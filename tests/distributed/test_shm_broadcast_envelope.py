# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the opt-in integrity envelope on shm_broadcast inline payloads.

See https://github.com/vllm-project/vllm/issues/27858 for context. The envelope
adds a fixed 16-byte header (magic + version + reserved + payload_len + crc32)
ahead of the existing inline payload so corruption surfaces with a structured
exception instead of an opaque pickle error.
"""

from unittest import mock

import numpy as np
import pytest

from vllm.distributed.device_communicators.shm_broadcast import (
    MessageQueue,
    ShmBroadcastCorruptionError,
)

MAGIC = b"VSB1"
HEADER_SIZE = 16  # magic(4) + version(1) + reserved(3) + payload_len(4) + crc32(4)
# In-chunk layout: [overflow:1][header:16][payload:...]
HEADER_OFFSET = 1
PAYLOAD_OFFSET = HEADER_OFFSET + HEADER_SIZE  # = 17


def _make_pair(max_chunk_bytes: int = 64 * 1024, max_chunks: int = 4):
    """Build a writer + local reader sharing one shm ring, single-process."""
    writer = MessageQueue(
        n_reader=1,
        n_local_reader=1,
        max_chunk_bytes=max_chunk_bytes,
        max_chunks=max_chunks,
    )
    reader = MessageQueue.create_from_handle(writer.export_handle(), rank=0)
    writer.wait_until_ready()
    reader.wait_until_ready()
    return writer, reader


def _shutdown(writer: MessageQueue, reader: MessageQueue) -> None:
    writer.shutdown()
    reader.shutdown()


def _chunk_bytes(mq: MessageQueue, chunk_idx: int, n: int) -> bytes:
    """Snapshot first n bytes of a chunk in shm."""
    start = chunk_idx * mq.buffer.max_chunk_bytes
    return bytes(mq.buffer.shared_memory.buf[start : start + n])


# --- envelope disabled (default) ------------------------------------------------


def test_envelope_disabled_no_magic_header():
    """Default off: writer must not insert the magic prefix into the buffer."""
    with mock.patch(
        "vllm.distributed.device_communicators.shm_broadcast."
        "VLLM_SHM_BROADCAST_VERIFY",
        new=False,
    ):
        writer, reader = _make_pair()
        try:
            writer.enqueue("hello")
            head = _chunk_bytes(writer, chunk_idx=0, n=PAYLOAD_OFFSET + 8)
            assert MAGIC not in head, (
                f"Magic should not appear when verification is disabled, "
                f"but found in: {head!r}"
            )
            assert reader.dequeue(timeout=1) == "hello"
        finally:
            _shutdown(writer, reader)


def test_envelope_disabled_roundtrip_unchanged():
    """Default off: ordinary roundtrip works for several object types."""
    with mock.patch(
        "vllm.distributed.device_communicators.shm_broadcast."
        "VLLM_SHM_BROADCAST_VERIFY",
        new=False,
    ):
        writer, reader = _make_pair()
        try:
            for obj in [
                "hello",
                42,
                [1, 2, 3],
                {"k": "v"},
                np.arange(50, dtype=np.int64),
            ]:
                writer.enqueue(obj)
                got = reader.dequeue(timeout=1)
                if isinstance(obj, np.ndarray):
                    assert np.array_equal(got, obj)
                else:
                    assert got == obj
        finally:
            _shutdown(writer, reader)


def test_envelope_disabled_does_not_raise_corruption():
    """Default off: corruption never surfaces as ShmBroadcastCorruptionError.

    Contract under test is intentionally narrow: with the envelope disabled we
    do not promise *which* legacy exception (UnpicklingError, AttributeError,
    UnicodeDecodeError, etc.) shows up — only that the new structured
    exception type is *not* used. Pickle itself is permissive about random bit
    flips, so we also do not require any exception at all.
    """
    with mock.patch(
        "vllm.distributed.device_communicators.shm_broadcast."
        "VLLM_SHM_BROADCAST_VERIFY",
        new=False,
    ):
        writer, reader = _make_pair()
        try:
            writer.enqueue({"k": "v" * 200})
            buf = writer.buffer.shared_memory.buf
            buf[3 + 20] ^= 0xFF  # somewhere inside the pickled bytes
            try:
                reader.dequeue(timeout=1)
            except ShmBroadcastCorruptionError:
                pytest.fail(
                    "Envelope disabled but reader raised "
                    "ShmBroadcastCorruptionError — disabled mode must keep "
                    "the legacy failure surface."
                )
            except Exception:
                pass  # any other failure is acceptable legacy behavior
        finally:
            _shutdown(writer, reader)


# --- envelope enabled -----------------------------------------------------------


def test_envelope_enabled_inserts_magic_header():
    """With verification on, the magic prefix sits at offset 1 in each chunk."""
    with mock.patch(
        "vllm.distributed.device_communicators.shm_broadcast."
        "VLLM_SHM_BROADCAST_VERIFY",
        new=True,
    ):
        writer, reader = _make_pair()
        try:
            writer.enqueue("hello")
            head = _chunk_bytes(writer, chunk_idx=0, n=PAYLOAD_OFFSET)
            assert head[HEADER_OFFSET : HEADER_OFFSET + 4] == MAGIC, (
                f"Expected magic {MAGIC!r} at offset {HEADER_OFFSET}, "
                f"got {head[HEADER_OFFSET : HEADER_OFFSET + 4]!r}"
            )
        finally:
            _shutdown(writer, reader)


def test_envelope_enabled_roundtrip():
    """With verification on, ordinary roundtrip still works end-to-end."""
    with mock.patch(
        "vllm.distributed.device_communicators.shm_broadcast."
        "VLLM_SHM_BROADCAST_VERIFY",
        new=True,
    ):
        writer, reader = _make_pair()
        try:
            for obj in [
                "hello",
                42,
                [1, 2, 3],
                {"k": "v"},
                np.arange(50, dtype=np.int64),
            ]:
                writer.enqueue(obj)
                got = reader.dequeue(timeout=1)
                if isinstance(obj, np.ndarray):
                    assert np.array_equal(got, obj)
                else:
                    assert got == obj
        finally:
            _shutdown(writer, reader)


def test_envelope_detects_payload_corruption():
    """Flip a byte inside the payload → structured corruption error."""
    with mock.patch(
        "vllm.distributed.device_communicators.shm_broadcast."
        "VLLM_SHM_BROADCAST_VERIFY",
        new=True,
    ):
        writer, reader = _make_pair()
        try:
            writer.enqueue({"k": "v" * 200})
            buf = writer.buffer.shared_memory.buf
            # Flip a byte well inside the payload region.
            buf[PAYLOAD_OFFSET + 8] ^= 0xFF
            with pytest.raises(ShmBroadcastCorruptionError) as exc_info:
                reader.dequeue(timeout=1)
            # Diagnostic exception should expose enough to investigate.
            err = exc_info.value
            assert hasattr(err, "expected_crc")
            assert hasattr(err, "actual_crc")
            assert err.expected_crc != err.actual_crc
        finally:
            _shutdown(writer, reader)


def test_envelope_detects_header_corruption():
    """Flip a byte inside the magic → structured corruption error."""
    with mock.patch(
        "vllm.distributed.device_communicators.shm_broadcast."
        "VLLM_SHM_BROADCAST_VERIFY",
        new=True,
    ):
        writer, reader = _make_pair()
        try:
            writer.enqueue("hello")
            buf = writer.buffer.shared_memory.buf
            buf[HEADER_OFFSET] ^= 0xFF  # corrupt magic[0]
            with pytest.raises(ShmBroadcastCorruptionError):
                reader.dequeue(timeout=1)
        finally:
            _shutdown(writer, reader)


def test_envelope_detects_length_corruption():
    """Flip a byte inside payload_len → structured corruption error."""
    with mock.patch(
        "vllm.distributed.device_communicators.shm_broadcast."
        "VLLM_SHM_BROADCAST_VERIFY",
        new=True,
    ):
        writer, reader = _make_pair()
        try:
            writer.enqueue("hello")
            buf = writer.buffer.shared_memory.buf
            # payload_len lives at bytes [9..13] within the chunk (offset 1+8 .. 1+12)
            buf[HEADER_OFFSET + 8] ^= 0xFF
            with pytest.raises(ShmBroadcastCorruptionError):
                reader.dequeue(timeout=1)
        finally:
            _shutdown(writer, reader)


# --- overflow / ZMQ path is unaffected -----------------------------------------


def test_envelope_overflow_path_passthrough():
    """Large payload goes through ZMQ; envelope must NOT touch that path."""
    with mock.patch(
        "vllm.distributed.device_communicators.shm_broadcast."
        "VLLM_SHM_BROADCAST_VERIFY",
        new=True,
    ):
        # Small chunks force the overflow branch.
        writer, reader = _make_pair(max_chunk_bytes=1024, max_chunks=4)
        try:
            large = "x" * (8 * 1024)  # > max_chunk_bytes
            writer.enqueue(large)
            head = _chunk_bytes(writer, chunk_idx=0, n=PAYLOAD_OFFSET + 8)
            assert head[0] == 1, "overflow flag should be set in shm slot"
            assert MAGIC not in head, (
                "Envelope header must not be written on the overflow path; "
                f"unexpectedly found magic in: {head!r}"
            )
            assert reader.dequeue(timeout=2) == large
        finally:
            _shutdown(writer, reader)
