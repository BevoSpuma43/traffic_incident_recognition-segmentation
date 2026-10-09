import errno
import os
from pathlib import Path
from threading import Event, Thread

import pytest

from cctv_incident import batch


def windows_permission_error(code):
    error = PermissionError(errno.EACCES, "File temporarily unavailable")
    error.winerror = code
    return error


@pytest.mark.parametrize("winerror", [5, 32])
def test_transient_windows_lock_retries_and_publishes_complete_content(
    tmp_path, monkeypatch, winerror
):
    target = tmp_path / "checkpoint.json"
    target.write_text("previous", encoding="utf-8")
    replace = os.replace
    attempts = []
    waits = []

    def temporarily_locked(source, destination):
        attempts.append(Path(source))
        if len(attempts) <= 2:
            assert target.read_text(encoding="utf-8") == "previous"
            assert Path(source).read_text(encoding="utf-8") == "updated"
            raise windows_permission_error(winerror)
        replace(source, destination)

    monkeypatch.setattr(batch.os, "replace", temporarily_locked)
    monkeypatch.setattr(batch.time, "sleep", waits.append)

    batch.atomic_write(target, "updated")

    assert target.read_text(encoding="utf-8") == "updated"
    assert len(set(attempts)) == 1  # Retry the already flushed temporary file.
    assert len(waits) == 2
    assert all(delay > 0 for delay in waits)
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("winerror", [5, 32])
def test_persistent_windows_lock_is_bounded_and_preserves_previous_content(
    tmp_path, monkeypatch, winerror
):
    target = tmp_path / "checkpoint.json"
    target.write_text("previous", encoding="utf-8")
    error = windows_permission_error(winerror)
    attempts = []
    waits = []

    def locked(source, destination):
        attempts.append(source)
        raise error

    monkeypatch.setattr(batch.os, "replace", locked)
    monkeypatch.setattr(batch.time, "sleep", waits.append)

    with pytest.raises(PermissionError) as raised:
        batch.atomic_write(target, "updated")

    assert raised.value is error
    assert 1 < len(attempts) <= 10
    assert len(waits) == len(attempts) - 1
    assert 0 < sum(waits) <= 2.0
    assert target.read_text(encoding="utf-8") == "previous"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "error",
    [
        PermissionError(errno.EACCES, "Non-Windows permission failure"),
        windows_permission_error(1314),
        OSError(errno.ENOSPC, "No space left"),
    ],
)
def test_unrelated_replace_errors_are_reported_immediately(tmp_path, monkeypatch, error):
    target = tmp_path / "checkpoint.json"
    target.write_text("previous", encoding="utf-8")
    attempts = []
    waits = []

    def fail(source, destination):
        attempts.append(source)
        raise error

    monkeypatch.setattr(batch.os, "replace", fail)
    monkeypatch.setattr(batch.time, "sleep", waits.append)

    with pytest.raises(type(error)) as raised:
        batch.atomic_write(target, "updated")

    assert raised.value is error
    assert len(attempts) == 1
    assert not waits
    assert target.read_text(encoding="utf-8") == "previous"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.skipif(os.name != "nt", reason="Requires Windows file sharing semantics")
def test_real_windows_reader_can_close_before_atomic_replacement_retries(tmp_path, monkeypatch):
    target = tmp_path / "checkpoint.json"
    target.write_text("previous", encoding="utf-8")
    reader = target.open(encoding="utf-8")
    retry_seen = Event()
    reader_closed = Event()
    failures = []
    sleep = batch.time.sleep

    def wait_for_reader(delay):
        retry_seen.set()
        if not reader_closed.wait(timeout=5):
            raise TimeoutError("Reader was not released")
        sleep(delay)

    def publish():
        try:
            batch.atomic_write(target, "updated")
        except Exception as exc:
            failures.append(exc)

    monkeypatch.setattr(batch.time, "sleep", wait_for_reader)
    writer = Thread(target=publish)
    writer.start()
    try:
        assert retry_seen.wait(timeout=5), "Expected a retry while the real reader is open"
        assert target.read_text(encoding="utf-8") == "previous"
    finally:
        reader.close()
        reader_closed.set()
        writer.join(timeout=5)

    assert not writer.is_alive()
    assert not failures
    assert target.read_text(encoding="utf-8") == "updated"
    assert not list(tmp_path.glob("*.tmp"))
