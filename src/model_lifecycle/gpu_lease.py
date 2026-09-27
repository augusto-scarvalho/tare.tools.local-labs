"""Cross-process GPU lease coordination for shared text and image workloads.

Coordinates admission between the text gateway and ComfyUI on a single GPU.
Uses Linux flock on a dedicated lease file with monotonic deadlines.
Stale metadata without a held kernel lock is never authoritative.
"""
from __future__ import annotations

import contextlib
import errno
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import stat
import threading
import time
from typing import Any, Callable, Iterator


def _get_fcntl():
    try:
        import fcntl
        return fcntl
    except ImportError:
        return None


class SharedGpuLease:
    """Exclusive cross-process GPU lease backed by Linux flock.

    Parameters:
        path: Path to the lease lock file. If None, coordination is disabled.
        timeout: Default bounded acquisition timeout in seconds.
    """

    def __init__(self, path: Path | str | None = None, timeout: float = 60.0) -> None:
        self.path: Path | None = Path(path) if path is not None else None
        self.timeout: float = float(timeout)
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("GPU wait timeout must be finite and positive")
        self._thread_lock = threading.Lock()
        self._active_receipt: dict[str, Any] | None = None

    @property
    def is_enabled(self) -> bool:
        return self.path is not None

    def _open_lock_file(self, *, create=True) -> int:
        assert self.path is not None
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if os.path.islink(self.path):
            raise ValueError(f"GPU lock path must not be a symlink: {self.path}")

        flags = os.O_RDWR | getattr(os, "O_NONBLOCK", 0)
        if create:
            flags |= os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW

        try:
            fd = os.open(self.path, flags, 0o600)
        except OSError as exc:
            if exc.errno == getattr(errno, "ELOOP", None):
                raise ValueError(f"GPU lock path must not be a symlink: {self.path}") from exc
            raise

        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            os.close(fd)
            raise ValueError(f"GPU lock path must be a regular file: {self.path}")

        if os.path.islink(self.path):
            os.close(fd)
            raise ValueError(f"GPU lock path must not be a symlink: {self.path}")

        if create:
            try:
                os.fchmod(fd, 0o600)
            except BaseException:
                os.close(fd)
                raise

        return fd

    @contextlib.contextmanager
    def hold(
        self,
        kind: str,
        request_id: str,
        *,
        timeout: float | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[dict[str, Any] | None]:
        """Acquire exclusive GPU lease for 'text' or 'image'.

        Yields a receipt containing the secret nonce.
        Raises TimeoutError if deadline expires, or InterruptedError if cancelled.
        """
        if not self.is_enabled:
            yield None
            return

        if kind not in ("text", "image"):
            raise ValueError(f"Invalid lease kind: {kind!r} (expected 'text' or 'image')")
        if not isinstance(request_id, str) or not request_id or len(request_id.encode()) > 128:
            raise ValueError("request_id must be a non-empty string of at most 128 UTF-8 bytes")

        fcntl_mod = _get_fcntl()
        if fcntl_mod is None:
            raise RuntimeError("fcntl is required for GPU lease coordination on POSIX systems")

        effective_timeout = self.timeout if timeout is None else float(timeout)
        if not math.isfinite(effective_timeout) or effective_timeout <= 0:
            raise ValueError("GPU wait timeout must be finite and positive")
        deadline = time.monotonic() + effective_timeout
        poll_interval = 0.05

        if cancelled is not None and cancelled():
            raise InterruptedError("GPU lease acquisition cancelled before wait")

        # First serialize threads in the same process
        acquired_thread_lock = False
        while True:
            if cancelled is not None and cancelled():
                raise InterruptedError("GPU lease acquisition cancelled while waiting for thread lock")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for GPU lease thread lock ({effective_timeout}s)")
            if self._thread_lock.acquire(timeout=min(poll_interval, max(0.001, remaining))):
                acquired_thread_lock = True
                break

        fd = None
        flock_acquired = False
        ticket = None
        try:
            fd = self._open_lock_file()
            ticket = self._take_ticket(fcntl_mod)
            while True:
                if cancelled is not None and cancelled():
                    raise InterruptedError("GPU lease acquisition cancelled while waiting for file lock")
                if self._first_in_line(fcntl_mod, ticket[0]):
                    try:
                        fcntl_mod.flock(fd, fcntl_mod.LOCK_EX | fcntl_mod.LOCK_NB)
                        flock_acquired = True
                        break
                    except (BlockingIOError, OSError) as exc:
                        if exc.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                            raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for GPU lease ({effective_timeout}s)")
                time.sleep(min(poll_interval, max(0.001, remaining)))
            self._drop_ticket(ticket)
            ticket = None

            nonce = secrets.token_hex(16)
            acquired_at = time.time()
            metadata = {
                "kind": kind,
                "request_id": request_id,
                "pid": os.getpid(),
                "nonce": nonce,
                "acquired_at": acquired_at,
            }
            raw = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, raw)
            os.ftruncate(fd, len(raw))
            os.fsync(fd)

            receipt = {
                "kind": kind,
                "request_id": request_id,
                "pid": os.getpid(),
                "nonce": nonce,
                "path": str(self.path),
                "acquired_at": acquired_at,
            }
            self._active_receipt = receipt

            yield receipt

        finally:
            if ticket is not None:
                self._drop_ticket(ticket)
            try:
                if flock_acquired and fd is not None:
                    with contextlib.suppress(Exception):
                        os.lseek(fd, 0, os.SEEK_SET)
                        os.ftruncate(fd, 0)
                        os.fsync(fd)
                    fcntl_mod.flock(fd, fcntl_mod.LOCK_UN)
            finally:
                self._active_receipt = None
                if fd is not None:
                    with contextlib.suppress(Exception):
                        os.close(fd)
                if acquired_thread_lock:
                    self._thread_lock.release()

    # First-come queue in front of the lease. Waiters hold an flock on their ticket, so a
    # ticket whose lock can be taken belongs to a dead process and is discarded. Tickets are
    # locked before they are renamed into the queue, so a visible ticket is never unlocked
    # by a live owner. Processes running older code skip the queue but still take the same
    # lease flock, so exclusion never depends on the queue.
    @property
    def _queue_dir(self) -> Path:
        assert self.path is not None
        return self.path.with_name(self.path.name + ".queue")

    def _take_ticket(self, fcntl_mod) -> tuple[str, int]:
        queue = self._queue_dir
        queue.mkdir(mode=0o700, exist_ok=True)
        name = f"{time.time_ns():020d}-{os.getpid()}-{secrets.token_hex(4)}"
        staging = queue / ("." + name)
        fd = os.open(staging, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            fcntl_mod.flock(fd, fcntl_mod.LOCK_EX | fcntl_mod.LOCK_NB)
            os.rename(staging, queue / name)
        except BaseException:
            os.close(fd)
            with contextlib.suppress(OSError):
                os.unlink(staging)
            raise
        return name, fd

    def _drop_ticket(self, ticket: tuple[str, int]) -> None:
        with contextlib.suppress(OSError):
            os.unlink(self._queue_dir / ticket[0])
        with contextlib.suppress(OSError):
            os.close(ticket[1])

    def _live_tickets(self, fcntl_mod) -> list[str]:
        live = []
        for name in sorted(n for n in os.listdir(self._queue_dir) if not n.startswith(".")):
            try:
                fd = os.open(self._queue_dir / name, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
            except OSError:
                continue  # removed by its owner meanwhile
            try:
                fcntl_mod.flock(fd, fcntl_mod.LOCK_EX | fcntl_mod.LOCK_NB)
            except (BlockingIOError, OSError):
                live.append(name)
            else:
                with contextlib.suppress(OSError):
                    os.unlink(self._queue_dir / name)  # owner died while waiting
            finally:
                os.close(fd)
        return live

    def _first_in_line(self, fcntl_mod, name: str) -> bool:
        ahead = [n for n in self._live_tickets(fcntl_mod) if n < name]
        return not ahead

    def status(self) -> dict[str, Any]:
        """Report GPU lease status.

        Only reports held if confirmed by flock. Stale metadata without flock
        is never reported as held. Does not leak nonce.
        """
        if not self.is_enabled:
            return {"enabled": False}

        fcntl_mod = _get_fcntl()
        if fcntl_mod is None:
            return {
                "enabled": True,
                "path": str(self.path),
                "held": False,
                "owner": None,
                "error": "fcntl_unavailable",
            }

        active = self._active_receipt
        if active is not None:
            return {
                "enabled": True,
                "path": str(self.path),
                "held": True,
                "owner": {
                    "kind": active["kind"],
                    "request_id": active["request_id"],
                    "pid": active["pid"],
                    "acquired_at": active["acquired_at"],
                },
            }

        assert self.path is not None
        if not self.path.exists():
            return {
                "enabled": True,
                "path": str(self.path),
                "held": False,
                "owner": None,
            }

        if os.path.islink(self.path):
            raise ValueError(f"GPU lock path must not be a symlink: {self.path}")

        fd = None
        try:
            fd = self._open_lock_file(create=False)
            try:
                fcntl_mod.flock(fd, fcntl_mod.LOCK_EX | fcntl_mod.LOCK_NB)
                fcntl_mod.flock(fd, fcntl_mod.LOCK_UN)
                return {
                    "enabled": True,
                    "path": str(self.path),
                    "held": False,
                    "owner": None,
                }
            except (BlockingIOError, OSError) as exc:
                if exc.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                    raise
                raw = b""
                with contextlib.suppress(Exception):
                    os.lseek(fd, 0, os.SEEK_SET)
                    raw = os.read(fd, 4096)
                owner_info = None
                if raw:
                    with contextlib.suppress(Exception):
                        meta = json.loads(raw.decode("utf-8"))
                        if isinstance(meta, dict):
                            owner_info = {
                                "kind": meta.get("kind"),
                                "request_id": meta.get("request_id"),
                                "pid": meta.get("pid"),
                                "acquired_at": meta.get("acquired_at"),
                            }
                return {
                    "enabled": True,
                    "path": str(self.path),
                    "held": True,
                    "owner": owner_info,
                }
        finally:
            if fd is not None:
                os.close(fd)

    def validate_image_lease(self, nonce: str) -> bool:
        """Validate that an external process holds an IMAGE lease matching nonce."""
        if not self.is_enabled:
            return False
        if not isinstance(nonce, str) or not nonce:
            return False

        fcntl_mod = _get_fcntl()
        if fcntl_mod is None:
            return False

        if self._active_receipt is not None:
            return False

        assert self.path is not None
        if not self.path.exists():
            return False
        if os.path.islink(self.path):
            raise ValueError(f"GPU lock path must not be a symlink: {self.path}")

        fd = None
        try:
            fd = self._open_lock_file(create=False)
            try:
                fcntl_mod.flock(fd, fcntl_mod.LOCK_EX | fcntl_mod.LOCK_NB)
                fcntl_mod.flock(fd, fcntl_mod.LOCK_UN)
                return False
            except (BlockingIOError, OSError) as exc:
                if exc.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                    return False

            os.lseek(fd, 0, os.SEEK_SET)
            raw = os.read(fd, 4096)
            if not raw:
                return False
            meta = json.loads(raw.decode("utf-8"))
            if not isinstance(meta, dict):
                return False
            if meta.get("kind") != "image":
                return False
            stored_nonce = meta.get("nonce")
            if not isinstance(stored_nonce, str) or not hmac.compare_digest(stored_nonce, nonce):
                return False
            return True
        except Exception:
            return False
        finally:
            if fd is not None:
                os.close(fd)
