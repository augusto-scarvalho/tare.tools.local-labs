"""Small, fail-closed lifecycle kernel for local research experiments.

The harness deliberately does not schedule work, interpret scientific gates, or
replace the backlog pipeline.  It gives runners a common way to preflight a live
endpoint, append raw samples, record restoration, seal immutable evidence, and
replay deterministic scorers without using the GPU again.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .collectors.request import RequestResult


JOURNAL_SCHEMA = "local-labs-experiment-journal-v1"
TERMINAL_SCHEMA = "local-labs-experiment-terminal-v1"
RECEIPT_SCHEMA = "local-labs-backlog-receipt-v1"
JOURNAL_NAME = "run.events.jsonl"
TERMINAL_NAME = "run.terminal.json"
SAMPLES_NAME = "samples.jsonl"
LOCK_NAME = ".experiment-run.lock"
_UNSET = object()
KNOWN_EVENT_TYPES = {
    "STARTED",
    "PREFLIGHT_PASSED",
    "PREFLIGHT_FAILED",
    "CHECKPOINT",
    "RESTORED",
    "RESTORE_FAILED",
    "SEALED",
    "ABORTED",
}


class HarnessError(RuntimeError):
    """Base class for a fail-closed harness error."""


class HarnessStateError(HarnessError):
    """The requested lifecycle transition is not legal."""


class PreflightError(HarnessError):
    """The physical endpoint did not satisfy its declared contract."""


@dataclass(frozen=True)
class PreflightContract:
    """Minimum observable contract for one cheap physical canary."""

    name: str
    require_answered: bool = True
    require_text: bool = True
    require_generated_tokens: bool = True
    require_server_timings: bool = False


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: pathlib.Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(_canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: pathlib.Path, value: Any) -> None:
    with path.open("ab") as stream:
        stream.write(_canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _event_digest(event: Mapping[str, Any]) -> str:
    return _sha256_value({key: value for key, value in event.items() if key != "event_sha256"})


def _receipt_with_fingerprint(receipt: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    payload = dict(receipt)
    if payload.get("schema") != RECEIPT_SCHEMA:
        raise HarnessStateError(f"receipt schema must be {RECEIPT_SCHEMA!r}")
    if payload.get("task_id") != task_id:
        raise HarnessStateError("receipt task_id does not match the active run")
    supplied = payload.pop("receipt_fingerprint", None)
    computed = _sha256_value(payload)
    if supplied is not None and supplied != computed:
        raise HarnessStateError("receipt_fingerprint does not match receipt content")
    payload["receipt_fingerprint"] = computed
    return payload


def _file_manifest(raw_dir: pathlib.Path, *, reject_temporary: bool = True) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path in {
            raw_dir / LOCK_NAME,
            raw_dir / TERMINAL_NAME,
        }:
            continue
        if reject_temporary and path.name.endswith(".tmp"):
            raise HarnessStateError(f"unfinished temporary artifact prevents terminal state: {path.name}")
        files[path.relative_to(raw_dir).as_posix()] = _sha256_file(path)
    return files


class ExperimentRun:
    """One exclusive experiment attempt rooted at a packet's ``raw`` directory."""

    def __init__(
        self,
        raw_dir: pathlib.Path | str,
        task_id: str,
        inputs: Any,
        *,
        requires_restoration: bool = False,
    ) -> None:
        self.raw_dir = pathlib.Path(raw_dir)
        self.task_id = task_id
        self.inputs = inputs
        self.requires_restoration = requires_restoration
        self.journal_path = self.raw_dir / JOURNAL_NAME
        self.samples_path = self.raw_dir / SAMPLES_NAME
        self.receipt_path = self.raw_dir / "receipt.json"
        self.terminal_path = self.raw_dir / TERMINAL_NAME
        self.lock_path = self.raw_dir / LOCK_NAME
        self._active = False
        self._terminal = False
        self._restored = False
        self._sample_count = 0
        self._seq = 0
        self._last_event_sha256: str | None = None

    def __enter__(self) -> "ExperimentRun":
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        if any(path.exists() for path in (
            self.journal_path,
            self.samples_path,
            self.receipt_path,
            self.terminal_path,
        )):
            raise HarnessStateError("run artifacts already exist; use a fresh successor packet")
        inputs_sha256 = _sha256_value(self.inputs)
        lock_payload = _canonical_bytes(
            {"task_id": self.task_id, "pid": os.getpid(), "started_at": _utc_now()}
        ) + b"\n"
        try:
            descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise HarnessStateError("another attempt owns this raw directory") from error
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(lock_payload)
                stream.flush()
                os.fsync(stream.fileno())
            self._active = True
            self._emit("STARTED", {
                "inputs_sha256": inputs_sha256,
                "requires_restoration": self.requires_restoration,
            })
        except BaseException:
            self._active = False
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass
            raise
        return self

    def __exit__(self, exc_type, exc, _traceback) -> bool:
        if self._terminal:
            return False
        if exc is not None:
            self.abort(f"{exc_type.__name__}: {exc}")
            return False
        self.abort("context exited without seal")
        raise HarnessStateError("experiment context exited without seal")

    def _require_active(self) -> None:
        if not self._active or self._terminal:
            raise HarnessStateError("run is not active")

    def _emit(self, event_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._require_active()
        event = {
            "schema": JOURNAL_SCHEMA,
            "seq": self._seq + 1,
            "at": _utc_now(),
            "type": event_type,
            "task_id": self.task_id,
            "prev_sha256": self._last_event_sha256,
            "payload": dict(payload),
        }
        event["event_sha256"] = _event_digest(event)
        _append_jsonl(self.journal_path, event)
        self._seq = event["seq"]
        self._last_event_sha256 = event["event_sha256"]
        return event

    def preflight(
        self,
        contract: PreflightContract,
        invoke: Callable[[], RequestResult],
    ) -> RequestResult:
        """Run one canary and reject contract failures before an expensive panel."""
        self._require_active()
        result = invoke()
        if not isinstance(result, RequestResult):
            errors = [f"invoke returned {type(result).__name__}, expected RequestResult"]
            summary: dict[str, Any] = {"result_type": type(result).__name__}
        else:
            errors = []
            if not result.ok:
                errors.append(f"transport/runtime failed: {result.error or 'unknown error'}")
            if contract.require_answered and not result.answered:
                errors.append(f"no usable answer: {result.error or 'answered=false'}")
            if contract.require_text and not result.text.strip():
                errors.append("response text is empty")
            if contract.require_generated_tokens and result.completion_tokens <= 0:
                errors.append("completion token count is not positive")
            if contract.require_server_timings and not (
                result.predicted_n and result.predicted_n > 0
                and result.predicted_ms and result.predicted_ms > 0
            ):
                errors.append("positive server generation timings are required")
            summary = {
                "ok": result.ok,
                "answered": result.answered,
                "completion_tokens": result.completion_tokens,
                "prompt_tokens": result.prompt_tokens,
                "predicted_n": result.predicted_n,
                "predicted_ms": result.predicted_ms,
                "text_sha256": hashlib.sha256(result.text.encode("utf-8")).hexdigest(),
                "reasoning_sha256": hashlib.sha256(result.reasoning_text.encode("utf-8")).hexdigest(),
                "error": result.error,
            }
        payload = {"contract": contract.__dict__, "result": summary, "errors": errors}
        self._emit("PREFLIGHT_FAILED" if errors else "PREFLIGHT_PASSED", payload)
        if errors:
            raise PreflightError("; ".join(errors))
        return result

    def record(self, sample: Mapping[str, Any]) -> None:
        """Append one raw, replayable sample without changing its schema."""
        self._require_active()
        _append_jsonl(self.samples_path, dict(sample))
        self._sample_count += 1

    def checkpoint(self, label: str, details: Mapping[str, Any] | None = None) -> None:
        """Emit a sparse progress milestone for the watcher."""
        self._emit("CHECKPOINT", {
            "label": label,
            "sample_count": self._sample_count,
            "details": dict(details or {}),
        })

    def restored(self, details: Mapping[str, Any], *, ok: bool = True) -> None:
        """Record the runner's physical restoration result."""
        self._restored = ok
        self._emit("RESTORED" if ok else "RESTORE_FAILED", {"details": dict(details)})

    def _write_terminal(self, status: str, *, error: str | None = None) -> dict[str, Any]:
        terminal = {
            "schema": TERMINAL_SCHEMA,
            "task_id": self.task_id,
            "status": status,
            "finished_at": _utc_now(),
            "sample_count": self._sample_count,
            "requires_restoration": self.requires_restoration,
            "restored": self._restored,
            "last_event_sha256": self._last_event_sha256,
            "files": _file_manifest(self.raw_dir, reject_temporary=status == "SEALED"),
        }
        if error is not None:
            terminal["error"] = error
        terminal["terminal_fingerprint"] = _sha256_value(terminal)
        _atomic_json(self.terminal_path, terminal)
        self._terminal = True
        self._active = False
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass
        return terminal

    def seal(self, receipt: Mapping[str, Any]) -> dict[str, Any]:
        """Atomically publish the receipt and terminal manifest after all evidence."""
        self._require_active()
        if self.requires_restoration and not self._restored:
            raise HarnessStateError("successful restoration is required before seal")
        payload = _receipt_with_fingerprint(receipt, self.task_id)
        _atomic_json(self.receipt_path, payload)
        self._emit("SEALED", {
            "receipt_sha256": _sha256_file(self.receipt_path),
            "receipt_fingerprint": payload["receipt_fingerprint"],
            "sample_count": self._sample_count,
        })
        self._write_terminal("SEALED")
        return payload

    def abort(self, error: str) -> dict[str, Any]:
        """Publish a structurally valid negative terminal without a scientific verdict."""
        if self._terminal:
            raise HarnessStateError("run already has a terminal state")
        self._require_active()
        self._emit("ABORTED", {"error": error, "sample_count": self._sample_count})
        return self._write_terminal("ABORTED", error=error)


def _verify_journal(path: pathlib.Path, task_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    events: list[dict[str, Any]] = []
    previous = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return [], [f"journal unavailable: {error}"]
    for index, line in enumerate(lines, 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"journal line {index} is invalid JSON: {error}")
            continue
        if not isinstance(event, dict):
            errors.append(f"journal line {index} is not an object")
            continue
        events.append(event)
        if event.get("schema") != JOURNAL_SCHEMA:
            errors.append(f"journal line {index} has unsupported schema")
        if event.get("seq") != index:
            errors.append(f"journal line {index} has non-monotonic seq")
        if event.get("task_id") != task_id:
            errors.append(f"journal line {index} has a different task_id")
        if event.get("prev_sha256") != previous:
            errors.append(f"journal line {index} breaks the hash chain")
        if event.get("event_sha256") != _event_digest(event):
            errors.append(f"journal line {index} has an invalid event digest")
        previous = event.get("event_sha256")
    if not events:
        errors.append("journal is empty")
    elif events[0].get("type") != "STARTED":
        errors.append("journal does not start with STARTED")
    return events, errors


def _verify_run(
    raw_dir: pathlib.Path | str,
    scorer: Callable[[list[dict[str, Any]]], Any] | None = None,
    *,
    expected_replay: Any = _UNSET,
) -> dict[str, Any]:
    """Verify a terminal run and optionally replay a deterministic scorer."""
    raw = pathlib.Path(raw_dir)
    errors: list[str] = []
    terminal_path = raw / TERMINAL_NAME
    try:
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"valid": False, "status": None, "errors": [f"terminal unavailable: {error}"]}
    if not isinstance(terminal, dict):
        return {
            "valid": False,
            "status": None,
            "errors": ["terminal is not a JSON object"],
        }
    supplied_terminal_fingerprint = terminal.get("terminal_fingerprint")
    terminal_content = dict(terminal)
    terminal_content.pop("terminal_fingerprint", None)
    if terminal.get("schema") != TERMINAL_SCHEMA:
        errors.append("unsupported terminal schema")
    try:
        if supplied_terminal_fingerprint != _sha256_value(terminal_content):
            errors.append("terminal fingerprint mismatch")
    except (TypeError, ValueError) as error:
        errors.append(f"terminal content is not canonical JSON: {error}")
    if (raw / LOCK_NAME).exists():
        errors.append("active run lock remains beside terminal state")

    expected_files = terminal.get("files")
    if not isinstance(expected_files, dict):
        errors.append("terminal file manifest is missing")
        expected_files = {}
    try:
        actual_files = _file_manifest(raw, reject_temporary=terminal.get("status") == "SEALED")
    except HarnessStateError as error:
        errors.append(str(error))
        actual_files = {}
    missing = sorted(set(expected_files) - set(actual_files))
    extra = sorted(set(actual_files) - set(expected_files))
    changed = sorted(
        path for path in set(expected_files) & set(actual_files)
        if expected_files[path] != actual_files[path]
    )
    if missing:
        errors.append(f"sealed files missing: {missing}")
    if extra:
        errors.append(f"unsealed files present: {extra}")
    if changed:
        errors.append(f"sealed files changed: {changed}")

    task_id = terminal.get("task_id")
    events, journal_errors = _verify_journal(raw / JOURNAL_NAME, task_id)
    errors.extend(journal_errors)
    if events and events[-1].get("event_sha256") != terminal.get("last_event_sha256"):
        errors.append("terminal does not bind the journal tail")
    expected_last_type = {"SEALED": "SEALED", "ABORTED": "ABORTED"}.get(terminal.get("status"))
    if expected_last_type is None:
        errors.append("terminal status is not SEALED or ABORTED")
    elif events and events[-1].get("type") != expected_last_type:
        errors.append("terminal status disagrees with the journal tail")
    if (
        terminal.get("status") == "SEALED"
        and terminal.get("requires_restoration")
        and not terminal.get("restored")
    ):
        errors.append("terminal claims a required restoration was not successful")

    receipt = None
    if terminal.get("status") == "SEALED":
        try:
            receipt = json.loads((raw / "receipt.json").read_text(encoding="utf-8"))
            if not isinstance(receipt, dict):
                raise TypeError("receipt is not a JSON object")
            normalized = dict(receipt)
            supplied = normalized.pop("receipt_fingerprint", None)
            if supplied != _sha256_value(normalized):
                errors.append("receipt fingerprint mismatch")
            if receipt.get("task_id") != task_id:
                errors.append("receipt task_id does not match terminal")
            if receipt.get("schema") != RECEIPT_SCHEMA:
                errors.append("receipt schema mismatch")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            errors.append(f"sealed receipt unavailable: {error}")

    samples: list[dict[str, Any]] = []
    samples_path = raw / SAMPLES_NAME
    if samples_path.exists():
        try:
            sample_lines = samples_path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            errors.append(f"samples unavailable: {error}")
            sample_lines = []
        for index, line in enumerate(sample_lines, 1):
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(f"sample line {index} is invalid JSON: {error}")
                continue
            if not isinstance(sample, dict):
                errors.append(f"sample line {index} is not an object")
                continue
            samples.append(sample)
    if len(samples) != terminal.get("sample_count"):
        errors.append("sample_count does not match samples.jsonl")

    if events:
        event_types = [event.get("type") for event in events]
        unknown_types = sorted(
            {str(event_type) for event_type in event_types if event_type not in KNOWN_EVENT_TYPES}
        )
        if unknown_types:
            errors.append(f"journal contains unknown event types: {unknown_types}")
        if event_types.count("STARTED") != 1:
            errors.append("journal must contain exactly one STARTED event")
        terminal_positions = [
            index for index, event_type in enumerate(event_types)
            if event_type in {"SEALED", "ABORTED"}
        ]
        if terminal_positions != [len(events) - 1]:
            errors.append("journal must contain exactly one terminal event at the end")
        if "PREFLIGHT_FAILED" in event_types:
            failed_at = event_types.index("PREFLIGHT_FAILED")
            if event_types[failed_at + 1:] != ["ABORTED"]:
                errors.append("PREFLIGHT_FAILED must transition directly to ABORTED")

        started = events[0]
        started_payload = started.get("payload")
        if not isinstance(started_payload, dict):
            errors.append("STARTED event payload is not an object")
        else:
            inputs_sha256 = started_payload.get("inputs_sha256")
            if not (
                isinstance(inputs_sha256, str)
                and len(inputs_sha256) == 64
                and all(character in "0123456789abcdef" for character in inputs_sha256)
            ):
                errors.append("STARTED inputs_sha256 is missing or invalid")
            if started_payload.get("requires_restoration") != terminal.get("requires_restoration"):
                errors.append("STARTED restoration contract does not match terminal")

        final_event = events[-1]
        final_payload = final_event.get("payload")
        if not isinstance(final_payload, dict):
            errors.append("terminal journal event payload is not an object")
        elif final_payload.get("sample_count") != len(samples):
            errors.append("terminal journal event sample_count does not match samples")

        if terminal.get("status") == "SEALED":
            restoration_events = [
                event for event in events
                if event.get("type") in {"RESTORED", "RESTORE_FAILED"}
            ]
            if terminal.get("requires_restoration") and (
                not restoration_events or restoration_events[-1].get("type") != "RESTORED"
            ):
                errors.append("SEALED journal lacks a final successful restoration event")
            if isinstance(final_payload, dict) and receipt is not None:
                receipt_path = raw / "receipt.json"
                if final_payload.get("receipt_sha256") != _sha256_file(receipt_path):
                    errors.append("SEALED journal receipt_sha256 mismatch")
                if final_payload.get("receipt_fingerprint") != receipt.get("receipt_fingerprint"):
                    errors.append("SEALED journal receipt_fingerprint mismatch")

    replay = None
    replay_matches = None
    if scorer is not None:
        try:
            replay = scorer(samples)
            if expected_replay is not _UNSET:
                replay_matches = _sha256_value(replay) == _sha256_value(expected_replay)
                if not replay_matches:
                    errors.append("replayed scorer output does not match expected output")
        except Exception as error:  # noqa: BLE001 - scorer failures are verification data
            errors.append(f"scorer replay failed: {type(error).__name__}: {error}")
    return {
        "valid": not errors,
        "status": terminal.get("status"),
        "task_id": task_id,
        "sample_count": len(samples),
        "receipt": receipt,
        "replay": replay,
        "replay_matches": replay_matches,
        "errors": errors,
    }


def verify_run(
    raw_dir: pathlib.Path | str,
    scorer: Callable[[list[dict[str, Any]]], Any] | None = None,
    *,
    expected_replay: Any = _UNSET,
) -> dict[str, Any]:
    """Total fail-closed verifier: malformed evidence is data, never a crash."""
    try:
        return _verify_run(raw_dir, scorer, expected_replay=expected_replay)
    except Exception as error:  # noqa: BLE001 - all malformed evidence fails closed
        return {
            "valid": False,
            "status": None,
            "errors": [f"verification failed: {type(error).__name__}: {error}"],
        }
