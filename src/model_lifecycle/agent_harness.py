"""Small fail-closed primitives for a repository-aware coding-agent harness.

The module is intentionally model-agnostic: it creates auditable contracts and
evidence packs before any model call, and validates test deltas afterwards.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


@dataclass(frozen=True)
class TaskContract:
    contract_id: str
    version: int
    objective: str
    constraints: tuple[str, ...]
    required_tests: tuple[str, ...]
    evidence: tuple[str, ...] = ()
    status: str = "OPEN"
    next_action: str = "inspect"
    parent_digest: str | None = None

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()

    def apply(self, delta: "ContractDelta") -> "TaskContract":
        if delta.base_digest != self.digest:
            raise ValueError("stale delta: base digest does not match current contract")
        if delta.contract_id != self.contract_id:
            raise ValueError("delta targets a different contract")
        return replace(
            self,
            version=self.version + 1,
            evidence=self.evidence + tuple(delta.evidence_append),
            status=delta.status if delta.status is not None else self.status,
            next_action=delta.next_action if delta.next_action is not None else self.next_action,
            parent_digest=self.digest,
        )


@dataclass(frozen=True)
class ContractDelta:
    contract_id: str
    base_digest: str
    evidence_append: tuple[str, ...] = ()
    status: str | None = None
    next_action: str | None = None


@dataclass(frozen=True)
class EvidenceChunk:
    path: str
    start_line: int
    end_line: int
    score: float
    text: str


@dataclass(frozen=True)
class RepositoryEvidencePack:
    query: str
    root: str
    chunks: tuple[EvidenceChunk, ...]
    source_digest: str
    approx_tokens: int


def _terms(text: str) -> set[str]:
    normalized = re.sub(r"[^A-Za-z0-9]+", " ", text.replace("_", " "))
    return {term.lower() for term in normalized.split() if len(term) >= 3}


def _structural_lines(path: Path, lines: list[str], query_terms: set[str]) -> set[int]:
    selected: set[int] = set()
    if path.suffix == ".py":
        try:
            tree = ast.parse("".join(lines))
            for node in ast.walk(tree):
                if (
                    isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    and query_terms.intersection(_terms(lines[node.lineno - 1]))
                ):
                    selected.add(node.lineno)
        except SyntaxError:
            pass
    elif path.suffix == ".md":
        selected.update(
            i for i, line in enumerate(lines, 1)
            if line.lstrip().startswith("#") and query_terms.intersection(_terms(line))
        )
    return selected


def build_evidence_pack(
    root: Path,
    query: str,
    *,
    include: tuple[str, ...] = ("*.py", "*.md"),
    max_files: int = 6,
    max_chunks: int = 18,
    context_lines: int = 2,
) -> RepositoryEvidencePack:
    """Rank lexical hits, then retain structural anchors plus local context."""
    query_terms = _terms(query)
    candidates: list[tuple[float, Path, list[str], list[int]]] = []
    seen: set[Path] = set()
    for pattern in include:
        for path in root.rglob(pattern):
            if path in seen or any(part in {".git", ".venv", "__pycache__", "runs", "_handoff"} for part in path.parts):
                continue
            seen.add(path)
            try:
                lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            except (UnicodeDecodeError, OSError):
                continue
            path_terms = _terms(str(path.relative_to(root)))
            hit_lines = [
                i for i, line in enumerate(lines, 1)
                if query_terms.intersection(_terms(line))
            ]
            if not hit_lines:
                continue
            file_terms = _terms("".join(lines))
            score = (
                50.0 * len(query_terms.intersection(path_terms))
                + 4.0 * len(query_terms.intersection(file_terms))
                + min(len(hit_lines), 10)
            )
            candidates.append((score, path, lines, hit_lines))
    candidates.sort(key=lambda row: (-row[0], str(row[1])))

    chunks: list[EvidenceChunk] = []
    digest = hashlib.sha256()
    for file_score, path, lines, hits in candidates[:max_files]:
        rel = str(path.relative_to(root)).replace("\\", "/")
        digest.update(rel.encode())
        digest.update("".join(lines).encode())
        structural = _structural_lines(path, lines, query_terms)
        ranked_anchors = sorted(
            set(hits) | structural,
            key=lambda line_no: (
                -len(query_terms.intersection(_terms(lines[line_no - 1]))),
                0 if line_no in structural else 1,
                line_no,
            ),
        )[:4]
        anchors = sorted(ranked_anchors)
        intervals: list[tuple[int, int]] = []
        for line_no in anchors:
            start = max(1, line_no - context_lines)
            end = min(len(lines), line_no + context_lines)
            if intervals and start <= intervals[-1][1] + 1:
                intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
            else:
                intervals.append((start, end))
        for start, end in intervals:
            text = "".join(lines[start - 1:end])
            local_hits = sum(1 for term in query_terms if term in _terms(text))
            chunks.append(EvidenceChunk(rel, start, end, file_score + local_hits, text))
    chunks.sort(key=lambda chunk: (-chunk.score, chunk.path, chunk.start_line))
    # Preserve at least one excerpt from every selected file before spending the
    # remaining budget on additional high-scoring excerpts. Without this
    # diversity rule a verbose test file can crowd out the implementation it
    # exercises even when both ranked highly.
    primary: list[EvidenceChunk] = []
    seen_paths: set[str] = set()
    for chunk in chunks:
        if chunk.path not in seen_paths:
            primary.append(chunk)
            seen_paths.add(chunk.path)
    primary_ids = {(chunk.path, chunk.start_line, chunk.end_line) for chunk in primary}
    remainder = [
        chunk for chunk in chunks
        if (chunk.path, chunk.start_line, chunk.end_line) not in primary_ids
    ]
    chunks = (primary + remainder)[:max_chunks]
    approx_tokens = sum(max(1, len(chunk.text.encode("utf-8")) // 4) for chunk in chunks)
    return RepositoryEvidencePack(
        query=query, root=str(root.resolve()), chunks=tuple(chunks),
        source_digest=digest.hexdigest(), approx_tokens=approx_tokens,
    )


def full_file_control(root: Path, paths: Iterable[str]) -> tuple[int, dict[str, str]]:
    payload = {path: (root / path).read_text(encoding="utf-8") for path in paths}
    return sum(max(1, len(text.encode("utf-8")) // 4) for text in payload.values()), payload


def test_baseline_non_weakening(
    before: Mapping[str, bool], after: Mapping[str, bool], *, require_same_tests: bool = True,
) -> dict[str, object]:
    missing = sorted(set(before) - set(after))
    regressions = sorted(name for name, passed in before.items() if passed and not after.get(name, False))
    additions = sorted(set(after) - set(before))
    passed = not regressions and (not require_same_tests or not missing)
    return {
        "pass": passed, "regressions": regressions, "missing": missing,
        "additions": additions, "before_passed": sum(before.values()),
        "after_passed": sum(after.values()),
    }


def deterministic_maintainability_gate(
    source: str, *, max_function_lines: int = 80, max_branches: int = 12,
) -> dict[str, object]:
    """Fail-closed anti-slop screen for deterministic agent-authored Python."""
    violations: list[dict[str, object]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"pass": False, "violations": [{"kind": "syntax", "detail": str(exc)}]}
    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            length = end - node.lineno + 1
            branches = sum(
                isinstance(child, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.Match))
                for child in ast.walk(node)
            )
            if length > max_function_lines:
                violations.append({"kind": "long_function", "line": node.lineno, "value": length})
            if branches > max_branches:
                violations.append({"kind": "branch_complexity", "line": node.lineno, "value": branches})
        elif isinstance(node, ast.ExceptHandler):
            broad = node.type is None or (
                isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}
            )
            if broad:
                violations.append({"kind": "broad_exception", "line": node.lineno})
            if any(isinstance(statement, ast.Pass) for statement in node.body):
                violations.append({"kind": "swallowed_exception", "line": node.lineno})
        elif isinstance(node, ast.Call):
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name == "sleep":
                violations.append({"kind": "blocking_sleep", "line": node.lineno})
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if any(module == "random" or module.startswith("random.") for module in modules):
                violations.append({"kind": "unseeded_random_dependency", "line": node.lineno})
    for line_no, line in enumerate(lines, 1):
        if re.search(r"\b(?:TODO|FIXME|HACK)\b", line, flags=re.IGNORECASE):
            violations.append({"kind": "unfinished_marker", "line": line_no})
    return {"pass": not violations, "violations": violations}
