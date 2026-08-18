"""Local artifact persistence for experiment runs."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    run_id: str
    path: str
    artifacts: tuple[str, ...]


class ArtifactStore(Protocol):
    def create_run(self, experiment_name: str, run_id: str) -> Path: ...

    def write_json(self, run_path: Path, name: str, value: Mapping[str, Any]) -> None: ...

    def write_jsonl(
        self, run_path: Path, name: str, values: Sequence[Mapping[str, Any]]
    ) -> None: ...

    def write_text(self, run_path: Path, name: str, value: str) -> None: ...


class LocalArtifactStore:
    """Persist run artifacts under a local directory."""

    def __init__(self, root: str | Path = "runs") -> None:
        self.root = Path(root)

    def create_run(self, experiment_name: str, run_id: str) -> Path:
        path = self.root / experiment_name / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, run_path: Path, name: str, value: Mapping[str, Any]) -> None:
        self._atomic_write(
            run_path / name, json.dumps(value, indent=2, sort_keys=True, default=str)
        )

    def write_jsonl(self, run_path: Path, name: str, values: Sequence[Mapping[str, Any]]) -> None:
        content = "".join(json.dumps(value, sort_keys=True, default=str) + "\n" for value in values)
        self._atomic_write(run_path / name, content)

    def write_text(self, run_path: Path, name: str, value: str) -> None:
        self._atomic_write(run_path / name, value)

    def _atomic_write(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            os.unlink(temporary_name)
            raise


def manifest_for(run_path: Path, run_id: str) -> ArtifactManifest:
    artifacts = tuple(
        sorted(str(path.relative_to(run_path)) for path in run_path.rglob("*") if path.is_file())
    )
    return ArtifactManifest(run_id=run_id, path=str(run_path), artifacts=artifacts)
