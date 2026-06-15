"""Helpers for artifact provenance, run compatibility, and safer output writes."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


_CACHED_RUN_ID: str | None = None


def utc_now_iso() -> str:
    """Return the current UTC time in a stable ISO-8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def current_run_id() -> str:
    """Return the active run identifier, generating one when missing."""
    global _CACHED_RUN_ID
    configured_run_id = os.environ.get("PIPELINE_RUN_ID", "").strip()
    if configured_run_id:
        _CACHED_RUN_ID = configured_run_id
        return configured_run_id

    if _CACHED_RUN_ID is None:
        timestamp_text = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        _CACHED_RUN_ID = f"run_{timestamp_text}_pid{os.getpid()}"
        os.environ["PIPELINE_RUN_ID"] = _CACHED_RUN_ID
    return _CACHED_RUN_ID


def pipeline_profile() -> str:
    """Return a plain-text label for the active pipeline profile."""
    explicit_profile = os.environ.get("PIPELINE_PROFILE", "").strip().lower()
    if explicit_profile:
        return explicit_profile
    if os.environ.get("FAST_TEST_MODE", "0") == "1":
        return "fast_test"
    return "standard"


def artifact_metadata_path(output_path: Path) -> Path:
    """Return the sidecar metadata path for one CSV or other artifact."""
    return output_path.with_suffix(f"{output_path.suffix}.meta.json")


def versioned_artifact_path(output_path: Path, *, run_id: str | None = None) -> Path:
    """Return a run-scoped copy path for one artifact."""
    resolved_run_id = (run_id or current_run_id()).strip()
    return output_path.with_name(f"{output_path.stem}__{resolved_run_id}{output_path.suffix}")


def compute_file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_fingerprint(path: Path) -> dict[str, Any]:
    """Return a compact fingerprint for one dependency file."""
    resolved_path = Path(path)
    if not resolved_path.exists():
        return {
            "path": str(resolved_path),
            "exists": False,
        }

    stat_result = resolved_path.stat()
    return {
        "path": str(resolved_path),
        "exists": True,
        "size_bytes": int(stat_result.st_size),
        "modified_time_ns": int(stat_result.st_mtime_ns),
        "sha256": compute_file_sha256(resolved_path),
    }


def collect_dependency_fingerprints(paths: list[Path] | tuple[Path, ...] | None) -> list[dict[str, Any]]:
    """Return fingerprints for a list of dependency paths."""
    if not paths:
        return []
    return [dependency_fingerprint(Path(path)) for path in paths]


def load_artifact_metadata(output_path: Path) -> dict[str, Any] | None:
    """Load one artifact's sidecar metadata when it exists."""
    metadata_path = artifact_metadata_path(Path(output_path))
    if not metadata_path.exists():
        return None

    with metadata_path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def artifact_run_id(output_path: Path) -> str | None:
    """Return the run ID stored beside one artifact when present."""
    metadata = load_artifact_metadata(Path(output_path))
    if not metadata:
        return None
    run_id = str(metadata.get("run_id", "")).strip()
    return run_id or None


def artifacts_share_run_id(paths: list[Path] | tuple[Path, ...]) -> bool:
    """Return whether all artifacts with metadata belong to the same run."""
    run_ids = {
        run_id
        for run_id in (artifact_run_id(Path(path)) for path in paths)
        if run_id
    }
    return len(run_ids) <= 1


def _write_metadata(metadata_path: Path, metadata: dict[str, Any]) -> None:
    """Write one metadata JSON file with deterministic formatting."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, indent=2, sort_keys=True)
        file_handle.write("\n")


def _should_update_canonical(*, research_grade: bool, canonical_policy: str) -> bool:
    """Return whether the canonical path should be refreshed for this write."""
    normalized_policy = str(canonical_policy).strip().lower()
    if normalized_policy == "never":
        return False
    if normalized_policy == "always":
        return True
    if normalized_policy != "auto":
        raise ValueError(f"Unknown canonical write policy: {canonical_policy}")
    return research_grade or os.environ.get("ALLOW_EXPLORATORY_CANONICAL_OVERWRITE", "0") == "1"


def write_dataframe_artifact(
    df: pd.DataFrame,
    output_path: Path,
    *,
    producer: str,
    current_ticker: str | None = None,
    dependencies: list[Path] | tuple[Path, ...] | None = None,
    parameters: dict[str, Any] | None = None,
    research_grade: bool = True,
    canonical_policy: str = "auto",
) -> dict[str, Any]:
    """Write one DataFrame artifact with versioned and canonical provenance."""
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    run_id = current_run_id()
    versioned_path = versioned_artifact_path(resolved_output_path, run_id=run_id)
    df.to_csv(versioned_path, index=False)

    metadata = {
        "artifact_path": str(resolved_output_path),
        "versioned_artifact_path": str(versioned_path),
        "run_id": run_id,
        "created_at_utc": utc_now_iso(),
        "producer": str(producer),
        "ticker": str(current_ticker).strip().upper() if current_ticker else "",
        "research_grade": bool(research_grade),
        "pipeline_profile": pipeline_profile(),
        "row_count": int(len(df)),
        "column_names": [str(column) for column in df.columns.tolist()],
        "canonical_policy": str(canonical_policy),
        "dependencies": collect_dependency_fingerprints(dependencies),
        "parameters": parameters or {},
    }
    _write_metadata(artifact_metadata_path(versioned_path), metadata)

    canonical_updated = _should_update_canonical(
        research_grade=research_grade,
        canonical_policy=canonical_policy,
    )
    if canonical_updated:
        df.to_csv(resolved_output_path, index=False)
        canonical_metadata = dict(metadata)
        canonical_metadata["canonical_updated"] = True
        _write_metadata(artifact_metadata_path(resolved_output_path), canonical_metadata)

    return {
        "canonical_path": resolved_output_path,
        "versioned_path": versioned_path,
        "canonical_updated": canonical_updated,
        "metadata": metadata,
    }
