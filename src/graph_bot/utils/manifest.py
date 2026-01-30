from __future__ import annotations

import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class RunManifest:
    """
    A ledger for tracking runs in a JSONL file.
    Supports concurrent writes using file locking.
    """

    def __init__(self, manifest_path: Optional[Path] = None):
        if manifest_path is None:
            # Default to outputs/run_manifest.jsonl
            self.manifest_path = Path("outputs/run_manifest.jsonl")
        else:
            self.manifest_path = manifest_path

        # Ensure parent directory exists
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def _append_log(self, data: Dict[str, Any]) -> None:
        """Appends a JSON line to the manifest file with file locking."""
        data["timestamp"] = datetime.now(timezone.utc).isoformat()

        with self.manifest_path.open("a", encoding="utf-8") as f:
            try:
                # Exclusive lock (blocking)
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(json.dumps(data) + "\n")
                f.flush()
            finally:
                # Release lock
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def log_start(self, run_id: str, config: Dict[str, Any]) -> None:
        """Logs the start of a run."""
        self._append_log({"run_id": run_id, "status": "STARTED", "config": config})

    def log_end(
        self, run_id: str, status: str, metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Logs the end of a run.
        status should be "COMPLETED" or "FAILED".
        """
        self._append_log({"run_id": run_id, "status": status, "metrics": metrics or {}})
