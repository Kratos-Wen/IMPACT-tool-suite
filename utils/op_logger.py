import csv
import json
from datetime import datetime
from typing import Dict, Any, List
import os


class OperationLogger:
    """
    Lightweight operation logger.
    - enabled: toggle logging.
    - call log(event, **fields) to append an entry.
    - call flush(path) to write CSV to disk.
    """

    def __init__(self, enabled: bool = False, max_rows: int = 20000):
        self.enabled = enabled
        self.max_rows = max(1, int(max_rows))
        self._rows: List[Dict[str, Any]] = []
        self._dirty = False
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self._started_dt = datetime.now()
        self.session_id = self._started_dt.strftime("%Y%m%dT%H%M%S")

    def _serialize_value(self, value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple, set, dict)):
            try:
                return json.dumps(value, ensure_ascii=False, sort_keys=True)
            except Exception:
                return str(value)
        return str(value)

    def log(self, event: str, **fields):
        if not self.enabled:
            return
        now = datetime.now()
        row = {
            "timestamp": now.isoformat(timespec="milliseconds"),
            "event": event,
            "session_id": self.session_id,
            "elapsed_ms": int((now - self._started_dt).total_seconds() * 1000.0),
        }
        for key, value in fields.items():
            row[str(key)] = self._serialize_value(value)
        self._rows.append(row)
        self._dirty = True
        if len(self._rows) > self.max_rows:
            del self._rows[: len(self._rows) - self.max_rows]

    def flush(self, path: str):
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # collect all keys for consistent columns
        keys = ["timestamp", "event"]
        for r in self._rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in self._rows:
                writer.writerow({k: r.get(k, "") for k in keys})
        self._dirty = False

    def clear(self):
        self._rows.clear()
        self._dirty = False

    def rows(self) -> List[Dict[str, Any]]:
        return [dict(row) for row in self._rows]

    def is_dirty(self) -> bool:
        return bool(self._dirty)

    def has_rows(self) -> bool:
        return bool(self._rows)
