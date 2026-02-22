from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


class JSONTaskOutputCache:
    """Persistent JSON cache for task outputs keyed by task/patient/plan/input fingerprint."""

    def __init__(self, path: str | Path, *, max_entries: int = 20000):
        self.path = Path(path)
        self.max_entries = int(max_entries)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._data: Dict[str, Any] = {"version": 1, "entries": {}}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("entries"), dict):
                self._data = {"version": int(raw.get("version", 1)), "entries": raw["entries"]}
        except Exception:
            self._data = {"version": 1, "entries": {}}

    def _persist_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    @staticmethod
    def stable_hash(value: Any) -> str:
        txt = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(txt.encode("utf-8")).hexdigest()

    def make_key(self, meta: Dict[str, Any]) -> str:
        # Keep the original meta for debugging while using a stable compact hash key.
        return self.stable_hash(meta)

    def get(self, meta: Dict[str, Any]) -> Optional[Any]:
        key = self.make_key(meta)
        with self._lock:
            row = (self._data.get("entries") or {}).get(key)
            if not isinstance(row, dict):
                self._misses += 1
                return None
            self._hits += 1
            row["last_accessed_at"] = time.time()
            return row.get("payload")

    def set(self, meta: Dict[str, Any], payload: Any) -> None:
        key = self.make_key(meta)
        with self._lock:
            entries = self._data.setdefault("entries", {})
            entries[key] = {
                "meta": meta,
                "payload": payload,
                "saved_at": time.time(),
                "last_accessed_at": time.time(),
            }
            if len(entries) > self.max_entries:
                # Remove oldest accessed entries first.
                ordered = sorted(
                    entries.items(),
                    key=lambda kv: float((kv[1] or {}).get("last_accessed_at") or (kv[1] or {}).get("saved_at") or 0.0),
                )
                for k, _ in ordered[: max(1, len(entries) - self.max_entries)]:
                    entries.pop(k, None)
            try:
                self._persist_locked()
            except Exception:
                pass

    def clear(self) -> None:
        with self._lock:
            self._data = {"version": 1, "entries": {}}
            try:
                self._persist_locked()
            except Exception:
                pass

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            entries = self._data.get("entries") or {}
            return {
                "entries": len(entries),
                "hits": int(self._hits),
                "misses": int(self._misses),
                "path": str(self.path),
            }
