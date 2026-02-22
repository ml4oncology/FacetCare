from __future__ import annotations

import json
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class SeenRecord:
    date: str
    risk_probability: float


class DedupStore:
    def seen_within(self, patient_id: str, *, days: int, as_of: str) -> bool:
        raise NotImplementedError

    def mark_seen(self, patient_id: str, *, as_of: str, risk_probability: float) -> None:
        raise NotImplementedError


class JSONFileDedupStore(DedupStore):
    def __init__(self, path: str):
        self.path = Path(path)
        self._data: Dict[str, List[SeenRecord]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._data = {}
            return
        obj = json.loads(self.path.read_text(encoding="utf-8"))
        out: Dict[str, List[SeenRecord]] = {}
        for pid, records in (obj or {}).items():
            out[pid] = [SeenRecord(**r) for r in (records or [])]
        self._data = out

    def _save(self) -> None:
        obj = {pid: [r.__dict__ for r in recs] for pid, recs in self._data.items()}
        self.path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    def seen_within(self, patient_id: str, *, days: int, as_of: str) -> bool:
        if days <= 0:
            return False
        recs = self._data.get(patient_id, [])
        if not recs:
            return False
        d_asof = dt.date.fromisoformat(as_of)
        for r in recs:
            try:
                d = dt.date.fromisoformat(r.date)
            except Exception:
                continue
            if (d_asof - d).days <= days:
                return True
        return False

    def mark_seen(self, patient_id: str, *, as_of: str, risk_probability: float) -> None:
        self._data.setdefault(patient_id, []).append(SeenRecord(date=as_of, risk_probability=float(risk_probability)))
        self._save()
