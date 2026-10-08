from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


NO_NOUN_TOKEN = "__NO_NOUN__"


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def _norm_key(value: Any) -> str:
    return _norm_text(value).lower()


def _safe_bool01(value: Any) -> bool:
    text = _norm_key(value)
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    try:
        return bool(int(float(text)))
    except Exception:
        return False


@dataclass
class HOIOntology:
    relations: Dict[str, Set[str]] = field(default_factory=dict)
    verb_display: Dict[str, str] = field(default_factory=dict)
    noun_display: Dict[str, str] = field(default_factory=dict)
    source_path: str = ""
    format_hint: str = ""

    def add_relation(self, verb: Any, noun: Any) -> None:
        verb_name = _norm_text(verb).lower()
        noun_name = _norm_text(noun).lower()
        if not verb_name or not noun_name:
            return
        verb_key = _norm_key(verb_name)
        noun_key = _norm_key(noun_name)
        self.relations.setdefault(verb_key, set()).add(noun_key)
        self.verb_display.setdefault(verb_key, verb_name)
        self.noun_display.setdefault(noun_key, noun_name)

    def add_no_noun(self, verb: Any) -> None:
        self.add_relation(verb, NO_NOUN_TOKEN)

    def allow_no_noun(self, verb: Any) -> bool:
        verb_key = _norm_key(verb)
        if not verb_key:
            return False
        return _norm_key(NO_NOUN_TOKEN) in set(self.relations.get(verb_key) or set())

    def has_verb(self, verb: Any) -> bool:
        return _norm_key(verb) in self.relations

    def is_allowed(self, verb: Any, noun: Any) -> bool:
        verb_key = _norm_key(verb)
        noun_key = _norm_key(noun)
        if not verb_key:
            return False
        allowed = set(self.relations.get(verb_key) or set())
        if not allowed:
            return True
        return noun_key in allowed

    def allowed_noun_names(self, verb: Any) -> List[str]:
        verb_key = _norm_key(verb)
        allowed = [
            self.noun_display.get(noun_key, noun_key)
            for noun_key in sorted(self.relations.get(verb_key, set()))
            if noun_key != _norm_key(NO_NOUN_TOKEN)
        ]
        return [name for name in allowed if _norm_text(name)]

    def allowed_noun_ids(self, verb: Any, name_to_id: Dict[str, Any]) -> List[int]:
        if not name_to_id:
            return []
        verb_key = _norm_key(verb)
        allowed = set(self.relations.get(verb_key) or set())
        if not allowed:
            out = []
            for name, object_id in name_to_id.items():
                try:
                    out.append(int(object_id))
                except Exception:
                    continue
            return sorted(set(out))
        out: List[int] = []
        for name, object_id in name_to_id.items():
            if _norm_key(name) not in allowed:
                continue
            try:
                out.append(int(object_id))
            except Exception:
                continue
        return sorted(set(out))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_hint": self.format_hint,
            "relations": {
                self.verb_display.get(verb_key, verb_key): [
                    self.noun_display.get(noun_key, noun_key)
                    for noun_key in sorted(list(noun_keys or set()))
                ]
                for verb_key, noun_keys in sorted(self.relations.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "HOIOntology":
        if not isinstance(payload, dict):
            return cls()
        out = cls(
            source_path=_norm_text(payload.get("source_path")),
            format_hint=_norm_text(payload.get("format_hint")),
        )
        relations = payload.get("relations", {}) or {}
        if isinstance(relations, dict):
            for verb_name, noun_names in relations.items():
                verb_text = _norm_text(verb_name)
                if not verb_text:
                    continue
                if isinstance(noun_names, (list, tuple, set)):
                    for noun_name in list(noun_names):
                        out.add_relation(verb_text, noun_name)
        return out

    @classmethod
    def from_csv(cls, path: str) -> "HOIOntology":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            return cls(source_path=path)

        header = [_norm_text(cell) for cell in rows[0]]
        norm_header = [_norm_key(cell) for cell in header]
        ontology = cls(source_path=path)

        if {"verb", "noun", "allowed"}.issubset(set(norm_header)):
            ontology.format_hint = "long_form"
            verb_idx = norm_header.index("verb")
            noun_idx = norm_header.index("noun")
            allowed_idx = norm_header.index("allowed")
            for row in rows[1:]:
                if not row:
                    continue
                verb_name = row[verb_idx] if verb_idx < len(row) else ""
                noun_name = row[noun_idx] if noun_idx < len(row) else ""
                allowed = row[allowed_idx] if allowed_idx < len(row) else ""
                if not _safe_bool01(allowed):
                    continue
                noun_text = _norm_text(noun_name) or NO_NOUN_TOKEN
                ontology.add_relation(verb_name, noun_text)
            return ontology

        ontology.format_hint = "matrix"
        if not header:
            return ontology
        noun_headers = [_norm_text(cell) for cell in header[1:]]
        for row in rows[1:]:
            if not row:
                continue
            verb_name = row[0] if row else ""
            if not _norm_text(verb_name):
                continue
            for idx, noun_name in enumerate(noun_headers, start=1):
                allowed = row[idx] if idx < len(row) else ""
                if not _safe_bool01(allowed):
                    continue
                noun_text = _norm_text(noun_name) or NO_NOUN_TOKEN
                ontology.add_relation(verb_name, noun_text)
        return ontology


def ontology_noun_required(
    ontology: Optional[HOIOntology],
    verb_name: Any,
) -> bool:
    text = _norm_text(verb_name)
    if not text:
        return False
    if ontology is None or not ontology.has_verb(text):
        return True
    return not ontology.allow_no_noun(text)


def ontology_allowed_noun_ids(
    ontology: Optional[HOIOntology],
    verb_name: Any,
    name_to_id: Dict[str, Any],
) -> List[int]:
    if ontology is None:
        out: List[int] = []
        for object_id in name_to_id.values():
            try:
                out.append(int(object_id))
            except Exception:
                continue
        return sorted(set(out))
    return ontology.allowed_noun_ids(verb_name, name_to_id)


def filter_allowed_object_candidates(
    candidates: Sequence[Dict[str, Any]],
    allowed_ids: Iterable[int],
) -> List[Dict[str, Any]]:
    allowed = {int(v) for v in list(allowed_ids or [])}
    if not allowed:
        return [dict(item) for item in list(candidates or []) if isinstance(item, dict)]
    out: List[Dict[str, Any]] = []
    for item in list(candidates or []):
        if not isinstance(item, dict):
            continue
        try:
            object_id = int(item.get("object_id"))
        except Exception:
            continue
        if object_id in allowed:
            out.append(dict(item))
    return out
