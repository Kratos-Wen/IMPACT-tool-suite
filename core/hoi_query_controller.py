from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from core.hoi_completion import build_onset_centric_completion


FIELD_NAMES = (
    "interaction_start",
    "functional_contact_onset",
    "interaction_end",
    "verb",
    "noun_object_id",
)

FIELD_ALIASES = {
    "target_object_id": "noun_object_id",
}

FIELD_LABELS = {
    "interaction_start": "start",
    "functional_contact_onset": "onset",
    "interaction_end": "end",
    "verb": "verb",
    "noun_object_id": "noun",
    "target_object_id": "noun",
    "instrument_object_id": "instrument",
    "tool_object_id": "instrument",
    "anomaly_label": "anomaly",
}

TEMPORAL_FIELDS = (
    "interaction_start",
    "functional_contact_onset",
    "interaction_end",
)

OBJECT_FIELDS = (
    "noun_object_id",
    "target_object_id",
    "instrument_object_id",
    "tool_object_id",
)

EVIDENCE_TIME_FIELDS = (
    ("start", "interaction_start"),
    ("onset", "functional_contact_onset"),
    ("end", "interaction_end"),
)

EVIDENCE_ROLE_FIELDS = (
    ("noun", "noun_object_id"),
    ("instrument", "instrument_object_id"),
)

DEFAULT_COST = {
    "temporal": 0.28,
    "semantic": 0.22,
    "object": 0.52,
    "review": 0.34,
}

DEFAULT_PROPAGATION = {
    "temporal": 0.90,
    "semantic": 0.72,
    "object": 0.66,
    "review": 0.78,
}

DEFAULT_RISK = {
    "temporal": 0.10,
    "semantic": 0.14,
    "object": 0.42,
    "review": 0.16,
}

DEFAULT_AUTHORITY_POLICY = {
    "name": "default",
    "completion_bundle": {
        "safe_local_enabled": True,
        "safe_local_min_reliability": 0.80,
        "safe_local_max_risk": 0.15,
        "safe_local_min_acceptance": 0.58,
        "human_confirm_min_reliability": 0.48,
    },
    "suggestion": {
        "manual_review_min_support": 6,
        "manual_review_max_acceptance": 0.35,
        "safe_local_downgrade_max_risk": 0.32,
    },
    "execution": {
        "safe_local_max_risk": 0.18,
        "safe_local_min_reliability": 0.78,
        "safe_local_min_support_if_available": 2,
    },
}

AUTHORITY_POLICY_PRESETS = {
    "default": {},
    "conservative": {
        "name": "conservative",
        "completion_bundle": {
            "safe_local_min_reliability": 0.86,
            "safe_local_max_risk": 0.12,
            "safe_local_min_acceptance": 0.64,
            "human_confirm_min_reliability": 0.54,
        },
        "suggestion": {
            "manual_review_min_support": 5,
            "manual_review_max_acceptance": 0.40,
            "safe_local_downgrade_max_risk": 0.26,
        },
        "execution": {
            "safe_local_max_risk": 0.15,
            "safe_local_min_reliability": 0.82,
            "safe_local_min_support_if_available": 2,
        },
    },
    "no_safe_local": {
        "name": "no_safe_local",
        "completion_bundle": {
            "safe_local_enabled": False,
            "human_confirm_min_reliability": 0.48,
        },
    },
}

_UNSET = object()


def _now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _is_handtrack_source(source: Any) -> bool:
    return _safe_text(source).lower().startswith("handtrack_once")


def _canonical_field_name(field_name: str) -> str:
    text = _safe_text(field_name)
    return FIELD_ALIASES.get(text, text)


def _sync_hand_aliases(hand_data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(hand_data, dict):
        return {}
    noun_value = hand_data.get("noun_object_id", hand_data.get("target_object_id"))
    hand_data["noun_object_id"] = _safe_int(noun_value)
    hand_data["target_object_id"] = _safe_int(noun_value)
    instrument_value = hand_data.get(
        "instrument_object_id",
        hand_data.get("tool_object_id"),
    )
    hand_data["instrument_object_id"] = _safe_int(instrument_value)
    hand_data["tool_object_id"] = _safe_int(instrument_value)
    return hand_data


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base or {})
    for key, value in dict(override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(dict(out.get(key) or {}), value)
        else:
            out[key] = value
    return out


def resolve_authority_policy(authority_policy: Optional[Any] = None) -> Dict[str, Any]:
    policy = _deep_merge_dict(DEFAULT_AUTHORITY_POLICY, {})
    if authority_policy is None:
        return policy
    if isinstance(authority_policy, str):
        preset = _deep_merge_dict({}, AUTHORITY_POLICY_PRESETS.get(_safe_text(authority_policy).lower(), {}))
        return _deep_merge_dict(policy, preset)
    if isinstance(authority_policy, dict):
        preset_name = _safe_text(authority_policy.get("preset")).lower()
        if preset_name:
            policy = _deep_merge_dict(policy, AUTHORITY_POLICY_PRESETS.get(preset_name, {}))
        policy = _deep_merge_dict(policy, authority_policy)
        return policy
    return policy


def evidence_slot_name(role: str, point: str) -> str:
    role_txt = _safe_text(role).lower() or "object"
    point_txt = _safe_text(point).lower() or "frame"
    return f"{role_txt}_{point_txt}"


def normalize_sparse_evidence_state(raw: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        slot = _safe_text(key)
        if not slot:
            continue
        out[slot] = {
            "status": _safe_text(value.get("status")) or "unknown",
            "source": _safe_text(value.get("source")) or "unknown",
            "frame": _safe_int(value.get("frame")),
            "role": _safe_text(value.get("role")),
            "time_key": _safe_text(value.get("time_key")),
            "time_label": _safe_text(value.get("time_label")),
            "object_id": _safe_int(value.get("object_id")),
            "object_name": _safe_text(value.get("object_name")),
            "note": _safe_text(value.get("note")),
        }
    return out


def _field_is_empty(field_name: str, value: Any) -> bool:
    if field_name == "verb":
        return not _safe_text(value)
    return value is None


def _normalize_state_entry(field_name: str, entry: Any, current_value: Any) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        entry = {}
    status = _safe_text(entry.get("status"))
    source = _safe_text(entry.get("source"))
    if status == "empty":
        status = "missing"
    if source == "empty":
        source = "missing"
    if not status:
        status = "missing" if _field_is_empty(field_name, current_value) else "confirmed"
    if not source:
        source = "unknown"
    return {
        "status": status,
        "source": source,
        "updated_at": _safe_text(entry.get("updated_at")) or _now_text(),
        "note": _safe_text(entry.get("note")),
        "value": entry.get("value", current_value),
    }


def _normalize_suggestion_entry(entry: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, dict):
        return None
    meta = dict(entry.get("meta") or {}) if isinstance(entry.get("meta"), dict) else {}
    display_value = _safe_text(entry.get("display_value")) or _safe_text(
        meta.get("display_value")
    )
    explicit_empty = bool(meta.get("explicit_empty"))
    if (
        entry.get("value") is None
        and not _safe_text(entry.get("value"))
        and not explicit_empty
        and not display_value
    ):
        return None
    confidence = entry.get("confidence")
    try:
        confidence = None if confidence is None else float(confidence)
    except Exception:
        confidence = None
    out = {
        "value": entry.get("value"),
        "display_value": display_value,
        "source": _safe_text(entry.get("source")) or "unknown",
        "status": _safe_text(entry.get("status")) or "suggested",
        "created_at": _safe_text(entry.get("created_at")) or _now_text(),
        "reason": _safe_text(entry.get("reason")),
        "confidence": confidence,
        "safe_to_apply": bool(entry.get("safe_to_apply", True)),
        "review_recommended": bool(entry.get("review_recommended", True)),
        "meta": meta,
    }
    return out


def _flatten_noun_source_decision(meta: Any) -> Dict[str, Any]:
    if not isinstance(meta, dict):
        return {}
    decision = dict(meta.get("source_decision") or {})
    if not decision:
        return {}
    return {
        "noun_primary_source": _safe_text(decision.get("preferred_source")),
        "noun_primary_family": _safe_text(decision.get("preferred_family")),
        "noun_source_margin": _safe_float(decision.get("score_margin")),
        "semantic_source_acceptance_est": _safe_float(
            decision.get("source_a_acceptance")
        ),
        "semantic_source_score": _safe_float(decision.get("source_a_score")),
        "semantic_source_support": int(decision.get("source_a_support", 0) or 0),
        "detector_source_acceptance_est": _safe_float(
            decision.get("source_b_acceptance")
        ),
        "detector_source_score": _safe_float(decision.get("source_b_score")),
        "detector_source_support": int(decision.get("source_b_support", 0) or 0),
        "noun_source_decision_basis": _safe_text(decision.get("decision_basis")),
    }


def ensure_hand_annotation_state(hand_data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(hand_data, dict):
        return {}
    _sync_hand_aliases(hand_data)
    raw_state = hand_data.get("_field_state")
    if not isinstance(raw_state, dict):
        raw_state = {}
    raw_suggestions = hand_data.get("_field_suggestions")
    if not isinstance(raw_suggestions, dict):
        raw_suggestions = {}
    if "noun_object_id" not in raw_state and "target_object_id" in raw_state:
        raw_state["noun_object_id"] = raw_state.get("target_object_id")
    if "noun_object_id" not in raw_suggestions and "target_object_id" in raw_suggestions:
        raw_suggestions["noun_object_id"] = raw_suggestions.get("target_object_id")

    state: Dict[str, Dict[str, Any]] = {}
    for field_name in FIELD_NAMES:
        state[field_name] = _normalize_state_entry(
            field_name, raw_state.get(field_name), hand_data.get(field_name)
        )

    suggestions: Dict[str, Dict[str, Any]] = {}
    for field_name in FIELD_NAMES:
        row = _normalize_suggestion_entry(raw_suggestions.get(field_name))
        if row is not None:
            suggestions[field_name] = row

    hand_data["_field_state"] = state
    hand_data["_field_suggestions"] = suggestions
    _sync_hand_aliases(hand_data)
    return hand_data


def hydrate_existing_field_state(
    hand_data: Dict[str, Any], default_source: str = "loaded_annotation"
) -> Dict[str, Any]:
    hand_data = ensure_hand_annotation_state(hand_data)
    state = hand_data.get("_field_state", {})
    for field_name in FIELD_NAMES:
        value = hand_data.get(field_name)
        row = dict(state.get(field_name, {}))
        if _field_is_empty(field_name, value):
            if row.get("status") not in ("suggested", "confirmed"):
                row["status"] = "missing"
                row["source"] = row.get("source") or "missing"
        else:
            if row.get("status") in ("", None, "empty", "missing"):
                row["status"] = "confirmed"
                row["source"] = default_source
                row["updated_at"] = _now_text()
                row["value"] = value
        state[field_name] = _normalize_state_entry(field_name, row, value)
    hand_data["_field_state"] = state
    return hand_data


def get_field_state(hand_data: Dict[str, Any], field_name: str) -> Dict[str, Any]:
    field_name = _canonical_field_name(field_name)
    ensure_hand_annotation_state(hand_data)
    return dict((hand_data.get("_field_state") or {}).get(field_name, {}))


def get_field_suggestion(hand_data: Dict[str, Any], field_name: str) -> Dict[str, Any]:
    field_name = _canonical_field_name(field_name)
    ensure_hand_annotation_state(hand_data)
    return dict((hand_data.get("_field_suggestions") or {}).get(field_name, {}))


def field_blocks_automation(hand_data: Dict[str, Any], field_name: str) -> bool:
    state = get_field_state(hand_data, field_name)
    return _safe_text(state.get("status")).lower() in {"confirmed", "locked"}


def set_field_confirmation(
    hand_data: Dict[str, Any],
    field_name: str,
    *,
    source: str,
    value: Any = _UNSET,
    status: str = "confirmed",
    note: str = "",
) -> Dict[str, Any]:
    field_name = _canonical_field_name(field_name)
    ensure_hand_annotation_state(hand_data)
    if value is not _UNSET:
        hand_data[field_name] = value
        if field_name == "noun_object_id":
            hand_data["target_object_id"] = value
    current_value = hand_data.get(field_name)
    hand_data["_field_state"][field_name] = {
        "status": _safe_text(status) or "confirmed",
        "source": _safe_text(source) or "unknown",
        "updated_at": _now_text(),
        "note": _safe_text(note),
        "value": current_value,
    }
    if status == "confirmed":
        hand_data["_field_suggestions"].pop(field_name, None)
    return dict(hand_data["_field_state"][field_name])


def set_field_suggestion(
    hand_data: Dict[str, Any],
    field_name: str,
    value: Any,
    *,
    source: str,
    confidence: Optional[float] = None,
    reason: str = "",
    safe_to_apply: bool = True,
    review_recommended: bool = True,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    field_name = _canonical_field_name(field_name)
    ensure_hand_annotation_state(hand_data)
    if field_blocks_automation(hand_data, field_name):
        hand_data["_field_suggestions"].pop(field_name, None)
        return {}
    hand_data["_field_suggestions"][field_name] = {
        "value": value,
        "source": _safe_text(source) or "unknown",
        "status": "suggested",
        "created_at": _now_text(),
        "reason": _safe_text(reason),
        "confidence": confidence,
        "safe_to_apply": bool(safe_to_apply),
        "review_recommended": bool(review_recommended),
        "meta": dict(meta or {}),
    }
    return dict(hand_data["_field_suggestions"][field_name])


def clear_field_suggestion(hand_data: Dict[str, Any], field_name: str) -> None:
    field_name = _canonical_field_name(field_name)
    ensure_hand_annotation_state(hand_data)
    hand_data["_field_suggestions"].pop(field_name, None)


def clear_field_value(
    hand_data: Dict[str, Any], field_name: str, *, source: str = "manual_clear"
) -> None:
    field_name = _canonical_field_name(field_name)
    ensure_hand_annotation_state(hand_data)
    hand_data[field_name] = "" if field_name == "verb" else None
    if field_name == "noun_object_id":
        hand_data["target_object_id"] = None
    hand_data["_field_state"][field_name] = {
        "status": "missing",
        "source": _safe_text(source) or "manual_clear",
        "updated_at": _now_text(),
        "note": "",
        "value": hand_data[field_name],
    }
    hand_data["_field_suggestions"].pop(field_name, None)


def apply_field_suggestion(
    hand_data: Dict[str, Any],
    field_name: str,
    *,
    source: Optional[str] = None,
    as_status: str = "suggested",
) -> bool:
    field_name = _canonical_field_name(field_name)
    ensure_hand_annotation_state(hand_data)
    if field_blocks_automation(hand_data, field_name):
        return False
    suggestion = get_field_suggestion(hand_data, field_name)
    if not suggestion:
        return False
    set_field_confirmation(
        hand_data,
        field_name,
        source=source or suggestion.get("source") or "suggested_apply",
        value=suggestion.get("value"),
        status=as_status,
        note=suggestion.get("reason", ""),
    )
    return True


def _score_query(
    *,
    base_priority: float,
    surface: str,
    selected_bonus: float = 0.0,
    propagation_gain: Optional[float] = None,
    human_cost: Optional[float] = None,
    overwrite_risk: Optional[float] = None,
) -> Dict[str, float]:
    propagation = (
        DEFAULT_PROPAGATION.get(surface, 0.5)
        if propagation_gain is None
        else max(0.0, min(1.0, float(propagation_gain)))
    )
    cost = (
        DEFAULT_COST.get(surface, 0.4)
        if human_cost is None
        else max(0.0, min(1.0, float(human_cost)))
    )
    risk = (
        DEFAULT_RISK.get(surface, 0.2)
        if overwrite_risk is None
        else max(0.0, min(1.0, float(overwrite_risk)))
    )
    voi = float(base_priority + 0.24 * propagation - 0.14 * cost - 0.22 * risk + selected_bonus)
    return {
        "base_priority": float(base_priority),
        "propagation_gain": float(propagation),
        "human_cost_est": float(cost),
        "overwrite_risk": float(risk),
        "voi_score": float(voi),
    }


def _apply_authority_policy(
    query: Dict[str, Any],
    *,
    authority_policy: Optional[Any] = None,
) -> Dict[str, Any]:
    out = dict(query or {})
    policy = resolve_authority_policy(
        authority_policy
        if authority_policy is not None
        else out.get("authority_policy")
    )
    completion_policy = dict(policy.get("completion_bundle") or {})
    suggestion_policy = dict(policy.get("suggestion") or {})
    action_kind = _safe_text(out.get("action_kind")).lower() or "query"
    authority = _safe_text(out.get("authority_level")).lower() or "human_only"
    apply_mode = _safe_text(out.get("apply_mode"))
    risk = max(0.0, min(1.0, float(out.get("overwrite_risk", 0.0) or 0.0)))
    acceptance = max(0.0, min(1.0, float(out.get("acceptance_prob_est", 0.0) or 0.0)))
    reliability = max(
        0.0,
        min(
            1.0,
            float(
                out.get("calibrated_reliability", out.get("acceptance_prob_est", 0.0))
                or 0.0
            ),
        ),
    )
    support = int(out.get("empirical_support_n", 0) or 0)

    note = ""
    reason_code = ""
    if apply_mode == "apply_completion_bundle":
        safe_local_enabled = bool(completion_policy.get("safe_local_enabled", True))
        safe_local_min_reliability = float(
            completion_policy.get("safe_local_min_reliability", 0.80) or 0.80
        )
        safe_local_max_risk = float(
            completion_policy.get("safe_local_max_risk", 0.15) or 0.15
        )
        safe_local_min_acceptance = float(
            completion_policy.get("safe_local_min_acceptance", 0.58) or 0.58
        )
        human_confirm_min_reliability = float(
            completion_policy.get("human_confirm_min_reliability", 0.48) or 0.48
        )
        if (
            safe_local_enabled
            and reliability >= safe_local_min_reliability
            and risk <= safe_local_max_risk
            and acceptance >= safe_local_min_acceptance
        ):
            out["action_kind"] = "propagate"
            out["authority_level"] = "safe_local"
            out["interaction_form"] = "bundle_accept"
            out["safe_apply"] = True
            note = "Calibrated reliability and risk support safe local completion."
            reason_code = "bundle_safe_local"
        elif reliability >= human_confirm_min_reliability:
            out["action_kind"] = "suggest"
            out["authority_level"] = "human_confirm"
            out["interaction_form"] = "bundle_accept"
            out["safe_apply"] = True
            note = "Completion remains local but now requires human confirmation."
            reason_code = "bundle_human_confirm"
        else:
            out["action_kind"] = "query"
            out["authority_level"] = "human_only"
            out["interaction_form"] = "manual_edit"
            out["apply_mode"] = ""
            out["safe_apply"] = False
            note = "Completion confidence is too low for automated execution."
            reason_code = "bundle_manual_only_low_reliability"
    elif action_kind == "suggest":
        manual_review_min_support = int(
            suggestion_policy.get("manual_review_min_support", 6) or 6
        )
        manual_review_max_acceptance = float(
            suggestion_policy.get("manual_review_max_acceptance", 0.35) or 0.35
        )
        safe_local_downgrade_max_risk = float(
            suggestion_policy.get("safe_local_downgrade_max_risk", 0.32) or 0.32
        )
        if support >= manual_review_min_support and acceptance < manual_review_max_acceptance:
            out["action_kind"] = "query"
            out["authority_level"] = "human_only"
            out["interaction_form"] = "manual_edit"
            out["apply_mode"] = ""
            out["safe_apply"] = False
            note = "Empirical acceptance is low; direct manual review is safer."
            reason_code = "suggest_manual_only_low_acceptance"
        elif risk >= safe_local_downgrade_max_risk and authority == "safe_local":
            out["action_kind"] = "suggest"
            out["authority_level"] = "human_confirm"
            out["interaction_form"] = "accept_suggestion"
            out["safe_apply"] = True
            note = "High overwrite risk prevents automatic authority transfer."
            reason_code = "suggest_downgraded_risk"
    if note:
        out["authority_policy_reason"] = note
    out["authority_policy_code"] = reason_code or "policy_passthrough"
    out["authority_policy_name"] = _safe_text(policy.get("name")) or "default"
    out["authority_policy_thresholds"] = {
        "safe_local_max_risk": float(
            dict(policy.get("execution") or {}).get("safe_local_max_risk", 0.18) or 0.18
        ),
        "safe_local_min_reliability": float(
            dict(policy.get("execution") or {}).get("safe_local_min_reliability", 0.78)
            or 0.78
        ),
        "safe_local_min_support_if_available": int(
            dict(policy.get("execution") or {}).get(
                "safe_local_min_support_if_available", 2
            )
            or 2
        ),
    }
    return out


def build_query_candidates(
    hand_rows: Sequence[Dict[str, Any]],
    *,
    selected_event_id: Optional[int] = None,
    selected_hand: Optional[str] = None,
    calibrator: Optional[Any] = None,
    authority_policy: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    queries: List[Dict[str, Any]] = []

    for row in list(hand_rows or []):
        if not isinstance(row, dict):
            continue
        event_id = row.get("event_id")
        hand = _safe_text(row.get("hand"))
        selected_bonus = 0.0
        if event_id == selected_event_id:
            selected_bonus += 0.04
        if hand and hand == _safe_text(selected_hand):
            selected_bonus += 0.03

        field_state = dict(row.get("field_state") or {})
        field_suggestions = dict(row.get("field_suggestions") or {})
        if "noun_object_id" not in field_state and "target_object_id" in field_state:
            field_state["noun_object_id"] = dict(field_state.get("target_object_id") or {})
        if "noun_object_id" not in field_suggestions and "target_object_id" in field_suggestions:
            field_suggestions["noun_object_id"] = dict(field_suggestions.get("target_object_id") or {})
        bbox_errors = list(row.get("bbox_errors") or [])
        sparse_evidence_state = normalize_sparse_evidence_state(
            row.get("sparse_evidence_state")
        )
        consistency_flags = list(row.get("consistency_flags") or [])
        videomae_candidates = list(row.get("videomae_candidates") or [])
        handtrack_prior = dict(row.get("handtrack_prior") or {})
        handtrack_onset_frame = _safe_int(handtrack_prior.get("onset_frame"))
        has_handtrack_onset_prior = handtrack_onset_frame is not None
        local_completion = build_onset_centric_completion(row, calibrator=calibrator)
        local_suggested_field_names = {
            _canonical_field_name(str(item.get("field_name") or "").strip())
            for item in list(local_completion.get("suggested_fields") or [])
            if isinstance(item, dict)
        }

        start = row.get("interaction_start")
        onset = row.get("functional_contact_onset")
        end = row.get("interaction_end")
        verb = _safe_text(row.get("verb"))
        target = row.get("noun_object_id", row.get("target_object_id"))
        noun_only_mode = True
        noun_required = bool(row.get("noun_required"))

        def _review_recommended_for(field_name: str) -> bool:
            suggestion = dict(field_suggestions.get(field_name) or {})
            if not suggestion:
                return True
            return bool(suggestion.get("review_recommended", True))

        def _append(query: Dict[str, Any]) -> None:
            query.setdefault("event_id", event_id)
            query.setdefault("hand", hand)
            query.setdefault("annotator_id", row.get("annotator_id") or "")
            query.setdefault("selected_bonus", selected_bonus)
            apply_mode = _safe_text(query.get("apply_mode"))
            safe_apply = bool(query.get("safe_apply"))
            if not _safe_text(query.get("action_kind")):
                if apply_mode == "apply_completion_bundle":
                    query["action_kind"] = "propagate"
                elif safe_apply:
                    query["action_kind"] = "suggest"
                else:
                    query["action_kind"] = "query"
            if not _safe_text(query.get("authority_level")):
                if apply_mode == "apply_completion_bundle":
                    query["authority_level"] = "safe_local"
                elif safe_apply:
                    query["authority_level"] = "human_confirm"
                else:
                    query["authority_level"] = "human_only"
            if not _safe_text(query.get("interaction_form")):
                if apply_mode == "confirm_current":
                    query["interaction_form"] = "confirm_current"
                elif apply_mode == "apply_completion_bundle":
                    query["interaction_form"] = "bundle_accept"
                elif apply_mode == "apply_suggestion":
                    query["interaction_form"] = "accept_suggestion"
                elif str(query.get("field_name") or "").strip() == "bbox":
                    query["interaction_form"] = "draw_keyframe_box"
                elif str(query.get("field_name") or "").strip() in OBJECT_FIELDS:
                    query["interaction_form"] = "choose_object"
                elif str(query.get("surface") or "").strip().lower() == "review":
                    query["interaction_form"] = "review_conflict"
                else:
                    query["interaction_form"] = "manual_edit"
            if calibrator is not None:
                try:
                    query = calibrator.calibrate_query(query)
                except Exception:
                    pass
            query = _apply_authority_policy(query, authority_policy=authority_policy)
            query.setdefault(
                "sparse_evidence_summary",
                dict(row.get("sparse_evidence_summary") or {}),
            )
            query.setdefault(
                "query_id",
                "|".join(
                    [
                        str(event_id),
                        hand or "",
                        str(query.get("query_type") or ""),
                        str(query.get("field_name") or ""),
                        str(query.get("action_kind") or ""),
                        str(query.get("authority_level") or ""),
                        str(query.get("target_frame") or ""),
                        str(query.get("target_slot") or ""),
                        str(query.get("suggested_value") or ""),
                        ",".join(
                            f"{_safe_text(item.get('field_name'))}:{item.get('value')}"
                            for item in list(query.get("completion_fields") or [])
                            if isinstance(item, dict)
                        ),
                    ]
                ),
            )
            queries.append(query)

        safe_completion_fields = list(local_completion.get("safe_fields") or [])
        if safe_completion_fields:
            safe_handtrack_onset = next(
                (
                    dict(item)
                    for item in safe_completion_fields
                    if str(item.get("field_name") or "").strip() == "functional_contact_onset"
                    and _is_handtrack_source(item.get("source"))
                ),
                {},
            )
            completion_names = [
                FIELD_LABELS.get(str(item.get("field_name") or ""), str(item.get("field_name") or "field"))
                for item in safe_completion_fields
            ]
            completion_score = _score_query(
                base_priority=0.99 if safe_handtrack_onset else (0.97 if "functional_contact_onset" in [str(item.get("field_name")) for item in safe_completion_fields] else 0.91),
                surface="temporal" if any(str(item.get("field_name")) in TEMPORAL_FIELDS for item in safe_completion_fields) else "semantic",
                selected_bonus=selected_bonus,
                propagation_gain=0.98 if safe_handtrack_onset else 0.97,
                human_cost=0.12 if safe_handtrack_onset else 0.16,
                overwrite_risk=0.03 if safe_handtrack_onset else 0.04,
            )
            _append(
                {
                    "query_type": "complete_local_event",
                    "surface": "event",
                    "field_name": "completion_bundle",
                    "summary": local_completion.get("summary")
                    or "Apply onset-centered local completion.",
                    "reason": local_completion.get("reason")
                    or "Onset-centered structured completion can safely fill unresolved local event fields.",
                    "target_frame": local_completion.get("anchor_frame"),
                    "safe_apply": True,
                    "apply_mode": "apply_completion_bundle",
                    "completion_fields": safe_completion_fields,
                    "target_slot": ",".join(completion_names),
                    "suggested_source": (
                        safe_handtrack_onset.get("source")
                        or next(
                            (
                                str(item.get("source") or "").strip()
                                for item in safe_completion_fields
                                if str(item.get("source") or "").strip()
                            ),
                            "",
                        )
                    ),
                    "hand_conditioned": bool(safe_handtrack_onset),
                    "handtrack_prior": dict(handtrack_prior) if safe_handtrack_onset else {},
                    **completion_score,
                }
            )

        suggested_temporal_fields = [
            dict(item)
            for item in list(local_completion.get("suggested_fields") or [])
            if str(item.get("field_name") or "").strip() in TEMPORAL_FIELDS
            and not bool(item.get("safe_to_apply"))
        ]
        for time_item in suggested_temporal_fields:
            field_name = str(time_item.get("field_name") or "").strip()
            field_label = FIELD_LABELS.get(field_name, field_name)
            suggested_value = _safe_int(time_item.get("value"))
            decision_hint = _safe_text(time_item.get("decision_hint")).lower()
            suggested_source = _safe_text(time_item.get("source"))
            is_handtrack_onset = (
                field_name == "functional_contact_onset"
                and _is_handtrack_source(suggested_source)
            )
            action_kind = "suggest" if decision_hint == "human_confirm" else "query"
            authority_level = (
                "human_confirm" if decision_hint == "human_confirm" else "human_only"
            )
            apply_mode = "apply_suggestion" if decision_hint == "human_confirm" else ""
            base_priority = (
                0.96
                if is_handtrack_onset
                else (0.87 if field_name == "interaction_start" else 0.85)
            )
            score = _score_query(
                base_priority=base_priority,
                surface="temporal",
                selected_bonus=selected_bonus,
                propagation_gain=0.96 if is_handtrack_onset else (0.88 if field_name == "interaction_start" else 0.84),
                human_cost=0.20 if is_handtrack_onset else 0.28,
                overwrite_risk=0.05 if is_handtrack_onset else 0.09,
            )
            _append(
                {
                    "query_type": (
                        "confirm_hand_conditioned_onset"
                        if is_handtrack_onset and decision_hint == "human_confirm"
                        else (
                            "review_hand_conditioned_onset"
                            if is_handtrack_onset
                            else f"complete_{field_name}"
                        )
                    ),
                    "surface": "event",
                    "field_name": field_name,
                    "summary": (
                        (
                            f"Confirm hand-conditioned onset prior at frame {suggested_value}."
                            if decision_hint == "human_confirm" and suggested_value is not None
                            else (
                                f"Review hand-conditioned onset prior near frame {suggested_value}."
                                if suggested_value is not None
                                else "Review hand-conditioned onset prior."
                            )
                        )
                        if is_handtrack_onset
                        else (
                            f"Confirm suggested {field_label} at frame {suggested_value}."
                            if decision_hint == "human_confirm" and suggested_value is not None
                            else (
                                f"Review {field_label} near frame {suggested_value}."
                                if suggested_value is not None
                                else f"Review {field_label}."
                            )
                        )
                    ),
                    "reason": _safe_text(time_item.get("reason"))
                    or (
                        "Persistent hand-track motion proposed a hand-conditioned onset prior that should be confirmed before downstream semantic review."
                        if is_handtrack_onset
                        else "Onset-centered local completion proposed a temporal field that still needs manual review."
                    ),
                    "target_frame": local_completion.get("anchor_frame")
                    or suggested_value
                    or row.get("functional_contact_onset")
                    or row.get("interaction_start")
                    or row.get("interaction_end"),
                    "safe_apply": False,
                    "apply_mode": apply_mode,
                    "suggested_value": suggested_value,
                    "suggested_source": suggested_source,
                    "suggested_confidence": time_item.get("confidence"),
                    "action_kind": action_kind,
                    "authority_level": authority_level,
                    "calibrated_reliability": time_item.get("calibrated_reliability"),
                    "acceptance_prob_est": time_item.get("acceptance_prior"),
                    "empirical_support_n": time_item.get("empirical_support"),
                    "hand_conditioned": bool(is_handtrack_onset),
                    "handtrack_prior": dict(handtrack_prior) if is_handtrack_onset else {},
                    **score,
                }
            )

        onset_state = dict(field_state.get("functional_contact_onset") or {})
        onset_suggestion = dict(field_suggestions.get("functional_contact_onset") or {})
        onset_source = (
            onset_state.get("source")
            or onset_suggestion.get("source")
            or row.get("onset_source")
        )
        onset_is_handtrack = _is_handtrack_source(onset_source)
        if (
            onset_state.get("status") == "suggested"
            and onset is not None
            and _review_recommended_for("functional_contact_onset")
        ):
            score = _score_query(
                base_priority=0.96 if onset_is_handtrack else 0.93,
                surface="temporal",
                selected_bonus=selected_bonus,
                propagation_gain=0.96 if onset_is_handtrack else None,
                human_cost=0.20 if onset_is_handtrack else None,
                overwrite_risk=0.05 if onset_is_handtrack else 0.06,
            )
            _append(
                {
                    "query_type": "confirm_hand_conditioned_onset" if onset_is_handtrack else "confirm_suggested_onset",
                    "surface": "event",
                    "field_name": "functional_contact_onset",
                    "summary": (
                        f"Confirm hand-conditioned onset prior at frame {int(onset)}."
                        if onset_is_handtrack
                        else f"Confirm onset at frame {int(onset)}."
                    ),
                    "reason": onset_suggestion.get("reason")
                    or (
                        "Persistent hand-track motion suggested the current onset and downstream semantic decode now depends on confirming it."
                        if onset_is_handtrack
                        else "Onset was suggested and should be confirmed before downstream review."
                    ),
                    "target_frame": int(onset),
                    "safe_apply": True,
                    "apply_mode": "confirm_current",
                    "suggested_value": onset,
                    "suggested_source": onset_source,
                    "action_kind": "suggest",
                    "authority_level": "human_confirm",
                    "hand_conditioned": bool(onset_is_handtrack),
                    "handtrack_prior": dict(handtrack_prior) if onset_is_handtrack else {},
                    **score,
                }
            )
        elif onset is None and start is not None and end is not None and not has_handtrack_onset_prior:
            midpoint = int(round((int(start) + int(end)) / 2.0))
            score = _score_query(
                base_priority=0.95,
                surface="temporal",
                selected_bonus=selected_bonus,
                propagation_gain=0.96,
                overwrite_risk=0.05,
            )
            _append(
                {
                    "query_type": "fill_missing_onset",
                    "surface": "event",
                    "field_name": "functional_contact_onset",
                    "summary": f"Add onset between start {int(start)} and end {int(end)}.",
                    "reason": "Onset anchors the sparse-evidence protocol and contracts the feasible event graph.",
                    "target_frame": midpoint,
                    "safe_apply": True,
                    "apply_mode": "apply_suggestion",
                    "suggested_value": midpoint,
                    "suggested_source": "midpoint_inference",
                    "action_kind": "suggest",
                    "authority_level": "human_confirm",
                    **score,
                }
            )

        verb_state = dict(field_state.get("verb") or {})
        verb_suggestion = dict(field_suggestions.get("verb") or {})
        if (
            verb_state.get("status") == "suggested"
            and verb
            and _review_recommended_for("verb")
        ):
            score = _score_query(
                base_priority=0.89,
                surface="semantic",
                selected_bonus=selected_bonus,
                overwrite_risk=0.07,
            )
            _append(
                {
                    "query_type": "confirm_suggested_verb",
                    "surface": "event",
                    "field_name": "verb",
                    "summary": f"Confirm suggested verb '{verb}'.",
                    "reason": verb_suggestion.get("reason")
                    or "The current action label was suggested by the model and still needs confirmation.",
                    "target_frame": row.get("functional_contact_onset")
                    or row.get("interaction_start")
                    or row.get("interaction_end"),
                    "safe_apply": True,
                    "apply_mode": "confirm_current",
                    "suggested_value": verb,
                    "suggested_source": verb_state.get("source") or verb_suggestion.get("source"),
                    "action_kind": "suggest",
                    "authority_level": "human_confirm",
                    **score,
                }
            )
        elif not verb:
            top_candidate = videomae_candidates[0] if videomae_candidates else {}
            suggested_label = _safe_text(top_candidate.get("label"))
            confidence = top_candidate.get("score")
            score = _score_query(
                base_priority=0.84 if suggested_label else 0.70,
                surface="semantic",
                selected_bonus=selected_bonus,
                propagation_gain=0.74 if suggested_label else 0.48,
                overwrite_risk=0.11 if suggested_label else 0.18,
            )
            _append(
                {
                    "query_type": "fill_missing_verb",
                    "surface": "event",
                    "field_name": "verb",
                    "summary": (
                        f"Confirm verb '{suggested_label}'."
                        if suggested_label
                        else "Choose a verb for this HOI event."
                    ),
                    "reason": (
                        "VideoMAE already produced a candidate and confirming it will reduce semantic ambiguity."
                        if suggested_label
                        else "This event has temporal support but no action label yet."
                    ),
                    "target_frame": row.get("functional_contact_onset")
                    or row.get("interaction_start")
                    or row.get("interaction_end"),
                    "safe_apply": bool(suggested_label),
                    "apply_mode": "apply_suggestion" if suggested_label else "",
                    "suggested_value": suggested_label or None,
                    "suggested_source": "videomae_top1" if suggested_label else "",
                    "suggested_confidence": confidence,
                    "action_kind": "suggest" if suggested_label else "query",
                    "authority_level": "human_confirm" if suggested_label else "human_only",
                    **score,
                }
            )

        if (
            target is None
            and noun_required
            and "noun_object_id" not in local_suggested_field_names
        ):
            score = _score_query(
                base_priority=0.74,
                surface="object",
                selected_bonus=selected_bonus,
                propagation_gain=0.60,
                overwrite_risk=0.20,
            )
            _append(
                {
                    "query_type": "assign_target",
                    "surface": "objects",
                    "field_name": "noun_object_id",
                    "summary": "Assign a noun object.",
                    "reason": "Noun identity is still missing for this event.",
                    "target_frame": row.get("functional_contact_onset")
                    or row.get("interaction_start")
                    or row.get("interaction_end"),
                    "safe_apply": False,
                    "action_kind": "query",
                    "authority_level": "human_only",
                    **score,
                }
            )

        target_suggestion = dict(field_suggestions.get("noun_object_id") or {})
        target_suggestion_meta = dict(target_suggestion.get("meta") or {})
        explicit_no_noun = bool(target_suggestion_meta.get("explicit_empty"))
        if (
            target is None
            and dict(field_state.get("noun_object_id") or {}).get("status") == "suggested"
            and explicit_no_noun
            and _review_recommended_for("noun_object_id")
        ):
            score = _score_query(
                base_priority=0.80,
                surface="object",
                selected_bonus=selected_bonus,
                propagation_gain=0.66,
                human_cost=0.20,
                overwrite_risk=0.06,
            )
            _append(
                {
                    "query_type": "confirm_no_noun_needed",
                    "surface": "objects",
                    "field_name": "noun_object_id",
                    "summary": "Confirm that no noun/object is needed.",
                    "reason": target_suggestion.get("reason")
                    or "The current verb can be completed without a noun/object under the ontology.",
                    "target_frame": row.get("functional_contact_onset")
                    or row.get("interaction_start")
                    or row.get("interaction_end"),
                    "safe_apply": True,
                    "apply_mode": "confirm_current",
                    "suggested_value": None,
                    "suggested_source": target_suggestion.get("source"),
                    "suggested_confidence": target_suggestion.get("confidence"),
                    "action_kind": "suggest",
                    "authority_level": "human_confirm",
                    **score,
                }
            )
        elif (
            target is not None
            and dict(field_state.get("noun_object_id") or {}).get("status") == "suggested"
            and _review_recommended_for("noun_object_id")
        ):
            noun_source_fields = _flatten_noun_source_decision(target_suggestion_meta)
            noun_primary_family = _safe_text(noun_source_fields.get("noun_primary_family")).lower()
            noun_reason = target_suggestion.get("reason") or "The current noun suggestion still benefits from human confirmation."
            if noun_primary_family == "semantic":
                noun_reason = (
                    noun_reason
                    + " Empirical source comparison currently keeps semantic noun decoding as the primary cue and treats detector boxes as grounding support."
                )
            score = _score_query(
                base_priority=0.82,
                surface="object",
                selected_bonus=selected_bonus,
                propagation_gain=0.70,
                human_cost=0.30,
                overwrite_risk=0.08,
            )
            _append(
                {
                    "query_type": "confirm_suggested_noun",
                    "surface": "objects",
                    "field_name": "noun_object_id",
                    "summary": (
                        f"Confirm suggested noun '{target_suggestion.get('display_value') or dict(target_suggestion.get('meta') or {}).get('display_value') or target}'."
                    ),
                    "reason": noun_reason,
                    "target_frame": row.get("functional_contact_onset")
                    or row.get("interaction_start")
                    or row.get("interaction_end"),
                    "safe_apply": True,
                    "apply_mode": "confirm_current",
                    "suggested_value": target,
                    "suggested_source": target_suggestion.get("source"),
                    "suggested_confidence": target_suggestion.get("confidence"),
                    "action_kind": "suggest",
                    "authority_level": "human_confirm",
                    **noun_source_fields,
                    **score,
                }
            )

        suggested_role_fields = [
            dict(item)
            for item in list(local_completion.get("suggested_fields") or [])
            if str(item.get("field_name") or "").strip()
            in ("noun_object_id",)
            and not bool(item.get("safe_to_apply"))
        ]
        for role_item in suggested_role_fields:
            field_name = str(role_item.get("field_name") or "").strip()
            field_label = FIELD_LABELS.get(field_name, field_name)
            display_value = _safe_text(role_item.get("display_value")) or _safe_text(role_item.get("value"))
            role_meta = dict(role_item.get("meta") or {})
            role_source = _safe_text(role_item.get("source"))
            role_is_hand_conditioned = bool(role_meta.get("hand_conditioned")) or role_source.lower().startswith("hand_conditioned")
            noun_source_fields = _flatten_noun_source_decision(role_meta)
            noun_primary_family = _safe_text(noun_source_fields.get("noun_primary_family")).lower()
            role_summary = f"Review suggested {field_label} '{display_value}'."
            role_reason = _safe_text(role_item.get("reason")) or "Onset-centered local completion produced a role candidate that still needs confirmation."
            if role_is_hand_conditioned and field_name == "noun_object_id" and noun_primary_family == "detector_grounding":
                role_summary = f"Review detector-grounded noun '{display_value}'."
                role_reason = (
                    role_reason
                    + " Empirical source comparison currently prefers hand-conditioned detector grounding over the semantic noun alternative."
                )
            score = _score_query(
                base_priority=0.81,
                surface="object",
                selected_bonus=selected_bonus,
                propagation_gain=0.72,
                human_cost=0.34,
                overwrite_risk=0.08,
            )
            _append(
                {
                    "query_type": f"suggest_{field_name}",
                    "surface": "objects",
                    "field_name": field_name,
                    "summary": role_summary,
                    "reason": role_reason,
                    "target_frame": local_completion.get("anchor_frame")
                    or row.get("functional_contact_onset")
                    or row.get("interaction_start")
                    or row.get("interaction_end"),
                    "safe_apply": False,
                    "suggested_value": role_item.get("value"),
                    "suggested_source": role_source,
                    "suggested_confidence": role_item.get("confidence"),
                    "hand_conditioned": bool(role_is_hand_conditioned),
                    "action_kind": (
                        "suggest"
                        if _safe_text(role_item.get("decision_hint")).lower() == "human_confirm"
                        else "query"
                    ),
                    "authority_level": (
                        "human_confirm"
                        if _safe_text(role_item.get("decision_hint")).lower() == "human_confirm"
                        else "human_only"
                    ),
                    "apply_mode": (
                        "apply_suggestion"
                        if _safe_text(role_item.get("decision_hint")).lower() == "human_confirm"
                        else ""
                    ),
                    "calibrated_reliability": role_item.get("calibrated_reliability"),
                    "acceptance_prob_est": role_item.get("acceptance_prior"),
                    "empirical_support_n": role_item.get("empirical_support"),
                    **noun_source_fields,
                    **score,
                }
            )

        if bbox_errors:
            time_rank = {"Onset": 0, "Start": 1, "End": 2}
            role_rank = {"Noun": 0, "Instrument": 1, "Target": 2}
            ordered_bbox_errors = sorted(
                [dict(err) for err in bbox_errors],
                key=lambda err: (
                    time_rank.get(_safe_text(err.get("time_label")).title(), 3),
                    role_rank.get(_safe_text(err.get("role")).title(), 2),
                    _safe_int(err.get("frame")) if _safe_int(err.get("frame")) is not None else 10**9,
                ),
            )
            for err in ordered_bbox_errors:
                time_label = _safe_text(err.get("time_label")).title() or "Key"
                role_label = _safe_text(err.get("role")).title() or "Object"
                frame = _safe_int(err.get("frame"))
                time_key = _safe_text(err.get("time_key")).lower()
                target_slot = evidence_slot_name(role_label, time_label)
                base_priority = {
                    "onset": 0.92,
                    "start": 0.84,
                    "end": 0.80,
                }.get(time_key, 0.78)
                if role_label.lower() == "instrument":
                    base_priority += 0.02
                score = _score_query(
                    base_priority=base_priority,
                    surface="object",
                    selected_bonus=selected_bonus,
                    propagation_gain=0.94 if time_key == "onset" else 0.82,
                    human_cost=0.56,
                    overwrite_risk=0.26,
                )
                object_name = _safe_text(err.get("object_name"))
                object_text = f" '{object_name}'" if object_name else ""
                _append(
                    {
                        "query_type": "verify_sparse_evidence",
                        "surface": "objects",
                        "field_name": "bbox",
                        "summary": (
                            f"Acquire {role_label.lower()}{object_text} evidence at {time_label.lower()} "
                            f"(frame {frame})."
                            if frame is not None
                            else f"Acquire {role_label.lower()}{object_text} evidence at {time_label.lower()}."
                        ),
                        "reason": (
                            "Sparse evidence is organized around start / onset / end; resolving this slot improves event grounding."
                        ),
                        "target_frame": frame,
                        "target_role": role_label.lower(),
                        "target_time_key": time_key,
                        "target_time_label": time_label,
                        "target_slot": target_slot,
                        "safe_apply": False,
                        "action_kind": "query",
                        "authority_level": "human_only",
                        **score,
                    }
                )

        if consistency_flags:
            first_flag = dict(consistency_flags[0])
            flag_text = _safe_text(first_flag.get("flag")).replace("_", " ") or "consistency issue"
            score = _score_query(
                base_priority=0.83,
                surface="review",
                selected_bonus=selected_bonus,
                propagation_gain=0.79,
                overwrite_risk=0.12,
            )
            _append(
                {
                    "query_type": "resolve_consistency_flag",
                    "surface": "review",
                    "field_name": first_flag.get("flag") or "consistency",
                    "summary": f"Resolve consistency issue: {flag_text}.",
                    "reason": "The event graph contains an inconsistent temporal or semantic state.",
                    "target_frame": first_flag.get("contact_onset_frame")
                    or first_flag.get("start_frame")
                    or first_flag.get("end_frame"),
                    "safe_apply": False,
                    "action_kind": "query",
                    "authority_level": "human_only",
                    **score,
                }
            )

        if sparse_evidence_state:
            for slot_name, slot_state in sparse_evidence_state.items():
                if _safe_text(slot_state.get("status")) != "suggested":
                    continue
                frame = _safe_int(slot_state.get("frame"))
                role_label = _safe_text(slot_state.get("role")).title() or "Object"
                time_label = _safe_text(slot_state.get("time_label")).title() or "Key"
                source = _safe_text(slot_state.get("source")) or "suggested_evidence"
                score = _score_query(
                    base_priority=0.86 if time_label.lower() == "onset" else 0.80,
                    surface="object",
                    selected_bonus=selected_bonus,
                    propagation_gain=0.90,
                    human_cost=0.38,
                    overwrite_risk=0.10,
                )
                _append(
                    {
                        "query_type": "confirm_sparse_evidence",
                        "surface": "objects",
                        "field_name": "bbox",
                        "summary": (
                            f"Review suggested {role_label.lower()} evidence at {time_label.lower()} "
                            f"(frame {frame})."
                            if frame is not None
                            else f"Review suggested {role_label.lower()} evidence at {time_label.lower()}."
                        ),
                        "reason": _safe_text(slot_state.get("note"))
                        or "Suggested sparse evidence should be confirmed before finalization.",
                        "target_frame": frame,
                        "target_role": role_label.lower(),
                        "target_time_key": _safe_text(slot_state.get("time_key")).lower(),
                        "target_time_label": time_label,
                        "target_slot": slot_name,
                        "safe_apply": False,
                        "suggested_source": source,
                        "action_kind": "suggest",
                        "authority_level": "human_confirm",
                        **score,
                    }
                )

    queries.sort(
        key=lambda row: (
            float(row.get("voi_score", 0.0) or 0.0),
            float(row.get("base_priority", 0.0) or 0.0),
            -float(row.get("human_cost_est", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return queries
