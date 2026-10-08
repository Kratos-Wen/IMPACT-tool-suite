from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


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


def _field_state_entry(field_state: Dict[str, Any], field_name: str) -> Dict[str, Any]:
    if field_name == "noun_object_id":
        row = (field_state or {}).get("noun_object_id")
        if row is None:
            row = (field_state or {}).get("target_object_id")
        return dict(row or {})
    return dict((field_state or {}).get(field_name) or {})


def _is_confirmed(field_state: Dict[str, Any], field_name: str) -> bool:
    row = _field_state_entry(field_state, field_name)
    return _safe_text(row.get("status")).lower() == "confirmed"


def _normalize_object_candidates(raw: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in list(raw or []):
        if not isinstance(item, dict):
            continue
        object_id = _safe_int(item.get("object_id"))
        if object_id is None:
            continue
        frames = []
        for frame in list(item.get("frames") or []):
            frame_int = _safe_int(frame)
            if frame_int is not None and frame_int not in frames:
                frames.append(frame_int)
        out.append(
            {
                "object_id": int(object_id),
                "object_name": _safe_text(item.get("object_name")) or f"Object {object_id}",
                "support": max(0, int(item.get("support", 0) or 0)),
                "onset_support": max(0, int(item.get("onset_support", 0) or 0)),
                "support_score": max(0.0, float(item.get("support_score", item.get("support", 0.0)) or 0.0)),
                "onset_support_score": max(
                    0.0,
                    float(item.get("onset_support_score", item.get("onset_support", 0.0)) or 0.0),
                ),
                "candidate_score": max(0.0, float(item.get("candidate_score", 0.0) or 0.0)),
                "candidate_gap": max(0.0, float(item.get("candidate_gap", 0.0) or 0.0)),
                "hand_conditioned": bool(item.get("hand_conditioned")),
                "best_frame": _safe_int(item.get("best_frame")),
                "best_bbox": dict(item.get("best_bbox") or {}),
                "hand_proximity_max": max(0.0, float(item.get("hand_proximity_max", 0.0) or 0.0)),
                "yolo_confidence_max": max(0.0, float(item.get("yolo_confidence_max", 0.0) or 0.0)),
                "frames": frames,
            }
        )
    out.sort(
        key=lambda row: (
            float(row.get("candidate_score", 0.0) or 0.0),
            float(row.get("onset_support_score", 0.0) or 0.0),
            float(row.get("support_score", 0.0) or 0.0),
            -int(row.get("object_id", 0) or 0),
        ),
        reverse=True,
    )
    return out


def _unique_role_candidate(
    candidates: Sequence[Dict[str, Any]], exclude_ids: Sequence[int]
) -> Optional[Dict[str, Any]]:
    excluded = {int(v) for v in list(exclude_ids or []) if _safe_int(v) is not None}
    usable = [dict(item) for item in list(candidates or []) if int(item.get("object_id", -1)) not in excluded]
    if len(usable) == 1:
        return usable[0]
    return None


def _dominant_role_candidate(
    candidates: Sequence[Dict[str, Any]],
    exclude_ids: Sequence[int],
) -> Optional[Dict[str, Any]]:
    excluded = {int(v) for v in list(exclude_ids or []) if _safe_int(v) is not None}
    usable = [
        dict(item)
        for item in list(candidates or [])
        if int(item.get("object_id", -1)) not in excluded
    ]
    if not usable:
        return None
    top = dict(usable[0])
    top_score = _safe_float(top.get("candidate_score"))
    top_gap = _safe_float(top.get("candidate_gap"))
    if len(usable) > 1 and top_gap <= 0.0:
        second_score = _safe_float(usable[1].get("candidate_score"))
        top_gap = max(0.0, top_score - second_score)
    if top_score < 0.52:
        return None
    if len(usable) > 1 and top_gap < 0.08:
        return None
    top["candidate_gap"] = float(top_gap)
    return top


def build_onset_centric_completion(
    row: Dict[str, Any],
    *,
    calibrator: Optional[Any] = None,
) -> Dict[str, Any]:
    if not isinstance(row, dict):
        return {}

    field_state = dict(row.get("field_state") or {})
    field_suggestions = dict(row.get("field_suggestions") or {})
    start = _safe_int(row.get("interaction_start"))
    onset = _safe_int(row.get("functional_contact_onset"))
    end = _safe_int(row.get("interaction_end"))
    event_start = _safe_int(row.get("event_start_frame"))
    event_end = _safe_int(row.get("event_end_frame"))
    verb = _safe_text(row.get("verb"))
    target = _safe_int(row.get("noun_object_id", row.get("target_object_id")))
    handtrack_prior = dict(row.get("handtrack_prior") or {})
    handtrack_onset = _safe_int(handtrack_prior.get("onset_frame"))
    handtrack_confidence = _safe_float(handtrack_prior.get("confidence"))
    noun_only_mode = True
    noun_required = bool(row.get("noun_required"))
    allowed_noun_ids = {
        int(v)
        for v in list(row.get("allowed_noun_ids") or [])
        if _safe_int(v) is not None
    }
    videomae_candidates = list(row.get("videomae_candidates") or [])
    object_candidates = _normalize_object_candidates(row.get("object_candidates"))
    if allowed_noun_ids:
        object_candidates = [
            dict(item)
            for item in object_candidates
            if _safe_int(item.get("object_id")) in allowed_noun_ids
        ]

    noun_suggestion = dict(field_suggestions.get("noun_object_id") or field_suggestions.get("target_object_id") or {})
    noun_suggestion_source = _safe_text(noun_suggestion.get("source")).lower()
    noun_suggestion_conf = _safe_float(noun_suggestion.get("confidence"))
    noun_source_decision = dict(row.get("noun_source_decision") or {})
    preferred_noun_source = _safe_text(noun_source_decision.get("preferred_source")).lower()
    preferred_noun_family = _safe_text(noun_source_decision.get("preferred_family")).lower()
    has_strong_semantic_noun = (
        noun_suggestion_source.startswith("semantic_adapter")
        and _safe_int(noun_suggestion.get("value")) is not None
        and noun_suggestion_conf >= 0.78
    )
    semantic_noun_preferred = (
        preferred_noun_family == "semantic"
        or preferred_noun_source.startswith("semantic_adapter")
    )
    detector_grounding_preferred = (
        preferred_noun_family == "detector_grounding"
        or preferred_noun_source.startswith("hand_conditioned")
    )

    suggestions: List[Dict[str, Any]] = []

    if (
        onset is None
        and handtrack_onset is not None
        and not _is_confirmed(field_state, "functional_contact_onset")
    ):
        suggestions.append(
            {
                "field_name": "functional_contact_onset",
                "value": int(handtrack_onset),
                "source": "handtrack_once_onset_prior",
                "confidence": float(handtrack_confidence),
                "safe_to_apply": bool(handtrack_confidence >= 0.78),
                "reason": "Persistent hand-track motion proposes a hand-conditioned onset prior inside the current event segment.",
                "meta": {
                    "hand_conditioned": True,
                    "handtrack_prior": dict(handtrack_prior),
                },
            }
        )

    if onset is not None:
        if start is None and not _is_confirmed(field_state, "interaction_start"):
            if event_start is not None and event_start <= onset:
                candidate = event_start
                confidence = 0.82
                reason = "Use the local event start as the temporal support preceding the confirmed onset."
            else:
                candidate = onset
                confidence = 0.56
                reason = "Fall back to onset as a conservative local start when earlier support is missing."
            suggestions.append(
                {
                    "field_name": "interaction_start",
                    "value": int(candidate),
                    "source": "onset_local_completion",
                    "confidence": confidence,
                    "safe_to_apply": True,
                    "reason": reason,
                }
            )

        if end is None and not _is_confirmed(field_state, "interaction_end"):
            if event_end is not None and event_end >= onset:
                candidate = event_end
                confidence = 0.82
                reason = "Use the local event end as the temporal support following the confirmed onset."
            else:
                candidate = onset
                confidence = 0.56
                reason = "Fall back to onset as a conservative local end when later support is missing."
            suggestions.append(
                {
                    "field_name": "interaction_end",
                    "value": int(candidate),
                    "source": "onset_local_completion",
                    "confidence": confidence,
                    "safe_to_apply": True,
                    "reason": reason,
                }
            )

    if not verb and not _is_confirmed(field_state, "verb"):
        top = dict(videomae_candidates[0] or {}) if videomae_candidates else {}
        label = _safe_text(top.get("label"))
        if label:
            suggestions.append(
                {
                    "field_name": "verb",
                    "value": label,
                    "source": "videomae_top1",
                    "confidence": float(top.get("score") or 0.0),
                    "safe_to_apply": True,
                    "reason": "VideoMAE already ranked a verb around this onset-centered event window.",
                }
            )

    if onset is not None and noun_only_mode and target is None and noun_required and not _is_confirmed(field_state, "noun_object_id"):
        candidate = _unique_role_candidate(object_candidates, [])
        if candidate:
            suggestions.append(
                {
                    "field_name": "noun_object_id",
                    "value": int(candidate["object_id"]),
                    "display_value": candidate["object_name"],
                    "support": int(candidate.get("support", 0) or 0),
                    "onset_support": int(candidate.get("onset_support", 0) or 0),
                    "source": "onset_noun_completion",
                    "confidence": 0.58,
                    "safe_to_apply": False,
                    "reason": "Only one onset-centered noun candidate remains after ontology filtering.",
                    "meta": {
                        "hand_conditioned": bool(candidate.get("hand_conditioned", True)),
                        "source_decision": dict(noun_source_decision or {}),
                    },
                }
            )
        else:
            candidate = _dominant_role_candidate(object_candidates, [])
            allow_hand_conditioned_prior = (
                detector_grounding_preferred
                or (not semantic_noun_preferred and not has_strong_semantic_noun)
            )
            if candidate and allow_hand_conditioned_prior:
                candidate_score = _safe_float(candidate.get("candidate_score"))
                candidate_gap = _safe_float(candidate.get("candidate_gap"))
                confidence = min(0.92, 0.48 + 0.34 * candidate_score + 0.18 * candidate_gap)
                suggestions.append(
                    {
                        "field_name": "noun_object_id",
                        "value": int(candidate["object_id"]),
                        "display_value": candidate["object_name"],
                        "support": int(candidate.get("support", 0) or 0),
                        "onset_support": int(candidate.get("onset_support", 0) or 0),
                        "source": "hand_conditioned_noun_prior",
                        "confidence": float(confidence),
                        "safe_to_apply": False,
                        "reason": "Hand-conditioned object support ranks this noun candidate highest by fusing box confidence, onset-local proximity, and hand-motion context.",
                        "decision_hint": "human_confirm" if confidence >= 0.74 else "human_only",
                        "meta": {
                            "hand_conditioned": bool(candidate.get("hand_conditioned", True)),
                            "source_decision": dict(noun_source_decision or {}),
                            "candidate_score": float(candidate_score),
                            "candidate_gap": float(candidate_gap),
                            "best_frame": candidate.get("best_frame"),
                            "best_bbox": dict(candidate.get("best_bbox") or {}),
                            "hand_proximity_max": float(candidate.get("hand_proximity_max", 0.0) or 0.0),
                            "yolo_confidence_max": float(candidate.get("yolo_confidence_max", 0.0) or 0.0),
                        },
                    }
                )

    if not suggestions:
        return {}

    safe_fields = [dict(item) for item in suggestions if bool(item.get("safe_to_apply"))]
    role_fields = [
        dict(item)
        for item in suggestions
        if str(item.get("field_name") or "").strip() == "noun_object_id"
    ]
    handtrack_onset_fields = [
        dict(item)
        for item in suggestions
        if str(item.get("field_name") or "").strip() == "functional_contact_onset"
        and _is_handtrack_source(item.get("source"))
    ]
    anchor_frame = onset
    if anchor_frame is None:
        anchor_frame = handtrack_onset if handtrack_onset is not None else (
            start if start is not None else end
        )

    if safe_fields:
        safe_handtrack_onset = next(
            (
                dict(item)
                for item in safe_fields
                if str(item.get("field_name") or "").strip() == "functional_contact_onset"
                and _is_handtrack_source(item.get("source"))
            ),
            {},
        )
        safe_names = [FIELD_LABELS.get(str(item.get("field_name")), str(item.get("field_name"))) for item in safe_fields]
        if safe_handtrack_onset:
            onset_value = _safe_int(safe_handtrack_onset.get("value"))
            remaining_safe_names = [
                name
                for item, name in zip(safe_fields, safe_names)
                if not (
                    str(item.get("field_name") or "").strip() == "functional_contact_onset"
                    and _is_handtrack_source(item.get("source"))
                )
            ]
            if len(safe_names) == 1 and onset_value is not None:
                summary = f"Apply hand-conditioned onset prior at frame {onset_value}."
            elif onset_value is not None and remaining_safe_names:
                summary = (
                    "Apply hand-conditioned onset prior and complete local event: add "
                    + (
                        remaining_safe_names[0]
                        if len(remaining_safe_names) == 1
                        else ", ".join(remaining_safe_names[:-1]) + f", and {remaining_safe_names[-1]}"
                    )
                    + "."
                )
            else:
                summary = "Apply hand-conditioned onset prior and complete local event."
            reason = (
                "Persistent hand-track motion supplies a hand-conditioned onset prior that can be locally decoded without touching confirmed content."
            )
        elif len(safe_names) == 1:
            summary = f"Complete local event around onset: add {safe_names[0]}."
            reason = (
                "Onset-conditioned local completion can safely fill unresolved temporal or semantic fields without touching confirmed content."
            )
        else:
            summary = "Complete local event around onset: add " + ", ".join(safe_names[:-1]) + f", and {safe_names[-1]}."
            reason = (
                "Onset-conditioned local completion can safely fill unresolved temporal or semantic fields without touching confirmed content."
            )
    elif role_fields:
        role_item = role_fields[0]
        role_label = FIELD_LABELS.get(str(role_item.get("field_name")), "role")
        display_value = _safe_text(role_item.get("display_value")) or f"Object {role_item.get('value')}"
        summary = f"Review suggested {role_label} '{display_value}' around the onset."
        reason = _safe_text(role_item.get("reason")) or "Onset-centered evidence yields a role candidate that still requires human confirmation."
    else:
        first = dict(suggestions[0])
        field_label = FIELD_LABELS.get(str(first.get("field_name")), str(first.get("field_name")))
        first_value = _safe_int(first.get("value"))
        if str(first.get("field_name") or "").strip() == "functional_contact_onset" and _is_handtrack_source(first.get("source")):
            summary = (
                f"Review hand-conditioned onset prior near frame {first_value}."
                if first_value is not None
                else "Review hand-conditioned onset prior."
            )
            reason = _safe_text(first.get("reason")) or "Persistent hand-track motion proposed an onset candidate that still needs confirmation."
        else:
            summary = f"Review onset-centered completion for {field_label}."
            reason = _safe_text(first.get("reason")) or "Local onset-conditioned completion is available for this unresolved field."

    completion = {
        "anchor_frame": anchor_frame,
        "safe_fields": safe_fields,
        "suggested_fields": suggestions,
        "summary": summary,
        "reason": reason,
        "hand_conditioned_onset_prior": bool(handtrack_onset_fields),
        "handtrack_prior": dict(handtrack_prior) if handtrack_onset_fields else {},
    }
    if calibrator is not None:
        try:
            completion = calibrator.calibrate_completion(row, completion)
        except Exception:
            pass
    return completion
