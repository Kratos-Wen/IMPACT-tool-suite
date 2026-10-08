from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple


RESOLUTION_EVENTS = {"hoi_query_apply", "hoi_query_reject"}
LATENCY_EVENTS = {"hoi_query_focus", "hoi_query_apply", "hoi_query_reject"}

TEMPORAL_FIELDS = {
    "interaction_start",
    "functional_contact_onset",
    "interaction_end",
}
OBJECT_FIELDS = {"noun_object_id", "target_object_id", "bbox"}

DEFAULT_ACCEPT_PRIOR_BY_SOURCE = {
    "videomae_top1": 0.66,
    "handtrack_once": 0.68,
    "handtrack_once_onset": 0.74,
    "handtrack_once_onset_prior": 0.76,
    "semantic_adapter_onset": 0.74,
    "semantic_adapter_verb": 0.72,
    "semantic_adapter_noun": 0.72,
    "semantic_adapter_no_noun": 0.70,
    "midpoint_inference": 0.58,
    "onset_local_completion": 0.74,
    "onset_role_completion": 0.56,
    "onset_noun_completion": 0.58,
    "hand_conditioned_noun_prior": 0.60,
    "hand_conditioned_object_candidate": 0.58,
    "detector_grounding": 0.58,
    "query_controller": 0.60,
    "unknown": 0.56,
}

DEFAULT_ACCEPT_PRIOR_BY_ACTION = {
    "query": 0.52,
    "suggest": 0.64,
    "propagate": 0.76,
}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _source_family(source: Any) -> str:
    text = _safe_text(source).lower()
    if not text:
        return "unknown"
    if text.startswith("semantic_adapter"):
        return "semantic"
    if text.startswith("hand_conditioned") or text == "detector_grounding":
        return "detector_grounding"
    if text.startswith("handtrack_once"):
        return "handtrack"
    if text.startswith("videomae"):
        return "videomae"
    if text.startswith("onset_"):
        return "onset_completion"
    return text


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in list(values or [])]
    if not vals:
        return 0.0
    return float(sum(vals) / max(1, len(vals)))


def _quantile(values: Sequence[float], q: float) -> float:
    vals = sorted(float(v) for v in list(values or []))
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    q = _clamp(q)
    pos = q * (len(vals) - 1)
    lo = int(pos)
    hi = min(len(vals) - 1, lo + 1)
    frac = pos - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def _read_csv_rows(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    except Exception:
        return []


@dataclass
class _RateStat:
    success: int = 0
    total: int = 0


@dataclass
class _CostStat:
    total_ms: float = 0.0
    count: int = 0


class HOIEmpiricalCalibrator:
    """
    Lightweight empirical calibration layer built from operation logs.

    It does not train a heavy model. Instead, it learns:
    - human acceptance probability
    - human interaction cost
    from resolved query traces, then uses those estimates to calibrate:
    - query priority
    - query authority
    - local completion reliability
    """

    def __init__(self, rows: Optional[Sequence[Dict[str, Any]]] = None):
        self.rows = [dict(row) for row in list(rows or []) if isinstance(row, dict)]
        self._rate_stats: DefaultDict[Tuple[Any, ...], _RateStat] = defaultdict(_RateStat)
        self._cost_stats: DefaultDict[Tuple[Any, ...], _CostStat] = defaultdict(_CostStat)
        self._latencies_ms: List[float] = []
        self._accept_default = 0.60
        self._cost_default_ms = 1800.0
        self._cost_norm_ref_ms = 3200.0
        self._build()

    @classmethod
    def from_sources(
        cls,
        *,
        csv_paths: Optional[Iterable[str]] = None,
        live_rows: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> "HOIEmpiricalCalibrator":
        rows: List[Dict[str, Any]] = []
        seen_paths = set()
        for path in list(csv_paths or []):
            fp = os.path.abspath(str(path))
            if not os.path.isfile(fp) or fp in seen_paths:
                continue
            seen_paths.add(fp)
            rows.extend(_read_csv_rows(fp))
        for row in list(live_rows or []):
            if isinstance(row, dict):
                rows.append(dict(row))
        return cls(rows)

    def _acceptance_key_specs(self, row: Dict[str, Any]) -> List[Tuple[Tuple[Any, ...], float]]:
        query_type = _safe_text(row.get("query_type")) or "unknown"
        field_name = _safe_text(row.get("field")) or _safe_text(row.get("field_name")) or "unknown"
        action_kind = _safe_text(row.get("action_kind")) or "unknown"
        authority = _safe_text(row.get("authority_level")) or "unknown"
        interaction_form = _safe_text(row.get("interaction_form")) or "unknown"
        source = _safe_text(row.get("suggested_source")) or _safe_text(row.get("source")) or ""
        source_family = _source_family(source)
        annotator = (
            _safe_text(row.get("annotator_id"))
            or _safe_text(row.get("validator"))
            or "unknown"
        )
        return [
            (("annotator_query_field_authority", annotator, query_type, field_name, authority), 1.06),
            (("annotator_field_action", annotator, field_name, action_kind), 1.00),
            (("annotator_query_type", annotator, query_type), 0.92),
            (("query_type_field_authority", query_type, field_name, authority), 1.00),
            (("query_type_field", query_type, field_name), 0.96),
            (("source_field", source or "unknown", field_name), 0.92),
            (("source_family_field", source_family, field_name), 0.88),
            (("source_family_action", source_family, action_kind), 0.76),
            (("source_family", source_family), 0.70),
            (("field_action", field_name, action_kind), 0.88),
            (("query_type", query_type), 0.80),
            (("field_name", field_name), 0.72),
            (("action_kind", action_kind), 0.62),
            (("authority_level", authority), 0.56),
            (("interaction_form", interaction_form), 0.50),
        ]

    def _cost_key_specs(self, row: Dict[str, Any]) -> List[Tuple[Tuple[Any, ...], float]]:
        query_type = _safe_text(row.get("query_type")) or "unknown"
        field_name = _safe_text(row.get("field")) or _safe_text(row.get("field_name")) or "unknown"
        action_kind = _safe_text(row.get("action_kind")) or "unknown"
        authority = _safe_text(row.get("authority_level")) or "unknown"
        interaction_form = _safe_text(row.get("interaction_form")) or "unknown"
        source = _safe_text(row.get("suggested_source")) or _safe_text(row.get("source")) or ""
        source_family = _source_family(source)
        annotator = (
            _safe_text(row.get("annotator_id"))
            or _safe_text(row.get("validator"))
            or "unknown"
        )
        return [
            (("annotator_query_field", annotator, query_type, field_name), 1.04),
            (("annotator_field_action", annotator, field_name, action_kind), 0.96),
            (("annotator_interaction_form", annotator, interaction_form), 0.86),
            (("query_type_field", query_type, field_name), 1.00),
            (("source_family_field", source_family, field_name), 0.84),
            (("source_family_action", source_family, action_kind), 0.74),
            (("field_action", field_name, action_kind), 0.86),
            (("interaction_form", interaction_form), 0.80),
            (("query_type", query_type), 0.74),
            (("field_name", field_name), 0.68),
            (("authority_level", authority), 0.56),
            (("action_kind", action_kind), 0.50),
        ]

    def _build(self) -> None:
        accept_success = 0
        accept_total = 0
        latency_values: List[float] = []

        for row in self.rows:
            event = _safe_text(row.get("event"))
            if event in RESOLUTION_EVENTS:
                resolve_kind = _safe_text(row.get("resolve_kind"))
                success = event == "hoi_query_apply" and resolve_kind not in {"", "unknown"}
                if event == "hoi_query_reject":
                    success = False
                accept_total += 1
                if success:
                    accept_success += 1
                for key, _weight in self._acceptance_key_specs(row):
                    stat = self._rate_stats[key]
                    stat.total += 1
                    stat.success += int(bool(success))

            if event in LATENCY_EVENTS:
                latency_ms = _safe_float(row.get("query_latency_ms"))
                if latency_ms <= 0:
                    continue
                latency_values.append(latency_ms)
                for key, _weight in self._cost_key_specs(row):
                    stat = self._cost_stats[key]
                    stat.total_ms += float(latency_ms)
                    stat.count += 1

        if accept_total > 0:
            self._accept_default = float(accept_success) / float(max(1, accept_total))
        if latency_values:
            self._latencies_ms = list(latency_values)
            self._cost_default_ms = _mean(latency_values)
            self._cost_norm_ref_ms = max(
                self._cost_default_ms,
                _quantile(latency_values, 0.85),
                1200.0,
            )

    def _posterior_rate(
        self,
        success: int,
        total: int,
        *,
        prior: float,
        strength: float = 4.0,
    ) -> float:
        return _clamp((float(success) + float(strength) * float(prior)) / (float(total) + float(strength)))

    def _lookup_acceptance(
        self,
        specs: Sequence[Tuple[Tuple[Any, ...], float]],
        *,
        prior: float,
    ) -> Tuple[float, int]:
        weighted_sum = 0.0
        total_weight = 0.0
        support = 0
        for key, base_weight in list(specs or []):
            stat = self._rate_stats.get(tuple(key))
            if not stat or stat.total <= 0:
                continue
            support += int(stat.total)
            posterior = self._posterior_rate(
                stat.success,
                stat.total,
                prior=prior,
                strength=4.0,
            )
            weight = float(base_weight) * min(1.0, float(stat.total) / 8.0)
            weighted_sum += posterior * weight
            total_weight += weight
        if total_weight <= 0:
            return float(prior), 0
        return float(weighted_sum / total_weight), int(support)

    def _lookup_cost_ms(
        self,
        specs: Sequence[Tuple[Tuple[Any, ...], float]],
    ) -> Tuple[float, int]:
        weighted_sum = 0.0
        total_weight = 0.0
        support = 0
        for key, base_weight in list(specs or []):
            stat = self._cost_stats.get(tuple(key))
            if not stat or stat.count <= 0:
                continue
            support += int(stat.count)
            mean_ms = float(stat.total_ms) / float(max(1, stat.count))
            # Empirical Bayes shrinkage toward the global mean to avoid overfitting tiny buckets.
            shrunk = (
                float(mean_ms) * min(1.0, float(stat.count) / 6.0)
                + float(self._cost_default_ms) * max(0.0, 1.0 - float(stat.count) / 6.0)
            )
            weight = float(base_weight) * min(1.0, float(stat.count) / 8.0)
            weighted_sum += shrunk * weight
            total_weight += weight
        if total_weight <= 0:
            return float(self._cost_default_ms), 0
        return float(weighted_sum / total_weight), int(support)

    def estimate_query_acceptance(self, query: Dict[str, Any]) -> Dict[str, Any]:
        action_kind = _safe_text(query.get("action_kind")).lower() or "query"
        source = _safe_text(query.get("suggested_source")).lower() or "unknown"
        prior = DEFAULT_ACCEPT_PRIOR_BY_SOURCE.get(
            source,
            DEFAULT_ACCEPT_PRIOR_BY_ACTION.get(action_kind, self._accept_default),
        )
        value, support = self._lookup_acceptance(
            self._acceptance_key_specs(query),
            prior=prior,
        )
        return {
            "value": float(value),
            "support": int(support),
        }

    def estimate_query_cost(self, query: Dict[str, Any]) -> Dict[str, Any]:
        mean_ms, support = self._lookup_cost_ms(self._cost_key_specs(query))
        norm = _clamp(float(mean_ms) / float(max(1.0, self._cost_norm_ref_ms)))
        return {
            "value": float(norm),
            "latency_ms": float(mean_ms),
            "support": int(support),
        }

    def _blend_empirical_with_runtime_confidence(
        self,
        empirical_value: float,
        empirical_support: int,
        runtime_confidence: float,
        *,
        support_ref: float = 48.0,
    ) -> float:
        # The lookup support is an aggregated bucket support count rather than a raw row count,
        # so the backoff reference needs to be wider than a simple per-row threshold.
        alpha = _clamp(float(empirical_support) / float(max(1.0, support_ref)))
        return _clamp(alpha * float(empirical_value) + (1.0 - alpha) * float(runtime_confidence))

    def estimate_source_acceptance(
        self,
        *,
        field_name: str,
        source: str,
        action_kind: str = "suggest",
        authority_level: str = "human_confirm",
        interaction_form: str = "accept_suggestion",
        query_type: str = "",
    ) -> Dict[str, Any]:
        safe_field = _safe_text(field_name) or "unknown"
        safe_source = _safe_text(source).lower() or "unknown"
        safe_action = _safe_text(action_kind).lower() or "suggest"
        safe_query_type = _safe_text(query_type) or f"{safe_action}_{safe_field}"
        meta = self.estimate_query_acceptance(
            {
                "query_type": safe_query_type,
                "field_name": safe_field,
                "field": safe_field,
                "action_kind": safe_action,
                "authority_level": _safe_text(authority_level) or "human_confirm",
                "interaction_form": _safe_text(interaction_form) or "accept_suggestion",
                "suggested_source": safe_source,
                "source": safe_source,
            }
        )
        return {
            "source": safe_source,
            "source_family": _source_family(safe_source),
            "value": float(meta.get("value", DEFAULT_ACCEPT_PRIOR_BY_SOURCE.get(safe_source, self._accept_default))),
            "support": int(meta.get("support", 0)),
        }

    def compare_field_sources(
        self,
        *,
        field_name: str,
        source_a: str,
        source_b: str,
        runtime_confidence_a: float = 0.0,
        runtime_confidence_b: float = 0.0,
        action_kind: str = "suggest",
        authority_level: str = "human_confirm",
        interaction_form: str = "accept_suggestion",
        query_type: str = "",
    ) -> Dict[str, Any]:
        meta_a = self.estimate_source_acceptance(
            field_name=field_name,
            source=source_a,
            action_kind=action_kind,
            authority_level=authority_level,
            interaction_form=interaction_form,
            query_type=query_type,
        )
        meta_b = self.estimate_source_acceptance(
            field_name=field_name,
            source=source_b,
            action_kind=action_kind,
            authority_level=authority_level,
            interaction_form=interaction_form,
            query_type=query_type,
        )
        score_a = self._blend_empirical_with_runtime_confidence(
            float(meta_a.get("value", 0.0) or 0.0),
            int(meta_a.get("support", 0) or 0),
            _clamp(runtime_confidence_a),
        )
        score_b = self._blend_empirical_with_runtime_confidence(
            float(meta_b.get("value", 0.0) or 0.0),
            int(meta_b.get("support", 0) or 0),
            _clamp(runtime_confidence_b),
        )
        preferred = "a"
        if score_b > score_a + 1e-6:
            preferred = "b"
        return {
            "field_name": _safe_text(field_name) or "unknown",
            "source_a": str(meta_a.get("source") or "unknown"),
            "source_a_family": str(meta_a.get("source_family") or "unknown"),
            "source_a_acceptance": float(meta_a.get("value", 0.0) or 0.0),
            "source_a_support": int(meta_a.get("support", 0) or 0),
            "source_a_runtime_confidence": _clamp(runtime_confidence_a),
            "source_a_score": float(score_a),
            "source_b": str(meta_b.get("source") or "unknown"),
            "source_b_family": str(meta_b.get("source_family") or "unknown"),
            "source_b_acceptance": float(meta_b.get("value", 0.0) or 0.0),
            "source_b_support": int(meta_b.get("support", 0) or 0),
            "source_b_runtime_confidence": _clamp(runtime_confidence_b),
            "source_b_score": float(score_b),
            "preferred_side": preferred,
            "preferred_source": str(meta_a.get("source") if preferred == "a" else meta_b.get("source")),
            "preferred_family": str(meta_a.get("source_family") if preferred == "a" else meta_b.get("source_family")),
            "score_margin": float(abs(score_a - score_b)),
            "decision_basis": "empirical_acceptance_with_runtime_confidence_backoff",
        }

    def estimate_completion_reliability(
        self,
        row: Dict[str, Any],
        suggestion: Dict[str, Any],
    ) -> Dict[str, Any]:
        field_name = _safe_text(suggestion.get("field_name"))
        source = _safe_text(suggestion.get("source")).lower() or "unknown"
        confidence = _safe_float(suggestion.get("confidence"))
        if confidence <= 0:
            confidence = 0.50

        accept_meta = self.estimate_query_acceptance(
            {
                "query_type": f"suggest_{field_name or 'field'}",
                "field_name": field_name,
                "field": field_name,
                "action_kind": "suggest",
                "authority_level": "human_confirm",
                "interaction_form": "accept_suggestion",
                "suggested_source": source,
            }
        )
        base_accept = float(accept_meta.get("value", DEFAULT_ACCEPT_PRIOR_BY_SOURCE.get(source, 0.56)))
        support = int(accept_meta.get("support", 0))

        field_state = dict(row.get("field_state") or {})
        onset_state = dict(field_state.get("functional_contact_onset") or {})
        onset_confirmed = _safe_text(onset_state.get("status")).lower() == "confirmed"
        onset_present = _safe_int(row.get("functional_contact_onset")) is not None

        sparse_summary = dict(row.get("sparse_evidence_summary") or {})
        sparse_expected = max(0, int(sparse_summary.get("expected", 0) or 0))
        sparse_confirmed = max(0, int(sparse_summary.get("confirmed", 0) or 0))
        sparse_ratio = (
            float(sparse_confirmed) / float(sparse_expected)
            if sparse_expected > 0
            else (1.0 if onset_present else 0.0)
        )

        support_score = _clamp(_safe_float(suggestion.get("support")) / 3.0)
        onset_support_score = _clamp(_safe_float(suggestion.get("onset_support")) / 2.0)
        object_support = _clamp(0.55 * onset_support_score + 0.45 * support_score)

        if field_name in TEMPORAL_FIELDS:
            reliability = (
                0.46 * base_accept
                + 0.22 * confidence
                + 0.18 * (1.0 if onset_present else 0.0)
                + 0.14 * sparse_ratio
            )
        elif field_name == "verb":
            reliability = (
                0.40 * base_accept
                + 0.36 * confidence
                + 0.12 * (1.0 if onset_present else 0.0)
                + 0.12 * sparse_ratio
            )
        elif field_name in OBJECT_FIELDS:
            reliability = (
                0.34 * base_accept
                + 0.10 * confidence
                + 0.20 * (1.0 if onset_confirmed else (0.6 if onset_present else 0.0))
                + 0.24 * object_support
                + 0.12 * sparse_ratio
            )
        else:
            reliability = 0.50 * base_accept + 0.25 * confidence + 0.25 * sparse_ratio

        reliability = _clamp(reliability)
        originally_safe = bool(suggestion.get("safe_to_apply"))
        if originally_safe:
            if reliability >= 0.80:
                decision_hint = "safe_local"
            elif reliability >= 0.48:
                decision_hint = "human_confirm"
            else:
                decision_hint = "manual_only"
        else:
            decision_hint = "human_confirm" if reliability >= 0.52 else "manual_only"

        return {
            "reliability": float(reliability),
            "acceptance_prior": float(base_accept),
            "support": int(support),
            "decision_hint": decision_hint,
        }

    def calibrate_completion(
        self,
        row: Dict[str, Any],
        completion: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(completion, dict) or not completion:
            return {}
        out = dict(completion)
        calibrated: List[Dict[str, Any]] = []
        safe_fields: List[Dict[str, Any]] = []
        for item in list(completion.get("suggested_fields") or []):
            if not isinstance(item, dict):
                continue
            row_item = dict(item)
            meta = self.estimate_completion_reliability(row, row_item)
            row_item["calibrated_reliability"] = float(meta.get("reliability", 0.0))
            row_item["acceptance_prior"] = float(meta.get("acceptance_prior", 0.0))
            row_item["empirical_support"] = int(meta.get("support", 0))
            row_item["decision_hint"] = str(meta.get("decision_hint") or "manual_only")
            row_item["safe_to_apply"] = bool(row_item.get("safe_to_apply")) and row_item["decision_hint"] == "safe_local"
            calibrated.append(row_item)
            if row_item["safe_to_apply"]:
                safe_fields.append(dict(row_item))
        out["suggested_fields"] = calibrated
        out["safe_fields"] = safe_fields
        out["calibration_source"] = "empirical_bayes"
        if calibrated:
            out["max_reliability"] = max(float(item.get("calibrated_reliability", 0.0) or 0.0) for item in calibrated)
        return out

    def calibrate_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(query, dict) or not query:
            return dict(query or {})
        out = dict(query)
        heuristic_cost = _safe_float(out.get("human_cost_est"))
        heuristic_risk = _safe_float(out.get("overwrite_risk"))
        heuristic_prop = _safe_float(out.get("propagation_gain"))
        base_priority = _safe_float(out.get("base_priority"))
        selected_bonus = _safe_float(out.get("selected_bonus"))

        cost_meta = self.estimate_query_cost(out)
        accept_meta = self.estimate_query_acceptance(out)

        empirical_cost = float(cost_meta.get("value", heuristic_cost))
        acceptance_prob = float(accept_meta.get("value", self._accept_default))
        empirical_support = int(accept_meta.get("support", 0))

        blended_cost = _clamp(0.55 * heuristic_cost + 0.45 * empirical_cost)
        blended_risk = _clamp(0.62 * heuristic_risk + 0.38 * (1.0 - acceptance_prob))

        action_kind = _safe_text(out.get("action_kind")).lower() or "query"
        if action_kind == "propagate":
            blended_prop = _clamp(0.58 * heuristic_prop + 0.42 * acceptance_prob)
        elif action_kind == "suggest":
            blended_prop = _clamp(0.68 * heuristic_prop + 0.32 * acceptance_prob)
        else:
            blended_prop = _clamp(0.80 * heuristic_prop + 0.20 * acceptance_prob)

        out["human_cost_est"] = float(blended_cost)
        out["overwrite_risk"] = float(blended_risk)
        out["propagation_gain"] = float(blended_prop)
        out["empirical_cost_est"] = float(empirical_cost)
        out["empirical_cost_ms"] = float(cost_meta.get("latency_ms", 0.0))
        out["acceptance_prob_est"] = float(acceptance_prob)
        out["empirical_support_n"] = int(empirical_support)
        out["calibration_source"] = "empirical_bayes"

        # Conservative authority downgrade when the data consistently suggests poor acceptance.
        if (
            action_kind == "suggest"
            and _safe_text(out.get("authority_level")).lower() == "human_confirm"
            and empirical_support >= 6
            and acceptance_prob < 0.35
        ):
            out["action_kind"] = "query"
            out["authority_level"] = "human_only"
            out["interaction_form"] = "manual_edit"
            out["apply_mode"] = ""
            out["safe_apply"] = False
            out["calibration_note"] = "Empirical acceptance is low; require direct manual review."

        out["voi_score"] = float(
            base_priority
            + 0.24 * float(out.get("propagation_gain", 0.0) or 0.0)
            - 0.14 * float(out.get("human_cost_est", 0.0) or 0.0)
            - 0.22 * float(out.get("overwrite_risk", 0.0) or 0.0)
            + selected_bonus
            + 0.08 * (acceptance_prob - 0.5)
        )
        return out
