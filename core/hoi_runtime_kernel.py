from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from core.hoi_query_controller import build_query_candidates
from core.semantic_adapter import predict_with_adapter


@dataclass
class SemanticRuntimeRequest:
    feature: Sequence[float]
    package: Any
    allowed_noun_ids: Sequence[int]
    noun_required: bool
    allow_no_noun: bool
    external_verb_scores: Optional[Dict[str, float]] = None
    noun_support_scores: Optional[Dict[int, float]] = None
    noun_onset_support_scores: Optional[Dict[int, float]] = None
    allowed_nouns_by_verb: Optional[Dict[str, Sequence[int]]] = None
    allow_no_noun_by_verb: Optional[Dict[str, bool]] = None
    clamp_onset_ratio: Optional[float] = None
    clamp_verb_label: str = ""
    clamp_noun_exists: Optional[bool] = None
    clamp_noun_object_id: Optional[int] = None
    clamp_unknown_noun: bool = False
    anchor_onset_ratio: Optional[float] = None
    anchor_onset_half_width: Optional[float] = None
    anchor_onset_weight: float = 0.0
    exclude_onset_ratio: Optional[float] = None
    exclude_onset_half_width: Optional[float] = None
    exclude_onset_weight: float = 0.0
    refinement_passes: int = 1
    refine_feature_fn: Optional[Callable[..., Sequence[float]]] = None


@dataclass
class SemanticRuntimeResult:
    prediction: Dict[str, Any] = field(default_factory=dict)
    feature: List[float] = field(default_factory=list)
    pass_trace: List[Dict[str, Any]] = field(default_factory=list)


def _prediction_signature(prediction: Dict[str, Any]) -> tuple:
    best = dict((prediction.get("structured") or {}).get("best") or {})
    return (
        round(float(best.get("onset_ratio", 0.5) or 0.5), 4),
        str(best.get("verb_label") or ""),
        bool(best.get("noun_exists")),
        best.get("noun_object_id"),
        bool(best.get("noun_is_unknown")),
    )


def _predict_once(
    feature: Sequence[float],
    request: SemanticRuntimeRequest,
) -> Dict[str, Any]:
    return dict(
        predict_with_adapter(
            feature=[float(v) for v in list(feature or [])],
            package=request.package,
            allowed_noun_ids=list(request.allowed_noun_ids or []),
            noun_required=bool(request.noun_required),
            allow_no_noun=bool(request.allow_no_noun),
            external_verb_scores=dict(request.external_verb_scores or {}),
            noun_support_scores=dict(request.noun_support_scores or {}),
            noun_onset_support_scores=dict(request.noun_onset_support_scores or {}),
            allowed_nouns_by_verb={
                str(key): list(value or [])
                for key, value in dict(request.allowed_nouns_by_verb or {}).items()
            },
            allow_no_noun_by_verb={
                str(key): bool(value)
                for key, value in dict(request.allow_no_noun_by_verb or {}).items()
            },
            clamp_onset_ratio=request.clamp_onset_ratio,
            clamp_verb_label=str(request.clamp_verb_label or ""),
            clamp_noun_exists=request.clamp_noun_exists,
            clamp_noun_object_id=request.clamp_noun_object_id,
            clamp_unknown_noun=bool(request.clamp_unknown_noun),
            anchor_onset_ratio=request.anchor_onset_ratio,
            anchor_onset_half_width=request.anchor_onset_half_width,
            anchor_onset_weight=float(request.anchor_onset_weight or 0.0),
            exclude_onset_ratio=request.exclude_onset_ratio,
            exclude_onset_half_width=request.exclude_onset_half_width,
            exclude_onset_weight=float(request.exclude_onset_weight or 0.0),
        )
        or {}
    )


def run_event_local_semantic_decode(
    request: SemanticRuntimeRequest,
) -> SemanticRuntimeResult:
    current_feature = [float(v) for v in list(request.feature or [])]
    prediction = _predict_once(current_feature, request)
    if not prediction:
        return SemanticRuntimeResult(prediction={}, feature=current_feature, pass_trace=[])

    pass_trace: List[Dict[str, Any]] = []
    refinement_passes = max(1, int(request.refinement_passes or 1))
    if refinement_passes <= 1 or not callable(request.refine_feature_fn):
        return SemanticRuntimeResult(
            prediction=prediction,
            feature=current_feature,
            pass_trace=pass_trace,
        )

    for pass_index in range(1, refinement_passes):
        coarse_best = dict((prediction.get("structured") or {}).get("best") or {})
        refined_feature = list(
            request.refine_feature_fn(
                current_feature,
                refined_onset_ratio=(
                    request.clamp_onset_ratio
                    if request.clamp_onset_ratio is not None
                    else coarse_best.get("onset_ratio")
                ),
                refined_verb=str(request.clamp_verb_label or coarse_best.get("verb_label") or ""),
                refined_noun_exists=(
                    request.clamp_noun_exists
                    if request.clamp_noun_exists is not None
                    else coarse_best.get("noun_exists")
                ),
                refined_noun_object_id=(
                    request.clamp_noun_object_id
                    if request.clamp_noun_object_id is not None
                    else coarse_best.get("noun_object_id")
                ),
            )
            or []
        )
        if refined_feature == current_feature:
            break

        refined_prediction = _predict_once(refined_feature, request)
        if not refined_prediction:
            break

        coarse_joint = float(coarse_best.get("joint_prob") or 0.0)
        refined_best = dict((refined_prediction.get("structured") or {}).get("best") or {})
        refined_joint = float(refined_best.get("joint_prob") or 0.0)
        use_refined = refined_joint >= (coarse_joint - 1e-6)
        pass_trace.append(
            {
                "pass_index": int(pass_index),
                "coarse_joint": float(coarse_joint),
                "refined_joint": float(refined_joint),
                "used_refined": bool(use_refined),
            }
        )
        if not use_refined:
            break

        previous_signature = _prediction_signature(prediction)
        prediction = refined_prediction
        current_feature = [float(v) for v in list(refined_feature or [])]
        if _prediction_signature(prediction) == previous_signature:
            break

    return SemanticRuntimeResult(
        prediction=prediction,
        feature=current_feature,
        pass_trace=pass_trace,
    )


def build_runtime_query_candidates(
    hand_rows: Sequence[Dict[str, Any]],
    *,
    selected_event_id: Optional[int] = None,
    selected_hand: Optional[str] = None,
    calibrator: Optional[Any] = None,
    authority_policy: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    return list(
        build_query_candidates(
            hand_rows,
            selected_event_id=selected_event_id,
            selected_hand=selected_hand,
            calibrator=calibrator,
            authority_policy=authority_policy,
        )
        or []
    )
