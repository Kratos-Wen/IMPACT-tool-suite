from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.hoi_empirical_calibration import HOIEmpiricalCalibrator
from core.hoi_eval_utils import is_annotation_json, iter_annotation_paths
from core.hoi_ontology import HOIOntology, ontology_allowed_noun_ids, ontology_noun_required
from core.hoi_query_controller import _apply_authority_policy
from core.hoi_runtime_kernel import SemanticRuntimeRequest, run_event_local_semantic_decode
from core.onset_guidance import build_local_onset_window, build_onset_band
from core.semantic_adapter import load_adapter_package
from core.videomae_v2_logic import (
    VideoMAEHandler,
    aggregate_precomputed_feature_cache,
    load_precomputed_feature_cache,
)


ACTOR_IDS = {
    "left": "Left_hand",
    "left_hand": "Left_hand",
    "right": "Right_hand",
    "right_hand": "Right_hand",
}


@dataclass
class GTEvent:
    clip_id: str
    video_path: str
    hand: str
    event_id: str
    start: int
    onset: int
    end: int
    verb: str
    noun_id: Optional[int]
    noun_label: str


@dataclass
class ClipData:
    clip_id: str
    video_path: str
    frame_count: int
    frame_width: int
    frame_height: int
    object_library: Dict[int, Dict[str, Any]]
    noun_name_to_id: Dict[str, int]
    noun_id_to_name: Dict[int, str]
    bboxes_by_frame: Dict[int, List[Dict[str, Any]]]
    events: List[GTEvent]
    handtracks: Dict[str, Any]
    videomae_cache: Dict[str, Any]


@dataclass
class VariantSpec:
    name: str
    use_handtrack: bool
    use_oracle_grounding: bool
    use_source_arbitration: bool
    description: str
    use_hop: bool = True
    use_scr: bool = True
    use_tsc: bool = True
    use_lock_aware: bool = True


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def _norm_key(value: Any) -> str:
    return re.sub(r"\s+", " ", _norm_text(value)).strip().lower()


def _bounded01(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if math.isnan(out) or math.isinf(out):
        out = float(default)
    return max(0.0, min(1.0, float(out)))


def _round(value: Any, digits: int = 4) -> float:
    return round(float(value), digits)


def _pct(numerator: float, denominator: float) -> float:
    if float(denominator) <= 0.0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def _mean(values: Iterable[float]) -> float:
    rows = [float(v) for v in list(values or [])]
    return float(mean(rows)) if rows else 0.0


def _median(values: Iterable[float]) -> float:
    rows = [float(v) for v in list(values or [])]
    return float(median(rows)) if rows else 0.0


def _normalize_hand_name(value: Any) -> str:
    key = _norm_key(value).replace("-", "_")
    return ACTOR_IDS.get(key, str(value or "").strip())


def _canonical_handedness_actor(value: Any) -> str:
    key = _norm_key(value).replace("-", "_")
    if key.startswith("left"):
        return "Left_hand"
    if key.startswith("right"):
        return "Right_hand"
    return ""


def _track_handedness_stats(track: Dict[str, Any]) -> Dict[str, Any]:
    frames = list((track or {}).get("frames") or [])
    left_weight = 0.0
    right_weight = 0.0
    vote_rows = 0
    noninterp_rows = 0
    for row in frames:
        if not isinstance(row, dict) or bool(row.get("interpolated", False)):
            continue
        noninterp_rows += 1
        hint_actor = _canonical_handedness_actor(row.get("handedness"))
        if not hint_actor:
            continue
        vote_rows += 1
        weight = max(0.05, _bounded01(row.get("handedness_score", 0.0), 0.0))
        if hint_actor == "Left_hand":
            left_weight += float(weight)
        elif hint_actor == "Right_hand":
            right_weight += float(weight)
    total_weight = float(left_weight + right_weight)
    majority_actor = ""
    if left_weight > right_weight + 1e-6:
        majority_actor = "Left_hand"
    elif right_weight > left_weight + 1e-6:
        majority_actor = "Right_hand"
    purity = (max(left_weight, right_weight) / total_weight) if total_weight > 1e-6 else 0.0
    return {
        "frame_count": int(len(frames)),
        "noninterp_rows": int(noninterp_rows),
        "vote_rows": int(vote_rows),
        "vote_weight": float(total_weight),
        "majority_actor": str(majority_actor),
        "purity": float(purity),
    }


def _repair_track_actor_consistency(tracks_out: Dict[str, Any]) -> Dict[str, Any]:
    repaired = {str(actor_id): dict(track or {}) for actor_id, track in dict(tracks_out or {}).items()}
    repaired.setdefault("Left_hand", {})
    repaired.setdefault("Right_hand", {})
    stats = {
        actor_id: _track_handedness_stats(repaired.get(actor_id) or {})
        for actor_id in ("Left_hand", "Right_hand")
    }

    def _is_strong_opposite(actor_id: str) -> bool:
        track_stats = dict(stats.get(actor_id) or {})
        majority_actor = str(track_stats.get("majority_actor") or "")
        return bool(
            majority_actor
            and majority_actor != actor_id
            and float(track_stats.get("purity", 0.0) or 0.0) >= 0.72
            and float(track_stats.get("vote_weight", 0.0) or 0.0) >= 1.25
            and int(track_stats.get("vote_rows", 0) or 0) >= 3
        )

    def _is_weak(actor_id: str) -> bool:
        track_stats = dict(stats.get(actor_id) or {})
        return bool(
            int(track_stats.get("frame_count", 0) or 0) < 3
            or (
                float(track_stats.get("vote_weight", 0.0) or 0.0) < 0.75
                and int(track_stats.get("vote_rows", 0) or 0) < 2
            )
        )

    left_wrong = _is_strong_opposite("Left_hand")
    right_wrong = _is_strong_opposite("Right_hand")
    if left_wrong and right_wrong:
        repaired["Left_hand"], repaired["Right_hand"] = (
            dict(repaired.get("Right_hand") or {}),
            dict(repaired.get("Left_hand") or {}),
        )
    elif left_wrong and _is_weak("Right_hand"):
        repaired["Right_hand"] = dict(repaired.get("Left_hand") or {})
        repaired["Left_hand"] = {}
    elif right_wrong and _is_weak("Left_hand"):
        repaired["Left_hand"] = dict(repaired.get("Right_hand") or {})
        repaired["Right_hand"] = {}

    return {
        str(actor_id): dict(track or {})
        for actor_id, track in repaired.items()
        if dict(track or {}).get("frames")
    }


VERB_TEMPLATE_RATIOS: Dict[str, float] = {
    "hold": 0.02,
    "screw": 0.13,
    "unscrew": 0.15,
    "hand_tighten": 0.24,
    "hand_loose": 0.24,
    "press": 0.24,
    "strip": 0.25,
    "crimp": 0.25,
    "twist": 0.25,
    "place": 0.26,
    "move": 0.31,
    "insert": 0.36,
    "transfer": 0.37,
    "align": 0.42,
    "flip": 0.47,
    "pick_up": 0.50,
    "pull": 0.50,
    "remove": 0.52,
    "ajust": 0.42,
    "put_down": 0.65,
    "cut": 0.72,
}

VERB_FAMILY_DEFAULT_RATIOS: Dict[str, float] = {
    "boundary": 0.02,
    "early": 0.24,
    "mid": 0.45,
    "late": 0.66,
}

VERB_FAMILY_WINDOW_RATIOS: Dict[str, float] = {
    "boundary": 0.12,
    "early": 0.18,
    "mid": 0.22,
    "late": 0.18,
}

VERB_FAMILY_BAND_RATIOS: Dict[str, float] = {
    "boundary": 0.08,
    "early": 0.12,
    "mid": 0.14,
    "late": 0.12,
}

VERB_FAMILY_LABELS: Dict[str, str] = {
    "hold": "boundary",
    "screw": "early",
    "unscrew": "early",
    "hand_tighten": "early",
    "hand_loose": "early",
    "press": "early",
    "strip": "early",
    "crimp": "early",
    "twist": "early",
    "place": "early",
    "move": "mid",
    "insert": "mid",
    "transfer": "mid",
    "align": "mid",
    "flip": "mid",
    "pick_up": "mid",
    "pull": "mid",
    "remove": "mid",
    "ajust": "mid",
    "put_down": "late",
    "cut": "late",
}


def _norm_verb_token(value: Any) -> str:
    return _norm_key(value).replace("-", "_")


def _verb_family_for_label(label: Any) -> str:
    token = _norm_verb_token(label)
    return str(VERB_FAMILY_LABELS.get(token) or "mid")


def _template_ratio_for_label(label: Any) -> float:
    token = _norm_verb_token(label)
    family = _verb_family_for_label(token)
    return float(VERB_TEMPLATE_RATIOS.get(token, VERB_FAMILY_DEFAULT_RATIOS.get(family, 0.45)))


def _semantic_prior_from_scores(candidate_scores: Dict[str, float]) -> Dict[str, Any]:
    scores = {
        str(label): max(0.0, float(score))
        for label, score in dict(candidate_scores or {}).items()
        if _norm_text(label)
    }
    if not scores:
        return {
            "verb_label": "",
            "verb_score": 0.0,
            "family": "mid",
            "family_confidence": 0.0,
            "template_ratio": float(VERB_FAMILY_DEFAULT_RATIOS["mid"]),
        }
    best_label = max(scores.items(), key=lambda item: float(item[1]))[0]
    best_score = float(scores.get(best_label, 0.0) or 0.0)
    total_score = float(sum(float(v) for v in scores.values()))
    family_scores: Dict[str, float] = defaultdict(float)
    for label, score in scores.items():
        family_scores[_verb_family_for_label(label)] += float(score)
    family = _verb_family_for_label(best_label)
    family_conf = (
        float(family_scores.get(family, 0.0) or 0.0) / float(max(1e-6, total_score))
        if total_score > 1e-6
        else 0.0
    )
    return {
        "verb_label": str(best_label),
        "verb_score": float(best_score),
        "family": str(family),
        "family_confidence": float(_bounded01(family_conf, 0.0)),
        "template_ratio": float(_template_ratio_for_label(best_label)),
    }


def _row_motion(row: Dict[str, Any]) -> float:
    return float(row.get("motion", 0.0) or 0.0)


PHASE_PRIOR_BY_FAMILY: Dict[str, Dict[str, float]] = {
    "boundary": {
        "boundary_start": 0.98,
        "contact_valley": 0.62,
        "stabilization_start": 0.80,
        "approach_peak": 0.18,
    },
    "early": {
        "boundary_start": 0.42,
        "contact_valley": 0.96,
        "stabilization_start": 0.84,
        "approach_peak": 0.32,
    },
    "mid": {
        "boundary_start": 0.18,
        "contact_valley": 0.94,
        "stabilization_start": 0.76,
        "approach_peak": 0.54,
    },
    "late": {
        "boundary_start": 0.08,
        "contact_valley": 0.84,
        "stabilization_start": 0.92,
        "approach_peak": 0.38,
    },
}


def _local_peak_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seq = [dict(row or {}) for row in list(rows or [])]
    if len(seq) <= 2:
        return seq
    peaks: List[Dict[str, Any]] = []
    for idx, row in enumerate(seq):
        cur = _row_motion(row)
        prev = _row_motion(seq[idx - 1]) if idx > 0 else cur
        nxt = _row_motion(seq[idx + 1]) if idx + 1 < len(seq) else cur
        if cur >= prev - 1e-6 and cur >= nxt - 1e-6:
            peaks.append(dict(row))
    return peaks or seq


def _local_valley_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seq = [dict(row or {}) for row in list(rows or [])]
    if len(seq) <= 2:
        return seq
    valleys: List[Dict[str, Any]] = []
    for idx, row in enumerate(seq):
        cur = _row_motion(row)
        prev = _row_motion(seq[idx - 1]) if idx > 0 else cur
        nxt = _row_motion(seq[idx + 1]) if idx + 1 < len(seq) else cur
        if cur <= prev + 1e-6 and cur <= nxt + 1e-6:
            valleys.append(dict(row))
    return valleys or seq


def _stable_run_starts(
    rows: Sequence[Dict[str, Any]],
    *,
    low_threshold: float,
    min_run: int = 2,
) -> List[Dict[str, Any]]:
    seq = sorted(
        [dict(row or {}) for row in list(rows or [])],
        key=lambda item: int(item.get("frame", 0) or 0),
    )
    if not seq:
        return []
    out: List[Dict[str, Any]] = []
    run_start: Optional[int] = None
    for idx in range(len(seq) + 1):
        is_low = (
            idx < len(seq)
            and _row_motion(seq[idx]) <= float(low_threshold) + 1e-6
        )
        if is_low and run_start is None:
            run_start = idx
        if (not is_low or idx >= len(seq)) and run_start is not None:
            run = seq[run_start:idx]
            if len(run) >= int(min_run):
                item = dict(run[0] or {})
                item["run_length"] = int(len(run))
                item["run_mean_motion"] = float(
                    sum(_row_motion(row) for row in run) / float(max(1, len(run)))
                )
                out.append(item)
            run_start = None
    return out


def _segment_handedness_purity(rows: Sequence[Dict[str, Any]], hand_key: str) -> float:
    target_actor = _canonical_handedness_actor(hand_key)
    if not target_actor:
        return 1.0
    target_weight = 0.0
    other_weight = 0.0
    total_weight = 0.0
    for row in list(rows or []):
        if not isinstance(row, dict) or bool(row.get("interpolated", False)):
            continue
        actor = _canonical_handedness_actor(row.get("handedness"))
        if not actor:
            continue
        weight = max(0.05, _bounded01(row.get("handedness_score", 0.0), 0.0))
        total_weight += float(weight)
        if actor == target_actor:
            target_weight += float(weight)
        else:
            other_weight += float(weight)
    if total_weight <= 1e-6:
        return 1.0
    if target_weight <= 1e-6 and other_weight > 1e-6:
        return 0.0
    return _bounded01(float(target_weight) / float(total_weight), 1.0)


def _best_template_candidate(
    rows: Sequence[Dict[str, Any]],
    *,
    target_frame: int,
    search_left: int,
    search_right: int,
) -> Dict[str, Any]:
    search_rows = [
        dict(row or {})
        for row in list(rows or [])
        if search_left <= int(row.get("frame", target_frame) or target_frame) <= search_right
    ]
    if not search_rows:
        search_rows = [dict(row or {}) for row in list(rows or [])]
    if not search_rows:
        return {}
    valley_rows = _local_valley_rows(search_rows)
    min_motion = min(_row_motion(row) for row in search_rows)
    max_motion = max(_row_motion(row) for row in search_rows)
    motion_span = float(max(1e-6, max_motion - min_motion))
    window_span = float(max(3, search_right - search_left + 1))
    best_row: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    for row in valley_rows:
        frame = int(row.get("frame", target_frame) or target_frame)
        motion = _row_motion(row)
        motion_gain = 1.0 - ((motion - min_motion) / motion_span)
        dist_gain = 1.0 - (abs(int(frame) - int(target_frame)) / window_span)
        score = (
            0.60 * float(_bounded01(motion_gain, 0.0))
            + 0.40 * float(_bounded01(dist_gain, 0.0))
            - (0.08 if bool(row.get("interpolated", False)) else 0.0)
        )
        if (
            best_score is None
            or float(score) > float(best_score)
            or (
                abs(float(score) - float(best_score)) <= 1e-6
                and abs(int(frame) - int(target_frame))
                < abs(int((best_row or {}).get("frame", target_frame) or target_frame) - int(target_frame))
            )
        ):
            best_row = dict(row)
            best_score = float(score)
    return dict(best_row or {})


def _best_peak_candidate(
    rows: Sequence[Dict[str, Any]],
    *,
    target_frame: int,
    search_left: int,
    search_right: int,
) -> Dict[str, Any]:
    search_rows = [
        dict(row or {})
        for row in list(rows or [])
        if search_left <= int(row.get("frame", target_frame) or target_frame) <= search_right
    ]
    if not search_rows:
        search_rows = [dict(row or {}) for row in list(rows or [])]
    if not search_rows:
        return {}
    peak_rows = _local_peak_rows(search_rows)
    min_motion = min(_row_motion(row) for row in search_rows)
    max_motion = max(_row_motion(row) for row in search_rows)
    motion_span = float(max(1e-6, max_motion - min_motion))
    window_span = float(max(3, search_right - search_left + 1))
    best_row: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    for row in peak_rows:
        frame = int(row.get("frame", target_frame) or target_frame)
        motion = _row_motion(row)
        motion_gain = (motion - min_motion) / motion_span
        dist_gain = 1.0 - (abs(int(frame) - int(target_frame)) / window_span)
        score = (
            0.68 * float(_bounded01(motion_gain, 0.0))
            + 0.32 * float(_bounded01(dist_gain, 0.0))
            - (0.08 if bool(row.get("interpolated", False)) else 0.0)
        )
        if (
            best_score is None
            or float(score) > float(best_score)
            or (
                abs(float(score) - float(best_score)) <= 1e-6
                and abs(int(frame) - int(target_frame))
                < abs(int((best_row or {}).get("frame", target_frame) or target_frame) - int(target_frame))
            )
        ):
            best_row = dict(row)
            best_score = float(score)
    return dict(best_row or {})


def _phase_prior_score(family: str, candidate_kind: str) -> float:
    table = dict(PHASE_PRIOR_BY_FAMILY.get(str(family or "mid"), PHASE_PRIOR_BY_FAMILY["mid"]) or {})
    return float(table.get(str(candidate_kind or ""), 0.18))


def _build_hop_band(
    *,
    start: int,
    end: int,
    onset_frame: int,
    family: str,
    candidate_kind: str,
    rows: Sequence[Dict[str, Any]],
    chosen_row: Dict[str, Any],
    confidence: float,
) -> Dict[str, Any]:
    segment_len = max(1, int(end) - int(start))
    base_half = max(2, int(round(float(VERB_FAMILY_BAND_RATIOS.get(family, 0.14) or 0.14) * float(segment_len))))
    chosen_motion = _row_motion(chosen_row)
    min_motion = min([chosen_motion] + [_row_motion(row) for row in list(rows or [])])
    peak_motion = max([chosen_motion] + [_row_motion(row) for row in list(rows or [])])
    cluster_half = max(2, int(round(0.20 * float(segment_len))))
    if str(candidate_kind or "") == "approach_peak":
        threshold = float(chosen_motion - 0.18 * max(0.0, float(chosen_motion) - float(min_motion)))
        cluster_rows = [
            dict(row or {})
            for row in list(rows or [])
            if abs(int(row.get("frame", onset_frame) or onset_frame) - int(onset_frame)) <= cluster_half
            and _row_motion(row) >= threshold - 1e-6
        ]
    else:
        threshold = float(chosen_motion + 0.18 * max(0.0, float(peak_motion) - float(chosen_motion)))
        cluster_rows = [
            dict(row or {})
            for row in list(rows or [])
            if abs(int(row.get("frame", onset_frame) or onset_frame) - int(onset_frame)) <= cluster_half
            and _row_motion(row) <= threshold + 1e-6
            and (
                str(candidate_kind or "") != "stabilization_start"
                or int(row.get("frame", onset_frame) or onset_frame) >= int(onset_frame)
            )
        ]
    if family == "boundary":
        left_frame = int(start)
        if cluster_rows:
            right_frame = min(int(end), max(int(row.get("frame", onset_frame) or onset_frame) for row in cluster_rows))
        else:
            right_frame = min(int(end), int(onset_frame) + int(base_half))
    elif cluster_rows:
        left_frame = max(int(start), min(int(row.get("frame", onset_frame) or onset_frame) for row in cluster_rows))
        right_frame = min(int(end), max(int(row.get("frame", onset_frame) or onset_frame) for row in cluster_rows))
    else:
        left_frame = max(int(start), int(onset_frame) - int(base_half))
        right_frame = min(int(end), int(onset_frame) + int(base_half))
    if right_frame - left_frame < 2:
        left_frame = max(int(start), int(onset_frame) - int(base_half))
        right_frame = min(int(end), int(onset_frame) + int(base_half))
    if right_frame < left_frame:
        left_frame = int(onset_frame)
        right_frame = int(onset_frame)
    return {
        "center_frame": int(onset_frame),
        "start_frame": int(left_frame),
        "end_frame": int(right_frame),
        "width": int(max(1, right_frame - left_frame + 1)),
        "segment_start": int(start),
        "segment_end": int(end),
        "status": "suggested",
        "confidence_proxy": float(_bounded01(confidence, 0.0)),
        "center_ratio": float(_bounded01((int(onset_frame) - int(start)) / float(max(1, segment_len)), 0.5)),
        "left_ratio": float(_bounded01((int(left_frame) - int(start)) / float(max(1, segment_len)), 0.0)),
        "right_ratio": float(_bounded01((int(right_frame) - int(start)) / float(max(1, segment_len)), 1.0)),
    }


def _read_label_list(path: str) -> List[str]:
    rows: List[str] = []
    if not path or not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = _norm_text(raw)
            if not line:
                continue
            head = _norm_text(line.split(",", 1)[0])
            if head:
                rows.append(head)
    return rows


def _discover_existing_path(candidates: Sequence[str]) -> str:
    for path in list(candidates or []):
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    return ""


def _discover_nouns_path(dataset_root: str, explicit_path: str = "") -> str:
    if explicit_path:
        return os.path.abspath(explicit_path)
    root = os.path.abspath(str(dataset_root or "").strip())
    return _discover_existing_path(
        [
            os.path.join(root, "nouns.txt"),
            os.path.join(root, "tar.txt"),
            os.path.join(root, "ins.txt"),
            os.path.join(ROOT, "test_files", "GoodVideosSegments", "nouns.txt"),
            os.path.join(ROOT, "test_files", "nouns.txt"),
        ]
    )


def _discover_verbs_path(dataset_root: str, explicit_path: str = "") -> str:
    if explicit_path:
        return os.path.abspath(explicit_path)
    root = os.path.abspath(str(dataset_root or "").strip())
    return _discover_existing_path(
        [
            os.path.join(root, "verbs.txt"),
            os.path.join(ROOT, "test_files", "GoodVideosSegments", "verbs.txt"),
            os.path.join(ROOT, "test_files", "verbs.txt"),
        ]
    )


def _discover_ontology_path(dataset_root: str, explicit_path: str = "") -> str:
    if explicit_path:
        return os.path.abspath(explicit_path)
    root = os.path.abspath(str(dataset_root or "").strip())
    return _discover_existing_path(
        [
            os.path.join(root, "verb_noun_ontology.csv"),
            os.path.join(root, "hoi_ontology.csv"),
            os.path.join(root, "ontology.csv"),
            os.path.join(ROOT, "test_files", "GoodVideosSegments", "verb_noun_ontology.csv"),
            os.path.join(ROOT, "test_files", "verb_noun_ontology.csv"),
        ]
    )


def _discover_gt_paths(dataset_root: str, recursive: bool = False) -> List[str]:
    root = os.path.abspath(str(dataset_root or "").strip())
    if not root:
        return []
    if bool(recursive):
        return sorted(iter_annotation_paths(root))
    paths: List[str] = []
    if os.path.isfile(root):
        return [root] if is_annotation_json(root) else []
    if not os.path.isdir(root):
        return []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if is_annotation_json(path):
            paths.append(path)
    return sorted(paths)


def _normalize_video_stem_for_match(value: Any) -> str:
    text = _norm_text(value)
    if not text:
        return ""
    text = os.path.splitext(os.path.basename(text))[0]
    text = re.sub(r"_rgb_a$", "_rgb", text, flags=re.IGNORECASE)
    return _norm_key(text)


def _build_video_index(dataset_root: str, recursive: bool = False) -> Dict[str, List[str]]:
    root = os.path.abspath(str(dataset_root or "").strip())
    index: Dict[str, List[str]] = defaultdict(list)
    if bool(recursive):
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if name.lower().endswith(".mp4"):
                    path = os.path.join(dirpath, name)
                    index[_normalize_video_stem_for_match(name)].append(path)
    else:
        for name in os.listdir(root):
            if name.lower().endswith(".mp4"):
                path = os.path.join(root, name)
                index[_normalize_video_stem_for_match(name)].append(path)
    return {str(key): sorted(list(values or [])) for key, values in index.items()}


def _resolve_video_path_for_gt(gt_path: str, data: Dict[str, Any], video_index: Dict[str, List[str]]) -> str:
    gt_path = os.path.abspath(gt_path)
    gt_dir = os.path.dirname(gt_path)
    gt_name = os.path.splitext(os.path.basename(gt_path))[0]
    sibling_mp4s = sorted(
        os.path.join(gt_dir, name)
        for name in os.listdir(gt_dir)
        if name.lower().endswith(".mp4")
    )

    desired_names: List[str] = []
    raw_video_path = _norm_text(data.get("video_path"))
    raw_video_id = _norm_text(data.get("video_id"))
    derived_name = re.sub(r"_hoi_bbox$", "", gt_name, flags=re.IGNORECASE)
    derived_name = re.sub(
        r"_(manual|assist|full_assist|fullassist|validation)$",
        "",
        derived_name,
        flags=re.IGNORECASE,
    )
    for candidate in [raw_video_path, raw_video_id, derived_name]:
        stem = os.path.splitext(os.path.basename(candidate))[0]
        if stem:
            desired_names.append(f"{stem}.mp4")

    for name in desired_names:
        path = os.path.join(gt_dir, name)
        if os.path.isfile(path):
            return os.path.abspath(path)

    desired_norms = {
        _normalize_video_stem_for_match(item)
        for item in [raw_video_path, raw_video_id, derived_name]
        if _normalize_video_stem_for_match(item)
    }
    for path in sibling_mp4s:
        if _normalize_video_stem_for_match(path) in desired_norms:
            return os.path.abspath(path)

    if len(sibling_mp4s) == 1:
        return os.path.abspath(sibling_mp4s[0])

    for norm in sorted(desired_norms):
        candidates = list(video_index.get(norm) or [])
        if len(candidates) == 1:
            return os.path.abspath(candidates[0])
        same_dir = [path for path in candidates if os.path.dirname(os.path.abspath(path)) == gt_dir]
    raise FileNotFoundError(
        f"Unable to resolve video for GT file: {gt_path}\n"
        f"Requested names: {desired_names}\n"
        f"Sibling mp4s: {[os.path.basename(path) for path in sibling_mp4s]}"
    )


def _object_name_for_id(object_library: Dict[int, Dict[str, Any]], object_id: Optional[int]) -> str:
    if object_id is None:
        return ""
    info = dict(object_library.get(int(object_id)) or {})
    return _norm_text(info.get("label")) or _norm_text(info.get("category"))


def _find_handtrack_file(handtracks_dir: str, clip_id: str) -> str:
    root = os.path.abspath(str(handtracks_dir or "").strip())
    if not root or not os.path.isdir(root):
        return ""
    matches = sorted(
        os.path.join(root, name)
        for name in os.listdir(root)
        if name.startswith(f"{clip_id}.handtracks.") and name.endswith(".json")
    )
    return matches[0] if matches else ""


def _normalize_handtrack_payload(payload: Optional[dict]) -> Dict[str, Any]:
    out = dict(payload or {})
    tracks_out: Dict[str, Any] = {}
    for actor_id, track in dict(out.get("tracks") or {}).items():
        rows = []
        frame_map = {}
        raw_frames = list((track or {}).get("frames") or [])
        
        centers: Dict[int, Tuple[float, float]] = {}
        for row in raw_frames:
            f_id = _safe_int(row.get("frame", row.get("f")))
            if f_id is None: continue
            bbox = list(row.get("bbox") or [])
            if not bbox and "x1" in row:
                bbox = [float(row.get("x1",0)), float(row.get("y1",0)), float(row.get("x2",0)), float(row.get("y2",0))]
            if len(bbox) >= 4:
                centers[int(f_id)] = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)

        for i, row in enumerate(raw_frames):
            if not isinstance(row, dict):
                continue
            frame = _safe_int(row.get("frame", row.get("f")))
            if frame is None:
                continue
            
            bbox = list(row.get("bbox") or [])
            if not bbox and "x1" in row:
                bbox = [float(row.get("x1", 0.0)), float(row.get("y1", 0.0)), float(row.get("x2", 0.0)), float(row.get("y2", 0.0))]
            if len(bbox) < 4:
                bbox = [0.0, 0.0, 0.0, 0.0]

            center = list(row.get("center") or [])
            if not center and "cx" in row:
                center = [float(row.get("cx", 0.0)), float(row.get("cy", 0.0))]
            if len(center) < 2:
                center = [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5] if len(bbox) >= 4 else [0.0, 0.0]

            motion = _safe_float(row.get("motion"))
            if "motion" not in row or motion <= 0:
                prev_c = centers.get(int(frame) - 1)
                curr_c = (center[0], center[1])
                if prev_c:
                    dx = (curr_c[0] - prev_c[0]) * 1920
                    dy = (curr_c[1] - prev_c[1]) * 1080
                    motion = math.hypot(dx, dy)
                else:
                    motion = 0.0

            clean = {
                "frame": int(frame),
                "bbox": [float(v) for v in bbox[:4]],
                "center": [float(v) for v in center[:2]],
                "area": float(row.get("area", 0.0) or 0.0),
                "motion": float(motion),
                "handedness": str(row.get("handedness") or "").strip().lower(),
                "handedness_score": float(row.get("handedness_score", 0.0) or 0.0),
                "detection_confidence": float(row.get("detection_confidence", 0.0) or 0.0),
                "interpolated": bool(row.get("interpolated", False)),
            }
            rows.append(clean)
            frame_map[int(frame)] = clean
            
        rows.sort(key=lambda item: int(item.get("frame", 0)))
        
        if len(rows) > 3:
            for i in range(1, len(rows) - 1):
                rows[i]["motion"] = 0.5 * rows[i]["motion"] + 0.25 * rows[i-1]["motion"] + 0.25 * rows[i+1]["motion"]
            
        norm_actor = _normalize_hand_name(actor_id)
        tracks_out[norm_actor] = {
            "frame_count": int((track or {}).get("frame_count", out.get("frame_count", 0)) or 0),
            "coverage": float((track or {}).get("coverage", 0.0) or 0.0),
            "motion_peak_frame": (
                None
                if (track or {}).get("motion_peak_frame") is None
                else int((track or {}).get("motion_peak_frame"))
            ),
            "motion_peak_score": float((track or {}).get("motion_peak_score", 0.0) or 0.0),
            "frames": rows,
            "frame_map": frame_map,
        }
    out["tracks"] = _repair_track_actor_consistency(tracks_out)
    out["frame_count"] = int(out.get("frame_count", 0) or 0)
    out["stride"] = int(out.get("stride", 1) or 1)
    return out


def _load_handtracks(handtracks_dir: str, clip_id: str) -> Dict[str, Any]:
    path = _find_handtrack_file(handtracks_dir, clip_id)
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return _normalize_handtrack_payload(payload)


def _ensure_cache_dir(path: str) -> str:
    target = os.path.abspath(str(path or "").strip())
    os.makedirs(target, exist_ok=True)
    return target


def _expand_calibration_inputs(paths: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in list(paths or []):
        path = os.path.abspath(str(raw or "").strip())
        if not path:
            continue
        candidates: List[str] = []
        if os.path.isdir(path):
            for dirpath, _dirnames, filenames in os.walk(path):
                for name in sorted(filenames):
                    if name.lower().endswith(".ops.log.csv"):
                        candidates.append(os.path.join(dirpath, name))
        elif os.path.isfile(path):
            candidates.append(path)
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
    return out


def _cache_output_path(cache_dir: str, clip_id: str, stride: int, window_span: int) -> str:
    return os.path.join(
        cache_dir,
        f"{clip_id}.stride{int(stride)}.span{int(window_span)}.videomae_cache.npz",
    )


def _ensure_videomae_cache(
    *,
    video_path: str,
    clip_id: str,
    cache_dir: str,
    handler: VideoMAEHandler,
    stride: int,
    window_span: int,
) -> Dict[str, Any]:
    output_path = _cache_output_path(cache_dir, clip_id, stride, window_span)
    cache = load_precomputed_feature_cache(output_path)
    if cache is not None:
        return cache
    payload, err = handler.build_dense_feature_cache(
        video_path,
        stride=int(stride),
        window_span=int(window_span),
    )
    if payload is None:
        raise RuntimeError(f"Failed to build VideoMAE cache for {clip_id}: {err}")
    ok, save_msg = handler.save_dense_feature_cache(payload, output_path)
    if not ok:
        raise RuntimeError(f"Failed to save VideoMAE cache for {clip_id}: {save_msg}")
    cache = load_precomputed_feature_cache(output_path)
    if cache is None:
        raise RuntimeError(f"Saved VideoMAE cache could not be reloaded for {clip_id}")
    return cache


def _event_noun_id(
    event: Dict[str, Any],
    tracks: Dict[str, Any],
    noun_name_to_id: Dict[str, int],
    object_library: Dict[int, Dict[str, Any]],
) -> Optional[int]:
    noun_id = _safe_int(event.get("noun_object_id"))
    if noun_id is not None:
        return int(noun_id)
    links = dict(event.get("links") or {})
    target_track_id = _norm_text(links.get("target_track_id"))
    if target_track_id:
        track = dict(tracks.get(target_track_id) or {})
        track_object_id = _safe_int(track.get("object_id"))
        if track_object_id is not None:
            return int(track_object_id)
    interaction = dict(event.get("interaction") or {})
    target_name = _norm_text(interaction.get("target")) or _norm_text(interaction.get("noun"))
    if target_name:
        for name, object_id in noun_name_to_id.items():
            if _norm_key(name) == _norm_key(target_name):
                return int(object_id)
    noun_label = _norm_text(interaction.get("target")) or _norm_text(interaction.get("noun"))
    if noun_label:
        for object_id, info in object_library.items():
            label = _norm_text((info or {}).get("label")) or _norm_text((info or {}).get("category"))
            if _norm_key(label) == _norm_key(noun_label):
                return int(object_id)
    return None


def _load_clip_data(
    *,
    dataset_root: str,
    annotation_root: str = "",
    nouns_path: str,
    ontology: HOIOntology,
    handtracks_dir: str,
    cache_dir: str,
    videomae_handler: VideoMAEHandler,
    stride: int,
    window_span: int,
    clip_ids: Optional[Sequence[str]] = None,
    recursive: bool = False,
) -> List[ClipData]:
    root = os.path.abspath(str(dataset_root or "").strip())
    annotation_src = os.path.abspath(str(annotation_root or dataset_root or "").strip())
    clip_filter = {_norm_text(item) for item in list(clip_ids or []) if _norm_text(item)}
    noun_names = [name for name in _read_label_list(nouns_path) if _norm_text(name)]
    noun_name_to_id = {str(name): int(idx) for idx, name in enumerate(noun_names)}
    noun_id_to_name = {int(idx): str(name) for idx, name in enumerate(noun_names)}
    gt_paths = _discover_gt_paths(annotation_src, recursive=bool(recursive))
    video_index = _build_video_index(root, recursive=bool(recursive))

    clips: List[ClipData] = []
    for gt_path in gt_paths:
        name = os.path.basename(gt_path)
        with open(gt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_clip_id = re.sub(r"_hoi_bbox$", "", os.path.splitext(name)[0], flags=re.IGNORECASE)
        video_path = _resolve_video_path_for_gt(gt_path, data, video_index)
        clip_id = os.path.splitext(os.path.basename(video_path))[0]
        candidate_ids = {raw_clip_id, clip_id, _norm_text(data.get("video_id"))}
        if clip_filter and not any(item in clip_filter for item in candidate_ids if item):
            continue

        raw_library = dict(data.get("object_library") or {})
        object_library: Dict[int, Dict[str, Any]] = {}
        for raw_id, info in raw_library.items():
            object_id = _safe_int(raw_id)
            if object_id is None:
                object_id = _safe_int((info or {}).get("object_id"))
            if object_id is None:
                continue
            object_library[int(object_id)] = dict(info or {})

        tracks = dict(data.get("tracks") or {})
        bboxes_by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for _track_id, track in tracks.items():
            if not isinstance(track, dict):
                continue
            object_id = _safe_int(track.get("object_id"))
            if object_id is None:
                continue
            label = _object_name_for_id(object_library, object_id) or _norm_text(track.get("category"))
            for row in list(track.get("boxes") or []):
                if not isinstance(row, dict):
                    continue
                frame = _safe_int(row.get("frame"))
                bbox = list(row.get("bbox") or [])
                if frame is None or len(bbox) < 4:
                    continue
                bboxes_by_frame[int(frame)].append(
                    {
                        "id": int(object_id),
                        "label": str(label),
                        "source": "oracle_box",
                        "confidence": 1.0,
                        "frame": int(frame),
                        "x1": float(bbox[0]),
                        "y1": float(bbox[1]),
                        "x2": float(bbox[2]),
                        "y2": float(bbox[3]),
                    }
                )

        events: List[GTEvent] = []
        hoi_events = dict(data.get("hoi_events") or {})
        for raw_hand, hand_rows in hoi_events.items():
            hand = _normalize_hand_name(raw_hand)
            if not isinstance(hand_rows, list):
                continue
            for item in hand_rows:
                if not isinstance(item, dict):
                    continue
                start = _safe_int(item.get("start_frame"))
                onset = _safe_int(item.get("contact_onset_frame"))
                end = _safe_int(item.get("end_frame"))
                if start is None or onset is None or end is None:
                    continue
                if end < start:
                    start, end = end, start
                noun_id = _event_noun_id(item, tracks, noun_name_to_id, object_library)
                noun_label = (
                    _object_name_for_id(object_library, noun_id)
                    or _norm_text((item.get("interaction") or {}).get("target"))
                    or _norm_text((item.get("interaction") or {}).get("noun"))
                )
                events.append(
                    GTEvent(
                        clip_id=clip_id,
                        video_path=video_path,
                        hand=hand,
                        event_id=_norm_text(item.get("event_id")) or f"{hand}_{len(events) + 1:03d}",
                        start=int(start),
                        onset=int(onset),
                        end=int(end),
                        verb=_norm_text(item.get("verb")),
                        noun_id=None if noun_id is None else int(noun_id),
                        noun_label=str(noun_label),
                    )
                )

        handtracks = _load_handtracks(handtracks_dir, clip_id)
        videomae_cache = _ensure_videomae_cache(
            video_path=video_path,
            clip_id=clip_id,
            cache_dir=cache_dir,
            handler=videomae_handler,
            stride=int(stride),
            window_span=int(window_span),
        )
        frame_size = list(data.get("frame_size") or [0, 0])
        clips.append(
            ClipData(
                clip_id=clip_id,
                video_path=video_path,
                frame_count=int(data.get("frame_count", 0) or 0),
                frame_width=int(frame_size[0] if len(frame_size) > 0 else 0),
                frame_height=int(frame_size[1] if len(frame_size) > 1 else 0),
                object_library=object_library,
                noun_name_to_id=dict(noun_name_to_id),
                noun_id_to_name=dict(noun_id_to_name),
                bboxes_by_frame={int(k): [dict(v) for v in vals] for k, vals in bboxes_by_frame.items()},
                events=events,
                handtracks=handtracks,
                videomae_cache=videomae_cache,
            )
        )
    return clips


def _make_state(event: GTEvent) -> Dict[str, Any]:
    return {
        "interaction_start": int(event.start),
        "functional_contact_onset": None,
        "interaction_end": int(event.end),
        "verb": "",
        "noun_object_id": None,
        "_field_state": {
            "functional_contact_onset": {},
            "verb": {},
            "noun_object_id": {},
        },
    }


def _set_confirmed_field(state: Dict[str, Any], field_name: str, value: Any) -> Dict[str, Any]:
    out = {
        "interaction_start": state.get("interaction_start"),
        "functional_contact_onset": state.get("functional_contact_onset"),
        "interaction_end": state.get("interaction_end"),
        "verb": state.get("verb"),
        "noun_object_id": state.get("noun_object_id"),
        "_field_state": {
            "functional_contact_onset": dict((state.get("_field_state") or {}).get("functional_contact_onset") or {}),
            "verb": dict((state.get("_field_state") or {}).get("verb") or {}),
            "noun_object_id": dict((state.get("_field_state") or {}).get("noun_object_id") or {}),
        },
    }
    out[field_name] = value
    out["_field_state"][field_name] = {"status": "confirmed", "source": "oracle"}
    return out


class OfflineHOIRuntime:
    def __init__(
        self,
        *,
        clip: ClipData,
        package: Any,
        ontology: HOIOntology,
        variant: VariantSpec,
        refinement_passes: int,
        calibrator: Optional[HOIEmpiricalCalibrator] = None,
        enable_handtrack_fusion: bool = False,
        enable_cross_hand_exclusion: bool = True,
    ) -> None:
        self.clip = clip
        self.package = package
        self.ontology = ontology
        self.variant = variant
        self.refinement_passes = int(refinement_passes) if variant.use_scr else 0
        self.calibrator = calibrator or HOIEmpiricalCalibrator()
        self.enable_handtrack_fusion = bool(enable_handtrack_fusion) and variant.use_hop
        self.enable_cross_hand_exclusion = bool(enable_cross_hand_exclusion)
        self.verb_labels = [str(v) for v in list(package.verb_labels or [])]
        self.noun_ids = [int(v) for v in list(package.noun_ids or [])]
        self.feature_layout = dict(getattr(package, "feature_layout", {}) or {})
        self._unguided_candidate_cache: Dict[Tuple[int, int], Dict[str, float]] = {}

    def noun_required_for_verb(self, verb_name: Any) -> bool:
        return ontology_noun_required(self.ontology, verb_name)

    def allowed_noun_ids_for_verb(self, verb_name: Any) -> List[int]:
        return ontology_allowed_noun_ids(
            self.ontology,
            verb_name,
            self.clip.noun_name_to_id,
        )

    def noun_exists_prior_for_verb(self, verb_name: Any) -> float:
        text = _norm_text(verb_name)
        if not text:
            return 0.5
        if self.ontology is None or not self.ontology.has_verb(text):
            return 0.75
        return 0.15 if self.ontology.allow_no_noun(text) else 0.95

    def get_field_state(self, state: Dict[str, Any], field_name: str) -> Dict[str, Any]:
        return dict((state.get("_field_state") or {}).get(field_name) or {})

    def semantic_source_is_runtime_lock(self, source: Any) -> bool:
        text = _norm_text(source).lower()
        if not text:
            return False
        model_prefixes = (
            "semantic_adapter",
            "videomae",
            "onset_local_completion",
            "onset_noun_completion",
            "onset_role_completion",
            "query_controller",
        )
        return not any(text.startswith(prefix) for prefix in model_prefixes)

    def semantic_reinfer_hint(self, state: Dict[str, Any]) -> Dict[str, Any]:
        onset_anchor_ratio = None
        onset_anchor_half_width = None
        onset_value = _safe_int(state.get("functional_contact_onset"))
        start = _safe_int(state.get("interaction_start"))
        end = _safe_int(state.get("interaction_end"))
        onset_state = self.get_field_state(state, "functional_contact_onset")
        onset_status = _norm_text(onset_state.get("status")).lower()
        if onset_value is not None and onset_status != "confirmed" and start is not None and end is not None:
            if end < start:
                start, end = end, start
            segment_len = max(1, int(end) - int(start))
            onset_anchor_ratio = _bounded01(
                (int(onset_value) - int(start)) / float(segment_len),
                0.5,
            )
            onset_anchor_half_width = 0.08 if onset_status == "suggested" else 0.12
        return {
            "onset_anchor_ratio": onset_anchor_ratio,
            "onset_anchor_half_width": onset_anchor_half_width,
        }

    def cross_hand_context(self, event: GTEvent, state: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enable_cross_hand_exclusion:
            return {}
        start = _safe_int(state.get("interaction_start"))
        end = _safe_int(state.get("interaction_end"))
        if start is None or end is None:
            return {}
        if end < start:
            start, end = end, start
        segment_len = max(1, int(end) - int(start))
        mid = float(start + end) * 0.5
        rows: List[Dict[str, Any]] = []
        for other in list(self.clip.events or []):
            if other.clip_id != event.clip_id or other.hand == event.hand:
                continue
            onset = _safe_int(other.onset)
            if onset is None or onset < start or onset > end:
                continue
            overlap = max(0.0, min(float(end), float(other.end)) - max(float(start), float(other.start)))
            union = max(float(end), float(other.end)) - min(float(start), float(other.start))
            overlap_ratio = float(overlap / union) if union > 0.0 else 0.0
            matched_event_id = bool(event.event_id and other.event_id and str(event.event_id) == str(other.event_id))
            exclude_weight = 0.16 if matched_event_id else (0.12 if overlap_ratio > 0.0 else 0.08)
            rows.append(
                {
                    "hand": other.hand,
                    "event_id": other.event_id,
                    "onset_frame": int(onset),
                    "onset_ratio": _bounded01((int(onset) - int(start)) / float(segment_len), 0.5),
                    "status": "confirmed",
                    "exclude_weight": float(exclude_weight),
                    "overlap_ratio": float(overlap_ratio),
                    "distance_to_mid": abs(float(onset) - mid),
                }
            )
        rows.sort(
            key=lambda row: (
                -float(row.get("exclude_weight", 0.0)),
                -float(row.get("overlap_ratio", 0.0)),
                float(row.get("distance_to_mid", 0.0)),
            )
        )
        return {
            "other_hands": rows,
            "exclusive_count": int(len(rows)),
            "primary_exclusion": dict(rows[0]) if rows else {},
        }

    def filter_allowed_object_candidates(
        self,
        candidates: Sequence[Dict[str, Any]],
        verb_name: Any,
    ) -> List[Dict[str, Any]]:
        allowed_ids = {
            int(v)
            for v in list(self.allowed_noun_ids_for_verb(verb_name) or [])
            if _safe_int(v) is not None
        }
        if not allowed_ids:
            return [dict(item) for item in list(candidates or []) if isinstance(item, dict)]
        return [
            dict(item)
            for item in list(candidates or [])
            if isinstance(item, dict) and _safe_int(item.get("object_id")) in allowed_ids
        ]

    def detector_grounding_noun_candidate(self, event: GTEvent, state: Dict[str, Any], verb_name: Any) -> Dict[str, Any]:
        candidates = self.collect_event_object_candidates(event, state)
        candidates = self.filter_allowed_object_candidates(candidates, verb_name)
        return dict((candidates or [None])[0] or {})

    def best_grounding_candidate_for_noun(
        self,
        event: GTEvent,
        state: Dict[str, Any],
        noun_object_id: Any,
        verb_name: Any,
    ) -> Dict[str, Any]:
        noun_id = _safe_int(noun_object_id)
        if noun_id is None:
            return {}
        for item in self.filter_allowed_object_candidates(self.collect_event_object_candidates(event, state), verb_name):
            if _safe_int(item.get("object_id")) == int(noun_id):
                return dict(item)
        return {}

    def _confidence_only_completion_authority(
        self,
        *,
        reliability: float,
        overwrite_risk: float,
    ) -> Dict[str, Any]:
        reliability = _bounded01(reliability, 0.0)
        overwrite_risk = _bounded01(overwrite_risk, 0.0)
        if reliability >= 0.80 and overwrite_risk <= 0.15:
            return {
                "action_kind": "propagate",
                "authority_level": "safe_local",
                "interaction_form": "bundle_accept",
                "apply_mode": "apply_completion_bundle",
                "safe_apply": True,
                "authority_policy_code": "confidence_only_safe_local",
                "authority_policy_name": "confidence_only",
            }
        if reliability >= 0.48:
            return {
                "action_kind": "suggest",
                "authority_level": "human_confirm",
                "interaction_form": "accept_suggestion",
                "apply_mode": "apply_suggestion",
                "safe_apply": True,
                "authority_policy_code": "confidence_only_human_confirm",
                "authority_policy_name": "confidence_only",
            }
        return {
            "action_kind": "query",
            "authority_level": "human_only",
            "interaction_form": "manual_edit",
            "apply_mode": "",
            "safe_apply": False,
            "authority_policy_code": "confidence_only_manual_only",
            "authority_policy_name": "confidence_only",
        }

    def _tsc_completion_decision(
        self,
        *,
        field_name: str,
        source: str,
        query_type: str,
        runtime_reliability: float,
        overwrite_risk: float,
        propagation_gain: float,
        human_cost_est: float,
        base_priority: float,
    ) -> Dict[str, Any]:
        runtime_reliability = _bounded01(runtime_reliability, 0.0)
        overwrite_risk = _bounded01(overwrite_risk, 0.0)
        propagation_gain = _bounded01(propagation_gain, 0.0)
        human_cost_est = _bounded01(human_cost_est, 0.0)
        base_priority = float(base_priority)
        if not self.variant.use_tsc:
            out = self._confidence_only_completion_authority(
                reliability=float(runtime_reliability),
                overwrite_risk=float(overwrite_risk),
            )
            out["acceptance_prob_est"] = float(runtime_reliability)
            out["calibrated_reliability"] = float(runtime_reliability)
            out["empirical_support_n"] = 0
            out["human_cost_est"] = float(human_cost_est)
            out["overwrite_risk"] = float(overwrite_risk)
            out["propagation_gain"] = float(propagation_gain)
            out["base_priority"] = float(base_priority)
            out["selected_bonus"] = 0.0
            out["query_type"] = str(query_type or "")
            out["field_name"] = str(field_name or "")
            out["field"] = str(field_name or "")
            out["suggested_source"] = str(source or "")
            out["source"] = str(source or "")
            return out

        query = {
            "query_type": str(query_type or f"complete_{field_name}"),
            "field_name": str(field_name or ""),
            "field": str(field_name or ""),
            "suggested_source": str(source or ""),
            "source": str(source or ""),
            "action_kind": "propagate",
            "authority_level": "safe_local",
            "interaction_form": "bundle_accept",
            "apply_mode": "apply_completion_bundle",
            "safe_apply": True,
            "base_priority": float(base_priority),
            "propagation_gain": float(propagation_gain),
            "human_cost_est": float(human_cost_est),
            "overwrite_risk": float(overwrite_risk),
            "selected_bonus": 0.0,
            "acceptance_prob_est": float(runtime_reliability),
            "calibrated_reliability": float(runtime_reliability),
        }
        query = self.calibrator.calibrate_query(query)
        empirical_accept = _bounded01(query.get("acceptance_prob_est", runtime_reliability), runtime_reliability)
        query["calibrated_reliability"] = _bounded01(
            0.72 * float(runtime_reliability) + 0.28 * float(empirical_accept),
            runtime_reliability,
        )
        query = _apply_authority_policy(query)
        return dict(query or {})

    def estimate_noun_source_decision(
        self,
        event: GTEvent,
        state: Dict[str, Any],
        *,
        verb_name: Any,
        semantic_noun_id: Any,
        semantic_confidence: float,
    ) -> Dict[str, Any]:
        if not self.variant.use_source_arbitration:
            noun_id = _safe_int(semantic_noun_id)
            return {
                "preferred_source": "semantic_adapter_noun" if noun_id is not None else "",
                "preferred_family": "semantic" if noun_id is not None else "",
                "semantic_noun_id": noun_id,
                "semantic_confidence": float(semantic_confidence or 0.0),
                "detector_noun_id": None,
                "detector_confidence": 0.0,
                "detector_candidate": {},
                "grounding_candidate": {},
                "score_margin": float(semantic_confidence or 0.0),
                "decision_basis": "source_arbitration_disabled",
            }
        try:
            semantic_noun_id = int(semantic_noun_id) if semantic_noun_id is not None else None
        except Exception:
            semantic_noun_id = None
        detector_candidate = self.detector_grounding_noun_candidate(event, state, verb_name)
        detector_noun_id = _safe_int(detector_candidate.get("object_id"))
        detector_confidence = float((detector_candidate or {}).get("candidate_score", 0.0) or 0.0)
        grounding_candidate = self.best_grounding_candidate_for_noun(
            event,
            state,
            semantic_noun_id,
            verb_name,
        )
        semantic_confidence = float(semantic_confidence or 0.0)
        noun_required = bool(self.noun_required_for_verb(verb_name))

        def _policy_meta(
            *,
            source: str,
            reliability: float,
            risk: float,
            base_priority: float,
        ) -> Dict[str, Any]:
            return self._tsc_completion_decision(
                field_name="noun_object_id",
                source=str(source or "unknown"),
                query_type="complete_noun_object_id",
                runtime_reliability=float(reliability),
                overwrite_risk=float(risk),
                propagation_gain=0.74 if noun_required else 0.58,
                human_cost_est=0.42,
                base_priority=float(base_priority),
            )

        if semantic_noun_id is None and detector_noun_id is None:
            return {}
        if semantic_noun_id is None:
            policy = _policy_meta(
                source="hand_conditioned_noun_prior",
                reliability=float(detector_confidence),
                risk=0.30,
                base_priority=0.88,
            )
            return {
                "preferred_source": "hand_conditioned_noun_prior",
                "preferred_family": "detector_grounding",
                "semantic_noun_id": None,
                "semantic_confidence": float(semantic_confidence),
                "detector_noun_id": detector_noun_id,
                "detector_confidence": float(detector_confidence),
                "detector_candidate": dict(detector_candidate or {}),
                "grounding_candidate": dict(grounding_candidate or {}),
                "score_margin": float(detector_confidence),
                "decision_basis": "detector_fallback_no_semantic_noun",
                "apply_detector_override": bool(
                    _norm_text(policy.get("authority_level")).lower() == "safe_local"
                ),
                **dict(policy or {}),
            }
        if detector_noun_id is None:
            policy = _policy_meta(
                source="semantic_adapter_noun",
                reliability=float(semantic_confidence),
                risk=0.10,
                base_priority=0.80,
            )
            return {
                "preferred_source": "semantic_adapter_noun",
                "preferred_family": "semantic",
                "semantic_noun_id": int(semantic_noun_id),
                "semantic_confidence": float(semantic_confidence),
                "detector_noun_id": None,
                "detector_confidence": float(detector_confidence),
                "detector_candidate": {},
                "grounding_candidate": dict(grounding_candidate or {}),
                "score_margin": float(semantic_confidence),
                "decision_basis": "semantic_fallback_no_detector_candidate",
                "apply_detector_override": False,
                **dict(policy or {}),
            }

        if not self.variant.use_tsc:
            prefer_detector = float(detector_confidence) > float(semantic_confidence) + 1e-6
            comparison = {
                "field_name": "noun_object_id",
                "source_a": "semantic_adapter_noun",
                "source_a_family": "semantic",
                "source_a_acceptance": float(semantic_confidence),
                "source_a_support": 0,
                "source_a_runtime_confidence": float(semantic_confidence),
                "source_a_score": float(semantic_confidence),
                "source_b": "hand_conditioned_noun_prior",
                "source_b_family": "detector_grounding",
                "source_b_acceptance": float(detector_confidence),
                "source_b_support": 0,
                "source_b_runtime_confidence": float(detector_confidence),
                "source_b_score": float(detector_confidence),
                "preferred_side": "b" if prefer_detector else "a",
                "preferred_source": "hand_conditioned_noun_prior" if prefer_detector else "semantic_adapter_noun",
                "preferred_family": "detector_grounding" if prefer_detector else "semantic",
                "score_margin": float(abs(float(detector_confidence) - float(semantic_confidence))),
                "decision_basis": "confidence_only_arbitration",
            }
        else:
            comparison = self.calibrator.compare_field_sources(
                field_name="noun_object_id",
                source_a="semantic_adapter_noun",
                source_b="hand_conditioned_noun_prior",
                runtime_confidence_a=float(semantic_confidence),
                runtime_confidence_b=float(detector_confidence),
                action_kind="suggest",
                authority_level="human_confirm",
                interaction_form="accept_suggestion",
                query_type="suggest_noun_object_id",
            )
        preferred_side = _norm_text(comparison.get("preferred_side")).lower()
        preferred_source = _norm_text(comparison.get("preferred_source")).lower()
        preferred_family = _norm_text(comparison.get("preferred_family")).lower()
        source_a_score = float(comparison.get("source_a_score", semantic_confidence) or semantic_confidence)
        source_b_score = float(comparison.get("source_b_score", detector_confidence) or detector_confidence)
        preferred_score = source_a_score if preferred_side == "a" else source_b_score
        preferred_acceptance = (
            float(comparison.get("source_a_acceptance", semantic_confidence) or semantic_confidence)
            if preferred_side == "a"
            else float(comparison.get("source_b_acceptance", detector_confidence) or detector_confidence)
        )
        sources_agree = bool(
            semantic_noun_id is not None
            and detector_noun_id is not None
            and int(semantic_noun_id) == int(detector_noun_id)
        )
        if sources_agree:
            overwrite_risk = 0.06
        elif preferred_family == "detector_grounding":
            overwrite_risk = max(
                0.22,
                _bounded01(
                    0.34
                    - 0.18 * float(comparison.get("score_margin", 0.0) or 0.0)
                    - 0.10 * float(detector_confidence),
                    0.22,
                ),
            )
        else:
            overwrite_risk = max(
                0.10,
                _bounded01(
                    0.18
                    - 0.12 * float(comparison.get("score_margin", 0.0) or 0.0)
                    - 0.08 * float(semantic_confidence),
                    0.10,
                ),
            )
        policy = _policy_meta(
            source=str(preferred_source or "semantic_adapter_noun"),
            reliability=_bounded01(
                0.74 * float(preferred_score) + 0.26 * float(preferred_acceptance),
                preferred_score,
            ),
            risk=float(overwrite_risk),
            base_priority=0.90 if preferred_family == "detector_grounding" and not sources_agree else 0.82,
        )
        comparison["semantic_noun_id"] = int(semantic_noun_id)
        comparison["semantic_confidence"] = float(semantic_confidence)
        comparison["detector_noun_id"] = int(detector_noun_id)
        comparison["detector_confidence"] = float(detector_confidence)
        comparison["detector_candidate"] = dict(detector_candidate or {})
        comparison["grounding_candidate"] = dict(grounding_candidate or {})
        comparison["semantic_matches_detector"] = bool(int(semantic_noun_id) == int(detector_noun_id))
        comparison["apply_detector_override"] = bool(
            preferred_family == "detector_grounding"
            and _norm_text(policy.get("authority_level")).lower() == "safe_local"
        )
        comparison["preferred_interaction"] = str(policy.get("interaction_form") or "")
        comparison.update(dict(policy or {}))
        return comparison

    def handtrack_track(self, hand_key: str) -> Dict[str, Any]:
        if not self.variant.use_handtrack:
            return {}
        return dict((self.clip.handtracks.get("tracks") or {}).get(str(hand_key)) or {})

    def unguided_segment_candidate_scores(self, start_frame: int, end_frame: int) -> Dict[str, float]:
        start = _safe_int(start_frame)
        end = _safe_int(end_frame)
        if start is None or end is None:
            return {}
        if end < start:
            start, end = end, start
        key = (int(start), int(end))
        cached = self._unguided_candidate_cache.get(key)
        if cached is not None:
            return dict(cached)
        summary = aggregate_precomputed_feature_cache(
            self.clip.videomae_cache,
            start_frame=int(start),
            end_frame=int(end),
            onset_band=None,
            top_k=max(5, int(len(self.verb_labels) or 0)),
        )
        scores = {
            str(item.get("label") or ""): float(item.get("score") or 0.0)
            for item in list(summary.get("candidates") or [])
            if _norm_text(item.get("label"))
        }
        self._unguided_candidate_cache[key] = dict(scores)
        return dict(scores)

    def handtrack_segment_prior(self, hand_key: str, start_frame: int, end_frame: int) -> Dict[str, Any]:
        if not self.variant.use_handtrack:
            return {}
        start = _safe_int(start_frame)
        end = _safe_int(end_frame)
        if start is None or end is None:
            return {}
        if end < start:
            start, end = end, start
        track = self.handtrack_track(hand_key)
        frame_map = dict(track.get("frame_map") or {})
        if not frame_map:
            return {}
        segment_len = max(1, int(end) - int(start))
        rows = [
            dict(frame_map.get(int(frame)) or {})
            for frame in range(int(start), int(end) + 1)
            if int(frame) in frame_map
        ]
        if len(rows) < 3:
            return {}
        coverage = float(len(rows)) / float(max(1, segment_len + 1))
        handedness_purity = float(_segment_handedness_purity(rows, str(hand_key)))
        peak_rows = _local_peak_rows(rows)
        peak_row = max(peak_rows or rows, key=lambda row: float(row.get("motion", 0.0) or 0.0))
        peak_motion = float(peak_row.get("motion", 0.0) or 0.0)
        min_motion = float(min(_row_motion(row) for row in rows))
        avg_motion = float(
            sum(float(row.get("motion", 0.0) or 0.0) for row in rows)
            / float(max(1, len(rows)))
        )
        if peak_motion <= 1e-6:
            return {}
        dominance = peak_motion / float(max(1e-4, avg_motion))
        coarse_prior = _semantic_prior_from_scores(self.unguided_segment_candidate_scores(int(start), int(end)))
        family = str(coarse_prior.get("family") or "mid")
        template_ratio = float(coarse_prior.get("template_ratio", VERB_FAMILY_DEFAULT_RATIOS["mid"]) or VERB_FAMILY_DEFAULT_RATIOS["mid"])
        target_frame = int(round(int(start) + float(template_ratio) * float(segment_len)))
        target_frame = max(int(start), min(int(end), int(target_frame)))
        half = max(2, int(round(float(VERB_FAMILY_WINDOW_RATIOS.get(family, 0.22) or 0.22) * float(segment_len))))
        search_left = max(int(start), int(target_frame) - int(half))
        search_right = min(int(end), int(target_frame) + int(half))
        if family == "late":
            search_right = int(end)
        if family == "boundary":
            search_right = min(int(end), int(start) + max(3, int(round(0.14 * float(segment_len)))))
        search_rows = [
            dict(row or {})
            for row in rows
            if int(search_left) <= int(row.get("frame", target_frame) or target_frame) <= int(search_right)
        ] or [dict(row or {}) for row in rows]

        low_threshold = float(min_motion + 0.28 * max(0.0, float(peak_motion) - float(min_motion)))
        stable_min_run = max(2, int(round(0.08 * float(max(3, len(rows))))))
        stable_rows = _stable_run_starts(search_rows if family != "late" else rows, low_threshold=float(low_threshold), min_run=int(stable_min_run))

        phase_candidates: List[Dict[str, Any]] = []
        boundary_rows = [
            dict(row or {})
            for row in rows
            if int(row.get("frame", start) or start) <= min(int(end), int(start) + max(3, int(round(0.12 * float(segment_len)))))
        ]
        if boundary_rows:
            boundary_row = min(
                [dict(row or {}) for row in boundary_rows if not bool(row.get("interpolated", False))] or boundary_rows,
                key=lambda row: (
                    abs(int(row.get("frame", start) or start) - int(start)),
                    float(row.get("motion", 0.0) or 0.0),
                ),
            )
            phase_candidates.append({"phase_kind": "boundary_start", **dict(boundary_row or {})})
        peak_candidate = _best_peak_candidate(
            rows,
            target_frame=int(target_frame),
            search_left=int(search_left),
            search_right=int(search_right),
        )
        if peak_candidate:
            phase_candidates.append({"phase_kind": "approach_peak", **dict(peak_candidate or {})})
        valley_candidate = _best_template_candidate(
            rows,
            target_frame=int(target_frame),
            search_left=int(search_left),
            search_right=int(search_right),
        )
        if valley_candidate:
            phase_candidates.append({"phase_kind": "contact_valley", **dict(valley_candidate or {})})
        if stable_rows:
            stable_target = max(int(start), int(target_frame) - max(1, int(round(0.06 * float(segment_len)))))
            stable_row = min(
                stable_rows,
                key=lambda row: (
                    abs(int(row.get("frame", stable_target) or stable_target) - int(stable_target)),
                    float(row.get("run_mean_motion", row.get("motion", 0.0)) or 0.0),
                ),
            )
            phase_candidates.append({"phase_kind": "stabilization_start", **dict(stable_row or {})})
        if not phase_candidates:
            return {}

        motion_span = float(max(1e-6, float(peak_motion) - float(min_motion)))
        template_window = max(0.12, float(VERB_FAMILY_WINDOW_RATIOS.get(family, 0.22) or 0.22))
        score_rows: List[Dict[str, Any]] = []
        left_context = max(2, int(round(0.22 * float(segment_len))))
        right_context = max(2, int(round(0.18 * float(segment_len))))
        stable_ref = max(2, int(round(0.08 * float(segment_len))))
        for candidate in phase_candidates:
            frame = int(candidate.get("frame", target_frame) or target_frame)
            motion = float(candidate.get("motion", 0.0) or 0.0)
            motion_norm = _bounded01((float(motion) - float(min_motion)) / float(motion_span), 0.0)
            low_motion_score = 1.0 - float(motion_norm)
            template_score = _bounded01(
                1.0 - abs(int(frame) - int(target_frame)) / float(max(3, search_right - search_left + 1)),
                0.0,
            )
            start_score = _bounded01(
                1.0 - abs(int(frame) - int(start)) / float(max(3, segment_len)),
                0.0,
            )
            prev_peak_score = 0.0
            for row in peak_rows:
                peak_frame = int(row.get("frame", frame) or frame)
                if 0 <= int(frame) - int(peak_frame) <= int(left_context):
                    prev_peak_score = max(
                        float(prev_peak_score),
                        _bounded01((float(_row_motion(row)) - float(min_motion)) / float(motion_span), 0.0),
                    )
            post_low_rows = [
                dict(row or {})
                for row in rows
                if 0 <= int(row.get("frame", frame) or frame) - int(frame) <= int(right_context)
                and float(_row_motion(row)) <= float(low_threshold) + 1e-6
            ]
            post_stability = _bounded01(float(len(post_low_rows)) / float(max(1, stable_ref)), 0.0)
            run_score = _bounded01(float(candidate.get("run_length", 0) or 0) / float(max(1, stable_ref)), 0.0)
            phase_kind = str(candidate.get("phase_kind") or "contact_valley")
            phase_prior = _phase_prior_score(str(family), str(phase_kind))
            if phase_kind == "boundary_start":
                candidate_score = (
                    0.40 * float(phase_prior)
                    + 0.30 * float(start_score)
                    + 0.20 * float(post_stability)
                    + 0.10 * float(low_motion_score)
                )
            elif phase_kind == "approach_peak":
                candidate_score = (
                    0.34 * float(phase_prior)
                    + 0.26 * float(template_score)
                    + 0.22 * float(motion_norm)
                    + 0.18 * float(post_stability)
                )
            elif phase_kind == "stabilization_start":
                candidate_score = (
                    0.34 * float(phase_prior)
                    + 0.22 * float(template_score)
                    + 0.18 * float(low_motion_score)
                    + 0.16 * float(post_stability)
                    + 0.10 * float(run_score)
                )
            else:
                candidate_score = (
                    0.36 * float(phase_prior)
                    + 0.24 * float(template_score)
                    + 0.22 * float(low_motion_score)
                    + 0.18 * float(0.55 * float(prev_peak_score) + 0.45 * float(post_stability))
                )
            if bool(candidate.get("interpolated", False)):
                candidate_score -= 0.06
            score_rows.append(
                {
                    **dict(candidate or {}),
                    "phase_kind": str(phase_kind),
                    "phase_score": float(_bounded01(candidate_score, 0.0)),
                }
            )
        score_rows = sorted(score_rows, key=lambda row: float(row.get("phase_score", 0.0) or 0.0), reverse=True)
        chosen_row = dict((score_rows or [peak_row])[0] or {})
        candidate_kind = str(chosen_row.get("phase_kind") or "contact_valley")
        candidate_source = str(f"{family}_{candidate_kind}")
        onset_frame = int(chosen_row.get("frame", start) or start)
        onset_ratio = _bounded01((onset_frame - start) / float(max(1, segment_len)), 0.5)
        dominance_norm = _bounded01(dominance / 3.0, 0.0)
        template_agreement = _bounded01(
            1.0 - abs(float(onset_ratio) - float(template_ratio)) / float(max(1e-6, template_window)),
            0.0,
        )
        support_half = max(2, int(round(0.20 * float(segment_len))))
        chosen_motion = float(chosen_row.get("motion", 0.0) or 0.0)
        if candidate_kind == "approach_peak":
            support_cut = float(chosen_motion - 0.18 * max(0.0, float(chosen_motion) - float(min_motion)))
            support_frames = [
                int(row.get("frame", onset_frame) or onset_frame)
                for row in rows
                if abs(int(row.get("frame", onset_frame) or onset_frame) - int(onset_frame)) <= int(support_half)
                and float(row.get("motion", 0.0) or 0.0) >= float(support_cut) - 1e-6
            ]
        else:
            support_cut = float(chosen_motion + 0.18 * max(0.0, float(peak_motion) - float(chosen_motion)))
            support_frames = [
                int(row.get("frame", onset_frame) or onset_frame)
                for row in rows
                if abs(int(row.get("frame", onset_frame) or onset_frame) - int(onset_frame)) <= int(support_half)
                and float(row.get("motion", 0.0) or 0.0) <= float(support_cut) + 1e-6
                and (
                    candidate_kind != "stabilization_start"
                    or int(row.get("frame", onset_frame) or onset_frame) >= int(onset_frame)
                )
            ]
        support_ratio = _bounded01(float(len(support_frames)) / float(max(1, stable_ref)), 0.0)
        if len(score_rows) <= 1:
            margin_score = 1.0
        else:
            margin_score = _bounded01(
                (
                    float(score_rows[0].get("phase_score", 0.0) or 0.0)
                    - float(score_rows[1].get("phase_score", 0.0) or 0.0)
                ) / 0.20,
                0.0,
            )
        track_quality = _bounded01(
            0.35 * float(coverage)
            + 0.20 * float(coarse_prior.get("family_confidence", 0.0) or 0.0)
            + 0.20 * float(dominance_norm)
            + 0.25 * float(handedness_purity),
            0.0,
        )
        confidence = _bounded01(
            0.32 * float(track_quality)
            + 0.24 * float(chosen_row.get("phase_score", 0.0) or 0.0)
            + 0.16 * float(template_agreement)
            + 0.14 * float(margin_score)
            + 0.14 * float(support_ratio),
            0.0,
        )
        if bool(chosen_row.get("interpolated", False)):
            confidence *= 0.90
        if float(coverage) < 0.28:
            return {}
        if float(handedness_purity) < 0.70:
            return {}
        if float(confidence) < 0.58:
            return {}
        if float(margin_score) < 0.06 and float(confidence) < 0.70:
            return {}
        if int(len(support_frames)) < 3 and float(confidence) < 0.72:
            return {}
        onset_band = _build_hop_band(
            start=int(start),
            end=int(end),
            onset_frame=int(onset_frame),
            family=str(family),
            candidate_kind=str(candidate_kind),
            rows=rows,
            chosen_row=chosen_row,
            confidence=float(confidence),
        )
        return {
            "hand": str(hand_key),
            "source": "handtrack_semantic_phase_prior",
            "onset_frame": int(onset_frame),
            "onset_ratio": float(onset_ratio),
            "onset_band": dict(onset_band or {}),
            "band_width": float(
                float((onset_band or {}).get("right_ratio", onset_ratio) or onset_ratio)
                - float((onset_band or {}).get("left_ratio", onset_ratio) or onset_ratio)
            ),
            "confidence": float(confidence),
            "coverage": float(coverage),
            "motion_peak": float(peak_motion),
            "support_frame_count": int(len(support_frames)),
            "family": str(family),
            "template_ratio": float(template_ratio),
            "coarse_verb_label": str(coarse_prior.get("verb_label") or ""),
            "coarse_verb_score": float(coarse_prior.get("verb_score", 0.0) or 0.0),
            "candidate_source": str(candidate_source),
            "candidate_phase": str(candidate_kind),
            "candidate_score": float(chosen_row.get("phase_score", 0.0) or 0.0),
            "handedness_purity": float(handedness_purity),
            "phase_margin": float(margin_score),
        }

    def primary_onset_context(self, event: GTEvent, state: Dict[str, Any]) -> Dict[str, Any]:
        start = int(state.get("interaction_start"))
        end = int(state.get("interaction_end"))
        onset = _safe_int(state.get("functional_contact_onset"))
        onset_state = self.get_field_state(state, "functional_contact_onset")
        track_prior = self.handtrack_segment_prior(event.hand, start, end) if self.variant.use_hop else {}
        onset_status = str(onset_state.get("status") or "").strip().lower()
        onset_source = "event_state"
        onset_for_context = onset
        if onset_for_context is None and track_prior:
            onset_for_context = int(track_prior.get("onset_frame", start) or start)
            onset_status = "handtrack_prior"
            onset_source = str(track_prior.get("source") or "handtrack_once")
            band = dict(track_prior.get("onset_band") or {})
        else:
            band = build_onset_band(
                start,
                end,
                onset_frame=onset_for_context,
                onset_status=onset_state.get("status"),
            )
        return {
            "hand": event.hand,
            "start_frame": int(start),
            "end_frame": int(end),
            "onset_frame": onset_for_context,
            "onset_status": onset_status,
            "onset_band": band,
            "onset_source": onset_source,
            "handtrack_prior": dict(track_prior or {}),
        }

    def primary_local_onset_context(self, event: GTEvent, state: Dict[str, Any]) -> Dict[str, Any]:
        onset_context = self.primary_onset_context(event, state)
        return build_local_onset_window(
            state.get("interaction_start"),
            state.get("interaction_end"),
            onset_frame=onset_context.get("onset_frame"),
            onset_band=onset_context.get("onset_band"),
        )

    def precomputed_videomae_summary(self, event: GTEvent, state: Dict[str, Any]) -> Dict[str, Any]:
        onset_context = self.primary_onset_context(event, state)
        local_context = self.primary_local_onset_context(event, state)
        summary = aggregate_precomputed_feature_cache(
            self.clip.videomae_cache,
            start_frame=state.get("interaction_start"),
            end_frame=state.get("interaction_end"),
            onset_band=dict(onset_context.get("onset_band") or {}),
            top_k=5,
        )
        local_summary = {}
        if local_context:
            local_summary = aggregate_precomputed_feature_cache(
                self.clip.videomae_cache,
                start_frame=local_context.get("start_frame"),
                end_frame=local_context.get("end_frame"),
                onset_band=None,
                top_k=5,
            )
        return {
            "segment_feature": list(summary.get("segment_feature") or []),
            "candidates": [dict(row) for row in list(summary.get("candidates") or [])],
            "local_segment_feature": list(local_summary.get("segment_feature") or []),
            "local_candidates": [dict(row) for row in list(local_summary.get("candidates") or [])],
        }

    def primary_videomae_scores(self, summary: Dict[str, Any]) -> Dict[str, float]:
        global_scores = {
            str(item.get("label") or ""): float(item.get("score") or 0.0)
            for item in list(summary.get("candidates") or [])
        }
        local_scores = {
            str(item.get("label") or ""): float(item.get("score") or 0.0)
            for item in list(summary.get("local_candidates") or [])
        }
        merged: Dict[str, float] = {}
        for key in set(global_scores.keys()) | set(local_scores.keys()):
            merged[str(key)] = 0.65 * float(global_scores.get(key, 0.0)) + 0.35 * float(local_scores.get(key, 0.0))
        return merged

    def compute_sparse_evidence_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        expected = 0
        confirmed = 0
        missing = 0
        blocked = 0
        noun_id = _safe_int(state.get("noun_object_id"))
        for key in ("interaction_start", "functional_contact_onset", "interaction_end"):
            frame = _safe_int(state.get(key))
            if noun_id is None or frame is None:
                blocked += 1
                continue
            expected += 1
            rows = list(self.clip.bboxes_by_frame.get(int(frame), []) or []) if self.variant.use_oracle_grounding else []
            match = next((box for box in rows if _safe_int(box.get("id")) == int(noun_id)), None)
            if match is not None:
                confirmed += 1
            else:
                missing += 1
        return {
            "expected": int(expected),
            "confirmed": int(confirmed),
            "missing": int(missing),
            "blocked": int(blocked),
        }

    def box_center_xy(self, box: Optional[dict]) -> Tuple[float, float]:
        if not isinstance(box, dict):
            return 0.0, 0.0
        return (
            (float(box.get("x1", 0.0) or 0.0) + float(box.get("x2", 0.0) or 0.0)) * 0.5,
            (float(box.get("y1", 0.0) or 0.0) + float(box.get("y2", 0.0) or 0.0)) * 0.5,
        )

    def box_diag(self, box: Optional[dict]) -> float:
        if not isinstance(box, dict):
            return 0.0
        x1 = float(box.get("x1", 0.0) or 0.0)
        y1 = float(box.get("y1", 0.0) or 0.0)
        x2 = float(box.get("x2", 0.0) or 0.0)
        y2 = float(box.get("y2", 0.0) or 0.0)
        return float(max(0.0, math.hypot(x2 - x1, y2 - y1)))

    def box_iou(self, box_a: Optional[dict], box_b: Optional[dict]) -> float:
        if not isinstance(box_a, dict) or not isinstance(box_b, dict):
            return 0.0
        ax1, ay1, ax2, ay2 = [float(box_a.get(key, 0.0) or 0.0) for key in ("x1", "y1", "x2", "y2")]
        bx1, by1, bx2, by2 = [float(box_b.get(key, 0.0) or 0.0) for key in ("x1", "y1", "x2", "y2")]
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        if denom <= 1e-6:
            return 0.0
        return float(inter / denom)

    def box_edge_gap(self, box_a: Optional[dict], box_b: Optional[dict]) -> float:
        if not isinstance(box_a, dict) or not isinstance(box_b, dict):
            return float("inf")
        ax1, ay1, ax2, ay2 = [float(box_a.get(key, 0.0) or 0.0) for key in ("x1", "y1", "x2", "y2")]
        bx1, by1, bx2, by2 = [float(box_b.get(key, 0.0) or 0.0) for key in ("x1", "y1", "x2", "y2")]
        dx = max(0.0, ax1 - bx2, bx1 - ax2)
        dy = max(0.0, ay1 - by2, by1 - ay2)
        return float(math.hypot(dx, dy))

    def box_confidence_score(self, box: Optional[dict]) -> float:
        if not isinstance(box, dict):
            return 0.0
        conf = box.get("confidence")
        if conf is not None:
            return _bounded01(conf, 0.0)
        return 0.5

    def hand_object_proximity_score(self, hand_box: Optional[dict], object_box: Optional[dict]) -> float:
        if not isinstance(hand_box, dict) or not isinstance(object_box, dict):
            return 0.0
        frame_diag = float(max(1.0, math.hypot(self.clip.frame_width, self.clip.frame_height)))
        hand_diag = float(max(1.0, self.box_diag(hand_box)))
        object_diag = float(max(1.0, self.box_diag(object_box)))
        hx, hy = self.box_center_xy(hand_box)
        ox, oy = self.box_center_xy(object_box)
        center_dist = float(math.hypot(ox - hx, oy - hy))
        center_scale = max(24.0, 0.95 * hand_diag + 0.55 * object_diag)
        center_score = _bounded01(1.0 - center_dist / float(max(1.0, center_scale)), 0.0)
        gap = self.box_edge_gap(hand_box, object_box)
        gap_score = _bounded01(1.0 - gap / float(max(1.0, 0.18 * frame_diag)), 0.0)
        iou_score = _bounded01(self.box_iou(hand_box, object_box) * 3.0, 0.0)
        return _bounded01(max(gap_score, 0.60 * center_score + 0.40 * iou_score), 0.0)

    def frame_hand_box(self, hand_key: str, frame: int) -> Dict[str, Any]:
        if not self.variant.use_handtrack:
            return {}
        track = self.handtrack_track(str(hand_key))
        row = dict((track.get("frame_map") or {}).get(int(frame)) or {})
        bbox = list(row.get("bbox") or [])
        if len(bbox) < 4:
            return {}
        return {
            "id": str(hand_key),
            "label": str(hand_key),
            "source": "handtrack_once",
            "frame": int(frame),
            "x1": float(bbox[0]),
            "y1": float(bbox[1]),
            "x2": float(bbox[2]),
            "y2": float(bbox[3]),
            "confidence": float(row.get("detection_confidence", 0.0) or 0.0),
            "locked": True,
            "synthetic": True,
        }

    def event_object_candidate_frames(self, event: GTEvent, state: Dict[str, Any]) -> List[int]:
        start = _safe_int(state.get("interaction_start"))
        onset = _safe_int(state.get("functional_contact_onset"))
        end = _safe_int(state.get("interaction_end"))
        if start is None and onset is None and end is None:
            return []
        if start is not None and end is not None and end < start:
            start, end = end, start
        frames: List[int] = []

        def _add(frame_value: Optional[int]) -> None:
            if frame_value is None:
                return
            frame_int = _safe_int(frame_value)
            if frame_int is None:
                return
            if start is not None and frame_int < start:
                return
            if end is not None and frame_int > end:
                return
            if frame_int not in frames:
                frames.append(frame_int)

        for value in (start, onset, end):
            _add(value)
        if onset is not None:
            _add(onset - 1)
            _add(onset + 1)
        track = self.handtrack_track(str(event.hand))
        motion_peak_frame = _safe_int(track.get("motion_peak_frame"))
        _add(motion_peak_frame)
        if motion_peak_frame is not None:
            _add(motion_peak_frame - 1)
            _add(motion_peak_frame + 1)
        return frames

    def collect_event_object_candidates(self, event: GTEvent, state: Dict[str, Any]) -> List[dict]:
        if not self.variant.use_oracle_grounding:
            return []
        onset_frame = _safe_int(state.get("functional_contact_onset"))
        frames = self.event_object_candidate_frames(event, state)
        if not frames:
            return []
        track = self.handtrack_track(str(event.hand))
        frame_map = dict(track.get("frame_map") or {})
        motion_peak_frame = _safe_int(track.get("motion_peak_frame"))
        max_motion = max(
            [1e-6]
            + [float((frame_map.get(int(frame)) or {}).get("motion", 0.0) or 0.0) for frame in frames]
        )
        candidates: Dict[int, Dict[str, Any]] = {}
        for frame in frames:
            hand_box = self.frame_hand_box(str(event.hand), int(frame))
            track_row = dict(frame_map.get(int(frame)) or {})
            motion_value = float(track_row.get("motion", 0.0) or 0.0)
            motion_weight = _bounded01(motion_value / float(max_motion), 0.0)
            onset_weight = 0.0
            if onset_frame is not None:
                if int(frame) == int(onset_frame):
                    onset_weight = 1.0
                elif abs(int(frame) - int(onset_frame)) == 1:
                    onset_weight = 0.55
            peak_weight = 0.0
            if motion_peak_frame is not None:
                if int(frame) == int(motion_peak_frame):
                    peak_weight = 1.0
                elif abs(int(frame) - int(motion_peak_frame)) == 1:
                    peak_weight = 0.55
            for box in list(self.clip.bboxes_by_frame.get(int(frame), []) or []):
                object_id = _safe_int(box.get("id"))
                if object_id is None:
                    continue
                row = candidates.setdefault(
                    int(object_id),
                    {
                        "object_id": int(object_id),
                        "object_name": _object_name_for_id(self.clip.object_library, object_id) or f"Object {int(object_id)}",
                        "support": 0,
                        "onset_support": 0,
                        "frames": [],
                        "support_score": 0.0,
                        "onset_support_score": 0.0,
                        "motion_support_score": 0.0,
                        "hand_proximity_max": 0.0,
                        "yolo_confidence_max": 0.0,
                        "best_frame": None,
                        "best_bbox": {},
                        "best_frame_score": -1.0,
                        "_hand_proximity_sum": 0.0,
                        "_confidence_sum": 0.0,
                        "_frame_count": 0,
                    },
                )
                yolo_conf = self.box_confidence_score(box)
                proximity = self.hand_object_proximity_score(hand_box, box)
                motion_alignment = max(peak_weight, motion_weight) * max(0.15, proximity)
                support_score = 0.40 + 0.35 * proximity + 0.25 * yolo_conf
                onset_support_score = onset_weight * max(0.20, 0.60 * proximity + 0.40 * yolo_conf)
                frame_score = (
                    0.32 * yolo_conf
                    + 0.34 * proximity
                    + 0.20 * onset_weight
                    + 0.14 * max(motion_weight, peak_weight)
                )
                row["support"] = int(row.get("support", 0) or 0) + 1
                if int(frame) not in row["frames"]:
                    row["frames"].append(int(frame))
                if onset_weight >= 0.99:
                    row["onset_support"] = int(row.get("onset_support", 0) or 0) + 1
                row["support_score"] = float(row.get("support_score", 0.0) or 0.0) + float(support_score)
                row["onset_support_score"] = float(row.get("onset_support_score", 0.0) or 0.0) + float(onset_support_score)
                row["motion_support_score"] = float(row.get("motion_support_score", 0.0) or 0.0) + float(motion_alignment)
                row["hand_proximity_max"] = max(float(row.get("hand_proximity_max", 0.0) or 0.0), float(proximity))
                row["yolo_confidence_max"] = max(float(row.get("yolo_confidence_max", 0.0) or 0.0), float(yolo_conf))
                row["_hand_proximity_sum"] = float(row.get("_hand_proximity_sum", 0.0) or 0.0) + float(proximity)
                row["_confidence_sum"] = float(row.get("_confidence_sum", 0.0) or 0.0) + float(yolo_conf)
                row["_frame_count"] = int(row.get("_frame_count", 0) or 0) + 1
                if float(frame_score) >= float(row.get("best_frame_score", -1.0) or -1.0):
                    row["best_frame_score"] = float(frame_score)
                    row["best_frame"] = int(frame)
                    row["best_bbox"] = {
                        "x1": float(box.get("x1", 0.0) or 0.0),
                        "y1": float(box.get("y1", 0.0) or 0.0),
                        "x2": float(box.get("x2", 0.0) or 0.0),
                        "y2": float(box.get("y2", 0.0) or 0.0),
                    }
        rows = [dict(item) for item in candidates.values()]
        if not rows:
            return []
        max_support_score = max([1e-6] + [float(item.get("support_score", 0.0) or 0.0) for item in rows])
        max_onset_support_score = max([1e-6] + [float(item.get("onset_support_score", 0.0) or 0.0) for item in rows])
        max_motion_support_score = max([1e-6] + [float(item.get("motion_support_score", 0.0) or 0.0) for item in rows])
        cleaned_rows: List[dict] = []
        for item in rows:
            frame_count = max(1, int(item.pop("_frame_count", 0) or 0))
            hand_proximity_mean = float(item.pop("_hand_proximity_sum", 0.0) or 0.0) / float(frame_count)
            yolo_confidence_mean = float(item.pop("_confidence_sum", 0.0) or 0.0) / float(frame_count)
            support_norm = _bounded01(float(item.get("support_score", 0.0) or 0.0) / float(max_support_score), 0.0)
            onset_norm = _bounded01(float(item.get("onset_support_score", 0.0) or 0.0) / float(max_onset_support_score), 0.0)
            motion_norm = _bounded01(float(item.get("motion_support_score", 0.0) or 0.0) / float(max_motion_support_score), 0.0)
            item["hand_proximity_mean"] = float(hand_proximity_mean)
            item["yolo_confidence_mean"] = float(yolo_confidence_mean)
            item["hand_conditioned"] = bool(float(item.get("hand_proximity_max", 0.0) or 0.0) > 0.0 or motion_norm > 0.0)
            item["candidate_score"] = _bounded01(
                0.24 * float(item.get("yolo_confidence_max", 0.0) or 0.0)
                + 0.22 * float(yolo_confidence_mean)
                + 0.24 * float(item.get("hand_proximity_max", 0.0) or 0.0)
                + 0.16 * float(onset_norm)
                + 0.08 * float(support_norm)
                + 0.06 * float(motion_norm),
                0.0,
            )
            cleaned_rows.append(item)
        cleaned_rows.sort(
            key=lambda item: (
                float(item.get("candidate_score", 0.0) or 0.0),
                float(item.get("onset_support_score", 0.0) or 0.0),
                float(item.get("support_score", 0.0) or 0.0),
                -int(item.get("object_id", 0) or 0),
            ),
            reverse=True,
        )
        for idx, item in enumerate(cleaned_rows):
            next_score = (
                float(cleaned_rows[idx + 1].get("candidate_score", 0.0) or 0.0)
                if idx + 1 < len(cleaned_rows)
                else 0.0
            )
            item["candidate_rank"] = int(idx + 1)
            item["candidate_gap"] = float(max(0.0, float(item.get("candidate_score", 0.0) or 0.0) - next_score))
        return cleaned_rows

    def semantic_refine_feature(
        self,
        feature: Sequence[float],
        *,
        refined_onset_ratio: Optional[float] = None,
        refined_verb: str = "",
        refined_noun_exists: Optional[bool] = None,
        refined_noun_object_id: Optional[int] = None,
    ) -> List[float]:
        values = [float(v) for v in list(feature or [])]
        if len(values) < 10:
            return values
        if refined_onset_ratio is not None:
            values[0] = _bounded01(refined_onset_ratio, 0.5)
            values[1] = 1.0
            if values[2] < 0.5:
                values[3] = 1.0
        refined_verb = _norm_text(refined_verb)
        if refined_verb and refined_verb in self.verb_labels:
            values[7] = 1.0 if self.noun_required_for_verb(refined_verb) else 0.0
            values[9] = float(self.noun_exists_prior_for_verb(refined_verb))
            verb_offset = 10
            verb_pos = verb_offset + self.verb_labels.index(refined_verb)
            if 0 <= verb_pos < len(values):
                values[verb_pos] = max(float(values[verb_pos]), 0.85)
            local_verb_offset = verb_offset + len(self.verb_labels)
            local_verb_pos = local_verb_offset + self.verb_labels.index(refined_verb)
            if 0 <= local_verb_pos < len(values):
                values[local_verb_pos] = max(float(values[local_verb_pos]), 0.90)
        if refined_noun_exists is not None:
            values[8] = 1.0 if bool(refined_noun_exists) else 0.0
        noun_id = _safe_int(refined_noun_object_id)
        if noun_id is not None and noun_id in self.noun_ids:
            noun_index = self.noun_ids.index(noun_id)
            support_offset = int(self.feature_layout.get("noun_support_offset", 0) or 0)
            support_dim = int(self.feature_layout.get("noun_support_dim", 0) or 0)
            onset_support_offset = int(self.feature_layout.get("noun_onset_support_offset", 0) or 0)
            onset_support_dim = int(self.feature_layout.get("noun_onset_support_dim", 0) or 0)
            support_pos = support_offset + noun_index
            onset_support_pos = onset_support_offset + noun_index
            if 0 <= support_pos < min(len(values), support_offset + support_dim):
                values[support_pos] = max(float(values[support_pos]), 0.88)
            if 0 <= onset_support_pos < min(len(values), onset_support_offset + onset_support_dim):
                values[onset_support_pos] = max(float(values[onset_support_pos]), 0.94)
        return values

    def semantic_feature_sample(self, event: GTEvent, state: Dict[str, Any]) -> Dict[str, Any]:
        start = int(state.get("interaction_start"))
        end = int(state.get("interaction_end"))
        if end < start:
            start, end = end, start
        segment_len = max(1, end - start)
        onset = _safe_int(state.get("functional_contact_onset"))
        onset_ratio = 0.5
        has_onset = 0.0
        if onset is not None:
            onset_ratio = _bounded01((int(onset) - start) / float(segment_len), 0.5)
            has_onset = 1.0
        onset_state = self.get_field_state(state, "functional_contact_onset")
        sparse_summary = self.compute_sparse_evidence_summary(state)

        # Mirror the runtime onset-context path. When onset is still unresolved
        # and a handtrack prior exists, that prior should condition the onset
        # band and onset-local VideoMAE sampling. No separate handtrack scalar
        # side-channel is injected into the adapter input. The editable state
        # features below still reflect the actual event state and are not
        # overwritten by handtrack.
        video_summary = self.precomputed_videomae_summary(event, state)

        candidate_scores = self.primary_videomae_scores(video_summary)
        local_candidate_scores = {
            str(item.get("label") or ""): float(item.get("score") or 0.0)
            for item in list(video_summary.get("local_candidates") or [])
        }
        support_map: Dict[int, float] = {}
        onset_support_map: Dict[int, float] = {}
        if self.variant.use_oracle_grounding:
            object_candidates = self.collect_event_object_candidates(event, state)
            support_map = {
                int(item.get("object_id")): float(item.get("support_score", item.get("support", 0.0)) or 0.0)
                for item in object_candidates
                if item.get("object_id") is not None
            }
            onset_support_map = {
                int(item.get("object_id")): float(item.get("onset_support_score", item.get("onset_support", 0.0)) or 0.0)
                for item in object_candidates
                if item.get("object_id") is not None
            }
        max_support = max([1.0] + list(support_map.values()))
        max_onset_support = max([1.0] + list(onset_support_map.values()))
        noun_id = _safe_int(state.get("noun_object_id"))

        feature: List[float] = [
            _bounded01(onset_ratio, 0.5),
            has_onset,
            1.0 if str(onset_state.get("status") or "").strip().lower() == "confirmed" else 0.0,
            1.0 if str(onset_state.get("status") or "").strip().lower() == "suggested" else 0.0,
            _bounded01(segment_len / 240.0),
            _bounded01(float(sparse_summary.get("confirmed", 0) or 0) / float(max(1, sparse_summary.get("expected", 0) or 0))),
            _bounded01(float(sparse_summary.get("missing", 0) or 0) / float(max(1, sparse_summary.get("expected", 0) or 0))),
            1.0 if self.noun_required_for_verb(state.get("verb")) else 0.0,
            1.0 if noun_id is not None else 0.0,
            float(self.noun_exists_prior_for_verb(state.get("verb"))),
        ]
        feature.extend(float(candidate_scores.get(label, 0.0)) for label in self.verb_labels)
        feature.extend(float(local_candidate_scores.get(label, 0.0)) for label in self.verb_labels)
        feature.extend(float(support_map.get(noun_idx, 0.0)) / float(max_support) for noun_idx in self.noun_ids)
        feature.extend(float(onset_support_map.get(noun_idx, 0.0)) / float(max_onset_support) for noun_idx in self.noun_ids)
        feature.extend([float(v) for v in list(video_summary.get("segment_feature") or [])])
        feature.extend([float(v) for v in list(video_summary.get("local_segment_feature") or [])])
        if len(feature) != int(getattr(self.package, "feature_dim", 0) or 0):
            raise RuntimeError(
                f"Feature dim mismatch for {event.clip_id}/{event.event_id}: "
                f"expected {getattr(self.package, 'feature_dim', 0)}, got {len(feature)}"
            )
        return {
            "feature": [float(v) for v in feature],
            "support_map": support_map,
            "onset_support_map": onset_support_map,
            "videomae_scores": candidate_scores,
        }

    def runtime_constraints(self, state: Dict[str, Any]) -> Dict[str, Any]:
        constraints = {
            "clamp_onset_ratio": None,
            "clamp_verb_label": "",
            "clamp_noun_exists": None,
            "clamp_noun_object_id": None,
            "clamp_unknown_noun": False,
            "locked_fields": [],
        }
        if not self.variant.use_lock_aware:
            return constraints

        start = _safe_int(state.get("interaction_start"))
        end = _safe_int(state.get("interaction_end"))
        if start is not None and end is not None and end < start:
            start, end = end, start
        segment_len = max(1, int(end) - int(start)) if start is not None and end is not None else 1
        onset = _safe_int(state.get("functional_contact_onset"))
        onset_state = self.get_field_state(state, "functional_contact_onset")
        onset_source = _norm_text(onset_state.get("source"))
        onset_locked = _norm_text(onset_state.get("status")).lower() == "confirmed" or self.semantic_source_is_runtime_lock(onset_source)
        if onset is not None and start is not None and onset_locked:
            constraints["clamp_onset_ratio"] = _bounded01((int(onset) - int(start)) / float(segment_len), 0.5)
            constraints["locked_fields"].append("onset")
        verb_name = _norm_text(state.get("verb"))
        verb_state = self.get_field_state(state, "verb")
        verb_source = _norm_text(verb_state.get("source"))
        verb_locked = _norm_text(verb_state.get("status")).lower() == "confirmed" or self.semantic_source_is_runtime_lock(verb_source)
        if verb_name and verb_locked:
            constraints["clamp_verb_label"] = verb_name
            constraints["locked_fields"].append("verb")
        noun_value = _safe_int(state.get("noun_object_id"))
        noun_state = self.get_field_state(state, "noun_object_id")
        noun_source = _norm_text(noun_state.get("source"))
        noun_locked = _norm_text(noun_state.get("status")).lower() == "confirmed" or self.semantic_source_is_runtime_lock(noun_source)
        if noun_value is not None and noun_locked:
            constraints["clamp_noun_exists"] = True
            constraints["clamp_noun_object_id"] = int(noun_value)
            constraints["locked_fields"].append("noun")
        elif noun_value is None and noun_locked and self.ontology is not None and verb_name and self.ontology.allow_no_noun(verb_name):
            constraints["clamp_noun_exists"] = False
            constraints["locked_fields"].append("noun_absent")
        return constraints

    def fuse_onset_with_handtrack(
        self,
        *,
        state: Dict[str, Any],
        prediction: Dict[str, Any],
        track_prior: Dict[str, Any],
        runtime_constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        structured = dict((prediction.get("structured") or {}).get("best") or {})
        structured_joint_conf = float(structured.get("joint_prob", 0.0) or 0.0)
        structured_onset_ratio = float(
            structured.get("onset_ratio", prediction.get("onset_ratio", 0.5)) or 0.5
        )
        structured_band = dict(structured.get("onset_band") or prediction.get("onset_band") or {})
        structured_band_width = float(
            structured.get("band_width")
            or (
                float(structured_band.get("right_ratio", 0.0) or 0.0)
                - float(structured_band.get("left_ratio", 0.0) or 0.0)
            )
            or 0.0
        )
        handtrack_conf = float(track_prior.get("confidence", 0.0) or 0.0)
        fused_onset_ratio = float(structured_onset_ratio)
        fused_onset_band = dict(structured_band)
        fused_band_width = float(structured_band_width)
        if (
            self.enable_handtrack_fusion
            and track_prior
            and runtime_constraints.get("clamp_onset_ratio") is None
        ):
            sem_weight = max(
                1e-6,
                0.55 * float(prediction.get("onset_confidence", 0.0) or 0.0)
                + 0.45 * structured_joint_conf,
            )
            track_weight = max(1e-6, handtrack_conf)
            total_weight = sem_weight + track_weight
            fused_onset_ratio = _bounded01(
                (
                    sem_weight * float(structured_onset_ratio)
                    + track_weight
                    * float(track_prior.get("onset_ratio", structured_onset_ratio) or structured_onset_ratio)
                )
                / float(max(1e-6, total_weight)),
                structured_onset_ratio,
            )
            track_band = dict(track_prior.get("onset_band") or {})
            if track_band:
                fused_onset_band = {
                    "center_ratio": float(fused_onset_ratio),
                    "left_ratio": _bounded01(
                        (
                            sem_weight
                            * float(structured_band.get("left_ratio", fused_onset_ratio) or fused_onset_ratio)
                            + track_weight
                            * float(track_band.get("left_ratio", fused_onset_ratio) or fused_onset_ratio)
                        )
                        / float(max(1e-6, total_weight)),
                        fused_onset_ratio,
                    ),
                    "right_ratio": _bounded01(
                        (
                            sem_weight
                            * float(structured_band.get("right_ratio", fused_onset_ratio) or fused_onset_ratio)
                            + track_weight
                            * float(track_band.get("right_ratio", fused_onset_ratio) or fused_onset_ratio)
                        )
                        / float(max(1e-6, total_weight)),
                        fused_onset_ratio,
                    ),
                }
                fused_onset_band["left_ratio"] = min(
                    float(fused_onset_band["left_ratio"]),
                    float(fused_onset_ratio),
                )
                fused_onset_band["right_ratio"] = max(
                    float(fused_onset_band["right_ratio"]),
                    float(fused_onset_ratio),
                )
                fused_band_width = float(
                    fused_onset_band["right_ratio"] - fused_onset_band["left_ratio"]
                )
        return {
            "ratio": float(fused_onset_ratio),
            "band": dict(fused_onset_band),
            "band_width": float(fused_band_width),
            "handtrack_confidence": float(handtrack_conf),
            "structured_joint_confidence": float(structured_joint_conf),
        }

    def resolve_prediction_output(
        self,
        *,
        event: GTEvent,
        state: Dict[str, Any],
        prediction: Dict[str, Any],
        track_prior: Dict[str, Any],
        runtime_constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        structured = dict((prediction.get("structured") or {}).get("best") or {})
        onset_meta = self.fuse_onset_with_handtrack(
            state=state,
            prediction=prediction,
            track_prior=track_prior,
            runtime_constraints=runtime_constraints,
        )
        resolved_verb = _norm_text(structured.get("verb_label")) or _norm_text(
            ((prediction.get("verb_candidates") or [{}])[0]).get("label")
        )
        resolved_noun_exists = bool(structured.get("noun_exists", False))
        resolved_noun_unknown = bool(structured.get("noun_is_unknown"))
        resolved_noun_id = (
            _safe_int(structured.get("noun_object_id"))
            if resolved_noun_exists and not resolved_noun_unknown
            else None
        )
        noun_source_decision: Dict[str, Any] = {}
        detector_candidate: Dict[str, Any] = {}
        detector_preferred = False
        if self.variant.use_oracle_grounding:
            noun_source_decision = self.estimate_noun_source_decision(
                event,
                state,
                verb_name=resolved_verb or state.get("verb"),
                semantic_noun_id=resolved_noun_id,
                semantic_confidence=float(structured.get("joint_prob", 0.0) or 0.0),
            )
            detector_candidate = dict(noun_source_decision.get("detector_candidate") or {})
            detector_preferred = (
                self.variant.use_source_arbitration
                and _norm_text(noun_source_decision.get("preferred_family")).lower() == "detector_grounding"
            )
            detector_auto_apply = bool(noun_source_decision.get("apply_detector_override"))
            if detector_auto_apply:
                detector_noun_id = _safe_int(noun_source_decision.get("detector_noun_id"))
                if detector_noun_id is not None:
                    resolved_noun_id = int(detector_noun_id)
                    resolved_noun_exists = True
                    resolved_noun_unknown = False
        else:
            detector_auto_apply = False
        return {
            "resolved_onset_ratio": float(onset_meta.get("ratio", 0.5) or 0.5),
            "resolved_onset_band": dict(onset_meta.get("band") or {}),
            "resolved_onset_band_width": float(onset_meta.get("band_width", 0.0) or 0.0),
            "resolved_verb": str(resolved_verb or ""),
            "resolved_noun_id": None if resolved_noun_id is None else int(resolved_noun_id),
            "resolved_noun_exists": bool(resolved_noun_exists),
            "resolved_noun_unknown": bool(resolved_noun_unknown),
            "noun_source_decision": dict(noun_source_decision or {}),
            "detector_candidate": dict(detector_candidate or {}),
            "detector_preferred": bool(detector_preferred),
            "detector_auto_apply": bool(detector_auto_apply),
            "structured_joint_confidence": float(onset_meta.get("structured_joint_confidence", 0.0) or 0.0),
            "handtrack_confidence": float(onset_meta.get("handtrack_confidence", 0.0) or 0.0),
        }

    def decode(self, event: GTEvent, state: Dict[str, Any]) -> Dict[str, Any]:
        sample = self.semantic_feature_sample(event, state)
        current_verb = _norm_text(state.get("verb"))
        allowed_noun_ids = self.allowed_noun_ids_for_verb(current_verb)
        allowed_nouns_by_verb = {str(label): list(self.allowed_noun_ids_for_verb(label)) for label in self.verb_labels}
        allow_no_noun_by_verb = {str(label): (not self.noun_required_for_verb(label)) for label in self.verb_labels}
        constraints = self.runtime_constraints(state)
        track_prior = (
            self.handtrack_segment_prior(
                event.hand,
                _safe_int(state.get("interaction_start")) or event.start,
                _safe_int(state.get("interaction_end")) or event.end,
            )
            if self.variant.use_hop
            else {}
        )
        reinfer_hint = self.semantic_reinfer_hint(state)
        anchor_onset_ratio = reinfer_hint.get("onset_anchor_ratio")
        anchor_onset_half_width = reinfer_hint.get("onset_anchor_half_width")
        anchor_onset_weight = (
            0.34 if anchor_onset_ratio is not None and constraints.get("clamp_onset_ratio") is None else 0.0
        )
        cross_hand_context = self.cross_hand_context(event, state)
        primary_exclusion = dict(cross_hand_context.get("primary_exclusion") or {})
        exclude_onset_ratio = primary_exclusion.get("onset_ratio")
        exclude_onset_half_width = 0.08 if primary_exclusion else None
        exclude_onset_weight = float(primary_exclusion.get("exclude_weight", 0.0) or 0.0)
        result = run_event_local_semantic_decode(
            SemanticRuntimeRequest(
                feature=list(sample.get("feature") or []),
                package=self.package,
                allowed_noun_ids=allowed_noun_ids,
                noun_required=self.noun_required_for_verb(current_verb),
                allow_no_noun=(not self.noun_required_for_verb(current_verb)),
                external_verb_scores=dict(sample.get("videomae_scores") or {}),
                noun_support_scores=dict(sample.get("support_map") or {}),
                noun_onset_support_scores=dict(sample.get("onset_support_map") or {}),
                allowed_nouns_by_verb=allowed_nouns_by_verb,
                allow_no_noun_by_verb=allow_no_noun_by_verb,
                clamp_onset_ratio=constraints.get("clamp_onset_ratio"),
                clamp_verb_label=constraints.get("clamp_verb_label") or "",
                clamp_noun_exists=constraints.get("clamp_noun_exists"),
                clamp_noun_object_id=constraints.get("clamp_noun_object_id"),
                clamp_unknown_noun=bool(constraints.get("clamp_unknown_noun")),
                anchor_onset_ratio=anchor_onset_ratio,
                anchor_onset_half_width=anchor_onset_half_width,
                anchor_onset_weight=anchor_onset_weight,
                exclude_onset_ratio=exclude_onset_ratio,
                exclude_onset_half_width=exclude_onset_half_width,
                exclude_onset_weight=exclude_onset_weight,
                refinement_passes=self.refinement_passes,
                refine_feature_fn=self.semantic_refine_feature,
            )
        )
        prediction = dict(result.prediction or {})
        resolved = self.resolve_prediction_output(
            event=event,
            state=state,
            prediction=prediction,
            track_prior=track_prior,
            runtime_constraints=constraints,
        )
        return {
            "prediction": prediction,
            "pass_trace": [dict(row) for row in list(result.pass_trace or [])],
            "feature": [float(v) for v in list(result.feature or [])],
            "track_prior": dict(track_prior or {}),
            "cross_hand_context": dict(cross_hand_context or {}),
            "runtime_constraints": dict(constraints or {}),
            **dict(resolved or {}),
        }

    def onset_frame_from_prediction(self, event: GTEvent, prediction: Dict[str, Any]) -> Optional[int]:
        best = dict((prediction.get("structured") or {}).get("best") or {})
        onset_ratio = best.get("onset_ratio")
        if onset_ratio is None:
            onset_ratio = prediction.get("onset_ratio")
        if onset_ratio is None:
            return None
        start = int(event.start)
        end = int(event.end)
        segment_len = max(1, end - start)
        frame = int(round(start + float(onset_ratio) * float(segment_len)))
        return max(start, min(end, frame))


def _candidate_rank(rows: Sequence[Dict[str, Any]], target: Any, *, key: str) -> Optional[int]:
    for idx, row in enumerate(list(rows or []), start=1):
        if key == "label":
            if _norm_text(row.get(key)) == _norm_text(target):
                return int(idx)
        else:
            row_value = _safe_int(row.get(key))
            target_value = _safe_int(target)
            if row_value is not None and target_value is not None and int(row_value) == int(target_value):
                return int(idx)
    return None


def _object_candidate_rank(rows: Sequence[Dict[str, Any]], target_object_id: Any) -> Optional[int]:
    target_value = _safe_int(target_object_id)
    if target_value is None:
        return None
    for idx, row in enumerate(list(rows or []), start=1):
        if _safe_int((row or {}).get("object_id")) == int(target_value):
            return int(idx)
    return None


def _best_verb(prediction: Dict[str, Any]) -> str:
    best = dict((prediction.get("structured") or {}).get("best") or {})
    return _norm_text(best.get("verb_label")) or _norm_text((prediction.get("verb_candidates") or [{}])[0].get("label"))


def _best_noun_id(prediction: Dict[str, Any]) -> Optional[int]:
    best = dict((prediction.get("structured") or {}).get("best") or {})
    if bool(best.get("noun_exists")):
        return _safe_int(best.get("noun_object_id"))
    return None


def _best_noun_exists(prediction: Dict[str, Any]) -> bool:
    best = dict((prediction.get("structured") or {}).get("best") or {})
    return bool(best.get("noun_exists"))


def _event_record(
    *,
    runtime: OfflineHOIRuntime,
    event: GTEvent,
    variant: VariantSpec,
) -> Dict[str, Any]:
    initial_state = _make_state(event)

    # Handtrack does not directly overwrite editable state in the offline
    # protocol, but it can condition onset-band temporal sampling and onset
    # fusion while the onset field is still unresolved.

    one_shot = runtime.decode(event, initial_state)
    one_pred = dict(one_shot.get("prediction") or {})
    one_track_prior = dict(one_shot.get("track_prior") or {})
    one_onset_ratio = one_shot.get("resolved_onset_ratio")
    if one_onset_ratio is None:
        one_onset_frame = runtime.onset_frame_from_prediction(event, one_pred)
    else:
        segment_len = max(1, int(event.end) - int(event.start))
        one_onset_frame = int(round(int(event.start) + float(one_onset_ratio) * float(segment_len)))
        one_onset_frame = max(int(event.start), min(int(event.end), int(one_onset_frame)))
    one_verb = _norm_text(one_shot.get("resolved_verb")) or _best_verb(one_pred)
    one_noun_id = _safe_int(one_shot.get("resolved_noun_id"))
    one_noun_exists = bool(one_shot.get("resolved_noun_exists")) if "resolved_noun_exists" in one_shot else _best_noun_exists(one_pred)
    one_noun_source_decision = dict(one_shot.get("noun_source_decision") or {})
    one_detector_candidate = dict(one_shot.get("detector_candidate") or {})

    state_onset = _set_confirmed_field(initial_state, "functional_contact_onset", int(event.onset))
    after_onset = runtime.decode(event, state_onset)
    after_onset_pred = dict(after_onset.get("prediction") or {})
    stage_verb = _norm_text(after_onset.get("resolved_verb")) or _best_verb(after_onset_pred)

    state_verb = _set_confirmed_field(state_onset, "verb", str(event.verb))
    after_verb = runtime.decode(event, state_verb)
    after_verb_pred = dict(after_verb.get("prediction") or {})
    stage_noun_id = _safe_int(after_verb.get("resolved_noun_id"))
    stage_noun_exists = bool(after_verb.get("resolved_noun_exists")) if "resolved_noun_exists" in after_verb else _best_noun_exists(after_verb_pred)
    stage_noun_source_decision = dict(after_verb.get("noun_source_decision") or {})
    stage_detector_candidate = dict(after_verb.get("detector_candidate") or {})

    onset_err = abs(int(one_onset_frame) - int(event.onset)) if one_onset_frame is not None else None
    onset_band = dict(one_shot.get("resolved_onset_band") or one_pred.get("onset_band") or {})
    band_left_ratio = onset_band.get("left_ratio")
    band_right_ratio = onset_band.get("right_ratio")
    segment_len = max(1, int(event.end) - int(event.start))
    gt_onset_ratio = float(int(event.onset) - int(event.start)) / float(segment_len)
    onset_band_covered = False
    onset_band_width_frames = None
    if band_left_ratio is not None and band_right_ratio is not None:
        onset_band_covered = float(band_left_ratio) <= float(gt_onset_ratio) <= float(band_right_ratio)
        onset_band_width_frames = float(band_right_ratio) - float(band_left_ratio)
        onset_band_width_frames = float(onset_band_width_frames) * float(segment_len)

    onset_strict = onset_err == 0 if onset_err is not None else False
    onset_tol2 = onset_err is not None and int(onset_err) <= 2
    onset_tol5 = onset_err is not None and int(onset_err) <= 5

    verb_one_rank = _candidate_rank(one_pred.get("verb_candidates") or [], event.verb, key="label")
    verb_stage_rank = _candidate_rank(after_onset_pred.get("verb_candidates") or [], event.verb, key="label")
    one_detector_auto_apply = bool(one_shot.get("detector_auto_apply"))
    stage_detector_auto_apply = bool(after_verb.get("detector_auto_apply"))
    handtrack_prior_onset_frame = _safe_int(one_track_prior.get("onset_frame"))
    handtrack_prior_onset_err = (
        abs(int(handtrack_prior_onset_frame) - int(event.onset))
        if handtrack_prior_onset_frame is not None
        else None
    )
    noun_one_rank = (
        _object_candidate_rank([one_detector_candidate], event.noun_id)
        if one_detector_auto_apply and one_detector_candidate
        else _candidate_rank(one_pred.get("noun_candidates") or [], event.noun_id, key="object_id")
    )
    noun_stage_rank = (
        _object_candidate_rank([stage_detector_candidate], event.noun_id)
        if stage_detector_auto_apply and stage_detector_candidate
        else _candidate_rank(after_verb_pred.get("noun_candidates") or [], event.noun_id, key="object_id")
    )

    verb_one_correct = _norm_text(one_verb) == _norm_text(event.verb)
    verb_stage_correct = _norm_text(stage_verb) == _norm_text(event.verb)
    noun_one_correct = (_safe_int(one_noun_id) == _safe_int(event.noun_id)) and bool(one_noun_exists)
    noun_stage_correct = (_safe_int(stage_noun_id) == _safe_int(event.noun_id)) and bool(stage_noun_exists)

    corrections_strict = int(not onset_strict) + int(not verb_stage_correct) + int(not noun_stage_correct)
    corrections_tol2 = int(not onset_tol2) + int(not verb_stage_correct) + int(not noun_stage_correct)
    corrections_tol5 = int(not onset_tol5) + int(not verb_stage_correct) + int(not noun_stage_correct)

    return {
        "variant": variant.name,
        "clip_id": event.clip_id,
        "hand": event.hand,
        "event_id": event.event_id,
        "start_frame": int(event.start),
        "gt_onset_frame": int(event.onset),
        "end_frame": int(event.end),
        "gt_verb": str(event.verb),
        "gt_noun_id": None if event.noun_id is None else int(event.noun_id),
        "gt_noun_label": str(event.noun_label),
        "one_shot_onset_frame": None if one_onset_frame is None else int(one_onset_frame),
        "one_shot_onset_abs_err": None if onset_err is None else int(onset_err),
        "one_shot_onset_exact": bool(onset_strict),
        "one_shot_onset_tol2": bool(onset_tol2),
        "one_shot_onset_tol5": bool(onset_tol5),
        "one_shot_onset_band_covered": bool(onset_band_covered),
        "one_shot_onset_band_width_frames": None if onset_band_width_frames is None else float(onset_band_width_frames),
        "handtrack_prior_available": bool(handtrack_prior_onset_frame is not None),
        "handtrack_prior_onset_frame": None if handtrack_prior_onset_frame is None else int(handtrack_prior_onset_frame),
        "handtrack_prior_onset_abs_err": None if handtrack_prior_onset_err is None else int(handtrack_prior_onset_err),
        "handtrack_prior_onset_tol2": bool(handtrack_prior_onset_err is not None and int(handtrack_prior_onset_err) <= 2),
        "handtrack_prior_onset_tol5": bool(handtrack_prior_onset_err is not None and int(handtrack_prior_onset_err) <= 5),
        "handtrack_prior_confidence": float(one_track_prior.get("confidence", 0.0) or 0.0),
        "handtrack_prior_coverage": float(one_track_prior.get("coverage", 0.0) or 0.0),
        "handtrack_prior_motion_peak": float(one_track_prior.get("motion_peak", 0.0) or 0.0),
        "handtrack_prior_support_frame_count": int(one_track_prior.get("support_frame_count", 0) or 0),
        "handtrack_prior_band_width": float(one_track_prior.get("band_width", 0.0) or 0.0),
        "handtrack_prior_family": str(one_track_prior.get("family") or ""),
        "handtrack_prior_template_ratio": float(one_track_prior.get("template_ratio", 0.0) or 0.0),
        "handtrack_prior_coarse_verb": str(one_track_prior.get("coarse_verb_label") or ""),
        "handtrack_prior_coarse_verb_score": float(one_track_prior.get("coarse_verb_score", 0.0) or 0.0),
        "handtrack_prior_candidate_phase": str(one_track_prior.get("candidate_phase") or ""),
        "handtrack_prior_candidate_score": float(one_track_prior.get("candidate_score", 0.0) or 0.0),
        "handtrack_prior_handedness_purity": float(one_track_prior.get("handedness_purity", 0.0) or 0.0),
        "handtrack_prior_phase_margin": float(one_track_prior.get("phase_margin", 0.0) or 0.0),
        "handtrack_prior_candidate_source": str(one_track_prior.get("candidate_source") or ""),
        "one_shot_verb": str(one_verb),
        "one_shot_verb_correct": bool(verb_one_correct),
        "one_shot_verb_rank": None if verb_one_rank is None else int(verb_one_rank),
        "one_shot_noun_id": None if one_noun_id is None else int(one_noun_id),
        "one_shot_noun_correct": bool(noun_one_correct),
        "one_shot_noun_rank": None if noun_one_rank is None else int(noun_one_rank),
        "one_shot_noun_source": str(one_noun_source_decision.get("preferred_source") or ("semantic_adapter_noun" if one_noun_id is not None else "")),
        "one_shot_noun_family": str(one_noun_source_decision.get("preferred_family") or ("semantic" if one_noun_id is not None else "")),
        "one_shot_noun_action": str(one_noun_source_decision.get("action_kind") or ""),
        "one_shot_noun_authority": str(one_noun_source_decision.get("authority_level") or ""),
        "one_shot_noun_interaction_form": str(one_noun_source_decision.get("interaction_form") or ""),
        "one_shot_noun_safe_apply": bool(one_noun_source_decision.get("safe_apply")),
        "one_shot_noun_detector_auto_apply": bool(one_detector_auto_apply),
        "after_gt_onset_verb": str(stage_verb),
        "after_gt_onset_verb_correct": bool(verb_stage_correct),
        "after_gt_onset_verb_rank": None if verb_stage_rank is None else int(verb_stage_rank),
        "after_gt_onset_gt_verb_noun_id": None if stage_noun_id is None else int(stage_noun_id),
        "after_gt_onset_gt_verb_noun_correct": bool(noun_stage_correct),
        "after_gt_onset_gt_verb_noun_rank": None if noun_stage_rank is None else int(noun_stage_rank),
        "after_gt_onset_gt_verb_noun_source": str(stage_noun_source_decision.get("preferred_source") or ("semantic_adapter_noun" if stage_noun_id is not None else "")),
        "after_gt_onset_gt_verb_noun_family": str(stage_noun_source_decision.get("preferred_family") or ("semantic" if stage_noun_id is not None else "")),
        "after_gt_onset_gt_verb_noun_action": str(stage_noun_source_decision.get("action_kind") or ""),
        "after_gt_onset_gt_verb_noun_authority": str(stage_noun_source_decision.get("authority_level") or ""),
        "after_gt_onset_gt_verb_noun_interaction_form": str(stage_noun_source_decision.get("interaction_form") or ""),
        "after_gt_onset_gt_verb_noun_safe_apply": bool(stage_noun_source_decision.get("safe_apply")),
        "after_gt_onset_gt_verb_noun_detector_auto_apply": bool(stage_detector_auto_apply),
        "one_shot_full_event_exact": bool(onset_strict and verb_one_correct and noun_one_correct),
        "one_shot_full_event_tol2": bool(onset_tol2 and verb_one_correct and noun_one_correct),
        "one_shot_full_event_tol5": bool(onset_tol5 and verb_one_correct and noun_one_correct),
        "sequential_zero_edit_exact": bool(corrections_strict == 0),
        "sequential_zero_edit_tol2": bool(corrections_tol2 == 0),
        "sequential_zero_edit_tol5": bool(corrections_tol5 == 0),
        "sequential_corrections_exact": int(corrections_strict),
        "sequential_corrections_tol2": int(corrections_tol2),
        "sequential_corrections_tol5": int(corrections_tol5),
        "sequential_help_reduction_exact_pct": _pct(3 - corrections_strict, 3),
        "sequential_help_reduction_tol2_pct": _pct(3 - corrections_tol2, 3),
        "sequential_help_reduction_tol5_pct": _pct(3 - corrections_tol5, 3),
    }


def _finalize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [dict(row) for row in list(records or [])]
    events = len(rows)
    onset_errs = [float(row["one_shot_onset_abs_err"]) for row in rows if row.get("one_shot_onset_abs_err") is not None]
    handtrack_prior_errs = [
        float(row["handtrack_prior_onset_abs_err"])
        for row in rows
        if row.get("handtrack_prior_onset_abs_err") is not None
    ]
    handtrack_prior_confidences = [
        float(row["handtrack_prior_confidence"])
        for row in rows
        if row.get("handtrack_prior_available")
    ]
    handtrack_prior_coverages = [
        float(row["handtrack_prior_coverage"])
        for row in rows
        if row.get("handtrack_prior_available")
    ]
    handtrack_prior_supports = [
        float(row["handtrack_prior_support_frame_count"])
        for row in rows
        if row.get("handtrack_prior_available")
    ]
    handtrack_prior_band_widths = [
        float(row["handtrack_prior_band_width"])
        for row in rows
        if row.get("handtrack_prior_available")
    ]
    handtrack_prior_candidate_scores = [
        float(row["handtrack_prior_candidate_score"])
        for row in rows
        if row.get("handtrack_prior_available")
    ]
    handtrack_prior_handedness_purities = [
        float(row["handtrack_prior_handedness_purity"])
        for row in rows
        if row.get("handtrack_prior_available")
    ]
    handtrack_prior_phase_margins = [
        float(row["handtrack_prior_phase_margin"])
        for row in rows
        if row.get("handtrack_prior_available")
    ]
    onset_band_widths = [
        float(row["one_shot_onset_band_width_frames"])
        for row in rows
        if row.get("one_shot_onset_band_width_frames") is not None
    ]
    verb_one_ranks = [int(row["one_shot_verb_rank"]) for row in rows if row.get("one_shot_verb_rank") is not None]
    verb_stage_ranks = [int(row["after_gt_onset_verb_rank"]) for row in rows if row.get("after_gt_onset_verb_rank") is not None]
    noun_one_ranks = [int(row["one_shot_noun_rank"]) for row in rows if row.get("one_shot_noun_rank") is not None]
    noun_stage_ranks = [int(row["after_gt_onset_gt_verb_noun_rank"]) for row in rows if row.get("after_gt_onset_gt_verb_noun_rank") is not None]
    corrections_exact = [int(row["sequential_corrections_exact"]) for row in rows]
    corrections_tol2 = [int(row["sequential_corrections_tol2"]) for row in rows]
    corrections_tol5 = [int(row["sequential_corrections_tol5"]) for row in rows]
    one_shot_detector_rows = sum(1 for row in rows if str(row.get("one_shot_noun_family") or "").strip().lower() == "detector_grounding")
    stage_detector_rows = sum(1 for row in rows if str(row.get("after_gt_onset_gt_verb_noun_family") or "").strip().lower() == "detector_grounding")
    one_shot_safe_local_rows = sum(1 for row in rows if str(row.get("one_shot_noun_authority") or "").strip().lower() == "safe_local")
    one_shot_human_confirm_rows = sum(1 for row in rows if str(row.get("one_shot_noun_authority") or "").strip().lower() == "human_confirm")
    one_shot_human_only_rows = sum(1 for row in rows if str(row.get("one_shot_noun_authority") or "").strip().lower() == "human_only")
    stage_safe_local_rows = sum(1 for row in rows if str(row.get("after_gt_onset_gt_verb_noun_authority") or "").strip().lower() == "safe_local")
    stage_human_confirm_rows = sum(1 for row in rows if str(row.get("after_gt_onset_gt_verb_noun_authority") or "").strip().lower() == "human_confirm")
    stage_human_only_rows = sum(1 for row in rows if str(row.get("after_gt_onset_gt_verb_noun_authority") or "").strip().lower() == "human_only")
    one_shot_detector_auto_apply_rows = sum(1 for row in rows if row.get("one_shot_noun_detector_auto_apply"))
    stage_detector_auto_apply_rows = sum(1 for row in rows if row.get("after_gt_onset_gt_verb_noun_detector_auto_apply"))
    return {
        "events": int(events),
        "clips": int(len({str(row.get('clip_id')) for row in rows})),
        "onset_mae_frames": _round(_mean(onset_errs)),
        "onset_median_ae_frames": _round(_median(onset_errs)),
        "onset_exact_pct": _round(_pct(sum(1 for row in rows if row.get("one_shot_onset_exact")), events)),
        "onset_acc_2f_pct": _round(_pct(sum(1 for row in rows if row.get("one_shot_onset_tol2")), events)),
        "onset_acc_5f_pct": _round(_pct(sum(1 for row in rows if row.get("one_shot_onset_tol5")), events)),
        "onset_band_coverage_pct": _round(_pct(sum(1 for row in rows if row.get("one_shot_onset_band_covered")), events)),
        "onset_band_width_mean_frames": _round(_mean(onset_band_widths)),
        "handtrack_prior_available_pct": _round(_pct(sum(1 for row in rows if row.get("handtrack_prior_available")), events)),
        "handtrack_prior_onset_mae_frames": _round(_mean(handtrack_prior_errs)),
        "handtrack_prior_onset_acc_2f_pct": _round(_pct(sum(1 for row in rows if row.get("handtrack_prior_onset_tol2")), events)),
        "handtrack_prior_onset_acc_5f_pct": _round(_pct(sum(1 for row in rows if row.get("handtrack_prior_onset_tol5")), events)),
        "handtrack_prior_confidence_mean": _round(_mean(handtrack_prior_confidences)),
        "handtrack_prior_coverage_mean": _round(_mean(handtrack_prior_coverages)),
        "handtrack_prior_support_frame_count_mean": _round(_mean(handtrack_prior_supports)),
        "handtrack_prior_band_width_mean_ratio": _round(_mean(handtrack_prior_band_widths)),
        "handtrack_prior_candidate_score_mean": _round(_mean(handtrack_prior_candidate_scores)),
        "handtrack_prior_handedness_purity_mean": _round(_mean(handtrack_prior_handedness_purities)),
        "handtrack_prior_phase_margin_mean": _round(_mean(handtrack_prior_phase_margins)),
        "verb_one_shot_top1_pct": _round(_pct(sum(1 for row in rows if row.get("one_shot_verb_correct")), events)),
        "verb_one_shot_recall3_pct": _round(_pct(sum(1 for row in rows if (row.get("one_shot_verb_rank") or 999) <= 3), events)),
        "verb_one_shot_recall5_pct": _round(_pct(sum(1 for row in rows if (row.get("one_shot_verb_rank") or 999) <= 5), events)),
        "verb_after_gt_onset_top1_pct": _round(_pct(sum(1 for row in rows if row.get("after_gt_onset_verb_correct")), events)),
        "verb_after_gt_onset_recall3_pct": _round(_pct(sum(1 for row in rows if (row.get("after_gt_onset_verb_rank") or 999) <= 3), events)),
        "verb_after_gt_onset_recall5_pct": _round(_pct(sum(1 for row in rows if (row.get("after_gt_onset_verb_rank") or 999) <= 5), events)),
        "verb_gt_rank_mean_one_shot": _round(_mean(verb_one_ranks)),
        "verb_gt_rank_mean_after_gt_onset": _round(_mean(verb_stage_ranks)),
        "noun_one_shot_top1_pct": _round(_pct(sum(1 for row in rows if row.get("one_shot_noun_correct")), events)),
        "noun_one_shot_recall3_pct": _round(_pct(sum(1 for row in rows if (row.get("one_shot_noun_rank") or 999) <= 3), events)),
        "noun_one_shot_recall5_pct": _round(_pct(sum(1 for row in rows if (row.get("one_shot_noun_rank") or 999) <= 5), events)),
        "noun_after_gt_onset_verb_top1_pct": _round(_pct(sum(1 for row in rows if row.get("after_gt_onset_gt_verb_noun_correct")), events)),
        "noun_after_gt_onset_verb_recall3_pct": _round(_pct(sum(1 for row in rows if (row.get("after_gt_onset_gt_verb_noun_rank") or 999) <= 3), events)),
        "noun_after_gt_onset_verb_recall5_pct": _round(_pct(sum(1 for row in rows if (row.get("after_gt_onset_gt_verb_noun_rank") or 999) <= 5), events)),
        "noun_gt_rank_mean_one_shot": _round(_mean(noun_one_ranks)),
        "noun_gt_rank_mean_after_gt_onset_verb": _round(_mean(noun_stage_ranks)),
        "noun_one_shot_detector_primary_pct": _round(_pct(one_shot_detector_rows, events)),
        "noun_after_gt_onset_verb_detector_primary_pct": _round(_pct(stage_detector_rows, events)),
        "noun_one_shot_detector_auto_apply_pct": _round(_pct(one_shot_detector_auto_apply_rows, events)),
        "noun_after_gt_onset_verb_detector_auto_apply_pct": _round(_pct(stage_detector_auto_apply_rows, events)),
        "noun_one_shot_safe_local_pct": _round(_pct(one_shot_safe_local_rows, events)),
        "noun_one_shot_human_confirm_pct": _round(_pct(one_shot_human_confirm_rows, events)),
        "noun_one_shot_human_only_pct": _round(_pct(one_shot_human_only_rows, events)),
        "noun_after_gt_onset_verb_safe_local_pct": _round(_pct(stage_safe_local_rows, events)),
        "noun_after_gt_onset_verb_human_confirm_pct": _round(_pct(stage_human_confirm_rows, events)),
        "noun_after_gt_onset_verb_human_only_pct": _round(_pct(stage_human_only_rows, events)),
        "one_shot_full_event_exact_pct": _round(_pct(sum(1 for row in rows if row.get("one_shot_full_event_exact")), events)),
        "one_shot_full_event_tol2_pct": _round(_pct(sum(1 for row in rows if row.get("one_shot_full_event_tol2")), events)),
        "one_shot_full_event_tol5_pct": _round(_pct(sum(1 for row in rows if row.get("one_shot_full_event_tol5")), events)),
        "sequential_residual_corrections_exact_per_event": _round(_mean(corrections_exact)),
        "sequential_residual_corrections_tol2_per_event": _round(_mean(corrections_tol2)),
        "sequential_residual_corrections_tol5_per_event": _round(_mean(corrections_tol5)),
        "sequential_action_reduction_exact_pct": _round(_pct((3 * events) - sum(corrections_exact), 3 * max(events, 1))),
        "sequential_action_reduction_tol2_pct": _round(_pct((3 * events) - sum(corrections_tol2), 3 * max(events, 1))),
        "sequential_action_reduction_tol5_pct": _round(_pct((3 * events) - sum(corrections_tol5), 3 * max(events, 1))),
        "sequential_zero_edit_exact_pct": _round(_pct(sum(1 for row in rows if row.get("sequential_zero_edit_exact")), events)),
        "sequential_zero_edit_tol2_pct": _round(_pct(sum(1 for row in rows if row.get("sequential_zero_edit_tol2")), events)),
        "sequential_zero_edit_tol5_pct": _round(_pct(sum(1 for row in rows if row.get("sequential_zero_edit_tol5")), events)),
        "sequential_one_or_less_edit_exact_pct": _round(_pct(sum(1 for row in rows if int(row.get("sequential_corrections_exact", 99)) <= 1), events)),
        "sequential_one_or_less_edit_tol2_pct": _round(_pct(sum(1 for row in rows if int(row.get("sequential_corrections_tol2", 99)) <= 1), events)),
        "sequential_one_or_less_edit_tol5_pct": _round(_pct(sum(1 for row in rows if int(row.get("sequential_corrections_tol5", 99)) <= 1), events)),
        "sequential_two_or_less_edit_exact_pct": _round(_pct(sum(1 for row in rows if int(row.get("sequential_corrections_exact", 99)) <= 2), events)),
        "sequential_two_or_less_edit_tol2_pct": _round(_pct(sum(1 for row in rows if int(row.get("sequential_corrections_tol2", 99)) <= 2), events)),
        "sequential_two_or_less_edit_tol5_pct": _round(_pct(sum(1 for row in rows if int(row.get("sequential_corrections_tol5", 99)) <= 2), events)),
        "paper_metrics": {
            "onset_mae_frames": _round(_mean(onset_errs)),
            "onset_acc_5f_pct": _round(_pct(sum(1 for row in rows if row.get("one_shot_onset_tol5")), events)),
            "verb_top1_after_gt_onset_pct": _round(_pct(sum(1 for row in rows if row.get("after_gt_onset_verb_correct")), events)),
            "noun_top1_after_gt_onset_verb_pct": _round(_pct(sum(1 for row in rows if row.get("after_gt_onset_gt_verb_noun_correct")), events)),
            "residual_corrections_per_event_exact": _round(_mean(corrections_exact)),
            "residual_corrections_per_event_tol5": _round(_mean(corrections_tol5)),
            "zero_edit_event_rate_tol5_pct": _round(_pct(sum(1 for row in rows if row.get("sequential_zero_edit_tol5")), events)),
            "one_or_less_edit_rate_tol5_pct": _round(_pct(sum(1 for row in rows if int(row.get("sequential_corrections_tol5", 99)) <= 1), events)),
        },
    }


def _group_records(records: Sequence[Dict[str, Any]], key_name: str) -> Dict[str, Any]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in list(records or []):
        buckets[str(row.get(key_name) or "")].append(dict(row))
    return {str(key): _finalize_records(bucket) for key, bucket in sorted(buckets.items())}


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    items = [dict(row) for row in list(rows or [])]
    if not items:
        return
    fieldnames: List[str] = []
    for row in items:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(items)


def run_sequential_runtime_eval(
    *,
    dataset_root: str,
    adapter_path: str,
    videomae_weights: str,
    videomae_verbs: str = "",
    handtracks_dir: str = "",
    nouns_path: str = "",
    verbs_path: str = "",
    ontology_path: str = "",
    cache_dir: str = os.path.join("analysis", "videomae_cache"),
    stride: int = 4,
    window_span: int = 16,
    refinement_passes: int = 2,
    clip_ids: Optional[Sequence[str]] = None,
    recursive: bool = False,
    annotation_root: str = "",
    calibration_inputs: Optional[Sequence[str]] = None,
    enable_handtrack_fusion: bool = False,
    enable_cross_hand_exclusion: bool = True,
    enable_source_arbitration: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    nouns_path = _discover_nouns_path(dataset_root, nouns_path)
    verbs_path = _discover_verbs_path(dataset_root, verbs_path)
    ontology_path = _discover_ontology_path(dataset_root, ontology_path)
    if not nouns_path:
        raise SystemExit("Failed to locate noun/object list. Pass --nouns explicitly.")
    if not ontology_path:
        raise SystemExit("Failed to locate ontology CSV. Pass --ontology explicitly.")
    ontology = HOIOntology.from_csv(ontology_path)

    package = load_adapter_package(adapter_path)
    if package is None:
        raise SystemExit(f"Failed to load semantic adapter: {adapter_path}")

    cache_dir = _ensure_cache_dir(cache_dir)
    videomae_handler = VideoMAEHandler()
    ok, msg = videomae_handler.load_model(videomae_weights, videomae_verbs or verbs_path or None)
    if not ok:
        raise SystemExit(f"Failed to load VideoMAE encoder:\n{msg}")

    clips = _load_clip_data(
        dataset_root=dataset_root,
        annotation_root=annotation_root or dataset_root,
        nouns_path=nouns_path,
        ontology=ontology,
        handtracks_dir=handtracks_dir,
        cache_dir=cache_dir,
        videomae_handler=videomae_handler,
        stride=int(stride),
        window_span=int(window_span),
        clip_ids=[item.strip() for item in list(clip_ids or []) if item and str(item).strip()],
        recursive=bool(recursive),
    )
    if not clips:
        raise SystemExit(
            f"No annotation clips were discovered under: {os.path.abspath(annotation_root or dataset_root)}"
        )

    has_any_handtracks = any(bool((clip.handtracks.get("tracks") or {})) for clip in clips)
    calibration_paths = _expand_calibration_inputs(
        [str(item or "").strip() for item in list(calibration_inputs or []) if str(item or "").strip()]
    )
    calibrator = HOIEmpiricalCalibrator.from_sources(csv_paths=calibration_paths)

    variants = [
        VariantSpec(
            name="semantic_only",
            use_handtrack=False,
            use_oracle_grounding=False,
            use_source_arbitration=False,
            use_hop=False,
            description="Frozen encoder + semantic adapter only. No handtrack prior and no object support.",
        ),
        VariantSpec(
            name="full_assist_oracle",
            use_handtrack=True,
            use_oracle_grounding=True,
            use_source_arbitration=bool(enable_source_arbitration),
            description="Full IMPACT-HOI system with oracle grounding.",
        ),
        VariantSpec(
            name="wo_scr",
            use_handtrack=True,
            use_oracle_grounding=True,
            use_source_arbitration=True,
            use_scr=False,
            description="Full Assist without Semantic Constraint Refinement (SCR).",
        ),
        VariantSpec(
            name="wo_tsc",
            use_handtrack=True,
            use_oracle_grounding=True,
            use_source_arbitration=True,
            use_tsc=False,
            description="Full Assist with Confidence-Only Policy (No TSC).",
        ),
        VariantSpec(
            name="wo_lock_aware",
            use_handtrack=True,
            use_oracle_grounding=True,
            use_source_arbitration=True,
            use_lock_aware=False,
            description="Full Assist without Lock-Constrained Decoding (No Clamping).",
        ),
        VariantSpec(
            name="wo_hop",
            use_handtrack=True,
            use_oracle_grounding=True,
            use_source_arbitration=True,
            use_hop=False,
            description="Full Assist without Hand-guided Onset Prior (HOP).",
        ),
    ]

    all_rows: List[Dict[str, Any]] = []
    variant_payload: Dict[str, Any] = {}
    for variant in variants:
        variant_rows: List[Dict[str, Any]] = []
        for clip in clips:
            runtime = OfflineHOIRuntime(
                clip=clip,
                package=package,
                ontology=ontology,
                variant=variant,
                refinement_passes=int(refinement_passes),
                calibrator=calibrator,
                enable_handtrack_fusion=bool(enable_handtrack_fusion),
                enable_cross_hand_exclusion=bool(enable_cross_hand_exclusion),
            )
            for event in clip.events:
                variant_rows.append(_event_record(runtime=runtime, event=event, variant=variant))
        all_rows.extend(variant_rows)
        variant_payload[variant.name] = {
            "description": variant.description,
            "summary": _finalize_records(variant_rows),
            "per_hand_summary": _group_records(variant_rows, "hand"),
            "per_file": _group_records(variant_rows, "clip_id"),
        }

    payload = {
        "mode": "full_assist_sequential_oracle_runtime",
        "dataset_root": os.path.abspath(dataset_root),
        "annotation_root": os.path.abspath(annotation_root or dataset_root),
        "nouns_path": nouns_path,
        "verbs_path": verbs_path,
        "ontology_path": ontology_path,
        "handtracks_dir": os.path.abspath(handtracks_dir) if str(handtracks_dir or "").strip() else "",
        "calibration_inputs": calibration_paths,
        "recursive_scan": bool(recursive),
        "handtracks_available": bool(has_any_handtracks),
        "assumptions": {
            "start_end_source": "ground_truth",
            "sequential_order": ["onset", "verb", "noun"],
            "sequential_rule": (
                "At each stage the system predicts the next unresolved field(s). "
                "If prediction != GT, oracle human correction is applied and the corrected field is clamped "
                "before re-decoding the remaining unresolved fields."
            ),
            "variants": [variant.name for variant in variants],
            "videomae_stride": int(stride),
            "videomae_window_span": int(window_span),
            "refinement_passes": int(refinement_passes),
            "handtrack_post_fusion_enabled": bool(enable_handtrack_fusion),
            "cross_hand_exclusion_enabled": bool(enable_cross_hand_exclusion),
            "source_arbitration_enabled_for_oracle_grounding": bool(enable_source_arbitration),
            "note": (
                "semantic_only is the conservative runtime baseline. "
                "Any *_oracle_grounding variant is an upper bound because GT boxes are used as grounding support. "
                "Cross-hand exclusion is approximated from opposing-hand GT events whose onset falls inside the current segment."
            ),
        },
        "variants": variant_payload,
    }
    return payload, all_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Full Assist offline under sequential oracle clamp-and-redecode. "
            "GT start/end are given; onset, verb, and noun are evaluated stage-wise."
        )
    )
    parser.add_argument("dataset_root", help="Dataset folder containing videos and annotation JSON files.")
    parser.add_argument("--annotation-root", default="", help="Optional annotation directory or file. Defaults to dataset_root.")
    parser.add_argument("--adapter", required=True, help="Semantic adapter package path.")
    parser.add_argument("--videomae-weights", required=True, help="Frozen VideoMAE checkpoint path.")
    parser.add_argument("--videomae-verbs", default="", help="Optional VideoMAE verb-label path.")
    parser.add_argument("--handtracks-dir", default="", help="Directory containing *.handtracks.*.json files.")
    parser.add_argument("--calibration-input", action="append", default=[], help="Optional .ops.log.csv file or directory used to seed source-aware offline calibration. Repeatable.")
    parser.add_argument("--nouns", default="", help="Optional noun/object list path. Defaults are auto-discovered.")
    parser.add_argument("--verbs", default="", help="Optional verb list path. Defaults are auto-discovered.")
    parser.add_argument("--ontology", default="", help="Optional ontology CSV path. Defaults are auto-discovered.")
    parser.add_argument("--cache-dir", default=os.path.join("analysis", "videomae_cache"), help="Directory for cached VideoMAE features.")
    parser.add_argument("--out", default=os.path.join("analysis", "goodvideos_full_assist_sequential_runtime.json"), help="Output JSON path.")
    parser.add_argument("--per-event-csv", default=os.path.join("analysis", "goodvideos_full_assist_sequential_runtime_per_event.csv"), help="Per-event CSV path.")
    parser.add_argument("--stride", type=int, default=4, help="VideoMAE dense-cache stride in frames. Default: 4.")
    parser.add_argument("--window-span", type=int, default=16, help="VideoMAE dense-cache window span. Default: 16.")
    parser.add_argument("--refinement-passes", type=int, default=2, help="Semantic refinement passes. Default: 2.")
    parser.add_argument("--clip-ids", default="", help="Optional comma-separated clip ids for smoke runs.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search annotation JSON files and sibling videos under dataset_root or --annotation-root.")
    parser.add_argument("--enable-handtrack-fusion", action="store_true", help="Enable legacy post-decode handtrack/semantic onset fusion. Disabled by default so HOP acts only as a prior on onset-band sampling.")
    parser.add_argument("--disable-handtrack-fusion", action="store_true", help="Deprecated alias. Post-decode handtrack fusion is already disabled by default.")
    parser.add_argument("--disable-cross-hand-exclusion", action="store_true", help="Disable the runtime's opposing-hand onset exclusion prior.")
    parser.add_argument("--disable-source-arbitration", action="store_true", help="Disable source-aware semantic-vs-grounding noun arbitration for oracle-grounding variants.")
    args = parser.parse_args()

    payload, all_rows = run_sequential_runtime_eval(
        dataset_root=args.dataset_root,
        annotation_root=args.annotation_root,
        adapter_path=args.adapter,
        videomae_weights=args.videomae_weights,
        videomae_verbs=args.videomae_verbs,
        handtracks_dir=args.handtracks_dir,
        calibration_inputs=args.calibration_input,
        nouns_path=args.nouns,
        verbs_path=args.verbs,
        ontology_path=args.ontology,
        cache_dir=args.cache_dir,
        stride=int(args.stride),
        window_span=int(args.window_span),
        refinement_passes=int(args.refinement_passes),
        clip_ids=[item.strip() for item in str(args.clip_ids or "").split(",") if item.strip()],
        recursive=bool(args.recursive),
        enable_handtrack_fusion=bool(args.enable_handtrack_fusion) and not bool(args.disable_handtrack_fusion),
        enable_cross_hand_exclusion=not bool(args.disable_cross_hand_exclusion),
        enable_source_arbitration=not bool(args.disable_source_arbitration),
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    _write_csv(args.per_event_csv, all_rows)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.per_event_csv}")
    for variant_name, variant_info in dict(payload.get("variants") or {}).items():
        paper = dict((variant_info.get("summary") or {}).get("paper_metrics") or {})
        print(
            f"[{variant_name}] "
            f"onset_mae={paper['onset_mae_frames']}, "
            f"onset_acc5={paper['onset_acc_5f_pct']}%, "
            f"verb_top1={paper['verb_top1_after_gt_onset_pct']}%, "
            f"noun_top1={paper['noun_top1_after_gt_onset_verb_pct']}%, "
            f"corr_exact={paper['residual_corrections_per_event_exact']}, "
            f"corr_tol5={paper['residual_corrections_per_event_tol5']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
