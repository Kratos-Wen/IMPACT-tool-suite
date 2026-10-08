import os
import json
import sys
import glob
import argparse
import random
import math
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.hoi_ontology import HOIOntology, ontology_noun_required
from core.videomae_v2_logic import load_precomputed_feature_cache, aggregate_precomputed_feature_cache
from core.onset_guidance import build_onset_band, build_local_onset_window
from core.semantic_adapter import train_adapter_from_feedback


def _safe_int(v):
    try: return int(v)
    except: return None

def _safe_float(v):
    try: return float(v)
    except: return 0.0

def _bounded01(v, default=0.0):
    try: return max(0.0, min(1.0, float(v)))
    except: return default


def _norm_text(value):
    return str(value or "").strip()


def _norm_key(value):
    return _norm_text(value).lower().replace(" ", "_")


def _normalize_hand_name(value):
    key = str(value or "").strip().lower().replace("-", "_")
    if key in {"left", "left_hand"}:
        return "Left_hand"
    if key in {"right", "right_hand"}:
        return "Right_hand"
    return str(value or "").strip()


def _canonical_handedness_actor(value):
    key = str(value or "").strip().lower().replace("-", "_")
    if key.startswith("left"):
        return "Left_hand"
    if key.startswith("right"):
        return "Right_hand"
    return ""


def _track_handedness_stats(track):
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


def _repair_track_actor_consistency(tracks_out):
    repaired = {str(actor_id): dict(track or {}) for actor_id, track in dict(tracks_out or {}).items()}
    repaired.setdefault("Left_hand", {})
    repaired.setdefault("Right_hand", {})
    stats = {
        actor_id: _track_handedness_stats(repaired.get(actor_id) or {})
        for actor_id in ("Left_hand", "Right_hand")
    }

    def _is_strong_opposite(actor_id):
        track_stats = dict(stats.get(actor_id) or {})
        majority_actor = str(track_stats.get("majority_actor") or "")
        return bool(
            majority_actor
            and majority_actor != actor_id
            and float(track_stats.get("purity", 0.0) or 0.0) >= 0.72
            and float(track_stats.get("vote_weight", 0.0) or 0.0) >= 1.25
            and int(track_stats.get("vote_rows", 0) or 0) >= 3
        )

    def _is_weak(actor_id):
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


def _norm_verb_token(value):
    return _norm_key(value).replace("-", "_")


VERB_TEMPLATE_RATIOS = {
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

VERB_FAMILY_DEFAULT_RATIOS = {
    "boundary": 0.02,
    "early": 0.24,
    "mid": 0.45,
    "late": 0.66,
}

VERB_FAMILY_WINDOW_RATIOS = {
    "boundary": 0.12,
    "early": 0.18,
    "mid": 0.22,
    "late": 0.18,
}

VERB_FAMILY_BAND_RATIOS = {
    "boundary": 0.08,
    "early": 0.12,
    "mid": 0.14,
    "late": 0.12,
}

VERB_FAMILY_LABELS = {
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


def _verb_family_for_label(label):
    token = _norm_verb_token(label)
    return str(VERB_FAMILY_LABELS.get(token) or "mid")


def _template_ratio_for_label(label):
    token = _norm_verb_token(label)
    family = _verb_family_for_label(token)
    return float(VERB_TEMPLATE_RATIOS.get(token, VERB_FAMILY_DEFAULT_RATIOS.get(family, 0.45)))


def _semantic_prior_from_scores(candidate_scores):
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
    family_scores = defaultdict(float)
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


def _row_motion(row):
    return float(row.get("motion", 0.0) or 0.0)


PHASE_PRIOR_BY_FAMILY = {
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


def _local_peak_rows(rows):
    seq = [dict(row or {}) for row in list(rows or [])]
    if len(seq) <= 2:
        return seq
    peaks = []
    for idx, row in enumerate(seq):
        cur = _row_motion(row)
        prev = _row_motion(seq[idx - 1]) if idx > 0 else cur
        nxt = _row_motion(seq[idx + 1]) if idx + 1 < len(seq) else cur
        if cur >= prev - 1e-6 and cur >= nxt - 1e-6:
            peaks.append(dict(row))
    return peaks or seq


def _local_valley_rows(rows):
    seq = [dict(row or {}) for row in list(rows or [])]
    if len(seq) <= 2:
        return seq
    valleys = []
    for idx, row in enumerate(seq):
        cur = _row_motion(row)
        prev = _row_motion(seq[idx - 1]) if idx > 0 else cur
        nxt = _row_motion(seq[idx + 1]) if idx + 1 < len(seq) else cur
        if cur <= prev + 1e-6 and cur <= nxt + 1e-6:
            valleys.append(dict(row))
    return valleys or seq


def _stable_run_starts(rows, *, low_threshold, min_run=2):
    seq = sorted(
        [dict(row or {}) for row in list(rows or [])],
        key=lambda item: int(item.get("frame", 0) or 0),
    )
    if not seq:
        return []
    out = []
    run_start = None
    for idx in range(len(seq) + 1):
        is_low = idx < len(seq) and _row_motion(seq[idx]) <= float(low_threshold) + 1e-6
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


def _segment_handedness_purity(rows, hand_key):
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


def _best_template_candidate(rows, *, target_frame, search_left, search_right):
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
    best_row = None
    best_score = None
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


def _best_peak_candidate(rows, *, target_frame, search_left, search_right):
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
    best_row = None
    best_score = None
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


def _phase_prior_score(family, candidate_kind):
    table = dict(PHASE_PRIOR_BY_FAMILY.get(str(family or "mid"), PHASE_PRIOR_BY_FAMILY["mid"]) or {})
    return float(table.get(str(candidate_kind or ""), 0.18))


def _build_hop_band(*, start, end, onset_frame, family, candidate_kind, rows, chosen_row, confidence):
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


def _unguided_segment_candidate_scores(cache, start_frame, end_frame, verb_labels, candidate_cache=None):
    start = _safe_int(start_frame)
    end = _safe_int(end_frame)
    if start is None or end is None or not cache:
        return {}
    if end < start:
        start, end = end, start
    key = (int(start), int(end))
    if isinstance(candidate_cache, dict) and key in candidate_cache:
        return dict(candidate_cache.get(key) or {})
    summary = aggregate_precomputed_feature_cache(
        cache,
        start_frame=int(start),
        end_frame=int(end),
        onset_band=None,
        top_k=max(5, int(len(verb_labels or []) or 0)),
    )
    scores = {
        str(item.get("label") or ""): float(item.get("score") or 0.0)
        for item in list(summary.get("candidates") or [])
        if _norm_text(item.get("label"))
    }
    if isinstance(candidate_cache, dict):
        candidate_cache[key] = dict(scores)
    return dict(scores)


def _object_name_for_id(object_library, object_id):
    if object_id is None:
        return ""
    info = dict((object_library or {}).get(int(object_id)) or {})
    return _norm_text(info.get("label")) or _norm_text(info.get("category"))


def _event_noun_id(event, tracks, noun_name_to_id, object_library):
    noun_id = _safe_int((event or {}).get("noun_object_id"))
    if noun_id is not None:
        return int(noun_id)
    links = dict((event or {}).get("links") or {})
    target_track_id = _norm_text(links.get("target_track_id"))
    if target_track_id:
        track = dict((tracks or {}).get(target_track_id) or {})
        track_object_id = _safe_int(track.get("object_id"))
        if track_object_id is not None:
            return int(track_object_id)
    interaction = dict((event or {}).get("interaction") or {})
    target_name = _norm_text(interaction.get("target")) or _norm_text(interaction.get("noun"))
    if target_name:
        for name, object_id in dict(noun_name_to_id or {}).items():
            if _norm_key(name) == _norm_key(target_name):
                return int(object_id)
    noun_label = _norm_text(interaction.get("target")) or _norm_text(interaction.get("noun"))
    if noun_label:
        for object_id, info in dict(object_library or {}).items():
            label = _norm_text((info or {}).get("label")) or _norm_text((info or {}).get("category"))
            if _norm_key(label) == _norm_key(noun_label):
                return int(object_id)
    return None


def _build_bboxes_by_frame(gt_data, object_library):
    tracks = dict((gt_data or {}).get("tracks") or {})
    bboxes_by_frame = defaultdict(list)
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
                    "confidence": 1.0,
                    "frame": int(frame),
                    "x1": float(bbox[0]),
                    "y1": float(bbox[1]),
                    "x2": float(bbox[2]),
                    "y2": float(bbox[3]),
                }
            )
    return {int(k): [dict(v) for v in vals] for k, vals in bboxes_by_frame.items()}


def _box_center_xy(box):
    if not isinstance(box, dict):
        return 0.0, 0.0
    return (
        (float(box.get("x1", 0.0) or 0.0) + float(box.get("x2", 0.0) or 0.0)) * 0.5,
        (float(box.get("y1", 0.0) or 0.0) + float(box.get("y2", 0.0) or 0.0)) * 0.5,
    )


def _box_diag(box):
    if not isinstance(box, dict):
        return 0.0
    x1 = float(box.get("x1", 0.0) or 0.0)
    y1 = float(box.get("y1", 0.0) or 0.0)
    x2 = float(box.get("x2", 0.0) or 0.0)
    y2 = float(box.get("y2", 0.0) or 0.0)
    return float(max(0.0, math.hypot(x2 - x1, y2 - y1)))


def _box_iou(box_a, box_b):
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


def _box_edge_gap(box_a, box_b):
    if not isinstance(box_a, dict) or not isinstance(box_b, dict):
        return float("inf")
    ax1, ay1, ax2, ay2 = [float(box_a.get(key, 0.0) or 0.0) for key in ("x1", "y1", "x2", "y2")]
    bx1, by1, bx2, by2 = [float(box_b.get(key, 0.0) or 0.0) for key in ("x1", "y1", "x2", "y2")]
    dx = max(0.0, ax1 - bx2, bx1 - ax2)
    dy = max(0.0, ay1 - by2, by1 - ay2)
    return float(math.hypot(dx, dy))


def _box_confidence_score(box):
    if not isinstance(box, dict):
        return 0.0
    conf = box.get("confidence")
    if conf is not None:
        return _bounded01(conf, 0.0)
    return 0.5


def _hand_object_proximity_score(hand_box, object_box, frame_width, frame_height):
    if not isinstance(hand_box, dict) or not isinstance(object_box, dict):
        return 0.0
    frame_diag = float(max(1.0, math.hypot(float(frame_width or 0.0), float(frame_height or 0.0))))
    hand_diag = float(max(1.0, _box_diag(hand_box)))
    object_diag = float(max(1.0, _box_diag(object_box)))
    hx, hy = _box_center_xy(hand_box)
    ox, oy = _box_center_xy(object_box)
    center_dist = float(math.hypot(ox - hx, oy - hy))
    center_scale = max(24.0, 0.95 * hand_diag + 0.55 * object_diag)
    center_score = _bounded01(1.0 - center_dist / float(max(1.0, center_scale)), 0.0)
    gap = _box_edge_gap(hand_box, object_box)
    gap_score = _bounded01(1.0 - gap / float(max(1.0, 0.18 * frame_diag)), 0.0)
    iou_score = _bounded01(_box_iou(hand_box, object_box) * 3.0, 0.0)
    return _bounded01(max(gap_score, 0.60 * center_score + 0.40 * iou_score), 0.0)


def _frame_hand_box(handtrack_data, hand_key, frame):
    tracks = dict((handtrack_data or {}).get("tracks") or {})
    track = dict(tracks.get(_normalize_hand_name(hand_key)) or {})
    row = dict((track.get("frame_map") or {}).get(int(frame)) or {})
    bbox = list(row.get("bbox") or [])
    if len(bbox) < 4:
        return {}
    return {
        "id": str(hand_key),
        "label": str(hand_key),
        "frame": int(frame),
        "x1": float(bbox[0]),
        "y1": float(bbox[1]),
        "x2": float(bbox[2]),
        "y2": float(bbox[3]),
        "confidence": float(row.get("detection_confidence", 0.0) or 0.0),
    }


def _event_object_candidate_frames(start_f, end_f, onset_f, hand_key, handtrack_data):
    frames = []

    def _add(frame_value):
        frame_int = _safe_int(frame_value)
        if frame_int is None:
            return
        if frame_int < int(start_f) or frame_int > int(end_f):
            return
        if frame_int not in frames:
            frames.append(int(frame_int))

    for value in (start_f, onset_f, end_f):
        _add(value)
    if onset_f is not None:
        _add(int(onset_f) - 1)
        _add(int(onset_f) + 1)
    tracks = dict((handtrack_data or {}).get("tracks") or {})
    track = dict(tracks.get(_normalize_hand_name(hand_key)) or {})
    motion_peak_frame = _safe_int(track.get("motion_peak_frame"))
    _add(motion_peak_frame)
    if motion_peak_frame is not None:
        _add(int(motion_peak_frame) - 1)
        _add(int(motion_peak_frame) + 1)
    return frames


def _collect_object_candidates(
    *,
    start_f,
    end_f,
    onset_f,
    hand_key,
    handtrack_data,
    bboxes_by_frame,
    frame_width,
    frame_height,
):
    frames = _event_object_candidate_frames(start_f, end_f, onset_f, hand_key, handtrack_data)
    if not frames:
        return []
    track = dict((dict((handtrack_data or {}).get("tracks") or {}).get(_normalize_hand_name(hand_key)) or {}))
    frame_map = dict(track.get("frame_map") or {})
    motion_peak_frame = _safe_int(track.get("motion_peak_frame"))
    max_motion = max(
        [1e-6]
        + [float((frame_map.get(int(frame)) or {}).get("motion", 0.0) or 0.0) for frame in frames]
    )
    candidates = {}
    for frame in frames:
        hand_box = _frame_hand_box(handtrack_data, hand_key, int(frame))
        track_row = dict(frame_map.get(int(frame)) or {})
        motion_value = float(track_row.get("motion", 0.0) or 0.0)
        motion_weight = _bounded01(motion_value / float(max_motion), 0.0)
        onset_weight = 0.0
        if onset_f is not None:
            if int(frame) == int(onset_f):
                onset_weight = 1.0
            elif abs(int(frame) - int(onset_f)) == 1:
                onset_weight = 0.55
        peak_weight = 0.0
        if motion_peak_frame is not None:
            if int(frame) == int(motion_peak_frame):
                peak_weight = 1.0
            elif abs(int(frame) - int(motion_peak_frame)) == 1:
                peak_weight = 0.55
        for box in list((bboxes_by_frame or {}).get(int(frame), []) or []):
            object_id = _safe_int(box.get("id"))
            if object_id is None:
                continue
            row = candidates.setdefault(
                int(object_id),
                {
                    "object_id": int(object_id),
                    "support_score": 0.0,
                    "onset_support_score": 0.0,
                    "motion_support_score": 0.0,
                    "hand_proximity_max": 0.0,
                    "yolo_confidence_max": 0.0,
                },
            )
            yolo_conf = _box_confidence_score(box)
            proximity = _hand_object_proximity_score(hand_box, box, frame_width, frame_height)
            motion_alignment = max(peak_weight, motion_weight) * max(0.15, proximity)
            support_score = 0.40 + 0.35 * proximity + 0.25 * yolo_conf
            onset_support_score = onset_weight * max(0.20, 0.60 * proximity + 0.40 * yolo_conf)
            row["support_score"] = float(row.get("support_score", 0.0) or 0.0) + float(support_score)
            row["onset_support_score"] = float(row.get("onset_support_score", 0.0) or 0.0) + float(onset_support_score)
            row["motion_support_score"] = float(row.get("motion_support_score", 0.0) or 0.0) + float(motion_alignment)
            row["hand_proximity_max"] = max(float(row.get("hand_proximity_max", 0.0) or 0.0), float(proximity))
            row["yolo_confidence_max"] = max(float(row.get("yolo_confidence_max", 0.0) or 0.0), float(yolo_conf))
    rows = [dict(item) for item in candidates.values()]
    if not rows:
        return []
    max_support_score = max([1e-6] + [float(item.get("support_score", 0.0) or 0.0) for item in rows])
    max_onset_support_score = max([1e-6] + [float(item.get("onset_support_score", 0.0) or 0.0) for item in rows])
    out = []
    for item in rows:
        item["candidate_score"] = _bounded01(
            0.30 * float(item.get("yolo_confidence_max", 0.0) or 0.0)
            + 0.34 * float(item.get("hand_proximity_max", 0.0) or 0.0)
            + 0.22 * _bounded01(float(item.get("onset_support_score", 0.0) or 0.0) / float(max_onset_support_score), 0.0)
            + 0.14 * _bounded01(float(item.get("support_score", 0.0) or 0.0) / float(max_support_score), 0.0),
            0.0,
        )
        out.append(item)
    out.sort(key=lambda item: float(item.get("candidate_score", 0.0) or 0.0), reverse=True)
    return out


def _support_maps_for_state(
    *,
    start_f,
    end_f,
    onset_f,
    hand_key,
    handtrack_data,
    bboxes_by_frame,
    frame_width,
    frame_height,
):
    candidates = _collect_object_candidates(
        start_f=start_f,
        end_f=end_f,
        onset_f=onset_f,
        hand_key=hand_key,
        handtrack_data=handtrack_data,
        bboxes_by_frame=bboxes_by_frame,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    support_map = {
        int(item.get("object_id")): float(item.get("support_score", 0.0) or 0.0)
        for item in candidates
        if _safe_int(item.get("object_id")) is not None
    }
    onset_support_map = {
        int(item.get("object_id")): float(item.get("onset_support_score", 0.0) or 0.0)
        for item in candidates
        if _safe_int(item.get("object_id")) is not None
    }
    return support_map, onset_support_map


def _noun_exists_prior_for_verb(ontology, verb_label):
    text = _norm_text(verb_label)
    if not text:
        return 0.5
    if ontology is None or not ontology.has_verb(text):
        return 0.75
    return 0.15 if ontology.allow_no_noun(text) else 0.95


def _normalize_handtrack_payload(payload):
    out = dict(payload or {})
    tracks_out = {}
    for actor_id, track in dict(out.get("tracks") or {}).items():
        rows = []
        frame_map = {}
        raw_frames = list((track or {}).get("frames") or [])
        centers = {}
        for row in raw_frames:
            f_id = _safe_int(row.get("frame", row.get("f")))
            if f_id is None: continue
            bbox = list(row.get("bbox") or [])
            if not bbox and "x1" in row:
                bbox = [float(row.get("x1",0)), float(row.get("y1",0)),
                        float(row.get("x2",0)), float(row.get("y2",0))]
            if len(bbox) >= 4:
                centers[int(f_id)] = ((bbox[0]+bbox[2])*0.5, (bbox[1]+bbox[3])*0.5)

        for row in raw_frames:
            if not isinstance(row, dict): continue
            frame = _safe_int(row.get("frame", row.get("f")))
            if frame is None: continue
            bbox = list(row.get("bbox") or [])
            if not bbox and "x1" in row:
                bbox = [float(row.get("x1",0)), float(row.get("y1",0)),
                        float(row.get("x2",0)), float(row.get("y2",0))]
            if len(bbox) < 4: bbox = [0,0,0,0]
            center = list(row.get("center") or [])
            if not center and "cx" in row:
                center = [float(row.get("cx",0)), float(row.get("cy",0))]
            if len(center) < 2:
                center = [(bbox[0]+bbox[2])*0.5, (bbox[1]+bbox[3])*0.5]
            motion = _safe_float(row.get("motion"))
            if "motion" not in row or motion <= 0:
                prev_c = centers.get(int(frame)-1)
                if prev_c:
                    dx = (center[0]-prev_c[0])*1920
                    dy = (center[1]-prev_c[1])*1080
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
        rows.sort(key=lambda it: int(it.get("frame",0)))
        if len(rows) > 3:
            for i in range(1, len(rows)-1):
                rows[i]["motion"] = (0.5*rows[i]["motion"]
                                     + 0.25*rows[i-1]["motion"]
                                     + 0.25*rows[i+1]["motion"])
        norm_actor = _canonical_handedness_actor(actor_id) or _normalize_hand_name(actor_id)
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
    return out


def handtrack_segment_prior(tracks_data, hand_key, start, end, cache=None, verb_labels=None, candidate_cache=None):
    """Semantic-gated, phase-aware HOP aligned with the offline evaluator."""
    key_lower = str(hand_key).strip().lower().replace("-", "_")
    norm_map = {
        "left": "Left_hand",
        "left_hand": "Left_hand",
        "right": "Right_hand",
        "right_hand": "Right_hand",
    }
    norm_key = norm_map.get(key_lower, hand_key)
    tracks_dict = dict(tracks_data.get("tracks") or {})
    track = tracks_dict.get(norm_key)
    if not track:
        for k, v in tracks_dict.items():
            if str(k).lower().replace("_", "") == key_lower.replace("_", ""):
                track = v
                break
    if not track:
        return {}
    frame_map = dict(track.get("frame_map") or {})
    if not frame_map:
        return {}
    segment_len = max(1, int(end) - int(start))
    rows = [dict(frame_map.get(int(f)) or {}) for f in range(int(start), int(end) + 1) if int(f) in frame_map]
    if len(rows) < 3:
        return {}
    coverage = float(len(rows)) / float(max(1, segment_len + 1))
    handedness_purity = float(_segment_handedness_purity(rows, str(norm_key)))
    peak_rows = _local_peak_rows(rows)
    peak_row = max(peak_rows or rows, key=lambda row: float(row.get("motion", 0.0) or 0.0))
    peak_motion = float(peak_row.get("motion", 0.0) or 0.0)
    min_motion = float(min(_row_motion(row) for row in rows))
    avg_motion = sum(float(row.get("motion", 0.0) or 0.0) for row in rows) / max(1, len(rows))
    if peak_motion <= 1e-6:
        return {}
    dominance = peak_motion / float(max(1e-4, avg_motion))
    coarse_prior = _semantic_prior_from_scores(
        _unguided_segment_candidate_scores(
            cache,
            int(start),
            int(end),
            verb_labels,
            candidate_cache=candidate_cache,
        )
    )
    family = str(coarse_prior.get("family") or "mid")
    template_ratio = float(
        coarse_prior.get("template_ratio", VERB_FAMILY_DEFAULT_RATIOS["mid"])
        or VERB_FAMILY_DEFAULT_RATIOS["mid"]
    )
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
    stable_rows = _stable_run_starts(
        search_rows if family != "late" else rows,
        low_threshold=float(low_threshold),
        min_run=int(stable_min_run),
    )

    phase_candidates = []
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
    score_rows = []
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
        "hand": str(norm_key),
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


def _load_handtrack_file(handtracks_dir, clip_id):
    matches = sorted(glob.glob(os.path.join(handtracks_dir, f"{clip_id}.handtracks.*.json")))
    if not matches: return None
    with open(matches[0], 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return _normalize_handtrack_payload(raw)


def build_sample(
    cache, start_f, end_f,
    sim_onset_frame, sim_onset_ratio, sim_has_onset, sim_status,
    gt_onset_ratio, verb_labels, noun_list, v_idx, n_idx,
    onset_context_frame=None, onset_context_band=None,
    current_verb_label="",
    current_noun_id=None,
    support_map=None,
    onset_support_map=None,
    ontology=None,
    supervised_fields=None,
):
    segment_len = max(1, end_f - start_f)
    visual_onset_frame = onset_context_frame
    if visual_onset_frame is None:
        visual_onset_frame = sim_onset_frame
    if onset_context_band:
        band = dict(onset_context_band)
    else:
        band = build_onset_band(
            start_f, end_f,
            onset_frame=visual_onset_frame,
            onset_status=sim_status,
        )
    local_win = build_local_onset_window(
        start_f, end_f,
        onset_frame=visual_onset_frame,
        onset_band=band,
    )
    global_p = aggregate_precomputed_feature_cache(
        cache, start_frame=start_f, end_frame=end_f, onset_band=band, top_k=5)
    local_p = aggregate_precomputed_feature_cache(
        cache, start_frame=local_win.get('start_frame', start_f),
        end_frame=local_win.get('end_frame', end_f), onset_band=None, top_k=5)
    if not global_p or not local_p: return None

    g_map = {str(it.get('label','')): float(it.get('score',0)) for it in global_p.get("candidates",[])}
    l_map = {str(it.get('label','')): float(it.get('score',0)) for it in local_p.get("candidates",[])}
    merged = {k: 0.65*g_map.get(k,0)+0.35*l_map.get(k,0) for k in set(g_map)|set(l_map)}

    current_verb_label = _norm_text(current_verb_label)
    noun_required = bool(ontology_noun_required(ontology, current_verb_label)) if current_verb_label else False
    noun_exists_prior = _noun_exists_prior_for_verb(ontology, current_verb_label) if current_verb_label else 0.5
    current_noun_id = _safe_int(current_noun_id)
    feat = [
        _bounded01(sim_onset_ratio, 0.5),          # [0] onset ratio state
        float(sim_has_onset),                      # [1] onset exists in state
        1.0 if sim_status=="confirmed" else 0.0,  # [2] onset confirmed
        1.0 if sim_status=="suggested" else 0.0,  # [3] onset suggested
        _bounded01(segment_len / 240.0),          # [4] normalized segment length
        0.0,                                      # [5] sparse evidence confirmed ratio
        0.0,                                      # [6] sparse evidence missing ratio
        1.0 if noun_required else 0.0,            # [7] noun-required prior from current verb
        1.0 if current_noun_id is not None else 0.0,  # [8] noun already exists in state
        float(noun_exists_prior),                 # [9] noun-exists prior from current verb
    ]
    feat.extend(float(merged.get(l,0)) for l in verb_labels)
    feat.extend(float(l_map.get(l,0)) for l in verb_labels)
    support_map = {int(k): float(v) for k, v in dict(support_map or {}).items()}
    onset_support_map = {int(k): float(v) for k, v in dict(onset_support_map or {}).items()}
    max_support = max([1.0] + list(support_map.values()))
    max_onset_support = max([1.0] + list(onset_support_map.values()))
    feat.extend(float(support_map.get(noun_idx, 0.0)) / float(max_support) for noun_idx in range(len(noun_list)))
    feat.extend(float(onset_support_map.get(noun_idx, 0.0)) / float(max_onset_support) for noun_idx in range(len(noun_list)))
    feat.extend(float(v) for v in (global_p.get("segment_feature") or []))
    feat.extend(float(v) for v in (local_p.get("segment_feature") or []))

    all_fields = ["functional_contact_onset", "verb", "noun_object_id"]
    supervised_fields = [str(v) for v in list(supervised_fields or all_fields) if str(v)]
    return json.dumps({
        "feature": feat,
        "targets": {
            "onset_ratio": float(gt_onset_ratio),
            "verb_index": int(v_idx),
            "noun_exists": 1.0 if n_idx >= 0 else 0.0,
            "noun_index": int(n_idx),
        },
        "meta": {
            "edited_fields": list(supervised_fields),
        },
    })


def main():
    parser = argparse.ArgumentParser(description="IMPACT_HOI Clean Ablation Training v8 (strict HOP, no handtrack side-channel)")
    parser.add_argument("--gt-root", type=str, default="paper_workspace/gt")
    parser.add_argument("--feature-root", type=str, default="analysis/videomae_cache")
    parser.add_argument("--handtracks-dir", type=str, default="runtime_artifacts")
    parser.add_argument("--train-list", type=str, default="train_list.txt")
    parser.add_argument("--verbs", type=str, default="test_files/verbs.txt")
    parser.add_argument("--nouns", type=str, default="test_files/nouns.txt")
    parser.add_argument("--ontology", type=str, default="test_files/verb_noun_ontology.csv")
    parser.add_argument("--output", type=str, default="cluster_train_adapter.pt")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()

    with open(args.verbs, 'r') as f:
        verb_labels = [l.strip() for l in f if l.strip()]
    with open(args.nouns, 'r') as f:
        noun_list = [l.strip() for l in f if l.strip()]
    ontology = HOIOntology.from_csv(args.ontology) if args.ontology and os.path.isfile(args.ontology) else None
    noun_name_to_id = {str(name): int(idx) for idx, name in enumerate(noun_list)}

    all_json = sorted(glob.glob(os.path.join(args.gt_root, "*.json")))
    if os.path.exists(args.train_list):
        with open(args.train_list, "r") as f:
            valid_stems = {line.strip() for line in f if line.strip()}
        json_files = [f for f in all_json if os.path.basename(f) in valid_stems]
        print(f"[*] Train split: {len(json_files)}/{len(all_json)} clips")
    else:
        json_files = all_json
        print(f"[*] No train_list, using ALL {len(all_json)} clips")

    random.seed(42)
    feedback_rows = []
    feat_dim = None
    ht_found = 0
    ht_missing = 0

    for gt_path in json_files:
        clip_id = os.path.basename(gt_path).replace("_hoi_bbox.json", "")
        npz_match = glob.glob(os.path.join(args.feature_root, f"*{clip_id}*.npz"))
        if not npz_match:
            print(f"[skip] Missing NPZ features for {clip_id}")
            continue
        cache = load_precomputed_feature_cache(npz_match[0])
        if not cache:
            print(f"[skip] Failed to load NPZ cache for {clip_id}")
            continue
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        raw_library = dict(gt_data.get("object_library") or {})
        object_library = {}
        for raw_id, info in raw_library.items():
            object_id = _safe_int(raw_id)
            if object_id is None:
                object_id = _safe_int((info or {}).get("object_id"))
            if object_id is None:
                continue
            object_library[int(object_id)] = dict(info or {})
        tracks = dict(gt_data.get("tracks") or {})
        bboxes_by_frame = _build_bboxes_by_frame(gt_data, object_library)
        frame_size = list(gt_data.get("frame_size") or [0, 0])
        frame_width = int(frame_size[0] if len(frame_size) > 0 else 0)
        frame_height = int(frame_size[1] if len(frame_size) > 1 else 0)
        handtrack_data = _load_handtrack_file(args.handtracks_dir, clip_id)
        if not handtrack_data:
            print(f"[warn] Missing handtrack JSON for {clip_id} (using Mode A fallback)")
        clip_candidate_cache = {}

        events = []
        if "hoi_events" in gt_data:
            for h in ["left_hand", "right_hand"]:
                for ev in gt_data["hoi_events"].get(h, []):
                    ev["_hand_key"] = h
                    events.append(ev)
        else:
            events = gt_data.get("actions", [])

        for ev in events:
            start_f = _safe_int(ev.get('start_frame'))
            end_f = _safe_int(ev.get('end_frame'))
            if start_f is None or end_f is None:
                print(f"  [skip] Invalid start/end frames in event of {clip_id}")
                continue
            if end_f < start_f: start_f, end_f = end_f, start_f
            segment_len = max(1, end_f - start_f)

            onset_f = _safe_int(ev.get('contact_onset_frame'))
            if onset_f is None: onset_f = start_f
            gt_onset_ratio = _bounded01((onset_f - start_f) / float(segment_len), 0.5)

            verb_label = _norm_text(ev.get('verb', ''))
            interaction = ev.get('interaction') or {}
            noun_object_id = _event_noun_id(ev, tracks, noun_name_to_id, object_library)
            noun_label = (
                _object_name_for_id(object_library, noun_object_id)
                or _norm_text(interaction.get('target', ev.get('target', '')))
                or _norm_text(interaction.get('noun', ''))
            )
            v_idx = verb_labels.index(verb_label) if verb_label in verb_labels else -1
            if noun_object_id is not None and 0 <= int(noun_object_id) < len(noun_list):
                n_idx = int(noun_object_id)
            else:
                n_idx = noun_list.index(noun_label) if noun_label in noun_list else -1
            base_support_map = {}
            base_onset_support_map = {}
            if handtrack_data and bboxes_by_frame:
                base_support_map, base_onset_support_map = _support_maps_for_state(
                    start_f=start_f,
                    end_f=end_f,
                    onset_f=None,
                    hand_key=ev.get("_hand_key", "right_hand"),
                    handtrack_data=handtrack_data,
                    bboxes_by_frame=bboxes_by_frame,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            onset_support_map = {}
            onset_onset_support_map = {}
            if handtrack_data and bboxes_by_frame:
                onset_support_map, onset_onset_support_map = _support_maps_for_state(
                    start_f=start_f,
                    end_f=end_f,
                    onset_f=onset_f,
                    hand_key=ev.get("_hand_key", "right_hand"),
                    handtrack_data=handtrack_data,
                    bboxes_by_frame=bboxes_by_frame,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )

            ht_prior = handtrack_segment_prior(
                handtrack_data,
                ev.get("_hand_key","right_hand"),
                start_f,
                end_f,
                cache=cache,
                verb_labels=verb_labels,
                candidate_cache=clip_candidate_cache,
            ) if handtrack_data else {}
            if ht_prior:
                ht_found += 1
            else:
                ht_missing += 1


            row = build_sample(
                cache, start_f, end_f,
                sim_onset_frame=None, sim_onset_ratio=0.5,
                sim_has_onset=0.0, sim_status="",
                onset_context_frame=None, onset_context_band=None,
                gt_onset_ratio=gt_onset_ratio,
                verb_labels=verb_labels, noun_list=noun_list,
                v_idx=v_idx, n_idx=n_idx,
                ontology=ontology,
                supervised_fields=["functional_contact_onset", "verb", "noun_object_id"],
            )
            if row:
                feedback_rows.append(row)
                if feat_dim is None:
                    feat_dim = len(json.loads(row)["feature"])


            row = build_sample(
                cache, start_f, end_f,
                sim_onset_frame=None, sim_onset_ratio=0.5,
                sim_has_onset=0.0, sim_status="",
                onset_context_frame=(ht_prior.get("onset_frame") if ht_prior else None),
                onset_context_band=(ht_prior.get("onset_band") if ht_prior else None),
                gt_onset_ratio=gt_onset_ratio,
                verb_labels=verb_labels, noun_list=noun_list,
                v_idx=v_idx, n_idx=n_idx,
                ontology=ontology,
                supervised_fields=["functional_contact_onset", "verb", "noun_object_id"],
            )
            if row:
                feedback_rows.append(row)

            row = build_sample(
                cache, start_f, end_f,
                sim_onset_frame=onset_f, sim_onset_ratio=gt_onset_ratio,
                sim_has_onset=1.0, sim_status="confirmed",
                onset_context_frame=None, onset_context_band=None,
                gt_onset_ratio=gt_onset_ratio,
                verb_labels=verb_labels, noun_list=noun_list,
                v_idx=v_idx, n_idx=n_idx,
                ontology=ontology,
                supervised_fields=["verb", "noun_object_id"],
            )
            if row:
                feedback_rows.append(row)

            row = build_sample(
                cache, start_f, end_f,
                sim_onset_frame=None, sim_onset_ratio=0.5,
                sim_has_onset=0.0, sim_status="",
                onset_context_frame=None, onset_context_band=None,
                gt_onset_ratio=gt_onset_ratio,
                verb_labels=verb_labels, noun_list=noun_list,
                v_idx=v_idx, n_idx=n_idx,
                support_map=base_support_map,
                onset_support_map=base_onset_support_map,
                ontology=ontology,
                supervised_fields=["functional_contact_onset", "verb", "noun_object_id"],
            )
            if row:
                feedback_rows.append(row)

            row = build_sample(
                cache, start_f, end_f,
                sim_onset_frame=None, sim_onset_ratio=0.5,
                sim_has_onset=0.0, sim_status="",
                onset_context_frame=(ht_prior.get("onset_frame") if ht_prior else None),
                onset_context_band=(ht_prior.get("onset_band") if ht_prior else None),
                gt_onset_ratio=gt_onset_ratio,
                verb_labels=verb_labels, noun_list=noun_list,
                v_idx=v_idx, n_idx=n_idx,
                support_map=base_support_map,
                onset_support_map=base_onset_support_map,
                ontology=ontology,
                supervised_fields=["functional_contact_onset", "verb", "noun_object_id"],
            )
            if row:
                feedback_rows.append(row)

            row = build_sample(
                cache, start_f, end_f,
                sim_onset_frame=onset_f, sim_onset_ratio=gt_onset_ratio,
                sim_has_onset=1.0, sim_status="confirmed",
                onset_context_frame=None, onset_context_band=None,
                gt_onset_ratio=gt_onset_ratio,
                verb_labels=verb_labels, noun_list=noun_list,
                v_idx=v_idx, n_idx=n_idx,
                support_map=onset_support_map,
                onset_support_map=onset_onset_support_map,
                ontology=ontology,
                supervised_fields=["verb", "noun_object_id"],
            )
            if row:
                feedback_rows.append(row)

            row = build_sample(
                cache, start_f, end_f,
                sim_onset_frame=onset_f, sim_onset_ratio=gt_onset_ratio,
                sim_has_onset=1.0, sim_status="confirmed",
                onset_context_frame=None, onset_context_band=None,
                gt_onset_ratio=gt_onset_ratio,
                verb_labels=verb_labels, noun_list=noun_list,
                v_idx=v_idx, n_idx=n_idx,
                current_verb_label=verb_label,
                ontology=ontology,
                supervised_fields=["noun_object_id"],
            )
            if row:
                feedback_rows.append(row)

            row = build_sample(
                cache, start_f, end_f,
                sim_onset_frame=onset_f, sim_onset_ratio=gt_onset_ratio,
                sim_has_onset=1.0, sim_status="confirmed",
                onset_context_frame=None, onset_context_band=None,
                gt_onset_ratio=gt_onset_ratio,
                verb_labels=verb_labels, noun_list=noun_list,
                v_idx=v_idx, n_idx=n_idx,
                current_verb_label=verb_label,
                support_map=onset_support_map,
                onset_support_map=onset_onset_support_map,
                ontology=ontology,
                supervised_fields=["noun_object_id"],
            )
            if row:
                feedback_rows.append(row)

    if not feedback_rows:
        print("[ERROR] No valid samples!")
        return

    print(f"[*] Handtrack coverage: {ht_found} found, {ht_missing} missing")
    log_path = args.output + ".jsonl"
    with open(log_path, "w") as f:
        for r in feedback_rows:
            f.write(r + "\n")

    print(f"[*] Training v8. Samples: {len(feedback_rows)} | Dim: {feat_dim}")
    ok, msg, _ = train_adapter_from_feedback(
        feedback_path=log_path, output_path=args.output,
        feature_dim=feat_dim, verb_labels=verb_labels,
        noun_ids=list(range(len(noun_list))),
        epochs=args.epochs, batch_size=32,
    )
    print(f"[*] Done: {ok} | {msg}")


if __name__ == "__main__":
    main()
