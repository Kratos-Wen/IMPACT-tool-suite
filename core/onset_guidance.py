from __future__ import annotations

from typing import Any, Dict, List, Optional


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _linspace_int(start: int, end: int, count: int) -> List[int]:
    count = max(1, int(count))
    if count == 1 or start == end:
        return [int(round((int(start) + int(end)) / 2.0))]
    step = float(end - start) / float(count - 1)
    return [int(round(start + idx * step)) for idx in range(count)]


def build_onset_band(
    start_frame: Any,
    end_frame: Any,
    *,
    onset_frame: Any = None,
    onset_status: str = "",
) -> Dict[str, Any]:
    start = _safe_int(start_frame)
    end = _safe_int(end_frame)
    if start is None or end is None:
        return {}
    if end < start:
        start, end = end, start

    onset = _safe_int(onset_frame)
    if onset is None:
        onset = int(round((start + end) / 2.0))
    onset = max(start, min(onset, end))

    span = max(0, end - start)
    status = str(onset_status or "").strip().lower()
    if status == "confirmed":
        ratio = 0.10
        confidence = 0.82
    elif status == "suggested":
        ratio = 0.18
        confidence = 0.62
    else:
        ratio = 0.22
        confidence = 0.46

    min_half_width = 1 if span <= 2 else 2
    half_width = int(round(max(min_half_width, float(span) * ratio)))
    half_width = min(max(1, span // 2 if span > 1 else 1), half_width)

    band_start = max(start, onset - half_width)
    band_end = min(end, onset + half_width)
    if band_end < band_start:
        band_start = band_end = onset

    return {
        "center_frame": int(onset),
        "start_frame": int(band_start),
        "end_frame": int(band_end),
        "width": int(max(1, band_end - band_start + 1)),
        "segment_start": int(start),
        "segment_end": int(end),
        "status": status or ("suggested" if onset_frame is not None else "missing"),
        "confidence_proxy": float(confidence),
    }


def build_temporal_sample_indices(
    start_frame: Any,
    end_frame: Any,
    *,
    num_samples: int,
    onset_band: Optional[Dict[str, Any]] = None,
) -> List[int]:
    start = _safe_int(start_frame)
    end = _safe_int(end_frame)
    if start is None or end is None:
        return []
    if end < start:
        start, end = end, start

    total = max(1, int(num_samples))
    if onset_band and _safe_int(onset_band.get("center_frame")) is not None:
        center = int(onset_band["center_frame"])
        local_start = _safe_int(onset_band.get("start_frame"))
        local_end = _safe_int(onset_band.get("end_frame"))
        if local_start is None or local_end is None:
            local_start = local_end = center
        local_start = max(start, min(local_start, end))
        local_end = max(start, min(local_end, end))
        if local_end < local_start:
            local_start, local_end = local_end, local_start
        if local_end - local_start < 2:
            local_start = max(start, center - 1)
            local_end = min(end, center + 1)

        local_count = max(4, total // 2)
        global_count = max(4, total - local_count)
        local_indices = _linspace_int(local_start, local_end, local_count)
        global_indices = _linspace_int(start, end, global_count)

        selected = []
        selected_set = set()
        for idx in list(local_indices) + list(global_indices):
            idx = max(start, min(int(idx), end))
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)

        if len(selected) < total:
            for idx in _linspace_int(start, end, total * 2):
                idx = max(start, min(int(idx), end))
                if idx not in selected_set:
                    selected.append(idx)
                    selected_set.add(idx)
                if len(selected) >= total:
                    break

        if len(selected) > total:
            local_set = {max(start, min(int(v), end)) for v in list(local_indices)}
            ranked = sorted(
                selected,
                key=lambda idx: (
                    0 if idx in local_set else 1,
                    abs(int(idx) - int(center)),
                    int(idx),
                ),
            )
            keep = sorted(ranked[:total])
            return [int(v) for v in keep]

        return [int(v) for v in sorted(selected)]

    return [int(v) for v in _linspace_int(start, end, total)]


def build_local_onset_window(
    start_frame: Any,
    end_frame: Any,
    *,
    onset_frame: Any = None,
    onset_band: Optional[Dict[str, Any]] = None,
    width_ratio: float = 0.16,
    min_width: int = 8,
) -> Dict[str, Any]:
    start = _safe_int(start_frame)
    end = _safe_int(end_frame)
    if start is None or end is None:
        return {}
    if end < start:
        start, end = end, start

    onset = _safe_int(onset_frame)
    if onset is None and onset_band:
        onset = _safe_int(onset_band.get("center_frame"))
    if onset is None:
        onset = int(round((start + end) / 2.0))
    onset = max(start, min(onset, end))

    span = max(1, end - start + 1)
    base_width = int(round(max(int(min_width), float(span) * float(width_ratio))))
    if onset_band:
        band_start = _safe_int(onset_band.get("start_frame"))
        band_end = _safe_int(onset_band.get("end_frame"))
        if band_start is not None and band_end is not None:
            band_width = max(1, abs(int(band_end) - int(band_start)) + 1)
            base_width = max(base_width, int(round(float(band_width) * 1.2)))
    base_width = max(3, min(int(span), int(base_width)))
    half = max(1, int(round((base_width - 1) / 2.0)))
    local_start = max(start, onset - half)
    local_end = min(end, onset + half)
    if local_end - local_start + 1 < base_width:
        deficit = base_width - (local_end - local_start + 1)
        extend_left = deficit // 2
        extend_right = deficit - extend_left
        local_start = max(start, local_start - extend_left)
        local_end = min(end, local_end + extend_right)
    return {
        "center_frame": int(onset),
        "start_frame": int(local_start),
        "end_frame": int(local_end),
        "width": int(max(1, local_end - local_start + 1)),
        "segment_start": int(start),
        "segment_end": int(end),
        "kind": "onset_local_window",
    }
