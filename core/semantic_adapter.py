from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


UNKNOWN_NOUN_ID = -1
UNKNOWN_NOUN_LABEL = "__UNKNOWN_NOUN__"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _ensure_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


class SemanticAdapterNet:  # pragma: no cover - thin wrapper around runtime torch model
    def __new__(
        cls,
        feature_dim: int,
        hidden_dim: int,
        verb_count: int,
        noun_count: int,
        onset_bins: int = 0,
        feature_layout: Optional[Dict[str, Any]] = None,
        video_adapter_rank: int = 0,
        video_adapter_alpha: float = 1.0,
    ):
        torch, nn, _F = _ensure_torch()
        normalized_feature_layout = _sanitize_feature_layout(
            feature_layout,
            feature_dim=int(feature_dim),
        )
        adapter_rank = max(0, int(video_adapter_rank))
        adapter_alpha = max(0.0, float(video_adapter_alpha or 0.0))

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.feature_dim = int(feature_dim)
                self.hidden_dim = int(hidden_dim)
                self.verb_count = int(verb_count)
                self.noun_count = int(noun_count)
                self.onset_bins = int(onset_bins)
                self.feature_layout = dict(normalized_feature_layout or {})
                self.video_adapter_rank = int(adapter_rank)
                self.video_adapter_alpha = float(adapter_alpha)
                self.videomae_feature_offset = int(
                    self.feature_layout.get("videomae_feature_offset") or 0
                )
                self.videomae_feature_dim = int(
                    self.feature_layout.get("videomae_feature_dim") or 0
                )
                self.videomae_local_feature_offset = int(
                    self.feature_layout.get("videomae_local_feature_offset") or 0
                )
                self.videomae_local_feature_dim = int(
                    self.feature_layout.get("videomae_local_feature_dim") or 0
                )
                if (
                    self.video_adapter_rank > 0
                    and self.video_adapter_alpha > 0.0
                    and self.videomae_feature_dim > 0
                ):
                    self.videomae_feature_norm = nn.LayerNorm(self.videomae_feature_dim)
                    self.videomae_feature_down = nn.Linear(
                        self.videomae_feature_dim,
                        self.video_adapter_rank,
                        bias=False,
                    )
                    self.videomae_feature_up = nn.Linear(
                        self.video_adapter_rank,
                        self.videomae_feature_dim,
                        bias=False,
                    )
                    nn.init.kaiming_uniform_(
                        self.videomae_feature_down.weight,
                        a=math.sqrt(5.0),
                    )
                    nn.init.zeros_(self.videomae_feature_up.weight)
                else:
                    self.videomae_feature_norm = None
                    self.videomae_feature_down = None
                    self.videomae_feature_up = None
                if (
                    self.video_adapter_rank > 0
                    and self.video_adapter_alpha > 0.0
                    and self.videomae_local_feature_dim > 0
                ):
                    self.videomae_local_feature_norm = nn.LayerNorm(
                        self.videomae_local_feature_dim
                    )
                    self.videomae_local_feature_down = nn.Linear(
                        self.videomae_local_feature_dim,
                        self.video_adapter_rank,
                        bias=False,
                    )
                    self.videomae_local_feature_up = nn.Linear(
                        self.video_adapter_rank,
                        self.videomae_local_feature_dim,
                        bias=False,
                    )
                    nn.init.kaiming_uniform_(
                        self.videomae_local_feature_down.weight,
                        a=math.sqrt(5.0),
                    )
                    nn.init.zeros_(self.videomae_local_feature_up.weight)
                else:
                    self.videomae_local_feature_norm = None
                    self.videomae_local_feature_down = None
                    self.videomae_local_feature_up = None
                self.trunk = nn.Sequential(
                    nn.Linear(self.feature_dim, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.08),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                )
                self.onset_mean = nn.Linear(self.hidden_dim, 1)
                self.onset_logvar = nn.Linear(self.hidden_dim, 1)
                self.onset_bin_head = (
                    nn.Linear(self.hidden_dim, self.onset_bins)
                    if self.onset_bins > 1
                    else None
                )
                self.verb_head = (
                    nn.Linear(self.hidden_dim, self.verb_count)
                    if self.verb_count > 0
                    else None
                )
                self.noun_exist_head = nn.Linear(self.hidden_dim, 1)
                self.noun_head = (
                    nn.Linear(self.hidden_dim, self.noun_count)
                    if self.noun_count > 0
                    else None
                )

            def _apply_low_rank_video_adapter(self, chunk, norm_layer, down_layer, up_layer):
                if (
                    chunk is None
                    or chunk.numel() <= 0
                    or norm_layer is None
                    or down_layer is None
                    or up_layer is None
                ):
                    return chunk
                residual = up_layer(torch.tanh(down_layer(norm_layer(chunk))))
                scale = float(self.video_adapter_alpha) / float(
                    max(1, self.video_adapter_rank)
                )
                return chunk + residual * scale

            def _adapt_video_features(self, features):
                if (
                    features is None
                    or features.ndim != 2
                    or self.video_adapter_rank <= 0
                    or self.video_adapter_alpha <= 0.0
                ):
                    return features
                specs = []
                if (
                    self.videomae_feature_dim > 0
                    and self.videomae_feature_down is not None
                    and self.videomae_feature_up is not None
                ):
                    specs.append(
                        (
                            "global",
                            int(self.videomae_feature_offset),
                            int(self.videomae_feature_dim),
                        )
                    )
                if (
                    self.videomae_local_feature_dim > 0
                    and self.videomae_local_feature_down is not None
                    and self.videomae_local_feature_up is not None
                ):
                    specs.append(
                        (
                            "local",
                            int(self.videomae_local_feature_offset),
                            int(self.videomae_local_feature_dim),
                        )
                    )
                if not specs:
                    return features
                specs.sort(key=lambda row: int(row[1]))
                segments = []
                cursor = 0
                feature_cols = int(features.shape[-1])
                for name, offset, dim in specs:
                    offset = max(0, min(int(offset), feature_cols))
                    dim = max(0, min(int(dim), feature_cols - offset))
                    if dim <= 0 or offset < cursor:
                        continue
                    if offset > cursor:
                        segments.append(features[:, cursor:offset])
                    chunk = features[:, offset : offset + dim]
                    if name == "global":
                        adapted = self._apply_low_rank_video_adapter(
                            chunk,
                            self.videomae_feature_norm,
                            self.videomae_feature_down,
                            self.videomae_feature_up,
                        )
                    else:
                        adapted = self._apply_low_rank_video_adapter(
                            chunk,
                            self.videomae_local_feature_norm,
                            self.videomae_local_feature_down,
                            self.videomae_local_feature_up,
                        )
                    segments.append(adapted)
                    cursor = offset + dim
                if cursor < feature_cols:
                    segments.append(features[:, cursor:])
                if not segments:
                    return features
                adapted = (
                    torch.cat(segments, dim=-1) if len(segments) > 1 else segments[0]
                )
                return adapted if int(adapted.shape[-1]) == feature_cols else features

            def forward(self, features):
                adapted_features = self._adapt_video_features(features)
                h = self.trunk(adapted_features)
                out = {
                    "onset_mean": torch.sigmoid(self.onset_mean(h)).squeeze(-1),
                    "onset_logvar": torch.clamp(
                        self.onset_logvar(h).squeeze(-1), min=-6.0, max=3.0
                    ),
                    "noun_exist_logit": self.noun_exist_head(h).squeeze(-1),
                }
                if self.onset_bin_head is not None:
                    out["onset_bin_logits"] = self.onset_bin_head(h)
                if self.verb_head is not None:
                    out["verb_logits"] = self.verb_head(h)
                if self.noun_head is not None:
                    out["noun_logits"] = self.noun_head(h)
                return out

        return _Net()


@dataclass
class SemanticAdapterPackage:
    feature_dim: int
    hidden_dim: int
    verb_labels: List[str]
    noun_ids: List[int]
    state_dict: Dict[str, Any]
    onset_bins: int = 0
    feature_layout: Dict[str, int] = field(default_factory=dict)
    video_adapter_rank: int = 0
    video_adapter_alpha: float = 0.0
    sample_count: int = 0
    epochs: int = 0
    mean_loss: float = 0.0
    trained_at: str = ""
    supports_unknown_noun: bool = True
    unknown_noun_id: int = UNKNOWN_NOUN_ID
    calibration: Dict[str, Any] = field(default_factory=dict)
    structured_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_dim": int(self.feature_dim),
            "hidden_dim": int(self.hidden_dim),
            "verb_labels": list(self.verb_labels or []),
            "noun_ids": [int(v) for v in list(self.noun_ids or [])],
            "state_dict": dict(self.state_dict or {}),
            "onset_bins": int(self.onset_bins),
            "feature_layout": {
                str(k): int(v)
                for k, v in dict(self.feature_layout or {}).items()
            },
            "video_adapter_rank": int(self.video_adapter_rank),
            "video_adapter_alpha": float(self.video_adapter_alpha),
            "sample_count": int(self.sample_count),
            "epochs": int(self.epochs),
            "mean_loss": float(self.mean_loss),
            "trained_at": str(self.trained_at or ""),
            "supports_unknown_noun": bool(self.supports_unknown_noun),
            "unknown_noun_id": int(self.unknown_noun_id),
            "calibration": dict(self.calibration or {}),
            "structured_stats": dict(self.structured_stats or {}),
        }


def _load_feedback_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path or not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_adapter_package(path: str) -> Optional[SemanticAdapterPackage]:
    if not path or not os.path.isfile(path):
        return None
    torch, _nn, _F = _ensure_torch()
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        return None
    return SemanticAdapterPackage(
        feature_dim=int(payload.get("feature_dim") or 0),
        hidden_dim=int(payload.get("hidden_dim") or 0),
        verb_labels=[str(v) for v in list(payload.get("verb_labels") or [])],
        noun_ids=[int(v) for v in list(payload.get("noun_ids") or [])],
        state_dict=dict(payload.get("state_dict") or {}),
        onset_bins=int(payload.get("onset_bins") or 0),
        feature_layout={
            str(k): int(v)
            for k, v in dict(payload.get("feature_layout") or {}).items()
        },
        video_adapter_rank=int(payload.get("video_adapter_rank") or 0),
        video_adapter_alpha=float(payload.get("video_adapter_alpha") or 0.0),
        sample_count=int(payload.get("sample_count") or 0),
        epochs=int(payload.get("epochs") or 0),
        mean_loss=float(payload.get("mean_loss") or 0.0),
        trained_at=str(payload.get("trained_at") or ""),
        supports_unknown_noun=bool(payload.get("supports_unknown_noun", True)),
        unknown_noun_id=int(payload.get("unknown_noun_id", UNKNOWN_NOUN_ID)),
        calibration=dict(payload.get("calibration") or {}),
        structured_stats=dict(payload.get("structured_stats") or {}),
    )


def _safe_log_prob(value: float, floor: float = 1e-6) -> float:
    return math.log(max(float(floor), float(value)))


def _sanitize_feature_layout(
    feature_layout: Optional[Dict[str, Any]],
    *,
    feature_dim: int,
) -> Dict[str, int]:
    raw = dict(feature_layout or {})
    normalized: Dict[str, int] = {}
    keys = (
        "scalar_dim",
        "global_verb_offset",
        "global_verb_dim",
        "local_verb_offset",
        "local_verb_dim",
        "noun_support_offset",
        "noun_support_dim",
        "noun_onset_support_offset",
        "noun_onset_support_dim",
        "videomae_feature_offset",
        "videomae_feature_dim",
        "videomae_local_feature_offset",
        "videomae_local_feature_dim",
    )
    feature_dim = max(0, int(feature_dim))
    for key in keys:
        normalized[str(key)] = max(0, _safe_int(raw.get(key), 0))
    for offset_key, dim_key in (
        ("videomae_feature_offset", "videomae_feature_dim"),
        ("videomae_local_feature_offset", "videomae_local_feature_dim"),
    ):
        offset = max(0, min(int(normalized.get(offset_key) or 0), feature_dim))
        dim = max(0, min(int(normalized.get(dim_key) or 0), feature_dim - offset))
        normalized[offset_key] = int(offset)
        normalized[dim_key] = int(dim)
    return normalized


def _ratio_to_bin_index(value: float, bins: int) -> int:
    bins = max(2, int(bins))
    return int(round(_clamp(value) * float(bins - 1)))


def _bin_index_to_ratio(index: int, bins: int) -> float:
    bins = max(2, int(bins))
    index = max(0, min(int(index), bins - 1))
    return float(index) / float(max(1, bins - 1))


def _prepare_rows(
    rows: Sequence[Dict[str, Any]],
    feature_dim: int,
) -> List[Dict[str, Any]]:
    def _normalize_feature_length(values: Sequence[Any], target_dim: int) -> List[float]:
        target_dim = max(0, int(target_dim))
        src = list(values or [])
        out: List[float] = []
        for idx in range(target_dim):
            if idx < len(src):
                out.append(_safe_float(src[idx], 0.0))
            else:
                out.append(0.0)
        return out

    valid_rows: List[Dict[str, Any]] = []
    for row in list(rows or []):
        feature = row.get("feature")
        targets = row.get("targets")
        if not isinstance(feature, list) or not isinstance(targets, dict):
            continue
        normalized = dict(row)
        normalized["feature"] = _normalize_feature_length(feature, int(feature_dim))
        valid_rows.append(normalized)
    return valid_rows


def _split_feedback_rows(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    valid_rows = list(rows or [])
    if len(valid_rows) < 10:
        return valid_rows, valid_rows
    calib_count = max(2, min(32, int(round(len(valid_rows) * 0.2))))
    if len(valid_rows) - calib_count < 4:
        return valid_rows, valid_rows
    return valid_rows[:-calib_count], valid_rows[-calib_count:]


def _row_supervision_weight(
    row: Dict[str, Any],
    field_names: Sequence[str],
) -> float:
    meta = dict((row or {}).get("meta") or {})
    edited = {str(v) for v in list(meta.get("edited_fields") or []) if str(v)}
    accepted = {str(v) for v in list(meta.get("accepted_fields") or []) if str(v)}
    untouched = {str(v) for v in list(meta.get("untouched_fields") or []) if str(v)}
    supervision_kind = str(meta.get("supervision_kind") or "").strip().lower()
    fields = {str(v) for v in list(field_names or []) if str(v)}
    if fields & edited:
        return 1.0
    if fields & accepted:
        return 0.65
    if fields & untouched:
        return 0.20
    if supervision_kind == "accepted":
        return 0.18
    if supervision_kind == "edited":
        return 0.30
    return 0.10


def _row_exclusive_anchor(row: Dict[str, Any]) -> Tuple[Optional[float], float]:
    meta = dict((row or {}).get("meta") or {})
    cross = dict(meta.get("cross_hand_context") or {})
    primary = dict(cross.get("primary_exclusion") or {})
    ratio = primary.get("onset_ratio")
    try:
        ratio = float(ratio) if ratio is not None else None
    except Exception:
        ratio = None
    weight = 0.0
    try:
        weight = float(primary.get("exclude_weight", 0.0) or 0.0)
    except Exception:
        weight = 0.0
    if ratio is None:
        return None, 0.0
    return _clamp(ratio), _clamp(weight, 0.0, 0.35)


def _tensorize_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    feature_dim: int,
    noun_count: int,
    include_unknown_noun: bool,
):
    torch, _nn, _F = _ensure_torch()
    device = torch.device("cpu")
    features = torch.tensor(
        [[_safe_float(v) for v in row.get("feature", [])] for row in list(rows or [])],
        dtype=torch.float32,
        device=device,
    )
    onset_targets = torch.tensor(
        [
            _clamp(_safe_float((row.get("targets") or {}).get("onset_ratio"), 0.5))
            for row in list(rows or [])
        ],
        dtype=torch.float32,
        device=device,
    )
    verb_targets = torch.tensor(
        [
            _safe_int((row.get("targets") or {}).get("verb_index"), -1)
            for row in list(rows or [])
        ],
        dtype=torch.long,
        device=device,
    )
    noun_exist_targets = torch.tensor(
        [
            _clamp(_safe_float((row.get("targets") or {}).get("noun_exists"), 0.0))
            for row in list(rows or [])
        ],
        dtype=torch.float32,
        device=device,
    )
    unknown_idx = int(noun_count) if include_unknown_noun else -1
    noun_targets_list: List[int] = []
    for row in list(rows or []):
        targets = row.get("targets") or {}
        noun_exists = _safe_float(targets.get("noun_exists"), 0.0) >= 0.5
        noun_index = _safe_int(targets.get("noun_index"), -1)
        if noun_exists and noun_index < 0 and include_unknown_noun:
            noun_index = unknown_idx
        noun_targets_list.append(noun_index)
    noun_targets = torch.tensor(noun_targets_list, dtype=torch.long, device=device)
    onset_weights = torch.tensor(
        [
            _row_supervision_weight(row, ("functional_contact_onset",))
            for row in list(rows or [])
        ],
        dtype=torch.float32,
        device=device,
    )
    verb_weights = torch.tensor(
        [
            _row_supervision_weight(row, ("verb",))
            for row in list(rows or [])
        ],
        dtype=torch.float32,
        device=device,
    )
    noun_weights = torch.tensor(
        [
            _row_supervision_weight(row, ("noun_object_id",))
            for row in list(rows or [])
        ],
        dtype=torch.float32,
        device=device,
    )
    exclusive_anchor = torch.tensor(
        [
            float(_row_exclusive_anchor(row)[0])
            if _row_exclusive_anchor(row)[0] is not None
            else -1.0
            for row in list(rows or [])
        ],
        dtype=torch.float32,
        device=device,
    )
    exclusive_weight = torch.tensor(
        [float(_row_exclusive_anchor(row)[1]) for row in list(rows or [])],
        dtype=torch.float32,
        device=device,
    )
    return (
        device,
        features,
        onset_targets,
        verb_targets,
        noun_exist_targets,
        noun_targets,
        onset_weights,
        verb_weights,
        noun_weights,
        exclusive_anchor,
        exclusive_weight,
    )


def _predict_raw(model, features):
    model.eval()
    with __import__("torch").no_grad():
        out = model(features)
    return out


def _fit_multiclass_temperature(logits, labels) -> float:
    torch, _nn, F = _ensure_torch()
    mask = labels >= 0
    if not bool(torch.any(mask)):
        return 1.0
    logits = logits[mask].detach()
    labels = labels[mask].detach()
    if logits.shape[0] < 4:
        return 1.0
    log_temp = torch.zeros(1, dtype=torch.float32, device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temp], lr=0.25, max_iter=50)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        temp = torch.exp(log_temp).clamp(min=0.05, max=10.0)
        loss = F.cross_entropy(logits / temp, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temp.detach()).cpu().item())


def _fit_binary_temperature(logits, labels) -> float:
    torch, _nn, F = _ensure_torch()
    logits = logits.detach()
    labels = labels.detach()
    if logits.shape[0] < 4:
        return 1.0
    log_temp = torch.zeros(1, dtype=torch.float32, device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temp], lr=0.25, max_iter=50)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        temp = torch.exp(log_temp).clamp(min=0.05, max=10.0)
        loss = F.binary_cross_entropy_with_logits(logits / temp, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temp.detach()).cpu().item())


def _softmax_with_temperature(logits, temperature: float) -> List[float]:
    torch, _nn, F = _ensure_torch()
    temp = max(0.05, float(temperature or 1.0))
    probs = F.softmax(logits / temp, dim=-1).detach().cpu().tolist()
    return [float(v) for v in list(probs or [])]


def _sigmoid_with_temperature(logit, temperature: float) -> float:
    torch, _nn, _F = _ensure_torch()
    temp = max(0.05, float(temperature or 1.0))
    return float(torch.sigmoid(logit / temp).detach().cpu().item())


def _best_binary_threshold(probs: Sequence[float], labels: Sequence[float]) -> float:
    if not probs:
        return 0.5
    best_threshold = 0.5
    best_score = -1.0
    for threshold in [i / 20.0 for i in range(2, 19)]:
        tp = fp = fn = 0.0
        for prob, label in zip(list(probs or []), list(labels or [])):
            pred = 1.0 if float(prob) >= threshold else 0.0
            truth = 1.0 if float(label) >= 0.5 else 0.0
            if pred >= 0.5 and truth >= 0.5:
                tp += 1.0
            elif pred >= 0.5 and truth < 0.5:
                fp += 1.0
            elif pred < 0.5 and truth >= 0.5:
                fn += 1.0
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        f1 = 0.0 if precision + recall <= 0.0 else (2.0 * precision * recall) / (precision + recall)
        if f1 > best_score:
            best_score = f1
            best_threshold = float(threshold)
    return float(best_threshold)


def _quantile(values: Sequence[float], q: float, default: float) -> float:
    seq = sorted(float(v) for v in list(values or []))
    if not seq:
        return float(default)
    q = _clamp(float(q), 0.0, 1.0)
    pos = q * float(max(0, len(seq) - 1))
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(seq[lo])
    alpha = pos - float(lo)
    return float(seq[lo] * (1.0 - alpha) + seq[hi] * alpha)


def _collect_structured_stats(
    rows: Sequence[Dict[str, Any]],
    *,
    verb_labels: Sequence[str],
    noun_ids: Sequence[int],
) -> Dict[str, Any]:
    verb_stats: Dict[str, Dict[str, Any]] = {}
    noun_stats: Dict[str, Dict[str, Any]] = {}
    noun_bonus: Dict[str, Dict[str, float]] = {}
    noun_to_verb_bonus: Dict[str, Dict[str, float]] = {}
    pair_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
    noun_keys = [str(int(v)) for v in list(noun_ids or [])]
    for row in list(rows or []):
        targets = row.get("targets") or {}
        verb_index = _safe_int(targets.get("verb_index"), -1)
        if verb_index < 0 or verb_index >= len(list(verb_labels or [])):
            continue
        verb_name = str(list(verb_labels or [])[verb_index])
        onset_ratio = _clamp(_safe_float(targets.get("onset_ratio"), 0.5))
        noun_exists = 1.0 if _safe_float(targets.get("noun_exists"), 0.0) >= 0.5 else 0.0
        noun_index = _safe_int(targets.get("noun_index"), -1)
        stat = verb_stats.setdefault(
            verb_name,
            {
                "count": 0,
                "onset_values": [],
                "no_noun_count": 0,
            },
        )
        stat["count"] += 1
        stat["onset_values"].append(onset_ratio)
        if noun_exists < 0.5:
            stat["no_noun_count"] += 1
        noun_key = "__NO_NOUN__"
        if noun_exists >= 0.5 and 0 <= noun_index < len(noun_keys):
            noun_key = noun_keys[noun_index]
        elif noun_exists >= 0.5:
            noun_key = "__UNKNOWN__"
        noun_bonus.setdefault(verb_name, {})
        noun_bonus[verb_name][noun_key] = noun_bonus[verb_name].get(noun_key, 0.0) + 1.0
        noun_to_verb_bonus.setdefault(noun_key, {})
        noun_to_verb_bonus[noun_key][verb_name] = (
            noun_to_verb_bonus[noun_key].get(verb_name, 0.0) + 1.0
        )
        noun_stat = noun_stats.setdefault(
            noun_key,
            {
                "count": 0,
                "onset_values": [],
            },
        )
        noun_stat["count"] += 1
        noun_stat["onset_values"].append(onset_ratio)
        pair_stat = pair_stats.setdefault(verb_name, {}).setdefault(
            noun_key,
            {
                "count": 0,
                "onset_values": [],
            },
        )
        pair_stat["count"] += 1
        pair_stat["onset_values"].append(onset_ratio)

    out_priors: Dict[str, Dict[str, Any]] = {}
    out_noun_priors: Dict[str, Dict[str, Any]] = {}
    out_bonus: Dict[str, Dict[str, float]] = {}
    out_noun_bonus: Dict[str, Dict[str, float]] = {}
    out_pair_priors: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for verb_name, stat in verb_stats.items():
        values = [float(v) for v in list(stat.get("onset_values") or [])]
        mean = sum(values) / max(1, len(values))
        variance = sum((v - mean) ** 2 for v in values) / max(1, len(values))
        out_priors[verb_name] = {
            "count": int(stat.get("count") or 0),
            "mean": float(mean),
            "std": float(max(0.04, math.sqrt(max(0.0, variance)))),
            "no_noun_rate": float(stat.get("no_noun_count", 0) or 0) / float(max(1, stat.get("count", 0) or 0)),
        }
        counts = noun_bonus.get(verb_name) or {}
        total = float(sum(counts.values()) or 0.0)
        smooth_den = total + float(len(counts) + 1)
        out_bonus[verb_name] = {
            str(noun_key): float(math.log((float(count) + 1.0) / max(1.0, smooth_den)))
            for noun_key, count in counts.items()
        }
        pair_rows = dict(pair_stats.get(verb_name) or {})
        if pair_rows:
            out_pair_priors[verb_name] = {}
            for noun_key, pair_stat in pair_rows.items():
                pair_values = [float(v) for v in list(pair_stat.get("onset_values") or [])]
                pair_mean = sum(pair_values) / max(1, len(pair_values))
                pair_variance = sum((v - pair_mean) ** 2 for v in pair_values) / max(1, len(pair_values))
                out_pair_priors[verb_name][str(noun_key)] = {
                    "count": int(pair_stat.get("count") or 0),
                    "mean": float(pair_mean),
                    "std": float(max(0.04, math.sqrt(max(0.0, pair_variance)))),
                }
    for noun_key, stat in noun_stats.items():
        values = [float(v) for v in list(stat.get("onset_values") or [])]
        mean = sum(values) / max(1, len(values))
        variance = sum((v - mean) ** 2 for v in values) / max(1, len(values))
        out_noun_priors[str(noun_key)] = {
            "count": int(stat.get("count") or 0),
            "mean": float(mean),
            "std": float(max(0.04, math.sqrt(max(0.0, variance)))),
        }
        counts = dict(noun_to_verb_bonus.get(noun_key) or {})
        total = float(sum(counts.values()) or 0.0)
        smooth_den = total + float(len(counts) + 1)
        out_noun_bonus[str(noun_key)] = {
            str(verb_name): float(math.log((float(count) + 1.0) / max(1.0, smooth_den)))
            for verb_name, count in counts.items()
        }
    return {
        "verb_onset_priors": out_priors,
        "verb_noun_bonus": out_bonus,
        "noun_onset_priors": out_noun_priors,
        "noun_verb_bonus": out_noun_bonus,
        "verb_noun_onset_priors": out_pair_priors,
    }


def _normalize_prob_map(raw_scores: Dict[Any, float]) -> Dict[Any, float]:
    cleaned = {k: max(0.0, float(v)) for k, v in dict(raw_scores or {}).items()}
    total = float(sum(cleaned.values()) or 0.0)
    if total <= 0.0:
        return {k: 0.0 for k in cleaned}
    return {k: float(v) / total for k, v in cleaned.items()}


def _blend_prob_maps(
    primary: Dict[Any, float],
    secondary: Optional[Dict[Any, float]] = None,
    *,
    primary_weight: float = 0.75,
) -> Dict[Any, float]:
    secondary = dict(secondary or {})
    keys = set(primary.keys()) | set(secondary.keys())
    if not keys:
        return {}
    primary_weight = _clamp(float(primary_weight), 0.0, 1.0)
    secondary_weight = 1.0 - primary_weight
    merged = {
        key: float(primary_weight) * float(primary.get(key, 0.0))
        + float(secondary_weight) * float(secondary.get(key, 0.0))
        for key in keys
    }
    return _normalize_prob_map(merged)


def _blend_scalar(
    primary: float,
    secondary: float,
    *,
    primary_weight: float = 0.75,
) -> float:
    primary_weight = _clamp(float(primary_weight), 0.0, 1.0)
    secondary_weight = 1.0 - primary_weight
    return _clamp(
        float(primary_weight) * float(primary)
        + float(secondary_weight) * float(secondary)
    )


def _log_bonus_to_prob_map(
    raw_scores: Optional[Dict[Any, float]],
    *,
    allowed_keys: Optional[Iterable[Any]] = None,
) -> Dict[Any, float]:
    scores = dict(raw_scores or {})
    if not scores:
        return {}
    if allowed_keys is None:
        keys = list(scores.keys())
    else:
        keys = [key for key in list(allowed_keys or []) if key in scores]
    if not keys:
        return {}
    merged: Dict[Any, float] = {}
    for key in keys:
        merged[key] = float(
            math.exp(float(scores.get(key, math.log(1e-6))))
        )
    return _normalize_prob_map(merged)


def _top_prob_entries(
    prob_map: Dict[Any, float],
    *,
    top_k: int = 3,
) -> List[Tuple[Any, float]]:
    rows = sorted(
        [
            (key, float(value))
            for key, value in dict(prob_map or {}).items()
            if float(value) > 0.0
        ],
        key=lambda row: float(row[1]),
        reverse=True,
    )
    return rows[: max(1, int(top_k))]


def _build_onset_distribution(
    onset_ratio: float,
    onset_std: float,
    *,
    bins: int = 21,
) -> List[Dict[str, float]]:
    std = max(0.02, float(onset_std))
    scores: List[float] = []
    ratios: List[float] = []
    for idx in range(max(3, int(bins))):
        ratio = float(idx) / float(max(1, bins - 1))
        ratios.append(ratio)
        z = (ratio - float(onset_ratio)) / std
        scores.append(math.exp(-0.5 * (z ** 2)))
    total = float(sum(scores) or 1.0)
    return [
        {"ratio": float(ratio), "score": float(score) / total}
        for ratio, score in zip(ratios, scores)
    ]


def _normalize_onset_distribution(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, float]]:
    clean: List[Dict[str, float]] = []
    total = 0.0
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        ratio = _clamp(_safe_float(row.get("ratio"), 0.5))
        score = max(0.0, _safe_float(row.get("score"), 0.0))
        clean.append({"ratio": float(ratio), "score": float(score)})
        total += float(score)
    if not clean:
        return [{"ratio": 0.5, "score": 1.0}]
    if total <= 0.0:
        uniform = 1.0 / float(len(clean))
        return [{"ratio": float(row["ratio"]), "score": float(uniform)} for row in clean]
    return [{"ratio": float(row["ratio"]), "score": float(row["score"]) / total} for row in clean]


def _blend_onset_distributions(
    primary: Sequence[Dict[str, Any]],
    secondary: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    primary_weight: float = 0.6,
) -> List[Dict[str, float]]:
    left = _normalize_onset_distribution(primary)
    right = _normalize_onset_distribution(secondary or [])
    key_scores: Dict[float, float] = {}
    primary_weight = _clamp(float(primary_weight), 0.0, 1.0)
    secondary_weight = 1.0 - primary_weight
    for row in left:
        key = float(row["ratio"])
        key_scores[key] = key_scores.get(key, 0.0) + primary_weight * float(row["score"])
    for row in right:
        key = float(row["ratio"])
        key_scores[key] = key_scores.get(key, 0.0) + secondary_weight * float(row["score"])
    return _normalize_onset_distribution(
        [{"ratio": float(k), "score": float(v)} for k, v in sorted(key_scores.items())]
    )


def _onset_expectation(rows: Sequence[Dict[str, Any]], default: float = 0.5) -> float:
    seq = _normalize_onset_distribution(rows)
    if not seq:
        return _clamp(default)
    return _clamp(sum(float(row["ratio"]) * float(row["score"]) for row in seq))


def _top_onset_candidates(
    rows: Sequence[Dict[str, Any]],
    *,
    top_k: int = 3,
) -> List[Dict[str, float]]:
    seq = sorted(
        _normalize_onset_distribution(rows),
        key=lambda row: (float(row.get("score") or 0.0), -abs(float(row.get("ratio") or 0.5) - 0.5)),
        reverse=True,
    )
    out: List[Dict[str, float]] = []
    seen: set = set()
    for row in seq:
        key = round(float(row.get("ratio") or 0.5), 4)
        if key in seen:
            continue
        seen.add(key)
        out.append({"ratio": float(row["ratio"]), "score": float(row["score"])})
        if len(out) >= max(1, int(top_k)):
            break
    if not out:
        out = [{"ratio": 0.5, "score": 1.0}]
    return out


def _onset_distribution_quantile(
    rows: Sequence[Dict[str, Any]],
    q: float,
    *,
    default: float = 0.5,
) -> float:
    seq = sorted(
        _normalize_onset_distribution(rows),
        key=lambda row: float(row.get("ratio") or 0.5),
    )
    if not seq:
        return _clamp(default)
    q = _clamp(float(q), 0.0, 1.0)
    cumulative = 0.0
    for row in seq:
        cumulative += float(row.get("score") or 0.0)
        if cumulative >= q:
            return _clamp(float(row.get("ratio") or default))
    return _clamp(float(seq[-1].get("ratio") or default))


def _distribution_half_width(
    rows: Sequence[Dict[str, Any]],
    *,
    center: float,
    default: float,
) -> float:
    left = _onset_distribution_quantile(rows, 0.16, default=float(center))
    right = _onset_distribution_quantile(rows, 0.84, default=float(center))
    derived = max(
        0.03,
        max(float(center) - float(left), float(right) - float(center)),
    )
    return float(max(0.03, 0.55 * float(default) + 0.45 * float(derived)))


def _weighted_onset_prior_distribution(
    rows: Sequence[Tuple[float, Dict[str, Any]]],
    *,
    bins: int,
) -> List[Dict[str, float]]:
    merged: List[Dict[str, float]] = []
    total_weight = 0.0
    for weight, prior in list(rows or []):
        weight = max(0.0, float(weight))
        if weight <= 1e-6 or not isinstance(prior, dict):
            continue
        mean = _clamp(_safe_float(prior.get("mean"), 0.5))
        std = max(0.04, _safe_float(prior.get("std"), 0.12))
        dist = _build_onset_distribution(mean, std, bins=max(9, int(bins)))
        for row in dist:
            merged.append(
                {
                    "ratio": float(row.get("ratio") or 0.5),
                    "score": float(row.get("score") or 0.0) * weight,
                }
            )
        total_weight += weight
    if total_weight <= 1e-6:
        return []
    return _normalize_onset_distribution(merged)


def _cooperative_refine_predictions(
    *,
    package: SemanticAdapterPackage,
    onset_distribution: Sequence[Dict[str, Any]],
    verb_probs: Dict[str, float],
    noun_exist_prob: float,
    noun_probs: Dict[int, float],
    unknown_noun_prob: float,
    noun_required: bool,
    allow_no_noun: bool,
    allowed_noun_ids: Iterable[int],
    allow_no_noun_by_verb: Optional[Dict[str, bool]] = None,
    clamp_onset_ratio: Optional[float] = None,
    clamp_verb_label: str = "",
    clamp_noun_exists: Optional[bool] = None,
    clamp_noun_object_id: Optional[int] = None,
    clamp_unknown_noun: bool = False,
) -> Dict[str, Any]:
    stats = dict(getattr(package, "structured_stats", {}) or {})
    verb_onset_priors = dict(stats.get("verb_onset_priors") or {})
    noun_onset_priors = dict(stats.get("noun_onset_priors") or {})
    verb_noun_onset_priors = dict(stats.get("verb_noun_onset_priors") or {})
    verb_noun_bonus = dict(stats.get("verb_noun_bonus") or {})
    noun_verb_bonus = dict(stats.get("noun_verb_bonus") or {})
    refined_onset_distribution = _normalize_onset_distribution(onset_distribution)
    refined_verb_probs = dict(verb_probs or {})
    refined_noun_probs = dict(noun_probs or {})
    refined_noun_exist_prob = float(noun_exist_prob)
    refined_unknown_noun_prob = float(unknown_noun_prob)
    cooperation_meta: Dict[str, Any] = {
        "enabled": False,
        "noun_exists_prior": None,
        "semantic_onset_prior_used": False,
        "verb_refined_from_noun": False,
        "noun_refined_from_verb": False,
    }

    if not refined_verb_probs:
        return {
            "onset_distribution": refined_onset_distribution,
            "verb_probs": refined_verb_probs,
            "noun_exist_prob": refined_noun_exist_prob,
            "noun_probs": refined_noun_probs,
            "unknown_noun_prob": refined_unknown_noun_prob,
            "meta": cooperation_meta,
        }

    allowed_noun_set = {
        int(v) for v in list(allowed_noun_ids or []) if _safe_int(v, -999999) != -999999
    }
    if not allowed_noun_set:
        allowed_noun_set = {int(v) for v in list(package.noun_ids or [])}

    if clamp_noun_exists is None:
        no_noun_prior = 0.0
        for verb_label, verb_prob in refined_verb_probs.items():
            allow_no_noun_local = bool(
                (allow_no_noun_by_verb or {}).get(verb_label, allow_no_noun)
            )
            learned_no_noun = _clamp(
                _safe_float(
                    (verb_onset_priors.get(verb_label) or {}).get("no_noun_rate"),
                    1.0 if allow_no_noun_local else 0.0,
                )
            )
            if allow_no_noun_local:
                no_noun_prior += float(verb_prob) * float(learned_no_noun)
        noun_exists_prior = _clamp(1.0 - no_noun_prior)
        refined_noun_exist_prob = _blend_scalar(
            refined_noun_exist_prob,
            noun_exists_prior,
            primary_weight=0.74,
        )
        cooperation_meta["noun_exists_prior"] = float(noun_exists_prior)
        cooperation_meta["enabled"] = True

    if (
        refined_noun_probs
        and clamp_noun_object_id is None
        and not bool(clamp_unknown_noun)
    ):
        noun_prior_from_verbs: Dict[int, float] = {}
        noun_keys = [str(int(v)) for v in sorted(allowed_noun_set)]
        for verb_label, verb_prob in _top_prob_entries(refined_verb_probs, top_k=4):
            local_prior = _log_bonus_to_prob_map(
                dict(verb_noun_bonus.get(str(verb_label)) or {}),
                allowed_keys=noun_keys,
            )
            if not local_prior:
                continue
            for noun_key, local_prob in local_prior.items():
                try:
                    noun_id = int(noun_key)
                except Exception:
                    continue
                noun_prior_from_verbs[noun_id] = noun_prior_from_verbs.get(noun_id, 0.0) + float(verb_prob) * float(local_prob)
        if noun_prior_from_verbs:
            refined_noun_probs = _blend_prob_maps(
                refined_noun_probs,
                noun_prior_from_verbs,
                primary_weight=0.76,
            )
            cooperation_meta["noun_refined_from_verb"] = True
            cooperation_meta["enabled"] = True

    if clamp_verb_label == "":
        verb_prior_from_nouns: Dict[str, float] = {}
        noun_top = _top_prob_entries(refined_noun_probs, top_k=3)
        for noun_id, noun_prob in noun_top:
            local_prior = _log_bonus_to_prob_map(
                dict(noun_verb_bonus.get(str(int(noun_id))) or {}),
                allowed_keys=list(refined_verb_probs.keys()),
            )
            if not local_prior:
                continue
            for verb_label, local_prob in local_prior.items():
                verb_prior_from_nouns[str(verb_label)] = verb_prior_from_nouns.get(str(verb_label), 0.0) + float(refined_noun_exist_prob) * float(noun_prob) * float(local_prob)
        if refined_noun_exist_prob < 0.999:
            local_prior = _log_bonus_to_prob_map(
                dict(noun_verb_bonus.get("__NO_NOUN__") or {}),
                allowed_keys=list(refined_verb_probs.keys()),
            )
            for verb_label, local_prob in local_prior.items():
                verb_prior_from_nouns[str(verb_label)] = verb_prior_from_nouns.get(str(verb_label), 0.0) + float(1.0 - refined_noun_exist_prob) * float(local_prob)
        if verb_prior_from_nouns:
            refined_verb_probs = _blend_prob_maps(
                refined_verb_probs,
                verb_prior_from_nouns,
                primary_weight=0.72,
            )
            cooperation_meta["verb_refined_from_noun"] = True
            cooperation_meta["enabled"] = True

    if clamp_onset_ratio is None:
        onset_prior_rows: List[Tuple[float, Dict[str, Any]]] = []
        noun_rows = _top_prob_entries(refined_noun_probs, top_k=3)
        no_noun_mass = max(0.0, 1.0 - float(refined_noun_exist_prob))
        for verb_label, verb_prob in _top_prob_entries(refined_verb_probs, top_k=3):
            pair_priors = dict(verb_noun_onset_priors.get(str(verb_label)) or {})
            used_pair = False
            for noun_id, noun_prob in noun_rows:
                pair_prior = dict(pair_priors.get(str(int(noun_id))) or {})
                if int(pair_prior.get("count", 0) or 0) <= 0:
                    continue
                onset_prior_rows.append(
                    (float(verb_prob) * float(refined_noun_exist_prob) * float(noun_prob), pair_prior)
                )
                used_pair = True
            if no_noun_mass > 1e-6:
                pair_prior = dict(pair_priors.get("__NO_NOUN__") or {})
                if int(pair_prior.get("count", 0) or 0) > 0:
                    onset_prior_rows.append(
                        (float(verb_prob) * float(no_noun_mass), pair_prior)
                    )
                    used_pair = True
            if not used_pair:
                prior = dict(verb_onset_priors.get(str(verb_label)) or {})
                if int(prior.get("count", 0) or 0) > 0:
                    onset_prior_rows.append((float(verb_prob), prior))
        for noun_id, noun_prob in noun_rows:
            prior = dict(noun_onset_priors.get(str(int(noun_id))) or {})
            if int(prior.get("count", 0) or 0) > 0:
                onset_prior_rows.append(
                    (0.35 * float(refined_noun_exist_prob) * float(noun_prob), prior)
                )
        if no_noun_mass > 1e-6:
            prior = dict(noun_onset_priors.get("__NO_NOUN__") or {})
            if int(prior.get("count", 0) or 0) > 0:
                onset_prior_rows.append((0.35 * float(no_noun_mass), prior))
        semantic_onset_prior = _weighted_onset_prior_distribution(
            onset_prior_rows,
            bins=max(21, len(list(refined_onset_distribution or [])) or 21),
        )
        if semantic_onset_prior:
            top_verb_prob = max([0.0] + [float(v) for v in refined_verb_probs.values()])
            top_noun_prob = max([0.0] + [float(v) for v in refined_noun_probs.values()])
            semantic_weight = _clamp(
                0.18 + 0.18 * float(top_verb_prob) + 0.10 * float(top_noun_prob),
                0.18,
                0.42,
            )
            refined_onset_distribution = _blend_onset_distributions(
                refined_onset_distribution,
                semantic_onset_prior,
                primary_weight=1.0 - semantic_weight,
            )
            cooperation_meta["semantic_onset_prior_used"] = True
            cooperation_meta["enabled"] = True

    return {
        "onset_distribution": refined_onset_distribution,
        "verb_probs": refined_verb_probs,
        "noun_exist_prob": float(refined_noun_exist_prob),
        "noun_probs": refined_noun_probs,
        "unknown_noun_prob": float(refined_unknown_noun_prob),
        "meta": cooperation_meta,
    }


def _structured_decode(
    *,
    package: SemanticAdapterPackage,
    base_onset_ratio: float,
    onset_half_width: float,
    onset_confidence: float,
    onset_candidates: Optional[Sequence[Dict[str, Any]]] = None,
    verb_probs: Dict[str, float],
    noun_exist_prob: float,
    noun_probs: Dict[int, float],
    unknown_noun_prob: float,
    noun_required: bool,
    allow_no_noun: bool,
    allowed_noun_ids: Iterable[int],
    allowed_nouns_by_verb: Optional[Dict[str, Iterable[int]]] = None,
    allow_no_noun_by_verb: Optional[Dict[str, bool]] = None,
    clamp_onset_ratio: Optional[float] = None,
    clamp_verb_label: str = "",
    clamp_noun_exists: Optional[bool] = None,
    clamp_noun_object_id: Optional[int] = None,
    clamp_unknown_noun: bool = False,
) -> Dict[str, Any]:
    base_allowed_nouns = {int(v) for v in list(allowed_noun_ids or [])}
    if not base_allowed_nouns:
        base_allowed_nouns = {int(v) for v in list(package.noun_ids or [])}

    verb_priors = dict((package.structured_stats or {}).get("verb_onset_priors") or {})
    verb_noun_bonus = dict((package.structured_stats or {}).get("verb_noun_bonus") or {})
    clamp_verb_label = str(clamp_verb_label or "").strip()
    if clamp_verb_label:
        forced_prob = float(max(1e-6, verb_probs.get(clamp_verb_label, 1.0)))
        verb_candidates = [{"label": clamp_verb_label, "prob": forced_prob}]
    else:
        verb_candidates = sorted(
            [{"label": str(label), "prob": float(prob)} for label, prob in verb_probs.items()],
            key=lambda row: float(row.get("prob") or 0.0),
            reverse=True,
        )[:5]
    if not verb_candidates and package.verb_labels:
        verb_candidates = [{"label": str(package.verb_labels[0]), "prob": 1.0}]

    candidate_rows: List[Dict[str, Any]] = []
    onset_half_width = max(0.03, float(onset_half_width))
    onset_confidence = _clamp(float(onset_confidence))
    if clamp_onset_ratio is not None:
        onset_rows = [{"ratio": _clamp(float(clamp_onset_ratio)), "score": 1.0}]
    else:
        onset_rows = _top_onset_candidates(onset_candidates or [], top_k=3)
    for verb_row in verb_candidates:
        verb_label = str(verb_row.get("label") or "")
        verb_prob = float(verb_row.get("prob") or 0.0)
        prior = dict(verb_priors.get(verb_label) or {})
        current_bonus = dict(verb_noun_bonus.get(verb_label) or {})
        allowed_nouns = {
            int(v)
            for v in list((allowed_nouns_by_verb or {}).get(verb_label, base_allowed_nouns) or [])
        }
        if not allowed_nouns:
            allowed_nouns = set(base_allowed_nouns)
        allow_no_noun_local = bool(
            (allow_no_noun_by_verb or {}).get(verb_label, allow_no_noun)
        )
        noun_required_local = not allow_no_noun_local if verb_label in dict(allow_no_noun_by_verb or {}) else bool(noun_required)
        if clamp_noun_exists is None:
            exist_options = [1]
            if not noun_required_local and allow_no_noun_local:
                exist_options = [0, 1]
        else:
            exist_options = [1 if bool(clamp_noun_exists) else 0]
        for onset_row in onset_rows:
            onset_ratio = _clamp(_safe_float(onset_row.get("ratio"), base_onset_ratio))
            onset_prob = max(1e-6, _safe_float(onset_row.get("score"), onset_confidence))
            adjusted_onset = float(onset_ratio)
            psi_vo = 0.0
            if prior:
                prior_mean = _clamp(_safe_float(prior.get("mean"), onset_ratio))
                prior_std = max(0.04, _safe_float(prior.get("std"), onset_half_width))
                if clamp_onset_ratio is None:
                    adjusted_onset = _clamp(0.72 * float(onset_ratio) + 0.28 * prior_mean)
                psi_vo = -0.18 * abs(float(onset_ratio) - prior_mean) / max(0.04, prior_std + onset_half_width)
            score_base = (
                _safe_log_prob(verb_prob)
                + 0.12 * _safe_log_prob(onset_confidence)
                + 0.88 * _safe_log_prob(onset_prob)
                + float(psi_vo)
            )
            for noun_exists in exist_options:
                exist_prob = float(noun_exist_prob if noun_exists else (1.0 - noun_exist_prob))
                score_exist = score_base + _safe_log_prob(exist_prob)
                if noun_exists:
                    noun_candidates: List[Tuple[int, float, bool]] = []
                    if clamp_unknown_noun:
                        noun_candidates = [(int(package.unknown_noun_id), float(max(unknown_noun_prob, 1e-6)), True)]
                    elif clamp_noun_object_id is not None:
                        forced_noun = int(clamp_noun_object_id)
                        if not allowed_nouns or forced_noun in allowed_nouns:
                            noun_candidates = [(forced_noun, float(max(noun_probs.get(forced_noun, 1.0), 1e-6)), False)]
                    else:
                        for noun_id, noun_prob in noun_probs.items():
                            try:
                                noun_id = int(noun_id)
                            except Exception:
                                continue
                            if allowed_nouns and noun_id not in allowed_nouns:
                                continue
                            noun_candidates.append((noun_id, float(noun_prob), False))
                        noun_candidates.sort(key=lambda row: float(row[1]), reverse=True)
                        noun_candidates = noun_candidates[:5]
                        noun_candidates.append((int(package.unknown_noun_id), float(unknown_noun_prob), True))
                    for noun_id, noun_prob, is_unknown in noun_candidates:
                        noun_key = "__UNKNOWN__" if is_unknown else str(int(noun_id))
                        psi_vn = float(current_bonus.get(noun_key, current_bonus.get("__NO_NOUN__", 0.0)))
                        score = score_exist + _safe_log_prob(noun_prob) + 0.25 * psi_vn
                        if is_unknown:
                            score -= 0.20
                        candidate_rows.append(
                            {
                                "verb_label": verb_label,
                                "noun_exists": True,
                                "noun_object_id": None if is_unknown else int(noun_id),
                                "noun_is_unknown": bool(is_unknown),
                                "onset_ratio": float(adjusted_onset),
                                "raw_onset_ratio": float(onset_ratio),
                                "score": float(score),
                            }
                        )
                else:
                    if clamp_noun_object_id is not None or clamp_unknown_noun:
                        continue
                    psi_ve = 0.10 if allow_no_noun_local else -2.50
                    candidate_rows.append(
                        {
                            "verb_label": verb_label,
                            "noun_exists": False,
                            "noun_object_id": None,
                            "noun_is_unknown": False,
                            "onset_ratio": float(adjusted_onset),
                            "raw_onset_ratio": float(onset_ratio),
                            "score": float(score_exist + psi_ve),
                        }
                    )

    if not candidate_rows:
        return {}
    max_score = max(float(row.get("score") or 0.0) for row in candidate_rows)
    normalizer = sum(math.exp(float(row.get("score") or 0.0) - max_score) for row in candidate_rows)
    for row in candidate_rows:
        row["joint_prob"] = float(math.exp(float(row.get("score") or 0.0) - max_score) / max(1e-6, normalizer))
    candidate_rows.sort(key=lambda row: float(row.get("joint_prob") or 0.0), reverse=True)
    best = dict(candidate_rows[0])
    second_prob = float(candidate_rows[1].get("joint_prob") or 0.0) if len(candidate_rows) > 1 else 0.0
    band_width = float(onset_half_width * 2.0)
    risk_score = (
        1.0
        - float(best.get("joint_prob") or 0.0)
        + min(1.0, band_width / 0.30) * 0.18
        + (0.18 if bool(best.get("noun_is_unknown")) else 0.0)
        + (0.10 if abs(float(noun_exist_prob) - 0.5) < 0.12 else 0.0)
    )
    best["joint_margin"] = float(max(0.0, float(best.get("joint_prob") or 0.0) - second_prob))
    best["risk_score"] = float(_clamp(risk_score, 0.0, 1.5))
    best["band_width"] = float(band_width)
    best["candidate_count"] = int(len(candidate_rows))
    return {
        "best": best,
        "top_candidates": candidate_rows[:5],
    }


def train_adapter_from_feedback(
    *,
    feedback_path: str,
    output_path: str,
    feature_dim: int,
    verb_labels: Sequence[str],
    noun_ids: Sequence[int],
    hidden_dim: int = 96,
    onset_bins: int = 21,
    feature_layout: Optional[Dict[str, Any]] = None,
    video_adapter_rank: int = 0,
    video_adapter_alpha: float = 1.0,
    epochs: int = 12,
    batch_size: int = 16,
    lr: float = 1e-3,
    min_samples: int = 8,
    init_package_path: str = "",
) -> Tuple[bool, str, Optional[SemanticAdapterPackage]]:
    rows = _load_feedback_rows(feedback_path)
    if len(rows) < int(min_samples):
        return False, f"Need at least {int(min_samples)} feedback samples, found {len(rows)}.", None

    valid_rows = _prepare_rows(rows, int(feature_dim))
    if len(valid_rows) < int(min_samples):
        return False, f"Only {len(valid_rows)} feedback rows matched the current feature schema.", None

    train_rows, calib_rows = _split_feedback_rows(valid_rows)
    include_unknown_noun = True
    noun_count = len(list(noun_ids or []))
    onset_bins = max(0, int(onset_bins))
    normalized_feature_layout = _sanitize_feature_layout(
        feature_layout,
        feature_dim=int(feature_dim),
    )
    video_adapter_rank = max(0, int(video_adapter_rank))
    video_adapter_alpha = max(0.0, float(video_adapter_alpha or 0.0))
    torch, _nn, F = _ensure_torch()
    (
        device,
        features,
        onset_targets,
        verb_targets,
        noun_exist_targets,
        noun_targets,
        onset_weights,
        verb_weights,
        noun_weights,
        exclusive_anchor,
        exclusive_weight,
    ) = _tensorize_rows(
        train_rows,
        feature_dim=int(feature_dim),
        noun_count=noun_count,
        include_unknown_noun=include_unknown_noun,
    )
    model = SemanticAdapterNet(
        feature_dim=int(feature_dim),
        hidden_dim=int(hidden_dim),
        verb_count=len(list(verb_labels or [])),
        noun_count=int(noun_count + (1 if include_unknown_noun else 0)),
        onset_bins=int(onset_bins),
        feature_layout=normalized_feature_layout,
        video_adapter_rank=video_adapter_rank,
        video_adapter_alpha=video_adapter_alpha,
    )
    model.to(device)
    loaded_init_path = ""
    init_package_path = str(init_package_path or "").strip()
    if init_package_path and os.path.isfile(init_package_path):
        init_package = load_adapter_package(init_package_path)
        if init_package is not None:
            compatible = (
                int(getattr(init_package, "feature_dim", 0) or 0) == int(feature_dim)
                and list(getattr(init_package, "verb_labels", []) or []) == [str(v) for v in list(verb_labels or [])]
                and [int(v) for v in list(getattr(init_package, "noun_ids", []) or [])] == [int(v) for v in list(noun_ids or [])]
                and int(getattr(init_package, "onset_bins", 0) or 0) == int(onset_bins)
                and dict(getattr(init_package, "feature_layout", {}) or {}) == dict(normalized_feature_layout or {})
                and int(getattr(init_package, "video_adapter_rank", 0) or 0) == int(video_adapter_rank)
            )
            if compatible:
                try:
                    model.load_state_dict(dict(getattr(init_package, "state_dict", {}) or {}), strict=False)
                    loaded_init_path = init_package_path
                except Exception:
                    loaded_init_path = ""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

    sample_count = features.shape[0]
    batch_size = max(1, int(batch_size))
    losses: List[float] = []
    for _epoch in range(max(1, int(epochs))):
        perm = torch.randperm(sample_count, device=device)
        epoch_losses: List[float] = []
        for start in range(0, sample_count, batch_size):
            idx = perm[start : start + batch_size]
            batch_x = features[idx]
            batch_onset = onset_targets[idx]
            batch_onset_bins = torch.clamp(
                torch.round(batch_onset * float(max(1, onset_bins - 1))),
                min=0,
                max=max(0, onset_bins - 1),
            ).long()
            batch_verb = verb_targets[idx]
            batch_nexist = noun_exist_targets[idx]
            batch_noun = noun_targets[idx]
            batch_onset_w = onset_weights[idx]
            batch_verb_w = verb_weights[idx]
            batch_noun_w = noun_weights[idx]
            batch_exclusive_anchor = exclusive_anchor[idx]
            batch_exclusive_weight = exclusive_weight[idx]

            out = model(batch_x)
            loss = torch.tensor(0.0, dtype=torch.float32, device=device)

            mean = out["onset_mean"]
            logvar = out["onset_logvar"]
            onset_loss = 0.5 * (logvar + ((batch_onset - mean) ** 2) / torch.exp(logvar))
            onset_loss = onset_loss.reshape(-1)
            loss = loss + (
                (onset_loss * batch_onset_w).sum()
                / torch.clamp(batch_onset_w.sum(), min=1e-6)
            )
            exclusion_mask = batch_exclusive_anchor >= 0.0
            if torch.any(exclusion_mask):
                sigma = 0.08
                z = (mean[exclusion_mask] - batch_exclusive_anchor[exclusion_mask]) / sigma
                exclusion_penalty = torch.exp(-0.5 * (z ** 2))
                exclusion_w = batch_exclusive_weight[exclusion_mask]
                loss = loss + 0.08 * (
                    (exclusion_penalty * exclusion_w).sum()
                    / torch.clamp(exclusion_w.sum(), min=1e-6)
                )
            if onset_bins > 1 and "onset_bin_logits" in out and out["onset_bin_logits"].numel() > 0:
                onset_bin_loss = F.cross_entropy(
                    out["onset_bin_logits"],
                    batch_onset_bins,
                    reduction="none",
                )
                loss = loss + 0.35 * (
                    (onset_bin_loss * batch_onset_w).sum()
                    / torch.clamp(batch_onset_w.sum(), min=1e-6)
                )

            if "verb_logits" in out and out["verb_logits"].numel() > 0:
                mask = batch_verb >= 0
                if torch.any(mask):
                    verb_loss = F.cross_entropy(
                        out["verb_logits"][mask],
                        batch_verb[mask],
                        reduction="none",
                    )
                    verb_w = batch_verb_w[mask]
                    loss = loss + (
                        (verb_loss * verb_w).sum()
                        / torch.clamp(verb_w.sum(), min=1e-6)
                    )

            noun_exist_loss = F.binary_cross_entropy_with_logits(
                out["noun_exist_logit"].reshape(-1),
                batch_nexist.reshape(-1),
                reduction="none",
            )
            loss = loss + (
                (noun_exist_loss * batch_noun_w).sum()
                / torch.clamp(batch_noun_w.sum(), min=1e-6)
            )

            if "noun_logits" in out and out["noun_logits"].numel() > 0:
                mask = batch_noun >= 0
                if torch.any(mask):
                    noun_loss = F.cross_entropy(
                        out["noun_logits"][mask],
                        batch_noun[mask],
                        reduction="none",
                    )
                    noun_w = batch_noun_w[mask]
                    loss = loss + (
                        (noun_loss * noun_w).sum()
                        / torch.clamp(noun_w.sum(), min=1e-6)
                    )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        if epoch_losses:
            losses.extend(epoch_losses)

    (
        _device,
        calib_features,
        calib_onset,
        calib_verb,
        calib_nexist,
        calib_noun,
        _calib_onset_w,
        _calib_verb_w,
        _calib_noun_w,
        _calib_exclusive_anchor,
        _calib_exclusive_weight,
    ) = _tensorize_rows(
        calib_rows,
        feature_dim=int(feature_dim),
        noun_count=noun_count,
        include_unknown_noun=include_unknown_noun,
    )
    calib_out = _predict_raw(model, calib_features)
    calib_onset_bins = torch.clamp(
        torch.round(calib_onset * float(max(1, onset_bins - 1))),
        min=0,
        max=max(0, onset_bins - 1),
    ).long()
    verb_temperature = 1.0
    noun_temperature = 1.0
    noun_exist_temperature = 1.0
    onset_bin_temperature = 1.0
    noun_exist_threshold = 0.5
    onset_radius_q90 = 0.10
    onset_radius_q95 = 0.14
    onset_mae = 0.0
    if calib_features.shape[0] > 0:
        if "verb_logits" in calib_out:
            verb_temperature = _fit_multiclass_temperature(
                calib_out["verb_logits"], calib_verb
            )
        if onset_bins > 1 and "onset_bin_logits" in calib_out:
            onset_bin_temperature = _fit_multiclass_temperature(
                calib_out["onset_bin_logits"], calib_onset_bins
            )
        if "noun_logits" in calib_out:
            noun_temperature = _fit_multiclass_temperature(
                calib_out["noun_logits"], calib_noun
            )
        noun_exist_temperature = _fit_binary_temperature(
            calib_out["noun_exist_logit"], calib_nexist
        )
        noun_exist_probs = [
            _sigmoid_with_temperature(logit, noun_exist_temperature)
            for logit in list(calib_out["noun_exist_logit"])
        ]
        noun_exist_threshold = _best_binary_threshold(
            noun_exist_probs,
            [float(v) for v in calib_nexist.detach().cpu().tolist()],
        )
        onset_preds = [float(v) for v in calib_out["onset_mean"].detach().cpu().tolist()]
        onset_targets_eval = [float(v) for v in calib_onset.detach().cpu().tolist()]
        onset_errors = [abs(p - t) for p, t in zip(onset_preds, onset_targets_eval)]
        onset_radius_q90 = _quantile(onset_errors, 0.90, default=0.10)
        onset_radius_q95 = _quantile(onset_errors, 0.95, default=0.14)
        onset_mae = float(sum(onset_errors) / max(1, len(onset_errors)))

    calibration = {
        "verb_temperature": float(verb_temperature),
        "noun_temperature": float(noun_temperature),
        "noun_exist_temperature": float(noun_exist_temperature),
        "onset_bin_temperature": float(onset_bin_temperature),
        "noun_exist_threshold": float(noun_exist_threshold),
        "onset_radius_q90": float(max(0.03, onset_radius_q90)),
        "onset_radius_q95": float(max(0.05, onset_radius_q95)),
        "onset_mae": float(max(0.0, onset_mae)),
    }
    structured_stats = _collect_structured_stats(
        train_rows,
        verb_labels=verb_labels,
        noun_ids=noun_ids,
    )

    package = SemanticAdapterPackage(
        feature_dim=int(feature_dim),
        hidden_dim=int(hidden_dim),
        verb_labels=[str(v) for v in list(verb_labels or [])],
        noun_ids=[int(v) for v in list(noun_ids or [])],
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        onset_bins=int(onset_bins),
        feature_layout=normalized_feature_layout,
        video_adapter_rank=video_adapter_rank,
        video_adapter_alpha=video_adapter_alpha,
        sample_count=int(len(valid_rows)),
        epochs=int(epochs),
        mean_loss=float(sum(losses) / max(1, len(losses))),
        trained_at=str(
            __import__("datetime").datetime.now().isoformat(timespec="seconds")
        ),
        supports_unknown_noun=True,
        unknown_noun_id=int(UNKNOWN_NOUN_ID),
        calibration=calibration,
        structured_stats=structured_stats,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(package.to_dict(), output_path)
    if loaded_init_path:
        return True, (
            f"Trained semantic adapter on {len(valid_rows)} samples "
            f"(warm-started from {os.path.basename(loaded_init_path)})."
        ), package
    return True, f"Trained semantic adapter on {len(valid_rows)} samples.", package


def predict_with_adapter(
    *,
    feature: Sequence[float],
    package: SemanticAdapterPackage,
    allowed_noun_ids: Optional[Iterable[int]] = None,
    noun_required: bool = True,
    allow_no_noun: bool = False,
    external_verb_scores: Optional[Dict[str, float]] = None,
    noun_support_scores: Optional[Dict[int, float]] = None,
    noun_onset_support_scores: Optional[Dict[int, float]] = None,
    allowed_nouns_by_verb: Optional[Dict[str, Iterable[int]]] = None,
    allow_no_noun_by_verb: Optional[Dict[str, bool]] = None,
    clamp_onset_ratio: Optional[float] = None,
    clamp_verb_label: str = "",
    clamp_noun_exists: Optional[bool] = None,
    clamp_noun_object_id: Optional[int] = None,
    clamp_unknown_noun: bool = False,
    anchor_onset_ratio: Optional[float] = None,
    anchor_onset_half_width: Optional[float] = None,
    anchor_onset_weight: float = 0.0,
    exclude_onset_ratio: Optional[float] = None,
    exclude_onset_half_width: Optional[float] = None,
    exclude_onset_weight: float = 0.0,
) -> Dict[str, Any]:
    if package is None:
        return {}
    if len(list(feature or [])) != int(package.feature_dim):
        return {}
    torch, _nn, F = _ensure_torch()
    model = SemanticAdapterNet(
        feature_dim=int(package.feature_dim),
        hidden_dim=int(package.hidden_dim),
        verb_count=len(list(package.verb_labels or [])),
        noun_count=len(list(package.noun_ids or []))
        + (1 if package.supports_unknown_noun else 0),
        onset_bins=int(getattr(package, "onset_bins", 0) or 0),
        feature_layout=dict(getattr(package, "feature_layout", {}) or {}),
        video_adapter_rank=int(getattr(package, "video_adapter_rank", 0) or 0),
        video_adapter_alpha=float(
            getattr(package, "video_adapter_alpha", 0.0) or 0.0
        ),
    )
    model.load_state_dict(dict(package.state_dict or {}), strict=False)
    model.eval()
    calibration = dict(package.calibration or {})
    clamp_verb_label = str(clamp_verb_label or "").strip()
    onset_distribution: List[Dict[str, float]] = []
    onset_candidates: List[Dict[str, float]] = []
    with torch.no_grad():
        x = torch.tensor([[float(v) for v in list(feature or [])]], dtype=torch.float32)
        out = model(x)
        onset_ratio = float(out["onset_mean"][0].cpu().item())
        onset_std = math.sqrt(float(torch.exp(out["onset_logvar"][0]).cpu().item()))
        onset_bins = max(0, int(getattr(package, "onset_bins", 0) or 0))
        onset_bin_distribution: List[Dict[str, float]] = []
        if onset_bins > 1 and "onset_bin_logits" in out:
            onset_bin_probs = _softmax_with_temperature(
                out["onset_bin_logits"][0],
                float(calibration.get("onset_bin_temperature", 1.0)),
            )
            onset_bin_distribution = [
                {
                    "ratio": _bin_index_to_ratio(idx, onset_bins),
                    "score": float(onset_bin_probs[idx]),
                }
                for idx in range(min(onset_bins, len(onset_bin_probs)))
            ]

        verb_probs: Dict[str, float] = {}
        if "verb_logits" in out:
            calibrated = _softmax_with_temperature(
                out["verb_logits"][0],
                float(calibration.get("verb_temperature", 1.0)),
            )
            verb_probs = {
                str(label): float(calibrated[idx])
                for idx, label in enumerate(list(package.verb_labels or []))
                if idx < len(calibrated)
            }
            if external_verb_scores:
                verb_probs = _blend_prob_maps(
                    verb_probs,
                    _normalize_prob_map(external_verb_scores),
                    primary_weight=0.76,
                )

        noun_exist_prob = _sigmoid_with_temperature(
            out["noun_exist_logit"][0],
            float(calibration.get("noun_exist_temperature", 1.0)),
        )

        noun_probs: Dict[int, float] = {}
        unknown_noun_prob = 0.0
        if "noun_logits" in out:
            calibrated = _softmax_with_temperature(
                out["noun_logits"][0],
                float(calibration.get("noun_temperature", 1.0)),
            )
            actual_count = len(list(package.noun_ids or []))
            for idx, noun_id in enumerate(list(package.noun_ids or [])):
                if idx >= len(calibrated):
                    continue
                noun_probs[int(noun_id)] = float(calibrated[idx])
            if package.supports_unknown_noun and len(calibrated) > actual_count:
                unknown_noun_prob = float(calibrated[actual_count])

        if noun_probs:
            support_scores = _normalize_prob_map(
                {int(k): float(v) for k, v in dict(noun_support_scores or {}).items()}
            )
            onset_support_scores = _normalize_prob_map(
                {int(k): float(v) for k, v in dict(noun_onset_support_scores or {}).items()}
            )
            merged = {}
            for noun_id in noun_probs.keys():
                merged[int(noun_id)] = (
                    0.66 * float(noun_probs.get(noun_id, 0.0))
                    + 0.22 * float(support_scores.get(int(noun_id), 0.0))
                    + 0.12 * float(onset_support_scores.get(int(noun_id), 0.0))
                )
            noun_probs = _normalize_prob_map(merged)
            top_known = max(noun_probs.values()) if noun_probs else 0.0
            unknown_noun_prob = max(float(unknown_noun_prob), max(0.05, 0.35 - float(top_known)))

    gaussian_onset_distribution = _build_onset_distribution(
        onset_ratio,
        max(onset_std, 0.02),
        bins=max(21, int(getattr(package, "onset_bins", 0) or 21)),
    )
    if onset_bin_distribution:
        onset_distribution = _blend_onset_distributions(
            onset_bin_distribution,
            gaussian_onset_distribution,
            primary_weight=0.58,
        )
    else:
        onset_distribution = _normalize_onset_distribution(gaussian_onset_distribution)
    onset_ratio = _onset_expectation(onset_distribution, default=onset_ratio)

    if clamp_onset_ratio is not None:
        onset_ratio = _clamp(float(clamp_onset_ratio))
        onset_distribution = [{"ratio": float(onset_ratio), "score": 1.0}]
        onset_std = min(float(onset_std), 0.03)

    if clamp_verb_label:
        if clamp_verb_label in list(package.verb_labels or []):
            verb_probs = {
                str(label): (1.0 if str(label) == clamp_verb_label else 0.0)
                for label in list(package.verb_labels or [])
            }
        else:
            clamp_verb_label = ""

    if clamp_noun_exists is not None:
        noun_exist_prob = 1.0 if bool(clamp_noun_exists) else 0.0
    if clamp_unknown_noun:
        noun_exist_prob = 1.0
        noun_probs = {}
        unknown_noun_prob = 1.0
    elif clamp_noun_object_id is not None:
        noun_exist_prob = 1.0
        forced_id = int(clamp_noun_object_id)
        noun_probs = {forced_id: 1.0}
        unknown_noun_prob = 0.0

    cooperative_refinement = _cooperative_refine_predictions(
        package=package,
        onset_distribution=onset_distribution,
        verb_probs=verb_probs,
        noun_exist_prob=noun_exist_prob,
        noun_probs=noun_probs,
        unknown_noun_prob=unknown_noun_prob,
        noun_required=bool(noun_required),
        allow_no_noun=bool(allow_no_noun),
        allowed_noun_ids=allowed_noun_ids,
        allow_no_noun_by_verb=allow_no_noun_by_verb,
        clamp_onset_ratio=clamp_onset_ratio,
        clamp_verb_label=clamp_verb_label,
        clamp_noun_exists=clamp_noun_exists,
        clamp_noun_object_id=clamp_noun_object_id,
        clamp_unknown_noun=bool(clamp_unknown_noun),
    )
    onset_distribution = list(
        cooperative_refinement.get("onset_distribution") or onset_distribution
    )
    verb_probs = dict(cooperative_refinement.get("verb_probs") or verb_probs)
    noun_exist_prob = float(
        cooperative_refinement.get("noun_exist_prob", noun_exist_prob)
    )
    noun_probs = dict(cooperative_refinement.get("noun_probs") or noun_probs)
    unknown_noun_prob = float(
        cooperative_refinement.get("unknown_noun_prob", unknown_noun_prob)
    )
    onset_ratio = _onset_expectation(onset_distribution, default=onset_ratio)

    if clamp_onset_ratio is None and anchor_onset_ratio is not None:
        anchor_ratio = _clamp(float(anchor_onset_ratio))
        anchor_half_width = max(
            0.03,
            float(
                anchor_onset_half_width
                if anchor_onset_half_width is not None
                else 0.10
            ),
        )
        anchor_weight = _clamp(float(anchor_onset_weight or 0.0), 0.0, 0.55)
        if anchor_weight > 1e-6:
            anchor_distribution = _build_onset_distribution(
                anchor_ratio,
                max(0.02, anchor_half_width / 1.6),
                bins=max(21, int(getattr(package, "onset_bins", 0) or 21)),
            )
            onset_distribution = _blend_onset_distributions(
                onset_distribution,
                anchor_distribution,
                primary_weight=1.0 - anchor_weight,
            )
            onset_ratio = _onset_expectation(onset_distribution, default=onset_ratio)
    if clamp_onset_ratio is None and exclude_onset_ratio is not None:
        exclusion_ratio = _clamp(float(exclude_onset_ratio))
        exclusion_half_width = max(
            0.03,
            float(
                exclude_onset_half_width
                if exclude_onset_half_width is not None
                else 0.10
            ),
        )
        exclusion_weight = _clamp(float(exclude_onset_weight or 0.0), 0.0, 0.35)
        if exclusion_weight > 1e-6:
            sigma = max(0.02, exclusion_half_width / 1.6)
            excluded_rows: List[Dict[str, float]] = []
            for row in list(onset_distribution or []):
                ratio = _clamp(_safe_float(row.get("ratio"), 0.5))
                score = max(0.0, _safe_float(row.get("score"), 0.0))
                z = (ratio - exclusion_ratio) / sigma
                penalty = max(
                    1e-4,
                    1.0 - exclusion_weight * math.exp(-0.5 * (z ** 2)),
                )
                excluded_rows.append(
                    {"ratio": float(ratio), "score": float(score) * float(penalty)}
                )
            onset_distribution = _normalize_onset_distribution(excluded_rows)
            onset_ratio = _onset_expectation(onset_distribution, default=onset_ratio)

    onset_half_width = max(
        float(calibration.get("onset_radius_q90", 0.10)),
        min(0.24, max(0.03, float(onset_std) * 1.6)),
    )
    onset_half_width = min(
        0.24,
        _distribution_half_width(
            onset_distribution,
            center=onset_ratio,
            default=onset_half_width,
        ),
    )
    if clamp_onset_ratio is not None:
        onset_half_width = min(float(onset_half_width), 0.04)
    onset_band = {
        "center_ratio": _clamp(onset_ratio),
        "left_ratio": _clamp(onset_ratio - onset_half_width),
        "right_ratio": _clamp(onset_ratio + onset_half_width),
    }
    band_width = float(onset_band["right_ratio"] - onset_band["left_ratio"])
    onset_confidence = _clamp(1.0 - band_width / 0.50)
    if clamp_onset_ratio is not None:
        onset_confidence = max(float(onset_confidence), 0.98)
    onset_candidates = _top_onset_candidates(onset_distribution, top_k=3)

    structured = _structured_decode(
        package=package,
        base_onset_ratio=onset_ratio,
        onset_half_width=onset_half_width,
        onset_confidence=onset_confidence,
        onset_candidates=onset_candidates,
        verb_probs=verb_probs,
        noun_exist_prob=noun_exist_prob,
        noun_probs=noun_probs,
        unknown_noun_prob=unknown_noun_prob,
        noun_required=bool(noun_required),
        allow_no_noun=bool(allow_no_noun),
        allowed_noun_ids=allowed_noun_ids,
        allowed_nouns_by_verb=allowed_nouns_by_verb,
        allow_no_noun_by_verb=allow_no_noun_by_verb,
        clamp_onset_ratio=clamp_onset_ratio,
        clamp_verb_label=clamp_verb_label,
        clamp_noun_exists=clamp_noun_exists,
        clamp_noun_object_id=clamp_noun_object_id,
        clamp_unknown_noun=bool(clamp_unknown_noun),
    )

    verb_rows = [
        {"label": str(label), "score": float(score)}
        for label, score in sorted(verb_probs.items(), key=lambda item: float(item[1]), reverse=True)
    ]
    noun_rows = [
        {"object_id": int(noun_id), "score": float(score), "is_unknown": False}
        for noun_id, score in sorted(noun_probs.items(), key=lambda item: float(item[1]), reverse=True)
    ]
    if package.supports_unknown_noun:
        noun_rows.append(
            {
                "object_id": int(package.unknown_noun_id),
                "score": float(unknown_noun_prob),
                "is_unknown": True,
                "display_value": "Unknown / Other",
            }
        )
        noun_rows.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)

    best = dict((structured or {}).get("best") or {})
    if best:
        best["onset_band"] = dict(onset_band)
        best["onset_confidence"] = float(onset_confidence)
        best["noun_exists_prob"] = float(noun_exist_prob)

    return {
        "onset_ratio": _clamp(onset_ratio),
        "onset_std": float(onset_std),
        "onset_confidence": float(onset_confidence),
        "onset_band": onset_band,
        "onset_distribution": onset_distribution,
        "onset_candidates": onset_candidates,
        "verb_candidates": verb_rows,
        "noun_exists_prob": float(noun_exist_prob),
        "noun_exists_threshold": float(calibration.get("noun_exist_threshold", 0.5)),
        "noun_candidates": noun_rows,
        "structured": structured,
        "calibration": dict(calibration),
        "runtime_constraints": {
            "clamp_onset_ratio": None if clamp_onset_ratio is None else float(_clamp(clamp_onset_ratio)),
            "clamp_verb_label": str(clamp_verb_label or ""),
            "clamp_noun_exists": None if clamp_noun_exists is None else bool(clamp_noun_exists),
            "clamp_noun_object_id": None if clamp_noun_object_id is None else int(clamp_noun_object_id),
            "clamp_unknown_noun": bool(clamp_unknown_noun),
            "anchor_onset_ratio": None if anchor_onset_ratio is None else float(_clamp(anchor_onset_ratio)),
            "anchor_onset_half_width": None if anchor_onset_half_width is None else float(max(0.0, anchor_onset_half_width)),
            "anchor_onset_weight": float(max(0.0, anchor_onset_weight)),
            "exclude_onset_ratio": None if exclude_onset_ratio is None else float(_clamp(exclude_onset_ratio)),
            "exclude_onset_half_width": None if exclude_onset_half_width is None else float(max(0.0, exclude_onset_half_width)),
            "exclude_onset_weight": float(max(0.0, exclude_onset_weight)),
        },
        "cooperative_refinement": dict(
            cooperative_refinement.get("meta") or {}
        ),
    }
