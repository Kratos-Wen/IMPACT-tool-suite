import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from core.onset_guidance import build_temporal_sample_indices


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def load_precomputed_feature_cache(path: str) -> Optional[Dict[str, Any]]:
    cache_path = str(path or "").strip()
    if not cache_path or not os.path.isfile(cache_path):
        return None
    try:
        import numpy as np
    except Exception:
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            meta_blob = payload.get("meta_json")
            meta_text = ""
            if meta_blob is not None and len(meta_blob) > 0:
                meta_text = str(meta_blob[0])
            meta = json.loads(meta_text) if meta_text else {}
            return {
                "path": cache_path,
                "meta": dict(meta or {}),
                "window_starts": payload["window_starts"],
                "window_ends": payload["window_ends"],
                "window_centers": payload["window_centers"],
                "segment_features": payload["segment_features"],
                "verb_scores": payload["verb_scores"] if "verb_scores" in payload.files else None,
            }
    except Exception:
        return None


def aggregate_precomputed_feature_cache(
    cache: Optional[Dict[str, Any]],
    *,
    start_frame: Any,
    end_frame: Any,
    onset_band: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    if not isinstance(cache, dict):
        return {}
    start = _safe_int(start_frame)
    end = _safe_int(end_frame)
    if start is None or end is None:
        return {}
    if end < start:
        start, end = end, start

    try:
        import numpy as np
    except Exception:
        return {}

    starts = cache.get("window_starts")
    ends = cache.get("window_ends")
    centers = cache.get("window_centers")
    features = cache.get("segment_features")
    if starts is None or ends is None or centers is None or features is None:
        return {}
    if len(starts) <= 0 or len(features) <= 0:
        return {}

    starts = np.asarray(starts, dtype=np.int32).reshape(-1)
    ends = np.asarray(ends, dtype=np.int32).reshape(-1)
    centers = np.asarray(centers, dtype=np.int32).reshape(-1)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != starts.shape[0]:
        return {}

    segment_span = max(1, end - start + 1)
    overlap = np.maximum(0, np.minimum(ends, end) - np.maximum(starts, start) + 1)
    weights = overlap.astype(np.float32) / float(segment_span)

    if onset_band:
        band_start = _safe_int(onset_band.get("start_frame"))
        band_end = _safe_int(onset_band.get("end_frame"))
        if band_start is not None and band_end is not None:
            if band_end < band_start:
                band_start, band_end = band_end, band_start
            band_span = max(1, band_end - band_start + 1)
            band_overlap = np.maximum(
                0, np.minimum(ends, band_end) - np.maximum(starts, band_start) + 1
            )
            weights += 0.65 * (band_overlap.astype(np.float32) / float(band_span))

    valid = weights > 0
    if not np.any(valid):
        midpoint = int(round((start + end) / 2.0))
        nearest = int(np.argmin(np.abs(centers - midpoint)))
        valid = np.zeros_like(weights, dtype=bool)
        valid[nearest] = True
        weights = valid.astype(np.float32)

    weights = weights[valid]
    feature_rows = features[valid]
    denom = float(np.maximum(weights.sum(), 1e-6))
    pooled_feature = ((feature_rows * weights[:, None]).sum(axis=0) / denom).astype(
        np.float32
    )

    labels = list((cache.get("meta") or {}).get("labels") or [])
    verb_scores = cache.get("verb_scores")
    candidates: List[Dict[str, Any]] = []
    if verb_scores is not None:
        score_rows = np.asarray(verb_scores, dtype=np.float32)
        if score_rows.ndim == 2 and score_rows.shape[0] == starts.shape[0]:
            score_rows = score_rows[valid]
            pooled_scores = ((score_rows * weights[:, None]).sum(axis=0) / denom).astype(
                np.float32
            )
            if labels and pooled_scores.shape[0] == len(labels):
                order = np.argsort(-pooled_scores)[: max(1, int(top_k or 5))]
                for idx in list(order):
                    if int(idx) < 0 or int(idx) >= len(labels):
                        continue
                    candidates.append(
                        {
                            "label": str(labels[int(idx)]),
                            "score": float(max(0.0, pooled_scores[int(idx)])),
                        }
                    )

    return {
        "segment_feature": [float(v) for v in pooled_feature.tolist()],
        "candidates": candidates,
        "meta": {
            "source": "precomputed_cache",
            "window_count": int(np.count_nonzero(valid)),
            "segment_start": int(start),
            "segment_end": int(end),
            "onset_band": dict(onset_band or {}),
        },
    }


class VideoMAEHandler:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.labels: List[str] = []
        self.enable_verb_prior = False
        self.weights_path: Optional[str] = None
        self.verb_list_path: Optional[str] = None
        self._yaml = None
        self._torch = None
        self._Image = None
        self._np = None
        self._decord = None
        self._VideoMAEImageProcessor = None
        self._VideoMAEForVideoClassification = None
        self._VideoMAEModel = None
        self._VideoMAEConfig = None
        self.last_predict_meta = {}
        self.segment_feature_dim = 32
        self.feature_only = False

    @staticmethod
    def _unwrap_state_dict(raw_state: Any) -> Dict[str, Any]:
        state = raw_state
        for key in ("model", "state_dict", "module"):
            if isinstance(state, dict) and key in state and isinstance(state.get(key), dict):
                state = state.get(key)
        return dict(state) if isinstance(state, dict) else {}

    @staticmethod
    def _looks_like_timm_videomae_backbone(state_dict: Dict[str, Any]) -> bool:
        keys = list(state_dict.keys())
        if not keys:
            return False
        return any(str(key).startswith("patch_embed.proj.") for key in keys) and any(
            re.match(r"blocks\.\d+\.", str(key)) for key in keys
        )

    def _convert_timm_backbone_to_hf(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        converted: Dict[str, Any] = {}
        for key, value in dict(state_dict or {}).items():
            text = str(key)
            if text.startswith("patch_embed.proj."):
                suffix = text[len("patch_embed.proj.") :]
                converted[f"embeddings.patch_embeddings.projection.{suffix}"] = value
                continue
            if text.startswith("norm."):
                suffix = text[len("norm.") :]
                converted[f"layernorm.{suffix}"] = value
                continue
            match = re.match(r"blocks\.(\d+)\.(.+)", text)
            if not match:
                continue
            layer_idx = int(match.group(1))
            suffix = match.group(2)
            prefix = f"encoder.layer.{layer_idx}"
            if suffix.startswith("norm1."):
                converted[f"{prefix}.layernorm_before.{suffix[len('norm1.'):]}"] = value
                continue
            if suffix.startswith("norm2."):
                converted[f"{prefix}.layernorm_after.{suffix[len('norm2.'):]}"] = value
                continue
            if suffix == "attn.q_bias":
                converted[f"{prefix}.attention.attention.q_bias"] = value
                continue
            if suffix == "attn.v_bias":
                converted[f"{prefix}.attention.attention.v_bias"] = value
                continue
            if suffix == "attn.qkv.weight":
                try:
                    q_weight, k_weight, v_weight = value.chunk(3, dim=0)
                    converted[f"{prefix}.attention.attention.query.weight"] = q_weight
                    converted[f"{prefix}.attention.attention.key.weight"] = k_weight
                    converted[f"{prefix}.attention.attention.value.weight"] = v_weight
                except Exception:
                    pass
                continue
            if suffix.startswith("attn.proj."):
                converted[
                    f"{prefix}.attention.output.dense.{suffix[len('attn.proj.'):]}"
                ] = value
                continue
            if suffix.startswith("mlp.fc1."):
                converted[f"{prefix}.intermediate.dense.{suffix[len('mlp.fc1.'):]}"] = value
                continue
            if suffix.startswith("mlp.fc2."):
                converted[f"{prefix}.output.dense.{suffix[len('mlp.fc2.'):]}"] = value
                continue
        return converted

    def _compress_segment_feature(self, raw_vector, out_dim: Optional[int] = None) -> List[float]:
        out_dim = max(4, int(out_dim or self.segment_feature_dim or 32))
        if raw_vector is None:
            return [0.0] * out_dim
        try:
            arr = self._np.asarray(raw_vector, dtype=self._np.float32).reshape(-1)
        except Exception:
            return [0.0] * out_dim
        if arr.size <= 0:
            return [0.0] * out_dim
        if arr.size == out_dim:
            return [float(v) for v in arr.tolist()]
        edges = self._np.linspace(0, int(arr.size), num=out_dim + 1, dtype=int)
        reduced: List[float] = []
        for idx in range(out_dim):
            start = int(edges[idx])
            end = int(edges[idx + 1])
            if end <= start:
                pick = min(max(0, start), int(arr.size) - 1)
                reduced.append(float(arr[pick]))
                continue
            chunk = arr[start:end]
            reduced.append(float(chunk.mean()))
        return reduced

    def _extract_segment_feature(self, outputs) -> List[float]:
        feature_vec = None
        try:
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states:
                last_hidden = hidden_states[-1]
                if last_hidden is not None:
                    pooled = last_hidden.mean(dim=1).detach().cpu().numpy()
                    if pooled.ndim >= 2 and pooled.shape[0] > 0:
                        feature_vec = pooled[0]
        except Exception:
            feature_vec = None
        if feature_vec is None:
            try:
                logits = getattr(outputs, "logits", None)
                if logits is not None:
                    feature_vec = logits[0].detach().cpu().numpy()
            except Exception:
                feature_vec = None
        return self._compress_segment_feature(feature_vec, self.segment_feature_dim)

    def _ensure_fixed_sample_count(
        self,
        indices: List[int],
        *,
        start: int,
        end: int,
        num_samples: int,
    ) -> List[int]:
        target = max(1, int(num_samples or 1))
        clean = [
            max(int(start), min(int(end), int(idx)))
            for idx in list(indices or [])
        ]
        if not clean:
            clean = [int(start)]
        if len(clean) > target:
            return [int(v) for v in clean[:target]]
        if len(clean) < target:
            if int(start) == int(end):
                clean.extend([int(start)] * (target - len(clean)))
            else:
                extra = [
                    int(round(v))
                    for v in self._np.linspace(int(start), int(end), target).tolist()
                ]
                pos = 0
                while len(clean) < target:
                    clean.append(extra[min(pos, len(extra) - 1)])
                    pos += 1
        return [int(v) for v in clean[:target]]

    def _ensure_runtime(self):
        if (
            self._yaml is not None
            and self._torch is not None
            and self._Image is not None
            and self._np is not None
            and self._decord is not None
            and self._VideoMAEImageProcessor is not None
            and self._VideoMAEForVideoClassification is not None
        ):
            if self.device is None:
                self.device = self._torch.device(
                    "cuda" if self._torch.cuda.is_available() else "cpu"
                )
            return True, None
        try:
            import yaml
            import torch
            from PIL import Image
            import numpy as np
            import decord
            from transformers import (
                VideoMAEConfig,
                VideoMAEForVideoClassification,
                VideoMAEImageProcessor,
                VideoMAEModel,
            )
        except Exception as ex:
            return (
                False,
                "VideoMAE optional dependencies are missing. "
                "Install Pillow, PyYAML, transformers, and decord to enable action ranking.\n"
                f"{ex}",
            )
        self._yaml = yaml
        self._torch = torch
        self._Image = Image
        self._np = np
        self._decord = decord
        self._VideoMAEImageProcessor = VideoMAEImageProcessor
        self._VideoMAEForVideoClassification = VideoMAEForVideoClassification
        self._VideoMAEModel = VideoMAEModel
        self._VideoMAEConfig = VideoMAEConfig
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return True, None

    def load_model(self, weights_path, verb_list_path=None):
        """Load the frozen video encoder and optional custom verb labels."""
        ok, err = self._ensure_runtime()
        if not ok:
            self.model = None
            self.processor = None
            return False, err

        self.weights_path = weights_path
        self.verb_list_path = verb_list_path
        self.feature_only = False

        self.labels = []
        if verb_list_path and os.path.exists(verb_list_path):
            with open(verb_list_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            data = self._yaml.safe_load(raw_text)
            if isinstance(data, dict) and "names" in data:
                names = data["names"]
                try:
                    ordered_keys = sorted(names.keys(), key=lambda x: int(x))
                except Exception:
                    ordered_keys = sorted(names.keys(), key=lambda x: str(x))
                self.labels = [str(names[i]) for i in ordered_keys]
            elif isinstance(data, list):
                self.labels = [str(v) for v in data]
            elif isinstance(data, str):
                self.labels = [line.strip() for line in raw_text.splitlines() if line.strip()]
        num_labels = len(self.labels)
        self.processor = self._VideoMAEImageProcessor()
        loaded_message = "Loaded frozen video encoder in feature-only mode."

        try:
            if weights_path and os.path.exists(weights_path):
                raw_state = self._torch.load(weights_path, map_location=self.device)
                state_dict = self._unwrap_state_dict(raw_state)
                if self._looks_like_timm_videomae_backbone(state_dict):
                    self.model = self._VideoMAEModel(self._VideoMAEConfig())
                    converted = self._convert_timm_backbone_to_hf(state_dict)
                    self.model.load_state_dict(converted, strict=False)
                    self.labels = []
                    self.enable_verb_prior = False
                    self.feature_only = True
                    loaded_message = (
                        "Loaded local generic VideoMAE encoder weights in feature-only mode."
                    )
                else:
                    self.model = self._VideoMAEForVideoClassification(
                        self._VideoMAEConfig(num_labels=max(1, int(num_labels or 1)))
                    )
                    self.model.load_state_dict(state_dict, strict=False)
                    self.enable_verb_prior = bool(self.labels)
                    self.feature_only = not bool(self.enable_verb_prior)
                    if self.enable_verb_prior:
                        loaded_message = f"Loaded local classification model with {num_labels} labels."
            else:
                model_id = "OpenGVLab/video-mae-v2-base-kinetics"
                self.processor = self._VideoMAEImageProcessor.from_pretrained(model_id)
                if num_labels > 0:
                    self.model = self._VideoMAEForVideoClassification.from_pretrained(
                        model_id,
                        num_labels=num_labels,
                        ignore_mismatched_sizes=True,
                    )
                else:
                    self.model = self._VideoMAEForVideoClassification.from_pretrained(
                        model_id
                    )
                    self.labels = []
                self.enable_verb_prior = bool(self.labels)
                self.feature_only = not bool(self.enable_verb_prior)
                if self.enable_verb_prior:
                    loaded_message = f"Loaded model with {num_labels} labels."
            self.model.to(self.device)
            self.model.eval()
            return True, loaded_message
        except Exception as ex:
            self.model = None
            self.processor = None
            self.enable_verb_prior = False
            self.feature_only = False
            return False, str(ex)

    def predict(
        self,
        video_path,
        start_frame,
        end_frame,
        top_k=5,
        *,
        onset_frame=None,
        onset_band=None,
    ):
        """Predict action labels for a given segment."""
        ok, err = self._ensure_runtime()
        if not ok:
            return None, err
        if self.model is None or self.processor is None:
            return None, "Model not loaded."

        try:
            vr = self._decord.VideoReader(video_path, num_threads=1)
            num_frames_available = len(vr)
            start = max(0, int(start_frame))
            end = min(num_frames_available - 1, int(end_frame))
            num_samples = 16
            indices = build_temporal_sample_indices(
                start,
                end,
                num_samples=num_samples,
                onset_band=onset_band,
            )
            if not indices:
                indices = self._np.linspace(start, end, num_samples).astype(int).tolist()
            indices = self._ensure_fixed_sample_count(
                indices,
                start=int(start),
                end=int(end),
                num_samples=int(num_samples),
            )
            video = vr.get_batch(indices).asnumpy()
            frames = [self._Image.fromarray(frame) for frame in video]
            inputs = self.processor(list(frames), return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with self._torch.no_grad():
                try:
                    outputs = self.model(**inputs, output_hidden_states=True)
                except TypeError:
                    outputs = self.model(**inputs)
                logits = getattr(outputs, "logits", None)
                top_probs = None
                top_indices = None
                if self.enable_verb_prior and self.labels and logits is not None:
                    probs = self._torch.nn.functional.softmax(logits, dim=-1)
                    top_probs, top_indices = self._torch.topk(
                        probs, k=min(top_k, len(self.labels))
                    )
                segment_feature = self._extract_segment_feature(outputs)

            results = []
            if self.enable_verb_prior and self.labels and top_probs is not None and top_indices is not None:
                for i in range(top_probs.size(1)):
                    idx = top_indices[0, i].item()
                    results.append(
                        {
                            "label": self.labels[idx]
                            if idx < len(self.labels)
                            else f"Unknown_{idx}",
                            "score": top_probs[0, i].item(),
                        }
                    )
            self.last_predict_meta = {
                "sampling_mode": "onset_aware" if onset_band else "uniform_segment",
                "feature_only": bool(not self.enable_verb_prior),
                "indices": [int(v) for v in list(indices or [])],
                "segment_start": int(start),
                "segment_end": int(end),
                "onset_frame": None if onset_frame is None else int(onset_frame),
                "onset_band": dict(onset_band or {}),
                "segment_feature": list(segment_feature or []),
                "segment_feature_dim": int(len(segment_feature or [])),
            }
            return results, None
        except Exception as ex:
            self.last_predict_meta = {}
            return None, str(ex)

    def build_dense_feature_cache(
        self,
        video_path,
        *,
        stride: int = 1,
        window_span: int = 16,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        ok, err = self._ensure_runtime()
        if not ok:
            return None, err
        if self.model is None or self.processor is None:
            return None, "Model not loaded."

        stride = max(1, int(stride or 1))
        window_span = max(16, int(window_span or 16))

        try:
            vr = self._decord.VideoReader(video_path, num_threads=1)
            frame_count = len(vr)
            if frame_count <= 0:
                return None, "Video has no readable frames."
            clip_start = max(0, int(start_frame or 0))
            clip_end = frame_count - 1 if end_frame is None else min(frame_count - 1, int(end_frame))
            if clip_end < clip_start:
                clip_start, clip_end = clip_end, clip_start

            centers = list(range(int(clip_start), int(clip_end) + 1, stride))
            if not centers or centers[-1] != int(clip_end):
                centers.append(int(clip_end))

            window_starts: List[int] = []
            window_ends: List[int] = []
            window_centers: List[int] = []
            feature_rows: List[List[float]] = []
            score_rows: List[List[float]] = []

            for center in centers:
                half = max(1, int(round(window_span / 2.0)))
                seg_start = max(int(clip_start), int(center) - half)
                seg_end = min(int(clip_end), seg_start + window_span - 1)
                if seg_end - seg_start + 1 < window_span:
                    seg_start = max(int(clip_start), seg_end - window_span + 1)
                indices = build_temporal_sample_indices(
                    seg_start,
                    seg_end,
                    num_samples=16,
                    onset_band=None,
                )
                if not indices:
                    continue
                indices = self._ensure_fixed_sample_count(
                    indices,
                    start=int(seg_start),
                    end=int(seg_end),
                    num_samples=16,
                )
                video = vr.get_batch(indices).asnumpy()
                frames = [self._Image.fromarray(frame) for frame in video]
                inputs = self.processor(list(frames), return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with self._torch.no_grad():
                    try:
                        outputs = self.model(**inputs, output_hidden_states=True)
                    except TypeError:
                        outputs = self.model(**inputs)
                    logits = getattr(outputs, "logits", None)
                    feature_rows.append(self._extract_segment_feature(outputs))
                    if self.enable_verb_prior and self.labels and logits is not None:
                        probs = self._torch.nn.functional.softmax(logits, dim=-1)
                        score_rows.append(
                            [float(v) for v in probs[0].detach().cpu().numpy().tolist()]
                        )
                window_starts.append(int(seg_start))
                window_ends.append(int(seg_end))
                window_centers.append(int(center))

            if not feature_rows:
                return None, "No windows were extracted from the video."

            meta = {
                "version": "VIDEOMAE_PRECOMPUTED_CACHE_V1",
                "video_path": os.path.abspath(str(video_path or "").strip()),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "weights_path": str(self.weights_path or ""),
                "verb_list_path": str(self.verb_list_path or ""),
                "labels": list(self.labels or []),
                "feature_dim": int(len(feature_rows[0]) if feature_rows else 0),
                "window_span": int(window_span),
                "stride": int(stride),
                "frame_count": int(frame_count),
                "start_frame": int(clip_start),
                "end_frame": int(clip_end),
            }
            return {
                "meta": meta,
                "window_starts": window_starts,
                "window_ends": window_ends,
                "window_centers": window_centers,
                "segment_features": feature_rows,
                "verb_scores": score_rows,
            }, None
        except Exception as ex:
            return None, str(ex)

    @staticmethod
    def save_dense_feature_cache(payload: Dict[str, Any], output_path: str) -> Tuple[bool, str]:
        target_path = str(output_path or "").strip()
        if not target_path:
            return False, "Missing output path."
        try:
            import numpy as np
        except Exception as ex:
            return False, f"NumPy is required to save cache files.\n{ex}"
        try:
            os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
            meta = dict((payload or {}).get("meta") or {})
            segment_features = np.asarray(
                (payload or {}).get("segment_features") or [], dtype=np.float32
            )
            verb_scores = np.asarray(
                (payload or {}).get("verb_scores") or [], dtype=np.float32
            )
            np.savez_compressed(
                target_path,
                meta_json=np.asarray([json.dumps(meta, ensure_ascii=True)], dtype=f"<U{max(32, len(json.dumps(meta, ensure_ascii=True)) + 1)}"),
                window_starts=np.asarray((payload or {}).get("window_starts") or [], dtype=np.int32),
                window_ends=np.asarray((payload or {}).get("window_ends") or [], dtype=np.int32),
                window_centers=np.asarray((payload or {}).get("window_centers") or [], dtype=np.int32),
                segment_features=segment_features,
                verb_scores=verb_scores,
            )
            return True, target_path
        except Exception as ex:
            return False, str(ex)
