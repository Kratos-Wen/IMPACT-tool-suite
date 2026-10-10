from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import defaultdict
from PyQt5.QtWidgets import (
    QAction,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QFileDialog,
    QMessageBox,
    QComboBox,
    QToolButton,
    QSpinBox,
    QSlider,
    QCheckBox,
    QShortcut,
    QMenu,
    QGroupBox,
    QFormLayout,
    QLineEdit,
    QSplitter,
    QListView,
    QSizePolicy,
    QInputDialog,
    QFrame,
    QDialog,
    QGridLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QDialogButtonBox,
    QRadioButton,
    QButtonGroup,
    QProgressDialog,
    QApplication,
    QTabWidget,
    QScrollArea,
    QTextBrowser,
    QGraphicsDropShadowEffect,
)
from ui.mixins import FrameControlMixin
from ui.autosave import AutosaveMixin
from ui.label_glossary import LabelGlossaryMixin, decorate_combo
from core.safe_storage import atomic_json
from ui.correction_propagation import CorrectionPropagationMixin
from ui.assembly_editor import AssemblyEditorMixin
from ui.frame_review import FrameReviewMixin
from core.assembly_timeline import (validate_timeline, resolve_object, noun_at, state_at,
    empty_timeline, reference_id, active_id, merge_at, track_segments)
from core.noun_aliases import normalize_noun_aliases
from core.anomaly_attributes import normalize as normalize_attributes, export_review, display_value, is_positive
from ui.attribute_selector import AttributeSelector
from ui.annotation_policy import AnnotationPolicyMixin, onset_resolved
from core.frame_review import valid_record
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer, QEvent
from PyQt5.QtWidgets import QStyle
from PyQt5.QtGui import QKeySequence, QColor, QPixmap
import copy
import json
import hashlib
from ui.video_player import VideoPlayer
from ui.label_panel import LabelPanel
from core.models import LabelDef
from core.hoi_query_controller import (
    apply_field_suggestion,
    clear_field_suggestion,
    clear_field_value,
    ensure_hand_annotation_state,
    get_field_state,
    get_field_suggestion,
    hydrate_existing_field_state,
    set_field_confirmation,
    set_field_suggestion,
)
from core.hoi_empirical_calibration import HOIEmpiricalCalibrator
from core.hoi_runtime_kernel import (
    SemanticRuntimeRequest,
    build_runtime_query_candidates,
    run_event_local_semantic_decode,
)
from core.hoi_ontology import (
    HOIOntology,
    NO_NOUN_TOKEN,
    filter_allowed_object_candidates,
    ontology_allowed_noun_ids,
    ontology_noun_required,
)
from core.onset_guidance import build_local_onset_window, build_onset_band
from core.semantic_adapter import (
    load_adapter_package,
    train_adapter_from_feedback,
)
from core.structured_event_graph import build_hoi_event_graph, save_event_graph_sidecar
from utils.constants import PRESET_COLORS, color_from_key
from utils.shortcut_settings import (
    load_shortcut_bindings,
    default_shortcut_bindings,
    shortcut_value,
    set_shortcut_key,
    load_ui_preferences,
    save_ui_preferences,
    save_logging_policy,
)
from utils.op_logger import OperationLogger
from ui.hoi_timeline import HOITimeline
from ui.widgets import ToggleSwitch, ClickToggleList
import os
import cv2
import re
import yaml
import time
import math
import urllib.request
from datetime import datetime
from core.videomae_v2_logic import (
    VideoMAEHandler,
    aggregate_precomputed_feature_cache,
    load_precomputed_feature_cache,
)

_NO_FIELD_VALUE = object()
from core.project_profile import PROFILE as _PROJECT_PROFILE
_DEFAULT_ANOMALY_LABEL = str(_PROJECT_PROFILE.get("default_anomaly_label", "unreviewed"))
_ANOMALY_LABEL_OPTIONS = tuple(_PROJECT_PROFILE.get("anomaly_labels") or [_DEFAULT_ANOMALY_LABEL])
_ANOMALY_LABEL_ALIASES = dict(_PROJECT_PROFILE.get("anomaly_aliases") or {})

def _apply_task_specific_category_mapping(category, *, for_detection=False):
    ignored = set(_PROJECT_PROFILE.get("detection_ignored_categories") or [])
    if for_detection and category in ignored:
        return ""
    return (_PROJECT_PROFILE.get("detection_matching_aliases") or {}).get(category, category)


def _anomaly_label_lookup_keys(value: Any) -> Tuple[str, str]:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    loose = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text, loose


def _default_mediapipe_runtime_dir(video_path: str = "") -> str:
    base = (
        os.path.dirname(os.path.abspath(str(video_path or "").strip()))
        if str(video_path or "").strip()
        else os.getcwd()
    )
    path = os.path.join(base, "runtime_artifacts", "mediapipe_models")
    os.makedirs(path, exist_ok=True)
    return path


def _mediapipe_hand_landmarker_model_file(video_path: str = "") -> str:
    out_path = os.path.join(
        _default_mediapipe_runtime_dir(video_path),
        "hand_landmarker.task",
    )
    if os.path.isfile(out_path):
        return out_path
    url = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )
    urllib.request.urlretrieve(url, out_path)
    return out_path


def _mediapipe_anatomical_label(value):
    label=str(value or '').strip().lower()
    # https://chuoling.github.io/mediapipe/solutions/hands.html#multi_handedness
    if not _PROJECT_PROFILE.get('video_mirrored',False):
        label={'left':'right','right':'left'}.get(label,label)
    return label


def _extract_hand_detections_tasks(results, width: int, height: int) -> List[Dict[str, Any]]:
    if not results:
        return []
    hand_landmarks = list(getattr(results, "hand_landmarks", []) or [])
    handedness_rows = list(getattr(results, "handedness", []) or [])
    detections: List[Dict[str, Any]] = []
    for idx, hand_lms in enumerate(hand_landmarks):
        xs = [float(getattr(lm, "x", 0.0) or 0.0) for lm in list(hand_lms or [])]
        ys = [float(getattr(lm, "y", 0.0) or 0.0) for lm in list(hand_lms or [])]
        if not xs or not ys:
            continue
        x1 = max(0.0, min(float(width), min(xs) * float(width)))
        y1 = max(0.0, min(float(height), min(ys) * float(height)))
        x2 = max(0.0, min(float(width), max(xs) * float(width)))
        y2 = max(0.0, min(float(height), max(ys) * float(height)))
        label = ""
        score = 0.0
        if idx < len(handedness_rows) and list(handedness_rows[idx] or []):
            cat = list(handedness_rows[idx] or [])[0]
            label = str(getattr(cat, "category_name", "") or "").strip().lower()
            try:
                score = float(getattr(cat, "score", 0.0) or 0.0)
            except Exception:
                score = 0.0
        detections.append(
            {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "cx": float(0.5 * (x1 + x2)),
                "cy": float(0.5 * (y1 + y2)),
                "area": float(max(1.0, max(0.0, x2 - x1) * max(0.0, y2 - y1))),
                "handedness": _mediapipe_anatomical_label(label),
                "handedness_score": float(score),
                "detection_confidence": float(score),
            }
        )
    return detections


class EditableTitleGroupBox(QGroupBox):
    """Group box whose title can be renamed by double-clicking the title band."""

    titleEdited = pyqtSignal(str)

    def __init__(self, title="", parent=None):
        super().__init__(title, parent)
        self._title_editor = QLineEdit(self)
        self._title_editor.hide()
        self._title_editor.editingFinished.connect(self._finish_title_edit)
        self.setToolTip("Double-click the title to rename it.")

    def _title_edit_geometry(self):
        fm = self.fontMetrics()
        left = 12
        top = 2
        height = max(24, fm.height() + 10)
        width = max(
            180,
            min(self.width() - 24, fm.horizontalAdvance(self.title() or "Title") + 36),
        )
        return left, top, width, height

    def _begin_title_edit(self):
        left, top, width, height = self._title_edit_geometry()
        self._title_editor.setGeometry(left, top, width, height)
        self._title_editor.setText(self.title())
        self._title_editor.show()
        self._title_editor.raise_()
        self._title_editor.setFocus()
        self._title_editor.selectAll()

    def _finish_title_edit(self):
        if not self._title_editor.isVisible():
            return
        new_title = self._title_editor.text().strip() or self.title()
        self._title_editor.hide()
        if new_title != self.title():
            self.setTitle(new_title)
            self.titleEdited.emit(new_title)

    def mouseDoubleClickEvent(self, event):
        left, top, width, height = self._title_edit_geometry()
        if left <= event.pos().x() <= left + width and top <= event.pos().y() <= top + height:
            self._begin_title_edit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._title_editor.isVisible():
            left, top, width, height = self._title_edit_geometry()
            self._title_editor.setGeometry(left, top, width, height)


class ActionLabelSelector(QDialog):
    def __init__(self, candidates, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Verb Suggestion")
        self.layout = QVBoxLayout(self)
        self.selected_label = None
        
        self.layout.addWidget(QLabel("Multiple high-confidence verb suggestions are available. Please select one:"))
        
        self.group = QButtonGroup(self)
        for i, cand in enumerate(candidates):
            rb = QRadioButton(f"{cand['label']} ({cand['score']:.2%})")
            if i == 0:
                rb.setChecked(True)
            self.group.addButton(rb, i)
            self.layout.addWidget(rb)
        
        self.btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btns.accepted.connect(self.accept)
        self.btns.rejected.connect(self.reject)
        self.layout.addWidget(self.btns)
        
        self.candidates = candidates

    def get_selected_label(self):
        idx = self.group.checkedId()
        if idx >= 0:
            return self.candidates[idx]['label']
        return self.candidates[0]['label']

class YoloTrainDialog(QDialog):
    def __init__(self, parent=None, default_epochs=10):
        super().__init__(parent)
        self.setWindowTitle("YOLO Incremental Training Config")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        form = QFormLayout()
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(default_epochs)
        form.addRow("Epochs:", self.epochs_spin)
        
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(8)
        form.addRow("Batch Size:", self.batch_spin)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(320, 1280)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        form.addRow("Image Size:", self.imgsz_spin)

        self.lr_edit = QLineEdit("0.01")
        form.addRow("Learning Rate (Fine-tuning):", self.lr_edit)
        
        self.train_split_spin = QSpinBox()
        self.train_split_spin.setRange(50, 95)
        self.train_split_spin.setValue(80)
        self.train_split_spin.setSuffix("% Train")
        form.addRow("Train/Val Split:", self.train_split_spin)

        self.output_dir_edit = QLineEdit(os.path.join(os.getcwd(), "incremental_train"))
        btn_out = QPushButton("...")
        btn_out.setFixedWidth(30)
        btn_out.clicked.connect(self._choose_out)
        h_out = QHBoxLayout()
        h_out.addWidget(self.output_dir_edit)
        h_out.addWidget(btn_out)
        form.addRow("Dataset Output Dir:", h_out)
        
        layout.addLayout(form)
        
        self.btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btn_box.accepted.connect(self.accept)
        self.btn_box.rejected.connect(self.reject)
        layout.addWidget(self.btn_box)

    def _choose_out(self):
        d = QFileDialog.getExistingDirectory(self, "Choose Output Directory")
        if d:
            self.output_dir_edit.setText(d)

    def get_config(self):
        return {
            "epochs": self.epochs_spin.value(),
            "batch": self.batch_spin.value(),
            "imgsz": self.imgsz_spin.value(),
            "lr0": float(self.lr_edit.text() or 0.01),
            "train_split": self.train_split_spin.value() / 100.0,
            "output_dir": self.output_dir_edit.text()
        }


class YoloTrainWorker(QThread):
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(str)

    def __init__(self, weights_path, data_yaml, config):
        super().__init__()
        self.weights_path = weights_path
        self.data_yaml = data_yaml
        self.config = config

    def run(self):
        try:
            from ultralytics import YOLO
            import torch
            
            self.progress.emit("Loading model...")
            model = YOLO(self.weights_path)
            
            self.progress.emit(f"Starting training for {self.config['epochs']} epochs...")
            # Note: project and name combined determine save dir
            results = model.train(
                data=self.data_yaml,
                epochs=self.config['epochs'],
                imgsz=self.config['imgsz'],
                batch=self.config['batch'],
                lr0=self.config['lr0'],
                device='cuda' if torch.cuda.is_available() else 'cpu',
                project=self.config['output_dir'],
                name='train_run',
                workers=0, # Avoid multiprocessing issues on Windows
                exist_ok=True
            )
            self.finished.emit(True, f"Training complete. Results saved to {results.save_dir}")
        except Exception as e:
            self.finished.emit(False, str(e))


class SemanticAdapterTrainWorker(QThread):
    finished = pyqtSignal(bool, str, object)
    progress = pyqtSignal(str)

    def __init__(
        self,
        feedback_path,
        model_path,
        feature_dim,
        verb_labels,
        noun_ids,
        config=None,
        init_model_path: str = "",
    ):
        super().__init__()
        self.feedback_path = feedback_path
        self.model_path = model_path
        self.feature_dim = int(feature_dim)
        self.verb_labels = [str(v) for v in list(verb_labels or [])]
        self.noun_ids = [int(v) for v in list(noun_ids or [])]
        self.config = dict(config or {})
        self.init_model_path = str(init_model_path or "").strip()

    def run(self):
        try:
            self.progress.emit("Training semantic adapter...")
            ok, msg, package = train_adapter_from_feedback(
                feedback_path=self.feedback_path,
                output_path=self.model_path,
                feature_dim=self.feature_dim,
                verb_labels=self.verb_labels,
                noun_ids=self.noun_ids,
                hidden_dim=int(self.config.get("hidden_dim", 96) or 96),
                onset_bins=int(self.config.get("onset_bins", 21) or 21),
                feature_layout=dict(self.config.get("feature_layout") or {}),
                video_adapter_rank=int(self.config.get("video_adapter_rank", 0) or 0),
                video_adapter_alpha=float(self.config.get("video_adapter_alpha", 0.0) or 0.0),
                epochs=int(self.config.get("epochs", 12) or 12),
                batch_size=int(self.config.get("batch_size", 16) or 16),
                lr=float(self.config.get("lr", 1e-3) or 1e-3),
                min_samples=int(self.config.get("min_samples", 8) or 8),
                init_package_path=self.init_model_path,
            )
            self.finished.emit(bool(ok), str(msg), package)
        except Exception as ex:
            self.finished.emit(False, str(ex), None)


class YoloInferenceWorker(QThread):
    finished = pyqtSignal(object)

    def __init__(self, model, requests, conf: float, iou: float, class_map=None):
        super().__init__()
        self.model = model
        self.requests = [dict(row or {}) for row in list(requests or [])]
        self.conf = float(conf)
        self.iou = float(iou)
        self.class_map = dict(class_map or {})

    def _class_name_for_id(self, cls_id: int):
        # Class IDs belong to the checkpoint; editor taxonomy order is independent.
        class_name = None
        if hasattr(self.model, 'names'):
            names = self.model.names
            if isinstance(names, dict):
                class_name = names.get(cls_id, names.get(str(cls_id)))
            else:
                try: class_name = names[int(cls_id)]
                except (IndexError, TypeError, ValueError): pass
        if class_name is None:
            class_name = self.class_map.get(cls_id, self.class_map.get(str(cls_id)))
        if class_name is None:return None
        from core.annotation_migration import canonical_noun
        return canonical_noun(str(class_name).lower(), _PROJECT_PROFILE)

    def _predict_once(self, frame_bgr, device: str):
        try:
            return self.model.predict(
                source=frame_bgr,
                conf=self.conf,
                iou=self.iou,
                device=device,
                verbose=False,
            ), None
        except Exception as ex:
            if device == "cuda":
                try:
                    return self.model.predict(
                        source=frame_bgr,
                        conf=self.conf,
                        iou=self.iou,
                        device="cpu",
                        verbose=False,
                    ), None
                except Exception as ex2:
                    return None, str(ex2)
            return None, str(ex)

    def run(self):
        payload = {
            "success": False,
            "error": "",
            "results": [],
            "frames": len(self.requests),
        }
        if self.model is None:
            payload["error"] = "YOLO model not loaded."
            self.finished.emit(payload)
            return

        device = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            device = "cpu"

        for request in self.requests:
            frame_bgr = request.get("frame_bgr")
            frame_idx = request.get("frame_idx")
            result_rows, err = self._predict_once(frame_bgr, device)
            if err:
                payload["error"] = err
                self.finished.emit(payload)
                return

            detections = []
            if result_rows:
                result = result_rows[0]
                boxes = getattr(result, "boxes", None)
                if boxes is not None and len(boxes) > 0:
                    try:
                        xyxy = boxes.xyxy.cpu().numpy()
                        cls_ids = boxes.cls.cpu().numpy().astype(int)
                        confs = (
                            boxes.conf.cpu().numpy()
                            if getattr(boxes, "conf", None) is not None
                            else None
                        )
                    except Exception as ex:
                        payload["error"] = f"Failed to parse YOLO outputs: {ex}"
                        self.finished.emit(payload)
                        return
                    for i in range(len(xyxy)):
                        cls_id = int(cls_ids[i])
                        class_name = self._class_name_for_id(cls_id)
                        detections.append(
                            {
                                "class_id": cls_id,
                                "class_name": class_name,
                                "xyxy": [float(v) for v in list(xyxy[i])],
                                "confidence": (
                                    None
                                    if confs is None
                                    else float(confs[i])
                                ),
                            }
                        )

            request_out = {
                "frame_idx": frame_idx,
                "include_ids": None
                if request.get("include_ids") is None
                else list(request.get("include_ids") or []),
                "replace_existing": bool(request.get("replace_existing")),
                "click_point": (
                    None
                    if request.get("click_point") is None
                    else list(request.get("click_point") or [])
                ),
                "auto_select": bool(request.get("auto_select")),
                "detections": detections,
            }
            payload["results"].append(request_out)

        payload["success"] = True
        self.finished.emit(payload)


class VideoMAEInferenceWorker(QThread):
    finished = pyqtSignal(object)

    def __init__(self, handler, request):
        super().__init__()
        self.handler = handler
        self.request = dict(request or {})

    def run(self):
        payload = {
            "success": False,
            "error": "",
            "request": dict(self.request),
            "candidates": [],
            "local_candidates": [],
            "segment_meta": {},
            "local_meta": {},
        }
        if self.handler is None:
            payload["error"] = "VideoMAE handler is unavailable."
            self.finished.emit(payload)
            return

        try:
            if bool(self.request.get("need_global")):
                raw_candidates, err = self.handler.predict(
                    self.request.get("video_path"),
                    int(self.request.get("start")),
                    int(self.request.get("end")),
                    onset_frame=(self.request.get("onset_context") or {}).get("onset_frame"),
                    onset_band=(self.request.get("onset_context") or {}).get("onset_band"),
                )
                if err:
                    payload["error"] = err
                    self.finished.emit(payload)
                    return
                payload["candidates"] = list(raw_candidates or [])
                payload["segment_meta"] = dict(self.handler.last_predict_meta or {})

            local_context = dict(self.request.get("local_onset_context") or {})
            if bool(self.request.get("need_local")) and local_context:
                raw_local, local_err = self.handler.predict(
                    self.request.get("video_path"),
                    int(local_context.get("start_frame")),
                    int(local_context.get("end_frame")),
                    onset_frame=local_context.get("center_frame"),
                    onset_band=local_context,
                )
                if not local_err and raw_local:
                    payload["local_candidates"] = list(raw_local or [])
                    payload["local_meta"] = dict(self.handler.last_predict_meta or {})

            payload["success"] = True
        except Exception as ex:
            payload["error"] = str(ex)

        self.finished.emit(payload)


class HandTrackBuildWorker(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(str)

    def __init__(
        self,
        video_path: str,
        actors_config: Optional[Sequence[Dict[str, Any]]] = None,
        *,
        max_hands: int = 2,
        det_conf: float = 0.5,
        track_conf: float = 0.5,
    ):
        super().__init__()
        self.video_path = str(video_path or "").strip()
        self.actors_config = [dict(row or {}) for row in list(actors_config or [])]
        self.max_hands = max(1, int(max_hands or 2))
        self.det_conf = float(det_conf or 0.5)
        self.track_conf = float(track_conf or 0.5)

    @staticmethod
    def _dynamic_stride(frame_count: int) -> int:
        frame_count = max(0, int(frame_count or 0))
        if frame_count <= 900:
            return 1
        if frame_count <= 2400:
            return 2
        if frame_count <= 5400:
            return 3
        return 4

    @staticmethod
    def _diag(width: int, height: int) -> float:
        width = max(1, int(width or 1))
        height = max(1, int(height or 1))
        return max(1.0, float((width ** 2 + height ** 2) ** 0.5))

    @staticmethod
    def _interp(a: float, b: float, alpha: float) -> float:
        return float(a) + (float(b) - float(a)) * float(alpha)

    def _actor_ids(self) -> Tuple[str, str]:
        left_actor = (
            str((self.actors_config[0] or {}).get("id") or "Left_hand")
            if self.actors_config
            else "Left_hand"
        )
        if len(self.actors_config) > 1:
            right_actor = str((self.actors_config[1] or {}).get("id") or left_actor)
        else:
            right_actor = left_actor
        return left_actor, right_actor

    def _extract_detections(self, results, width: int, height: int) -> List[Dict[str, Any]]:
        if not results or not getattr(results, "multi_hand_landmarks", None):
            return []
        handedness = []
        if getattr(results, "multi_handedness", None):
            for hd in list(results.multi_handedness or []):
                label = ""
                score = 0.0
                try:
                    label = str(hd.classification[0].label or "").strip().lower()
                except Exception:
                    label = ""
                try:
                    score = float(hd.classification[0].score or 0.0)
                except Exception:
                    score = 0.0
                handedness.append({"label": label, "score": score})
        while len(handedness) < len(list(results.multi_hand_landmarks or [])):
            handedness.append({"label": "", "score": 0.0})

        detections: List[Dict[str, Any]] = []
        for idx, hand_lms in enumerate(list(results.multi_hand_landmarks or [])):
            xs = [float(lm.x) for lm in list(hand_lms.landmark or [])]
            ys = [float(lm.y) for lm in list(hand_lms.landmark or [])]
            if not xs or not ys:
                continue
            x1 = max(0.0, min(float(width), min(xs) * float(width)))
            y1 = max(0.0, min(float(height), min(ys) * float(height)))
            x2 = max(0.0, min(float(width), max(xs) * float(width)))
            y2 = max(0.0, min(float(height), max(ys) * float(height)))
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            area = max(1.0, float(max(0.0, x2 - x1) * max(0.0, y2 - y1)))
            hand_info = dict(handedness[idx] or {}) if idx < len(handedness) else {}
            detections.append(
                {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "cx": float(cx),
                    "cy": float(cy),
                    "area": float(area),
                    "handedness": _mediapipe_anatomical_label(hand_info.get("label")),
                    "handedness_score": float(hand_info.get("score") or 0.0),
                    "detection_confidence": float(hand_info.get("score") or 0.0),
                }
            )
        return detections

    def _extract_detections_tasks(self, results, width: int, height: int) -> List[Dict[str, Any]]:
        return _extract_hand_detections_tasks(results, width, height)

    def _assign_detections(
        self,
        detections: Sequence[Dict[str, Any]],
        prev_centers: Dict[str, Tuple[float, float]],
    ) -> Dict[str, Dict[str, Any]]:
        left_actor, right_actor = self._actor_ids()
        assignments: Dict[str, Dict[str, Any]] = {}
        unresolved: List[Dict[str, Any]] = []
        for det in list(detections or []):
            hint = str(det.get("handedness") or "").strip().lower()
            desired = None
            if hint.startswith("left"):
                desired = left_actor
            elif hint.startswith("right"):
                desired = right_actor
            if desired and desired not in assignments:
                assignments[desired] = dict(det)
            else:
                unresolved.append(dict(det))
        remaining = [aid for aid in (left_actor, right_actor) if aid not in assignments]
        if unresolved and remaining:
            unresolved.sort(key=lambda row: float(row.get("cx", 0.0)))
            for det in list(unresolved):
                best_actor = None
                best_dist = None
                for actor_id in list(remaining):
                    center = prev_centers.get(actor_id)
                    if center is None:
                        continue
                    dist = (
                        (float(det.get("cx", 0.0)) - float(center[0])) ** 2
                        + (float(det.get("cy", 0.0)) - float(center[1])) ** 2
                    )
                    if best_dist is None or dist < best_dist:
                        best_actor = actor_id
                        best_dist = dist
                if best_actor is None and remaining:
                    best_actor = remaining[0] if len(remaining) == 1 else (
                        left_actor if float(det.get("cx", 0.0)) <= float(unresolved[-1].get("cx", 0.0)) else right_actor
                    )
                    if best_actor not in remaining:
                        best_actor = remaining[0]
                if best_actor is None:
                    continue
                assignments[best_actor] = dict(det)
                if best_actor in remaining:
                    remaining.remove(best_actor)
                if not remaining:
                    break
        return assignments

    def _finalize_track(
        self,
        rows: Dict[int, Dict[str, Any]],
        *,
        frame_count: int,
        stride: int,
        width: int,
        height: int,
    ) -> Dict[str, Any]:
        records = {int(k): dict(v or {}) for k, v in dict(rows or {}).items()}
        keys = sorted(records.keys())
        max_gap = max(6, int(stride) * 3)
        for idx in range(len(keys) - 1):
            f1 = int(keys[idx])
            f2 = int(keys[idx + 1])
            if f2 <= f1 + 1 or (f2 - f1) > max_gap:
                continue
            a = dict(records.get(f1) or {})
            b = dict(records.get(f2) or {})
            for frame in range(f1 + 1, f2):
                alpha = float(frame - f1) / float(max(1, f2 - f1))
                records[frame] = {
                    "x1": self._interp(a.get("x1", 0.0), b.get("x1", 0.0), alpha),
                    "y1": self._interp(a.get("y1", 0.0), b.get("y1", 0.0), alpha),
                    "x2": self._interp(a.get("x2", 0.0), b.get("x2", 0.0), alpha),
                    "y2": self._interp(a.get("y2", 0.0), b.get("y2", 0.0), alpha),
                    "cx": self._interp(a.get("cx", 0.0), b.get("cx", 0.0), alpha),
                    "cy": self._interp(a.get("cy", 0.0), b.get("cy", 0.0), alpha),
                    "area": self._interp(a.get("area", 1.0), b.get("area", 1.0), alpha),
                    "handedness": str(a.get("handedness") or b.get("handedness") or ""),
                    "handedness_score": self._interp(
                        a.get("handedness_score", 0.0),
                        b.get("handedness_score", 0.0),
                        alpha,
                    ),
                    "detection_confidence": self._interp(
                        a.get("detection_confidence", 0.0),
                        b.get("detection_confidence", 0.0),
                        alpha,
                    ),
                    "interpolated": True,
                }
        diag = self._diag(width, height)
        motion_rows: List[Tuple[int, float]] = []
        prev_entry = None
        for frame in sorted(records.keys()):
            row = dict(records.get(frame) or {})
            motion = 0.0
            if prev_entry is not None:
                dx = float(row.get("cx", 0.0)) - float(prev_entry.get("cx", 0.0))
                dy = float(row.get("cy", 0.0)) - float(prev_entry.get("cy", 0.0))
                spatial = ((dx ** 2 + dy ** 2) ** 0.5) / float(diag)
                prev_area = max(1.0, float(prev_entry.get("area", 1.0) or 1.0))
                curr_area = max(1.0, float(row.get("area", 1.0) or 1.0))
                area_delta = abs(math.log(curr_area / prev_area))
                motion = float(spatial + 0.18 * area_delta)
            row["motion"] = float(motion)
            records[frame] = row
            motion_rows.append((int(frame), float(motion)))
            prev_entry = row
        smoothed = {}
        for idx, (frame, _motion) in enumerate(motion_rows):
            vals = []
            for j in (idx - 1, idx, idx + 1):
                if 0 <= j < len(motion_rows):
                    vals.append(float(motion_rows[j][1]))
            smoothed[int(frame)] = float(sum(vals) / float(max(1, len(vals))))
        peak_frame = None
        peak_motion = 0.0
        for frame, score in smoothed.items():
            records.setdefault(int(frame), {})["motion_smooth"] = float(score)
            if peak_frame is None or float(score) > float(peak_motion):
                peak_frame = int(frame)
                peak_motion = float(score)
        frames_out = []
        for frame in sorted(records.keys()):
            row = dict(records.get(frame) or {})
            frames_out.append(
                {
                    "frame": int(frame),
                    "bbox": [
                        float(row.get("x1", 0.0)),
                        float(row.get("y1", 0.0)),
                        float(row.get("x2", 0.0)),
                        float(row.get("y2", 0.0)),
                    ],
                    "center": [float(row.get("cx", 0.0)), float(row.get("cy", 0.0))],
                    "area": float(row.get("area", 0.0)),
                    "motion": float(row.get("motion_smooth", row.get("motion", 0.0)) or 0.0),
                    "handedness": str(row.get("handedness") or "").strip().lower(),
                    "handedness_score": float(row.get("handedness_score", 0.0) or 0.0),
                    "detection_confidence": float(row.get("detection_confidence", 0.0) or 0.0),
                    "interpolated": bool(row.get("interpolated", False)),
                }
            )
        return {
            "frame_count": int(frame_count or 0),
            "coverage": float(len(frames_out)) / float(max(1, frame_count)),
            "motion_peak_frame": None if peak_frame is None else int(peak_frame),
            "motion_peak_score": float(peak_motion),
            "frames": frames_out,
        }

    def run(self):
        payload = {
            "success": False,
            "error": "",
            "video_path": self.video_path,
            "tracks": {},
            "frame_count": 0,
            "frame_size": [0, 0],
            "stride": 1,
            "backend": "",
        }
        if not self.video_path:
            payload["error"] = "Missing video path."
            self.finished.emit(payload)
            return
        try:
            import mediapipe as mp
        except Exception as ex:
            payload["error"] = f"MediaPipe is unavailable: {ex}"
            self.finished.emit(payload)
            return
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            payload["error"] = "Failed to open video."
            self.finished.emit(payload)
            return
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            payload["frame_count"] = int(frame_count)
            payload["frame_size"] = [int(width), int(height)]
            stride = self._dynamic_stride(frame_count)
            payload["stride"] = int(stride)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps <= 1e-6:
                fps = 30.0
            left_actor, right_actor = self._actor_ids()
            track_rows: Dict[str, Dict[int, Dict[str, Any]]] = {
                left_actor: {},
                right_actor: {},
            }
            prev_centers: Dict[str, Tuple[float, float]] = {}
            frame_idx = 0
            processed = 0
            solutions_hands = getattr(getattr(mp, "solutions", None), "hands", None)
            if solutions_hands is not None:
                payload["backend"] = "solutions"
                with solutions_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self.max_hands,
                    min_detection_confidence=self.det_conf,
                    min_tracking_confidence=self.track_conf,
                ) as hands:
                    while True:
                        ok, frame_bgr = cap.read()
                        if not ok:
                            break
                        if frame_idx % stride != 0:
                            frame_idx += 1
                            continue
                        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        results = hands.process(rgb)
                        detections = self._extract_detections(results, width, height)
                        assigned = self._assign_detections(detections, prev_centers)
                        for actor_id, det in dict(assigned or {}).items():
                            row = dict(det or {})
                            row["interpolated"] = False
                            track_rows.setdefault(str(actor_id), {})[int(frame_idx)] = row
                            prev_centers[str(actor_id)] = (
                                float(row.get("cx", 0.0)),
                                float(row.get("cy", 0.0)),
                            )
                        processed += 1
                        if processed % 90 == 0:
                            self.progress.emit(
                                f"Precomputing persistent hand tracks... {frame_idx + 1}/{max(1, frame_count)}"
                            )
                        frame_idx += 1
            else:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                model_path = _mediapipe_hand_landmarker_model_file(self.video_path)
                payload["backend"] = "tasks"
                options = vision.HandLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=model_path),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=self.max_hands,
                    min_hand_detection_confidence=self.det_conf,
                    min_tracking_confidence=self.track_conf,
                )
                with vision.HandLandmarker.create_from_options(options) as hands:
                    while True:
                        ok, frame_bgr = cap.read()
                        if not ok:
                            break
                        if frame_idx % stride != 0:
                            frame_idx += 1
                            continue
                        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        timestamp_ms = int(round((float(frame_idx) / float(fps)) * 1000.0))
                        results = hands.detect_for_video(mp_image, timestamp_ms)
                        detections = self._extract_detections_tasks(results, width, height)
                        assigned = self._assign_detections(detections, prev_centers)
                        for actor_id, det in dict(assigned or {}).items():
                            row = dict(det or {})
                            row["interpolated"] = False
                            track_rows.setdefault(str(actor_id), {})[int(frame_idx)] = row
                            prev_centers[str(actor_id)] = (
                                float(row.get("cx", 0.0)),
                                float(row.get("cy", 0.0)),
                            )
                        processed += 1
                        if processed % 90 == 0:
                            self.progress.emit(
                                f"Precomputing persistent hand tracks... {frame_idx + 1}/{max(1, frame_count)}"
                            )
                        frame_idx += 1
            payload["tracks"] = {
                str(actor_id): self._finalize_track(
                    dict(rows or {}),
                    frame_count=frame_count,
                    stride=stride,
                    width=width,
                    height=height,
                )
                for actor_id, rows in dict(track_rows or {}).items()
            }
            payload["success"] = True
        except Exception as ex:
            payload["error"] = str(ex)
        finally:
            try:
                cap.release()
            except Exception:
                pass
        self.finished.emit(payload)


class HOIWindow(LabelGlossaryMixin, AutosaveMixin, AnnotationPolicyMixin, FrameReviewMixin, AssemblyEditorMixin, CorrectionPropagationMixin, FrameControlMixin, QWidget):
    """
    HOI event construction annotator:
    - Single video.
    - Load YOLO bbox text (per-line: frame class xc yc w h [conf]; normalized if <=1).
    - Overlay bboxes on video; show current-frame boxes in a list.
    - Define verbs via LabelPanel; select two boxes + a verb to create/delete relations.
    """
    def __init__(
        self,
        parent=None,
        on_close=None,
        logger: OperationLogger = None,
    ):
        super().__init__(parent)

        self.setFocusPolicy(Qt.StrongFocus)
        self.setWindowTitle("IMPACT HOI")
        self.resize(1280, 840)
        self._on_close = on_close
        self.op_logger = logger or OperationLogger(False)
        self._ui_preferences = load_ui_preferences(default_ui_scale=0.85)
        self._ui_scale = float(self._ui_preferences.get("ui_scale", 0.85) or 0.85)
        self._show_quick_start_on_startup = bool(
            self._ui_preferences.get("show_quick_start_on_startup", False)
        )
        self._user_study_mode = False
        self._participant_code = ""
        self._experiment_mode_change_guard = False
        self._quick_start_auto_open_pending = False
        self._quick_start_dialog_open = False
        self._onboarding_dismissed_session = True
        self._spotlight_widget_ref = None
        self._spotlight_effect = None
        self._spotlight_timer = QTimer(self)
        self._spotlight_timer.setSingleShot(True)
        self._spotlight_timer.timeout.connect(self._clear_spotlight_widget)
        self._last_onboarding_focus_key = None
        self._find_boxes_guidance_signature = ""
        self._auto_find_boxes_signatures: Dict[str, str] = {}
        self._completion_status_refresh_pending = False
        self._selected_edit_box: Optional[dict] = None
        self._pending_draw_box_role: Optional[str] = None
        self._shortcut_bindings = load_shortcut_bindings()
        self._shortcut_defaults = default_shortcut_bindings()

        self.player = VideoPlayer()
        self.player.on_frame_advanced = self._on_frame_advanced
        self.player.on_playback_state_changed = self._on_player_playback_state_changed
        self.player.on_click_frame = self._on_video_canvas_click

        # --- Customizable Actor Module (Default: Left/Right Hand) ---
        self.actors_config = [
            {"id": "Left_hand", "label": "Left Hand", "short": "L"},
            {"id": "Right_hand", "label": "Right Hand", "short": "R"},
        ]

        # data
        self.bboxes: Dict[int, List[dict]] = {}  # frame -> list of boxes
        self.box_id_counter = 1
        self._loaded_target_names: List[str] = []
        self._loaded_target_norms: set = set()
        self._known_anomaly_labels: set = set(_ANOMALY_LABEL_OPTIONS)
        self.verbs: List[LabelDef] = []
        self.current_verb_idx = -1
        self._action_panel_sync = False
        self._pending_field_sources: Dict[str, str] = {}
        self.events: List[dict] = []  # Store all events (internal event structure)
        self.event_id_counter = 0  # Event ID counter
        self._reset_event_draft()  # Initialize event draft structure
        self.class_map: Dict[int, str] = {}
        self.labels_dir: str = ""
        self.start_offset = 0
        self.end_frame = None  # optional clamp
        self.raw_boxes: List[dict] = []  # keep raw frame indices for re-mapping
        self._bbox_revision = 0
        self.current_hands = {actor["id"]: None for actor in self.actors_config}
        self.selected_hand_label = None  # Current active actor ID
        self.selected_event_id = None
        self._required_loaded = False
        self.video_path = ""
        self.current_annotation_path = ""
        # YOLO current-frame detection
        self.yolo_model = None
        self.yolo_weights_path = ""
        self.yolo_conf = 0.25
        self.yolo_iou = 0.45
        self.yolo_existing_policy = None  # "append" or "replace"
        self.mp_hands = None
        self.mp_hands_error = None
        self.mp_hands_backend = ""
        self.mp_hands_max = 2
        self.mp_hands_conf = 0.5
        self.mp_hands_track_conf = 0.5
        self.mp_hands_swap = False
        self._handtrack_cache: Dict[str, Any] = {}
        self._handtrack_worker = None
        self._handtrack_cache_key = ""
        self._handtrack_status = {
            "ready": False,
            "building": False,
            "video_path": "",
            "cache_file": "",
            "backend": "",
            "error": "",
        }

        # VideoMAE V2 action detection
        self.videomae_handler = VideoMAEHandler()
        self.videomae_weights_path = ""
        self.videomae_verb_list_path = ""
        self._videomae_loaded_key = ""
        self._videomae_precomputed_cache: Dict[str, Any] = {}
        self._videomae_precomputed_cache_path = ""
        self._videomae_precomputed_cache_key = ""
        self._videomae_action_cache: Dict[str, List[dict]] = {}
        self._videomae_feature_cache: Dict[str, List[float]] = {}
        self._videomae_local_action_cache: Dict[str, List[dict]] = {}
        self._videomae_local_feature_cache: Dict[str, List[float]] = {}
        self._videomae_event_signatures: Dict[int, str] = {}
        self._videomae_auto_event_id: Optional[int] = None
        self._videomae_auto_force_refresh = False
        self._videomae_auto_timer = QTimer(self)
        self._videomae_auto_timer.setSingleShot(True)
        self._videomae_auto_timer.setInterval(400)
        self._videomae_auto_timer.timeout.connect(
            self._run_pending_action_label_refresh
        )
        self._yolo_infer_worker = None
        self._yolo_infer_context: Dict[str, Any] = {}
        self._videomae_infer_worker = None
        self._videomae_infer_request: Dict[str, Any] = {}
        self._videomae_pending_request: Optional[Dict[str, Any]] = None
        self._videomae_batch_requests: List[Dict[str, Any]] = []
        self._videomae_batch_progress = None
        self._full_assist_warning_signature = ""
        self._hoi_undo_stack = []
        self._hoi_redo_stack = []
        self._undo_block = False
        self._undo_limit = 50
        self.validation_enabled = False
        self.validation_session_active = False
        self.validation_started_at = None
        self.validation_modified = False
        self.validation_change_count = 0
        self.validator_name = ""
        self.validation_summary_enabled = True
        self.validation_op_logger = OperationLogger(self.op_logger.enabled)
        self._oplog_flush_timer = QTimer(self)
        self._oplog_flush_timer.setInterval(15000)
        self._oplog_flush_timer.timeout.connect(self._flush_live_operation_logs)
        self._oplog_flush_timer.start()
        self._validation_highlights = {}
        self._relation_highlights = {}
        self._hoi_saved_signature = None
        self._close_request_approved = False
        self._close_request_finalized = False
        self._experiment_mode = "full_assist"
        self._next_best_query: Optional[Dict[str, Any]] = None
        self._current_query_candidates: List[Dict[str, Any]] = []
        self._next_best_query_id = ""
        self._next_best_query_presented_at = 0.0
        self._query_state_revision = 0
        self._query_calibrator = None
        self._query_calibrator_signature = None
        self._query_calibration_dirty = True
        self._authority_policy_config = {"preset": "default"}
        self._slider_scrubbing = False
        self._dismissed_query_ids: Dict[str, int] = {}
        self._query_metrics = {
            "presented": 0,
            "focused": 0,
            "accepted": 0,
            "propagated": 0,
            "rejected": 0,
            "hand_conditioned_presented": 0,
            "hand_conditioned_focused": 0,
            "hand_conditioned_accepted": 0,
            "hand_conditioned_propagated": 0,
            "hand_conditioned_rejected": 0,
        }
        self._safe_execution_metrics = {
            "precheck_blocked": 0,
            "violations": 0,
            "rollbacks": 0,
        }
        self._clip_session_index = 0
        self._clip_session_id = ""
        self._annotation_ready_signature = ""
        self._annotation_readiness_phase = ""

        # controls (professional top chrome)
        self.toolbar_frame = QFrame(self)
        self.toolbar_frame.setObjectName("toolbarFrame")
        toolbar_layout = QVBoxLayout(self.toolbar_frame)
        toolbar_layout.setContentsMargins(6, 4, 6, 4)
        toolbar_layout.setSpacing(2)

        session_row = QHBoxLayout()
        session_row.setContentsMargins(0, 0, 0, 0)
        session_row.setSpacing(6)
        transport_row = QHBoxLayout()
        transport_row.setContentsMargins(0, 0, 0, 0)
        transport_row.setSpacing(6)
        toolbar_main_row = QHBoxLayout()
        toolbar_main_row.setContentsMargins(0, 0, 0, 0)
        toolbar_main_row.setSpacing(12)

        lbl_experiment_mode = QLabel("Mode")
        lbl_experiment_mode.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        session_row.addWidget(lbl_experiment_mode)
        self.combo_experiment_mode = QComboBox()
        self.combo_experiment_mode.addItem("Manual", "manual")
        self.combo_experiment_mode.addItem("Full Assist", "full_assist")
        self.combo_experiment_mode.setCurrentIndex(1)
        self.combo_experiment_mode.setMinimumWidth(120)
        self.combo_experiment_mode.setToolTip(
            "Manual: annotate directly. Full Assist: show action suggestions and box assistance."
        )
        self.combo_experiment_mode.currentIndexChanged.connect(
            self._on_experiment_mode_changed
        )
        session_row.addWidget(self.combo_experiment_mode)
        self.lbl_participant_code = QLabel("Participant No")
        self.lbl_participant_code.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.lbl_participant_code.setVisible(False)
        session_row.addWidget(self.lbl_participant_code)
        self.edit_participant_code = QLineEdit(self._participant_code)
        self.edit_participant_code.setPlaceholderText("")
        self.edit_participant_code.setClearButtonEnabled(True)
        self.edit_participant_code.setMinimumWidth(96)
        self.edit_participant_code.setMaximumWidth(132)
        self.edit_participant_code.setToolTip(
            "Participant/session code is disabled for annotation mode."
        )
        self.edit_participant_code.setVisible(False)
        self.edit_participant_code.editingFinished.connect(
            self._on_participant_code_changed
        )
        session_row.addWidget(self.edit_participant_code)
        self.lbl_clip_readiness_chip = QLabel("Load clip")
        self.lbl_clip_readiness_chip.setToolTip(
            "Load a video and the required study assets before timing starts."
        )
        self.lbl_clip_readiness_chip.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        session_row.addWidget(self.lbl_clip_readiness_chip)
        session_row.addStretch(1)

        self.lbl_validation = QLabel("Validate")
        self.lbl_validation.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        session_row.addWidget(self.lbl_validation)
        self.btn_validation = ToggleSwitch(self)
        self.btn_validation.setToolTip("Toggle validation on/off")
        self.btn_validation.toggled.connect(self._on_validation_toggled)
        session_row.addWidget(self.btn_validation)

        self.file_menu = QMenu(self)
        self.act_load_video = self.file_menu.addAction("Load Video...", self._load_video)
        self.act_new_trial = self.file_menu.addAction(
            "New Trial / Reset Workspace", self._new_trial_reset_workspace
        )
        self.file_menu.addSeparator()
        import_menu = self.file_menu.addMenu("Import")
        self.import_menu = import_menu
        self.act_import_targets = import_menu.addAction("Noun List...", self._import_targets)
        self.act_prune_object_library = import_menu.addAction(
            "Prune Object Library To Current Nouns",
            lambda: self._prune_object_library_to_loaded_targets(notify_user=True),
        )
        self.act_load_verbs = import_menu.addAction("Verb List...", self._load_verbs_txt)
        self.act_import_hoi_ontology = import_menu.addAction("Verb-Noun Ontology CSV...", self._import_hoi_ontology)
        import_menu.addSeparator()
        self.act_load_yaml = import_menu.addAction("Class Map (data.yaml)...", self._load_yaml)
        self.act_import_yolo_boxes = import_menu.addAction(
            "YOLO Boxes...", self._load_bboxes
        )
        self.act_load_hands_xml = import_menu.addAction(
            "Hands XML...", self._load_hands_xml
        )
        import_menu.addSeparator()
        self.act_load_annotations = import_menu.addAction("HOI Annotations...", self._load_annotations_json)
        self.act_shared_assembly = self.file_menu.addAction("Shared assembly...", self._open_assembly_editor)
        self.file_menu.addAction("Merge assemblies...", self._open_assembly_merge)
        self.file_menu.addAction("Use independent Object for selected hand", self._unlink_shared_assembly)
        self._install_frame_review()
        self._install_annotation_policy()
        self.act_frame_review_status=self.file_menu.addAction("Frame review: no active event")
        self.act_frame_review_status.setEnabled(False)

        detect_menu = self.file_menu.addMenu("Detection")
        self.detect_menu = detect_menu
        self.act_load_yolo_model = detect_menu.addAction(
            "Load YOLO Model...", self._load_yolo_model
        )
        self.act_detect_current_frame = detect_menu.addAction(
            "Detect Current Frame", self._detect_current_frame_combined
        )
        self.act_detect_selected_action = detect_menu.addAction(
            "Detect Selected Action", self._detect_selected_action
        )
        self.act_detect_all_actions = detect_menu.addAction(
            "Detect All Actions", self._detect_all_actions
        )
        detect_menu.addSeparator()
        self.act_precompute_hand_tracks = detect_menu.addAction(
            "Precompute Persistent Hand Tracks",
            self._precompute_persistent_hand_tracks,
        )
        self.act_swap_hands = detect_menu.addAction("Auto Swap Left / Right")
        self.act_swap_hands.setCheckable(True)
        self.act_swap_hands.toggled.connect(
            lambda on: setattr(self, "mp_hands_swap", on)
        )
        detect_menu.addSeparator()
        self.act_incremental_train_yolo = detect_menu.addAction(
            "YOLO Incremental Training...", self._incremental_train_yolo
        )

        assist_menu = self.file_menu.addMenu("Action Assist")
        self.assist_menu = assist_menu
        self.act_load_videomae_model = assist_menu.addAction(
            "Load Frozen Video Encoder...", self._load_videomae_model
        )
        self.act_load_videomae_cache = assist_menu.addAction(
            "Load Precomputed Encoder Cache...", self._load_videomae_cache
        )
        self.act_load_videomae_verb_list = assist_menu.addAction(
            "Load Optional Verb-Prior Labels...", self._load_videomae_verb_list
        )
        self.act_load_semantic_adapter = assist_menu.addAction(
            "Load Semantic Adapter Base...", self._load_semantic_adapter
        )
        assist_menu.addSeparator()
        self.act_review_selected_action_label = assist_menu.addAction(
            "Review Selected Verb Ranking...", self._detect_selected_action_label
        )
        self.act_auto_apply_action_labels = assist_menu.addAction(
            "Refresh Verb Candidates for All Events", self._detect_all_action_labels
        )

        save_menu = self.file_menu.addMenu("Save / Export")
        self.save_menu = save_menu
        self.act_save_annotations = save_menu.addAction("Save HOI Annotations...", self._save_annotations_json)
        self.act_save_hands_xml = save_menu.addAction("Export Hands XML...", self._save_hands_xml)
        self.file_menu.addSeparator()
        self.act_open_quick_start = self.file_menu.addAction("Quick Start / Help...", self._open_quick_start_dialog)
        self.act_label_guide = self.file_menu.addAction("Noun and Verb Guide...", self._open_label_guide)
        self.act_show_onboarding_banner = self.file_menu.addAction(
            "Show Onboarding Banner", self._show_onboarding_banner
        )
        self.act_open_settings = self.file_menu.addAction("Settings...", self._open_settings_dialog)

        self.btn_file_menu = QToolButton()
        self.btn_file_menu.setText("\u22EF")
        self.btn_file_menu.setToolTip(
            "Project, import/export, detection, and model actions"
        )
        self.btn_file_menu.setPopupMode(QToolButton.InstantPopup)
        self.btn_file_menu.setMenu(self.file_menu)
        self.btn_file_menu.setFixedWidth(30)
        session_row.addWidget(self.btn_file_menu)

        self.btn_quick_start = QToolButton()
        self.btn_quick_start.setText("?")
        self.btn_quick_start.setToolTip("Open Quick Start guide at any time (F1)")
        self.btn_quick_start.clicked.connect(self._open_quick_start_dialog)
        self.btn_quick_start.setFixedWidth(30)
        session_row.addWidget(self.btn_quick_start)

        self.btn_rew = QToolButton()
        self.btn_rew.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekBackward))
        self.btn_play = QToolButton()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.setToolTip("Play")
        self.btn_stop = QToolButton()
        self.btn_stop.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.btn_ff = QToolButton()
        self.btn_ff.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekForward))
        for b in (
            self.btn_rew,
            self.btn_play,
            self.btn_stop,
            self.btn_ff,
        ):
            b.setToolButtonStyle(Qt.ToolButtonIconOnly)
            b.setIconSize(QSize(20, 20))
        self.spin_jump = QSpinBox()
        self.spin_jump.setMinimum(0)
        self.spin_jump.setMaximum(0)
        self.spin_jump.setKeyboardTracking(False)
        self.btn_jump = QToolButton()
        self.btn_jump.setText("\u2192")
        self.btn_jump.setToolTip("Jump to frame")

        self.btn_rew.clicked.connect(lambda: self._seek_relative(-10))
        self.btn_ff.clicked.connect(lambda: self._seek_relative(+10))
        self.btn_play.clicked.connect(self._toggle_play_pause)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_jump.clicked.connect(self._jump_to_spin)

        self.combo_verb = QComboBox()
        self.combo_verb.setMinimumWidth(96)
        self.combo_verb.setEditable(True)

        self.btn_detect = QToolButton(self)
        self.btn_detect.setText("Frame")
        self.btn_detect.setToolTip(
            "Detect objects + hands on current frame (Ctrl+Shift+D)"
        )
        self.btn_detect.clicked.connect(self._detect_current_frame_combined)
        self.btn_detect_action = QToolButton(self)
        self.btn_detect_action.setText("Action")
        self.btn_detect_action.setToolTip(
            "Detect assigned objects on the selected action's start, onset, and end frames."
        )
        self.btn_detect_action.clicked.connect(self._detect_selected_action)

        transport_row.addWidget(self.btn_rew)
        transport_row.addWidget(self.btn_play)
        transport_row.addWidget(self.btn_stop)
        transport_row.addWidget(self.btn_ff)
        transport_row.addSpacing(8)
        transport_row.addWidget(QLabel("Jump"))
        transport_row.addWidget(self.spin_jump)
        transport_row.addWidget(self.btn_jump)
        transport_row.addSpacing(8)
        transport_row.addWidget(QLabel("Start"))
        self.spin_start_offset = QSpinBox()
        self.spin_start_offset.setMinimum(0)
        self.spin_start_offset.valueChanged.connect(self._on_offset_changed)
        transport_row.addWidget(self.spin_start_offset)
        transport_row.addWidget(QLabel("End"))
        self.spin_end_frame = QSpinBox()
        self.spin_end_frame.setMinimum(0)
        self.spin_end_frame.setMaximum(10**9)
        self.spin_end_frame.valueChanged.connect(self._on_offset_changed)
        transport_row.addWidget(self.spin_end_frame)
        transport_row.addStretch(1)

        self.chk_edit_boxes = QCheckBox("\u270E Edit Existing Boxes")
        self.chk_edit_boxes.setToolTip(
            "Toggle moving/resizing existing boxes. Ctrl+drag on the video canvas always creates a new box."
        )
        self.chk_edit_boxes.setChecked(True)
        self.chk_edit_boxes.toggled.connect(self._on_edit_boxes_toggled)
        self.chk_edit_boxes.hide()

        self.draw_mode_widget = QWidget(self)
        draw_mode_row = QHBoxLayout(self.draw_mode_widget)
        draw_mode_row.setContentsMargins(0, 0, 0, 0)
        draw_mode_row.setSpacing(6)
        lbl_draw_mode = QLabel("Draw")
        lbl_draw_mode.setObjectName("captionLabel")
        draw_mode_row.addWidget(lbl_draw_mode)
        self.rad_draw_none = QRadioButton("Manual")
        self.rad_draw_inst = QRadioButton("Instrument")
        self.rad_draw_target = QRadioButton("Noun")
        self.rad_draw_none.setChecked(True)
        self.rad_draw_none.setToolTip("Draw boxes and enter labels manually.")
        self.rad_draw_inst.setToolTip(
            "New boxes inherit the selected event's instrument/tool label."
        )
        self.rad_draw_target.setToolTip(
            "New boxes inherit the selected event's noun/object label."
        )
        draw_mode_row.addWidget(self.rad_draw_none)
        draw_mode_row.addWidget(self.rad_draw_inst)
        draw_mode_row.addWidget(self.rad_draw_target)
        self.rad_draw_none.toggled.connect(self._update_inline_edit_boxes_button_state)
        self.rad_draw_inst.toggled.connect(self._update_inline_edit_boxes_button_state)
        self.rad_draw_target.toggled.connect(self._update_inline_edit_boxes_button_state)
        for widget in (self.rad_draw_none, self.rad_draw_inst, self.rad_draw_target):
            widget.setEnabled(False)
        self.draw_mode_widget.hide()

        toolbar_main_row.addLayout(session_row, 0)
        toolbar_main_row.addLayout(transport_row, 1)
        toolbar_layout.addLayout(toolbar_main_row)

        self.onboarding_banner = QFrame(self)
        self.onboarding_banner.setStyleSheet(
            "background: #F8FAFC; border: 1px solid #D7DEE8; border-radius: 16px;"
        )
        onboarding_layout = QVBoxLayout(self.onboarding_banner)
        onboarding_layout.setContentsMargins(14, 12, 14, 12)
        onboarding_layout.setSpacing(8)

        onboarding_header = QHBoxLayout()
        onboarding_header.setContentsMargins(0, 0, 0, 0)
        onboarding_header.setSpacing(8)
        self.lbl_onboarding_title = QLabel("Getting Started")
        self.lbl_onboarding_title.setStyleSheet(
            "background: #2563EB; color: white; border-radius: 999px; padding: 4px 10px; font-weight: 700;"
        )
        self.lbl_onboarding_title.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        onboarding_header.addWidget(self.lbl_onboarding_title)
        self.lbl_onboarding_progress = QLabel("0 / 6 ready")
        self.lbl_onboarding_progress.setObjectName("statusSubtle")
        onboarding_header.addWidget(self.lbl_onboarding_progress)
        onboarding_header.addStretch(1)
        self.btn_onboarding_open_guide = QPushButton("Open Guide")
        self.btn_onboarding_open_guide.setAutoDefault(False)
        self.btn_onboarding_open_guide.clicked.connect(self._open_quick_start_dialog)
        self.btn_onboarding_hide = QToolButton()
        self.btn_onboarding_hide.setText("Hide")
        self.btn_onboarding_hide.setAutoRaise(True)
        self.btn_onboarding_hide.clicked.connect(self._dismiss_onboarding_banner)
        onboarding_header.addWidget(self.btn_onboarding_open_guide)
        onboarding_header.addWidget(self.btn_onboarding_hide)
        onboarding_layout.addLayout(onboarding_header)

        self.lbl_onboarding_summary = QLabel(
            "Start by loading a video and importing the required project lists."
        )
        self.lbl_onboarding_summary.setWordWrap(True)
        self.lbl_onboarding_summary.setStyleSheet("color: #0F172A; font-weight: 600;")
        onboarding_layout.addWidget(self.lbl_onboarding_summary)

        self.onboarding_steps_row = QHBoxLayout()
        self.onboarding_steps_row.setContentsMargins(0, 0, 0, 0)
        self.onboarding_steps_row.setSpacing(6)
        self._onboarding_step_chips = {}
        onboarding_chip_defs = (
            ("video", "Video"),
            ("target", "Nouns"),
            ("verb", "Verbs"),
            ("event", "Event"),
        )
        for key, label in onboarding_chip_defs:
            chip = QLabel(label)
            chip.setAlignment(Qt.AlignCenter)
            chip.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.onboarding_steps_row.addWidget(chip)
            self._onboarding_step_chips[key] = chip
        self.onboarding_steps_row.addStretch(1)
        onboarding_layout.addLayout(self.onboarding_steps_row)

        self.onboarding_actions_row = QHBoxLayout()
        self.onboarding_actions_row.setContentsMargins(0, 0, 0, 0)
        self.onboarding_actions_row.setSpacing(8)
        onboarding_layout.addLayout(self.onboarding_actions_row)

        # video row (main canvas above, inspector below)
        video_row = QHBoxLayout()
        video_row.setContentsMargins(0, 0, 0, 0)
        video_row.setSpacing(8)
        video_row.addWidget(self.player, 1)

        self.list_objects = QListWidget()
        self.list_objects.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_objects.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_objects.customContextMenuRequested.connect(self._on_object_list_menu)
        self.list_objects.itemSelectionChanged.connect(self._on_object_selection)
        self.list_objects.setEnabled(False)
        self.list_objects.setAlternatingRowColors(True)
        self.list_objects.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)

        hand_row = QHBoxLayout()
        hand_row.setContentsMargins(0, 0, 0, 0)
        hand_row.setSpacing(8)
        lbl_actors = QLabel("Actors")
        lbl_actors.setObjectName("captionLabel")
        hand_row.addWidget(lbl_actors)

        self.actor_layout = QHBoxLayout()
        hand_row.addLayout(self.actor_layout)

        self.actor_controls = {}
        self._rebuild_actor_checkboxes()

        self.btn_config_actors = QPushButton("Manage")
        self.btn_config_actors.setToolTip("Configure actors (Add/Remove/Rename)")
        self.btn_config_actors.clicked.connect(self._on_configure_actors)
        hand_row.addWidget(self.btn_config_actors)

        self.btn_swap_draft = QPushButton("\u21C4")
        self.btn_swap_draft.setToolTip(
            "Swap actor boxes on the current frame (first two actors)."
        )
        self.btn_swap_draft.setStyleSheet("color: #111;")
        self.btn_swap_draft.clicked.connect(self._swap_frame_hands)
        hand_row.addWidget(self.btn_swap_draft)
        hand_row.addStretch(1)

        self.label_panel = LabelPanel(
            self.verbs,
            on_add=self._on_verb_added,
            on_remove_idx=self._on_verb_removed,
            on_rename=self._on_verb_renamed,
            on_search_matches=None,
            on_select_idx=self._on_verb_selected,
            manage_storage=False,
        )

        for le in self.label_panel.findChildren(QLineEdit):
            ph = le.placeholderText().lower()
            if "search" in ph:
                le.setPlaceholderText("Search action library")
            elif "new" in ph:
                le.setPlaceholderText("Add action to library")
        self._init_verb_color_combo()
        self.label_panel.set_verb_only(True)
        self.label_panel.set_admin_visible(False)
        self.label_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)

        self.group_library = QGroupBox("Objects")
        self.group_library.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lib_layout = QVBoxLayout()
        lib_layout.setContentsMargins(8, 8, 8, 8)
        lib_layout.setSpacing(6)

        object_tools_row = QHBoxLayout()
        object_tools_row.setContentsMargins(0, 0, 0, 0)
        object_tools_row.setSpacing(6)
        object_tools_row.addWidget(self.btn_detect)
        object_tools_row.addWidget(self.btn_detect_action)
        self.btn_object_tools = QToolButton()
        self.btn_object_tools.setText("...")
        self.btn_object_tools.setPopupMode(QToolButton.InstantPopup)
        self.btn_object_tools.setToolTip("More object tools")
        self.object_tools_menu = QMenu(self.btn_object_tools)
        self.object_tools_menu.addAction("Detect All", self._detect_all_actions)
        self.object_tools_menu.addSeparator()
        self.object_tools_menu.addAction(self.act_load_yolo_model)
        self.object_tools_menu.addAction(self.act_incremental_train_yolo)
        self.object_tools_menu.addSeparator()
        self.object_tools_menu.addAction(self.act_swap_hands)
        self.btn_object_tools.setMenu(self.object_tools_menu)
        object_tools_row.addWidget(self.btn_object_tools)
        object_tools_row.addStretch(1)
        lib_layout.addLayout(object_tools_row)
        edit_tools_row = QHBoxLayout()
        edit_tools_row.setContentsMargins(0, 0, 0, 0)
        edit_tools_row.setSpacing(6)
        edit_tools_row.addWidget(self.chk_edit_boxes)
        edit_tools_row.addStretch(1)
        lib_layout.addLayout(edit_tools_row)
        lib_layout.addWidget(self.draw_mode_widget)

        link_form = QFormLayout()
        self.combo_target = QComboBox()
        self.combo_target.addItem("None", None)

        self._enable_combo_search(self.combo_target, placeholder="Search noun / object...")

        noun_label = QLabel("Noun")
        noun_label.setObjectName("nounFieldLabel")
        link_form.addRow(noun_label, self.combo_target)
        self._noun_form_label = noun_label
        self.combo_instrument = QComboBox()
        self.combo_instrument.addItem("", None)
        self._enable_combo_search(
            self.combo_instrument,
            placeholder="Search instrument / tool...",
            auto_popup=True,
        )
        self.combo_instrument.setEditText("")
        instrument_label = QLabel("Instrument")
        instrument_label.setObjectName("instrumentFieldLabel")
        link_form.addRow(instrument_label, self.combo_instrument)
        self._instrument_form_label = instrument_label
        self.combo_anomaly = AttributeSelector()
        self._populate_anomaly_combo(self.combo_anomaly)
        anomaly_label = QLabel("Anomaly")
        anomaly_label.setObjectName("anomalyFieldLabel")
        link_form.addRow(anomaly_label, self.combo_anomaly)
        self._anomaly_form_label = anomaly_label
        lib_layout.addLayout(link_form)

        lbl_current_objects = QLabel("Current Frame")
        lbl_current_objects.setObjectName("captionLabel")
        lib_layout.addWidget(lbl_current_objects)
        lib_layout.addWidget(self.list_objects, 1)
        self.group_library.setLayout(lib_layout)

        hand_widget = QWidget()
        hand_layout = QHBoxLayout(hand_widget)
        hand_layout.setContentsMargins(0, 0, 0, 0)
        hand_layout.addLayout(hand_row)

        self.group_action = QGroupBox("Action")
        self.group_action.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        action_layout = QVBoxLayout(self.group_action)
        action_layout.setContentsMargins(8, 8, 8, 8)
        action_layout.setSpacing(6)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.addWidget(QLabel("Final"))
        self.combo_verb.setPlaceholderText("Select or type action...")
        action_row.addWidget(self.combo_verb, 1)
        self.btn_suggest_action_label = QToolButton()
        self.btn_suggest_action_label.setText("\u21BB")
        self.btn_suggest_action_label.setToolTip(
            "Refresh the action suggestion for the selected event."
        )
        self.btn_suggest_action_label.clicked.connect(self._refresh_selected_action_label)
        self.btn_action_tools = QToolButton()
        self.btn_action_tools.setText("\u22EF")
        self.btn_action_tools.setPopupMode(QToolButton.InstantPopup)
        self.btn_action_tools.setToolTip("Action tools")
        self.action_tools_menu = QMenu(self.btn_action_tools)
        self.act_toggle_verb_library_admin = QAction("Manage Verb Library", self)
        self.act_toggle_verb_library_admin.setCheckable(True)
        self.act_toggle_verb_library_admin.toggled.connect(
            self._set_verb_library_admin_mode
        )
        self.action_tools_menu.addAction(self.act_toggle_verb_library_admin)
        self.action_tools_menu.addSeparator()
        self.act_batch_apply_action_labels = QAction(
            "Refresh Verb Candidates for All Events",
            self,
        )
        self.act_batch_apply_action_labels.triggered.connect(self._detect_all_action_labels)
        self.action_tools_menu.addAction(self.act_batch_apply_action_labels)
        self.btn_action_tools.setMenu(self.action_tools_menu)
        action_row.addWidget(self.btn_suggest_action_label)
        action_row.addWidget(self.btn_action_tools)
        action_layout.addLayout(action_row)
        action_layout.addWidget(self.label_panel, 1)

        self.label_panel.verb_list.itemClicked.connect(self._on_verb_library_item_clicked)
        self.label_panel.verb_list.setToolTip(
            "Click a verb to apply it to the current action."
        )

        self.event_status_card = QFrame(self)
        self.event_status_card.setObjectName("statusCard")
        event_status_layout = QVBoxLayout(self.event_status_card)
        event_status_layout.setContentsMargins(12, 12, 12, 12)
        event_status_layout.setSpacing(8)
        event_header = QHBoxLayout()
        event_header.setContentsMargins(0, 0, 0, 0)
        self.lbl_event_title = QLabel("No event selected")
        self.lbl_event_title.setObjectName("statusTitle")
        event_header.addWidget(self.lbl_event_title, 1)
        self.lbl_event_actor_chip = QLabel("Idle")
        self.lbl_event_actor_chip.setAlignment(Qt.AlignCenter)
        event_header.addWidget(self.lbl_event_actor_chip)
        self.lbl_event_health_chip = QLabel("Waiting")
        self.lbl_event_health_chip.setAlignment(Qt.AlignCenter)
        event_header.addWidget(self.lbl_event_health_chip)
        event_status_layout.addLayout(event_header)
        self.lbl_event_frames = QLabel("Frames: ?")
        self.lbl_event_frames.setObjectName("statusSubtle")
        event_status_layout.addWidget(self.lbl_event_frames)
        self.lbl_event_meta = QLabel("Verb: ?   Noun: ?")
        self.lbl_event_meta.setWordWrap(True)
        event_status_layout.addWidget(self.lbl_event_meta)
        self.lbl_event_status = QLabel("No HOI event selected.")
        self.lbl_event_status.setWordWrap(True)
        self.lbl_event_status.setObjectName("statusSubtle")
        event_status_layout.addWidget(self.lbl_event_status)
        self.lbl_hand_support_status = QLabel("Hand support: -")
        self.lbl_hand_support_status.setWordWrap(True)
        self.lbl_hand_support_status.setObjectName("statusSubtle")
        event_status_layout.addWidget(self.lbl_hand_support_status)
        keyframe_row = QHBoxLayout()
        keyframe_row.setContentsMargins(0, 0, 0, 0)
        keyframe_row.setSpacing(6)
        keyframe_row.addStretch(1)
        self.btn_jump_start_chip = QPushButton("Start")
        self.btn_jump_start_chip.setProperty("compactChip", True)
        self.btn_jump_start_chip.clicked.connect(self._jump_to_selected_start)
        self.btn_jump_onset_chip = QPushButton("Onset")
        self.btn_jump_onset_chip.setProperty("compactChip", True)
        self.btn_jump_onset_chip.clicked.connect(self._jump_to_selected_onset)
        self.btn_jump_end_chip = QPushButton("End")
        self.btn_jump_end_chip.setProperty("compactChip", True)
        self.btn_jump_end_chip.clicked.connect(self._jump_to_selected_end)
        keyframe_row.addWidget(self.btn_jump_start_chip)
        keyframe_row.addWidget(self.btn_jump_onset_chip)
        keyframe_row.addWidget(self.btn_jump_end_chip)
        keyframe_row.addStretch(1)
        event_status_layout.addLayout(keyframe_row)

        self.event_tab = QWidget()
        event_layout = QVBoxLayout(self.event_tab)
        event_layout.setContentsMargins(0, 0, 0, 0)
        event_layout.setSpacing(8)
        event_layout.addWidget(self.event_status_card)
        event_layout.addWidget(hand_widget)
        event_layout.addWidget(self.group_action)
        event_layout.addStretch(1)

        self.objects_tab = QWidget()
        objects_layout = QVBoxLayout(self.objects_tab)
        objects_layout.setContentsMargins(0, 0, 0, 0)
        objects_layout.setSpacing(8)
        objects_layout.addWidget(self.group_library)
        objects_layout.addStretch(1)

        self.review_status_card = QFrame(self)
        self.review_status_card.setObjectName("statusCard")
        review_status_layout = QVBoxLayout(self.review_status_card)
        review_status_layout.setContentsMargins(12, 12, 12, 12)
        review_status_layout.setSpacing(8)
        review_header = QHBoxLayout()
        review_header.setContentsMargins(0, 0, 0, 0)
        self.lbl_review_title = QLabel("Review Summary")
        self.lbl_review_title.setObjectName("statusTitle")
        review_header.addWidget(self.lbl_review_title, 1)
        self.lbl_validation_chip = QLabel("Validation Off")
        self.lbl_validation_chip.setAlignment(Qt.AlignCenter)
        review_header.addWidget(self.lbl_validation_chip)
        self.lbl_incomplete_chip = QLabel("Incomplete n/a")
        self.lbl_incomplete_chip.setAlignment(Qt.AlignCenter)
        review_header.addWidget(self.lbl_incomplete_chip)
        review_status_layout.addLayout(review_header)
        self.lbl_review_status = QLabel("No review issues yet.")
        self.lbl_review_status.setObjectName("statusSubtle")
        self.lbl_review_status.setWordWrap(True)
        review_status_layout.addWidget(self.lbl_review_status)
        self.lbl_frame_review_status=QLabel("Frame review: no active event")
        review_status_layout.addWidget(self.lbl_frame_review_status)
        review_nav_row = QHBoxLayout()
        review_nav_row.setContentsMargins(0, 0, 0, 0)
        review_nav_row.setSpacing(6)
        self.btn_review_prev = QPushButton("Prev")
        self.btn_review_prev.clicked.connect(lambda: self._jump_incomplete(-1))
        self.btn_review_next = QPushButton("Next")
        self.btn_review_next.clicked.connect(lambda: self._jump_incomplete(+1))
        review_nav_row.addWidget(self.btn_review_prev)
        review_nav_row.addWidget(self.btn_review_next)
        self.chk_auto_jump_incomplete = QCheckBox("Auto-jump after loading")
        self.chk_auto_jump_incomplete.setChecked(True)
        self.chk_auto_jump_incomplete.setToolTip("Select the first field or box issue after loading annotations.")
        review_nav_row.addWidget(self.chk_auto_jump_incomplete)
        review_nav_row.addStretch(1)
        review_status_layout.addLayout(review_nav_row)

        self.next_query_card = QFrame(self)
        self.next_query_card.setObjectName("statusCard")
        next_query_layout = QVBoxLayout(self.next_query_card)
        next_query_layout.setContentsMargins(12, 12, 12, 12)
        next_query_layout.setSpacing(8)
        next_query_header = QHBoxLayout()
        next_query_header.setContentsMargins(0, 0, 0, 0)
        self.lbl_next_query_title = QLabel("Next Best Query")
        self.lbl_next_query_title.setObjectName("statusTitle")
        next_query_header.addWidget(self.lbl_next_query_title, 1)
        self.lbl_next_query_surface_chip = QLabel("Idle")
        self.lbl_next_query_surface_chip.setAlignment(Qt.AlignCenter)
        next_query_header.addWidget(self.lbl_next_query_surface_chip)
        self.lbl_next_query_action_chip = QLabel("Action --")
        self.lbl_next_query_action_chip.setAlignment(Qt.AlignCenter)
        next_query_header.addWidget(self.lbl_next_query_action_chip)
        self.lbl_next_query_form_chip = QLabel("How --")
        self.lbl_next_query_form_chip.setAlignment(Qt.AlignCenter)
        next_query_header.addWidget(self.lbl_next_query_form_chip)
        self.lbl_next_query_authority_chip = QLabel("Authority --")
        self.lbl_next_query_authority_chip.setAlignment(Qt.AlignCenter)
        next_query_header.addWidget(self.lbl_next_query_authority_chip)
        self.lbl_next_query_score_chip = QLabel("VOI --")
        self.lbl_next_query_score_chip.setAlignment(Qt.AlignCenter)
        next_query_header.addWidget(self.lbl_next_query_score_chip)
        next_query_layout.addLayout(next_query_header)
        self.lbl_next_query_summary = QLabel("No pending query yet.")
        self.lbl_next_query_summary.setWordWrap(True)
        next_query_layout.addWidget(self.lbl_next_query_summary)
        self.lbl_next_query_reason = QLabel("The controller will surface the most valuable next supervision step here.")
        self.lbl_next_query_reason.setObjectName("statusSubtle")
        self.lbl_next_query_reason.setWordWrap(True)
        next_query_layout.addWidget(self.lbl_next_query_reason)
        next_query_metrics_row = QHBoxLayout()
        next_query_metrics_row.setContentsMargins(0, 0, 0, 0)
        next_query_metrics_row.setSpacing(6)
        self.lbl_next_query_prop_chip = QLabel("Prop --")
        self.lbl_next_query_prop_chip.setAlignment(Qt.AlignCenter)
        next_query_metrics_row.addWidget(self.lbl_next_query_prop_chip)
        self.lbl_next_query_cost_chip = QLabel("Cost --")
        self.lbl_next_query_cost_chip.setAlignment(Qt.AlignCenter)
        next_query_metrics_row.addWidget(self.lbl_next_query_cost_chip)
        self.lbl_next_query_risk_chip = QLabel("Risk --")
        self.lbl_next_query_risk_chip.setAlignment(Qt.AlignCenter)
        next_query_metrics_row.addWidget(self.lbl_next_query_risk_chip)
        next_query_metrics_row.addStretch(1)
        next_query_layout.addLayout(next_query_metrics_row)
        self.lbl_next_query_evidence = QLabel("Sparse evidence: --")
        self.lbl_next_query_evidence.setObjectName("statusSubtle")
        self.lbl_next_query_evidence.setWordWrap(True)
        next_query_layout.addWidget(self.lbl_next_query_evidence)
        self.lbl_next_query_metrics = QLabel(
            "Session: 0 shown · 0 focused · 0 accepted · 0 propagated · 0 rejected"
        )
        self.lbl_next_query_metrics.setObjectName("statusSubtle")
        self.lbl_next_query_metrics.setWordWrap(True)
        next_query_layout.addWidget(self.lbl_next_query_metrics)
        next_query_action_row = QHBoxLayout()
        next_query_action_row.setContentsMargins(0, 0, 0, 0)
        next_query_action_row.setSpacing(6)
        self.btn_next_query_focus = QPushButton("Focus")
        self.btn_next_query_focus.clicked.connect(self._focus_next_best_query)
        self.btn_next_query_apply = QPushButton("Apply")
        self.btn_next_query_apply.clicked.connect(self._apply_next_best_query)
        self.btn_next_query_reject = QPushButton("Dismiss")
        self.btn_next_query_reject.clicked.connect(self._reject_next_best_query)
        next_query_action_row.addWidget(self.btn_next_query_focus)
        next_query_action_row.addWidget(self.btn_next_query_apply)
        next_query_action_row.addWidget(self.btn_next_query_reject)
        next_query_action_row.addStretch(1)
        next_query_layout.addLayout(next_query_action_row)
        for widget in (
            self.lbl_next_query_surface_chip,
            self.lbl_next_query_form_chip,
            self.lbl_next_query_authority_chip,
            self.lbl_next_query_score_chip,
            self.lbl_next_query_prop_chip,
            self.lbl_next_query_cost_chip,
            self.lbl_next_query_risk_chip,
            self.lbl_next_query_metrics,
        ):
            widget.setVisible(False)

        self.review_tab = QWidget()
        review_layout = QVBoxLayout(self.review_tab)
        review_layout.setContentsMargins(0, 0, 0, 0)
        review_layout.setSpacing(8)
        review_layout.addWidget(self.review_status_card)
        review_layout.addWidget(self.next_query_card)
        review_layout.addStretch(1)

        self.event_tab_scroll = self._make_inspector_scroll(self.event_tab)
        self.objects_tab_scroll = self._make_inspector_scroll(self.objects_tab)
        self.review_tab_scroll = self._make_inspector_scroll(self.review_tab)

        self.inspector_tabs = QTabWidget(self)
        self.inspector_tabs.setObjectName("inspectorTabs")
        self.inspector_tabs.setDocumentMode(True)
        self.inspector_tabs.addTab(self.event_tab_scroll, "Event")
        self.inspector_tabs.addTab(self.objects_tab_scroll, "Objects")
        self.inspector_tabs.addTab(self.review_tab_scroll, "Review")
        self.inspector_tabs.setMinimumWidth(240)
        self.inspector_tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        top_block = QWidget(self)
        top_layout = QVBoxLayout(top_block)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)
        top_layout.addLayout(video_row, 1)
        top_layout.addWidget(self.toolbar_frame)
        top_layout.addWidget(self.onboarding_banner)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        top_layout.addWidget(self.slider)
        self.hoi_timeline = HOITimeline(
            get_segments_for_hand=self._hoi_segments_for_hand,
            get_color_for_verb=self._hoi_color_for_verb,
            on_select=self._on_hoi_timeline_select,
            on_update=self._on_hoi_timeline_update,
            on_create=self._on_hoi_timeline_create,
            on_delete=self._on_hoi_timeline_delete,
            on_activate=self._on_hoi_timeline_activate,
            on_hover=self._on_hoi_timeline_hover,
            get_frame_count=lambda: int(self.player.frame_count or 0),
            get_fps=lambda: int(self.player.frame_rate or 30),
            get_title_for_hand=self._hoi_title_for_hand,
            actors_config=self.actors_config,
            parent=self,
        )
        self.lbl_incomplete = QLabel("Incomplete: n/a")
        self.lbl_incomplete.setToolTip("No HOI completeness check available yet.")
        self.btn_incomplete_prev = QToolButton()
        self.btn_incomplete_prev.setText("<")
        self.btn_incomplete_prev.setToolTip("Previous incomplete segment (A)")
        self.btn_incomplete_prev.clicked.connect(lambda: self._jump_incomplete(-1))
        self.btn_incomplete_next = QToolButton()
        self.btn_incomplete_next.setText(">")
        self.btn_incomplete_next.setToolTip("Next incomplete segment (D)")
        self.btn_incomplete_next.clicked.connect(lambda: self._jump_incomplete(+1))
        self.btn_incomplete_prev.setEnabled(False)
        self.btn_incomplete_next.setEnabled(False)
        self._incomplete_issues = []
        self._incomplete_idx = -1

        timeline_container = QWidget()
        timeline_layout = QVBoxLayout(timeline_container)
        timeline_layout.setContentsMargins(0, 0, 0, 0)
        timeline_layout.setSpacing(4)
        self._inline_editor_sync = False
        self._advanced_inspector_visible = True
        self.inline_event_card = QFrame(self)
        self.inline_event_card.setObjectName("statusCard")
        inline_layout = QVBoxLayout(self.inline_event_card)
        inline_layout.setContentsMargins(10, 8, 10, 8)
        inline_layout.setSpacing(6)

        inline_header = QHBoxLayout()
        inline_header.setContentsMargins(0, 0, 0, 0)
        self.lbl_inline_event_title = QLabel("Quick Edit")
        self.lbl_inline_event_title.setObjectName("statusTitle")
        inline_header.addWidget(self.lbl_inline_event_title, 1)
        self.lbl_inline_event_step = QLabel("Waiting")
        self.lbl_inline_event_step.setAlignment(Qt.AlignCenter)
        inline_header.addWidget(self.lbl_inline_event_step)
        self.btn_inline_toggle_advanced = QToolButton()
        self.btn_inline_toggle_advanced.setAutoRaise(True)
        self.btn_inline_toggle_advanced.clicked.connect(self._toggle_advanced_inspector)
        inline_header.addWidget(self.btn_inline_toggle_advanced)
        inline_layout.addLayout(inline_header)

        self.lbl_inline_event_summary = QLabel(
            "Draw a temporal segment on the timeline to start quick editing."
        )
        self.lbl_inline_event_summary.setWordWrap(True)
        inline_layout.addWidget(self.lbl_inline_event_summary)

        inline_form = QHBoxLayout()
        inline_form.setContentsMargins(0, 0, 0, 0)
        inline_form.setSpacing(8)
        self.lbl_inline_verb = QLabel("Verb")
        inline_form.addWidget(self.lbl_inline_verb)
        self.combo_inline_verb = QComboBox()
        self.combo_inline_verb.setMinimumWidth(180)
        self._enable_combo_search(self.combo_inline_verb, placeholder="Choose verb...")
        self.combo_inline_verb.currentTextChanged.connect(self._on_inline_verb_changed)
        self.combo_inline_verb.activated[str].connect(self._on_inline_verb_changed)
        inline_form.addWidget(self.combo_inline_verb, 1)
        self.lbl_inline_noun = QLabel("Noun")
        inline_form.addWidget(self.lbl_inline_noun)
        self.combo_inline_noun = QComboBox()
        self.combo_inline_noun.setMinimumWidth(200)
        self._enable_combo_search(self.combo_inline_noun, placeholder="Choose noun...")
        self.combo_inline_noun.currentIndexChanged.connect(self._on_inline_noun_changed)
        self.combo_inline_noun.activated[int].connect(self._on_inline_noun_changed)
        inline_form.addWidget(self.combo_inline_noun, 1)
        try:
            verb_editor = self.combo_inline_verb.lineEdit()
            if verb_editor is not None:
                verb_editor.returnPressed.connect(
                    lambda: self._submit_inline_quick_edit_field("verb")
                )
        except Exception:
            pass
        try:
            noun_editor = self.combo_inline_noun.lineEdit()
            if noun_editor is not None:
                noun_editor.returnPressed.connect(
                    lambda: self._submit_inline_quick_edit_field("noun_object_id")
                )
        except Exception:
            pass
        inline_layout.addLayout(inline_form)

        inline_extra_form = QHBoxLayout()
        inline_extra_form.setContentsMargins(0, 0, 0, 0)
        inline_extra_form.setSpacing(8)
        self.lbl_inline_instrument = QLabel("Instrument")
        inline_extra_form.addWidget(self.lbl_inline_instrument)
        self.combo_inline_instrument = QComboBox()
        self.combo_inline_instrument.setMinimumWidth(180)
        self._enable_combo_search(
            self.combo_inline_instrument,
            placeholder="Search instrument / tool...",
            auto_popup=True,
        )
        self.combo_inline_instrument.currentIndexChanged.connect(
            self._on_inline_instrument_changed
        )
        self.combo_inline_instrument.activated[int].connect(
            self._on_inline_instrument_changed
        )
        try:
            inst_editor = self.combo_inline_instrument.lineEdit()
            if inst_editor is not None:
                inst_editor.returnPressed.connect(
                    self._on_inline_instrument_text_finished
                )
                inst_editor.editingFinished.connect(
                    self._on_inline_instrument_text_finished
                )
        except Exception:
            pass
        inline_extra_form.addWidget(self.combo_inline_instrument, 1)
        self.lbl_inline_anomaly = QLabel("Anomaly")
        inline_extra_form.addWidget(self.lbl_inline_anomaly)
        self.combo_inline_anomaly = AttributeSelector()
        self.combo_inline_anomaly.setMinimumWidth(220)
        self._populate_anomaly_combo(self.combo_inline_anomaly)
        self.combo_inline_anomaly.activated[int].connect(
            lambda _index: self._on_inline_anomaly_changed()
        )
        try:
            anomaly_editor = self.combo_inline_anomaly.lineEdit()
            if anomaly_editor is not None:
                anomaly_editor.editingFinished.connect(self._on_inline_anomaly_changed)
        except Exception:
            pass
        inline_extra_form.addWidget(self.combo_inline_anomaly, 1)
        inline_layout.addLayout(inline_extra_form)

        self.inline_grounding_card = QFrame(self.inline_event_card)
        self.inline_grounding_card.setObjectName("groundingAssistCard")
        grounding_layout = QHBoxLayout(self.inline_grounding_card)
        grounding_layout.setContentsMargins(8, 4, 8, 4)
        grounding_layout.setSpacing(6)
        self.lbl_inline_grounding_title = QLabel("Grounding")
        self.lbl_inline_grounding_title.setObjectName("groundingAssistTitle")
        self.lbl_inline_grounding_title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.lbl_inline_grounding_title.setMinimumWidth(82)
        grounding_layout.addWidget(self.lbl_inline_grounding_title)
        self.btn_inline_detect_objects = QPushButton("Find Boxes")
        self.btn_inline_detect_objects.setProperty("groundingAction", True)
        self.btn_inline_detect_objects.setMinimumWidth(108)
        self.btn_inline_detect_objects.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn_inline_detect_objects.setToolTip(
            "Run YOLO on the selected event's start, onset, and end frames to propose editable object boxes. "
            "You can also left-click the target object on the video canvas to request one local box on the current frame."
        )
        self.btn_inline_detect_objects.clicked.connect(
            self._inline_detect_selected_event_objects
        )
        self.lbl_inline_object_candidates = QLabel("Object candidates will appear here.")
        self.lbl_inline_object_candidates.setObjectName("groundingAssistText")
        self.lbl_inline_object_candidates.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.lbl_inline_object_candidates.setWordWrap(False)
        self.lbl_inline_object_candidates.setMinimumWidth(180)
        self.lbl_inline_object_candidates.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grounding_layout.addWidget(self.lbl_inline_object_candidates, 1)
        inline_candidate_row = QHBoxLayout()
        inline_candidate_row.setContentsMargins(0, 0, 0, 0)
        inline_candidate_row.setSpacing(6)
        self._inline_object_candidates = []
        self.inline_object_candidate_buttons = []
        for idx in range(3):
            btn = QPushButton("")
            btn.setProperty("objectCandidate", True)
            btn.setProperty("candidatePrimary", idx == 0)
            btn.setMinimumWidth(96)
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            btn.setVisible(False)
            btn.clicked.connect(
                lambda _checked=False, candidate_idx=idx: self._apply_inline_object_candidate_index(candidate_idx)
            )
            inline_candidate_row.addWidget(btn)
            self.inline_object_candidate_buttons.append(btn)
        grounding_layout.addLayout(inline_candidate_row)
        grounding_layout.addWidget(self.btn_inline_detect_objects)

        inline_actions = QHBoxLayout()
        inline_actions.setContentsMargins(0, 0, 0, 0)
        inline_actions.setSpacing(6)
        inline_actions.addWidget(self.inline_grounding_card, 1)
        inline_actions.addSpacing(2)
        self.btn_inline_edit_boxes = QPushButton("Edit Boxes")
        self.btn_inline_edit_boxes.setCheckable(True)
        self.btn_inline_edit_boxes.setProperty("compactChip", True)
        self.btn_inline_edit_boxes.setToolTip(
            "Enable moving/resizing existing boxes. Ctrl+drag always creates a new box. Shortcut: Ctrl+B"
        )
        self.btn_inline_edit_boxes.toggled.connect(self._set_edit_boxes_enabled)
        inline_actions.addWidget(self.btn_inline_edit_boxes)
        self.btn_correct_track = QPushButton("Track Correction")
        self.btn_correct_track.setToolTip("Correct one instance, then retrack a bounded interval. Keeps human anchors and other IDs; Undo supported.")
        self.btn_correct_track.clicked.connect(self._start_correction_propagation)
        inline_actions.addWidget(self.btn_correct_track)
        self.btn_inline_jump_start = QPushButton("Start")
        self.btn_inline_jump_start.setProperty("compactChip", True)
        self.btn_inline_jump_start.clicked.connect(self._jump_to_selected_start)
        inline_actions.addWidget(self.btn_inline_jump_start)
        self.btn_inline_jump_onset = QPushButton("Onset")
        self.btn_inline_jump_onset.setProperty("compactChip", True)
        self.btn_inline_jump_onset.clicked.connect(self._jump_to_selected_onset)
        inline_actions.addWidget(self.btn_inline_jump_onset)
        self.btn_inline_jump_end = QPushButton("End")
        self.btn_inline_jump_end.setProperty("compactChip", True)
        self.btn_inline_jump_end.clicked.connect(self._jump_to_selected_end)
        inline_actions.addWidget(self.btn_inline_jump_end)
        inline_actions.addSpacing(4)
        self.btn_inline_confirm_onset = QPushButton("Confirm Onset")
        self.btn_inline_confirm_onset.clicked.connect(self._confirm_inline_onset)
        inline_actions.addWidget(self.btn_inline_confirm_onset)
        self.btn_inline_apply_query = QPushButton("Use Recommendation")
        self.btn_inline_apply_query.clicked.connect(self._apply_inline_recommendation)
        inline_actions.addWidget(self.btn_inline_apply_query)
        inline_actions.addStretch(1)
        inline_layout.addLayout(inline_actions)

        self.lbl_inline_query_hint = QLabel("The most relevant next step will appear here.")
        self.lbl_inline_query_hint.setObjectName("inlineActionHint")
        self.lbl_inline_query_hint.setProperty("hintRole", "neutral")
        self.lbl_inline_query_hint.setAlignment(Qt.AlignCenter)
        self.lbl_inline_query_hint.setWordWrap(True)
        inline_layout.addWidget(self.lbl_inline_query_hint)

        timeline_layout.addWidget(self.inline_event_card, 0)
        timeline_layout.addWidget(self.hoi_timeline, 1)
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.addStretch(1)
        footer.addWidget(self.lbl_incomplete)
        footer.addSpacing(6)
        footer.addWidget(self.btn_incomplete_prev)
        footer.addWidget(self.btn_incomplete_next)
        timeline_layout.addLayout(footer)
        timeline_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.hoi_bottom_split = QSplitter(Qt.Horizontal, self)
        self.hoi_bottom_split.setChildrenCollapsible(False)
        self.hoi_bottom_split.setHandleWidth(10)
        self.hoi_bottom_split.setOpaqueResize(False)
        self.hoi_bottom_split.addWidget(self.inspector_tabs)
        self.hoi_bottom_split.addWidget(timeline_container)
        self.hoi_bottom_split.setStretchFactor(0, 0)
        self.hoi_bottom_split.setStretchFactor(1, 1)
        self.hoi_bottom_split.setSizes([280, 980])
        self.hoi_bottom_split.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.root_split = QSplitter(Qt.Vertical, self)
        self.root_split.setChildrenCollapsible(False)
        self.root_split.setHandleWidth(10)
        self.root_split.setOpaqueResize(False)
        self.root_split.addWidget(top_block)
        self.root_split.addWidget(self.hoi_bottom_split)
        self.root_split.setStretchFactor(0, 6)
        self.root_split.setStretchFactor(1, 2)
        self.root_split.setSizes([760, 180])
        root.addWidget(self.root_split, 1)
        self._set_advanced_inspector_visible(False)
        self.chk_filter_current = None

        # shortcuts (only within HOI window)
        self.sc_left = QShortcut(
            QKeySequence(Qt.Key_Left), self, activated=lambda: self._seek_relative(-1)
        )
        self.sc_left.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_right = QShortcut(
            QKeySequence(Qt.Key_Right), self, activated=lambda: self._seek_relative(+1)
        )
        self.sc_right.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_up = QShortcut(
            QKeySequence(Qt.Key_Up), self, activated=lambda: self._seek_seconds(-1)
        )
        self.sc_up.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_down = QShortcut(
            QKeySequence(Qt.Key_Down), self, activated=lambda: self._seek_seconds(+1)
        )
        self.sc_down.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_toggle_play = QShortcut(
            QKeySequence(Qt.Key_Space), self, activated=self._toggle_play_pause
        )
        self.sc_toggle_play.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_pause = QShortcut(QKeySequence("K"), self, activated=self._pause)
        self.sc_pause.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_detect = QShortcut(
            QKeySequence("Ctrl+Shift+D"),
            self,
            activated=self._detect_current_frame_combined,
        )
        self.sc_detect.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_toggle_edit_boxes = QShortcut(
            QKeySequence("Ctrl+B"),
            self,
            activated=self._toggle_edit_boxes_shortcut,
        )
        self.sc_toggle_edit_boxes.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_jump_start = QShortcut(
            QKeySequence("Z"), self, activated=self._jump_to_selected_start
        )
        self.sc_jump_start.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_jump_onset = QShortcut(
            QKeySequence("X"), self, activated=self._jump_to_selected_onset
        )
        self.sc_jump_onset.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_jump_end = QShortcut(
            QKeySequence("C"), self, activated=self._jump_to_selected_end
        )
        self.sc_jump_end.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_inc_prev = QShortcut(
            QKeySequence("A"), self, activated=lambda: self._jump_incomplete(-1)
        )
        self.sc_inc_prev.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_inc_next = QShortcut(
            QKeySequence("D"), self, activated=lambda: self._jump_incomplete(+1)
        )
        self.sc_inc_next.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_undo = QShortcut(QKeySequence.Undo, self, activated=self._hoi_undo)
        self.sc_undo.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_redo = QShortcut(QKeySequence.Redo, self, activated=self._hoi_redo)
        self.sc_redo.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_new_trial = QShortcut(
            QKeySequence("Ctrl+Shift+N"),
            self,
            activated=self._new_trial_reset_workspace,
        )
        self.sc_new_trial.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_open_settings = QShortcut(QKeySequence("Ctrl+,"), self, activated=self._open_settings_dialog)
        self.sc_open_settings.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_open_quick_start = QShortcut(QKeySequence("F1"), self, activated=self._open_quick_start_dialog)
        self.sc_open_quick_start.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_toggle_user_study_mode = QShortcut(
            QKeySequence("Ctrl+Alt+U"),
            self,
            activated=self._toggle_user_study_mode,
        )
        self.sc_toggle_user_study_mode.setContext(Qt.WidgetWithChildrenShortcut)
        self.apply_shortcut_settings(self._shortcut_bindings)

        # 3. Safely handle list_objects signals
        try:
            self.list_objects.itemSelectionChanged.disconnect()
        except (TypeError, RuntimeError):
            pass

        # 4. Connect new logic
        self.list_objects.itemSelectionChanged.connect(self._on_object_selection)
        # --- New: Global Object ID Registry ---
        # Structure: { "screwdriver_1": 0, "gear_1": 1, ... }
        self.global_object_map: Dict[str, int] = {}
        self.object_id_counter = 0

        self.entity_library = {}
        self.id_to_category = {}
        self._noun_only_mode = True
        self.hoi_ontology = HOIOntology()
        self.hoi_ontology_path = ""
        self.semantic_adapter_package = None
        self.semantic_adapter_base_model_path = ""
        self.semantic_adapter_active_model_path = ""
        self.semantic_adapter_model_path = ""
        self.semantic_feedback_path = ""
        self._semantic_feedback_pending = 0
        self._semantic_feedback_feature_dim = 0
        self._semantic_adapter_failed_load_paths: set = set()
        self._semantic_adapter_unavailable_reason = ""
        self._semantic_videomae_feature_dim = 32
        self._semantic_videomae_local_feature_dim = 32
        self._semantic_refinement_passes = 2
        self._semantic_reinfer_hints: Dict[str, Dict[str, Any]] = {}
        self._detector_grounding_only = True
        self._semantic_adapter_train_worker = None
        self._semantic_adapter_train_config = {
            "hidden_dim": 96,
            "onset_bins": 21,
            "video_adapter_rank": 8,
            "video_adapter_alpha": 8.0,
            "epochs": 12,
            "batch_size": 16,
            "lr": 1e-3,
            "min_samples": 8,
            "train_every": 6,
        }
        self._semantic_review_thresholds = {
            "onset_confidence": 0.86,
            "verb_confidence": 0.82,
            "noun_confidence": 0.80,
            "noun_exists": 0.62,
            "onset_band_width": 0.22,
            "risk_score": 0.34,
        }

        self.combo_verb.currentTextChanged.connect(self._on_hoi_meta_changed)
        self.combo_target.currentIndexChanged.connect(self._on_hoi_meta_changed)
        self.combo_instrument.currentIndexChanged.connect(self._on_hoi_meta_changed)
        try:
            instrument_editor = self.combo_instrument.lineEdit()
            if instrument_editor is not None:
                instrument_editor.returnPressed.connect(self._on_hoi_meta_changed)
                instrument_editor.editingFinished.connect(self._on_hoi_meta_changed)
        except Exception:
            pass
        self.combo_anomaly.activated[int].connect(
            lambda _index: self._on_hoi_meta_changed()
        )
        try:
            anomaly_editor = self.combo_anomaly.lineEdit()
            if anomaly_editor is not None:
                anomaly_editor.editingFinished.connect(self._on_hoi_meta_changed)
        except Exception:
            pass
        self._mark_hoi_saved()

        self._update_verb_combo()
        self._init_autosave()
        self._set_verb_library_admin_mode(False)
        self._update_draw_mode_visibility()
        self._update_inline_edit_boxes_button_state()
        self._apply_noun_only_mode_ui()
        self._set_ui_scale(getattr(self, "_ui_scale", 0.85), persist=False)
        self._apply_experiment_mode_ui()
        self._apply_user_study_ui_profile()
        self._set_validation_ui_state(False)
        self._update_incomplete_indicator()
        self._update_status_label()
        self._update_clip_readiness_ui()
        self._update_onboarding_banner()
        self._update_next_best_query_panel()
        self._update_play_pause_button()

    def _make_inspector_scroll(self, content: QWidget) -> QScrollArea:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        scroll.setWidget(content)
        return scroll

    def _apply_user_study_ui_profile(self) -> None:
        study_mode = bool(getattr(self, "_user_study_mode", True))
        self._user_study_ui_profile = study_mode

        combo = getattr(self, "combo_experiment_mode", None)
        if combo is not None:
            current_mode = str(combo.currentData() or "full_assist")
            keep_modes = [
                ("Manual", "manual"),
                ("Full Assist", "full_assist"),
            ]
            combo.blockSignals(True)
            combo.clear()
            for label, value in keep_modes:
                combo.addItem(label, value)
            target_mode = current_mode if current_mode in {"manual", "full_assist"} else "full_assist"
            target_index = max(0, combo.findData(target_mode))
            combo.setCurrentIndex(target_index)
            combo.blockSignals(False)
            combo.setToolTip(
                "Manual: annotate directly. Full Assist: show action suggestions and box assistance."
            )

        operator_only_actions = (
            getattr(self, "act_new_trial", None),
            getattr(self, "act_prune_object_library", None),
            getattr(self, "act_load_yaml", None),
            getattr(self, "act_import_yolo_boxes", None),
            getattr(self, "act_load_hands_xml", None),
            getattr(self, "act_toggle_verb_library_admin", None),
            getattr(self, "act_save_hands_xml", None),
            getattr(self, "act_show_onboarding_banner", None),
            getattr(self, "act_open_settings", None),
        )
        for action in operator_only_actions:
            if action is not None:
                action.setVisible(not study_mode)
                action.setEnabled(not study_mode)

        mode_sensitive_hidden_actions = (
            getattr(self, "act_detect_current_frame", None),
            getattr(self, "act_detect_selected_action", None),
            getattr(self, "act_detect_all_actions", None),
            getattr(self, "act_precompute_hand_tracks", None),
            getattr(self, "act_swap_hands", None),
            getattr(self, "act_incremental_train_yolo", None),
            getattr(self, "act_review_selected_action_label", None),
            getattr(self, "act_auto_apply_action_labels", None),
            getattr(self, "act_batch_apply_action_labels", None),
        )
        for action in mode_sensitive_hidden_actions:
            if action is not None:
                action.setVisible(not study_mode)
                if study_mode:
                    action.setEnabled(False)

        for widget in (
            getattr(self, "lbl_validation", None),
            getattr(self, "btn_validation", None),
            getattr(self, "btn_detect", None),
            getattr(self, "btn_detect_action", None),
            getattr(self, "btn_object_tools", None),
            getattr(self, "btn_action_tools", None),
            getattr(self, "btn_inline_toggle_advanced", None),
            getattr(self, "lbl_incomplete", None),
            getattr(self, "btn_incomplete_prev", None),
            getattr(self, "btn_incomplete_next", None),
            getattr(self, "lbl_validation_chip", None),
            getattr(self, "lbl_incomplete_chip", None),
            getattr(self, "btn_review_prev", None),
            getattr(self, "btn_review_next", None),
        ):
            if widget is not None:
                widget.setVisible(not study_mode)

        if getattr(self, "detect_menu", None) is not None:
            self.detect_menu.menuAction().setVisible(not study_mode)
        if getattr(self, "assist_menu", None) is not None:
            self.assist_menu.menuAction().setVisible(not study_mode)

        self._rebuild_object_tools_menu()
        self._rebuild_action_tools_menu()

        if getattr(self, "btn_file_menu", None) is not None:
            if study_mode:
                self.btn_file_menu.setToolTip("Load clip, import study files, and save.")
            else:
                self.btn_file_menu.setToolTip("Project, import/export, detection, and model actions")

        banner = getattr(self, "onboarding_banner", None)
        if banner is not None:
            banner.setVisible(False if study_mode else not bool(getattr(self, "_onboarding_dismissed_session", False)))

        if study_mode:
            self._set_advanced_inspector_visible(False)

        if getattr(self, "sc_new_trial", None) is not None:
            self.sc_new_trial.setEnabled(not study_mode)
        if getattr(self, "sc_detect", None) is not None:
            self.sc_detect.setEnabled(not study_mode)
        if getattr(self, "sc_open_settings", None) is not None:
            self.sc_open_settings.setEnabled(not study_mode)
        if getattr(self, "sc_open_quick_start", None) is not None:
            self.sc_open_quick_start.setEnabled(True)
        if getattr(self, "act_open_quick_start", None) is not None:
            self.act_open_quick_start.setVisible(True)
            self.act_open_quick_start.setEnabled(True)
        if getattr(self, "btn_quick_start", None) is not None:
            self.btn_quick_start.setVisible(True)
            self.btn_quick_start.setEnabled(True)

    def _rebuild_object_tools_menu(self) -> None:
        menu = getattr(self, "object_tools_menu", None)
        if menu is None:
            return
        study_mode = bool(getattr(self, "_user_study_mode", True))
        menu.clear()
        if not study_mode:
            if getattr(self, "act_detect_all_actions", None) is not None:
                menu.addAction(self.act_detect_all_actions)
            menu.addSeparator()
            if getattr(self, "act_load_yolo_model", None) is not None:
                menu.addAction(self.act_load_yolo_model)
            if getattr(self, "act_incremental_train_yolo", None) is not None:
                menu.addAction(self.act_incremental_train_yolo)
            menu.addSeparator()
            if getattr(self, "act_swap_hands", None) is not None:
                menu.addAction(self.act_swap_hands)
        btn = getattr(self, "btn_object_tools", None)
        if btn is not None:
            btn.setMenu(menu)
            btn.setVisible((not study_mode) and bool(menu.actions()))

    def _rebuild_action_tools_menu(self) -> None:
        menu = getattr(self, "action_tools_menu", None)
        if menu is None:
            return
        study_mode = bool(getattr(self, "_user_study_mode", True))
        menu.clear()
        if not study_mode:
            if getattr(self, "act_toggle_verb_library_admin", None) is not None:
                menu.addAction(self.act_toggle_verb_library_admin)
            menu.addSeparator()
            if getattr(self, "act_batch_apply_action_labels", None) is not None:
                menu.addAction(self.act_batch_apply_action_labels)
        btn = getattr(self, "btn_action_tools", None)
        if btn is not None:
            btn.setMenu(menu)
            btn.setVisible(not study_mode)

    def _toggle_user_study_mode(self) -> None:
        self._set_user_study_mode(False)

    def _set_user_study_mode(self, on: bool) -> None:
        on = False
        if on == bool(getattr(self, "_user_study_mode", True)):
            return
        self._user_study_mode = on
        self._persist_ui_preferences()
        self._apply_experiment_mode_ui()
        self._update_onboarding_banner()
        self._update_status_label()
        self._log("hoi_user_study_mode_changed", enabled=bool(on))

    def _normalized_ui_scale(self, value: Optional[float] = None) -> float:
        try:
            scale = float(self._ui_scale if value is None else value)
        except Exception:
            scale = 0.85
        if scale > 10.0:
            scale = scale / 100.0
        return max(0.80, min(1.25, scale))

    def _scaled_ui_px(self, base: float, min_px: int = 0) -> int:
        try:
            px = int(round(float(base) * self._normalized_ui_scale()))
        except Exception:
            px = int(round(float(base)))
        return max(int(min_px), px)

    def _reset_workspace_layout(self) -> None:
        total_w = max(1000, int(self.width() or 0) - 20)
        inspector_w = min(300, max(240, int(total_w * 0.20)))
        total_h = max(720, int(self.height() or 0) - 40)
        bottom_h = max(180, int(total_h * 0.21))
        top_h = max(500, total_h - bottom_h)
        try:
            self.hoi_bottom_split.setSizes([inspector_w, max(720, total_w - inspector_w)])
        except Exception:
            pass
        try:
            self.root_split.setSizes([top_h, bottom_h])
        except Exception:
            pass

    def _apply_logging_settings(self, oplog_enabled: bool, validation_summary_enabled: bool) -> None:
        handled = False
        parent = self.parentWidget()
        if parent is not None and hasattr(parent, "_set_logging_policy"):
            try:
                parent._set_logging_policy(bool(oplog_enabled), bool(validation_summary_enabled))
                handled = True
            except Exception:
                handled = False
        if not handled:
            self.set_logging_policy(bool(oplog_enabled), bool(validation_summary_enabled))
            ok_save, path_or_err = save_logging_policy({
                "ops_csv_enabled": bool(oplog_enabled),
                "validation_summary_enabled": bool(validation_summary_enabled),
            })
            if not ok_save:
                print(f"[LOG][ERROR] Failed to save logging policy: {path_or_err}")

    def _set_ui_scale(self, scale: float, persist: bool = False) -> None:
        self._ui_scale = self._normalized_ui_scale(scale)
        self._apply_professional_ui_style()
        self._apply_micro_interaction_icons()
        try:
            self.combo_experiment_mode.setMinimumWidth(self._scaled_ui_px(120, 104))
        except Exception:
            pass
        try:
            self.edit_participant_code.setMinimumWidth(self._scaled_ui_px(96, 84))
            self.edit_participant_code.setMaximumWidth(self._scaled_ui_px(132, 116))
        except Exception:
            pass
        try:
            self.btn_file_menu.setFixedWidth(self._scaled_ui_px(30, 24))
            self.btn_quick_start.setFixedWidth(self._scaled_ui_px(30, 24))
        except Exception:
            pass
        try:
            self.hoi_bottom_split.setHandleWidth(self._scaled_ui_px(10, 8))
            self.root_split.setHandleWidth(self._scaled_ui_px(10, 8))
        except Exception:
            pass
        try:
            if getattr(self, "hoi_timeline", None):
                self.hoi_timeline.updateGeometry()
                self.hoi_timeline.refresh()
        except Exception:
            pass
        self.updateGeometry()
        self.update()
        if persist:
            self._persist_ui_preferences()
            try:
                self._log("hoi_ui_scale_update", ui_scale=self._ui_scale)
            except Exception:
                pass

    def _persist_ui_preferences(self) -> None:
        ok_save, path_or_err = save_ui_preferences(
            {
                "ui_scale": self._ui_scale,
                "show_quick_start_on_startup": bool(
                    self._show_quick_start_on_startup
                ),
                "user_study_mode": False,
                "participant_code": "",
            }
        )
        if not ok_save:
            print(f"[UI][ERROR] Failed to save UI preferences: {path_or_err}")

    def _normalized_participant_code(self, value: Optional[str] = None) -> str:
        return ""

    def _safe_filename_token(self, value: Any, fallback: str = "") -> str:
        text = str(value or "").strip()
        if not text:
            text = str(fallback or "").strip()
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("._-")
        return text or str(fallback or "").strip("._-") or "item"

    def _require_participant_code_for_study_save(self) -> bool:
        return True

    def _semantic_participant_key(self) -> str:
        raw = self._normalized_participant_code()
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw or "").strip())
        safe = safe.strip("._-")
        return safe or "shared"

    def _repo_root_dir(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _semantic_runtime_artifacts_dir(self) -> str:
        path = os.path.join(
            self._repo_root_dir(),
            "runtime_artifacts",
            "participants",
            self._semantic_participant_key(),
        )
        os.makedirs(path, exist_ok=True)
        return path

    def _shared_semantic_adapter_file(self) -> str:
        return os.path.join(self._repo_root_dir(), "runtime_artifacts", "semantic_adapter.pt")

    def _switch_semantic_workspace_for_participant(self) -> None:
        self.semantic_adapter_model_path = ""
        self.semantic_adapter_active_model_path = ""
        self.semantic_feedback_path = ""
        self.semantic_adapter_package = None
        self._semantic_feedback_pending = 0
        self._semantic_feedback_feature_dim = 0
        self._log(
            "hoi_participant_code_changed",
            participant_code=self._normalized_participant_code(),
            semantic_workspace=self._semantic_runtime_artifacts_dir(),
        )
        self._log_annotation_ready_state("hoi_participant_code_changed")
        self._ensure_semantic_adapter_loaded()
        if self.selected_event_id is not None:
            event = self._find_event_by_id(self.selected_event_id)
            if event is not None:
                self._refresh_semantic_suggestions_for_event(self.selected_event_id, event)
                self._update_status_label()

    def _on_participant_code_changed(self) -> None:
        code = self._normalized_participant_code()
        editor = getattr(self, "edit_participant_code", None)
        if editor is not None and editor.text() != code:
            editor.blockSignals(True)
            editor.setText(code)
            editor.blockSignals(False)
        if code == getattr(self, "_participant_code", ""):
            return
        self._participant_code = code
        self._persist_ui_preferences()
        self._switch_semantic_workspace_for_participant()

    def _default_annotation_basename(self) -> str:
        base = self._safe_filename_token(
            os.path.splitext(os.path.basename(self.video_path or "annotation"))[0],
            fallback="annotation",
        )
        parts = []
        participant_code = self._normalized_participant_code()
        if participant_code:
            parts.append(self._safe_filename_token(participant_code, fallback="participant"))
        parts.append(base)
        parts.append(self._safe_filename_token(self._experiment_mode_key(), fallback="full_assist"))
        return "_".join(
            str(part).strip("_") for part in parts if str(part).strip("_")
        )

    def _open_settings_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Settings")
        dialog.resize(420, 260)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        grp_appearance = QGroupBox("Appearance", dialog)
        form_appearance = QFormLayout(grp_appearance)
        form_appearance.setContentsMargins(12, 12, 12, 12)
        combo_scale = QComboBox(grp_appearance)
        options = [
            (0.80, "80%"),
            (0.85, "85%"),
            (0.90, "90%"),
            (1.00, "100%"),
            (1.10, "110%"),
            (1.20, "120%"),
        ]
        current_scale = self._normalized_ui_scale()
        current_idx = 0
        best_diff = 999.0
        for idx, (value, label) in enumerate(options):
            combo_scale.addItem(label, value)
            diff = abs(value - current_scale)
            if diff < best_diff:
                best_diff = diff
                current_idx = idx
        combo_scale.setCurrentIndex(current_idx)
        form_appearance.addRow("UI Scale", combo_scale)
        lbl_scale_hint = QLabel("Scales fonts, button density, and toolbar icons.", grp_appearance)
        lbl_scale_hint.setObjectName("statusSubtle")
        lbl_scale_hint.setWordWrap(True)
        form_appearance.addRow("", lbl_scale_hint)
        chk_quick_start = QCheckBox("Show Quick Start on startup", grp_appearance)
        chk_quick_start.setChecked(bool(self._show_quick_start_on_startup))
        form_appearance.addRow("Onboarding", chk_quick_start)
        btn_reset_layout = QPushButton("Reset Layout", grp_appearance)
        btn_reset_layout.clicked.connect(self._reset_workspace_layout)
        form_appearance.addRow("Panels", btn_reset_layout)
        layout.addWidget(grp_appearance)

        grp_logging = QGroupBox("Evaluation Logging", dialog)
        logging_layout = QVBoxLayout(grp_logging)
        logging_layout.setContentsMargins(12, 12, 12, 12)
        chk_ops = QCheckBox("Write operations CSV", grp_logging)
        chk_ops.setChecked(bool(getattr(self.op_logger, "enabled", False)))
        chk_validation = QCheckBox("Write validation summary", grp_logging)
        chk_validation.setChecked(bool(getattr(self, "validation_summary_enabled", True)))
        logging_layout.addWidget(chk_ops)
        logging_layout.addWidget(chk_validation)
        layout.addWidget(grp_logging)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() != QDialog.Accepted:
            return

        self._set_ui_scale(float(combo_scale.currentData() or self._ui_scale), persist=True)
        self._show_quick_start_on_startup = bool(chk_quick_start.isChecked())
        self._persist_ui_preferences()
        self._apply_logging_settings(chk_ops.isChecked(), chk_validation.isChecked())

    def _docs_dir(self) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))

    def _quick_start_asset_path(self, name: str) -> str:
        return os.path.join(self._docs_dir(), "assets", name)

    def _make_quick_start_image(self, image_path: str, max_width: int = 980) -> QLabel:
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        if os.path.exists(image_path):
            pix = QPixmap(image_path)
            if not pix.isNull():
                if pix.width() > max_width:
                    pix = pix.scaledToWidth(max_width, Qt.SmoothTransformation)
                label.setPixmap(pix)
                return label
        label.setText(f"Missing help image:\n{os.path.basename(image_path)}")
        label.setObjectName("statusSubtle")
        return label

    def _make_quick_start_text(self, html: str) -> QTextBrowser:
        browser = QTextBrowser(self)
        browser.setOpenExternalLinks(False)
        browser.setReadOnly(True)
        browser.setFrameShape(QFrame.NoFrame)
        browser.setHtml(html)
        return browser

    def _maybe_open_quick_start_on_startup(self) -> None:
        if not bool(getattr(self, "_quick_start_auto_open_pending", False)):
            return
        self._quick_start_auto_open_pending = False
        if bool(getattr(self, "_user_study_mode", True)):
            return
        if not bool(getattr(self, "_show_quick_start_on_startup", True)):
            return
        if bool(getattr(self, "_quick_start_dialog_open", False)):
            return
        self._open_quick_start_dialog()

    def _make_onboarding_card(
        self,
        title: str,
        body_lines: List[str],
        accent: str = "#2563EB",
        tone: str = "neutral",
    ) -> QFrame:
        palette = {
            "neutral": ("#FFFFFF", "#D7DEE8", "#475569"),
            "blue": ("#EFF6FF", "#BFDBFE", "#1D4ED8"),
            "green": ("#F0FDF4", "#BBF7D0", "#15803D"),
            "amber": ("#FFFBEB", "#FDE68A", "#B45309"),
            "red": ("#FEF2F2", "#FECACA", "#B91C1C"),
        }
        bg, border, text_col = palette.get(tone, palette["neutral"])
        frame = QFrame(self)
        frame.setStyleSheet(
            f"background: {bg}; border: 1px solid {border}; border-radius: 14px;"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        chip = QLabel(title, frame)
        chip.setStyleSheet(
            f"background: {accent}; color: white; border-radius: 999px; padding: 4px 10px; font-weight: 600; border: none;"
        )
        chip.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addWidget(chip, 0, Qt.AlignLeft)

        for idx, line in enumerate(body_lines):
            lbl = QLabel(line, frame)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(
                f"color: {'#0F172A' if idx == 0 else text_col}; background: transparent; border: none;"
            )
            layout.addWidget(lbl)
        layout.addStretch(1)
        return frame

    def _make_onboarding_grid(self, cards: List[Dict[str, Any]], columns: int = 2) -> QWidget:
        host = QWidget(self)
        grid = QGridLayout(host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for idx, spec in enumerate(cards):
            row = idx // columns
            col = idx % columns
            card = self._make_onboarding_card(
                spec.get("title", ""),
                spec.get("body_lines", []),
                accent=spec.get("accent", "#2563EB"),
                tone=spec.get("tone", "neutral"),
            )
            grid.addWidget(card, row, col)
        for col in range(columns):
            grid.setColumnStretch(col, 1)
        return host

    def _clear_spotlight_widget(self) -> None:
        widget = getattr(self, "_spotlight_widget_ref", None)
        effect = getattr(self, "_spotlight_effect", None)
        if widget is not None:
            try:
                if widget.graphicsEffect() is effect:
                    widget.setGraphicsEffect(None)
            except Exception:
                pass
        self._spotlight_widget_ref = None
        self._spotlight_effect = None

    def _spotlight_widget(self, widget: Optional[QWidget], color: str = "#2563EB", duration_ms: int = 2200) -> None:
        if widget is None:
            return
        self._clear_spotlight_widget()
        try:
            effect = QGraphicsDropShadowEffect(widget)
            effect.setBlurRadius(32)
            effect.setOffset(0, 0)
            effect.setColor(QColor(color))
            widget.setGraphicsEffect(effect)
            self._spotlight_widget_ref = widget
            self._spotlight_effect = effect
            self._spotlight_timer.start(max(600, int(duration_ms)))
        except Exception:
            self._clear_spotlight_widget()

    def _onboarding_focus_file_menu(self) -> None:
        self._spotlight_widget(getattr(self, "btn_file_menu", None), "#2563EB")

    def _onboarding_focus_event_panel(self) -> None:
        self._focus_inspector_tab("event")
        self._spotlight_widget(getattr(self, "group_action", None), "#2563EB")

    def _onboarding_focus_objects_panel(self) -> None:
        self._focus_inspector_tab("objects")
        self._spotlight_widget(getattr(self, "group_library", None), "#EA580C")

    def _onboarding_focus_review_panel(self) -> None:
        self._focus_inspector_tab("review")
        self._spotlight_widget(getattr(self, "next_query_card", None), "#DC2626")

    def _onboarding_highlight_timeline(self) -> None:
        self._focus_timeline_workspace()
        self._spotlight_widget(getattr(self, "hoi_timeline", None), "#2563EB")

    def _onboarding_highlight_draw_tools(self) -> None:
        self._focus_inspector_tab("objects")
        target = getattr(self, "draw_mode_widget", None) or getattr(self, "chk_edit_boxes", None)
        self._spotlight_widget(target, "#EA580C")

    def _dismiss_onboarding_banner(self) -> None:
        self._onboarding_dismissed_session = True
        self._clear_spotlight_widget()
        if getattr(self, "onboarding_banner", None) is not None:
            self.onboarding_banner.hide()

    def _show_onboarding_banner(self) -> None:
        self._onboarding_dismissed_session = False
        self._update_onboarding_banner()

    def _set_onboarding_step_chip(self, widget: Optional[QLabel], text: str, done: bool) -> None:
        if widget is None:
            return
        if done:
            bg, fg, border = "#ECFDF3", "#027A48", "#86EFAC"
        else:
            bg, fg, border = "#F8FAFC", "#475467", "#D7DEE8"
        widget.setText(text)
        widget.setStyleSheet(
            f"background: {bg}; color: {fg}; border: 1px solid {border}; border-radius: 999px; padding: 4px 10px; font-weight: 600;"
        )

    def _clear_layout_widgets(self, layout) -> None:
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout_widgets(child_layout)

    def _make_banner_action_button(self, text: str, callback, primary: bool = False) -> QPushButton:
        btn = QPushButton(text, self.onboarding_banner)
        btn.setAutoDefault(False)
        btn.setCursor(Qt.PointingHandCursor)
        if callback is not None:
            btn.clicked.connect(callback)
        if primary:
            btn.setStyleSheet(
                "QPushButton { background: #2563EB; color: white; border: 1px solid #2563EB; border-radius: 10px; padding: 7px 12px; font-weight: 600; }"
                "QPushButton:hover { background: #1D4ED8; border-color: #1D4ED8; }"
            )
        else:
            btn.setStyleSheet(
                "QPushButton { background: white; color: #0F172A; border: 1px solid #D7DEE8; border-radius: 10px; padding: 7px 12px; font-weight: 600; }"
                "QPushButton:hover { background: #F8FAFC; border-color: #BFDBFE; }"
            )
        return btn

    def _focus_timeline_workspace(self) -> None:
        try:
            self.root_split.setSizes([740, 180])
        except Exception:
            pass
        try:
            self.hoi_bottom_split.setSizes([0, 1260] if not self._advanced_inspector_visible else [300, 960])
        except Exception:
            pass
        try:
            self.hoi_timeline.setFocus()
        except Exception:
            pass

    def _select_first_event(self) -> None:
        if not self.events:
            return
        ev = sorted(self.events, key=lambda x: int(x.get("event_id", 10**9)))[0]
        hand_key = None
        for actor in self.actors_config:
            hid = actor["id"]
            h = ev.get("hoi_data", {}).get(hid, {}) or {}
            if h.get("interaction_start") is not None or h.get("interaction_end") is not None:
                hand_key = hid
                break
        self._set_selected_event(int(ev.get("event_id")), hand_key)
        self._focus_inspector_tab("event")

    def _highlight_onboarding_target(self, focus_key: Optional[str]) -> None:
        key = str(focus_key or "").strip().lower()
        if not key:
            self._clear_spotlight_widget()
            self._last_onboarding_focus_key = None
            return
        if key == getattr(self, "_last_onboarding_focus_key", None):
            return
        self._last_onboarding_focus_key = key
        if key in {"load_video", "import_inst", "import_target", "import_verb", "save"}:
            self._onboarding_focus_file_menu()
        elif key in {"create_event", "select_event"}:
            self._onboarding_highlight_timeline()
        elif key in {"fill_verb", "confirm_onset"}:
            self._onboarding_focus_event_panel()
        elif key == "fill_links":
            self._onboarding_focus_objects_panel()
        elif key == "fill_boxes":
            self._onboarding_highlight_draw_tools()
        elif key == "review_ready":
            self._onboarding_focus_review_panel()

    def _current_onboarding_hand(self) -> Optional[dict]:
        if self.selected_event_id is not None and self.selected_hand_label in getattr(self, "event_draft", {}):
            hand_data = self.event_draft.get(self.selected_hand_label) or {}
            return self._ensure_hand_annotation_state(hand_data)
        if not self.events:
            return None
        ev = self.events[0]
        for actor in self.actors_config:
            hid = actor["id"]
            hand_data = ev.get("hoi_data", {}).get(hid, {}) or {}
            if any(
                [
                    hand_data.get("interaction_start") is not None,
                    hand_data.get("interaction_end") is not None,
                    hand_data.get("functional_contact_onset") is not None,
                    bool(hand_data.get("verb")),
                    self._hand_noun_object_id(hand_data) is not None,
                    self._hand_instrument_object_id(hand_data) is not None,
                    self._is_anomalous_label(hand_data.get("anomaly_label")),
                ]
            ):
                return self._ensure_hand_annotation_state(hand_data)
        return None

    def _update_onboarding_banner(self) -> None:
        banner = getattr(self, "onboarding_banner", None)
        if banner is None:
            return
        if bool(getattr(self, "_user_study_mode", True)):
            banner.hide()
            return
        if bool(getattr(self, "_onboarding_dismissed_session", False)):
            banner.hide()
            return

        has_video = bool(str(getattr(self, "video_path", "") or "").strip())
        has_targets = bool(getattr(self, "combo_target", None) and self.combo_target.count() > 1)
        has_verbs = bool(len(getattr(self, "verbs", []) or []))
        has_events = bool(len(getattr(self, "events", []) or []))
        has_any_boxes = bool(len(getattr(self, "raw_boxes", []) or []))
        hand_data = self._current_onboarding_hand()
        if hand_data:
            evidence = self._sparse_evidence_summary(hand_data)
            has_any_boxes = has_any_boxes or bool(int(evidence.get("confirmed", 0) or 0) > 0)

        steps = {
            "video": has_video,
            "target": has_targets,
            "verb": has_verbs,
            "event": has_events,
        }
        if not self._noun_only_mode:
            steps["boxes"] = has_any_boxes
        done_count = sum(1 for v in steps.values() if v)
        self.lbl_onboarding_progress.setText(f"{done_count} / {len(steps)} ready")
        for key, widget in self._onboarding_step_chips.items():
            self._set_onboarding_step_chip(widget, widget.text(), bool(steps.get(key)))

        specs = []
        summary = (
            "Use the guide to start a new HOI session."
            if not self._manual_mode_enabled()
            else "Use the guide to start a manual HOI annotation session."
        )
        title = "Getting Started"
        focus_key = None

        def _add_spec(text, callback, primary=False):
            specs.append((text, callback, primary))

        if not has_video:
            title = "Start here"
            summary = "Load a video to begin an annotation session."
            focus_key = "load_video"
            _add_spec("Load Video...", self._load_video, True)
            _add_spec("Start Manual", lambda: self._set_experiment_mode_key("manual"))
        elif not has_targets:
            title = "Project setup"
            summary = "Import the noun/object list so events can reference valid HOI objects."
            focus_key = "import_target"
            _add_spec("Import Noun List...", self._import_targets, True)
            _add_spec("Open Guide", self._open_quick_start_dialog)
        elif not has_verbs:
            title = "Project setup"
            summary = "Import the verb list before assigning actions to HOI events."
            focus_key = "import_verb"
            _add_spec("Import Verb List...", self._load_verbs_txt, True)
            _add_spec("Open Guide", self._open_quick_start_dialog)
        elif not has_events:
            title = "Create the first event"
            summary = "Create your first HOI event by dragging on an actor row in the timeline."
            focus_key = "create_event"
            _add_spec("Focus Timeline", self._onboarding_highlight_timeline, True)
            _add_spec("Start Manual", lambda: self._set_experiment_mode_key("manual"))
        elif self.selected_event_id is None:
            title = "Select an event"
            summary = "Select an existing event to continue filling timing, links, and evidence."
            focus_key = "select_event"
            _add_spec("Select First Event", self._select_first_event, True)
            _add_spec("Focus Timeline", self._onboarding_highlight_timeline)
        else:
            completion_state = self._hand_completion_state(
                hand_data,
                hand_key=self.selected_hand_label,
            )
            missing_fields = list(completion_state.get("missing") or [])
            suggested_fields = list(completion_state.get("suggested_fields") or [])
            if (
                "onset" in missing_fields
                or "functional_contact_onset" in suggested_fields
            ):
                title = "Set onset" if self._manual_mode_enabled() else "Confirm onset"
                summary = (
                    "The selected event still needs its functional-contact onset on the timeline."
                    if self._manual_mode_enabled()
                    else "The selected event still needs its functional-contact onset."
                )
                focus_key = "confirm_onset"
                _add_spec("Focus Event", self._onboarding_focus_event_panel, True)
                _add_spec("Open Review", self._onboarding_focus_review_panel)
            elif "verb" in missing_fields:
                title = "Fill action"
                summary = (
                    "The selected event still needs a verb. Choose it in the event controls."
                    if self._manual_mode_enabled()
                    else "The selected event still needs a verb. Use the Action panel or import/load verb assistance."
                )
                focus_key = "fill_verb"
                _add_spec("Focus Event", self._onboarding_focus_event_panel, True)
                _add_spec("Open Guide", self._open_quick_start_dialog)
            elif "noun" in missing_fields:
                title = "Link objects"
                summary = "The selected event still needs its noun/object assignment."
                focus_key = "fill_links"
                _add_spec("Focus Objects", self._onboarding_focus_objects_panel, True)
                if self._detection_assist_enabled():
                    _add_spec("Detect Action", self._detect_selected_action)
            elif suggested_fields:
                title = "Review suggestions"
                summary = (
                    "Some event fields still carry non-confirmed suggestions. Review and confirm them before saving."
                    if self._manual_mode_enabled()
                    else "Some event fields still carry suggestions. Confirm them in Quick Edit or Review before saving."
                )
                focus_key = "review_ready"
                _add_spec("Open Review", self._onboarding_focus_review_panel, True)
                _add_spec("Focus Event", self._onboarding_focus_event_panel)
            else:
                title = "Event completed" if self._manual_mode_enabled() else "Event completed"
                summary = (
                    "The selected event is completed. Continue editing if needed or save the annotation."
                    if self._manual_mode_enabled()
                    else "The selected event is completed. Review the event graph if needed, then save."
                )
                focus_key = "review_ready"
                _add_spec("Open Review", self._onboarding_focus_review_panel, True)
                _add_spec("Save...", self._save_annotations_json)

        self.lbl_onboarding_title.setText(title)
        self.lbl_onboarding_summary.setText(summary)
        self._clear_layout_widgets(self.onboarding_actions_row)
        for idx, (text, callback, primary) in enumerate(specs[:3]):
            self.onboarding_actions_row.addWidget(
                self._make_banner_action_button(text, callback, primary=bool(primary))
            )
        self.onboarding_actions_row.addStretch(1)
        banner.show()
        self._highlight_onboarding_target(focus_key)

    def _set_experiment_mode_key(self, mode_key: str) -> None:
        combo = getattr(self, "combo_experiment_mode", None)
        if combo is None:
            return
        idx = combo.findData(str(mode_key or ""))
        if idx >= 0 and idx != combo.currentIndex():
            combo.setCurrentIndex(idx)

    def _open_quick_start_dialog(self) -> None:
        if bool(getattr(self, "_quick_start_dialog_open", False)):
            return
        self._quick_start_dialog_open = True
        dialog = QDialog(self)
        dialog.setWindowTitle("Quick Start")
        dialog.resize(1120, 860)
        outer = QVBoxLayout(dialog)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        hero = QFrame(dialog)
        hero.setStyleSheet(
            "background: #F8FAFC; border: 1px solid #D7DEE8; border-radius: 16px;"
        )
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(16, 14, 16, 14)
        hero_layout.setSpacing(6)
        hero_title = QLabel("Quick Start")
        hero_title.setStyleSheet("font-size: 18px; font-weight: 700; color: #0F172A;")
        hero_text = QLabel(
            "Quick reference for the current HOI workflow: create an event, confirm onset, choose verb and noun, optionally choose instrument/anomaly, complete required boxes, and save structured annotations."
        )
        hero_text.setWordWrap(True)
        hero_text.setObjectName("statusSubtle")
        hero_hint = QLabel("Open this guide any time from the top-right ? button, the ... menu, or F1.")
        hero_hint.setWordWrap(True)
        hero_hint.setStyleSheet("color: #1D4ED8;")
        btn_show_banner = QPushButton("Show Banner", hero)
        btn_show_banner.setAutoDefault(False)
        btn_show_banner.clicked.connect(self._show_onboarding_banner)
        hero_layout.addWidget(hero_title)
        hero_layout.addWidget(hero_text)
        hero_layout.addWidget(hero_hint)
        hero_layout.addWidget(btn_show_banner, 0, Qt.AlignLeft)

        actions_row = QHBoxLayout()
        actions_row.setContentsMargins(0, 6, 0, 0)
        actions_row.setSpacing(8)

        def _after_close(callback):
            def _run():
                dialog.accept()
                QTimer.singleShot(0, callback)
            return _run

        def _make_action_button(text: str, primary: bool = False) -> QPushButton:
            btn = QPushButton(text, dialog)
            btn.setAutoDefault(False)
            btn.setCursor(Qt.PointingHandCursor)
            if primary:
                btn.setStyleSheet(
                    "QPushButton { background: #2563EB; color: white; border: 1px solid #2563EB; border-radius: 10px; padding: 8px 14px; font-weight: 600; }"
                    "QPushButton:hover { background: #1D4ED8; border-color: #1D4ED8; }"
                )
            else:
                btn.setStyleSheet(
                    "QPushButton { background: white; color: #0F172A; border: 1px solid #D7DEE8; border-radius: 10px; padding: 8px 14px; font-weight: 600; }"
                    "QPushButton:hover { background: #F8FAFC; border-color: #BFDBFE; }"
                )
            return btn

        btn_start_manual = _make_action_button("Start Manual", primary=True)
        btn_start_full = _make_action_button("Start Full Assist")
        btn_load_video = _make_action_button("Load Video...")

        btn_start_manual.clicked.connect(
            _after_close(lambda: self._set_experiment_mode_key("manual"))
        )
        btn_start_full.clicked.connect(
            _after_close(lambda: self._set_experiment_mode_key("full_assist"))
        )
        btn_load_video.clicked.connect(_after_close(self._load_video))

        actions_row.addWidget(btn_start_manual)
        actions_row.addWidget(btn_start_full)
        actions_row.addStretch(1)
        actions_row.addWidget(btn_load_video)
        hero_layout.addLayout(actions_row)
        outer.addWidget(hero)

        tabs = QTabWidget(dialog)
        outer.addWidget(tabs, 1)

        # Overview tab
        overview_tab = QWidget(dialog)
        overview_layout = QVBoxLayout(overview_tab)
        overview_layout.setContentsMargins(8, 8, 8, 8)
        overview_layout.setSpacing(10)
        overview_layout.addWidget(
            self._make_quick_start_image(
                self._quick_start_asset_path("hoi_quick_start_overview.png")
            )
        )
        overview_cards = self._make_onboarding_grid(
            [
                {
                    "title": "1. Load",
                    "body_lines": [
                        "Open the ... menu and load a video.",
                        (
                            "Then import the Noun List, Verb List, and optionally a Verb-Noun Ontology CSV."
                            if getattr(self, "_noun_only_mode", False)
                            else "Then import the object list, verb list, and ontology."
                        ),
                    ],
                    "accent": "#2563EB",
                    "tone": "blue",
                },
                {
                    "title": "2. Create",
                    "body_lines": [
                        "Drag on an actor row in the timeline to create a new HOI event.",
                        "The event stores start, onset, and end for the selected hand.",
                    ],
                    "accent": "#2563EB",
                    "tone": "neutral",
                },
                {
                    "title": "3. Confirm onset",
                    "body_lines": [
                        "After drawing start/end, Quick Edit prioritizes Confirm Onset.",
                        "Move the onset marker if needed, then press Confirm Onset.",
                    ],
                    "accent": "#2563EB",
                    "tone": "neutral",
                },
                {
                    "title": "4. Fill labels",
                    "body_lines": [
                        "Choose verb and noun. Choose Instrument only if the event uses a tool.",
                        "Review event attributes explicitly. Select Normal after checking, or record the observed attributes.",
                    ],
                    "accent": "#16A34A",
                    "tone": "green",
                },
            ],
            columns=2,
        )
        overview_layout.addWidget(overview_cards, 1)
        tabs.addTab(overview_tab, "Overview")

        # Boxes and verbs tab
        boxes_tab = QWidget(dialog)
        boxes_layout = QVBoxLayout(boxes_tab)
        boxes_layout.setContentsMargins(8, 8, 8, 8)
        boxes_layout.setSpacing(10)
        boxes_layout.addWidget(
            self._make_quick_start_image(
                self._quick_start_asset_path("hoi_quick_start_boxes_verbs.png")
            )
        )
        boxes_cards = self._make_onboarding_grid(
            [
                {
                    "title": "Verb and noun",
                    "body_lines": [
                        "Import verbs from ... -> Import -> Verb List...",
                        "Select a noun from the object list. Choosing a noun in Quick Edit confirms it immediately.",
                    ],
                    "accent": "#2563EB",
                    "tone": "blue",
                },
                {
                    "title": "Instrument and anomaly",
                    "body_lines": [
                        "Instrument is optional and exports empty when no tool is selected.",
                        "Anomaly exports normal unless you select a defined anomaly type.",
                    ],
                    "accent": "#16A34A",
                    "tone": "neutral",
                },
                {
                    "title": "Required boxes",
                    "body_lines": [
                        "Each event needs hand boxes on start/onset/end.",
                        "If noun or instrument is selected, those boxes are also required on start/onset/end.",
                    ],
                    "accent": "#EA580C",
                    "tone": "amber",
                },
                {
                    "title": "Draw or detect",
                    "body_lines": [
                        "Use Find Boxes to run MediaPipe/YOLO on event keyframes.",
                        "Use Ctrl + left-drag to manually add hand, noun, or instrument boxes.",
                    ],
                    "accent": "#16A34A",
                    "tone": "green",
                },
            ],
            columns=2,
        )
        boxes_layout.addWidget(boxes_cards, 1)
        tabs.addTab(boxes_tab, "Labels and Boxes")

        # Modes and review tab
        review_tab = QWidget(dialog)
        review_layout = QVBoxLayout(review_tab)
        review_layout.setContentsMargins(8, 8, 8, 8)
        review_layout.setSpacing(10)
        review_cards = self._make_onboarding_grid(
            [
                {
                    "title": "Manual",
                    "body_lines": [
                        "Everything by hand.",
                        "Best for strict baselines and first-time familiarization.",
                    ],
                    "accent": "#334155",
                    "tone": "neutral",
                },
                {
                    "title": "Full Assist",
                    "body_lines": [
                        "Shows action suggestions and box assistance.",
                        "Use this for the guided workflow during the study.",
                    ],
                    "accent": "#16A34A",
                    "tone": "green",
                },
                {
                    "title": "Review meanings",
                    "body_lines": [
                        "Suggested fields still need review before completion.",
                        "Required Boxes means labels are set but keyframe bbox evidence is missing.",
                    ],
                    "accent": "#2563EB",
                    "tone": "blue",
                },
                {
                    "title": "Save",
                    "body_lines": [
                        "Use ... -> Save / Export -> Save HOI Annotations...",
                        "The save check warns about missing timing, verb/noun, hand boxes, noun boxes, or instrument boxes.",
                    ],
                    "accent": "#7C3AED",
                    "tone": "neutral",
                },
                {
                    "title": "Reopen help",
                    "body_lines": [
                        "Press F1 any time.",
                        "The top-right ? button remains available during annotation and study-style workflows.",
                    ],
                    "accent": "#2563EB",
                    "tone": "neutral",
                },
            ],
            columns=2,
        )
        review_layout.addWidget(review_cards, 1)
        tabs.addTab(review_tab, "Modes and Review")

        chk_startup = QCheckBox("Show this guide on startup", dialog)
        chk_startup.setChecked(bool(self._show_quick_start_on_startup))
        chk_startup.setStyleSheet("color: #0F172A;")
        outer.addWidget(chk_startup)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=dialog)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        buttons.button(QDialogButtonBox.Close).clicked.connect(dialog.accept)
        outer.addWidget(buttons)

        try:
            dialog.exec_()
        finally:
            self._show_quick_start_on_startup = bool(chk_startup.isChecked())
            self._persist_ui_preferences()
            self._quick_start_dialog_open = False

    def _set_verb_library_admin_mode(self, on: bool) -> None:
        on = bool(on)
        if hasattr(self, "label_panel") and self.label_panel:
            self.label_panel.set_admin_visible(on)
        act = getattr(self, "act_toggle_verb_library_admin", None)
        if act is not None and act.isChecked() != on:
            act.blockSignals(True)
            act.setChecked(on)
            act.blockSignals(False)

    def _set_status_chip(self, widget: Optional[QLabel], text: str, tone: str = "neutral") -> None:
        if widget is None:
            return
        palette = {
            "neutral": ("#EEF2FF", "#3730A3"),
            "ok": ("#ECFDF3", "#027A48"),
            "warn": ("#FFFAEB", "#B54708"),
            "danger": ("#FEF3F2", "#B42318"),
        }
        bg, fg = palette.get(tone, palette["neutral"])
        widget.setText(text)
        widget.setStyleSheet(
            f"background: {bg}; color: {fg}; border: 1px solid {bg}; border-radius: 999px; padding: 3px 10px; font-weight: 600;"
        )

    def _set_status_card_tone(self, widget: Optional[QFrame], tone: str = "neutral") -> None:
        if widget is None:
            return
        palette = {
            "neutral": ("#F8FAFC", "#E4E7EC"),
            "active": ("#EFF6FF", "#93C5FD"),
            "ok": ("#F0FDF4", "#86EFAC"),
            "warn": ("#FFFBEB", "#FCD34D"),
            "danger": ("#FEF2F2", "#FCA5A5"),
        }
        bg, border = palette.get(tone, palette["neutral"])
        widget.setStyleSheet(
            f"background: {bg}; border: 1px solid {border}; border-radius: 10px;"
        )

    def _focus_inspector_tab(self, name: str) -> None:
        tabs = getattr(self, "inspector_tabs", None)
        if tabs is None:
            return
        key = str(name or "").strip().lower()
        mapping = {"event": 0, "objects": 1, "review": 2}
        idx = mapping.get(key)
        if idx is not None and 0 <= idx < tabs.count():
            tabs.setCurrentIndex(idx)

    def _refresh_widget_style(self, widget: Optional[QWidget]) -> None:
        if widget is None:
            return
        try:
            style = widget.style()
            style.unpolish(widget)
            style.polish(widget)
            widget.update()
        except Exception:
            pass

    def _set_inline_query_hint(self, text: str, role: str = "neutral") -> None:
        label = getattr(self, "lbl_inline_query_hint", None)
        if label is None:
            return
        normalized_role = str(role or "neutral").strip().lower()
        if normalized_role not in {"neutral", "action", "review", "complete"}:
            normalized_role = "neutral"
        label.setText(text)
        if label.property("hintRole") != normalized_role:
            label.setProperty("hintRole", normalized_role)
            self._refresh_widget_style(label)

    def _set_advanced_inspector_visible(self, visible: bool) -> None:
        self._advanced_inspector_visible = bool(visible)
        tabs = getattr(self, "inspector_tabs", None)
        splitter = getattr(self, "hoi_bottom_split", None)
        if tabs is not None:
            tabs.setVisible(self._advanced_inspector_visible)
        if splitter is not None:
            try:
                splitter.setSizes(
                    [320, 940] if self._advanced_inspector_visible else [0, 1260]
                )
            except Exception:
                pass
        btn = getattr(self, "btn_inline_toggle_advanced", None)
        if btn is not None:
            btn.setText("Hide Advanced" if self._advanced_inspector_visible else "Advanced")
            btn.setToolTip(
                "Hide the full left-side inspector."
                if self._advanced_inspector_visible
                else "Show the full left-side inspector."
            )

    def _toggle_advanced_inspector(self) -> None:
        self._set_advanced_inspector_visible(
            not bool(getattr(self, "_advanced_inspector_visible", True))
        )

    def _selected_hand_data(self) -> Optional[dict]:
        hand_key = str(getattr(self, "selected_hand_label", "") or "").strip()
        draft = getattr(self, "event_draft", None)
        if not hand_key or not isinstance(draft, dict):
            return None
        hand_data = draft.get(hand_key)
        if not isinstance(hand_data, dict):
            return None
        return hand_data

    def _set_combo_to_data(
        self,
        combo: Optional[QComboBox],
        value: Any,
        *,
        fallback_text: str = "",
    ) -> None:
        if combo is None:
            return
        idx = combo.findData(value)
        if idx < 0:
            normalized_int = None
            normalized_text = ""
            try:
                normalized_int = int(value) if value is not None else None
            except Exception:
                normalized_int = None
            try:
                normalized_text = str(value or "").strip()
            except Exception:
                normalized_text = ""
            for item_idx in range(combo.count()):
                item_data = combo.itemData(item_idx)
                if item_data == value:
                    idx = item_idx
                    break
                if normalized_int is not None:
                    try:
                        if int(item_data) == normalized_int:
                            idx = item_idx
                            break
                    except Exception:
                        pass
                if normalized_text:
                    try:
                        if str(item_data or "").strip() == normalized_text:
                            idx = item_idx
                            break
                    except Exception:
                        pass
        if idx >= 0:
            combo.setCurrentIndex(idx)
            return
        if combo.isEditable():
            combo.setCurrentText(str(fallback_text or ""))
        elif combo.count() > 0:
            combo.setCurrentIndex(0)

    def _inline_query_for_selected(self) -> dict:
        query = dict(getattr(self, "_next_best_query", {}) or {})
        if not query:
            return {}
        try:
            event_ok = int(query.get("event_id")) == int(self.selected_event_id)
        except Exception:
            event_ok = False
        hand_ok = str(query.get("hand") or "").strip() == str(
            self.selected_hand_label or ""
        ).strip()
        if not (event_ok and hand_ok):
            return {}
        return query if self._query_is_actionable(query) else {}

    def _query_event_hand_data(self, query: Optional[dict]) -> Optional[dict]:
        if not isinstance(query, dict):
            return None
        try:
            event_id = int(query.get("event_id"))
        except Exception:
            return None
        hand_key = str(query.get("hand") or "").strip()
        if not hand_key:
            return None
        event = self._find_event_by_id(event_id)
        if not isinstance(event, dict):
            return None
        hand_data = (event.get("hoi_data", {}) or {}).get(hand_key)
        return hand_data if isinstance(hand_data, dict) else None

    def _query_has_reviewable_current_value(
        self,
        hand_data: Optional[dict],
        field_name: str,
    ) -> bool:
        field_name = self._canonical_query_field_name(field_name)
        if not isinstance(hand_data, dict) or not field_name:
            return False
        if field_name == "verb":
            return bool(str(hand_data.get("verb") or "").strip())
        if field_name == "noun_object_id":
            if self._hand_noun_object_id(hand_data) is not None:
                return True
            suggestion = get_field_suggestion(hand_data, "noun_object_id")
            suggestion_meta = dict((suggestion or {}).get("meta") or {})
            if bool(suggestion_meta.get("explicit_empty")):
                return True
            return self._hand_has_explicit_no_noun_lock(hand_data)
        if field_name in (
            "interaction_start",
            "functional_contact_onset",
            "interaction_end",
        ):
            return hand_data.get(field_name) is not None
        return hand_data.get(field_name) is not None

    def _query_is_actionable(self, query: Optional[dict]) -> bool:
        if not isinstance(query, dict) or not query:
            return False
        hand_data = self._query_event_hand_data(query)
        if not isinstance(hand_data, dict):
            return False
        apply_mode = str(query.get("apply_mode") or "").strip()
        field_name = self._query_target_field_name(query)
        if apply_mode == "confirm_current":
            state = get_field_state(hand_data, field_name)
            if str(state.get("status") or "").strip().lower() == "confirmed":
                return False
            return self._query_has_reviewable_current_value(hand_data, field_name)
        if apply_mode == "apply_suggestion":
            state = get_field_state(hand_data, field_name)
            if str(state.get("status") or "").strip().lower() == "confirmed":
                return False
            return query.get("suggested_value") is not None
        if apply_mode == "apply_completion_bundle":
            for item in list(query.get("completion_fields") or []):
                if not isinstance(item, dict):
                    continue
                bundle_field = self._canonical_query_field_name(item.get("field_name"))
                if not bundle_field:
                    continue
                state = get_field_state(hand_data, bundle_field)
                if str(state.get("status") or "").strip().lower() == "confirmed":
                    continue
                if bool(item.get("safe_to_apply")):
                    return True
            return False
        return True

    def _query_confirm_button_label(self, query: Optional[dict]) -> str:
        query_type = str((query or {}).get("query_type") or "").strip().lower()
        if query_type == "confirm_no_noun_needed":
            return "Confirm No Noun"
        field_name = self._query_target_field_name(query)
        if field_name == "verb":
            return "Confirm Verb"
        if field_name == "noun_object_id":
            return "Confirm Noun"
        if field_name == "functional_contact_onset":
            return "Confirm Onset"
        if field_name == "interaction_start":
            return "Confirm Start"
        if field_name == "interaction_end":
            return "Confirm End"
        return "Confirm Current"

    def _inline_confirmable_field_value(
        self,
        field_name: str,
        hand_data: Optional[dict] = None,
    ) -> Any:
        field_name = self._canonical_query_field_name(field_name)
        hand_data = hand_data if isinstance(hand_data, dict) else self._selected_hand_data()
        if field_name == "verb":
            current = str((hand_data or {}).get("verb") or "").strip()
            if current:
                return current
            combo = getattr(self, "combo_inline_verb", None)
            if isinstance(combo, QComboBox) and combo.isEnabled():
                value = combo.currentData()
                text = str(value or combo.currentText() or "").strip()
                return text or None
            return None
        if field_name == "noun_object_id":
            current = self._hand_noun_object_id(hand_data)
            if current is not None:
                return current
            combo = getattr(self, "combo_inline_noun", None)
            if isinstance(combo, QComboBox) and combo.isEnabled():
                try:
                    current_index = int(combo.currentIndex())
                except Exception:
                    current_index = -1
                value = combo.currentData()
                if current_index > 0 and value is not None:
                    return value
            return None
        if not isinstance(hand_data, dict):
            return None
        return hand_data.get(field_name)

    def _inline_has_confirmable_current_value(
        self,
        field_name: str,
        hand_data: Optional[dict] = None,
    ) -> bool:
        field_name = self._canonical_query_field_name(field_name)
        hand_data = hand_data if isinstance(hand_data, dict) else self._selected_hand_data()
        current_value = self._inline_confirmable_field_value(field_name, hand_data)
        if field_name == "verb":
            return bool(str(current_value or "").strip())
        if field_name == "noun_object_id":
            if current_value is not None:
                return True
            return self._hand_has_explicit_no_noun_lock(hand_data)
        if field_name in (
            "interaction_start",
            "functional_contact_onset",
            "interaction_end",
        ):
            return current_value is not None
        return current_value is not None

    def _inline_primary_field_name(
        self,
        hand_data: Optional[dict] = None,
        *,
        allow_query_fallback: bool = True,
    ) -> str:
        hand_data = hand_data if isinstance(hand_data, dict) else self._selected_hand_data()
        if not isinstance(hand_data, dict):
            return ""
        onset_value = hand_data.get("functional_contact_onset")
        onset_state = get_field_state(hand_data, "functional_contact_onset")
        if (
            onset_value is not None
            and str(onset_state.get("status") or "").strip().lower() != "confirmed"
            and not self._manual_mode_enabled()
        ):
            return "functional_contact_onset"
        verb = str(hand_data.get("verb") or "").strip()
        verb_state = get_field_state(hand_data, "verb")
        if not verb or str(verb_state.get("status") or "").strip().lower() != "confirmed":
            return "verb"
        noun_required = self._noun_required_for_verb(verb)
        noun_value = self._hand_noun_object_id(hand_data)
        noun_state = get_field_state(hand_data, "noun_object_id")
        if noun_required and (
            noun_value is None
            or str(noun_state.get("status") or "").strip().lower() != "confirmed"
        ):
            return "noun_object_id"
        completion_state = self._hand_completion_state(
            hand_data,
            hand_key=getattr(self, "selected_hand_label", None),
        )
        if completion_state.get("complete"):
            return ""
        bbox_missing = list(completion_state.get("bbox_missing") or [])
        if bbox_missing:
            return "bbox_evidence"
        if not allow_query_fallback:
            return ""
        query = self._inline_query_for_selected()
        return self._query_target_field_name(query) or ""

    def _refresh_inline_primary_focus(self, *, delay_ms: int = 0) -> None:
        if self._manual_mode_enabled():
            return
        if self.selected_event_id is None or not str(self.selected_hand_label or "").strip():
            return

        def _apply() -> None:
            if self.selected_event_id is None or not str(self.selected_hand_label or "").strip():
                return
            field_name = self._inline_primary_field_name(self._selected_hand_data())
            if field_name:
                self._focus_inline_editor_field(field_name)
            else:
                self._clear_spotlight_widget()

        try:
            delay = max(0, int(delay_ms))
        except Exception:
            delay = 0
        if delay <= 0:
            _apply()
        else:
            QTimer.singleShot(delay, _apply)

    def _refresh_selected_event_runtime_views(
        self,
        event_id: Any,
        *,
        refresh_boxes: bool = True,
        refresh_focus: bool = False,
    ) -> None:
        try:
            selected_matches = int(self.selected_event_id) == int(event_id)
        except Exception:
            selected_matches = False
        if not selected_matches:
            return
        self._update_next_best_query_panel()
        self._update_status_label()
        if refresh_boxes:
            current_frame = int(getattr(self.player, "current_frame", 0) or 0)
            self._refresh_boxes_for_frame(current_frame, skip_events=True, lightweight=True)
        if refresh_focus:
            self._refresh_inline_primary_focus(delay_ms=30)

    def _apply_noun_choice(self, object_id: Any, source: str = "inline_editor") -> None:
        if not self.selected_hand_label:
            return
        try:
            normalized_object_id = int(object_id) if object_id is not None else None
        except Exception:
            normalized_object_id = object_id
        hand_data = None
        if isinstance(getattr(self, "event_draft", None), dict):
            hand_data = self.event_draft.get(self.selected_hand_label)
        current_noun = self._hand_noun_object_id(hand_data)
        try:
            normalized_current_noun = int(current_noun) if current_noun is not None else None
        except Exception:
            normalized_current_noun = current_noun
        noun_state = get_field_state(hand_data, "noun_object_id") if isinstance(hand_data, dict) else {}
        noun_confirmed = str(noun_state.get("status") or "").strip().lower() == "confirmed"
        noun_source_fields = self._flatten_noun_source_decision_fields(
            self._current_noun_source_decision(hand_data)
        )
        self._set_pending_field_source("noun_object_id", source)
        self._set_combo_to_data(getattr(self, "combo_target", None), normalized_object_id)
        inline_combo = getattr(self, "combo_inline_noun", None)
        if isinstance(inline_combo, QComboBox):
            try:
                inline_combo.blockSignals(True)
                self._set_combo_to_data(inline_combo, normalized_object_id)
            finally:
                inline_combo.blockSignals(False)
        if normalized_current_noun == normalized_object_id and not noun_confirmed:
            self._confirm_current_hand_field("noun_object_id", source=source)
        elif getattr(self, "combo_target", None) is not None:
            try:
                current_combo_noun = self.combo_target.currentData()
            except Exception:
                current_combo_noun = None
            try:
                normalized_combo_noun = int(current_combo_noun) if current_combo_noun is not None else None
            except Exception:
                normalized_combo_noun = current_combo_noun
            if normalized_combo_noun == normalized_object_id and normalized_current_noun != normalized_object_id:
                self._on_hoi_meta_changed()
        self._log(
            "hoi_apply_noun_choice",
            source=source,
            noun_object_id=normalized_object_id,
            event_id=self.selected_event_id,
            hand=self.selected_hand_label,
            **noun_source_fields,
        )

    def _filtered_object_candidates_for_hand(
        self,
        hand_key: str,
        hand_data: Optional[dict],
    ) -> List[dict]:
        candidates = self._collect_event_object_candidates(hand_key, hand_data)
        verb_value = str((hand_data or {}).get("verb") or "").strip()
        if bool(self._noun_only_mode) and verb_value:
            allowed_ids = self._allowed_noun_ids_for_verb(verb_value)
            candidates = filter_allowed_object_candidates(candidates, allowed_ids)
        preferred_noun_id = None
        current_noun = self._hand_noun_object_id(hand_data)
        if current_noun is not None:
            try:
                preferred_noun_id = int(current_noun)
            except Exception:
                preferred_noun_id = None
        if preferred_noun_id is None:
            preferred_noun_id = self._strong_semantic_noun_suggestion_id(hand_data)
        if preferred_noun_id is not None:
            matching = []
            remaining = []
            for item in list(candidates or []):
                try:
                    object_id = int(item.get("object_id"))
                except Exception:
                    object_id = None
                if object_id == int(preferred_noun_id):
                    matching.append(dict(item))
                else:
                    remaining.append(dict(item))
            if matching:
                candidates = list(matching) + list(remaining)
        return [dict(item) for item in list(candidates or []) if isinstance(item, dict)]

    def _best_grounding_candidate_for_noun(
        self,
        hand_key: str,
        hand_data: Optional[dict],
        noun_object_id: Any,
    ) -> Dict[str, Any]:
        try:
            noun_object_id = int(noun_object_id)
        except Exception:
            return {}
        for item in self._filtered_object_candidates_for_hand(hand_key, hand_data):
            try:
                object_id = int(item.get("object_id"))
            except Exception:
                continue
            if object_id == noun_object_id:
                return dict(item)
        return {}

    def _selected_noun_grounding_status(
        self,
        hand_key: str,
        hand_data: Optional[dict],
    ) -> Dict[str, Any]:
        hand_key = str(hand_key or "").strip()
        if not hand_key or not isinstance(hand_data, dict):
            return {}
        object_id = self._hand_noun_object_id(hand_data)
        try:
            object_id = int(object_id) if object_id is not None else None
        except Exception:
            object_id = None
        if object_id is None:
            return {}
        current_frame = int(getattr(self.player, "current_frame", 0) or 0)
        onset_frame = hand_data.get("functional_contact_onset")
        try:
            onset_frame = int(onset_frame) if onset_frame is not None else None
        except Exception:
            onset_frame = None
        current_box = self._best_matching_object_box_on_frame(current_frame, object_id)
        onset_box = {}
        if onset_frame is not None:
            onset_box = self._best_matching_object_box_on_frame(
                onset_frame,
                object_id,
                anchor_box=current_box or None,
            )
        candidate = self._best_grounding_candidate_for_noun(hand_key, hand_data, object_id)
        candidate_bbox = dict(candidate.get("best_bbox") or {}) if isinstance(candidate, dict) else {}
        candidate_frame = candidate.get("best_frame") if isinstance(candidate, dict) else None
        try:
            candidate_frame = int(candidate_frame) if candidate_frame is not None else None
        except Exception:
            candidate_frame = None
        if candidate_frame is None and candidate_bbox:
            try:
                candidate_frame = (
                    int(candidate_bbox.get("frame"))
                    if candidate_bbox.get("frame") is not None
                    else None
                )
            except Exception:
                candidate_frame = None
        return {
            "object_id": int(object_id),
            "current_frame": int(current_frame),
            "current_box": dict(current_box or {}),
            "onset_frame": onset_frame,
            "onset_box": dict(onset_box or {}),
            "candidate": dict(candidate or {}),
            "candidate_frame": candidate_frame,
            "candidate_bbox": dict(candidate_bbox or {}),
            "has_current_box": bool(current_box),
            "has_onset_box": bool(onset_box),
        }

    def _default_object_id_for_label(self, label: Any) -> Optional[int]:
        norm_label = self._norm_category(label)
        if not norm_label:
            return None
        for obj_name, uid in list(getattr(self, "global_object_map", {}).items() or []):
            if self._norm_category(obj_name) != norm_label:
                continue
            try:
                return int(uid)
            except Exception:
                return None
        category_to_default_id = self._build_category_to_default_id()
        try:
            resolved = category_to_default_id.get(norm_label)
            return int(resolved) if resolved is not None else None
        except Exception:
            return None

    def _sync_selected_hand_noun_after_box_relabel(
        self,
        previous_object_id: Any,
        updated_box: Optional[dict],
        *,
        source: str,
    ) -> None:
        if self.selected_event_id is None or not str(self.selected_hand_label or "").strip():
            return
        if not isinstance(updated_box, dict) or self._is_hand_label(updated_box.get("label")):
            return
        try:
            previous_object_id = int(previous_object_id)
        except Exception:
            return
        try:
            next_object_id = int(updated_box.get("id"))
        except Exception:
            return
        if next_object_id == previous_object_id:
            return
        hand_data = self.event_draft.get(self.selected_hand_label)
        if not isinstance(hand_data, dict):
            return
        current_noun = self._hand_noun_object_id(hand_data)
        try:
            current_noun = int(current_noun) if current_noun is not None else None
        except Exception:
            current_noun = None
        if current_noun != previous_object_id:
            return
        hand_data["noun_object_id"] = next_object_id
        hand_data["target_object_id"] = next_object_id
        self._set_hand_field_state(
            hand_data,
            "noun_object_id",
            source=source,
            status="confirmed",
        )
        self._apply_draft_to_selected_event()
        self._mark_validation_change()
        self._bump_query_state_revision()
        self._refresh_events()
        self._update_hoi_titles()
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.refresh()
        self._update_status_label()

    def _select_object_item_in_list(
        self,
        object_id: Any,
        best_bbox: Optional[dict] = None,
    ) -> None:
        try:
            object_id = int(object_id)
        except Exception:
            return
        best_match = None
        fallback_match = None
        for idx in range(self.list_objects.count()):
            item = self.list_objects.item(idx)
            box = item.data(Qt.UserRole)
            if not isinstance(box, dict):
                continue
            try:
                box_id = int(box.get("id"))
            except Exception:
                continue
            if box_id != object_id:
                continue
            if fallback_match is None:
                fallback_match = item
            if isinstance(best_bbox, dict):
                same_bbox = all(
                    abs(float(box.get(key, 0.0) or 0.0) - float(best_bbox.get(key, 0.0) or 0.0))
                    <= 2.0
                    for key in ("x1", "y1", "x2", "y2")
                )
                if same_bbox:
                    best_match = item
                    break
        chosen = best_match or fallback_match
        if chosen is None:
            return
        self.list_objects.blockSignals(True)
        self.list_objects.setCurrentItem(chosen)
        self.list_objects.blockSignals(False)
        try:
            self.list_objects.scrollToItem(chosen)
        except Exception:
            pass

    def _find_clicked_object_box(
        self,
        frame_idx: int,
        image_x: Optional[float],
        image_y: Optional[float],
    ) -> Optional[dict]:
        if image_x is None or image_y is None:
            return None
        try:
            px = float(image_x)
            py = float(image_y)
        except Exception:
            return None
        hits = []
        for box in list(self._frame_boxes_with_cached_hands(int(frame_idx)) or []):
            if not isinstance(box, dict):
                continue
            if self._is_hand_label(box.get("label")):
                continue
            try:
                x1 = float(box.get("x1", 0.0) or 0.0)
                y1 = float(box.get("y1", 0.0) or 0.0)
                x2 = float(box.get("x2", 0.0) or 0.0)
                y2 = float(box.get("y2", 0.0) or 0.0)
            except Exception:
                continue
            if x1 <= px <= x2 and y1 <= py <= y2:
                area = max(1.0, max(0.0, x2 - x1) * max(0.0, y2 - y1))
                hits.append((area, dict(box)))
        if not hits:
            return None
        hits.sort(key=lambda item: item[0])
        return dict(hits[0][1])

    def _select_box_item_in_list(
        self,
        *,
        object_id: Any = None,
        hand_key: str = "",
        best_bbox: Optional[dict] = None,
    ) -> bool:
        hand_key = str(hand_key or "").strip()
        target_id = None
        if object_id is not None:
            try:
                target_id = int(object_id)
            except Exception:
                target_id = None
        chosen = None
        for idx in range(self.list_objects.count()):
            item = self.list_objects.item(idx)
            box = item.data(Qt.UserRole)
            if not isinstance(box, dict):
                continue
            if hand_key:
                if self._normalize_hand_label(box.get("label")) == hand_key:
                    if isinstance(best_bbox, dict):
                        same_bbox = all(
                            abs(float(box.get(key, 0.0) or 0.0) - float(best_bbox.get(key, 0.0) or 0.0))
                            <= 2.0
                            for key in ("x1", "y1", "x2", "y2")
                        )
                        if same_bbox:
                            chosen = item
                            break
                    if chosen is None:
                        chosen = item
                continue
            if target_id is None:
                continue
            try:
                box_id = int(box.get("id"))
            except Exception:
                continue
            if box_id != target_id:
                continue
            if isinstance(best_bbox, dict):
                same_bbox = all(
                    abs(float(box.get(key, 0.0) or 0.0) - float(best_bbox.get(key, 0.0) or 0.0))
                    <= 2.0
                    for key in ("x1", "y1", "x2", "y2")
                )
                if same_bbox:
                    chosen = item
                    break
            if chosen is None:
                chosen = item
        if chosen is None:
            return False
        self.list_objects.blockSignals(True)
        self.list_objects.setCurrentItem(chosen)
        self.list_objects.blockSignals(False)
        try:
            self.list_objects.scrollToItem(chosen)
        except Exception:
            pass
        return True

    def _object_boxes_on_frame(
        self,
        frame: Any,
        object_id: Any,
    ) -> List[dict]:
        try:
            frame = int(frame)
            object_id = int(object_id)
        except Exception:
            return []
        out: List[dict] = []
        for box in list(self.bboxes.get(frame, []) or []):
            if not isinstance(box, dict):
                continue
            if self._is_hand_label(box.get("label")):
                continue
            try:
                box_id = int(box.get("id"))
            except Exception:
                continue
            if box_id != object_id:
                continue
            row = dict(box)
            row["frame"] = frame
            out.append(row)
        return out

    def _box_anchor_similarity(
        self,
        box: Optional[dict],
        anchor_box: Optional[dict],
    ) -> float:
        if not isinstance(box, dict) or not isinstance(anchor_box, dict):
            return 0.0
        iou = float(self._box_iou(box, anchor_box))
        gap = float(self._box_edge_gap(box, anchor_box))
        anchor_diag = float(max(1.0, self._box_diag(anchor_box)))
        gap_score = self._bounded01(1.0 - gap / float(max(1.0, 0.85 * anchor_diag)), 0.0)
        return float(0.65 * iou + 0.35 * gap_score)

    def _best_matching_object_box_on_frame(
        self,
        frame: Any,
        object_id: Any,
        *,
        anchor_box: Optional[dict] = None,
    ) -> Dict[str, Any]:
        matches = self._object_boxes_on_frame(frame, object_id)
        if not matches:
            return {}
        if not isinstance(anchor_box, dict):
            return dict(matches[0])
        ranked = sorted(
            list(matches or []),
            key=lambda row: (
                float(self._box_anchor_similarity(row, anchor_box)),
                float(self._box_confidence_score(row)),
            ),
            reverse=True,
        )
        return dict((ranked or [None])[0] or {})

    @staticmethod
    def _interpolate_box_geometry(
        box_a: Optional[dict],
        frame_a: int,
        box_b: Optional[dict],
        frame_b: int,
        target_frame: int,
    ) -> Dict[str, float]:
        if not isinstance(box_a, dict) and not isinstance(box_b, dict):
            return {}
        if not isinstance(box_a, dict):
            box_a = dict(box_b or {})
            frame_a = int(frame_b)
        if not isinstance(box_b, dict):
            box_b = dict(box_a or {})
            frame_b = int(frame_a)
        try:
            frame_a = int(frame_a)
            frame_b = int(frame_b)
            target_frame = int(target_frame)
        except Exception:
            return {}
        if frame_a == frame_b:
            alpha = 0.0
        else:
            alpha = float(target_frame - frame_a) / float(frame_b - frame_a)
        alpha = max(0.0, min(1.0, alpha))
        out: Dict[str, float] = {}
        for key in ("x1", "y1", "x2", "y2"):
            try:
                va = float((box_a or {}).get(key, 0.0) or 0.0)
                vb = float((box_b or {}).get(key, 0.0) or 0.0)
            except Exception:
                return {}
            out[key] = float((1.0 - alpha) * va + alpha * vb)
        return out

    def _project_candidate_box_to_onset_frame(
        self,
        hand_key: str,
        hand_data: Optional[dict],
        object_id: Any,
        *,
        preferred_bbox: Optional[dict] = None,
        preferred_frame: Any = None,
        role: str = "object",
        event_id: Any = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        if not hand_key or not isinstance(hand_data, dict):
            return {}
        try:
            object_id = int(object_id)
        except Exception:
            return {}
        onset_frame = hand_data.get("functional_contact_onset")
        try:
            onset_frame = int(onset_frame) if onset_frame is not None else None
        except Exception:
            onset_frame = None
        if onset_frame is None:
            return {}

        onset_match = self._best_matching_object_box_on_frame(
            onset_frame,
            object_id,
            anchor_box=preferred_bbox,
        )
        if onset_match:
            return dict(onset_match)

        candidate_frames = list(self._event_object_candidate_frames(hand_key, hand_data) or [])
        try:
            preferred_frame = int(preferred_frame) if preferred_frame is not None else None
        except Exception:
            preferred_frame = None
        if preferred_frame is not None and preferred_frame not in candidate_frames:
            candidate_frames.append(int(preferred_frame))
        candidate_frames = sorted({int(frame) for frame in list(candidate_frames or []) if frame is not None})

        matched_by_frame: Dict[int, Dict[str, Any]] = {}
        for frame in list(candidate_frames or []):
            if int(frame) == int(onset_frame):
                continue
            match = self._best_matching_object_box_on_frame(
                frame,
                object_id,
                anchor_box=preferred_bbox,
            )
            if match:
                matched_by_frame[int(frame)] = dict(match)
        if preferred_frame is not None and isinstance(preferred_bbox, dict):
            preferred_entry = dict(preferred_bbox)
            preferred_entry["frame"] = int(preferred_frame)
            matched_by_frame[int(preferred_frame)] = preferred_entry

        before_frames = sorted([frame for frame in matched_by_frame if int(frame) < int(onset_frame)])
        after_frames = sorted([frame for frame in matched_by_frame if int(frame) > int(onset_frame)])
        before_frame = before_frames[-1] if before_frames else None
        after_frame = after_frames[0] if after_frames else None
        reference_box = None
        projection_method = ""
        geometry: Dict[str, float] = {}
        source_frame = None

        if before_frame is not None and after_frame is not None:
            geometry = self._interpolate_box_geometry(
                matched_by_frame.get(int(before_frame)),
                int(before_frame),
                matched_by_frame.get(int(after_frame)),
                int(after_frame),
                int(onset_frame),
            )
            reference_box = dict(matched_by_frame.get(int(before_frame)) or matched_by_frame.get(int(after_frame)) or {})
            projection_method = "linear_interp"
            source_frame = f"{int(before_frame)}->{int(after_frame)}"
        elif before_frame is not None or after_frame is not None:
            nearest_frame = int(before_frame if before_frame is not None else after_frame)
            reference_box = dict(matched_by_frame.get(nearest_frame) or {})
            geometry = {
                key: float(reference_box.get(key, 0.0) or 0.0)
                for key in ("x1", "y1", "x2", "y2")
            }
            projection_method = "nearest_copy"
            source_frame = int(nearest_frame)
        elif isinstance(preferred_bbox, dict) and self._box_diag(preferred_bbox) > 1.0:
            reference_box = dict(preferred_bbox)
            geometry = {
                key: float(reference_box.get(key, 0.0) or 0.0)
                for key in ("x1", "y1", "x2", "y2")
            }
            projection_method = "best_frame_copy"
            source_frame = preferred_frame
        else:
            return {}

        if not geometry:
            return {}
        label_txt = str(
            (reference_box or {}).get("label")
            or self._object_name_for_id(object_id, fallback=f"Object {object_id}")
            or ""
        ).strip()
        if not label_txt:
            label_txt = f"Object {object_id}"
        class_id = (reference_box or {}).get("class_id")
        if class_id is None:
            class_id = self._class_id_for_label(label_txt)
        try:
            target_orig = int(onset_frame) - int(self.start_offset)
        except Exception:
            target_orig = int(onset_frame)
        new_rb = {
            "id": int(object_id),
            "orig_frame": int(target_orig),
            "label": label_txt,
            "source": f"onset_projected_{projection_method}",
            "locked": False,
            "x1": float(geometry.get("x1", 0.0) or 0.0),
            "y1": float(geometry.get("y1", 0.0) or 0.0),
            "x2": float(geometry.get("x2", 0.0) or 0.0),
            "y2": float(geometry.get("y2", 0.0) or 0.0),
        }
        if class_id is not None:
            new_rb["class_id"] = class_id
        projected_box = dict(new_rb)
        projected_box["frame"] = int(onset_frame)
        if dry_run:
            return projected_box
        self.raw_boxes.append(new_rb)
        self._rebuild_bboxes_from_raw()
        self._bump_bbox_revision()
        self._bump_query_state_revision()
        self._log(
            "hoi_project_object_box_to_onset",
            event_id=event_id if event_id is not None else self.selected_event_id,
            hand=hand_key,
            object_role=str(role or "object"),
            object_id=int(object_id),
            noun_object_id=int(object_id) if str(role or "") == "noun" else None,
            instrument_object_id=(
                int(object_id) if str(role or "") == "instrument" else None
            ),
            onset_frame=int(onset_frame),
            source_frame=source_frame,
            projection_method=projection_method,
        )
        return projected_box

    def _ensure_event_onset_grounding(self, event_id: Any, *, dry_run: bool = False) -> int:
        event = self._find_event_by_id(event_id)
        if not isinstance(event, dict):
            return 0
        projected_count = 0
        for actor in list(self.actors_config or []):
            hand_key = str((actor or {}).get("id") or "").strip()
            hand_data = (event.get("hoi_data", {}) or {}).get(hand_key, {}) or {}
            if not isinstance(hand_data, dict):
                continue
            onset_frame = hand_data.get("functional_contact_onset")
            try:
                onset_frame = int(onset_frame) if onset_frame is not None else None
            except Exception:
                onset_frame = None
            if onset_frame is None:
                continue
            role_ids = [
                ("noun", self._hand_noun_object_id(hand_data)),
                ("instrument", self._hand_instrument_object_id(hand_data)),
            ]
            for role, object_id in role_ids:
                try:
                    object_id = int(object_id) if object_id is not None else None
                except Exception:
                    object_id = None
                if object_id is None:
                    continue
                if self._best_matching_object_box_on_frame(onset_frame, object_id):
                    continue
                projected = self._project_candidate_box_to_onset_frame(
                    hand_key,
                    hand_data,
                    object_id,
                    role=role,
                    event_id=event_id,
                    dry_run=bool(dry_run),
                )
                if projected:
                    projected_count += 1
        return projected_count

    @staticmethod
    def _same_box_identity(box_a: Optional[dict], box_b: Optional[dict]) -> bool:
        if not isinstance(box_a, dict) or not isinstance(box_b, dict):
            return False
        id_a = box_a.get("id")
        id_b = box_b.get("id")
        if id_a is not None and id_b is not None and id_a != id_b:
            return False
        frame_a = box_a.get("orig_frame", box_a.get("frame"))
        frame_b = box_b.get("orig_frame", box_b.get("frame"))
        try:
            frame_a = int(frame_a) if frame_a is not None else None
        except Exception:
            frame_a = None
        try:
            frame_b = int(frame_b) if frame_b is not None else None
        except Exception:
            frame_b = None
        if frame_a is not None and frame_b is not None and frame_a != frame_b:
            return False
        for key in ("x1", "y1", "x2", "y2"):
            try:
                va = float(box_a.get(key, 0.0) or 0.0)
                vb = float(box_b.get(key, 0.0) or 0.0)
            except Exception:
                return False
            if abs(va - vb) > 2.0:
                return False
        return True

    def _set_selected_edit_box(
        self,
        box: Optional[dict],
        *,
        refresh: bool = True,
    ) -> None:
        next_box = dict(box) if isinstance(box, dict) else None
        changed = not self._same_box_identity(
            getattr(self, "_selected_edit_box", None),
            next_box,
        )
        self._selected_edit_box = next_box
        if next_box is None:
            try:
                self.list_objects.blockSignals(True)
                self.list_objects.clearSelection()
                self.list_objects.setCurrentItem(None)
            except Exception:
                pass
            finally:
                try:
                    self.list_objects.blockSignals(False)
                except Exception:
                    pass
        else:
            hand_key = self._normalize_hand_label(next_box.get("label"))
            if hand_key:
                self._select_box_item_in_list(hand_key=hand_key, best_bbox=next_box)
            else:
                self._select_box_item_in_list(
                    object_id=next_box.get("id"),
                    best_bbox=next_box,
                )
        if changed and refresh:
            frame = int(getattr(self.player, "current_frame", 0) or 0)
            self._refresh_boxes_for_frame(frame, skip_events=True, lightweight=True)

    def _on_edit_box_selected(self, box: Optional[dict]) -> None:
        self._set_selected_edit_box(box, refresh=True)

    def _materialize_synthetic_hand_box_for_edit(
        self,
        box: Optional[dict],
    ) -> Optional[dict]:
        if not self._is_synthetic_hand_box(box):
            return None
        hand_key = self._normalize_hand_label((box or {}).get("label"))
        if not hand_key:
            return None
        try:
            frame = int((box or {}).get("frame"))
        except Exception:
            frame = int(getattr(self.player, "current_frame", 0) or 0)

        existing_match = None
        for rb in list(getattr(self, "raw_boxes", []) or []):
            try:
                target_frame = int(rb.get("orig_frame", 0) or 0) + int(self.start_offset)
            except Exception:
                continue
            if target_frame != frame:
                continue
            if self._normalize_hand_label(rb.get("label")) != hand_key:
                continue
            existing_match = dict(rb)
            existing_match["frame"] = frame
            break
        if existing_match:
            self._selected_edit_box = dict(existing_match)
            self._refresh_boxes_for_frame(frame, skip_events=True, lightweight=True)
            return dict(existing_match)

        try:
            target_orig = int(frame) - int(self.start_offset)
        except Exception:
            target_orig = 0

        self._push_undo()
        new_rb = {
            "id": self.box_id_counter,
            "orig_frame": int(target_orig),
            "label": hand_key,
            "source": "materialized_handtrack",
            "locked": False,
            "x1": float(box.get("x1", 0.0) or 0.0),
            "y1": float(box.get("y1", 0.0) or 0.0),
            "x2": float(box.get("x2", 0.0) or 0.0),
            "y2": float(box.get("y2", 0.0) or 0.0),
        }
        self.box_id_counter += 1
        self.raw_boxes.append(new_rb)
        self._selected_edit_box = dict(new_rb)
        self._rebuild_bboxes_from_raw()
        self._bump_bbox_revision()
        self._bump_query_state_revision()
        self._refresh_boxes_for_frame(frame, skip_events=True, lightweight=True)
        self._select_box_item_in_list(hand_key=hand_key)
        self._log(
            "hoi_hand_box_materialize",
            frame=frame,
            hand=hand_key,
            source=self._box_source(box),
        )
        return dict(new_rb)

    def _current_visual_attention_targets(self) -> Dict[str, Any]:
        event_id = getattr(self, "selected_event_id", None)
        hand_key = str(getattr(self, "selected_hand_label", "") or "").strip()
        if event_id is None or not hand_key:
            return {}
        hand_data = self._selected_hand_data()
        if not isinstance(hand_data, dict):
            event = self._find_event_by_id(int(event_id))
            if isinstance(event, dict):
                hand_data = ((event.get("hoi_data", {}) or {}).get(hand_key) or {})
        if not isinstance(hand_data, dict):
            return {}
        target_id = self._hand_noun_object_id(hand_data)
        best_bbox = None
        best_frame = None
        proposal_candidates: List[Dict[str, Any]] = []
        if target_id is None:
            candidates = self._filtered_object_candidates_for_hand(hand_key, hand_data)
            if candidates:
                proposal_candidates = [
                    dict(item) for item in list(candidates[:3] or []) if isinstance(item, dict)
                ]
                top_candidate = dict((proposal_candidates or [])[0] or {})
                try:
                    target_id = int(top_candidate.get("object_id"))
                except Exception:
                    target_id = None
                best_bbox = dict(top_candidate.get("best_bbox") or {})
                try:
                    best_frame = int(top_candidate.get("best_frame"))
                except Exception:
                    best_frame = None
        return {
            "hand_key": hand_key,
            "target_id": target_id,
            "best_frame": best_frame,
            "best_bbox": best_bbox if isinstance(best_bbox, dict) else None,
            "proposal_candidates": proposal_candidates,
        }

    def _current_frame_object_proposal_styles(
        self,
        frame: int,
        hand_key: str,
        hand_data: Optional[dict],
    ) -> Dict[int, Dict[str, Any]]:
        if not isinstance(hand_data, dict):
            return {}
        if self._hand_noun_object_id(hand_data) is not None:
            return {}
        styles: Dict[int, Dict[str, Any]] = {}
        palette = ["#F59E0B", "#FBBF24", "#FB7185"]
        candidates = [
            dict(item)
            for item in list(self._filtered_object_candidates_for_hand(hand_key, hand_data)[:3] or [])
            if isinstance(item, dict)
        ]
        frame = int(frame)
        for idx, candidate in enumerate(candidates):
            try:
                object_id = int(candidate.get("object_id"))
                best_frame = int(candidate.get("best_frame"))
            except Exception:
                continue
            if best_frame != frame:
                continue
            styles[object_id] = {
                "color": palette[min(idx, len(palette) - 1)],
                "thick": True,
                "dashed": True,
                "proposal_rank": int(idx + 1),
                "candidate_score": float(candidate.get("candidate_score", 0.0) or 0.0),
                "best_bbox": dict(candidate.get("best_bbox") or {}),
            }
        return styles

    def _focus_visual_support_for_selected_event(self) -> None:
        attention = self._current_visual_attention_targets()
        hand_key = str(attention.get("hand_key") or "").strip()
        if not hand_key:
            return
        hand_data = self._selected_hand_data()
        current_frame = int(getattr(self.player, "current_frame", 0) or 0)
        grounding_status = self._selected_noun_grounding_status(hand_key, hand_data)
        object_candidates = self._filtered_object_candidates_for_hand(hand_key, hand_data)
        target_id = attention.get("target_id")
        best_bbox = attention.get("best_bbox")
        selected = False
        if target_id is not None:
            selected = self._select_box_item_in_list(
                object_id=target_id,
                best_bbox=best_bbox,
            )
        if not selected:
            selected = self._select_box_item_in_list(hand_key=hand_key)
        grounding_guidance_signature = ""
        if grounding_status:
            onset_frame = grounding_status.get("onset_frame")
            onset_box = dict(grounding_status.get("onset_box") or {})
            current_box = dict(grounding_status.get("current_box") or {})
            if not onset_box:
                candidate_buttons = [
                    btn
                    for btn in list(getattr(self, "inline_object_candidate_buttons", []) or [])
                    if bool(getattr(btn, "isVisible", lambda: False)())
                ]
                target = None
                if candidate_buttons:
                    target = candidate_buttons[0]
                elif self._detection_assist_enabled():
                    target = getattr(self, "btn_inline_detect_objects", None)
                elif onset_frame is not None:
                    target = getattr(self, "btn_inline_jump_onset", None)
                grounding_guidance_signature = "|".join(
                    [
                        str(self.selected_event_id),
                        str(hand_key),
                        "missing_onset_grounding",
                        str(current_frame),
                        str(onset_frame if onset_frame is not None else ""),
                        str(len(candidate_buttons)),
                    ]
                )
                if grounding_guidance_signature != str(getattr(self, "_grounding_visibility_guidance_signature", "") or ""):
                    self._grounding_visibility_guidance_signature = grounding_guidance_signature
                    self._spotlight_widget(target, "#EA580C")
                    if target is not None:
                        try:
                            target.setFocus()
                        except Exception:
                            pass
                return
            if (
                onset_frame is not None
                and int(onset_frame) != int(current_frame)
                and not current_box
            ):
                grounding_guidance_signature = "|".join(
                    [
                        str(self.selected_event_id),
                        str(hand_key),
                        "jump_to_onset_grounding",
                        str(current_frame),
                        str(onset_frame),
                    ]
                )
                if grounding_guidance_signature != str(getattr(self, "_grounding_visibility_guidance_signature", "") or ""):
                    self._grounding_visibility_guidance_signature = grounding_guidance_signature
                    target = getattr(self, "btn_inline_jump_onset", None)
                    self._spotlight_widget(target, "#16A34A")
                    if target is not None:
                        try:
                            target.setFocus()
                        except Exception:
                            pass
                return
        self._grounding_visibility_guidance_signature = ""
        next_field = self._inline_primary_field_name(self._selected_hand_data())
        needs_box_guidance = (
            next_field == "noun_object_id"
            and target_id is None
            and not list(object_candidates or [])
            and not self._frame_has_object_boxes(int(getattr(self.player, "current_frame", 0) or 0))
        )
        if needs_box_guidance and self._detection_assist_enabled():
            target = getattr(self, "btn_inline_detect_objects", None)
            self._spotlight_widget(target, "#EA580C")
            if target is not None:
                try:
                    target.setFocus()
                except Exception:
                    pass
            return
        if next_field == "noun_object_id" and bool(getattr(self, "_advanced_inspector_visible", True)):
            self._focus_inspector_tab("objects")
            self._spotlight_widget(getattr(self, "group_library", None), "#EA580C")
        elif next_field == "noun_object_id":
            self._focus_inline_editor_field("noun_object_id")
        elif next_field == "verb":
            self._focus_inline_editor_field("verb")
        elif next_field == "functional_contact_onset":
            self._focus_inline_editor_field("functional_contact_onset")

    def _guide_find_boxes_if_needed(
        self,
        hand_key: str,
        hand_data: Optional[dict],
        *,
        object_candidates: Optional[Sequence[dict]] = None,
    ) -> None:
        if not self._detection_assist_enabled():
            return
        if not hand_key or not isinstance(hand_data, dict):
            return
        if self._inline_primary_field_name(hand_data) != "noun_object_id":
            return
        if self._hand_noun_object_id(hand_data) is not None:
            return
        candidates = list(object_candidates or [])
        current_frame = int(getattr(self.player, "current_frame", 0) or 0)
        needs_guidance = not candidates and not self._frame_has_object_boxes(current_frame)
        if not needs_guidance:
            return
        signature = "|".join(
            [
                str(self.selected_event_id),
                str(hand_key),
                str(current_frame),
                "find_boxes_needed",
            ]
        )
        if signature == str(getattr(self, "_find_boxes_guidance_signature", "") or ""):
            return
        self._find_boxes_guidance_signature = signature
        target = getattr(self, "btn_inline_detect_objects", None)
        self._spotlight_widget(target, "#EA580C")
        if target is not None:
            try:
                target.setFocus()
            except Exception:
                pass

    def _auto_find_boxes_cache_key(self, event_id: Any, hand_key: str) -> str:
        try:
            event_text = str(int(event_id))
        except Exception:
            event_text = str(event_id or "")
        return f"{event_text}|{str(hand_key or '').strip()}"

    def _clear_auto_find_boxes_attempt(self, event_id: Any, hand_key: str) -> None:
        cache = getattr(self, "_auto_find_boxes_signatures", None)
        if not isinstance(cache, dict):
            return
        cache.pop(self._auto_find_boxes_cache_key(event_id, hand_key), None)

    def _selected_hand_event_keyframes(self, hand_data: Optional[dict]) -> List[int]:
        frames = set()
        if not isinstance(hand_data, dict):
            return []
        for field_name in (
            "interaction_start",
            "functional_contact_onset",
            "interaction_end",
        ):
            try:
                value = hand_data.get(field_name)
                if value is not None:
                    frames.add(int(value))
            except Exception:
                continue
        return sorted(frames)

    def _auto_detect_selected_hand_noun_candidates(
        self,
        hand_key: str,
        hand_data: Optional[dict],
        include_ids: Sequence[int],
        *,
        signature: str,
    ) -> bool:
        if self.selected_event_id is None or not hand_key or not isinstance(hand_data, dict):
            return False
        if self._yolo_infer_worker is not None:
            return False
        keyframes = self._selected_hand_event_keyframes(hand_data)
        if not keyframes:
            return False
        include_id_list = [int(v) for v in sorted(set(include_ids or []))]
        if not include_id_list:
            return False
        frame_map = self._read_video_frames_bgr(keyframes)
        cache_key = self._auto_find_boxes_cache_key(self.selected_event_id, hand_key)
        self._auto_find_boxes_signatures[cache_key] = str(signature or "")
        started = False
        hand_auto_detected = 0
        try:
            hand_auto_detected = self._ensure_hand_boxes_for_detection_frames(
                frame_map,
                {int(frame_idx): [hand_key] for frame_idx in keyframes},
            )
            requests = []
            for frame_idx in keyframes:
                frame_bgr = frame_map.get(int(frame_idx))
                if frame_bgr is None:
                    continue
                requests.append(
                    {
                        "frame_idx": int(frame_idx),
                        "frame_bgr": frame_bgr,
                        "include_ids": list(include_id_list),
                        "replace_existing": False,
                    }
                )
            if not requests:
                return False
            started = self._start_yolo_inference(
                requests,
                context={
                    "mode": "action_keyframes",
                    "push_undo": not bool(hand_auto_detected),
                    "notify_empty": False,
                    "notify_errors": False,
                    "log_event": "hoi_auto_detect_noun_candidates",
                    "log_fields": {
                        "event_id": int(self.selected_event_id),
                        "hand": str(hand_key or ""),
                        "filtered": True,
                        "auto_requested": True,
                        "hand_auto_detected": int(hand_auto_detected),
                    },
                },
            )
        finally:
            if not started:
                self._clear_auto_find_boxes_attempt(self.selected_event_id, hand_key)
        return bool(started)

    def _maybe_auto_detect_selected_hand_noun_candidates(
        self,
        hand_key: str,
        hand_data: Optional[dict],
        *,
        object_candidates: Optional[Sequence[dict]] = None,
    ) -> bool:
        if self.selected_event_id is None or not hand_key or not isinstance(hand_data, dict):
            return False
        event_id = int(self.selected_event_id)
        cache_key = self._auto_find_boxes_cache_key(event_id, hand_key)
        if (
            self._manual_mode_enabled()
            or not self._detection_assist_enabled()
            or not self.player.cap
            or not self.class_map
            or not self.global_object_map
            or self.yolo_model is None
        ):
            self._clear_auto_find_boxes_attempt(event_id, hand_key)
            return False
        if self._inline_primary_field_name(hand_data) != "noun_object_id":
            self._clear_auto_find_boxes_attempt(event_id, hand_key)
            return False
        verb_value = str(hand_data.get("verb") or "").strip()
        verb_state = get_field_state(hand_data, "verb")
        if not verb_value or str(verb_state.get("status") or "").strip().lower() != "confirmed":
            self._clear_auto_find_boxes_attempt(event_id, hand_key)
            return False
        if self._hand_noun_object_id(hand_data) is not None:
            self._clear_auto_find_boxes_attempt(event_id, hand_key)
            return False
        current_candidates = [
            dict(item) for item in list(object_candidates or []) if isinstance(item, dict)
        ]
        if current_candidates:
            self._clear_auto_find_boxes_attempt(event_id, hand_key)
            return False

        noun_suggestion = get_field_suggestion(hand_data, "noun_object_id")
        suggestion_source = str(noun_suggestion.get("source") or "").strip().lower()
        suggestion_value = noun_suggestion.get("value")
        try:
            suggestion_value = int(suggestion_value) if suggestion_value is not None else None
        except Exception:
            suggestion_value = None
        review_recommended = bool(
            noun_suggestion.get("review_recommended", True)
        ) if noun_suggestion else True
        source_decision = self._current_noun_source_decision(hand_data)
        noun_primary_family = str(
            source_decision.get("preferred_family") or ""
        ).strip().lower()
        has_semantic_suggestion = (
            suggestion_value is not None and suggestion_source.startswith("semantic_adapter")
        )
        needs_detection = (
            noun_primary_family == "detector_grounding"
            or not has_semantic_suggestion
            or review_recommended
        )
        if not needs_detection:
            self._clear_auto_find_boxes_attempt(event_id, hand_key)
            return False

        include_ids = set()
        for noun_id in list(self._allowed_noun_ids_for_verb(verb_value) or []):
            try:
                include_ids.add(int(noun_id))
            except Exception:
                continue
        if not include_ids and suggestion_value is not None:
            include_ids.add(int(suggestion_value))
        if not include_ids:
            event = self._find_event_by_id(event_id)
            for noun_id in list(self._preferred_event_detection_ids(event) or []):
                try:
                    include_ids.add(int(noun_id))
                except Exception:
                    continue
        if not include_ids:
            self._clear_auto_find_boxes_attempt(event_id, hand_key)
            return False

        keyframes = self._selected_hand_event_keyframes(hand_data)
        if not keyframes:
            self._clear_auto_find_boxes_attempt(event_id, hand_key)
            return False

        suggestion_confidence = float(noun_suggestion.get("confidence", 0.0) or 0.0)
        signature = "|".join(
            [
                str(event_id),
                str(hand_key or ""),
                str(verb_value),
                ",".join(str(v) for v in keyframes),
                ",".join(str(v) for v in sorted(include_ids)),
                str(suggestion_value if suggestion_value is not None else ""),
                f"{suggestion_confidence:.4f}",
                "review" if review_recommended else "stable",
                str(noun_primary_family or ""),
            ]
        )
        if str(self._auto_find_boxes_signatures.get(cache_key) or "") == signature:
            return False
        started = self._auto_detect_selected_hand_noun_candidates(
            hand_key,
            hand_data,
            sorted(include_ids),
            signature=signature,
        )
        if not started:
            return False
        return True

    def _apply_inline_object_candidate_index(self, idx: int) -> None:
        if idx < 0 or idx >= len(list(self._inline_object_candidates or [])):
            return
        candidate = dict((self._inline_object_candidates or [])[idx] or {})
        object_id = candidate.get("object_id")
        if object_id is None:
            return
        self._apply_noun_choice(
            object_id,
            source="hand_conditioned_object_candidate",
        )
        best_frame = candidate.get("best_frame")
        try:
            best_frame = int(best_frame) if best_frame is not None else None
        except Exception:
            best_frame = None
        onset_frame = None
        onset_box = {}
        hand_key = str(getattr(self, "selected_hand_label", "") or "").strip()
        hand_data = self._selected_hand_data()
        if isinstance(hand_data, dict):
            try:
                onset_frame = (
                    int(hand_data.get("functional_contact_onset"))
                    if hand_data.get("functional_contact_onset") is not None
                    else None
                )
            except Exception:
                onset_frame = None
            if onset_frame is not None:
                onset_box = self._project_candidate_box_to_onset_frame(
                    hand_key,
                    hand_data,
                    object_id,
                    preferred_bbox=dict(candidate.get("best_bbox") or {}),
                    preferred_frame=best_frame,
                    role="noun",
                )
        target_frame = onset_frame if onset_frame is not None and onset_box else best_frame
        target_box = onset_box if onset_box else dict(candidate.get("best_bbox") or {})
        if target_frame is not None and self.player.cap:
            try:
                self.player.seek(target_frame)
            except Exception:
                pass
            self._refresh_boxes_for_frame(target_frame)
            self._set_frame_controls(target_frame)
            self._select_object_item_in_list(
                object_id,
                target_box,
            )
        self._log(
            "hoi_inline_object_candidate_pick",
            event_id=self.selected_event_id,
            hand=self.selected_hand_label,
            noun_object_id=object_id,
            best_frame=best_frame,
            onset_frame=onset_frame,
            visualized_frame=target_frame,
            onset_projected=bool(onset_box),
            source="hand_conditioned_object_candidate",
            candidate_score=float(candidate.get("candidate_score", 0.0) or 0.0),
            **self._flatten_noun_source_decision_fields(
                self._current_noun_source_decision(
                    (self.event_draft or {}).get(self.selected_hand_label, {})
                )
            ),
        )
        self._inline_object_candidates = []
        self._find_boxes_guidance_signature = ""
        self._update_status_label()

    def _inline_detect_selected_event_objects(self) -> None:
        if self.selected_event_id is None:
            return
        self._detect_action_active_items(int(self.selected_event_id))

    def _confirm_current_hand_field(self, field_name: str, source: str = "inline_editor_confirm") -> bool:
        field_name = self._canonical_query_field_name(field_name)
        hand_key = str(getattr(self, "selected_hand_label", "") or "").strip()
        if self.selected_event_id is None or not hand_key or not field_name:
            return False
        event = self._find_event_by_id(int(self.selected_event_id))
        if not isinstance(event, dict):
            return False
        hand_data = (event.get("hoi_data", {}) or {}).get(hand_key, {}) or {}
        self._ensure_hand_annotation_state(hand_data)
        current_value = self._inline_confirmable_field_value(field_name, hand_data)
        if field_name == "verb":
            if not str(current_value or "").strip():
                return False
        elif current_value is None:
            return False
        before_snapshot = self._field_snapshot(hand_data, field_name)
        if str(before_snapshot.get("status") or "").strip().lower() == "confirmed":
            return False
        before_hand = copy.deepcopy(hand_data)
        self._push_undo()
        self._set_hand_field_state(
            hand_data,
            field_name,
            source=source,
            value=current_value,
            status="confirmed",
        )
        draft = self.event_draft.get(hand_key) if isinstance(getattr(self, "event_draft", None), dict) else None
        if isinstance(draft, dict):
            self._set_hand_field_state(
                draft,
                field_name,
                source=source,
                value=current_value,
                status="confirmed",
            )
        after_snapshot = self._field_snapshot(hand_data, field_name)
        self._sync_event_frames(event)
        self._mark_validation_change()
        self._bump_query_state_revision()
        self._refresh_events()
        self._update_hoi_titles()
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.refresh()
        self._update_status_label()
        self._set_semantic_reinfer_hint(
            self.selected_event_id,
            hand_key,
            hand_data,
            reason=f"{field_name}_confirm",
            edited_fields=[field_name],
        )
        self._record_semantic_feedback(
            event,
            hand_key,
            reason=f"{field_name}_confirm",
            before_hand=before_hand,
            edited_fields=[field_name],
            supervision_kind="edited",
        )
        self._refresh_semantic_suggestions_for_event(int(self.selected_event_id), event)
        self._log(
            "hoi_inline_confirm_field",
            source=source,
            field=field_name,
            event_id=self.selected_event_id,
            hand=hand_key,
            old_value=before_snapshot.get("value"),
            new_value=after_snapshot.get("value"),
            old_status=before_snapshot.get("status"),
            new_status=after_snapshot.get("status"),
            old_source=before_snapshot.get("source"),
            new_source=after_snapshot.get("source"),
            old_source_family=before_snapshot.get("source_family"),
            new_source_family=after_snapshot.get("source_family"),
            rework_after_automation=bool(
                (before_snapshot.get("status") or before_snapshot.get("value") is not None)
                and self._is_automation_source(before_snapshot.get("source"))
            ),
        )
        if field_name == "functional_contact_onset":
            self._jump_to_selected_keyframe("functional_contact_onset")
            self._focus_visual_support_for_selected_event()
        return True

    def _focus_inline_editor_field(self, field_name: str) -> None:
        field_name = self._canonical_query_field_name(field_name)
        if field_name == "verb":
            target = getattr(self, "combo_inline_verb", None)
            spotlight_target = target
            if isinstance(target, QComboBox):
                try:
                    spotlight_target = target.lineEdit() or target
                except Exception:
                    spotlight_target = target
            self._spotlight_widget(spotlight_target or getattr(self, "inline_event_card", None), "#2563EB")
            self._focus_combo_editor(target, show_popup=True)
            return
        if field_name == "noun_object_id":
            target = getattr(self, "combo_inline_noun", None)
            if isinstance(target, QComboBox) and bool(target.isEnabled()):
                try:
                    spotlight_target = target.lineEdit() or target
                except Exception:
                    spotlight_target = target
                self._spotlight_widget(spotlight_target or getattr(self, "inline_event_card", None), "#EA580C")
                self._focus_combo_editor(target, show_popup=True)
            else:
                target = getattr(self, "btn_inline_detect_objects", None)
                self._spotlight_widget(target or getattr(self, "inline_event_card", None), "#EA580C")
            if target is not None and not isinstance(target, QComboBox):
                try:
                    target.setFocus()
                except Exception:
                    pass
            return
        if field_name == "instrument_object_id":
            target = getattr(self, "combo_inline_instrument", None)
            spotlight_target = target
            if isinstance(target, QComboBox):
                try:
                    spotlight_target = target.lineEdit() or target
                except Exception:
                    spotlight_target = target
            self._spotlight_widget(spotlight_target or getattr(self, "inline_event_card", None), "#16A34A")
            self._focus_combo_editor(target, show_popup=True)
            return
        if field_name == "anomaly_label":
            target = getattr(self, "combo_inline_anomaly", None)
            spotlight_target = target
            if isinstance(target, QComboBox):
                try:
                    spotlight_target = target.lineEdit() or target
                except Exception:
                    spotlight_target = target
            self._spotlight_widget(spotlight_target or getattr(self, "inline_event_card", None), "#DC2626")
            self._focus_combo_editor(target, show_popup=True)
            return
        if field_name == "bbox_evidence":
            target = getattr(self, "btn_inline_detect_objects", None)
            self._spotlight_widget(target or getattr(self, "inline_event_card", None), "#EA580C")
            if target is not None:
                try:
                    target.setFocus()
                except Exception:
                    pass
            return
        if field_name in ("interaction_start", "functional_contact_onset", "interaction_end"):
            if field_name == "interaction_start":
                target = getattr(self, "btn_inline_jump_start", None)
            elif field_name == "interaction_end":
                target = getattr(self, "btn_inline_jump_end", None)
            else:
                target = getattr(self, "btn_inline_confirm_onset", None) or getattr(
                    self, "btn_inline_jump_onset", None
                )
            self._spotlight_widget(target or getattr(self, "inline_event_card", None), "#2563EB")
            if target is not None:
                try:
                    target.setFocus()
                except Exception:
                    pass
            return
        self._spotlight_widget(getattr(self, "inline_event_card", None), "#2563EB")

    def _on_inline_verb_changed(self, text: str) -> None:
        if self._inline_editor_sync:
            return
        verb_name = str(text or "").strip()
        if not verb_name or not self.selected_hand_label:
            return
        hand_data = self._selected_hand_data() or {}
        current_verb = str(hand_data.get("verb") or "").strip()
        verb_state = get_field_state(hand_data, "verb")
        if verb_name == current_verb and str(verb_state.get("status") or "").strip().lower() != "confirmed":
            if self._confirm_current_hand_field("verb", source="inline_editor_confirm"):
                next_field = self._inline_primary_field_name(self._selected_hand_data())
                if next_field and next_field != "verb":
                    self._focus_inline_editor_field(next_field)
            return
        self._apply_verb_choice(verb_name, source="inline_editor")

    def _on_inline_noun_changed(self, _index: int) -> None:
        if self._inline_editor_sync or not self.selected_hand_label:
            return
        object_id = self.combo_inline_noun.currentData()
        hand_data = self._selected_hand_data() or {}
        current_id = self._hand_noun_object_id(hand_data)
        noun_state = get_field_state(hand_data, "noun_object_id")
        current_verb = str(hand_data.get("verb") or "").strip()
        allows_no_noun = bool(current_verb) and self._verb_allows_no_noun(current_verb)
        noun_confirmed = str(noun_state.get("status") or "").strip().lower() == "confirmed"
        if object_id == current_id and object_id is not None and not noun_confirmed:
            if self._confirm_current_hand_field("noun_object_id", source="inline_editor_confirm"):
                next_field = self._inline_primary_field_name(self._selected_hand_data())
                if next_field and next_field != "noun_object_id":
                    self._focus_inline_editor_field(next_field)
            return
        if object_id == current_id and not (
            object_id is None
            and allows_no_noun
            and str(noun_state.get("status") or "").strip().lower() != "confirmed"
        ):
            return
        if object_id is None and not allows_no_noun and self._noun_required_for_verb(current_verb):
            return
        source = "inline_editor_no_noun" if object_id is None else "inline_editor"
        self._apply_noun_choice(object_id, source=source)
        if object_id is not None:
            hand_after = self._selected_hand_data() or {}
            noun_after = get_field_state(hand_after, "noun_object_id")
            if str(noun_after.get("status") or "").strip().lower() != "confirmed":
                self._confirm_current_hand_field(
                    "noun_object_id",
                    source="inline_editor_confirm",
                )
        next_field = self._inline_primary_field_name(self._selected_hand_data())
        if next_field and next_field != "noun_object_id":
            self._focus_inline_editor_field(next_field)

    def _on_inline_instrument_changed(self, _index: int = 0) -> None:
        if self._inline_editor_sync or not self.selected_hand_label:
            return
        hand_data = self._selected_hand_data()
        if not isinstance(hand_data, dict):
            return
        current_id = self._hand_instrument_object_id(hand_data)
        object_id = self._combo_object_id(
            getattr(self, "combo_inline_instrument", None),
            fallback=current_id,
        )
        try:
            normalized_object_id = int(object_id) if object_id is not None else None
        except Exception:
            normalized_object_id = object_id
        try:
            normalized_current_id = int(current_id) if current_id is not None else None
        except Exception:
            normalized_current_id = current_id
        if normalized_current_id == normalized_object_id:
            return
        self._push_undo()
        hand_data["instrument_object_id"] = normalized_object_id
        hand_data["tool_object_id"] = normalized_object_id
        main_combo = getattr(self, "combo_instrument", None)
        if isinstance(main_combo, QComboBox):
            try:
                main_combo.blockSignals(True)
                self._set_combo_to_data(main_combo, normalized_object_id)
            finally:
                main_combo.blockSignals(False)
        self._apply_draft_to_selected_event()
        self._update_status_label()
        self._refresh_boxes_for_frame(
            int(getattr(self.player, "current_frame", 0) or 0),
            skip_events=True,
            lightweight=True,
        )
        self._log(
            "hoi_apply_instrument_choice",
            source="inline_editor",
            instrument_object_id=normalized_object_id,
            event_id=self.selected_event_id,
            hand=self.selected_hand_label,
        )

    def _on_inline_instrument_text_finished(self) -> None:
        if self._inline_editor_sync or not self.selected_hand_label:
            return
        combo = getattr(self, "combo_inline_instrument", None)
        if not isinstance(combo, QComboBox):
            return
        hand_data = self._selected_hand_data()
        if not isinstance(hand_data, dict):
            return
        current_id = self._hand_instrument_object_id(hand_data)
        object_id = self._combo_object_id(combo, fallback=current_id)
        try:
            combo.blockSignals(True)
            if object_id is None:
                combo.setCurrentIndex(0)
                combo.setEditText("")
            else:
                self._set_combo_to_data(combo, object_id)
        finally:
            combo.blockSignals(False)
        self._on_inline_instrument_changed()

    def _on_inline_anomaly_changed(self) -> None:
        if self._inline_editor_sync or not self.selected_hand_label:
            return
        hand_data = self._selected_hand_data()
        if not isinstance(hand_data, dict):
            return
        label = self._remember_anomaly_label(
            getattr(getattr(self, "combo_inline_anomaly", None), "currentText", lambda: "")()
        )
        current_label = self._normalize_anomaly_label(hand_data.get("anomaly_label"))
        if label == current_label:
            return
        self._push_undo()
        hand_data["anomaly_label"] = label
        hand_data["has_anomaly"] = self._is_anomalous_label(label)
        main_combo = getattr(self, "combo_anomaly", None)
        if isinstance(main_combo, QComboBox):
            try:
                main_combo.blockSignals(True)
                self._populate_anomaly_combo(main_combo, label)
            finally:
                main_combo.blockSignals(False)
        self._apply_draft_to_selected_event()
        self._update_status_label()
        self._log(
            "hoi_apply_anomaly_label",
            source="inline_editor",
            anomaly_label=label,
            has_anomaly=bool(hand_data.get("has_anomaly")),
            event_id=self.selected_event_id,
            hand=self.selected_hand_label,
        )

    def _submit_inline_quick_edit_field(self, field_name: str) -> None:
        if self.selected_event_id is None or not self.selected_hand_label:
            return
        field_name = self._canonical_query_field_name(field_name)
        query = self._inline_query_for_selected()
        target_field = self._query_target_field_name(query)
        primary_field = self._inline_primary_field_name(self._selected_hand_data())
        if target_field and target_field == field_name:
            desired_field = target_field
        elif primary_field and primary_field == field_name:
            desired_field = primary_field
        else:
            desired_field = field_name
        hand_data = self._selected_hand_data()
        if not self._inline_has_confirmable_current_value(desired_field, hand_data):
            self._apply_inline_recommendation()
            return
        if self._confirm_current_hand_field(
            desired_field,
            source="inline_editor_return_confirm",
        ):
            next_field = self._inline_primary_field_name(self._selected_hand_data())
            if next_field and next_field != desired_field:
                self._focus_inline_editor_field(next_field)
            return
        self._apply_inline_recommendation()

    def _apply_inline_recommendation(self) -> None:
        query = self._inline_query_for_selected()
        if query and self._apply_query_suggestion(query):
            return
        if query:
            field_name = self._query_target_field_name(query) or self._inline_primary_field_name(
                self._selected_hand_data()
            )
            hand_data = self._selected_hand_data()
            if (
                field_name
                and self._inline_has_confirmable_current_value(field_name, hand_data)
                and self._confirm_current_hand_field(
                    field_name,
                    source="inline_editor_confirm_current",
                )
            ):
                next_field = self._inline_primary_field_name(self._selected_hand_data())
                if next_field and next_field != field_name:
                    self._focus_inline_editor_field(next_field)
                return
            self._focus_query_for_manual_review(query)
            return
        primary_field = self._inline_primary_field_name(self._selected_hand_data())
        hand_data = self._selected_hand_data()
        if (
            primary_field
            and self._inline_has_confirmable_current_value(primary_field, hand_data)
            and self._confirm_current_hand_field(
                primary_field,
                source="inline_editor_confirm_current",
            )
        ):
            next_field = self._inline_primary_field_name(self._selected_hand_data())
            if next_field and next_field != primary_field:
                self._focus_inline_editor_field(next_field)
            return
        QMessageBox.information(
            self,
            "Recommendation",
            "There is no active recommendation for the selected event.",
        )

    def _confirm_inline_onset(self) -> None:
        query = self._inline_query_for_selected()
        query_field = self._query_target_field_name(query)
        if query and query_field == "functional_contact_onset" and str(query.get("apply_mode") or "").strip() == "confirm_current":
            if self._apply_query_suggestion(query):
                next_field = self._inline_primary_field_name(self._selected_hand_data())
                if next_field and next_field != "functional_contact_onset":
                    self._focus_inline_editor_field(next_field)
                return
        if self._confirm_current_hand_field("functional_contact_onset"):
            next_field = self._inline_primary_field_name(self._selected_hand_data())
            if next_field and next_field != "functional_contact_onset":
                self._focus_inline_editor_field(next_field)
            return
        hand_data = self._selected_hand_data() or {}
        onset_value = hand_data.get("functional_contact_onset")
        onset_state = get_field_state(hand_data, "functional_contact_onset") if isinstance(hand_data, dict) else {}
        if onset_value is None:
            message = "There is no onset value to confirm yet."
        elif str(onset_state.get("status") or "").strip().lower() == "confirmed":
            message = "The current onset is already confirmed."
        else:
            message = "The onset could not be confirmed from the current state. Review the event manually."
        QMessageBox.information(self, "Confirm Onset", message)

    def _on_hoi_timeline_activate(self, event_id: int, hand_key: str) -> None:
        self._set_selected_event(event_id, hand_key)
        self._focus_inline_editor_field(self._inline_primary_field_name())

    def _apply_inline_editor_mode_visibility(self) -> None:
        manual_mode = self._manual_mode_enabled()
        if getattr(self, "lbl_inline_event_step", None) is not None:
            self.lbl_inline_event_step.setVisible(not manual_mode)
        for widget in (
            getattr(self, "inline_grounding_card", None),
            getattr(self, "btn_inline_detect_objects", None),
            getattr(self, "lbl_inline_object_candidates", None),
            getattr(self, "btn_inline_confirm_onset", None),
            getattr(self, "btn_inline_apply_query", None),
            getattr(self, "lbl_inline_query_hint", None),
        ):
            if widget is not None:
                widget.setVisible(not manual_mode)
        if manual_mode:
            for btn in list(getattr(self, "inline_object_candidate_buttons", []) or []):
                try:
                    btn.setVisible(False)
                except Exception:
                    pass

    def _compact_inline_grounding_summary(self, max_chars: int = 72) -> None:
        label = getattr(self, "lbl_inline_object_candidates", None)
        if label is None:
            return
        full_text = str(label.text() or "").strip()
        label.setToolTip(full_text)
        if len(full_text) <= int(max_chars):
            return
        label.setText(full_text[: max(8, int(max_chars) - 3)].rstrip() + "...")

    def _update_inline_event_editor(self) -> None:
        if not hasattr(self, "inline_event_card"):
            return
        if bool(getattr(self, "_inline_editor_sync", False)):
            return
        hand_key = str(getattr(self, "selected_hand_label", "") or "").strip()
        hand_data = self._selected_hand_data()
        query = self._inline_query_for_selected()
        manual_mode = self._manual_mode_enabled()
        refresh_controls = [getattr(self, name, None) for name in
                            ('combo_inline_verb', 'combo_inline_noun', 'combo_inline_instrument', 'combo_inline_anomaly')]
        refresh_signal_states = [(control, control.signalsBlocked()) for control in refresh_controls if control is not None]
        self._inline_editor_sync = True
        try:
            self._apply_inline_editor_mode_visibility()
            if self.selected_event_id is None or not hand_key or not isinstance(hand_data, dict):
                self.lbl_inline_event_title.setText("Manual Edit" if manual_mode else "Quick Edit")
                self.lbl_inline_event_step.setText("Waiting")
                if getattr(self, "lbl_inline_grounding_title", None) is not None:
                    self.lbl_inline_grounding_title.setText("Object Grounding")
                self.lbl_inline_event_summary.setText(
                    "Draw a temporal segment on the timeline to start quick editing."
                    if not manual_mode
                    else "Draw a temporal segment on the timeline, then fill verb and noun here."
                )
                self._set_inline_query_hint(
                    (
                        "Create or select an event, then choose verb, noun, and confirm onset here."
                        if not manual_mode
                        else ""
                    ),
                    "neutral",
                )
                self.combo_inline_verb.blockSignals(True)
                self.combo_inline_verb.clear()
                self.combo_inline_verb.addItem("Choose verb...", "")
                self.combo_inline_verb.blockSignals(False)
                self.combo_inline_noun.blockSignals(True)
                self.combo_inline_noun.clear()
                self.combo_inline_noun.addItem("Choose noun...", None)
                self.combo_inline_noun.blockSignals(False)
                self.combo_inline_instrument.blockSignals(True)
                self.combo_inline_instrument.clear()
                self.combo_inline_instrument.addItem("", None)
                if self.combo_inline_instrument.lineEdit() is not None:
                    self.combo_inline_instrument.lineEdit().setPlaceholderText(
                        "Search instrument / tool..."
                    )
                self.combo_inline_instrument.setEditText("")
                self.combo_inline_instrument.blockSignals(False)
                self._populate_anomaly_combo(
                    self.combo_inline_anomaly,
                    _DEFAULT_ANOMALY_LABEL,
                )
                self._inline_object_candidates = []
                self.btn_inline_detect_objects.setText("Find Boxes")
                self.btn_inline_detect_objects.setEnabled(False)
                self.btn_inline_detect_objects.setToolTip(
                    "Run YOLO on the selected event's start, onset, and end frames to propose editable object boxes. "
                    "You can also left-click the target object on the video canvas to request one local box on the current frame."
                )
                self.lbl_inline_object_candidates.setText(
                    "Object candidates will appear here."
                )
                for btn in self.inline_object_candidate_buttons:
                    btn.setVisible(False)
                    btn.setEnabled(False)
                self.combo_inline_verb.setEnabled(False)
                self.combo_inline_noun.setEnabled(False)
                self.combo_inline_instrument.setEnabled(False)
                self.combo_inline_anomaly.setEnabled(False)
                self.btn_inline_jump_start.setEnabled(False)
                self.btn_inline_jump_onset.setEnabled(False)
                self.btn_inline_jump_end.setEnabled(False)
                self.btn_inline_confirm_onset.setEnabled(False)
                self.btn_inline_apply_query.setEnabled(False)
                self.btn_inline_apply_query.setText("Use Recommendation")
                return

            actor_label = self._get_actor_short_label(hand_key)
            verb_value = str(hand_data.get("verb") or "").strip()
            verb_suggestion = get_field_suggestion(hand_data, "verb")
            suggested_verb_value = str((verb_suggestion or {}).get("value") or "").strip()
            noun_value = self._hand_noun_object_id(hand_data)
            instrument_value = self._hand_instrument_object_id(hand_data)
            anomaly_value = self._normalize_anomaly_label(hand_data.get("anomaly_label"))
            noun_state = get_field_state(hand_data, "noun_object_id")
            start_value = hand_data.get("interaction_start")
            onset_value = hand_data.get("functional_contact_onset")
            end_value = hand_data.get("interaction_end")
            onset_state = get_field_state(hand_data, "functional_contact_onset")
            onset_confirmed = str(onset_state.get("status") or "").strip().lower() == "confirmed"
            object_candidates = self._filtered_object_candidates_for_hand(
                hand_key,
                hand_data,
            )
            noun_grounding_status = self._selected_noun_grounding_status(hand_key, hand_data)
            noun_source_decision = self._current_noun_source_decision(hand_data)
            noun_primary_family = str(
                noun_source_decision.get("preferred_family") or ""
            ).strip().lower()
            completion_state = self._hand_completion_state(
                hand_data,
                hand_key=hand_key,
            )
            primary_field = self._inline_primary_field_name(hand_data)
            step_map = {
                "verb": "Choose Verb",
                "noun_object_id": "Choose Noun",
                "bbox_evidence": "Ground Boxes",
                "functional_contact_onset": "Confirm Onset",
                "interaction_start": "Adjust Timing",
                "interaction_end": "Adjust Timing",
            }
            self.lbl_inline_event_title.setText(
                (
                    f"Manual Edit  Event {self.selected_event_id} {actor_label}"
                    if manual_mode
                    else f"Quick Edit  Event {self.selected_event_id} {actor_label}"
                ).strip()
            )
            missing = list(completion_state.get("inline_missing") or [])
            bbox_missing = list(completion_state.get("bbox_missing") or [])
            bbox_details = list(completion_state.get("bbox_details") or [])
            suggested_fields = list(completion_state.get("suggested_fields") or [])
            is_complete = bool(completion_state.get("complete"))
            if is_complete:
                self.lbl_inline_event_step.setText("Completed")
            elif bbox_missing:
                self.lbl_inline_event_step.setText("Ground Boxes")
            elif suggested_fields and not primary_field:
                self.lbl_inline_event_step.setText("Review")
            else:
                self.lbl_inline_event_step.setText(step_map.get(primary_field, "Ready"))

            if missing:
                if manual_mode:
                    if bbox_missing:
                        self.lbl_inline_event_summary.setText(
                            "Manual mode: add the required boxes: "
                            + "; ".join(bbox_details or bbox_missing)
                            + "."
                        )
                    elif not verb_value:
                        self.lbl_inline_event_summary.setText(
                            "Manual mode: choose the verb, then choose the noun if required."
                        )
                    elif self._noun_required_for_verb(verb_value) and noun_value is None:
                        self.lbl_inline_event_summary.setText(
                            "Manual mode: choose the noun for this event."
                        )
                    else:
                        self.lbl_inline_event_summary.setText(
                            "Manual mode: timing is edited directly on the timeline."
                        )
                elif not verb_value and suggested_verb_value:
                    self.lbl_inline_event_summary.setText(
                        f"Suggested verb: {suggested_verb_value}. Confirm it or choose another verb, then continue."
                    )
                else:
                    self.lbl_inline_event_summary.setText(
                        "Next: " + " -> ".join(missing) + "."
                    )
            elif suggested_fields:
                labels = [self._display_field_label(name) for name in suggested_fields]
                review_text = ", ".join(labels)
                self.lbl_inline_event_summary.setText(
                    (
                        "Manual mode: this event still has review-state fields: "
                        + review_text
                        + ". Reopen and confirm them before saving."
                    )
                    if manual_mode
                    else (
                        "This event has values for the core fields, but these still need confirmation: "
                        + review_text
                        + "."
                    )
                )
            else:
                self.lbl_inline_event_summary.setText(
                    "This event is completed. You can still reopen and adjust it here."
                    if not manual_mode
                    else "Manual mode: this event is completed. You can still adjust timeline, verb, noun, or boxes."
                )

            self.combo_inline_verb.blockSignals(True)
            self.combo_inline_verb.clear()
            self.combo_inline_verb.addItem("Choose verb...", "")
            for item in sorted([v.name for v in self.verbs]):
                self.combo_inline_verb.addItem(item, item)
            target_verb = verb_value or suggested_verb_value
            idx = self.combo_inline_verb.findData(target_verb)
            self.combo_inline_verb.setCurrentIndex(idx if idx >= 0 else 0)
            self.combo_inline_verb.blockSignals(False)
            self.combo_inline_verb.setEnabled(True)

            noun_label = "Noun"
            noun_placeholder = "Choose noun..."
            noun_enabled = bool(verb_value)
            allows_no_noun = bool(verb_value) and self._verb_allows_no_noun(verb_value)
            if verb_value and not self._noun_required_for_verb(verb_value):
                noun_label = "Noun (Optional)"
            elif not verb_value:
                noun_placeholder = "Choose verb first..."
            self.lbl_inline_noun.setText(noun_label)
            self.combo_inline_noun.blockSignals(True)
            self.combo_inline_noun.clear()
            self.combo_inline_noun.addItem(noun_placeholder, None)
            for idx in range(self.combo_target.count()):
                text = self.combo_target.itemText(idx)
                data = self.combo_target.itemData(idx)
                if idx == 0:
                    continue
                self.combo_inline_noun.addItem(text, data)
            target_noun = noun_value if noun_value is not None else (None if allows_no_noun else None)
            noun_idx = self.combo_inline_noun.findData(target_noun)
            self.combo_inline_noun.setCurrentIndex(noun_idx if noun_idx >= 0 else 0)
            self.combo_inline_noun.blockSignals(False)
            self.combo_inline_noun.setEnabled(noun_enabled)

            self.combo_inline_instrument.blockSignals(True)
            self.combo_inline_instrument.clear()
            self.combo_inline_instrument.addItem("", None)
            if self.combo_inline_instrument.lineEdit() is not None:
                self.combo_inline_instrument.lineEdit().setPlaceholderText(
                    "Search instrument / tool..."
                )
            source_combo = getattr(self, "combo_instrument", None) or getattr(
                self,
                "combo_target",
                None,
            )
            if isinstance(source_combo, QComboBox):
                for idx in range(source_combo.count()):
                    text = source_combo.itemText(idx)
                    data = source_combo.itemData(idx)
                    if idx == 0:
                        continue
                    self.combo_inline_instrument.addItem(text, data)
            inst_idx = self.combo_inline_instrument.findData(instrument_value)
            self.combo_inline_instrument.setCurrentIndex(inst_idx if inst_idx >= 0 else 0)
            if instrument_value is None:
                self.combo_inline_instrument.setEditText("")
            self.combo_inline_instrument.blockSignals(False)
            self.combo_inline_instrument.setEnabled(True)

            self._populate_anomaly_combo(self.combo_inline_anomaly, anomaly_value)
            self.combo_inline_anomaly.setEnabled(True)

            detection_assist_enabled = self._detection_assist_enabled()
            detection_enabled = detection_assist_enabled and self._yolo_infer_worker is None
            self.btn_inline_detect_objects.setEnabled(detection_enabled)
            grounding_title = "Object Grounding"
            noun_confirmed = (
                noun_value is not None
                and str(noun_state.get("status") or "").strip().lower() == "confirmed"
            )
            onset_grounding_box = dict(noun_grounding_status.get("onset_box") or {})
            current_grounding_box = dict(noun_grounding_status.get("current_box") or {})
            onset_grounding_frame = noun_grounding_status.get("onset_frame")
            current_editor_frame = noun_grounding_status.get("current_frame")
            noun_grounding_candidates: List[dict] = []
            if noun_value is not None:
                try:
                    noun_object_id = int(noun_value)
                except Exception:
                    noun_object_id = None
                if noun_object_id is not None:
                    for item in list(object_candidates or []):
                        if not isinstance(item, dict):
                            continue
                        try:
                            candidate_object_id = int(item.get("object_id"))
                        except Exception:
                            continue
                        if candidate_object_id == noun_object_id:
                            noun_grounding_candidates.append(dict(item))
            if bbox_missing:
                grounding_title = "Required Boxes"
                self.lbl_inline_object_candidates.setText(
                    "Missing required bbox evidence: "
                    + "; ".join(bbox_details or bbox_missing)
                    + "."
                )
                self.btn_inline_detect_objects.setText("Find Boxes")
                self.btn_inline_detect_objects.setToolTip(
                    (
                        "Run YOLO/MediaPipe on this event's start, onset, and end frames to fill missing hand, noun, and instrument boxes."
                    )
                    if detection_assist_enabled and self._yolo_infer_worker is None
                    else (
                        "Wait for the current detection job to finish."
                        if detection_assist_enabled
                        else "Detection assist is disabled in Manual mode. Draw or edit the missing boxes manually."
                    )
                )
                self._inline_object_candidates = []
                self._find_boxes_guidance_signature = ""
            elif noun_confirmed:
                if onset_grounding_box:
                    grounding_title = "Onset Grounding"
                    if (
                        onset_grounding_frame is not None
                        and current_editor_frame is not None
                        and int(current_editor_frame) != int(onset_grounding_frame)
                    ):
                        self.lbl_inline_object_candidates.setText(
                            f"Noun confirmed. The canonical box is on onset frame {int(onset_grounding_frame)}; jump to Onset to inspect or adjust it."
                        )
                    else:
                        self.lbl_inline_object_candidates.setText(
                            "Noun confirmed. The onset-frame box is visible and can be adjusted directly on the image."
                        )
                    self.btn_inline_detect_objects.setText("Refresh Boxes")
                    self.btn_inline_detect_objects.setToolTip(
                        "Re-run YOLO on this event only if you need a different object grounding."
                    )
                    self._inline_object_candidates = []
                elif noun_grounding_candidates:
                    grounding_title = "Place Onset Box"
                    if current_grounding_box:
                        self.lbl_inline_object_candidates.setText(
                            "Noun confirmed, but the annotation still has no onset-frame box. Boxes on other frames do not count; use the grounding button below to place one on onset."
                        )
                    else:
                        self.lbl_inline_object_candidates.setText(
                            "Noun confirmed, but the annotation still has no onset-frame box. The buttons below keep the noun fixed and only help place grounding on onset."
                        )
                    self.btn_inline_detect_objects.setText("Refresh Boxes")
                    self.btn_inline_detect_objects.setToolTip(
                        "Re-run YOLO on the selected event keyframes to refresh grounding candidates for the confirmed noun."
                    )
                    self._inline_object_candidates = [
                        dict(item) for item in list(noun_grounding_candidates[:3] or [])
                    ]
                else:
                    grounding_title = "Object Grounding"
                    if detection_assist_enabled and self._yolo_infer_worker is not None:
                        self.lbl_inline_object_candidates.setText(
                            "Noun confirmed, but the annotation still has no onset-frame box. Detection is running for this event."
                        )
                    elif detection_assist_enabled:
                        self.lbl_inline_object_candidates.setText(
                            "Noun confirmed, but the annotation still has no onset-frame box. Use Find Boxes, or jump to Onset and draw/edit it manually."
                        )
                    else:
                        self.lbl_inline_object_candidates.setText(
                            "Noun confirmed, but the annotation still has no onset-frame box. Jump to Onset and draw/edit it manually."
                        )
                    self.btn_inline_detect_objects.setText("Find Boxes")
                    self.btn_inline_detect_objects.setToolTip(
                        (
                            "Run YOLO on the selected event keyframes to place or refresh an editable object box for the confirmed noun. "
                            "You can also jump to onset and draw or edit the box manually."
                        )
                        if detection_assist_enabled and self._yolo_infer_worker is None
                        else (
                            "Wait for the current detection job to finish."
                            if detection_assist_enabled
                            else "Detection assist is disabled in Manual mode. Jump to onset and draw or edit the box manually."
                        )
                    )
                    self._inline_object_candidates = []
                self._find_boxes_guidance_signature = ""
            elif object_candidates:
                grounding_title = "Recommended Objects"
                if noun_primary_family == "detector_grounding":
                    self.lbl_inline_object_candidates.setText(
                        "Hand-conditioned grounding is currently the stronger noun cue; top object boxes are listed below."
                    )
                elif noun_primary_family == "semantic":
                    self.lbl_inline_object_candidates.setText(
                        "Semantic noun is currently stronger; the boxes below are grounding helpers."
                    )
                else:
                    self.lbl_inline_object_candidates.setText(
                        "Candidates ranked from hand-conditioned object support."
                    )
                self.btn_inline_detect_objects.setText("Refresh Boxes")
                self.btn_inline_detect_objects.setToolTip(
                    "Re-run YOLO on the selected event keyframes to refresh object candidates."
                )
                self._inline_object_candidates = [
                    dict(item) for item in list(object_candidates[:3] or [])
                ]
                self._find_boxes_guidance_signature = ""
            else:
                if detection_assist_enabled and self._yolo_infer_worker is not None:
                    self.lbl_inline_object_candidates.setText(
                        "Detection is running for the current clip/event. Candidate buttons will update if usable boxes are found."
                    )
                elif detection_assist_enabled:
                    self.lbl_inline_object_candidates.setText(
                        "No object boxes yet. Use Find Boxes or left-click the target on the current frame."
                    )
                else:
                    self.lbl_inline_object_candidates.setText(
                        "No object boxes yet. Manual mode keeps detection off, so draw or edit the box yourself."
                    )
                self.btn_inline_detect_objects.setText("Find Boxes")
                self.btn_inline_detect_objects.setToolTip(
                    (
                        "Run YOLO on the selected event's start, onset, and end frames to propose editable object boxes. "
                        "You can also left-click the target object on the video canvas to request one local box on the current frame."
                    )
                    if detection_assist_enabled and self._yolo_infer_worker is None
                    else (
                        "Wait for the current detection job to finish."
                        if detection_assist_enabled
                        else "Detection assist is disabled in Manual mode. Switch to Full Assist to use Find Boxes."
                    )
                )
                self._inline_object_candidates = []
            self._compact_inline_grounding_summary()
            self.lbl_inline_grounding_title.setText(grounding_title)
            for idx, btn in enumerate(self.inline_object_candidate_buttons):
                if manual_mode:
                    btn.setText("")
                    btn.setVisible(False)
                    btn.setEnabled(False)
                    btn.setProperty("candidatePrimary", False)
                    self._refresh_widget_style(btn)
                    continue
                if idx < len(self._inline_object_candidates):
                    candidate = dict(self._inline_object_candidates[idx] or {})
                    candidate_name = str(
                        candidate.get("object_name")
                        or candidate.get("display_value")
                        or candidate.get("object_id")
                    ).strip()
                    candidate_score = float(
                        candidate.get("candidate_score", 0.0) or 0.0
                    )
                    best_frame = candidate.get("best_frame")
                    tooltip_prefix = (
                        "Top recommended object\n"
                        if idx == 0
                        else f"Recommended object {idx + 1}\n"
                    )
                    btn.setText(candidate_name)
                    btn.setToolTip(
                        f"{tooltip_prefix}Frame: {best_frame}\n"
                        f"Score: {candidate_score:.2f}\n"
                        f"YOLO max: {float(candidate.get('yolo_confidence_max', 0.0) or 0.0):.2f}\n"
                        f"Hand proximity max: {float(candidate.get('hand_proximity_max', 0.0) or 0.0):.2f}"
                    )
                    btn.setProperty("candidatePrimary", idx == 0)
                    btn.setVisible(True)
                    btn.setEnabled(True)
                    self._refresh_widget_style(btn)
                else:
                    btn.setText("")
                    btn.setVisible(False)
                    btn.setEnabled(False)
                    btn.setProperty("candidatePrimary", False)
                    self._refresh_widget_style(btn)

            auto_detect_started = False
            if not manual_mode:
                auto_detect_started = self._maybe_auto_detect_selected_hand_noun_candidates(
                    hand_key,
                    hand_data,
                    object_candidates=self._inline_object_candidates,
                )
                if not auto_detect_started:
                    self._guide_find_boxes_if_needed(
                        hand_key,
                        hand_data,
                        object_candidates=self._inline_object_candidates,
                    )

            self.btn_inline_jump_start.setEnabled(start_value is not None)
            self.btn_inline_jump_onset.setEnabled(onset_value is not None)
            self.btn_inline_jump_end.setEnabled(end_value is not None)
            self.btn_inline_confirm_onset.setEnabled(onset_value is not None and not onset_confirmed)
            self.btn_inline_confirm_onset.setText(
                "Onset Confirmed" if onset_confirmed else "Confirm Onset"
            )
            self.btn_inline_confirm_onset.setToolTip(
                "Mark the current onset as confirmed."
                if not onset_confirmed
                else "The current onset is already confirmed."
            )

            if is_complete and not manual_mode:
                self.btn_inline_apply_query.setEnabled(False)
                self.btn_inline_apply_query.setText("Completed")
                self.btn_inline_apply_query.setToolTip(
                    "The selected event already has confirmed core fields."
                )
                self._set_inline_query_hint(
                    "This event is complete. Use Review only if you want to inspect evidence or reopen a field.",
                    "complete",
                )
            elif query and not manual_mode:
                can_apply = bool(query.get("safe_apply")) or (
                    str(query.get("action_kind") or "").strip().lower() == "suggest"
                    and query.get("suggested_value") is not None
                )
                confirmable_current = bool(
                    self._inline_has_confirmable_current_value(
                        self._query_target_field_name(query),
                        hand_data,
                    )
                )
                self.btn_inline_apply_query.setEnabled(True)
                action_text = str(self.btn_next_query_apply.text() or "").strip()
                self.btn_inline_apply_query.setText(
                    action_text
                    if can_apply
                    else (
                        self._query_confirm_button_label(query)
                        if confirmable_current
                        else "Review Manually"
                    )
                )
                summary_text = str(query.get("summary") or "Use the current recommendation.")
                if not can_apply:
                    summary_text = (
                        summary_text
                        + " This recommendation is review-only. Use the controls below to confirm it manually."
                    )
                self._set_inline_query_hint(
                    summary_text,
                    "action" if can_apply else "review",
                )
                self.btn_inline_apply_query.setToolTip(
                    (
                        str(query.get("reason") or "").strip()
                        or "Apply the current controller recommendation."
                    )
                    if can_apply
                    else (
                        "Confirm the current value without changing it."
                        if confirmable_current
                        else (
                        str(query.get("reason") or "").strip()
                        or "Focus the suggested control below and confirm it manually."
                        )
                    )
                )
            else:
                confirmable_current = bool(
                    primary_field
                    and self._inline_has_confirmable_current_value(hand_data=hand_data, field_name=primary_field)
                )
                self.btn_inline_apply_query.setEnabled(bool(confirmable_current) and not manual_mode)
                self.btn_inline_apply_query.setText(
                    self._query_confirm_button_label(
                        {"field_name": primary_field}
                    )
                    if confirmable_current
                    else "Use Recommendation"
                )
                self._set_inline_query_hint(
                    (
                        ""
                        if manual_mode
                        else (
                            "Confirm the current value to keep it as-is."
                            if confirmable_current
                            else "The controller will surface in-context recommendations for this event here."
                        )
                    ),
                    "action" if confirmable_current and not manual_mode else "neutral",
                )
                self.btn_inline_apply_query.setToolTip(
                    ""
                    if manual_mode
                    else (
                        "Confirm the current value without changing it."
                        if confirmable_current
                        else ""
                    )
                )
            if not query and not manual_mode:
                if (
                    primary_field == "noun_object_id"
                    and noun_value is None
                    and not list(object_candidates or [])
                ):
                    if detection_enabled:
                        self._set_inline_query_hint(
                            "Next: identify the noun. Press Find Boxes to run YOLO on this event, or left-click the target object on the current frame for one local box.",
                            "action",
                        )
                    else:
                        self._set_inline_query_hint(
                            "Next: identify the noun. Detection assist is off in Manual mode, so draw or edit the object box manually.",
                            "action",
                        )
                elif primary_field == "verb":
                    self._set_inline_query_hint(
                        "Next: choose or correct the verb. If the current suggestion is already right, use Confirm Verb.",
                        "action",
                    )
                elif primary_field == "noun_object_id":
                    self._set_inline_query_hint(
                        "Next: choose or correct the noun. Selecting a noun here, or clicking one matching box candidate below, confirms it immediately.",
                        "action",
                    )
                elif primary_field == "bbox_evidence":
                    self._set_inline_query_hint(
                        "Next: add the required boxes. Use Find Boxes, or draw/edit the missing hand, noun, or instrument boxes manually.",
                        "action",
                    )
                elif primary_field == "functional_contact_onset":
                    self._set_inline_query_hint(
                        "Next: refine the onset on the timeline, then press Confirm Onset.",
                        "action",
                    )
        finally:
            for control, previous in refresh_signal_states:
                control.blockSignals(previous)
            self._inline_editor_sync = False
            self._refresh_label_explanations()
            if bool(getattr(self, "_completion_status_refresh_pending", False)):
                self._completion_status_refresh_pending = False
                QTimer.singleShot(0, self._update_status_label)

    def _focus_combo_editor(
        self,
        combo: Optional[QComboBox],
        *,
        show_popup: bool = False,
    ) -> None:
        if combo is None:
            return
        try:
            combo.setFocus()
        except Exception:
            pass
        editor = None
        try:
            editor = combo.lineEdit()
        except Exception:
            editor = None
        if editor is not None:
            try:
                editor.setFocus()
                editor.selectAll()
            except Exception:
                pass
        if show_popup:
            self._show_combo_popup_soon(combo)

    def _query_target_field_name(self, query: Optional[dict]) -> str:
        if not isinstance(query, dict):
            return ""
        field_name = self._canonical_query_field_name(query.get("field_name"))
        if field_name and field_name != "completion_bundle":
            return field_name
        bundle_fields = [
            self._canonical_query_field_name(item.get("field_name"))
            for item in list(query.get("completion_fields") or [])
            if isinstance(item, dict)
        ]
        hand_data = self._query_event_hand_data(query)
        primary_field = self._inline_primary_field_name(
            hand_data,
            allow_query_fallback=False,
        )
        if primary_field and primary_field in bundle_fields:
            return primary_field
        for candidate in (
            "verb",
            "noun_object_id",
            "bbox_evidence",
            "instrument_object_id",
            "anomaly_label",
            "functional_contact_onset",
            "interaction_start",
            "interaction_end",
        ):
            if candidate in bundle_fields:
                return candidate
        return str(bundle_fields[0] or "").strip() if bundle_fields else ""

    def _query_focus_button_label(self, query: Optional[dict]) -> str:
        field_name = self._query_target_field_name(query)
        if field_name == "verb":
            return "Open Action"
        if field_name == "noun_object_id":
            return "Open Object"
        if field_name == "instrument_object_id":
            return "Open Instrument"
        if field_name == "anomaly_label":
            return "Open Anomaly"
        if field_name == "bbox_evidence":
            return "Open Boxes"
        if field_name in (
            "interaction_start",
            "functional_contact_onset",
            "interaction_end",
        ):
            return "Open Timing"
        return "Focus"

    def _route_query_target_control(
        self,
        query: Optional[dict],
        *,
        set_keyboard_focus: bool,
        allow_select_event: bool,
    ) -> None:
        if not isinstance(query, dict) or not query:
            return
        event_id = query.get("event_id")
        hand = str(query.get("hand") or "").strip()
        if allow_select_event and event_id is not None and hand:
            try:
                self._set_selected_event(int(event_id), hand)
            except Exception:
                pass
        elif event_id is not None and hand:
            try:
                if int(event_id) != int(self.selected_event_id) or hand != str(self.selected_hand_label or ""):
                    return
            except Exception:
                return
        field_name = self._query_target_field_name(query)
        if not field_name:
            field_name = self._inline_primary_field_name(self._selected_hand_data())
        if not bool(getattr(self, "_advanced_inspector_visible", True)):
            self._focus_inline_editor_field(field_name or self._inline_primary_field_name())
            return
        if field_name == "verb":
            self._focus_inspector_tab("event")
            self._spotlight_widget(getattr(self, "group_action", None), "#2563EB")
            if set_keyboard_focus:
                self._focus_combo_editor(getattr(self, "combo_verb", None), show_popup=True)
            return
        if field_name == "noun_object_id":
            self._focus_inspector_tab("objects")
            self._spotlight_widget(getattr(self, "group_library", None), "#EA580C")
            if set_keyboard_focus:
                self._focus_combo_editor(getattr(self, "combo_target", None), show_popup=True)
            return
        if field_name == "instrument_object_id":
            self._focus_inspector_tab("objects")
            self._spotlight_widget(getattr(self, "group_library", None), "#16A34A")
            if set_keyboard_focus:
                self._focus_combo_editor(
                    getattr(self, "combo_instrument", None),
                    show_popup=True,
                )
            return
        if field_name == "anomaly_label":
            self._focus_inspector_tab("objects")
            self._spotlight_widget(getattr(self, "group_library", None), "#DC2626")
            if set_keyboard_focus:
                self._focus_combo_editor(
                    getattr(self, "combo_anomaly", None),
                    show_popup=True,
                )
            return
        if field_name == "bbox_evidence":
            self._focus_inspector_tab("objects")
            target = getattr(self, "btn_inline_detect_objects", None)
            self._spotlight_widget(target or getattr(self, "group_library", None), "#EA580C")
            if set_keyboard_focus and target is not None:
                try:
                    target.setFocus()
                except Exception:
                    pass
            return
        if field_name in ("interaction_start", "functional_contact_onset", "interaction_end"):
            self._focus_inspector_tab("event")
            self._focus_timeline_workspace()
            timing_widget = getattr(self, "event_status_card", None)
            focus_widget = None
            if field_name == "interaction_start":
                focus_widget = getattr(self, "spin_start_offset", None)
            elif field_name == "interaction_end":
                focus_widget = getattr(self, "spin_end_frame", None)
            elif field_name == "functional_contact_onset":
                focus_widget = getattr(self, "btn_jump_onset_chip", None)
                if focus_widget is None or not focus_widget.isEnabled():
                    focus_widget = getattr(self, "hoi_timeline", None)
            self._spotlight_widget(focus_widget or timing_widget, "#2563EB")
            if set_keyboard_focus and focus_widget is not None:
                try:
                    focus_widget.setFocus()
                except Exception:
                    pass
            return
        self._focus_inline_editor_field(field_name or self._inline_primary_field_name())

    def _query_manual_review_hint(self, query: Optional[dict]) -> str:
        field_name = self._query_target_field_name(query)
        if not field_name:
            field_name = self._inline_primary_field_name(self._selected_hand_data())
        if field_name == "verb":
            return "Review now: choose or correct the verb."
        if field_name == "noun_object_id":
            return "Review now: choose or correct the noun/object."
        if field_name == "bbox_evidence":
            return "Review now: add the required hand, noun, or instrument boxes."
        if field_name == "functional_contact_onset":
            return "Review now: refine the onset on the timeline, then press Confirm Onset."
        if field_name == "interaction_start":
            return "Review now: adjust the event start on the timeline."
        if field_name == "interaction_end":
            return "Review now: adjust the event end on the timeline."
        return "Review now: inspect the highlighted control and confirm it manually."

    def _focus_query_for_manual_review(self, query: Optional[dict]) -> None:
        if not isinstance(query, dict) or not query:
            return
        self._spotlight_widget(getattr(self, "btn_inline_apply_query", None), "#DC2626", duration_ms=1200)
        self._focus_query_candidate(query)
        hint = self._query_manual_review_hint(query)
        try:
            self._set_inline_query_hint(hint, "review")
        except Exception:
            pass
        field_name = self._query_target_field_name(query) or self._inline_primary_field_name(
            self._selected_hand_data()
        )
        self._focus_inline_editor_field(field_name)
        try:
            QTimer.singleShot(
                120,
                lambda field=field_name: self._focus_inline_editor_field(field),
            )
        except Exception:
            pass

    def _update_inspector_tab_labels(self) -> None:
        tabs = getattr(self, "inspector_tabs", None)
        if tabs is None or tabs.count() < 3:
            return
        event_text = "Event"
        if self.selected_event_id is not None:
            event_text += " *"
        object_list = getattr(self, "list_objects", None)
        object_count = int(object_list.count()) if object_list is not None else 0
        objects_text = f"Objects ({object_count})" if object_count > 0 else "Objects"
        issue_count = len(getattr(self, "_incomplete_issues", []) or [])
        review_text = f"Review ({issue_count})" if issue_count > 0 else "Review"
        tabs.setTabText(0, event_text)
        tabs.setTabText(1, objects_text)
        tabs.setTabText(2, review_text)

    def _apply_micro_interaction_icons(self) -> None:
        style = self.style()
        icon_px = self._scaled_ui_px(18, 14)
        transport_w = self._scaled_ui_px(32, 26)
        chrome_w = self._scaled_ui_px(30, 24)
        try:
            self.btn_file_menu.setIcon(style.standardIcon(QStyle.SP_DialogOpenButton))
            self.btn_file_menu.setText("")
            self.btn_file_menu.setToolButtonStyle(Qt.ToolButtonIconOnly)
            self.btn_file_menu.setIconSize(QSize(icon_px, icon_px))
            self.btn_file_menu.setFixedWidth(chrome_w)
            self.btn_file_menu.setAutoRaise(True)
        except Exception:
            pass
        try:
            self.btn_quick_start.setIcon(style.standardIcon(QStyle.SP_MessageBoxQuestion))
            self.btn_quick_start.setText("")
            self.btn_quick_start.setToolButtonStyle(Qt.ToolButtonIconOnly)
            self.btn_quick_start.setIconSize(QSize(icon_px, icon_px))
            self.btn_quick_start.setFixedWidth(chrome_w)
            self.btn_quick_start.setAutoRaise(True)
        except Exception:
            pass
        try:
            for button in (self.btn_rew, self.btn_play, self.btn_stop, self.btn_ff):
                button.setIconSize(QSize(icon_px, icon_px))
                button.setFixedWidth(transport_w)
                button.setAutoRaise(True)
            self.btn_jump.setFixedWidth(self._scaled_ui_px(28, 24))
        except Exception:
            pass
        try:
            self.btn_suggest_action_label.setIcon(
                style.standardIcon(QStyle.SP_BrowserReload)
            )
            self.btn_suggest_action_label.setText("")
            self.btn_suggest_action_label.setToolButtonStyle(Qt.ToolButtonIconOnly)
            self.btn_suggest_action_label.setIconSize(QSize(icon_px, icon_px))
            self.btn_suggest_action_label.setAutoRaise(True)
        except Exception:
            pass
        try:
            self.btn_object_tools.setText("...")
            self.btn_object_tools.setToolButtonStyle(Qt.ToolButtonTextOnly)
            self.btn_object_tools.setAutoRaise(True)
            self.btn_action_tools.setText("...")
            self.btn_action_tools.setToolButtonStyle(Qt.ToolButtonTextOnly)
            self.btn_action_tools.setAutoRaise(True)
        except Exception:
            pass

    def _query_surface_tone(self, surface: str) -> str:
        key = str(surface or "").strip().lower()
        if key == "event":
            return "active"
        if key == "objects":
            return "warn"
        if key == "review":
            return "danger"
        return "neutral"

    def _query_surface_chip_tone(self, surface: str) -> str:
        key = str(surface or "").strip().lower()
        if key == "event":
            return "neutral"
        if key == "objects":
            return "warn"
        if key == "review":
            return "danger"
        return "neutral"

    def _query_action_label(self, action_kind: str) -> str:
        key = str(action_kind or "").strip().lower()
        if key == "query":
            return "Query"
        if key == "suggest":
            return "Suggest"
        if key == "propagate":
            return "Propagate"
        return "Action --"

    def _query_action_chip_tone(self, action_kind: str) -> str:
        key = str(action_kind or "").strip().lower()
        if key == "query":
            return "neutral"
        if key == "suggest":
            return "warn"
        if key == "propagate":
            return "ok"
        return "neutral"

    def _query_form_label(self, interaction_form: str) -> str:
        key = str(interaction_form or "").strip().lower()
        mapping = {
            "confirm_current": "Confirm",
            "bundle_accept": "Bundle",
            "accept_suggestion": "Accept",
            "draw_keyframe_box": "Draw Box",
            "choose_object": "Choose",
            "review_conflict": "Review",
            "manual_edit": "Edit",
        }
        return mapping.get(key, "How --")

    def _query_form_chip_tone(self, interaction_form: str) -> str:
        key = str(interaction_form or "").strip().lower()
        if key in ("bundle_accept", "accept_suggestion"):
            return "ok"
        if key in ("draw_keyframe_box", "choose_object", "review_conflict"):
            return "warn"
        return "neutral"

    def _query_authority_label(self, authority_level: str) -> str:
        key = str(authority_level or "").strip().lower()
        if key == "human_only":
            return "Human"
        if key == "human_confirm":
            return "Confirm"
        if key == "safe_local":
            return "Safe Auto"
        return "Authority --"

    def _query_authority_chip_tone(self, authority_level: str) -> str:
        key = str(authority_level or "").strip().lower()
        if key == "human_only":
            return "neutral"
        if key == "human_confirm":
            return "warn"
        if key == "safe_local":
            return "ok"
        return "neutral"

    def _record_query_metric(self, kind: str, amount: int = 1) -> None:
        key = str(kind or "").strip().lower()
        if not key:
            return
        self._query_metrics[key] = int(self._query_metrics.get(key, 0) or 0) + int(amount)
        self._update_query_session_metrics()

    def _query_source_label(self, query: Optional[dict]) -> str:
        if not isinstance(query, dict):
            return "query_controller"
        source = str(
            query.get("suggested_source")
            or query.get("source")
            or "query_controller"
        ).strip()
        return source or "query_controller"

    def _query_is_hand_conditioned(self, query: Optional[dict]) -> bool:
        if not isinstance(query, dict):
            return False
        if bool(query.get("hand_conditioned")):
            return True
        source = self._query_source_label(query).lower()
        query_type = str(query.get("query_type") or "").strip().lower()
        return (
            source.startswith("handtrack_once")
            or source.startswith("hand_conditioned")
            or "hand_conditioned" in query_type
        )

    def _next_query_compact_status(self, query: Optional[dict]) -> str:
        if not isinstance(query, dict) or not query:
            return "Waiting"
        action_kind = str(query.get("action_kind") or "").strip().lower()
        authority_level = str(query.get("authority_level") or "").strip().lower()
        if authority_level == "safe_local":
            return "Auto"
        if action_kind == "suggest":
            return "Suggest"
        if action_kind == "query":
            return "Review"
        if action_kind == "propagate":
            return "Propagate"
        return "Next"

    def _next_query_compact_evidence(
        self,
        query: Optional[dict],
        *,
        query_source: str,
        query_hand_conditioned: bool,
        evidence_expected: int,
        evidence_confirmed: int,
        evidence_missing: int,
    ) -> str:
        if query_hand_conditioned:
            base = "Source: hand-conditioned onset prior."
        elif query_source and query_source != "query_controller":
            base = f"Source: {query_source}."
        else:
            base = "Source: controller recommendation."
        if evidence_expected > 0:
            return (
                f"{base} Grounding {evidence_confirmed}/{evidence_expected}, "
                f"{evidence_missing} unresolved."
            )
        return f"{base} Add rough start/end or object support to unlock more suggestions."

    def _set_next_query_compact_tooltip(
        self,
        query: Optional[dict],
        *,
        summary: str,
        reason: str,
        policy_reason: str,
        query_source: str,
        query_hand_conditioned: bool,
        surface: str,
        action_kind: str,
        interaction_form: str,
        authority_level: str,
        voi: float,
        propagation: float,
        cost: float,
        risk: float,
        evidence_expected: int,
        evidence_confirmed: int,
        evidence_missing: int,
    ) -> None:
        metrics = dict(getattr(self, "_query_metrics", {}) or {})
        tooltip_lines = [
            summary or "Next supervision step.",
            reason or "Controller-selected step.",
        ]
        if policy_reason:
            tooltip_lines.append(f"Policy: {policy_reason}")
        tooltip_lines.extend(
            [
                "",
                f"Source: {query_source or 'query_controller'}",
                f"Hand-conditioned: {'true' if query_hand_conditioned else 'false'}",
                f"Surface: {surface or '--'}",
                f"Action: {action_kind or '--'}",
                f"Form: {interaction_form or '--'}",
                f"Authority: {authority_level or '--'}",
                f"VOI={voi:.2f} | Prop={propagation:.2f} | Cost={cost:.2f} | Risk={risk:.2f}",
            ]
        )
        if evidence_expected > 0:
            tooltip_lines.append(
                f"Grounding: {evidence_confirmed}/{evidence_expected} grounded, {evidence_missing} unresolved"
            )
        tooltip_lines.append(
            "Session: "
            f"{int(metrics.get('presented', 0) or 0)} shown | "
            f"{int(metrics.get('focused', 0) or 0)} focused | "
            f"{int(metrics.get('accepted', 0) or 0)} accepted | "
            f"{int(metrics.get('propagated', 0) or 0)} propagated | "
            f"{int(metrics.get('rejected', 0) or 0)} dismissed"
        )
        tooltip = "\n".join([line for line in tooltip_lines if line is not None])
        for widget in (
            getattr(self, "next_query_card", None),
            getattr(self, "lbl_next_query_title", None),
            getattr(self, "lbl_next_query_summary", None),
            getattr(self, "lbl_next_query_reason", None),
            getattr(self, "lbl_next_query_evidence", None),
            getattr(self, "btn_next_query_focus", None),
            getattr(self, "btn_next_query_apply", None),
            getattr(self, "btn_next_query_reject", None),
        ):
            if widget is not None:
                widget.setToolTip(tooltip)

    def _record_safe_execution_metric(self, kind: str, amount: int = 1) -> None:
        key = str(kind or "").strip().lower()
        if not key:
            return
        self._safe_execution_metrics[key] = int(
            self._safe_execution_metrics.get(key, 0) or 0
        ) + int(amount)

    def _reset_query_session_metrics(self) -> None:
        self._query_metrics = {
            "presented": 0,
            "focused": 0,
            "accepted": 0,
            "propagated": 0,
            "rejected": 0,
            "hand_conditioned_presented": 0,
            "hand_conditioned_focused": 0,
            "hand_conditioned_accepted": 0,
            "hand_conditioned_propagated": 0,
            "hand_conditioned_rejected": 0,
        }
        self._next_best_query = None
        self._current_query_candidates = []
        self._next_best_query_id = ""
        self._next_best_query_presented_at = 0.0
        self._safe_execution_metrics = {
            "precheck_blocked": 0,
            "violations": 0,
            "rollbacks": 0,
        }
        self._update_query_session_metrics()

    def _current_annotator_id(self) -> str:
        value = str(getattr(self, "validator_name", "") or "").strip()
        if value:
            return value
        for env_key in ("CVHCI_ANNOTATOR", "USERNAME", "USER"):
            try:
                val = str(os.environ.get(env_key, "") or "").strip()
            except Exception:
                val = ""
            if val:
                return val
        return "unknown_annotator"

    def _update_query_session_metrics_unused(self) -> None:
        if not hasattr(self, "lbl_next_query_metrics"):
            return
        metrics = dict(getattr(self, "_query_metrics", {}) or {})
        shown = int(metrics.get("presented", 0) or 0)
        accepted = int(metrics.get("accepted", 0) or 0)
        propagated = int(metrics.get("propagated", 0) or 0)
        rejected = int(metrics.get("rejected", 0) or 0)
        focused = int(metrics.get("focused", 0) or 0)
        self.lbl_next_query_metrics.setText(
            f"Session: {shown} shown · {focused} focused · {accepted} accepted · {propagated} propagated · {rejected} rejected"
        )

    def _update_query_session_metrics(self) -> None:
        if not hasattr(self, "lbl_next_query_metrics"):
            return
        metrics = dict(getattr(self, "_query_metrics", {}) or {})
        shown = int(metrics.get("presented", 0) or 0)
        accepted = int(metrics.get("accepted", 0) or 0)
        propagated = int(metrics.get("propagated", 0) or 0)
        rejected = int(metrics.get("rejected", 0) or 0)
        focused = int(metrics.get("focused", 0) or 0)
        hc_shown = int(metrics.get("hand_conditioned_presented", 0) or 0)
        hc_accepted = int(metrics.get("hand_conditioned_accepted", 0) or 0)
        hc_propagated = int(metrics.get("hand_conditioned_propagated", 0) or 0)
        hc_rejected = int(metrics.get("hand_conditioned_rejected", 0) or 0)
        current_query = dict(getattr(self, "_next_best_query", {}) or {})
        current_source = self._query_source_label(current_query)
        current_hand_conditioned = self._query_is_hand_conditioned(current_query)
        self.lbl_next_query_metrics.setText(
            "Session: "
            f"{shown} shown | {focused} focused | {accepted} accepted | {propagated} propagated | {rejected} rejected"
            f" | HC {hc_shown} shown | HC {hc_accepted} accepted | HC {hc_propagated} propagated | HC {hc_rejected} rejected"
            + (
                f" | Current: hand_conditioned={'true' if current_hand_conditioned else 'false'} | source={current_source}"
                if current_query
                else ""
            )
        )

    def _bump_query_state_revision(self) -> None:
        self._query_state_revision = int(getattr(self, "_query_state_revision", 0) or 0) + 1
        self._dismissed_query_ids = {}

    def _mark_query_calibration_dirty(self) -> None:
        self._query_calibration_dirty = True

    def _candidate_query_log_paths(self) -> List[str]:
        paths: List[str] = []
        seen = set()

        def _add_path(path: str) -> None:
            raw = str(path or "").strip()
            if not raw:
                return
            try:
                norm = os.path.abspath(raw)
            except Exception:
                norm = raw
            if norm in seen or not os.path.isfile(norm):
                return
            seen.add(norm)
            paths.append(norm)

        annotation_path = str(getattr(self, "current_annotation_path", "") or "").strip()
        if annotation_path:
            base = os.path.splitext(annotation_path)[0]
            _add_path(base + ".ops.log.csv")
            _add_path(base + ".validation.ops.log.csv")

        base_dir = ""
        if annotation_path:
            base_dir = os.path.dirname(annotation_path)
        elif self.video_path:
            base_dir = os.path.dirname(self.video_path)
        if base_dir and os.path.isdir(base_dir):
            try:
                recent = []
                for name in os.listdir(base_dir):
                    if not str(name).lower().endswith(".ops.log.csv"):
                        continue
                    fp = os.path.join(base_dir, name)
                    try:
                        mtime = os.path.getmtime(fp)
                    except Exception:
                        mtime = 0.0
                    recent.append((mtime, fp))
                recent.sort(reverse=True)
                for _mtime, fp in recent[:12]:
                    _add_path(fp)
            except Exception:
                pass
        return paths

    def _build_query_calibrator(self) -> HOIEmpiricalCalibrator:
        live_rows: List[Dict[str, Any]] = []
        for logger in (
            getattr(self, "op_logger", None),
            getattr(self, "validation_op_logger", None),
        ):
            if logger is None or not hasattr(logger, "rows"):
                continue
            try:
                live_rows.extend(list(logger.rows() or []))
            except Exception:
                continue
        return HOIEmpiricalCalibrator.from_sources(
            csv_paths=self._candidate_query_log_paths(),
            live_rows=live_rows,
        )

    def _query_calibrator_signature_value(self) -> tuple:
        path_meta = []
        for fp in self._candidate_query_log_paths():
            try:
                path_meta.append((fp, os.path.getmtime(fp), os.path.getsize(fp)))
            except Exception:
                path_meta.append((fp, 0.0, 0))
        live_count = 0
        for logger in (
            getattr(self, "op_logger", None),
            getattr(self, "validation_op_logger", None),
        ):
            if logger is None:
                continue
            try:
                if hasattr(logger, "_rows"):
                    live_count += len(getattr(logger, "_rows") or [])
                elif hasattr(logger, "rows"):
                    live_count += len(logger.rows() or [])
            except Exception:
                continue
        return (
            tuple(path_meta),
            int(live_count),
            str(getattr(self, "current_annotation_path", "") or ""),
            str(self.video_path or ""),
        )

    def _ensure_query_calibrator(self) -> HOIEmpiricalCalibrator:
        signature = self._query_calibrator_signature_value()
        if (
            self._query_calibrator is None
            or self._query_calibration_dirty
            or signature != self._query_calibrator_signature
        ):
            self._query_calibrator = self._build_query_calibrator()
            self._query_calibrator_signature = signature
            self._query_calibration_dirty = False
        return self._query_calibrator

    def _build_hoi_query_rows(self) -> List[dict]:
        rows: List[dict] = []
        graph = build_hoi_event_graph(
            self.events,
            video_path=self.video_path,
            annotation_path="",
            actors_config=self.actors_config,
            assembly_timeline=self._assembly_data(),
        )
        consistency_by_key: Dict[tuple, List[dict]] = {}
        for item in list(graph.get("consistency_flags", []) or []):
            if not isinstance(item, dict):
                continue
            key = (item.get("event_id"), item.get("actor_id") or item.get("hand"))
            consistency_by_key.setdefault(key, []).append(dict(item))

        for event in self.events:
            event_id = event.get("event_id")
            event_candidates = self._normalize_videomae_candidates(
                event.get("videomae_top5")
            )
            event_frames = event.get("frames")
            if not isinstance(event_frames, (list, tuple)) or len(event_frames) < 2:
                event_frames = self._compute_event_frames(event)
            event_start_frame = event_frames[0] if isinstance(event_frames, (list, tuple)) else None
            event_end_frame = event_frames[1] if isinstance(event_frames, (list, tuple)) else None
            for actor in self.actors_config:
                hand_key = actor["id"]
                hand_data = event.get("hoi_data", {}).get(hand_key, {}) or {}
                self._hydrate_hand_annotation_state(
                    hand_data, default_source="loaded_annotation"
                )
                state = copy.deepcopy(hand_data.get("_field_state", {}) or {})
                suggestions = copy.deepcopy(
                    hand_data.get("_field_suggestions", {}) or {}
                )
                start = hand_data.get("interaction_start")
                onset = hand_data.get("functional_contact_onset")
                end = hand_data.get("interaction_end")
                onset_state = dict((state.get("functional_contact_onset") or {}))
                onset_suggestion = dict(
                    (suggestions.get("functional_contact_onset") or {})
                )
                track_prior = (
                    self._handtrack_segment_prior(hand_key, start, end)
                    if start is not None and end is not None
                    else {}
                )
                onset_source = str(
                    onset_state.get("source")
                    or onset_suggestion.get("source")
                    or (
                        track_prior.get("source")
                        if onset is None and track_prior
                        else ""
                    )
                    or ""
                ).strip()
                verb = hand_data.get("verb")
                target = self._hand_noun_object_id(hand_data)
                instrument = self._hand_instrument_object_id(hand_data)
                anomaly_label = self._normalize_anomaly_label(hand_data.get("anomaly_label"))
                has_anomaly = self._is_anomalous_label(anomaly_label)
                noun_required = self._noun_required_for_verb(verb)
                allowed_noun_ids = self._allowed_noun_ids_for_verb(verb)
                has_info = any(
                    (
                        start is not None,
                        onset is not None,
                        end is not None,
                        bool(str(verb or "").strip()),
                        target is not None,
                        instrument is not None,
                        has_anomaly,
                        bool(suggestions),
                    )
                )
                if not has_info:
                    continue

                sparse_evidence = self._sparse_evidence_summary(hand_data)
                bbox_errors = list(self._validate_integrity(hand_data) or [])
                object_candidates = self._collect_event_object_candidates(
                    hand_key,
                    hand_data,
                )
                if self._noun_only_mode:
                    object_candidates = filter_allowed_object_candidates(
                        object_candidates,
                        allowed_noun_ids,
                    )
                rows.append(
                    {
                        "event_id": event_id,
                        "hand": hand_key,
                        "annotator_id": self._current_annotator_id(),
                        "interaction_start": start,
                        "functional_contact_onset": onset,
                        "interaction_end": end,
                        "event_start_frame": event_start_frame,
                        "event_end_frame": event_end_frame,
                        "verb": verb,
                        "target_object_id": target,
                        "noun_object_id": target,
                        "instrument_object_id": instrument,
                        "tool_object_id": instrument,
                        "anomaly_label": anomaly_label,
                        "has_anomaly": bool(has_anomaly),
                        "field_state": state,
                        "field_suggestions": suggestions,
                        "bbox_errors": bbox_errors,
                        "sparse_evidence_state": copy.deepcopy(
                            sparse_evidence.get("state", {}) or {}
                        ),
                        "sparse_evidence_summary": {
                            "expected": int(sparse_evidence.get("expected", 0) or 0),
                            "confirmed": int(
                                sparse_evidence.get("confirmed", 0) or 0
                            ),
                            "missing": int(sparse_evidence.get("missing", 0) or 0),
                            "blocked": int(sparse_evidence.get("blocked", 0) or 0),
                        },
                        "videomae_candidates": list(event_candidates or []),
                        "videomae_meta": copy.deepcopy(event.get("videomae_meta", {}) or {}),
                        "noun_only_mode": bool(self._noun_only_mode),
                        "noun_required": bool(noun_required),
                        "allowed_noun_ids": list(allowed_noun_ids or []),
                        "onset_band": (
                            dict(track_prior.get("onset_band") or {})
                            if onset is None and track_prior
                            else (
                                build_onset_band(
                                    start,
                                    end,
                                    onset_frame=onset,
                                    onset_status=(state.get("functional_contact_onset") or {}).get("status"),
                                )
                                if start is not None and end is not None
                                else {}
                            )
                        ),
                        "onset_source": onset_source,
                        "handtrack_prior": copy.deepcopy(track_prior or {}),
                        "object_candidates": object_candidates,
                        "noun_source_decision": copy.deepcopy(
                            hand_data.get("_noun_source_decision", {}) or {}
                        ),
                        "consistency_flags": list(
                            consistency_by_key.get((event_id, hand_key), []) or []
                        ),
                    }
                )
        return rows

    def _update_next_best_query_panel(self) -> None:
        if not hasattr(self, "lbl_next_query_summary"):
            return
        if not self._semantic_assist_enabled():
            self._next_best_query = None
            self._next_best_query_id = ""
            self._next_best_query_presented_at = 0.0
            self._current_query_candidates = []
            self.lbl_next_query_title.setText("Next Best Query")
            self.lbl_next_query_summary.setText("Manual mode: assistive recommendations are off.")
            self.lbl_next_query_reason.setText(
                "Manual mode keeps suggestions off so the event is filled directly by hand."
            )
            self.lbl_next_query_evidence.setText(
                "Use the timeline, Quick Edit, and manual boxes directly in this mode."
            )
            self._set_status_chip(
                getattr(self, "lbl_next_query_surface_chip", None), "Manual", "neutral"
            )
            self._set_status_chip(
                getattr(self, "lbl_next_query_action_chip", None), "Off", "neutral"
            )
            self._set_status_chip(
                getattr(self, "lbl_next_query_form_chip", None), "How --", "neutral"
            )
            self._set_status_chip(
                getattr(self, "lbl_next_query_authority_chip", None), "Authority --", "neutral"
            )
            self._set_status_chip(
                getattr(self, "lbl_next_query_score_chip", None), "VOI --", "neutral"
            )
            self._set_status_chip(
                getattr(self, "lbl_next_query_prop_chip", None), "Prop --", "neutral"
            )
            self._set_status_chip(
                getattr(self, "lbl_next_query_cost_chip", None), "Cost --", "neutral"
            )
            self._set_status_chip(
                getattr(self, "lbl_next_query_risk_chip", None), "Risk --", "neutral"
            )
            self._set_status_card_tone(getattr(self, "next_query_card", None), "neutral")
            self.btn_next_query_focus.setEnabled(False)
            self.btn_next_query_apply.setEnabled(False)
            self.btn_next_query_reject.setEnabled(False)
            self.btn_next_query_focus.setText("Focus")
            self.btn_next_query_apply.setText("Apply")
            self._set_next_query_compact_tooltip(
                {},
                summary="Manual mode",
                reason="Assistive query/controller outputs are disabled.",
                policy_reason="",
                query_source="manual_mode",
                query_hand_conditioned=False,
                surface="",
                action_kind="",
                interaction_form="",
                authority_level="",
                voi=0.0,
                propagation=0.0,
                cost=0.0,
                risk=0.0,
                evidence_expected=0,
                evidence_confirmed=0,
                evidence_missing=0,
            )
            self._update_inline_event_editor()
            self._update_query_session_metrics()
            return
        queries = build_runtime_query_candidates(
            self._build_hoi_query_rows(),
            selected_event_id=self.selected_event_id,
            selected_hand=self.selected_hand_label,
            calibrator=self._ensure_query_calibrator(),
            authority_policy=getattr(self, "_authority_policy_config", None),
        )
        self._current_query_candidates = list(queries or [])
        active_revision = int(getattr(self, "_query_state_revision", 0) or 0)
        queries = [
            dict(item)
            for item in list(queries or [])
            if int((getattr(self, "_dismissed_query_ids", {}) or {}).get(str(item.get("query_id") or "").strip(), -1)) != active_revision
        ]
        queries = [dict(item) for item in list(queries or []) if self._query_is_actionable(item)]
        self._current_query_candidates = list(queries or [])
        best = dict(queries[0]) if queries else None
        self._next_best_query = best
        if not best:
            self._next_best_query_id = ""
            self._next_best_query_presented_at = 0.0
            self.lbl_next_query_title.setText("Next Best Query")
            self.lbl_next_query_summary.setText("No recommendation yet.")
            self.lbl_next_query_reason.setText(
                "Create an event and mark a rough start/end to unlock onset and semantic suggestions."
            )
            self.lbl_next_query_evidence.setText(
                "Once a local event segment exists, the controller will recommend the next step here."
            )
            self._set_status_chip(
                getattr(self, "lbl_next_query_surface_chip", None), "Idle", "neutral"
            )
            self._set_status_chip(
                getattr(self, "lbl_next_query_action_chip", None), "Waiting", "neutral"
            )
            self._set_status_chip(
                getattr(self, "lbl_next_query_form_chip", None), "How --", "neutral"
            )
            self._set_status_chip(
                getattr(self, "lbl_next_query_authority_chip", None), "Authority --", "neutral"
            )
            self._set_status_chip(
                getattr(self, "lbl_next_query_score_chip", None), "VOI --", "neutral"
            )
            self._set_status_chip(
                getattr(self, "lbl_next_query_prop_chip", None), "Prop --", "neutral"
            )
            self._set_status_chip(
                getattr(self, "lbl_next_query_cost_chip", None), "Cost --", "neutral"
            )
            self._set_status_chip(
                getattr(self, "lbl_next_query_risk_chip", None), "Risk --", "neutral"
            )
            self._set_status_card_tone(getattr(self, "next_query_card", None), "neutral")
            self.btn_next_query_focus.setEnabled(False)
            self.btn_next_query_apply.setEnabled(False)
            self.btn_next_query_reject.setEnabled(False)
            self.btn_next_query_focus.setText("Focus")
            self.btn_next_query_apply.setText("Apply")
            self._set_next_query_compact_tooltip(
                {},
                summary="No recommendation yet.",
                reason="Create an event and mark a rough start/end to unlock onset and semantic suggestions.",
                policy_reason="",
                query_source="query_controller",
                query_hand_conditioned=False,
                surface="",
                action_kind="",
                interaction_form="",
                authority_level="",
                voi=0.0,
                propagation=0.0,
                cost=0.0,
                risk=0.0,
                evidence_expected=0,
                evidence_confirmed=0,
                evidence_missing=0,
            )
            self._update_inline_event_editor()
            self._update_query_session_metrics()
            return

        surface = str(best.get("surface") or "review").strip().title()
        event_id = best.get("event_id")
        hand = self._get_actor_short_label(best.get("hand"))
        query_id = str(best.get("query_id") or "").strip()
        query_source = self._query_source_label(best)
        query_hand_conditioned = self._query_is_hand_conditioned(best)
        if query_id and query_id != self._next_best_query_id:
            self._next_best_query_id = query_id
            self._next_best_query_presented_at = time.time()
            self._record_query_metric("presented")
            if query_hand_conditioned:
                self._record_query_metric("hand_conditioned_presented")
            self._log(
                "hoi_query_present",
                query_id=query_id,
                query_type=best.get("query_type"),
                event_id=event_id,
                hand=best.get("hand"),
                surface=best.get("surface"),
                field=best.get("field_name"),
                target_frame=best.get("target_frame"),
                target_slot=best.get("target_slot"),
                voi_score=best.get("voi_score"),
                propagation_gain=best.get("propagation_gain"),
                human_cost_est=best.get("human_cost_est"),
                overwrite_risk=best.get("overwrite_risk"),
                empirical_cost_est=best.get("empirical_cost_est"),
                empirical_cost_ms=best.get("empirical_cost_ms"),
                acceptance_prob_est=best.get("acceptance_prob_est"),
                empirical_support_n=best.get("empirical_support_n"),
                calibration_source=best.get("calibration_source"),
                calibration_note=best.get("calibration_note"),
                authority_policy_reason=best.get("authority_policy_reason"),
                authority_policy_code=best.get("authority_policy_code"),
                authority_policy_name=best.get("authority_policy_name"),
                action_kind=best.get("action_kind"),
                interaction_form=best.get("interaction_form"),
                authority_level=best.get("authority_level"),
                completion_fields=best.get("completion_fields"),
                suggested_source=best.get("suggested_source"),
                source=query_source,
                hand_conditioned=query_hand_conditioned,
                handtrack_prior=best.get("handtrack_prior"),
                suggested_confidence=best.get("suggested_confidence"),
                calibrated_reliability=best.get("calibrated_reliability"),
                noun_primary_source=best.get("noun_primary_source"),
                noun_primary_family=best.get("noun_primary_family"),
                noun_source_margin=best.get("noun_source_margin"),
                semantic_source_acceptance_est=best.get("semantic_source_acceptance_est"),
                semantic_source_score=best.get("semantic_source_score"),
                semantic_source_support=best.get("semantic_source_support"),
                detector_source_acceptance_est=best.get("detector_source_acceptance_est"),
                detector_source_score=best.get("detector_source_score"),
                detector_source_support=best.get("detector_source_support"),
                noun_source_decision_basis=best.get("noun_source_decision_basis"),
            )
        summary = str(best.get("summary") or "").strip()
        reason = str(best.get("reason") or "").strip()
        policy_reason = str(
            best.get("authority_policy_reason") or best.get("calibration_note") or ""
        ).strip()
        voi = float(best.get("voi_score", 0.0) or 0.0)
        propagation = float(best.get("propagation_gain", 0.0) or 0.0)
        cost = float(best.get("human_cost_est", 0.0) or 0.0)
        risk = float(best.get("overwrite_risk", 0.0) or 0.0)
        action_kind = str(best.get("action_kind") or "").strip().lower()
        interaction_form = str(best.get("interaction_form") or "").strip().lower()
        authority_level = str(best.get("authority_level") or "").strip().lower()
        evidence_summary = dict(best.get("sparse_evidence_summary") or {})
        evidence_expected = int(evidence_summary.get("expected", 0) or 0)
        evidence_confirmed = int(evidence_summary.get("confirmed", 0) or 0)
        evidence_missing = int(evidence_summary.get("missing", 0) or 0)
        location = f"Event {event_id} {hand}" if event_id is not None else hand
        self.lbl_next_query_title.setText(f"Next Best Query  {location}".strip())
        self.lbl_next_query_summary.setText(summary or "Review the next supervision step.")
        self.lbl_next_query_reason.setText(
            reason or "Controller-selected next supervision step."
        )
        self.lbl_next_query_evidence.setText(
            self._next_query_compact_evidence(
                best,
                query_source=query_source,
                query_hand_conditioned=query_hand_conditioned,
                evidence_expected=evidence_expected,
                evidence_confirmed=evidence_confirmed,
                evidence_missing=evidence_missing,
            )
        )
        self._set_status_chip(
            getattr(self, "lbl_next_query_surface_chip", None),
            surface,
            self._query_surface_chip_tone(surface),
        )
        self._set_status_chip(
            getattr(self, "lbl_next_query_action_chip", None),
            self._next_query_compact_status(best),
            self._query_action_chip_tone(action_kind),
        )
        self._set_status_chip(
            getattr(self, "lbl_next_query_form_chip", None),
            self._query_form_label(interaction_form),
            self._query_form_chip_tone(interaction_form),
        )
        self._set_status_chip(
            getattr(self, "lbl_next_query_authority_chip", None),
            self._query_authority_label(authority_level),
            self._query_authority_chip_tone(authority_level),
        )
        self._set_status_chip(
            getattr(self, "lbl_next_query_score_chip", None),
            f"VOI {voi:.2f}",
            "neutral",
        )
        self._set_status_chip(
            getattr(self, "lbl_next_query_prop_chip", None),
            f"Prop {propagation:.2f}",
            "ok" if propagation >= 0.85 else "neutral",
        )
        self._set_status_chip(
            getattr(self, "lbl_next_query_cost_chip", None),
            f"Cost {cost:.2f}",
            "warn" if cost >= 0.5 else "neutral",
        )
        self._set_status_chip(
            getattr(self, "lbl_next_query_risk_chip", None),
            f"Risk {risk:.2f}",
            "danger" if risk >= 0.3 else "neutral",
        )
        self._set_status_card_tone(
            getattr(self, "next_query_card", None),
            self._query_surface_tone(surface),
        )
        self.btn_next_query_focus.setEnabled(True)
        self.btn_next_query_focus.setText(self._query_focus_button_label(best))
        self.btn_next_query_reject.setEnabled(True)
        can_apply = bool(best.get("safe_apply")) or (
            str(best.get("action_kind") or "").strip().lower() == "suggest"
            and best.get("suggested_value") is not None
        )
        self.btn_next_query_apply.setEnabled(True)
        if not can_apply:
            self.btn_next_query_apply.setText("Review Manually")
        elif best.get("apply_mode") == "confirm_current":
            self.btn_next_query_apply.setText(self._query_confirm_button_label(best))
        elif best.get("apply_mode") == "apply_completion_bundle":
            self.btn_next_query_apply.setText("Apply Local Completion")
        elif str(best.get("action_kind") or "").strip().lower() == "suggest":
            self.btn_next_query_apply.setText("Accept Suggestion")
        else:
            self.btn_next_query_apply.setText("Apply Suggestion")
        self._set_next_query_compact_tooltip(
            best,
            summary=summary or "Review the next supervision step.",
            reason=reason or "Controller-selected next supervision step.",
            policy_reason=policy_reason,
            query_source=query_source,
            query_hand_conditioned=query_hand_conditioned,
            surface=surface,
            action_kind=action_kind,
            interaction_form=interaction_form,
            authority_level=authority_level,
            voi=voi,
            propagation=propagation,
            cost=cost,
            risk=risk,
            evidence_expected=evidence_expected,
            evidence_confirmed=evidence_confirmed,
            evidence_missing=evidence_missing,
        )
        self._route_query_target_control(
            best,
            set_keyboard_focus=False,
            allow_select_event=False,
        )
        self._update_inline_event_editor()
        self._update_query_session_metrics()

    def _focus_query_candidate(self, query: Optional[dict]) -> None:
        if not query:
            return
        event_id = query.get("event_id")
        hand = query.get("hand")
        query_source = self._query_source_label(query)
        query_hand_conditioned = self._query_is_hand_conditioned(query)
        surface = str(query.get("surface") or "").strip().lower()
        if event_id is not None and hand:
            self._set_selected_event(int(event_id), str(hand))
        self._route_query_target_control(
            query,
            set_keyboard_focus=True,
            allow_select_event=False,
        )
        frame = query.get("target_frame")
        if frame is not None and self.player.cap:
            try:
                target = int(frame)
                self.player.seek(target)
                self._refresh_boxes_for_frame(target)
                self._set_frame_controls(target)
            except Exception:
                pass
        latency_ms = None
        query_id = str(query.get("query_id") or "").strip()
        if query_id and query_id == getattr(self, "_next_best_query_id", ""):
            presented_at = float(getattr(self, "_next_best_query_presented_at", 0.0) or 0.0)
            if presented_at > 0:
                latency_ms = int(max(0.0, (time.time() - presented_at) * 1000.0))
        self._log(
            "hoi_query_focus",
            query_id=query_id,
            query_type=query.get("query_type"),
            event_id=event_id,
            hand=hand,
            surface=surface,
            field=query.get("field_name"),
            target_frame=query.get("target_frame"),
            target_slot=query.get("target_slot"),
            voi_score=query.get("voi_score"),
            propagation_gain=query.get("propagation_gain"),
            human_cost_est=query.get("human_cost_est"),
            overwrite_risk=query.get("overwrite_risk"),
            empirical_cost_est=query.get("empirical_cost_est"),
            empirical_cost_ms=query.get("empirical_cost_ms"),
            acceptance_prob_est=query.get("acceptance_prob_est"),
            empirical_support_n=query.get("empirical_support_n"),
            calibration_source=query.get("calibration_source"),
            authority_policy_reason=query.get("authority_policy_reason"),
            authority_policy_code=query.get("authority_policy_code"),
            authority_policy_name=query.get("authority_policy_name"),
            action_kind=query.get("action_kind"),
            interaction_form=query.get("interaction_form"),
            authority_level=query.get("authority_level"),
            completion_fields=query.get("completion_fields"),
            suggested_source=query.get("suggested_source"),
            suggested_confidence=query.get("suggested_confidence"),
            calibrated_reliability=query.get("calibrated_reliability"),
            source=query_source,
            hand_conditioned=query_hand_conditioned,
            handtrack_prior=query.get("handtrack_prior"),
            noun_primary_source=query.get("noun_primary_source"),
            noun_primary_family=query.get("noun_primary_family"),
            noun_source_margin=query.get("noun_source_margin"),
            semantic_source_acceptance_est=query.get("semantic_source_acceptance_est"),
            semantic_source_score=query.get("semantic_source_score"),
            semantic_source_support=query.get("semantic_source_support"),
            detector_source_acceptance_est=query.get("detector_source_acceptance_est"),
            detector_source_score=query.get("detector_source_score"),
            detector_source_support=query.get("detector_source_support"),
            noun_source_decision_basis=query.get("noun_source_decision_basis"),
            query_latency_ms=latency_ms,
        )
        self._record_query_metric("focused")
        if query_hand_conditioned:
            self._record_query_metric("hand_conditioned_focused")

    def _focus_next_best_query(self) -> None:
        self._focus_query_candidate(getattr(self, "_next_best_query", None))

    def _reject_next_best_query(self) -> None:
        query = getattr(self, "_next_best_query", None)
        if not query:
            return
        query_source = self._query_source_label(query)
        query_hand_conditioned = self._query_is_hand_conditioned(query)
        query_id = str(query.get("query_id") or "").strip()
        if query_id:
            self._dismissed_query_ids[query_id] = int(getattr(self, "_query_state_revision", 0) or 0)
        latency_ms = None
        if query_id and query_id == getattr(self, "_next_best_query_id", ""):
            presented_at = float(getattr(self, "_next_best_query_presented_at", 0.0) or 0.0)
            if presented_at > 0:
                latency_ms = int(max(0.0, (time.time() - presented_at) * 1000.0))
        self._log(
            "hoi_query_reject",
            query_id=query_id,
            query_type=query.get("query_type"),
            event_id=query.get("event_id"),
            hand=query.get("hand"),
            surface=query.get("surface"),
            field=query.get("field_name"),
            action_kind=query.get("action_kind"),
            interaction_form=query.get("interaction_form"),
            authority_level=query.get("authority_level"),
            voi_score=query.get("voi_score"),
            empirical_cost_est=query.get("empirical_cost_est"),
            empirical_cost_ms=query.get("empirical_cost_ms"),
            acceptance_prob_est=query.get("acceptance_prob_est"),
            empirical_support_n=query.get("empirical_support_n"),
            calibration_source=query.get("calibration_source"),
            authority_policy_reason=query.get("authority_policy_reason"),
            authority_policy_code=query.get("authority_policy_code"),
            authority_policy_name=query.get("authority_policy_name"),
            suggested_source=query.get("suggested_source"),
            source=query_source,
            hand_conditioned=query_hand_conditioned,
            handtrack_prior=query.get("handtrack_prior"),
            suggested_confidence=query.get("suggested_confidence"),
            calibrated_reliability=query.get("calibrated_reliability"),
            noun_primary_source=query.get("noun_primary_source"),
            noun_primary_family=query.get("noun_primary_family"),
            noun_source_margin=query.get("noun_source_margin"),
            semantic_source_acceptance_est=query.get("semantic_source_acceptance_est"),
            semantic_source_score=query.get("semantic_source_score"),
            semantic_source_support=query.get("semantic_source_support"),
            detector_source_acceptance_est=query.get("detector_source_acceptance_est"),
            detector_source_score=query.get("detector_source_score"),
            detector_source_support=query.get("detector_source_support"),
            noun_source_decision_basis=query.get("noun_source_decision_basis"),
            query_latency_ms=latency_ms,
        )
        self._mark_query_calibration_dirty()
        self._record_query_metric("rejected")
        if query_hand_conditioned:
            self._record_query_metric("hand_conditioned_rejected")
        self._update_next_best_query_panel()

    def _safe_execution_precheck(
        self,
        query: Optional[dict],
        hand_data: Optional[dict],
    ) -> tuple:
        if not isinstance(query, dict) or not isinstance(hand_data, dict):
            return False, "Invalid execution context.", {"stage": "precheck"}
        action_kind = str(query.get("action_kind") or "").strip().lower()
        authority_level = str(query.get("authority_level") or "").strip().lower()
        apply_mode = str(query.get("apply_mode") or "").strip()
        field_name = self._canonical_query_field_name(query.get("field_name"))
        overwrite_risk = float(query.get("overwrite_risk", 0.0) or 0.0)
        reliability = float(
            query.get(
                "calibrated_reliability",
                query.get("acceptance_prob_est", 0.0),
            )
            or 0.0
        )
        support = int(query.get("empirical_support_n", 0) or 0)
        authority_thresholds = dict(query.get("authority_policy_thresholds") or {})
        safe_local_max_risk = float(
            authority_thresholds.get("safe_local_max_risk", 0.18) or 0.18
        )
        safe_local_min_reliability = float(
            authority_thresholds.get("safe_local_min_reliability", 0.78) or 0.78
        )
        safe_local_min_support_if_available = int(
            authority_thresholds.get("safe_local_min_support_if_available", 2) or 2
        )

        if action_kind == "query":
            return False, "Manual-only queries cannot be executed automatically.", {
                "stage": "precheck",
                "reason": "manual_only",
            }
        if authority_level == "human_only":
            return False, "Authority policy requires direct manual review.", {
                "stage": "precheck",
                "reason": "authority_human_only",
            }
        if apply_mode == "apply_suggestion":
            if query.get("suggested_value") is None:
                return False, "No suggested value is available to apply.", {
                    "stage": "precheck",
                    "reason": "missing_suggestion_value",
                }
            if get_field_state(hand_data, field_name).get("status") == "confirmed":
                return False, "Confirmed fields are immutable.", {
                    "stage": "precheck",
                    "reason": "confirmed_immutable",
                }
        if apply_mode == "apply_completion_bundle":
            completion_fields = list(query.get("completion_fields") or [])
            if not completion_fields:
                return False, "No completion bundle is available.", {
                    "stage": "precheck",
                    "reason": "missing_bundle",
                }
            for item in completion_fields:
                if not isinstance(item, dict):
                    continue
                bundle_field = self._canonical_query_field_name(item.get("field_name"))
                if not bundle_field:
                    continue
                if get_field_state(hand_data, bundle_field).get("status") == "confirmed":
                    return False, f"Field '{bundle_field}' is already confirmed.", {
                        "stage": "precheck",
                        "reason": "confirmed_immutable",
                        "field": bundle_field,
                    }
                if not bool(item.get("safe_to_apply")):
                    return False, f"Field '{bundle_field}' is not marked safe to apply.", {
                        "stage": "precheck",
                        "reason": "unsafe_bundle_field",
                        "field": bundle_field,
                    }
        if authority_level == "safe_local":
            if overwrite_risk > safe_local_max_risk:
                return False, "Overwrite risk is too high for safe-local execution.", {
                    "stage": "precheck",
                    "reason": "risk_too_high",
                }
            if reliability < safe_local_min_reliability:
                return False, "Completion reliability is too low for safe-local execution.", {
                    "stage": "precheck",
                    "reason": "reliability_too_low",
                }
            if support > 0 and support < safe_local_min_support_if_available:
                return False, "Insufficient empirical support for safe-local authority.", {
                    "stage": "precheck",
                    "reason": "support_too_low",
                }
        return True, "", {"stage": "precheck", "reason": "ok"}

    def _safe_execution_postcheck(
        self,
        before_hand: Optional[dict],
        after_hand: Optional[dict],
        applied_fields: List[str],
    ) -> tuple:
        before = dict(before_hand or {})
        after = dict(after_hand or {})
        before_state = dict(before.get("_field_state") or {})
        after_state = dict(after.get("_field_state") or {})
        violations: List[str] = []
        for field_name, entry in before_state.items():
            status_before = str((entry or {}).get("status") or "").strip().lower()
            if status_before != "confirmed":
                continue
            before_value = before.get(field_name)
            after_value = after.get(field_name)
            after_status = str((after_state.get(field_name) or {}).get("status") or "").strip().lower()
            if before_value != after_value or after_status != "confirmed":
                violations.append(field_name)
        if violations:
            return False, f"Confirmed fields changed: {', '.join(violations)}", {
                "stage": "postcheck",
                "reason": "confirmed_field_changed",
                "fields": violations,
            }
        for field_name in list(applied_fields or []):
            after_status = str((after_state.get(field_name) or {}).get("status") or "").strip().lower()
            if after_status not in {"suggested", "confirmed"}:
                violations.append(field_name)
        if violations:
            return False, f"Applied fields did not remain reviewable: {', '.join(violations)}", {
                "stage": "postcheck",
                "reason": "applied_field_invalid",
                "fields": violations,
            }
        return True, "", {"stage": "postcheck", "reason": "ok"}

    def _apply_query_suggestion(self, query: Optional[dict]) -> bool:
        if not query:
            return False
        event_id = query.get("event_id")
        hand_key = query.get("hand")
        field_name = self._canonical_query_field_name(query.get("field_name"))
        action_kind = str(query.get("action_kind") or "").strip().lower()
        authority_level = str(query.get("authority_level") or "").strip().lower()
        apply_mode = str(query.get("apply_mode") or "").strip()
        query_source = self._query_source_label(query)
        query_hand_conditioned = self._query_is_hand_conditioned(query)
        if not apply_mode and action_kind == "suggest" and query.get("suggested_value") is not None:
            apply_mode = "apply_suggestion"
        can_apply = (
            apply_mode == "confirm_current"
            or bool(query.get("safe_apply"))
            or (
            action_kind == "suggest" and query.get("suggested_value") is not None
            )
        )
        if not can_apply:
            return False
        if event_id is None or not hand_key or not field_name:
            return False
        event = self._find_event_by_id(int(event_id))
        if not event:
            return False
        hand_data = event.get("hoi_data", {}).get(hand_key, {}) or {}
        self._ensure_hand_annotation_state(hand_data)
        draft = None
        if self.selected_event_id == int(event_id) and hand_key in self.event_draft:
            draft = self.event_draft.get(hand_key, {})
            self._ensure_hand_annotation_state(draft)

        ok_safe, safe_reason, safe_meta = self._safe_execution_precheck(query, hand_data)
        if not ok_safe:
            self._record_safe_execution_metric("precheck_blocked")
            self._log(
                "hoi_safe_execution_block",
                query_id=query.get("query_id"),
                query_type=query.get("query_type"),
                event_id=event_id,
                hand=hand_key,
                field=field_name,
                authority_level=authority_level,
                authority_policy_name=query.get("authority_policy_name"),
                authority_policy_code=query.get("authority_policy_code"),
                source=query_source,
                hand_conditioned=query_hand_conditioned,
                handtrack_prior=query.get("handtrack_prior"),
                reason=safe_meta.get("reason"),
                detail=safe_reason,
            )
            return False

        before_hand_snapshot = copy.deepcopy(hand_data)
        before_draft_snapshot = copy.deepcopy(draft) if isinstance(draft, dict) else None
        undo_pushed = False
        redo_snapshot = None

        def _ensure_query_undo() -> None:
            nonlocal undo_pushed, redo_snapshot
            if undo_pushed:
                return
            redo_snapshot = copy.deepcopy(getattr(self, "_hoi_redo_stack", []) or [])
            self._push_undo()
            undo_pushed = True

        def _rollback_query_undo() -> None:
            nonlocal undo_pushed, redo_snapshot
            if not undo_pushed:
                return
            if getattr(self, "_hoi_undo_stack", None):
                try:
                    self._hoi_undo_stack.pop()
                except Exception:
                    pass
            self._hoi_redo_stack = copy.deepcopy(redo_snapshot or [])
            undo_pushed = False

        applied = False
        applied_fields: List[str] = []
        if apply_mode == "confirm_current":
            state = get_field_state(hand_data, field_name)
            if state.get("status") != "confirmed":
                _ensure_query_undo()
                self._set_hand_field_state(
                    hand_data,
                    field_name,
                    source="query_confirm",
                    status="confirmed",
                )
                if isinstance(draft, dict):
                    self._set_hand_field_state(
                        draft,
                        field_name,
                        source="query_confirm",
                        status="confirmed",
                    )
                applied = True
                applied_fields = [field_name]
        elif apply_mode == "apply_suggestion":
            suggested_value = query.get("suggested_value")
            suggested_source = (
                str(query.get("suggested_source") or "").strip() or "query_controller"
            )
            reason = str(query.get("reason") or "").strip()
            confidence = query.get("suggested_confidence")
            applied_status = (
                "confirmed"
                if action_kind == "suggest" and authority_level == "human_confirm"
                else "suggested"
            )
            if get_field_state(hand_data, field_name).get("status") != "confirmed":
                _ensure_query_undo()
                self._suggest_hand_field(
                    hand_data,
                    field_name,
                    suggested_value,
                    source=suggested_source,
                    confidence=confidence,
                    reason=reason,
                    safe_to_apply=True,
                )
                applied = apply_field_suggestion(
                    hand_data,
                    field_name,
                    source=suggested_source,
                    as_status=applied_status,
                )
                if applied and isinstance(draft, dict):
                    self._suggest_hand_field(
                        draft,
                        field_name,
                        suggested_value,
                        source=suggested_source,
                        confidence=confidence,
                        reason=reason,
                        safe_to_apply=True,
                    )
                    apply_field_suggestion(
                        draft,
                        field_name,
                        source=suggested_source,
                        as_status=applied_status,
                    )
                applied_fields = [field_name]
        elif apply_mode == "apply_completion_bundle":
            completion_fields = list(query.get("completion_fields") or [])
            for item in completion_fields:
                if not isinstance(item, dict):
                    continue
                bundle_field = self._canonical_query_field_name(item.get("field_name"))
                if not bundle_field:
                    continue
                if get_field_state(hand_data, bundle_field).get("status") == "confirmed":
                    continue
                bundle_source = str(item.get("source") or "onset_local_completion").strip() or "onset_local_completion"
                bundle_reason = str(item.get("reason") or "").strip()
                bundle_confidence = item.get("confidence")
                bundle_value = item.get("value")
                if not applied:
                    _ensure_query_undo()
                self._suggest_hand_field(
                    hand_data,
                    bundle_field,
                    bundle_value,
                    source=bundle_source,
                    confidence=bundle_confidence,
                    reason=bundle_reason,
                    safe_to_apply=True,
                )
                if apply_field_suggestion(
                    hand_data,
                    bundle_field,
                    source=bundle_source,
                    as_status="suggested",
                ):
                    applied = True
                    applied_fields.append(bundle_field)
                    if isinstance(draft, dict):
                        self._suggest_hand_field(
                            draft,
                            bundle_field,
                            bundle_value,
                            source=bundle_source,
                            confidence=bundle_confidence,
                            reason=bundle_reason,
                            safe_to_apply=True,
                        )
                        apply_field_suggestion(
                            draft,
                            bundle_field,
                            source=bundle_source,
                            as_status="suggested",
                        )

        if not applied:
            _rollback_query_undo()
            return False

        ok_safe, safe_reason, safe_meta = self._safe_execution_postcheck(
            before_hand_snapshot,
            hand_data,
            applied_fields,
        )
        if not ok_safe:
            event.get("hoi_data", {})[hand_key] = before_hand_snapshot
            if isinstance(draft, dict) and before_draft_snapshot is not None:
                self.event_draft[hand_key] = before_draft_snapshot
            _rollback_query_undo()
            self._record_safe_execution_metric("violations")
            self._record_safe_execution_metric("rollbacks")
            self._log(
                "hoi_safe_execution_violation",
                query_id=query.get("query_id"),
                query_type=query.get("query_type"),
                event_id=event_id,
                hand=hand_key,
                field=field_name,
                authority_level=authority_level,
                authority_policy_name=query.get("authority_policy_name"),
                authority_policy_code=query.get("authority_policy_code"),
                source=query_source,
                hand_conditioned=query_hand_conditioned,
                handtrack_prior=query.get("handtrack_prior"),
                reason=safe_meta.get("reason"),
                detail=safe_reason,
            )
            self._log(
                "hoi_safe_execution_rollback",
                query_id=query.get("query_id"),
                event_id=event_id,
                hand=hand_key,
                source=query_source,
                hand_conditioned=query_hand_conditioned,
                handtrack_prior=query.get("handtrack_prior"),
                rolled_back_fields=applied_fields,
            )
            return False

        self._bump_query_state_revision()
        latency_ms = None
        query_id = str(query.get("query_id") or "").strip()
        if query_id and query_id == getattr(self, "_next_best_query_id", ""):
            presented_at = float(getattr(self, "_next_best_query_presented_at", 0.0) or 0.0)
            if presented_at > 0:
                latency_ms = int(max(0.0, (time.time() - presented_at) * 1000.0))
        self._sync_event_frames(event)
        refresh_videomae = any(
            name in ("interaction_start", "functional_contact_onset", "interaction_end", "verb")
            for name in list(applied_fields or [])
        ) or field_name in ("interaction_start", "functional_contact_onset", "interaction_end", "verb")
        if refresh_videomae:
            self._invalidate_videomae_candidates(int(event_id))
            event.pop("videomae_meta", None)
            event.pop("videomae_local_top5", None)
            event.pop("videomae_local_meta", None)
        if any(name in ("verb", "target_object_id", "noun_object_id") for name in applied_fields) or field_name in ("verb", "target_object_id", "noun_object_id"):
            if self.selected_event_id == int(event_id) and self.selected_hand_label == hand_key:
                self._load_hand_draft_to_ui(hand_key)
            if "verb" in applied_fields or field_name == "verb":
                self._update_action_top5_display(int(event_id))
        if self.selected_event_id == int(event_id) and self.selected_hand_label == hand_key:
            self._update_status_label()
        self._refresh_events()
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.refresh()
        if (
            self.selected_event_id == int(event_id)
            and self.selected_hand_label == hand_key
            and "functional_contact_onset" in [self._canonical_query_field_name(name) for name in list(applied_fields or [])]
        ):
            self._jump_to_selected_keyframe("functional_contact_onset")
            self._focus_visual_support_for_selected_event()
        resolve_kind = "safe_propagate"
        if action_kind == "suggest" and authority_level == "human_confirm":
            resolve_kind = "human_accept_suggestion"
        elif apply_mode == "confirm_current":
            resolve_kind = "human_confirm_current"
        self._log(
            "hoi_query_apply",
            query_id=query_id,
            query_type=query.get("query_type"),
            event_id=event_id,
            hand=hand_key,
            field=field_name,
            applied_fields=applied_fields,
            apply_mode=apply_mode,
            target_frame=query.get("target_frame"),
            target_slot=query.get("target_slot"),
            voi_score=query.get("voi_score"),
            propagation_gain=query.get("propagation_gain"),
            human_cost_est=query.get("human_cost_est"),
            overwrite_risk=query.get("overwrite_risk"),
            empirical_cost_est=query.get("empirical_cost_est"),
            empirical_cost_ms=query.get("empirical_cost_ms"),
            acceptance_prob_est=query.get("acceptance_prob_est"),
            empirical_support_n=query.get("empirical_support_n"),
            calibration_source=query.get("calibration_source"),
            authority_policy_reason=query.get("authority_policy_reason"),
            authority_policy_code=query.get("authority_policy_code"),
            authority_policy_name=query.get("authority_policy_name"),
            action_kind=query.get("action_kind"),
            interaction_form=query.get("interaction_form"),
            authority_level=query.get("authority_level"),
            completion_fields=query.get("completion_fields"),
            suggested_source=query.get("suggested_source"),
            source=query_source,
            hand_conditioned=query_hand_conditioned,
            handtrack_prior=query.get("handtrack_prior"),
            suggested_confidence=query.get("suggested_confidence"),
            calibrated_reliability=query.get("calibrated_reliability"),
            noun_primary_source=query.get("noun_primary_source"),
            noun_primary_family=query.get("noun_primary_family"),
            noun_source_margin=query.get("noun_source_margin"),
            semantic_source_acceptance_est=query.get("semantic_source_acceptance_est"),
            semantic_source_score=query.get("semantic_source_score"),
            semantic_source_support=query.get("semantic_source_support"),
            detector_source_acceptance_est=query.get("detector_source_acceptance_est"),
            detector_source_score=query.get("detector_source_score"),
            detector_source_support=query.get("detector_source_support"),
            noun_source_decision_basis=query.get("noun_source_decision_basis"),
            resolve_kind=resolve_kind,
            query_latency_ms=latency_ms,
        )
        self._mark_query_calibration_dirty()
        if refresh_videomae and self.selected_event_id == int(event_id):
            self._queue_action_label_refresh(int(event_id), delay_ms=90, force=True)
        if resolve_kind == "human_accept_suggestion":
            self._record_query_metric("accepted")
            if query_hand_conditioned:
                self._record_query_metric("hand_conditioned_accepted")
        elif apply_mode == "apply_completion_bundle":
            self._record_query_metric("propagated")
            if query_hand_conditioned:
                self._record_query_metric("hand_conditioned_propagated")
        else:
            self._record_query_metric("accepted")
            if query_hand_conditioned:
                self._record_query_metric("hand_conditioned_accepted")
        if apply_mode == "apply_completion_bundle" and applied_fields:
            self._log(
                "hoi_graph_propagate",
                query_id=query_id,
                event_id=event_id,
                hand=hand_key,
                propagated_fields=applied_fields,
                propagation_count=len(applied_fields),
                authority_level=query.get("authority_level"),
                interaction_form=query.get("interaction_form"),
                source=query_source,
                hand_conditioned=query_hand_conditioned,
                handtrack_prior=query.get("handtrack_prior"),
                noun_primary_source=query.get("noun_primary_source"),
                noun_primary_family=query.get("noun_primary_family"),
                noun_source_margin=query.get("noun_source_margin"),
                semantic_source_acceptance_est=query.get("semantic_source_acceptance_est"),
                semantic_source_score=query.get("semantic_source_score"),
                semantic_source_support=query.get("semantic_source_support"),
                detector_source_acceptance_est=query.get("detector_source_acceptance_est"),
                detector_source_score=query.get("detector_source_score"),
                detector_source_support=query.get("detector_source_support"),
                noun_source_decision_basis=query.get("noun_source_decision_basis"),
                query_latency_ms=latency_ms,
            )
            self._mark_query_calibration_dirty()
        if any(
            name
            in (
                "interaction_start",
                "functional_contact_onset",
                "interaction_end",
                "verb",
                "target_object_id",
                "noun_object_id",
            )
            for name in list(applied_fields or [])
        ):
            self._set_semantic_reinfer_hint(
                event_id,
                hand_key,
                hand_data,
                reason="query_apply",
                edited_fields=list(applied_fields or []),
            )
            self._record_semantic_feedback(
                event,
                hand_key,
                reason="query_apply",
                before_hand=before_hand_snapshot,
                accepted_fields=list(applied_fields or []),
                supervision_kind="accepted",
                resolve_kind=resolve_kind,
                authority_level=str(query.get("authority_level") or ""),
                query_id=query_id,
            )
            self._refresh_semantic_suggestions_for_event(int(event_id), event)
        return True

    def _apply_next_best_query(self) -> None:
        query = getattr(self, "_next_best_query", None)
        if self._apply_query_suggestion(query):
            return
        if query:
            self._focus_query_for_manual_review(query)
            return
        QMessageBox.information(
            self,
            "Next Best Query",
            "There is no active next-best query for the selected event.",
        )

    def _update_draw_mode_visibility(self) -> None:
        draw_widget = getattr(self, "draw_mode_widget", None)
        if draw_widget is not None:
            draw_widget.setVisible(bool(getattr(self, "chk_edit_boxes", None) and self.chk_edit_boxes.isChecked()))

    def _preferred_base_font_pt(self) -> float:
        app = QApplication.instance()
        base_font = app.font() if app is not None else self.font()
        try:
            size = float(base_font.pointSizeF())
        except Exception:
            size = float(base_font.pointSize() or 0)
        if size <= 0:
            size = 8.0
        scaled = size * self._normalized_ui_scale()
        return max(6.8, min(8.4, scaled))

    def _apply_professional_ui_style(self) -> None:
        body_pt = self._preferred_base_font_pt()
        caption_pt = max(6.3, body_pt - 0.30)
        title_pt = body_pt
        scale = self._normalized_ui_scale()
        group_margin_top = self._scaled_ui_px(12, 10)
        group_pad_top = self._scaled_ui_px(4, 3)
        tab_pad_v = self._scaled_ui_px(3, 2)
        tab_pad_h = self._scaled_ui_px(8, 6)
        tab_min_w = self._scaled_ui_px(52, 48)
        control_h = self._scaled_ui_px(22, 20)
        control_radius = self._scaled_ui_px(7, 6)
        control_pad_v = self._scaled_ui_px(1.5, 1)
        control_pad_h = self._scaled_ui_px(5, 4)
        compact_h = self._scaled_ui_px(19, 17)
        compact_pad_h = self._scaled_ui_px(5, 4)
        list_item_h = self._scaled_ui_px(22, 20)
        slider_handle = self._scaled_ui_px(14, 12)
        slider_margin = self._scaled_ui_px(5, 4)
        font = self.font()
        try:
            current_size = float(font.pointSizeF())
        except Exception:
            current_size = float(font.pointSize() or 0)
        if abs(current_size - body_pt) > 0.05:
            font.setPointSizeF(body_pt)
            self.setFont(font)
        self.setStyleSheet(
            f"""
            QFrame#toolbarFrame {{
                background: #F8FAFC;
                border: 1px solid #E4E7EC;
                border-radius: 10px;
            }}
            QWidget, QFrame, QLabel, QGroupBox, QPushButton, QToolButton, QComboBox, QLineEdit, QSpinBox, QListWidget, QTabWidget {{
                font-family: "Segoe UI", "Microsoft YaHei UI", "Tahoma", "Arial";
            }}
            QGroupBox {{
                background: #FFFFFF;
                border: 1px solid #D9DEE7;
                border-radius: 10px;
                margin-top: {group_margin_top}px;
                padding-top: {group_pad_top}px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
                color: #344054;
                font-weight: 600;
            }}
            QLabel#captionLabel {{
                color: #667085;
                font-size: {caption_pt:.1f}pt;
                font-weight: 600;
                text-transform: uppercase;
            }}
            QTabWidget::pane {{
                border: 1px solid #D9DEE7;
                background: #FFFFFF;
                border-radius: 10px;
                top: -1px;
            }}
            QTabBar::tab {{
                background: #F2F4F7;
                border: 1px solid #D0D5DD;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: {tab_pad_v}px {tab_pad_h}px;
                margin-right: 4px;
                color: #475467;
                min-width: {tab_min_w}px;
            }}
            QTabBar::tab:selected {{
                background: #FFFFFF;
                color: #101828;
                font-weight: 600;
            }}
            QSplitter::handle {{
                background: #E4E7EC;
                border-radius: 4px;
            }}
            QSplitter::handle:hover {{
                background: #CBD5E1;
            }}
            QSplitter::handle:horizontal {{
                width: 10px;
                margin: 0 1px;
            }}
            QSplitter::handle:vertical {{
                height: 10px;
                margin: 1px 0;
            }}
            QPushButton, QToolButton, QComboBox, QLineEdit, QSpinBox {{
                min-height: {control_h}px;
                border: 1px solid #D0D5DD;
                border-radius: {control_radius}px;
                background: #FFFFFF;
                padding: {control_pad_v}px {control_pad_h}px;
            }}
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QFrame#statusCard {{
                background: #F8FAFC;
                border: 1px solid #E4E7EC;
                border-radius: 10px;
            }}
            QLabel#statusTitle {{
                color: #101828;
                font-size: {title_pt:.1f}pt;
                font-weight: 700;
            }}
            QLabel#statusSubtle {{
                color: #667085;
            }}
            QFrame#groundingAssistCard {{
                background: #FFFFFF;
                border: 1px solid #E4E7EC;
                border-radius: 8px;
            }}
            QLabel#groundingAssistTitle {{
                color: #344054;
                font-size: {body_pt:.1f}pt;
                font-weight: 600;
            }}
            QLabel#groundingAssistText {{
                color: #475467;
                font-size: {body_pt:.1f}pt;
                font-weight: 600;
            }}
            QPushButton[objectCandidate="true"] {{
                min-height: {control_h}px;
                padding: {control_pad_v}px {control_pad_h}px;
                border-radius: {control_radius}px;
                background: #FFFFFF;
                border: 1px solid #D0D5DD;
                color: #344054;
                font-size: {body_pt + 0.2:.1f}pt;
                font-weight: 600;
            }}
            QPushButton[objectCandidate="true"]:hover {{
                background: #F9FAFB;
                border-color: #98A2B3;
            }}
            QPushButton[objectCandidate="true"][candidatePrimary="true"] {{
                background: #FFF7ED;
                border: 1px solid #FDBA74;
                color: #9A3412;
                font-weight: 700;
            }}
            QPushButton[objectCandidate="true"][candidatePrimary="true"]:hover {{
                background: #FFF1E6;
                border-color: #FB923C;
            }}
            QPushButton[groundingAction="true"] {{
                min-height: {control_h}px;
                padding: {control_pad_v}px {control_pad_h}px;
                background: #FFFFFF;
                border: 1px solid #D0D5DD;
                border-radius: {control_radius}px;
                color: #344054;
                font-weight: 600;
            }}
            QPushButton[groundingAction="true"]:hover {{
                background: #F9FAFB;
                border-color: #98A2B3;
            }}
            QLabel#inlineActionHint {{
                border-radius: 10px;
                padding: 10px 12px;
                font-size: {body_pt + 1.4:.1f}pt;
                font-weight: 700;
            }}
            QLabel#inlineActionHint[hintRole="neutral"] {{
                background: #F8FAFC;
                border: 1px solid #E4E7EC;
                color: #475467;
                font-weight: 600;
            }}
            QLabel#inlineActionHint[hintRole="action"] {{
                background: #EFF8FF;
                border: 1px solid #B2DDFF;
                color: #175CD3;
            }}
            QLabel#inlineActionHint[hintRole="review"] {{
                background: #FEF3F2;
                border: 1px solid #FECDCA;
                color: #B42318;
            }}
            QLabel#inlineActionHint[hintRole="complete"] {{
                background: #ECFDF3;
                border: 1px solid #ABEFC6;
                color: #027A48;
            }}
            QPushButton[compactChip="true"] {{
                min-height: {compact_h}px;
                padding: {control_pad_v}px {compact_pad_h}px;
            }}
            QPushButton:hover, QToolButton:hover {{
                background: #F9FAFB;
            }}
            QPushButton:disabled, QToolButton:disabled, QComboBox:disabled, QLineEdit:disabled, QSpinBox:disabled {{
                color: #98A2B3;
                background: #F2F4F7;
            }}
            QToolButton::menu-indicator {{
                image: none;
                width: 0px;
            }}
            QListWidget {{
                border: 1px solid #D0D5DD;
                border-radius: 8px;
                background: #FFFFFF;
                padding: 4px;
            }}
            QListWidget::item {{
                min-height: {control_h}px;
            }}
            QListWidget::item:selected {{
                background: #E0EAFF;
                color: #0F172A;
                border-radius: 6px;
            }}
            QSlider::groove:horizontal {{
                height: 6px;
                background: #E4E7EC;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                width: {slider_handle}px;
                margin: -{slider_margin}px 0;
                background: #2563EB;
                border-radius: {control_radius}px;
            }}
            """
        )

    def _set_shortcut_key(
        self, shortcut: Optional[QShortcut], sid: str, default_key: str
    ) -> None:
        set_shortcut_key(
            shortcut,
            shortcut_value(
                self._shortcut_bindings,
                self._shortcut_defaults,
                sid,
                default_key,
            ),
            default_key,
        )

    def apply_shortcut_settings(
        self, bindings: Optional[Dict[str, str]] = None
    ) -> None:
        self._shortcut_bindings = (
            load_shortcut_bindings() if bindings is None else dict(bindings)
        )
        self._shortcut_defaults = default_shortcut_bindings()
        for sid,(action,key) in getattr(self,"_frame_review_actions",{}).items():
            action.setShortcut(QKeySequence(shortcut_value(self._shortcut_bindings,self._shortcut_defaults,sid,key)))
        self._set_shortcut_key(getattr(self, "sc_left", None), "hoi.step_prev", "Left")
        self._set_shortcut_key(
            getattr(self, "sc_right", None), "hoi.step_next", "Right"
        )
        self._set_shortcut_key(
            getattr(self, "sc_up", None), "hoi.seek_prev_second", "Up"
        )
        self._set_shortcut_key(
            getattr(self, "sc_down", None), "hoi.seek_next_second", "Down"
        )
        self._set_shortcut_key(
            getattr(self, "sc_toggle_play", None), "hoi.play_pause", "Space"
        )
        self._set_shortcut_key(getattr(self, "sc_pause", None), "hoi.pause", "K")
        self._set_shortcut_key(
            getattr(self, "sc_detect", None), "hoi.detect", "Ctrl+Shift+D"
        )
        self._set_shortcut_key(
            getattr(self, "sc_toggle_edit_boxes", None),
            "hoi.toggle_edit_boxes",
            "Ctrl+B",
        )
        self._set_shortcut_key(getattr(self, "sc_undo", None), "hoi.undo", "Ctrl+Z")
        self._set_shortcut_key(getattr(self, "sc_redo", None), "hoi.redo", "Ctrl+Y")
        self._set_shortcut_key(
            getattr(self, "sc_open_settings", None), "hoi.open_settings", "Ctrl+,"
        )
        self._set_shortcut_key(
            getattr(self, "sc_open_quick_start", None), "hoi.open_quick_start", "F1"
        )

    # ---------- data helpers ----------
    def _enable_combo_search(
        self,
        combo: QComboBox,
        placeholder: str = "Search...",
        *,
        auto_popup: bool = False,
    ):
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.NoInsert)
        combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        if combo.lineEdit() is not None:
            combo.lineEdit().setPlaceholderText(placeholder)
        completer = combo.completer()
        if completer is not None:
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            try:
                completer.setFilterMode(Qt.MatchContains)
            except Exception:
                pass
        if auto_popup:
            self._enable_combo_autopopup(combo)

    def _enable_combo_autopopup(self, combo: Optional[QComboBox]) -> None:
        if not isinstance(combo, QComboBox):
            return
        targets = getattr(self, "_combo_autopopup_targets", None)
        if not isinstance(targets, dict):
            targets = {}
            self._combo_autopopup_targets = targets
        for obj in (combo, combo.lineEdit()):
            if obj is None:
                continue
            key = id(obj)
            if targets.get(key) is combo:
                continue
            targets[key] = combo
            try:
                obj.installEventFilter(self)
            except Exception:
                pass

    def _show_combo_popup_soon(
        self,
        combo: Optional[QComboBox],
        *,
        delay_ms: int = 90,
    ) -> None:
        if not isinstance(combo, QComboBox):
            return

        def _show() -> None:
            try:
                if not combo.isEnabled() or not combo.isVisible():
                    return
                view = combo.view()
                if view is not None and view.isVisible():
                    return
                combo.showPopup()
            except Exception:
                pass

        try:
            QTimer.singleShot(max(0, int(delay_ms)), _show)
        except Exception:
            pass

    def _normalize_combo_completion_text(self, combo: Optional[QComboBox]) -> None:
        if not isinstance(combo, QComboBox) or not combo.isEditable():
            return
        text = str(combo.currentText() or "").strip()
        if not text:
            return
        text_lower = text.lower()
        best_idx = -1
        best_len = -1
        for idx in range(combo.count()):
            item_text = str(combo.itemText(idx) or "").strip()
            if not item_text:
                continue
            candidates = [item_text]
            match = re.match(r"^\[(\d+)\]\s*(.+)$", item_text)
            if match and match.group(2):
                candidates.append(match.group(2).strip())
            for candidate in candidates:
                cand = str(candidate or "").strip()
                if not cand:
                    continue
                cand_lower = cand.lower()
                if text_lower == cand_lower or text_lower.endswith(cand_lower):
                    if len(cand_lower) > best_len:
                        best_idx = idx
                        best_len = len(cand_lower)
        if best_idx < 0:
            return
        try:
            combo.setCurrentIndex(best_idx)
            combo.setEditText(str(combo.itemText(best_idx) or ""))
        except Exception:
            pass

    def eventFilter(self, obj, event):
        try:
            targets = getattr(self, "_combo_autopopup_targets", {}) or {}
            combo = targets.get(id(obj))
            if isinstance(combo, QComboBox):
                if event.type() == QEvent.MouseButtonPress:
                    self._show_combo_popup_soon(combo)
                elif event.type() == QEvent.FocusOut:
                    self._normalize_combo_completion_text(combo)
                elif event.type() == QEvent.KeyPress and event.key() in (
                    Qt.Key_Tab,
                    Qt.Key_Backtab,
                    Qt.Key_Return,
                    Qt.Key_Enter,
                ):
                    self._normalize_combo_completion_text(combo)
        except Exception:
            pass
        return super().eventFilter(obj, event)

    def _normalize_anomaly_label(self, value: Any) -> str:
        return normalize_attributes(value, _PROJECT_PROFILE)

    def _is_anomalous_label(self, value: Any) -> bool:
        return is_positive(value, _PROJECT_PROFILE)

    def _remember_anomaly_label(self, value: Any) -> str:
        label = self._normalize_anomaly_label(value)
        if label in _ANOMALY_LABEL_OPTIONS:
            self._known_anomaly_labels.add(label)
        return label

    def _anomaly_label_options(self) -> List[str]:
        return [x for x in _ANOMALY_LABEL_OPTIONS if x not in ('normal','unreviewed','unknown')]

    def _populate_anomaly_combo(
        self,
        combo: Optional[QComboBox],
        selected: Any = None,
    ) -> None:
        if combo is None:
            return
        selected_label = self._remember_anomaly_label(selected)
        if isinstance(combo, AttributeSelector):
            combo.setOptions(self._anomaly_label_options(), selected_label,
                             _PROJECT_PROFILE.get('anomaly_display_names', {}))
            return
        was_blocked = combo.blockSignals(True)
        try:
            combo.clear()
            for label in self._anomaly_label_options():
                combo.addItem(label, label)
            if combo.findText(selected_label) < 0:
                combo.addItem(selected_label, selected_label)
            self._enable_combo_search(
                combo,
                placeholder="Choose anomaly type...",
                auto_popup=True,
            )
            combo.setCurrentText(selected_label)
        finally:
            combo.blockSignals(was_blocked)

    def _refresh_anomaly_combos(self, selected: Any = None) -> None:
        current = selected
        if current is None:
            try:
                current = self.combo_anomaly.currentText()
            except Exception:
                current = _DEFAULT_ANOMALY_LABEL
        self._populate_anomaly_combo(getattr(self, "combo_anomaly", None), current)
        self._populate_anomaly_combo(
            getattr(self, "combo_inline_anomaly", None),
            current,
        )

    def _init_verb_color_combo(self):
        combo = self.label_panel.combo
        if combo.findText("Auto") == -1:
            combo.insertItem(0, "Auto")
        combo.setCurrentText("Auto")
        combo.currentTextChanged.connect(self._on_verb_color_changed)

    def _on_verb_color_changed(self, color_key: str):
        if color_key == "Custom":
            return
        selected = self.label_panel.current_label_name()
        if not selected:
            return
        if self.label_panel.edit.text().strip():
            return
        idx = self.label_panel.index_of_label(selected)
        if idx < 0:
            return
        self.verbs[idx].color_name = color_key
        self.label_panel.refresh()
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.refresh()

    def _update_verb_combo(self):
        """[Restore] Populate the verb combobox from self.verbs list."""
        current = self.combo_verb.currentText()
        was_blocked = self.combo_verb.blockSignals(True)
        self.combo_verb.clear()

        sorted_verbs = sorted([v.name for v in self.verbs])
        self.combo_verb.addItems(sorted_verbs)

        idx = self.combo_verb.findText(current)
        if idx >= 0:
            self.combo_verb.setCurrentIndex(idx)
        elif current:
            self.combo_verb.setCurrentText(current)
        else:
            self.combo_verb.setCurrentIndex(-1)

        self.combo_verb.blockSignals(was_blocked)
        self._sync_action_panel_selection(self.combo_verb.currentText())
        self._update_inline_event_editor()
        self._refresh_label_explanations()

    def _hoi_color_for_verb(self, verb: str) -> Optional[QColor]:
        for v in self.verbs:
            if v.name == verb:
                color_name = v.color_name or ""
                if color_name.lower() == "auto":
                    return None
                return color_from_key(color_name)
        return None

    def _find_event_by_id(self, event_id: int) -> Optional[dict]:
        for ev in self.events:
            if ev.get("event_id") == event_id:
                return ev
        return None

    def _compute_event_frames(self, event: dict) -> tuple:
        times = []
        for actor in self.actors_config:
            hand = actor["id"]
            h = event.get("hoi_data", {}).get(hand, {})
            for key in (
                "interaction_start",
                "functional_contact_onset",
                "interaction_end",
            ):
                val = h.get(key)
                if val is not None:
                    try:
                        times.append(int(val))
                    except Exception:
                        continue
        if not times:
            return None, None
        return min(times), max(times)

    def _sync_event_frames(self, event: dict):
        ws, we = self._compute_event_frames(event)
        event["frames"] = [ws, we]

    def _apply_draft_to_selected_event(self):
        if self.selected_event_id is None:
            return
        ev = self._find_event_by_id(self.selected_event_id)
        if not ev:
            return
        actor_ids = {str(actor.get("id") or "").strip() for actor in self.actors_config}
        ev["hoi_data"] = {
            aid: copy.deepcopy(data)
            for aid, data in self.event_draft.items()
            if aid in actor_ids and isinstance(data, dict)
        }
        self._sync_event_frames(ev)

    def _set_selected_event(self, event_id: int, hand_key: Optional[str] = None):
        if self.selected_event_id == event_id and hand_key == self.selected_hand_label:
            return
        if self.selected_event_id is not None:
            self._save_ui_to_hand_draft(self.selected_hand_label)
            self._apply_draft_to_selected_event()

        ev = self._find_event_by_id(event_id)
        if not ev:
            return
        self._sync_event_frames(ev)
        self.event_id_counter = max(self.event_id_counter, event_id + 1)
        self.event_draft = {
            actor["id"]: copy.deepcopy(ev.get("hoi_data", {}).get(actor["id"], {}))
            for actor in self.actors_config
        }

        actor_ids = [a["id"] for a in self.actors_config]
        if hand_key not in actor_ids:
            hand_key = actor_ids[0]
            for aid in actor_ids:
                if (
                    ev.get("hoi_data", {})
                    .get(aid, {})
                    .get("interaction_start")
                    is not None
                ):
                    hand_key = aid
                    break
        self.selected_hand_label = hand_key
        self.selected_event_id = event_id

        for aid, chk in self.actor_controls.items():
            chk.blockSignals(True)
            chk.setChecked(aid == hand_key)
            chk.blockSignals(False)

        self._load_hand_draft_to_ui(hand_key)
        self._update_status_label()
        self.list_objects.setEnabled(True)

        if hasattr(self, "hoi_timeline") and self.hoi_timeline:
            self.hoi_timeline.set_selected(event_id, hand_key)
            self.hoi_timeline.refresh()

        self._update_action_top5_display(event_id)
        if not getattr(self, "_review_navigation_active", False):
            self._queue_action_label_refresh(event_id, delay_ms=250)
            self._refresh_semantic_suggestions_for_event(event_id, ev)
        self._update_overlay(self.player.current_frame)
        self._update_hoi_titles()

    def _hoi_segments_for_hand(self, hand_key: str) -> List[Dict]:
        segs = []
        for ev in self.events:
            h = ev.get("hoi_data", {}).get(hand_key, {})
            s = h.get("interaction_start")
            e = h.get("interaction_end")
            if s is None or e is None:
                continue
            onset = h.get("functional_contact_onset", s)
            verb = h.get("verb", "")
            base_color = QColor("#16a34a" if self._hand_completion_state(h, hand_key=hand_key).get("complete") else "#dc2626")
            segs.append(
                {
                    "event_id": ev.get("event_id"),
                    "start": int(s),
                    "end": int(e),
                    "onset": int(onset) if onset is not None else int(s),
                    "verb": verb,
                    "color": base_color,
                    "auto_color": base_color is None,
                }
            )
        segs = sorted(segs, key=lambda x: (x["start"], x.get("event_id", -1)))
        palette = list(PRESET_COLORS.values()) or [QColor(120, 120, 120)]
        prev_color = None
        for idx, seg in enumerate(segs):
            auto_color = bool(seg.get("auto_color", False))
            color = seg.get("color")
            if auto_color:
                start_idx = seg.get("event_id")
                if start_idx is None:
                    start_idx = idx
                try:
                    start_idx = abs(int(start_idx))
                except Exception:
                    start_idx = idx
                for offset in range(len(palette)):
                    cand = palette[(start_idx + offset) % len(palette)]
                    if prev_color is None or cand.name() != prev_color.name():
                        color = cand
                        break
            elif color is None:
                color = QColor(120, 120, 120)
            seg["color"] = color
            prev_color = color
        return segs

    def _hand_has_segment(self, ev: dict, hand_key: str) -> bool:
        h = ev.get("hoi_data", {}).get(hand_key, {})
        s = h.get("interaction_start")
        e = h.get("interaction_end")
        return s is not None and e is not None

    def _on_hoi_timeline_delete(self, event_id: int, hand_key: str):
        ev = self._find_event_by_id(event_id)
        if not ev:
            return
        self._push_undo()
        ev.get("hoi_data", {})[hand_key] = self._blank_hand_data()
        self._sync_event_frames(ev)

        if not any(
            self._hand_has_segment(ev, actor["id"]) for actor in self.actors_config
        ):
            if ev in self.events:
                self.events.remove(ev)
            if self.selected_event_id == event_id:
                self.selected_event_id = None
                self.selected_hand_label = None
                self._reset_event_draft()
                if getattr(self, "hoi_timeline", None):
                    self.hoi_timeline.set_selected(None, None)
        else:
            if (
                self.selected_event_id == event_id
                and self.selected_hand_label == hand_key
            ):
                # Find 'other' actor(s) for swap logic
                others = [a["id"] for a in self.actors_config if a["id"] != hand_key]
                other = others[0] if others else hand_key
                if self._hand_has_segment(ev, other):
                    self._set_selected_event(event_id, other)
                else:
                    self.selected_hand_label = None
                    self._reset_event_draft()

        self._update_status_label()
        self._bump_query_state_revision()
        self._refresh_events()
        self._update_hoi_titles()
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.refresh()
        self._log("hoi_segment_delete", event_id=event_id, hand=hand_key)

    def _get_actor_short_label(self, actor_id: str) -> str:
        """Helper for timeline titles."""
        for a in self.actors_config:
            if a["id"] == actor_id:
                return a.get("short", actor_id[:1])
        return actor_id[:1]

    def _get_actor_full_label(self, actor_id: str) -> str:
        for a in self.actors_config:
            if a["id"] == actor_id:
                return a.get("label", actor_id)
        return actor_id

    def _display_field_label(self, field_name: str) -> str:
        mapping = {
            "interaction_start": "start",
            "functional_contact_onset": "onset",
            "interaction_end": "end",
            "verb": "verb",
            "noun_object_id": "noun",
            "target_object_id": "noun",
            "instrument_object_id": "instrument",
            "tool_object_id": "instrument",
            "anomaly_label": "anomaly",
            "bbox_evidence": "boxes",
        }
        return mapping.get(self._canonical_query_field_name(field_name), str(field_name or ""))

    def _bbox_completion_state(self, hand_key, hand_data):
        state = {'missing': [], 'details': [], 'missing_roles': {}}
        if not hand_key or not isinstance(hand_data, dict):
            return state
        for field, label in (('interaction_start','start'),('functional_contact_onset','onset'),('interaction_end','end')):
            frame = hand_data.get(field)
            if type(frame) is not int:
                continue
            for entity in self._review_entities(hand_data, hand_key, frame):
                records = getattr(self, 'frame_review', {}).get(str(frame), {})
                if valid_record(records.get(entity), self._boxes_for_review(entity, frame)):
                    continue
                role = 'hand' if entity.startswith('H:') else 'instrument' if entity == 'O:'+str(self._hand_instrument_object_id(hand_data)) else 'noun'
                state['missing_roles'].setdefault(role, []).append(label)
        state['missing'] = [role+' bbox review' for role in state['missing_roles']]
        state['details'] = [role+': '+', '.join(labels) for role, labels in state['missing_roles'].items()]
        return state

    def _hand_completion_state(
        self,
        hand_data: Optional[dict],
        hand_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        state = {
            "has_data": False,
            "missing": [],
            "inline_missing": [],
            "suggested_fields": [],
            "bbox_missing": [],
            "bbox_details": [],
            "complete": False,
        }
        if not isinstance(hand_data, dict):
            return state

        self._ensure_hand_annotation_state(hand_data)
        start_value = hand_data.get("interaction_start")
        onset_value = hand_data.get("functional_contact_onset")
        end_value = hand_data.get("interaction_end")
        verb_value = str(hand_data.get("verb") or "").strip()
        noun_value = self._hand_noun_object_id(hand_data)
        instrument_value = self._hand_instrument_object_id(hand_data)
        anomaly_label = self._normalize_anomaly_label(hand_data.get("anomaly_label"))
        noun_required = self._noun_required_for_verb(verb_value)
        onset_state = get_field_state(hand_data, "functional_contact_onset")
        onset_confirmed = str(onset_state.get("status") or "").strip().lower() == "confirmed"

        has_data = any(
            [
                start_value is not None,
                onset_value is not None,
                end_value is not None,
                bool(verb_value),
                noun_value is not None and not hand_data.get('shared_assembly_ref'),
                instrument_value is not None,
                self._is_anomalous_label(anomaly_label),
            ]
        )
        missing = []
        if start_value is None or end_value is None:
            missing.append("start/end")
        if onset_value is None and not onset_resolved(hand_data):
            missing.append("onset")
        if not verb_value:
            missing.append("verb")
        if noun_required and noun_value is None:
            missing.append("noun")

        if hand_key is None:
            hand_key = str(getattr(self, "selected_hand_label", "") or "").strip()
        bbox_state = self._bbox_completion_state(hand_key, hand_data)
        bbox_missing = list(bbox_state.get("missing") or [])
        bbox_details = list(bbox_state.get("details") or [])
        for item in bbox_missing:
            if item not in missing:
                missing.append(item)

        inline_missing = list(missing)
        if (
            onset_value is not None
            and not onset_confirmed
            and not self._manual_mode_enabled()
        ):
            inline_missing.insert(0, "confirm onset")

        suggested_fields = []
        for field_name in (
            "interaction_start",
            "functional_contact_onset",
            "interaction_end",
            "verb",
            "noun_object_id",
        ):
            field_state = get_field_state(hand_data, field_name)
            if str(field_state.get("status") or "").strip().lower() == "suggested":
                suggested_fields.append(field_name)

        missing.extend(self._policy_missing(hand_data) if has_data else [])
        state.update(
            {
                "has_data": has_data,
                "missing": missing,
                "inline_missing": inline_missing,
                "suggested_fields": suggested_fields,
                "bbox_missing": bbox_missing,
                "bbox_details": bbox_details,
                "bbox_missing_roles": dict(bbox_state.get("missing_roles") or {}),
                "complete": bool(has_data and not missing and not suggested_fields),
            }
        )
        matching={ev['event_id'] for ev in self.events if ev.get('hoi_data',{}).get(hand_key)==hand_data}
        if not matching and self.selected_event_id is not None:matching={self.selected_event_id}
        if state['has_data'] and any(i['event_id'] in matching and i['hand']==hand_key for i in self._frame_review_issues()):
            state['missing'].append('frame review');state['complete']=False
        return state

    def _hoi_title_for_hand(self, hand_key: str) -> str:
        base = self._get_actor_full_label(hand_key)
        if self.selected_event_id is None:
            return f"{base}\nIdle"
        ev = self._find_event_by_id(self.selected_event_id)
        if not ev:
            return f"{base}\nIdle"
        h_data = {}
        if isinstance(getattr(self, "event_draft", None), dict):
            h_data = self.event_draft.get(hand_key, {}) or {}
        if not h_data:
            h_data = ev.get("hoi_data", {}).get(hand_key, {}) or {}
        completion_state = self._hand_completion_state(h_data, hand_key=hand_key)
        if not completion_state.get("has_data"):
            return f"{base}\nIdle"
        missing = list(completion_state.get("missing") or [])
        suggested_fields = list(completion_state.get("suggested_fields") or [])
        if missing:
            status = "Missing " + "/".join(missing[:2])
        elif suggested_fields:
            labels = [self._display_field_label(name) for name in suggested_fields[:2]]
            status = "Review " + "/".join(labels) if labels else "Review"
        else:
            status = "Completed"
        return f"{base}\n{status}"
    def _update_hoi_titles(self):
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.update_titles()

    def _current_hand_meta(self) -> dict:
        # New timeline-created events should start empty instead of inheriting
        # the previous event's semantic panel selection.
        return {
            "verb": "",
            "target_object_id": None,
            "noun_object_id": None,
            "instrument_object_id": None,
            "tool_object_id": None,
            "anomaly_label": _DEFAULT_ANOMALY_LABEL,
            "has_anomaly": False,
        }

    def _noun_field_label(self) -> str:
        return "noun"

    def _hand_noun_object_id(self, hand_data: Optional[dict]) -> Optional[int]:
        if not isinstance(hand_data, dict):
            return None
        return resolve_object(hand_data, self._assembly_data(), self._assembly_frame())

    def _hand_instrument_object_id(self, hand_data: Optional[dict]) -> Optional[int]:
        if not isinstance(hand_data, dict):
            return None
        return hand_data.get("instrument_object_id", hand_data.get("tool_object_id"))

    def _hand_field_locked_for_automation(
        self,
        hand_data: Optional[dict],
        field_name: str,
    ) -> bool:
        if not isinstance(hand_data, dict):
            return False
        state = get_field_state(hand_data, field_name)
        return str(state.get("status") or "").strip().lower() == "confirmed"

    def _sync_hand_alias_fields(self, hand_data: Optional[dict]) -> dict:
        if not isinstance(hand_data, dict):
            return {}
        noun_object_id = hand_data.get("noun_object_id", hand_data.get("target_object_id"))
        hand_data["target_object_id"] = noun_object_id
        hand_data["noun_object_id"] = noun_object_id
        instrument_object_id = hand_data.get(
            "instrument_object_id",
            hand_data.get("tool_object_id"),
        )
        hand_data["instrument_object_id"] = instrument_object_id
        hand_data["tool_object_id"] = instrument_object_id
        anomaly_label = self._remember_anomaly_label(hand_data.get("anomaly_label"))
        hand_data["anomaly_label"] = anomaly_label
        hand_data["has_anomaly"] = self._is_anomalous_label(anomaly_label)
        return hand_data

    def _export_field_state_aliases(self, state: Optional[dict]) -> dict:
        exported = copy.deepcopy(state or {})
        if "noun_object_id" in exported and "target_object_id" not in exported:
            exported["target_object_id"] = copy.deepcopy(exported.get("noun_object_id"))
        if "target_object_id" in exported and "noun_object_id" not in exported:
            exported["noun_object_id"] = copy.deepcopy(exported.get("target_object_id"))
        return exported

    def _export_field_suggestion_aliases(self, suggestions: Optional[dict]) -> dict:
        exported = copy.deepcopy(suggestions or {})
        if "noun_object_id" in exported and "target_object_id" not in exported:
            exported["target_object_id"] = copy.deepcopy(exported.get("noun_object_id"))
        if "target_object_id" in exported and "noun_object_id" not in exported:
            exported["noun_object_id"] = copy.deepcopy(exported.get("target_object_id"))
        for field_name, row in list(exported.items()):
            if not isinstance(row, dict):
                continue
            if str(field_name or "") == "verb" and row.get("value") is not None:
                row["value"] = self._canonical_label_name(row.get("value"))
            for key in (
                "display_value",
                "object_name",
                "label",
                "category",
                "verb",
                "noun",
                "target",
                "instrument",
                "tool",
            ):
                if key in row and isinstance(row.get(key), str):
                    row[key] = self._canonical_label_name(row.get(key))
        return exported

    def _export_sparse_evidence_state(self, sparse_state: Optional[dict]) -> dict:
        exported = copy.deepcopy(sparse_state or {})
        for row in list(exported.values()):
            if not isinstance(row, dict):
                continue
            for key in ("object_name", "label", "category", "noun", "instrument", "tool"):
                if key in row and isinstance(row.get(key), str):
                    row[key] = self._canonical_label_name(row.get(key))
        return exported

    def _canonical_query_field_name(self, field_name: str) -> str:
        text = str(field_name or "").strip()
        if text in ("target_object_id", "noun"):
            return "noun_object_id"
        if text in ("tool_object_id", "instrument", "tool"):
            return "instrument_object_id"
        if text == "anomaly":
            return "anomaly_label"
        return text

    def _active_noun_ids(self) -> List[int]:
        out: List[int] = []
        for _name, uid in sorted(self.global_object_map.items(), key=lambda x: x[1]):
            try:
                out.append(int(uid))
            except Exception:
                continue
        return out

    def _active_noun_names(self) -> List[str]:
        return [
            str(name)
            for name, _uid in sorted(self.global_object_map.items(), key=lambda x: x[1])
        ]

    def _noun_name_to_id(self) -> Dict[str, int]:
        return {str(name): int(uid) for name, uid in self.global_object_map.items()}

    def _noun_id_to_name(self) -> Dict[int, str]:
        out: Dict[int, str] = {}
        for name, uid in self.global_object_map.items():
            try:
                out[int(uid)] = str(name)
            except Exception:
                continue
        return out

    def _noun_required_for_verb(self, verb_name: Any) -> bool:
        return ontology_noun_required(getattr(self, "hoi_ontology", None), verb_name)

    def _allowed_noun_ids_for_verb(self, verb_name: Any) -> List[int]:
        return ontology_allowed_noun_ids(
            getattr(self, "hoi_ontology", None),
            verb_name,
            self._noun_name_to_id(),
        )

    def _noun_exists_prior_for_verb(self, verb_name: Any) -> float:
        text = str(verb_name or "").strip()
        if not text:
            return 0.5
        ontology = getattr(self, "hoi_ontology", None)
        if ontology is None or not ontology.has_verb(text):
            return 0.75
        return 0.15 if ontology.allow_no_noun(text) else 0.95

    def _verb_allows_no_noun(self, verb_name: Any) -> bool:
        return not self._noun_required_for_verb(verb_name)

    def _semantic_object_support_enabled(self) -> bool:
        return not bool(getattr(self, "_detector_grounding_only", True))

    @staticmethod
    def _source_family_label(source: Any) -> str:
        text = str(source or "").strip().lower()
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

    def _current_noun_source_decision(self, hand_data: Optional[dict]) -> Dict[str, Any]:
        if not isinstance(hand_data, dict):
            return {}
        return dict(hand_data.get("_noun_source_decision") or {})

    def _detector_grounding_noun_candidate(
        self,
        hand_key: str,
        hand_data: Optional[dict],
    ) -> Dict[str, Any]:
        if not isinstance(hand_data, dict):
            return {}
        candidates = self._collect_event_object_candidates(hand_key, hand_data)
        verb_value = str(hand_data.get("verb") or "").strip()
        if bool(self._noun_only_mode) and verb_value:
            allowed_ids = self._allowed_noun_ids_for_verb(verb_value)
            candidates = filter_allowed_object_candidates(candidates, allowed_ids)
        return dict((candidates or [None])[0] or {})

    def _estimate_noun_source_decision(
        self,
        hand_key: str,
        hand_data: Optional[dict],
        *,
        semantic_noun_id: Any,
        semantic_confidence: float,
        grounding_candidate: Optional[dict] = None,
    ) -> Dict[str, Any]:
        if not isinstance(hand_data, dict):
            return {}
        try:
            semantic_noun_id = int(semantic_noun_id) if semantic_noun_id is not None else None
        except Exception:
            semantic_noun_id = None
        detector_candidate = self._detector_grounding_noun_candidate(hand_key, hand_data)
        detector_noun_id = None
        try:
            detector_noun_id = (
                int(detector_candidate.get("object_id"))
                if detector_candidate.get("object_id") is not None
                else None
            )
        except Exception:
            detector_noun_id = None
        detector_confidence = float(
            (detector_candidate or {}).get("candidate_score", 0.0) or 0.0
        )
        semantic_confidence = float(semantic_confidence or 0.0)
        if semantic_noun_id is None and detector_noun_id is None:
            return {}
        if semantic_noun_id is None:
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
            }
        if detector_noun_id is None:
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
            }
        comparison = self._ensure_query_calibrator().compare_field_sources(
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
        comparison["semantic_noun_id"] = int(semantic_noun_id)
        comparison["semantic_confidence"] = float(semantic_confidence)
        comparison["detector_noun_id"] = int(detector_noun_id)
        comparison["detector_confidence"] = float(detector_confidence)
        comparison["detector_candidate"] = dict(detector_candidate or {})
        comparison["grounding_candidate"] = dict(grounding_candidate or {})
        comparison["semantic_matches_detector"] = bool(
            semantic_noun_id is not None
            and detector_noun_id is not None
            and int(semantic_noun_id) == int(detector_noun_id)
        )
        return comparison

    @staticmethod
    def _flatten_noun_source_decision_fields(decision: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        info = dict(decision or {})
        return {
            "noun_primary_source": str(info.get("preferred_source") or ""),
            "noun_primary_family": str(info.get("preferred_family") or ""),
            "noun_source_margin": float(info.get("score_margin", 0.0) or 0.0),
            "semantic_source_acceptance_est": float(
                info.get("source_a_acceptance", 0.0) or 0.0
            ),
            "semantic_source_score": float(info.get("source_a_score", 0.0) or 0.0),
            "semantic_source_support": int(info.get("source_a_support", 0) or 0),
            "detector_source_acceptance_est": float(
                info.get("source_b_acceptance", 0.0) or 0.0
            ),
            "detector_source_score": float(info.get("source_b_score", 0.0) or 0.0),
            "detector_source_support": int(info.get("source_b_support", 0) or 0),
            "semantic_noun_id": info.get("semantic_noun_id"),
            "detector_noun_id": info.get("detector_noun_id"),
            "noun_source_decision_basis": str(info.get("decision_basis") or ""),
        }

    def _strong_semantic_noun_suggestion_id(
        self,
        hand_data: Optional[dict],
    ) -> Optional[int]:
        if not isinstance(hand_data, dict):
            return None
        suggestion = get_field_suggestion(hand_data, "noun_object_id")
        source = str(suggestion.get("source") or "").strip().lower()
        if not source.startswith("semantic_adapter"):
            return None
        try:
            value = (
                int(suggestion.get("value"))
                if suggestion.get("value") is not None
                else None
            )
        except Exception:
            value = None
        if value is None:
            return None
        confidence = float(suggestion.get("confidence", 0.0) or 0.0)
        threshold = float(
            self._semantic_review_thresholds.get("noun_confidence", 0.80) or 0.80
        )
        if confidence < threshold:
            return None
        return int(value)

    def _hand_has_explicit_no_noun_lock(
        self,
        hand_data: Optional[dict],
        *,
        verb_name: Any = _NO_FIELD_VALUE,
    ) -> bool:
        if not isinstance(hand_data, dict):
            return False
        noun_state = get_field_state(hand_data, "noun_object_id")
        if str(noun_state.get("status") or "").strip().lower() != "confirmed":
            return False
        if self._hand_noun_object_id(hand_data) is not None:
            return False
        resolved_verb = hand_data.get("verb") if verb_name is _NO_FIELD_VALUE else verb_name
        return bool(str(resolved_verb or "").strip()) and self._verb_allows_no_noun(
            resolved_verb
        )

    def _set_hand_no_noun_confirmation(
        self,
        hand_data: Optional[dict],
        *,
        source: str = "manual_no_noun",
        status: str = "confirmed",
        note: str = "",
    ) -> None:
        if not isinstance(hand_data, dict):
            return
        hand_data["noun_object_id"] = None
        hand_data["target_object_id"] = None
        clear_field_suggestion(hand_data, "noun_object_id")
        self._set_hand_field_state(
            hand_data,
            "noun_object_id",
            source=source,
            value=None,
            status=status,
            note=note
            or "Confirmed that the current verb does not require a noun/object.",
        )

    def _bounded01(self, value: Any, default: float = 0.0) -> float:
        try:
            value = float(value)
        except Exception:
            value = float(default)
        return max(0.0, min(1.0, value))

    def _runtime_artifacts_dir(self) -> str:
        base = os.path.dirname(self.current_annotation_path) if self.current_annotation_path else os.getcwd()
        path = os.path.join(base, "runtime_artifacts")
        os.makedirs(path, exist_ok=True)
        return path

    def _handtrack_cache_file(self, video_path: Optional[str] = None) -> str:
        path = str(video_path or self.video_path or "").strip()
        norm = self._normalized_video_path(path)
        key = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16] if norm else "default"
        base = os.path.splitext(os.path.basename(path or "video"))[0] or "video"
        return os.path.join(
            self._runtime_artifacts_dir(),
            f"{base}.handtracks.{key}.json",
        )

    def _normalize_handtrack_payload(self, payload: Optional[dict]) -> Dict[str, Any]:
        out = dict(payload or {})
        tracks_out: Dict[str, Any] = {}
        for actor_id, track in dict(out.get("tracks") or {}).items():
            rows = []
            frame_map = {}
            for row in list((track or {}).get("frames") or []):
                if not isinstance(row, dict):
                    continue
                try:
                    frame = int(row.get("frame"))
                except Exception:
                    continue
                clean = {
                    "frame": int(frame),
                    "bbox": [float(v) for v in list(row.get("bbox") or [0.0, 0.0, 0.0, 0.0])[:4]],
                    "center": [float(v) for v in list(row.get("center") or [0.0, 0.0])[:2]],
                    "area": float(row.get("area", 0.0) or 0.0),
                    "motion": float(row.get("motion", 0.0) or 0.0),
                    "handedness": str(row.get("handedness") or "").strip().lower(),
                    "handedness_score": float(row.get("handedness_score", 0.0) or 0.0),
                    "detection_confidence": float(row.get("detection_confidence", 0.0) or 0.0),
                    "interpolated": bool(row.get("interpolated", False)),
                }
                rows.append(clean)
                frame_map[int(frame)] = clean
            rows.sort(key=lambda item: int(item.get("frame", 0)))
            tracks_out[str(actor_id)] = {
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
        out["tracks"] = tracks_out
        out["frame_count"] = int(out.get("frame_count", 0) or 0)
        out["stride"] = int(out.get("stride", 1) or 1)
        return out

    def _sanitize_handtrack_payload_for_save(self, payload: Optional[dict]) -> Dict[str, Any]:
        clean = dict(payload or {})
        if "video_path" in clean:
            clean["video_path"] = self._portable_path_token(clean.get("video_path"))
        if "cache_file" in clean:
            clean["cache_file"] = self._portable_path_token(clean.get("cache_file"))
        tracks = {}
        for actor_id, track in dict(clean.get("tracks") or {}).items():
            rows = []
            for row in list((track or {}).get("frames") or []):
                if not isinstance(row, dict):
                    continue
                rows.append(
                    {
                        "frame": int(row.get("frame", 0) or 0),
                        "bbox": [float(v) for v in list(row.get("bbox") or [0.0, 0.0, 0.0, 0.0])[:4]],
                        "center": [float(v) for v in list(row.get("center") or [0.0, 0.0])[:2]],
                        "area": float(row.get("area", 0.0) or 0.0),
                        "motion": float(row.get("motion", 0.0) or 0.0),
                        "handedness": str(row.get("handedness") or "").strip().lower(),
                        "handedness_score": float(row.get("handedness_score", 0.0) or 0.0),
                        "detection_confidence": float(row.get("detection_confidence", 0.0) or 0.0),
                        "interpolated": bool(row.get("interpolated", False)),
                    }
                )
            tracks[str(actor_id)] = {
                "frame_count": int((track or {}).get("frame_count", clean.get("frame_count", 0)) or 0),
                "coverage": float((track or {}).get("coverage", 0.0) or 0.0),
                "motion_peak_frame": (
                    None
                    if (track or {}).get("motion_peak_frame") is None
                    else int((track or {}).get("motion_peak_frame"))
                ),
                "motion_peak_score": float((track or {}).get("motion_peak_score", 0.0) or 0.0),
                "frames": rows,
            }
        clean["tracks"] = tracks
        return clean

    def _apply_handtrack_payload(
        self,
        payload: Optional[dict],
        *,
        cache_file: str = "",
        log_event: str = "",
    ) -> bool:
        normalized = self._normalize_handtrack_payload(payload)
        tracks = dict(normalized.get("tracks") or {})
        if not tracks:
            return False
        self._handtrack_cache = normalized
        self._handtrack_cache_key = self._normalized_video_path(
            str(normalized.get("video_path") or self.video_path or "").strip()
        )
        self._handtrack_status.update(
            {
                "ready": True,
                "building": False,
                "video_path": str(normalized.get("video_path") or self.video_path or ""),
                "cache_file": str(cache_file or self._handtrack_cache_file(normalized.get("video_path"))),
                "backend": str(normalized.get("backend") or self._handtrack_status.get("backend") or ""),
                "error": "",
            }
        )
        self._bump_query_state_revision()
        if getattr(self, "player", None) is not None:
            try:
                self._refresh_boxes_for_frame(int(getattr(self.player, "current_frame", 0) or 0), skip_events=True, lightweight=True)
            except Exception:
                pass
        self._update_status_label()
        if self.selected_event_id is not None:
            ev = self._find_event_by_id(self.selected_event_id)
            if ev:
                self._refresh_semantic_suggestions_for_event(int(self.selected_event_id), ev)
                self._queue_action_label_refresh(int(self.selected_event_id), delay_ms=60, force=True)
        if log_event:
            self._log(
                log_event,
                path=str(cache_file or ""),
                tracks=len(tracks),
                frame_count=int(normalized.get("frame_count", 0) or 0),
                stride=int(normalized.get("stride", 1) or 1),
            )
        self._log_annotation_ready_state(log_event or "hoi_handtrack_ready")
        return True

    def _load_handtrack_cache(self, video_path: Optional[str] = None) -> bool:
        cache_file = self._handtrack_cache_file(video_path)
        if not os.path.isfile(cache_file):
            return False
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return False
        return self._apply_handtrack_payload(
            payload,
            cache_file=cache_file,
            log_event="hoi_handtrack_cache_load",
        )

    def _save_handtrack_cache(self, payload: Optional[dict]) -> str:
        cache_file = self._handtrack_cache_file((payload or {}).get("video_path"))
        clean = self._sanitize_handtrack_payload_for_save(payload)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)
        return cache_file

    def _start_handtrack_precompute(self, video_path: Optional[str] = None, *, force: bool = False) -> bool:
        target_path = str(video_path or self.video_path or "").strip()
        if not target_path:
            return False
        target_key = self._normalized_video_path(target_path)
        if not force and self._handtrack_status.get("building"):
            return False
        if not force and self._handtrack_status.get("ready") and self._handtrack_cache_key == target_key:
            return True
        if not force and self._load_handtrack_cache(target_path):
            return True
        worker = HandTrackBuildWorker(
            target_path,
            self.actors_config,
            max_hands=self.mp_hands_max,
            det_conf=self.mp_hands_conf,
            track_conf=self.mp_hands_track_conf,
        )
        worker.progress.connect(self._on_handtrack_build_progress)
        worker.finished.connect(self._on_handtrack_build_finished)
        self._handtrack_worker = worker
        self._handtrack_cache = {}
        self._handtrack_cache_key = target_key
        self._handtrack_status.update(
            {
                "ready": False,
                "building": True,
                "video_path": target_path,
                "cache_file": self._handtrack_cache_file(target_path),
                "backend": "",
                "error": "",
            }
        )
        self._log("hoi_handtrack_build_start", video_path=target_path)
        self._log_annotation_ready_state("hoi_handtrack_build_start")
        self._update_status_label()
        worker.start()
        return True

    def _on_handtrack_build_progress(self, message: str) -> None:
        text = str(message or "").strip()
        if text:
            self._log("hoi_handtrack_build_progress", message=text)
        self._update_status_label()

    def _on_handtrack_build_finished(self, payload: Optional[dict]) -> None:
        self._handtrack_worker = None
        payload = dict(payload or {})
        payload_video = str(payload.get("video_path") or "").strip()
        payload_key = self._normalized_video_path(payload_video)
        current_key = self._normalized_video_path(self.video_path)
        if payload_key and current_key and payload_key != current_key:
            self._handtrack_status.update({"building": False})
            self._log(
                "hoi_handtrack_build_stale_ignored",
                payload_video=payload_video,
                current_video=self.video_path,
            )
            self._log_annotation_ready_state("hoi_handtrack_build_stale_ignored")
            return
        if not bool(payload.get("success")):
            self._handtrack_status.update(
                {
                    "ready": False,
                    "building": False,
                    "backend": str(payload.get("backend") or ""),
                    "error": str(payload.get("error") or "hand track build failed"),
                }
            )
            self._log(
                "hoi_handtrack_build_failed",
                error=str(payload.get("error") or "hand track build failed"),
            )
            self._log_annotation_ready_state("hoi_handtrack_build_failed")
            return
        cache_file = ""
        try:
            cache_file = self._save_handtrack_cache(payload)
        except Exception as ex:
            self._log("hoi_handtrack_cache_save_failed", error=str(ex))
        self._apply_handtrack_payload(
            payload,
            cache_file=cache_file,
            log_event="hoi_handtrack_build_finished",
        )

    def _handtrack_track(self, hand_key: str) -> Dict[str, Any]:
        return dict((self._handtrack_cache.get("tracks") or {}).get(str(hand_key)) or {})

    def _handtrack_segment_prior(
        self,
        hand_key: str,
        start_frame: Optional[int],
        end_frame: Optional[int],
    ) -> Dict[str, Any]:
        try:
            start = int(start_frame) if start_frame is not None else None
            end = int(end_frame) if end_frame is not None else None
        except Exception:
            start = end = None
        if start is None or end is None:
            return {}
        if end < start:
            start, end = end, start
        track = self._handtrack_track(hand_key)
        frame_map = dict(track.get("frame_map") or {})
        if not frame_map:
            return {}
        segment_len = max(1, end - start)
        rows = [
            dict(frame_map.get(int(frame)) or {})
            for frame in range(int(start), int(end) + 1)
            if int(frame) in frame_map
        ]
        if len(rows) < 3:
            return {}
        coverage = float(len(rows)) / float(max(1, segment_len + 1))
        peak_row = max(rows, key=lambda row: float(row.get("motion", 0.0) or 0.0))
        peak_motion = float(peak_row.get("motion", 0.0) or 0.0)
        avg_motion = float(sum(float(row.get("motion", 0.0) or 0.0) for row in rows) / float(max(1, len(rows))))
        if peak_motion <= 1e-6:
            return {}
        onset_frame = int(peak_row.get("frame", start) or start)
        onset_ratio = self._bounded01((onset_frame - start) / float(max(1, segment_len)), 0.5)
        dominance = peak_motion / float(max(1e-4, avg_motion))
        peak_norm = self._bounded01(peak_motion * 8.0, 0.0)
        dominance_norm = self._bounded01(dominance / 3.0, 0.0)
        confidence = self._bounded01(
            0.40 * coverage + 0.35 * peak_norm + 0.25 * dominance_norm,
            0.0,
        )
        if bool(peak_row.get("interpolated")):
            confidence *= 0.88
        support_cut = max(peak_motion * 0.35, avg_motion)
        support_frames = [
            int(row.get("frame", onset_frame) or onset_frame)
            for row in rows
            if float(row.get("motion", 0.0) or 0.0) >= support_cut
        ]
        if support_frames:
            left_frame = max(start, min(support_frames))
            right_frame = min(end, max(support_frames))
        else:
            radius = max(2, int(round(0.08 * float(segment_len))))
            left_frame = max(start, onset_frame - radius)
            right_frame = min(end, onset_frame + radius)
        onset_band = {
            "center_ratio": float(onset_ratio),
            "left_ratio": self._bounded01((left_frame - start) / float(max(1, segment_len)), onset_ratio),
            "right_ratio": self._bounded01((right_frame - start) / float(max(1, segment_len)), onset_ratio),
        }
        return {
            "hand": str(hand_key),
            "source": "handtrack_once",
            "onset_frame": int(onset_frame),
            "onset_ratio": float(onset_ratio),
            "onset_band": onset_band,
            "band_width": float(onset_band["right_ratio"] - onset_band["left_ratio"]),
            "confidence": float(confidence),
            "coverage": float(coverage),
            "motion_peak": float(peak_motion),
            "support_frame_count": int(len(support_frames)),
        }

    def _semantic_feedback_file(self) -> str:
        if not self.semantic_feedback_path:
            self.semantic_feedback_path = os.path.join(
                self._semantic_runtime_artifacts_dir(),
                "semantic_feedback.jsonl",
            )
        return self.semantic_feedback_path

    def _semantic_model_file(self) -> str:
        if not self.semantic_adapter_model_path:
            self.semantic_adapter_model_path = os.path.join(
                self._semantic_runtime_artifacts_dir(),
                "semantic_adapter.pt",
            )
        return self.semantic_adapter_model_path

    def _semantic_feedback_row_count(self) -> int:
        path = str(self._semantic_feedback_file() or "").strip()
        if not path or not os.path.isfile(path):
            return 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def _semantic_adapter_runtime_status(self) -> Dict[str, Any]:
        active_model_path = str(
            getattr(self, "semantic_adapter_active_model_path", "") or ""
        ).strip()
        participant_model_path = str(self._semantic_model_file() or "").strip()
        shared_model_path = str(self._shared_semantic_adapter_file() or "").strip()
        package = getattr(self, "semantic_adapter_package", None)
        runtime_error = str(getattr(self, "_semantic_adapter_unavailable_reason", "") or "")
        model_available = bool(
            (package is not None and self._semantic_adapter_matches_runtime(package))
            or (participant_model_path and os.path.isfile(participant_model_path))
            or (active_model_path and os.path.isfile(active_model_path))
            or (shared_model_path and os.path.isfile(shared_model_path))
        )
        if runtime_error:
            model_available = False
        sample_count = None
        if package is not None and self._semantic_adapter_matches_runtime(package):
            try:
                sample_count = int(getattr(package, "sample_count", 0) or 0)
            except Exception:
                sample_count = None
        training_running = bool(
            self._semantic_adapter_train_worker is not None
            and self._semantic_adapter_train_worker.isRunning()
        )
        return {
            "participant_code": self._normalized_participant_code(),
            "workspace": self._semantic_runtime_artifacts_dir(),
            "active_model_path": active_model_path or participant_model_path or shared_model_path,
            "model_available": bool(model_available),
            "runtime_error": runtime_error,
            "model_sample_count": sample_count,
            "feedback_rows": int(self._semantic_feedback_row_count()),
            "feedback_pending": int(getattr(self, "_semantic_feedback_pending", 0) or 0),
            "training_running": bool(training_running),
        }

    def _semantic_adapter_load_key(self, fp: str) -> str:
        path = str(fp or "").strip()
        if not path:
            return ""
        try:
            path = os.path.abspath(path)
        except Exception:
            pass
        return os.path.normcase(path)

    def _load_semantic_adapter_package_safely(
        self,
        fp: str,
        *,
        notify_user: bool = False,
        auto_discovered: bool = False,
    ):
        load_key = self._semantic_adapter_load_key(fp)
        self._semantic_adapter_unavailable_reason = ""
        try:
            package = load_adapter_package(fp)
        except Exception as ex:
            message = str(ex) or type(ex).__name__
            if load_key:
                getattr(self, "_semantic_adapter_failed_load_paths", set()).add(load_key)
            self._semantic_adapter_unavailable_reason = message
            self.semantic_adapter_package = None
            self.semantic_adapter_active_model_path = ""
            self._log(
                "hoi_auto_load_failed",
                asset="semantic_adapter",
                path=fp,
                error=message,
                error_type=type(ex).__name__,
                auto_discovered=bool(auto_discovered),
            )
            if notify_user:
                QMessageBox.warning(
                    self,
                    "Semantic Adapter",
                    "Semantic adapter suggestions are unavailable because PyTorch "
                    f"could not be loaded:\n{message}\n\nThe app will continue without "
                    "semantic adapter suggestions.",
                )
            return None
        if package is not None:
            failed_paths = getattr(self, "_semantic_adapter_failed_load_paths", set())
            if load_key and hasattr(failed_paths, "discard"):
                failed_paths.discard(load_key)
            self._semantic_adapter_unavailable_reason = ""
        return package

    def _load_semantic_adapter_from_path(
        self,
        fp: str,
        *,
        notify_user: bool = True,
        auto_discovered: bool = False,
        record_as_base: bool = True,
    ) -> bool:
        if not self._guard_asset_mutation(
            "semantic adapter",
            path=fp,
            notify_user=notify_user,
            auto_discovered=auto_discovered,
            full_assist_only=True,
        ):
            return False
        package = self._load_semantic_adapter_package_safely(
            fp,
            notify_user=notify_user,
            auto_discovered=auto_discovered,
        )
        if package is None:
            runtime_error = str(getattr(self, "_semantic_adapter_unavailable_reason", "") or "")
            if notify_user:
                if not runtime_error:
                    QMessageBox.warning(self, "Semantic Adapter", f"Failed to load adapter package:\n{fp}")
            elif not runtime_error:
                self._log("hoi_auto_load_failed", asset="semantic_adapter", path=fp, error="load_failed")
            return False
        if not self._semantic_adapter_matches_runtime(package):
            if notify_user:
                QMessageBox.warning(
                    self,
                    "Semantic Adapter",
                    "Adapter package does not match the current runtime feature schema.",
                )
            else:
                self._log("hoi_auto_load_failed", asset="semantic_adapter", path=fp, error="schema_mismatch")
            return False
        loaded_path = str(fp or "").strip()
        participant_model_path = str(self._semantic_model_file() or "").strip()
        prefer_participant_model = bool(
            record_as_base
            and participant_model_path
            and os.path.isfile(participant_model_path)
            and os.path.normcase(participant_model_path) != os.path.normcase(loaded_path)
        )
        if bool(record_as_base):
            self.semantic_adapter_base_model_path = loaded_path
        if prefer_participant_model:
            self.semantic_adapter_package = None
            self.semantic_adapter_active_model_path = ""
            self._ensure_semantic_adapter_loaded()
        else:
            self.semantic_adapter_active_model_path = loaded_path
            self.semantic_adapter_package = package
        self._log(
            "hoi_load_semantic_adapter",
            path=loaded_path,
            sample_count=int(getattr(package, "sample_count", 0) or 0),
            auto_discovered=bool(auto_discovered),
            record_as_base=bool(record_as_base),
            active_model_path=str(getattr(self, "semantic_adapter_active_model_path", "") or "").strip(),
            prefer_participant_model=bool(prefer_participant_model),
        )
        self._log_annotation_ready_state("hoi_load_semantic_adapter")
        if notify_user:
            QMessageBox.information(
                self,
                "Semantic Adapter",
                f"Loaded semantic adapter:\n{os.path.basename(loaded_path)}",
            )
        return True

    def _has_loaded_targets(self) -> bool:
        return bool(len(getattr(self, "global_object_map", {}) or {}) > 0 or getattr(self.combo_target, "count", lambda: 0)() > 1)

    def _has_loaded_verbs(self) -> bool:
        return bool(len(getattr(self, "verbs", []) or []) > 0)

    def _has_loaded_ontology(self) -> bool:
        return bool(len(getattr(getattr(self, "hoi_ontology", None), "relations", {}) or {}) > 0)

    def _has_loaded_class_map(self) -> bool:
        return bool(len(getattr(self, "class_map", {}) or {}) > 0)

    def _has_loaded_yolo_model(self) -> bool:
        return bool(getattr(self, "yolo_model", None) is not None and str(getattr(self, "yolo_weights_path", "") or "").strip())

    def _has_loaded_videomae_weights(self) -> bool:
        return bool(str(getattr(self, "videomae_weights_path", "") or "").strip())

    def _has_loaded_videomae_verb_list(self) -> bool:
        return bool(str(getattr(self, "videomae_verb_list_path", "") or "").strip())

    def _has_loaded_semantic_adapter(self) -> bool:
        explicit_path = str(getattr(self, "semantic_adapter_model_path", "") or "").strip()
        base_path = str(getattr(self, "semantic_adapter_base_model_path", "") or "").strip()
        shared_path = str(self._shared_semantic_adapter_file() or "").strip()
        default_path = self._semantic_model_file()
        return bool(
            getattr(self, "semantic_adapter_package", None) is not None
            or (explicit_path and os.path.isfile(explicit_path))
            or (base_path and os.path.isfile(base_path))
            or (shared_path and os.path.isfile(shared_path))
            or (default_path and os.path.isfile(default_path))
        )

    def _videomae_precomputed_ready(self) -> bool:
        cache = getattr(self, "_videomae_precomputed_cache", None)
        return bool(
            isinstance(cache, dict)
            and len(cache) > 0
            and str(getattr(self, "_videomae_precomputed_cache_key", "") or "")
            == self._normalized_video_path(self.video_path)
        )

    def _clear_precomputed_videomae_cache(self) -> None:
        self._videomae_precomputed_cache = {}
        self._videomae_precomputed_cache_path = ""
        self._videomae_precomputed_cache_key = ""

    def _load_precomputed_videomae_cache(
        self,
        fp: str,
        *,
        notify_user: bool = True,
        auto_discovered: bool = False,
    ) -> bool:
        cache_path = str(fp or "").strip()
        if not cache_path:
            return False
        if not self._guard_asset_mutation(
            "precomputed encoder cache",
            path=cache_path,
            notify_user=notify_user,
            auto_discovered=auto_discovered,
            full_assist_only=True,
        ):
            return False
        cache = load_precomputed_feature_cache(cache_path)
        if not isinstance(cache, dict) or not cache:
            if notify_user:
                QMessageBox.warning(
                    self,
                    "Precomputed Encoder Cache",
                    f"Failed to load precomputed cache:\n{cache_path}",
                )
            else:
                self._log(
                    "hoi_auto_load_failed",
                    asset="videomae_cache",
                    path=cache_path,
                    error="load_failed",
                )
            return False
        self._videomae_precomputed_cache = dict(cache)
        self._videomae_precomputed_cache_path = cache_path
        self._videomae_precomputed_cache_key = self._normalized_video_path(
            (cache.get("meta") or {}).get("video_path") or self.video_path
        )
        self._log(
            "hoi_load_videomae_cache",
            path=cache_path,
            auto_discovered=bool(auto_discovered),
        )
        self._log_annotation_ready_state("hoi_load_videomae_cache")
        if self.selected_event_id is not None:
            self._queue_action_label_refresh(self.selected_event_id, delay_ms=60, force=True)
        if notify_user:
            QMessageBox.information(
                self,
                "Precomputed Encoder Cache",
                f"Loaded precomputed encoder cache:\n{os.path.basename(cache_path)}",
            )
        return True

    def _precomputed_videomae_summary_for_event(self, event: Optional[dict]) -> Dict[str, Any]:
        if not isinstance(event, dict) or not self._videomae_precomputed_ready():
            return {}
        start, end = self._compute_event_frames(event)
        if start is None or end is None:
            return {}
        onset_context = self._primary_onset_context_for_event(event)
        local_context = self._primary_local_onset_context_for_event(event)
        summary = aggregate_precomputed_feature_cache(
            getattr(self, "_videomae_precomputed_cache", None),
            start_frame=start,
            end_frame=end,
            onset_band=dict(onset_context.get("onset_band") or {}),
            top_k=5,
        )
        local_summary = {}
        if local_context:
            local_summary = aggregate_precomputed_feature_cache(
                getattr(self, "_videomae_precomputed_cache", None),
                start_frame=local_context.get("start_frame"),
                end_frame=local_context.get("end_frame"),
                onset_band=None,
                top_k=5,
            )
        return {
            "segment_feature": list(summary.get("segment_feature") or []),
            "candidates": [dict(row) for row in list(summary.get("candidates") or [])],
            "segment_meta": dict(summary.get("meta") or {}),
            "local_segment_feature": list(local_summary.get("segment_feature") or []),
            "local_candidates": [dict(row) for row in list(local_summary.get("candidates") or [])],
            "local_meta": dict(local_summary.get("meta") or {}),
        }

    @staticmethod
    def _score_local_asset_candidate(path: str, kind: str) -> int:
        name = os.path.basename(str(path or "")).lower()
        stem, ext = os.path.splitext(name)
        ext = ext.lower()
        score = -1
        exact_scores = {
            "targets": {
                "nouns.txt": 100,
                "nouns_list.txt": 96,
                "noun_list.txt": 95,
                "objects.txt": 93,
                "object_list.txt": 92,
                "noun_object_list.txt": 98,
                "noun_objects.txt": 97,
                "targets.txt": 88,
            },
            "verbs": {
                "verbs.txt": 100,
                "verb_list.txt": 96,
                "actions.txt": 90,
                "action_list.txt": 89,
            },
            "ontology": {
                "verb_noun_ontology.csv": 100,
                "hoi_ontology.csv": 98,
                "ontology.csv": 92,
            },
            "class_map": {
                "data.yaml": 100,
                "data.yml": 100,
            },
            "videomae_verbs": {
                "videomae_verbs.yaml": 100,
                "videomae_verbs.yml": 100,
                "videomae_verbs.json": 99,
                "videomae_verbs.txt": 98,
                "videomae_verb_list.txt": 98,
            },
            "videomae_cache": {
                "videomae_cache.npz": 100,
            },
            "semantic_adapter": {
                "semantic_adapter.pt": 100,
                "semantic_adapter.pth": 99,
            },
        }
        score = max(score, int(exact_scores.get(kind, {}).get(name, -1)))

        if kind == "targets" and ext == ".txt":
            if re.search(r"(noun|object|target)", stem) and not re.search(r"(verb|action|videomae)", stem):
                score = max(score, 80)
        elif kind == "verbs" and ext == ".txt":
            if re.search(r"(verb|action)", stem) and "videomae" not in stem:
                score = max(score, 80)
        elif kind == "ontology" and ext == ".csv":
            if "ontology" in stem:
                score = max(score, 90)
            if re.search(r"verb.*noun|noun.*verb", stem):
                score = max(score, 88)
        elif kind == "class_map" and ext in {".yaml", ".yml"}:
            if "data" in stem or "class" in stem:
                score = max(score, 80)
        elif kind == "yolo_model" and ext in {".pt", ".pth"}:
            if "yolo" in stem and not re.search(r"(videomae|adapter|semantic)", stem):
                score = max(score, 90)
        elif kind == "videomae_weights" and ext in {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}:
            if re.search(r"(videomae|video_mae|vmae)", stem):
                score = max(score, 90)
            elif re.search(r"(kinetics|k400|k600|k700|k710|from_giant|from-giant)", stem) and re.search(r"(vit|video|mae)", stem):
                score = max(score, 84)
            elif re.search(r"(kinetics|k400|k600|k700|k710|from_giant|from-giant)", stem):
                score = max(score, 78)
        elif kind == "videomae_verbs" and ext in {".yaml", ".yml", ".json", ".txt"}:
            if "videomae" in stem and re.search(r"(verb|action|label|name)", stem):
                score = max(score, 85)
        elif kind == "videomae_cache" and ext == ".npz":
            if re.search(r"(videomae|video_mae|vmae)", stem) and "cache" in stem:
                score = max(score, 92)
            elif stem.endswith(".videomae_cache"):
                score = max(score, 95)
        elif kind == "semantic_adapter" and ext in {".pt", ".pth"}:
            if re.search(r"(semantic_adapter|adapter)", stem) and "videomae" not in stem and "yolo" not in stem:
                score = max(score, 84)
        return int(score)

    def _pick_local_asset_candidate(self, video_path: str, kind: str) -> str:
        video_dir = os.path.dirname(os.path.abspath(str(video_path or "").strip()))
        if not video_dir or not os.path.isdir(video_dir):
            return ""
        best_path = ""
        best_score = -1
        try:
            names = sorted(os.listdir(video_dir))
        except Exception:
            return ""
        for name in names:
            candidate = os.path.join(video_dir, name)
            if not os.path.isfile(candidate):
                continue
            score = self._score_local_asset_candidate(candidate, kind)
            if score > best_score:
                best_score = score
                best_path = candidate
        return best_path if best_score >= 0 else ""

    def _apply_profile_libraries(self):
        existing={label.name for label in self.verbs}
        added=False
        for name in _PROJECT_PROFILE.get('verbs',[]):
            if name not in existing:
                uid=max((label.id for label in self.verbs),default=-1)+1
                self.verbs.append(LabelDef(name=name,id=uid,color_name=self._color_for_index(uid)))
                existing.add(name);added=True
        if added:
            self._update_verb_combo();self.label_panel.refresh()
        if not _PROJECT_PROFILE.get('create_default_noun_instances',False):return
        known={self._norm_category(name) for name in self.global_object_map}
        combos=[getattr(self,name,None) for name in ('combo_target','combo_instrument','combo_inline_noun','combo_inline_instrument')]
        blocked=[(combo,combo.blockSignals(True)) for combo in combos if combo is not None]
        try:
            for name in _PROJECT_PROFILE.get('noun_classes',[]):
                if name=='assembly' or self._norm_category(name) in known:continue
                uid=self.object_id_counter;self.object_id_counter+=1
                self._register_object_entry(uid,name+'_1');known.add(self._norm_category(name))
        finally:
            for combo,previous in blocked:combo.blockSignals(previous)

    def _auto_load_task_bundle(self, video_path):
        from core.task_assets import resolve_task
        from core.project_profile import activate_project_profile
        global _DEFAULT_ANOMALY_LABEL, _ANOMALY_LABEL_OPTIONS, _ANOMALY_LABEL_ALIASES
        task = resolve_task(video_path)
        if task is None:
            return False
        data, paths = task["data"], task["paths"]
        # Validate every required input before changing profile or annotations.
        profile = paths.get("project_profile")
        if profile is None or not profile.is_file():
            raise ValueError("Task project_profile.json is missing")
        profile_data = json.loads(profile.read_text(encoding="utf-8-sig"))
        if not isinstance(profile_data, dict):
            raise ValueError("Invalid task project profile")
        activate_project_profile(profile)
        _DEFAULT_ANOMALY_LABEL = str(_PROJECT_PROFILE.get("default_anomaly_label", "unreviewed"))
        _ANOMALY_LABEL_OPTIONS = tuple(_PROJECT_PROFILE.get("anomaly_labels") or [_DEFAULT_ANOMALY_LABEL])
        _ANOMALY_LABEL_ALIASES = dict(_PROJECT_PROFILE.get("anomaly_aliases") or {})
        self._known_anomaly_labels = set(_ANOMALY_LABEL_OPTIONS)
        # Register checkpoints only: opening a video must not eagerly load GPU models.
        checkpoint = paths.get("sam_checkpoint")
        os.environ.pop("IMPACT_SAM2_CHECKPOINT", None)
        if checkpoint and checkpoint.is_file():
            os.environ["IMPACT_SAM2_CHECKPOINT"] = str(checkpoint)
        self.task_yolo_checkpoint = str(paths.get("yolo_checkpoint") or "")
        self._task_trial_id = str(data.get("trial_id") or "")
        resume = paths.get("resume_annotations")
        self._task_resume_path = str(resume) if resume else ""
        if data.get("status") != "ready" and not (resume and resume.is_file()):
            self._apply_profile_libraries()
            QMessageBox.warning(self, "Task not ready", data.get("reason") or "Assets or timeline verification are pending. No annotations imported.")
            return True
        expected = data.get("frame_count")
        if expected is not None and int(expected) != int(self.player.frame_count):
            raise ValueError("Video frame count differs from the task manifest")
        fps = data.get("fps")
        if fps is not None and abs(float(fps)-float(self.player.frame_rate)) > 0.01:
            raise ValueError("Video FPS differs from the task manifest")
        resume = paths.get("resume_annotations")
        annotation = resume if resume and resume.is_file() else paths.get("annotations")
        if annotation is None or not annotation.is_file():
            raise ValueError("Task annotations are missing")
        payload = json.loads(annotation.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict) or "tracks" not in payload or "hoi_events" not in payload:
            raise ValueError("Task requires editor-format annotations; review.json is retained separately")
        if payload.get("video_id") and str(payload["video_id"]) != str(data.get("trial_id")):
            raise ValueError("Annotation identity differs from the task")
        if annotation and annotation.is_file():
            if int(payload.get("frame_count", -1)) != int(self.player.frame_count):
                raise ValueError("Saved work frame count differs from this video")
            if abs(float(payload.get("fps", -1))-float(self.player.frame_rate)) > 0.01:
                raise ValueError("Saved work FPS differs from this video")
        self._reserve_autosave(video_path)
        self._load_annotations_v2(payload, annotation_path=str(annotation))
        self._apply_profile_libraries()
        self.current_annotation_path = str(resume or annotation)
        self._task_resume_path = str(resume) if resume else ""
        self._refresh_label_explanations()
        self._mark_query_calibration_dirty()
        self._mark_hoi_saved()
        self._log("hoi_task_bundle_loaded", manifest=str(task["manifest"]), annotations=str(annotation))
        self._update_onboarding_banner()
        self._open_autosave(str(annotation))
        if checkpoint and not checkpoint.is_file():
            QMessageBox.warning(self, "Tracking weights missing", "Annotations loaded. Download the bundle weights before using Track Correction.")
        return True

    def _auto_load_local_assets_for_video(self, video_path: str) -> None:
        path = str(video_path or "").strip()
        if not path:
            return
        loaded_assets: List[str] = []
        manual_mode = self._manual_mode_enabled()

        targets_path = self._pick_local_asset_candidate(path, "targets")
        if targets_path and not self._has_loaded_targets():
            if self._load_targets_from_path(targets_path, notify_user=False, auto_discovered=True) > 0:
                loaded_assets.append(f"Nouns: {os.path.basename(targets_path)}")

        verbs_path = self._pick_local_asset_candidate(path, "verbs")
        if verbs_path and not self._has_loaded_verbs():
            if self._load_verbs_from_path(verbs_path, notify_user=False, auto_discovered=True) > 0:
                loaded_assets.append(f"Verbs: {os.path.basename(verbs_path)}")

        ontology_path = self._pick_local_asset_candidate(path, "ontology")
        if ontology_path and not self._has_loaded_ontology():
            if self._load_ontology_from_path(ontology_path, notify_user=False, auto_discovered=True):
                loaded_assets.append(f"Ontology: {os.path.basename(ontology_path)}")

        if not manual_mode:
            videomae_cache_path = self._pick_local_asset_candidate(path, "videomae_cache")
            if videomae_cache_path and not self._videomae_precomputed_ready():
                if self._load_precomputed_videomae_cache(
                    videomae_cache_path,
                    notify_user=False,
                    auto_discovered=True,
                ):
                    loaded_assets.append(f"VideoMAE cache: {os.path.basename(videomae_cache_path)}")

            class_map_path = self._pick_local_asset_candidate(path, "class_map")
            if class_map_path and not self._has_loaded_class_map():
                if self._load_yaml_from_path(class_map_path, notify_user=False, auto_discovered=True):
                    loaded_assets.append(f"Class map: {os.path.basename(class_map_path)}")

            yolo_path = self._pick_local_asset_candidate(path, "yolo_model")
            if yolo_path and not self._has_loaded_yolo_model():
                if self._load_yolo_model_from_path(yolo_path, notify_user=False, auto_discovered=True):
                    loaded_assets.append(f"YOLO: {os.path.basename(yolo_path)}")

            videomae_weights_path = self._pick_local_asset_candidate(path, "videomae_weights")
            if videomae_weights_path and not self._has_loaded_videomae_weights():
                if self._store_videomae_weights_from_path(videomae_weights_path, notify_user=False, auto_discovered=True):
                    loaded_assets.append(f"VideoMAE weights: {os.path.basename(videomae_weights_path)}")

            videomae_verbs_path = self._pick_local_asset_candidate(path, "videomae_verbs")
            if videomae_verbs_path and not self._has_loaded_videomae_verb_list():
                if self._store_videomae_verb_list_from_path(videomae_verbs_path, notify_user=False, auto_discovered=True):
                    loaded_assets.append(f"VideoMAE verbs: {os.path.basename(videomae_verbs_path)}")

            semantic_adapter_path = self._pick_local_asset_candidate(path, "semantic_adapter")
            if semantic_adapter_path and not self._has_loaded_semantic_adapter():
                if self._load_semantic_adapter_from_path(semantic_adapter_path, notify_user=False, auto_discovered=True):
                    loaded_assets.append(f"Semantic adapter: {os.path.basename(semantic_adapter_path)}")

        if not loaded_assets:
            return
        self._update_onboarding_banner()
        self._log(
            "hoi_auto_load_local_assets",
            video_path=path,
            loaded_assets=list(loaded_assets),
            loaded_count=len(loaded_assets),
        )
        QMessageBox.information(
            self,
            "Auto Load",
            "Loaded project assets from the video folder:\n\n" + "\n".join(loaded_assets),
        )

    def _apply_noun_only_mode_ui(self) -> None:
        noun_only = True
        self._noun_only_mode = True
        if hasattr(self, "rad_draw_inst"):
            self.rad_draw_inst.setVisible(True)
            self.rad_draw_inst.setText("Instrument")
            self.rad_draw_inst.setToolTip(
                "New boxes inherit the selected event's instrument/tool label."
            )
        if hasattr(self, "rad_draw_target"):
            self.rad_draw_target.setText("Noun")
            self.rad_draw_target.setToolTip(
                "New boxes inherit the selected event's noun/object label."
            )
        if hasattr(self, "_noun_form_label") and self._noun_form_label is not None:
            self._noun_form_label.setText("Noun")
        if hasattr(self, "lbl_event_meta"):
            self.lbl_event_meta.setText("Verb: ?   Noun: ?   Instrument: ?   Anomaly: ?")

    def _semantic_verb_labels(self) -> List[str]:
        return [str(v.name) for v in sorted(self.verbs, key=lambda row: int(getattr(row, "id", 0)))]

    def _semantic_noun_ids(self) -> List[int]:
        return self._active_noun_ids()

    def _semantic_feature_layout(self) -> Dict[str, int]:
        verb_dim = len(self._semantic_verb_labels())
        noun_dim = len(self._semantic_noun_ids())
        scalar_dim = 10
        global_verb_offset = scalar_dim
        local_verb_offset = global_verb_offset + verb_dim
        noun_support_offset = local_verb_offset + verb_dim
        noun_onset_support_offset = noun_support_offset + noun_dim
        videomae_feature_offset = noun_onset_support_offset + noun_dim
        videomae_feature_dim = int(getattr(self, "_semantic_videomae_feature_dim", 0) or 0)
        videomae_local_feature_offset = videomae_feature_offset + videomae_feature_dim
        videomae_local_feature_dim = int(getattr(self, "_semantic_videomae_local_feature_dim", 0) or 0)
        return {
            "scalar_dim": int(scalar_dim),
            "global_verb_offset": int(global_verb_offset),
            "global_verb_dim": int(verb_dim),
            "local_verb_offset": int(local_verb_offset),
            "local_verb_dim": int(verb_dim),
            "noun_support_offset": int(noun_support_offset),
            "noun_support_dim": int(noun_dim),
            "noun_onset_support_offset": int(noun_onset_support_offset),
            "noun_onset_support_dim": int(noun_dim),
            "videomae_feature_offset": int(videomae_feature_offset),
            "videomae_feature_dim": int(videomae_feature_dim),
            "videomae_local_feature_offset": int(videomae_local_feature_offset),
            "videomae_local_feature_dim": int(videomae_local_feature_dim),
        }

    def _semantic_feature_dim_expected(self) -> int:
        layout = self._semantic_feature_layout()
        return int(
            layout["videomae_local_feature_offset"]
            + layout["videomae_local_feature_dim"]
        )

    def _primary_local_onset_context_for_event(self, event: Optional[dict]) -> Dict[str, Any]:
        if not event:
            return {}
        start, end = self._compute_event_frames(event)
        if start is None or end is None:
            return {}
        onset_context = self._primary_onset_context_for_event(event)
        return build_local_onset_window(
            start,
            end,
            onset_frame=onset_context.get("onset_frame"),
            onset_band=onset_context.get("onset_band"),
        )

    def _primary_videomae_scores_for_event(
        self,
        event: Optional[dict],
    ) -> Dict[str, float]:
        event = event or {}
        global_scores = {
            str(item.get("label") or ""): float(item.get("score") or 0.0)
            for item in self._normalize_videomae_candidates(event.get("videomae_top5"))
        }
        local_scores = {
            str(item.get("label") or ""): float(item.get("score") or 0.0)
            for item in self._normalize_videomae_candidates(event.get("videomae_local_top5"))
        }
        merged: Dict[str, float] = {}
        keys = set(global_scores.keys()) | set(local_scores.keys())
        for key in keys:
            merged[str(key)] = 0.65 * float(global_scores.get(key, 0.0)) + 0.35 * float(local_scores.get(key, 0.0))
        return merged

    def _cached_videomae_local_segment_feature(
        self,
        event_id: int,
        event: Optional[dict],
    ) -> List[float]:
        signature = self._videomae_signature_for_event(event)
        if not signature:
            return [0.0] * int(getattr(self, "_semantic_videomae_local_feature_dim", 0) or 0)
        rows = self._videomae_local_feature_cache.get(signature)
        return self._normalize_dense_feature(
            rows,
            length=int(getattr(self, "_semantic_videomae_local_feature_dim", 0) or 0),
        )

    def _cached_videomae_local_candidates(
        self,
        event_id: int,
        event: Optional[dict],
    ) -> List[dict]:
        signature = self._videomae_signature_for_event(event)
        if not signature:
            return []
        rows = []
        if event:
            rows = self._normalize_videomae_candidates(event.get("videomae_local_top5"))
        if not rows:
            rows = self._normalize_videomae_candidates(self._videomae_local_action_cache.get(signature))
        return rows

    def _normalize_dense_feature(
        self,
        values: Any,
        *,
        length: Optional[int] = None,
    ) -> List[float]:
        target_len = max(0, int(length if length is not None else getattr(self, "_semantic_videomae_feature_dim", 0) or 0))
        raw = list(values or [])
        out: List[float] = []
        for idx in range(target_len):
            if idx < len(raw):
                try:
                    out.append(float(raw[idx]))
                except Exception:
                    out.append(0.0)
            else:
                out.append(0.0)
        return out

    def _cached_videomae_segment_feature(
        self,
        event_id: int,
        event: Optional[dict],
    ) -> List[float]:
        signature = self._videomae_signature_for_event(event)
        if not signature:
            return [0.0] * int(getattr(self, "_semantic_videomae_feature_dim", 0) or 0)
        rows = self._videomae_feature_cache.get(signature)
        return self._normalize_dense_feature(
            rows,
            length=int(getattr(self, "_semantic_videomae_feature_dim", 0) or 0),
        )

    def _append_jsonl_row(self, path: str, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _semantic_feedback_schema(self) -> Dict[str, Any]:
        feature_layout = self._semantic_feature_layout()
        return {
            "verb_labels": self._semantic_verb_labels(),
            "noun_ids": self._semantic_noun_ids(),
            "feature_dim": int(self._semantic_feature_dim_expected()),
            "feature_layout": dict(feature_layout),
            "videomae_feature_dim": int(getattr(self, "_semantic_videomae_feature_dim", 0) or 0),
            "videomae_local_feature_dim": int(getattr(self, "_semantic_videomae_local_feature_dim", 0) or 0),
            "video_adapter_rank": int(self._semantic_adapter_train_config.get("video_adapter_rank", 0) or 0),
            "video_adapter_alpha": float(self._semantic_adapter_train_config.get("video_adapter_alpha", 0.0) or 0.0),
            "annotation_fields": [
                "interaction_start",
                "functional_contact_onset",
                "interaction_end",
                "verb",
                "noun_object_id",
                "instrument_object_id",
                "anomaly_label",
            ],
            "model_target_fields": [
                "functional_contact_onset",
                "verb",
                "noun_object_id",
            ],
        }

    def _semantic_refine_feature(
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
        layout = self._semantic_feature_layout()
        if refined_onset_ratio is not None:
            values[0] = self._bounded01(refined_onset_ratio, 0.5)
            values[1] = 1.0
            if values[2] < 0.5:
                values[3] = 1.0
        refined_verb = str(refined_verb or "").strip()
        verb_labels = self._semantic_verb_labels()
        if refined_verb and refined_verb in verb_labels:
            values[7] = 1.0 if self._noun_required_for_verb(refined_verb) else 0.0
            values[9] = float(self._noun_exists_prior_for_verb(refined_verb))
            verb_offset = 10
            verb_pos = verb_offset + verb_labels.index(refined_verb)
            if 0 <= verb_pos < len(values):
                values[verb_pos] = max(float(values[verb_pos]), 0.85)
            local_verb_offset = verb_offset + len(verb_labels)
            local_verb_pos = local_verb_offset + verb_labels.index(refined_verb)
            if 0 <= local_verb_pos < len(values):
                values[local_verb_pos] = max(float(values[local_verb_pos]), 0.90)
        if refined_noun_exists is not None:
            values[8] = 1.0 if bool(refined_noun_exists) else 0.0
        try:
            refined_noun_object_id = (
                int(refined_noun_object_id)
                if refined_noun_object_id is not None
                else None
            )
        except Exception:
            refined_noun_object_id = None
        noun_ids = self._semantic_noun_ids()
        if refined_noun_object_id is not None and refined_noun_object_id in noun_ids:
            noun_index = noun_ids.index(refined_noun_object_id)
            support_offset = int(layout.get("noun_support_offset", 0) or 0)
            support_dim = int(layout.get("noun_support_dim", 0) or 0)
            onset_support_offset = int(layout.get("noun_onset_support_offset", 0) or 0)
            onset_support_dim = int(layout.get("noun_onset_support_dim", 0) or 0)
            support_pos = support_offset + noun_index
            onset_support_pos = onset_support_offset + noun_index
            if 0 <= support_pos < min(len(values), support_offset + support_dim):
                values[support_pos] = max(float(values[support_pos]), 0.88)
            if 0 <= onset_support_pos < min(len(values), onset_support_offset + onset_support_dim):
                values[onset_support_pos] = max(float(values[onset_support_pos]), 0.94)
        return values

    def _semantic_feature_sample(
        self,
        event: Optional[dict],
        hand_key: str,
        hand_data: Optional[dict],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(event, dict) or not isinstance(hand_data, dict):
            return None
        start = hand_data.get("interaction_start")
        end = hand_data.get("interaction_end")
        if start is None or end is None:
            return None
        try:
            start = int(start)
            end = int(end)
        except Exception:
            return None
        if end < start:
            start, end = end, start
        segment_len = max(1, end - start)
        onset = hand_data.get("functional_contact_onset")
        onset_ratio = 0.5
        has_onset = 0.0
        if onset is not None:
            try:
                onset_ratio = self._bounded01((int(onset) - start) / float(segment_len), 0.5)
                has_onset = 1.0
            except Exception:
                onset_ratio = 0.5
        onset_state = get_field_state(hand_data, "functional_contact_onset")
        sparse_summary = self._sparse_evidence_summary(hand_data)
        candidates = self._normalize_videomae_candidates(event.get("videomae_top5"))
        candidate_scores = {str(item.get("label")): float(item.get("score") or 0.0) for item in candidates}
        local_candidates = self._normalize_videomae_candidates(event.get("videomae_local_top5"))
        local_candidate_scores = {str(item.get("label")): float(item.get("score") or 0.0) for item in local_candidates}
        support_map: Dict[int, float] = {}
        onset_support_map: Dict[int, float] = {}
        if self._semantic_object_support_enabled():
            object_candidates = self._collect_event_object_candidates(hand_key, hand_data)
            support_map = {
                int(item.get("object_id")): float(
                    item.get("support_score", item.get("support", 0.0)) or 0.0
                )
                for item in object_candidates
                if item.get("object_id") is not None
            }
            onset_support_map = {
                int(item.get("object_id")): float(
                    item.get(
                        "onset_support_score",
                        item.get("onset_support", 0.0),
                    )
                    or 0.0
                )
                for item in object_candidates
                if item.get("object_id") is not None
            }
        max_support = max([1.0] + list(support_map.values()))
        max_onset_support = max([1.0] + list(onset_support_map.values()))
        verb_labels = self._semantic_verb_labels()
        noun_ids = self._semantic_noun_ids()
        noun_id = self._hand_noun_object_id(hand_data)
        feature: List[float] = [
            self._bounded01(onset_ratio, 0.5),
            has_onset,
            1.0 if str(onset_state.get("status") or "").strip().lower() == "confirmed" else 0.0,
            1.0 if str(onset_state.get("status") or "").strip().lower() == "suggested" else 0.0,
            self._bounded01(segment_len / 240.0),
            self._bounded01(float(sparse_summary.get("confirmed", 0) or 0) / float(max(1, sparse_summary.get("expected", 0) or 0))),
            self._bounded01(float(sparse_summary.get("missing", 0) or 0) / float(max(1, sparse_summary.get("expected", 0) or 0))),
            1.0 if self._noun_required_for_verb(hand_data.get("verb")) else 0.0,
            1.0 if noun_id is not None else 0.0,
            float(self._noun_exists_prior_for_verb(hand_data.get("verb"))),
        ]
        feature.extend(float(candidate_scores.get(label, 0.0)) for label in verb_labels)
        feature.extend(float(local_candidate_scores.get(label, 0.0)) for label in verb_labels)
        feature.extend(
            float(support_map.get(noun_id, 0.0)) / float(max_support)
            for noun_id in noun_ids
        )
        feature.extend(
            float(onset_support_map.get(noun_id, 0.0)) / float(max_onset_support)
            for noun_id in noun_ids
        )
        feature.extend(
            self._cached_videomae_segment_feature(int(event.get("event_id") or -1), event)
        )
        feature.extend(
            self._cached_videomae_local_segment_feature(int(event.get("event_id") or -1), event)
        )

        verb_name = str(hand_data.get("verb") or "").strip()
        verb_index = verb_labels.index(verb_name) if verb_name in verb_labels else -1
        noun_index = noun_ids.index(int(noun_id)) if noun_id is not None and int(noun_id) in noun_ids else -1
        instrument_id = self._hand_instrument_object_id(hand_data)
        anomaly_label = self._normalize_anomaly_label(hand_data.get("anomaly_label"))
        sample = {
            "feature": [float(v) for v in feature],
            "targets": {
                "onset_ratio": self._bounded01(onset_ratio, 0.5),
                "verb_index": int(verb_index),
                "noun_exists": 1.0 if noun_id is not None else 0.0,
                "noun_index": int(noun_index),
            },
            "meta": {
                "event_id": event.get("event_id"),
                "hand": hand_key,
                "verb": verb_name,
                "noun_object_id": noun_id,
                "target_object_id": noun_id,
                "instrument_object_id": instrument_id,
                "tool_object_id": instrument_id,
                "anomaly_label": anomaly_label,
                "has_anomaly": self._is_anomalous_label(anomaly_label),
                "video_path": os.path.abspath(str(self.video_path or "").strip()) if str(self.video_path or "").strip() else "",
                "window": {
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "onset_frame": None if onset is None else int(onset),
                },
                "videomae_signature": str(self._videomae_signature_for_event(event) or ""),
                "onset_context": dict(self._primary_onset_context_for_event(event) or {}),
                "local_onset_context": dict(self._primary_local_onset_context_for_event(event) or {}),
                "event_hands": self._event_hand_runtime_summary(event, focus_hand=hand_key),
                "cross_hand_context": self._cross_hand_context_for_event(event, hand_key, hand_data),
                "noun_required": self._noun_required_for_verb(verb_name),
                "schema": self._semantic_feedback_schema(),
                "feature_layout": dict(self._semantic_feature_layout()),
                "videomae_feature_dim": int(getattr(self, "_semantic_videomae_feature_dim", 0) or 0),
                "videomae_local_feature_dim": int(getattr(self, "_semantic_videomae_local_feature_dim", 0) or 0),
            },
        }
        self._semantic_feedback_feature_dim = len(sample["feature"])
        return sample

    def _semantic_adapter_matches_runtime(self, package) -> bool:
        if package is None:
            return False
        if int(getattr(package, "feature_dim", 0) or 0) != int(self._semantic_feature_dim_expected()):
            return False
        if list(package.verb_labels or []) != self._semantic_verb_labels():
            return False
        if [int(v) for v in list(package.noun_ids or [])] != self._semantic_noun_ids():
            return False
        package_layout = dict(getattr(package, "feature_layout", {}) or {})
        if package_layout:
            runtime_layout = self._semantic_feature_layout()
            for key in (
                "videomae_feature_offset",
                "videomae_feature_dim",
                "videomae_local_feature_offset",
                "videomae_local_feature_dim",
            ):
                if int(package_layout.get(key, 0) or 0) != int(runtime_layout.get(key, 0) or 0):
                    return False
        return True

    def _ensure_semantic_adapter_loaded(self) -> None:
        participant_model_path = self._semantic_model_file()
        base_model_path = str(getattr(self, "semantic_adapter_base_model_path", "") or "").strip()
        active_model_path = str(getattr(self, "semantic_adapter_active_model_path", "") or "").strip()
        shared_model_path = str(self._shared_semantic_adapter_file() or "").strip()
        if (
            self.semantic_adapter_package is not None
            and self._semantic_adapter_matches_runtime(self.semantic_adapter_package)
            and active_model_path
        ):
            if os.path.isfile(active_model_path):
                return
            self.semantic_adapter_package = None
            self.semantic_adapter_active_model_path = ""

        candidate_paths: List[str] = []
        if participant_model_path:
            candidate_paths.append(participant_model_path)
        if (
            base_model_path
            and os.path.normcase(base_model_path)
            != os.path.normcase(participant_model_path or "")
        ):
            candidate_paths.append(base_model_path)
        if (
            shared_model_path
            and os.path.normcase(shared_model_path)
            != os.path.normcase(participant_model_path or "")
            and os.path.normcase(shared_model_path)
            != os.path.normcase(base_model_path or "")
        ):
            candidate_paths.append(shared_model_path)

        for model_path in candidate_paths:
            if not model_path or not os.path.isfile(model_path):
                continue
            load_key = self._semantic_adapter_load_key(model_path)
            if load_key and load_key in getattr(self, "_semantic_adapter_failed_load_paths", set()):
                continue
            package = self._load_semantic_adapter_package_safely(model_path)
            if package is not None and self._semantic_adapter_matches_runtime(package):
                self.semantic_adapter_package = package
                self.semantic_adapter_active_model_path = model_path
                if not base_model_path and os.path.normcase(model_path) == os.path.normcase(shared_model_path or ""):
                    self.semantic_adapter_base_model_path = model_path
                return
        self.semantic_adapter_package = None
        self.semantic_adapter_active_model_path = ""

    def _semantic_source_is_runtime_lock(self, source: Any) -> bool:
        text = str(source or "").strip().lower()
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

    def _semantic_runtime_constraints(
        self,
        event: Optional[dict],
        hand_data: Optional[dict],
    ) -> Dict[str, Any]:
        constraints: Dict[str, Any] = {
            "clamp_onset_ratio": None,
            "clamp_verb_label": "",
            "clamp_noun_exists": None,
            "clamp_noun_object_id": None,
            "clamp_unknown_noun": False,
            "locked_fields": [],
        }
        if not isinstance(event, dict) or not isinstance(hand_data, dict):
            return constraints
        start = hand_data.get("interaction_start")
        end = hand_data.get("interaction_end")
        try:
            start = int(start)
            end = int(end)
        except Exception:
            start = None
            end = None
        if start is not None and end is not None and end < start:
            start, end = end, start
        segment_len = max(1, int(end) - int(start)) if start is not None and end is not None else 1

        onset = hand_data.get("functional_contact_onset")
        onset_state = get_field_state(hand_data, "functional_contact_onset")
        onset_source = str(onset_state.get("source") or "").strip()
        if onset is not None and start is not None:
            if onset_state.get("status") == "confirmed" or self._semantic_source_is_runtime_lock(onset_source):
                try:
                    constraints["clamp_onset_ratio"] = self._bounded01((int(onset) - int(start)) / float(segment_len), 0.5)
                    constraints["locked_fields"].append("onset")
                except Exception:
                    pass

        verb_name = str(hand_data.get("verb") or "").strip()
        verb_state = get_field_state(hand_data, "verb")
        verb_source = str(verb_state.get("source") or "").strip()
        if verb_name and (
            verb_state.get("status") == "confirmed"
            or self._semantic_source_is_runtime_lock(verb_source)
        ):
            constraints["clamp_verb_label"] = verb_name
            constraints["locked_fields"].append("verb")

        noun_state = get_field_state(hand_data, "noun_object_id")
        noun_source = str(noun_state.get("source") or "").strip()
        noun_value = self._hand_noun_object_id(hand_data)
        if noun_value is not None and (
            noun_state.get("status") == "confirmed"
            or self._semantic_source_is_runtime_lock(noun_source)
        ):
            try:
                constraints["clamp_noun_exists"] = True
                constraints["clamp_noun_object_id"] = int(noun_value)
                constraints["locked_fields"].append("noun")
            except Exception:
                pass
        elif self._hand_has_explicit_no_noun_lock(hand_data, verb_name=verb_name):
            constraints["clamp_noun_exists"] = False
            constraints["locked_fields"].append("noun_absent")
        return constraints

    def _record_semantic_feedback(
        self,
        event: Optional[dict],
        hand_key: str,
        *,
        reason: str,
        before_hand: Optional[dict] = None,
        edited_fields: Optional[Sequence[str]] = None,
        accepted_fields: Optional[Sequence[str]] = None,
        supervision_kind: str = "",
        resolve_kind: str = "",
        authority_level: str = "",
        query_id: str = "",
    ) -> None:
        hand_after = ((event or {}).get("hoi_data", {}) or {}).get(hand_key)
        sample = self._semantic_feature_sample(event, hand_key, hand_after)
        if not sample:
            return
        edited = {
            self._canonical_query_field_name(name)
            for name in list(edited_fields or [])
            if self._canonical_query_field_name(name)
        }
        accepted = {
            self._canonical_query_field_name(name)
            for name in list(accepted_fields or [])
            if self._canonical_query_field_name(name)
        }
        if isinstance(before_hand, dict) and isinstance(hand_after, dict):
            for field_name in self._semantic_feedback_fields():
                before_snapshot = self._field_snapshot(before_hand, field_name)
                after_snapshot = self._field_snapshot(hand_after, field_name)
                if (
                    before_snapshot.get("value") != after_snapshot.get("value")
                    or before_snapshot.get("status") != after_snapshot.get("status")
                    or before_snapshot.get("source") != after_snapshot.get("source")
                ):
                    edited.add(str(field_name))
        untouched_fields: List[str] = []
        if isinstance(hand_after, dict):
            for field_name in self._semantic_feedback_fields():
                if field_name in edited or field_name in accepted:
                    continue
                snapshot = self._field_snapshot(hand_after, field_name)
                if snapshot.get("value") is not None or str(snapshot.get("status") or "").strip():
                    untouched_fields.append(str(field_name))
        supervision_kind = str(supervision_kind or "").strip() or (
            "accepted"
            if accepted
            else ("edited" if edited else "passive")
        )
        sample["meta"]["reason"] = str(reason or "").strip()
        sample["meta"]["recorded_at"] = datetime.now().isoformat(timespec="seconds")
        sample["meta"]["edited_fields"] = sorted(str(name) for name in edited if name)
        sample["meta"]["accepted_fields"] = sorted(str(name) for name in accepted if name)
        sample["meta"]["untouched_fields"] = sorted(str(name) for name in untouched_fields if name)
        sample["meta"]["supervision_kind"] = supervision_kind
        sample["meta"]["resolve_kind"] = str(resolve_kind or "").strip()
        sample["meta"]["authority_level"] = str(authority_level or "").strip()
        sample["meta"]["query_id"] = str(query_id or "").strip()
        sample["meta"]["participant_code"] = self._normalized_participant_code()
        sample["meta"]["semantic_model_path"] = self._semantic_model_file()
        self._append_jsonl_row(self._semantic_feedback_file(), sample)
        self._semantic_feedback_pending += 1
        self._log(
            "hoi_semantic_feedback_record",
            reason=reason,
            event_id=sample["meta"].get("event_id"),
            hand=hand_key,
            feature_dim=len(sample.get("feature") or []),
            supervision_kind=supervision_kind,
            edited_fields=sample["meta"].get("edited_fields"),
            accepted_fields=sample["meta"].get("accepted_fields"),
        )
        self._maybe_schedule_semantic_training()

    def _maybe_schedule_semantic_training(self) -> None:
        train_every = int(self._semantic_adapter_train_config.get("train_every", 6) or 6)
        min_samples = int(self._semantic_adapter_train_config.get("min_samples", 8) or 8)
        if self._semantic_feedback_pending < train_every:
            return
        if self._semantic_feedback_feature_dim <= 0:
            return
        if len(self._semantic_verb_labels()) <= 0:
            return
        if self._semantic_adapter_train_worker is not None and self._semantic_adapter_train_worker.isRunning():
            return
        feedback_rows = 0
        try:
            feedback_rows = sum(1 for _ in open(self._semantic_feedback_file(), "r", encoding="utf-8"))
        except Exception:
            feedback_rows = 0
        if feedback_rows < min_samples:
            return
        init_model_path = str(getattr(self, "semantic_adapter_active_model_path", "") or "").strip()
        worker = SemanticAdapterTrainWorker(
            self._semantic_feedback_file(),
            self._semantic_model_file(),
            self._semantic_feedback_feature_dim,
            self._semantic_verb_labels(),
            self._semantic_noun_ids(),
            config={
                **dict(self._semantic_adapter_train_config or {}),
                "feature_layout": dict(self._semantic_feature_layout()),
            },
            init_model_path=init_model_path,
        )
        self._log(
            "hoi_semantic_training_started",
            participant_code=self._normalized_participant_code(),
            feedback_path=self._semantic_feedback_file(),
            model_path=self._semantic_model_file(),
            init_model_path=init_model_path,
            feedback_rows=int(feedback_rows),
        )
        worker.finished.connect(self._on_semantic_training_finished)
        self._semantic_adapter_train_worker = worker
        self._log_annotation_ready_state("hoi_semantic_training_started")
        worker.start()

    def _on_semantic_training_finished(self, ok: bool, message: str, package: object) -> None:
        worker = self.sender()
        worker_model_path = ""
        if worker is not None:
            worker_model_path = str(getattr(worker, "model_path", "") or "").strip()
        current_model_path = str(self._semantic_model_file() or "").strip()
        self._log(
            "hoi_semantic_training_finished",
            ok=bool(ok),
            message=message,
            model_path=worker_model_path or current_model_path,
            participant_code=self._normalized_participant_code(),
        )
        if worker_model_path and current_model_path and os.path.normcase(worker_model_path) != os.path.normcase(current_model_path):
            self._semantic_adapter_train_worker = None
            return
        if ok and package is not None and self._semantic_adapter_matches_runtime(package):
            self.semantic_adapter_package = package
            self.semantic_adapter_active_model_path = worker_model_path or current_model_path
            self._semantic_feedback_pending = 0
            if self.selected_event_id is not None:
                event = self._find_event_by_id(self.selected_event_id)
                if event is not None:
                    self._refresh_semantic_suggestions_for_event(self.selected_event_id, event)
                    self._refresh_selected_event_runtime_views(
                        self.selected_event_id,
                        refresh_boxes=True,
                        refresh_focus=False,
                    )
        self._semantic_adapter_train_worker = None
        self._log_annotation_ready_state("hoi_semantic_training_finished")

    def _semantic_review_recommended(
        self,
        kind: str,
        confidence: float,
        *,
        exists_prob: Optional[float] = None,
        band_width: Optional[float] = None,
        risk_score: Optional[float] = None,
    ) -> bool:
        thresholds = dict(getattr(self, "_semantic_review_thresholds", {}) or {})
        risk_limit = float(thresholds.get("risk_score", 0.34))
        if risk_score is not None and float(risk_score) >= risk_limit:
            return True
        if kind == "onset":
            if band_width is not None and float(band_width) > float(thresholds.get("onset_band_width", 0.22)):
                return True
            return float(confidence) < float(thresholds.get("onset_confidence", 0.86))
        if kind == "verb":
            return float(confidence) < float(thresholds.get("verb_confidence", 0.82))
        if kind == "noun":
            noun_conf = float(thresholds.get("noun_confidence", 0.80))
            exist_conf = float(thresholds.get("noun_exists", 0.62))
            if exists_prob is not None and float(exists_prob) < exist_conf:
                return True
            return float(confidence) < noun_conf
        return True

    def _refresh_semantic_suggestions_for_event(self, event_id: int, event: Optional[dict] = None) -> None:
        event = event or self._find_event_by_id(event_id)
        if not isinstance(event, dict):
            return
        if not self._semantic_assist_enabled():
            for actor in self.actors_config:
                hand_key = actor["id"]
                hand_data = event.get("hoi_data", {}).get(hand_key, {}) or {}
                if not isinstance(hand_data, dict):
                    continue
                self._ensure_hand_annotation_state(hand_data)
                self._clear_hand_runtime_suggestions(hand_data)
                if self.selected_event_id == int(event_id) and hand_key in self.event_draft:
                    self.event_draft[hand_key] = copy.deepcopy(hand_data)
            self._refresh_selected_event_runtime_views(
                event_id,
                refresh_boxes=True,
                refresh_focus=False,
            )
            return
        self._ensure_semantic_adapter_loaded()
        package = getattr(self, "semantic_adapter_package", None)
        for actor in self.actors_config:
            hand_key = actor["id"]
            hand_data = event.get("hoi_data", {}).get(hand_key, {}) or {}
            self._ensure_hand_annotation_state(hand_data)
            start = hand_data.get("interaction_start")
            end = hand_data.get("interaction_end")
            track_prior = self._handtrack_segment_prior(hand_key, start, end)
            sample = (
                self._semantic_feature_sample(event, hand_key, hand_data)
                if package is not None
                else None
            )
            if package is None or not sample:
                onset_state = get_field_state(hand_data, "functional_contact_onset")
                if (
                    track_prior
                    and onset_state.get("status") != "confirmed"
                    and start is not None
                    and end is not None
                ):
                    prior_conf = float(track_prior.get("confidence", 0.0) or 0.0)
                    review_recommended = bool(prior_conf < 0.78)
                    self._suggest_hand_field(
                        hand_data,
                        "functional_contact_onset",
                        int(track_prior.get("onset_frame", start) or start),
                        source="handtrack_once_onset",
                        confidence=prior_conf,
                        reason="Persistent hand-track motion proposes the most likely onset inside the current hand-conditioned segment.",
                        safe_to_apply=True,
                        review_recommended=review_recommended,
                        meta={
                            "onset_band": dict(track_prior.get("onset_band") or {}),
                            "handtrack_prior": dict(track_prior),
                            "risk_score": float(max(0.0, 1.0 - prior_conf)),
                        },
                    )
                    if hand_data.get("functional_contact_onset") is None or not review_recommended:
                        apply_field_suggestion(
                            hand_data,
                            "functional_contact_onset",
                            source="handtrack_once_onset",
                            as_status="suggested",
                        )
                if self.selected_event_id == int(event_id) and hand_key in self.event_draft:
                    self.event_draft[hand_key] = copy.deepcopy(hand_data)
                continue
            current_verb = str(hand_data.get("verb") or "").strip()
            verb_labels = self._semantic_verb_labels()
            allowed_noun_ids = self._allowed_noun_ids_for_verb(current_verb)
            allowed_nouns_by_verb = {
                str(label): list(self._allowed_noun_ids_for_verb(label))
                for label in verb_labels
            }
            allow_no_noun_by_verb = {
                str(label): (not self._noun_required_for_verb(label))
                for label in verb_labels
            }
            object_candidates = self._collect_event_object_candidates(
                hand_key,
                hand_data,
            )
            if self._semantic_object_support_enabled():
                support_map = {
                    int(item.get("object_id")): float(
                        item.get("support_score", item.get("support", 0.0)) or 0.0
                    )
                    for item in object_candidates
                    if item.get("object_id") is not None
                }
                onset_support_map = {
                    int(item.get("object_id")): float(
                        item.get(
                            "onset_support_score",
                            item.get("onset_support", 0.0),
                        )
                        or 0.0
                    )
                    for item in object_candidates
                    if item.get("object_id") is not None
                }
            else:
                support_map = {}
                onset_support_map = {}
            videomae_scores = self._primary_videomae_scores_for_event(event)
            runtime_constraints = self._semantic_runtime_constraints(event, hand_data)
            base_feature = [float(v) for v in list(sample.get("feature") or [])]
            reinfer_hint = self._consume_semantic_reinfer_hint(event_id, hand_key)
            anchor_onset_ratio = reinfer_hint.get("onset_anchor_ratio")
            anchor_onset_half_width = reinfer_hint.get("onset_anchor_half_width")
            anchor_onset_weight = (
                0.34 if anchor_onset_ratio is not None and runtime_constraints.get("clamp_onset_ratio") is None else 0.0
            )
            cross_hand_context = self._cross_hand_context_for_event(event, hand_key, hand_data)
            primary_exclusion = dict(cross_hand_context.get("primary_exclusion") or {})
            exclude_onset_ratio = primary_exclusion.get("onset_ratio")
            exclude_onset_half_width = 0.08 if primary_exclusion else None
            exclude_onset_weight = float(primary_exclusion.get("exclude_weight", 0.0) or 0.0)
            runtime_result = run_event_local_semantic_decode(
                SemanticRuntimeRequest(
                    feature=base_feature,
                    package=package,
                    allowed_noun_ids=allowed_noun_ids,
                    noun_required=self._noun_required_for_verb(current_verb),
                    allow_no_noun=(not self._noun_required_for_verb(current_verb)),
                    external_verb_scores=videomae_scores,
                    noun_support_scores=support_map,
                    noun_onset_support_scores=onset_support_map,
                    allowed_nouns_by_verb=allowed_nouns_by_verb,
                    allow_no_noun_by_verb=allow_no_noun_by_verb,
                    clamp_onset_ratio=runtime_constraints.get("clamp_onset_ratio"),
                    clamp_verb_label=runtime_constraints.get("clamp_verb_label") or "",
                    clamp_noun_exists=runtime_constraints.get("clamp_noun_exists"),
                    clamp_noun_object_id=runtime_constraints.get("clamp_noun_object_id"),
                    clamp_unknown_noun=bool(runtime_constraints.get("clamp_unknown_noun")),
                    anchor_onset_ratio=anchor_onset_ratio,
                    anchor_onset_half_width=anchor_onset_half_width,
                    anchor_onset_weight=anchor_onset_weight,
                    exclude_onset_ratio=exclude_onset_ratio,
                    exclude_onset_half_width=exclude_onset_half_width,
                    exclude_onset_weight=exclude_onset_weight,
                    refinement_passes=max(1, int(getattr(self, "_semantic_refinement_passes", 1) or 1)),
                    refine_feature_fn=self._semantic_refine_feature,
                )
            )
            prediction = dict(runtime_result.prediction or {})
            if not prediction:
                continue
            for pass_info in list(runtime_result.pass_trace or []):
                self._log(
                    "hoi_semantic_refine_pass",
                    event_id=event_id,
                    hand=hand_key,
                    pass_index=int(pass_info.get("pass_index", 0) or 0),
                    coarse_joint=float(pass_info.get("coarse_joint", 0.0) or 0.0),
                    refined_joint=float(pass_info.get("refined_joint", 0.0) or 0.0),
                    used_refined=bool(pass_info.get("used_refined")),
                )
            cooperative_meta = dict(prediction.get("cooperative_refinement") or {})
            if cooperative_meta:
                self._log(
                    "hoi_semantic_cooperation",
                    event_id=event_id,
                    hand=hand_key,
                    enabled=bool(cooperative_meta.get("enabled")),
                    reinfer_reason=str(reinfer_hint.get("reason") or ""),
                    reinfer_edited_fields=list(reinfer_hint.get("edited_fields") or []),
                    onset_anchor_ratio=anchor_onset_ratio,
                    onset_anchor_half_width=anchor_onset_half_width,
                    cross_hand_exclusive_count=int(cross_hand_context.get("exclusive_count", 0) or 0),
                    exclude_onset_ratio=exclude_onset_ratio,
                    exclude_onset_weight=exclude_onset_weight,
                    noun_exists_prior=cooperative_meta.get("noun_exists_prior"),
                    semantic_onset_prior_used=bool(cooperative_meta.get("semantic_onset_prior_used")),
                    verb_refined_from_noun=bool(cooperative_meta.get("verb_refined_from_noun")),
                    noun_refined_from_verb=bool(cooperative_meta.get("noun_refined_from_verb")),
                )
            start = hand_data.get("interaction_start")
            end = hand_data.get("interaction_end")
            if start is None or end is None:
                continue
            start = int(start)
            end = int(end)
            segment_len = max(1, end - start)
            structured = dict((prediction.get("structured") or {}).get("best") or {})
            structured_joint_conf = float(structured.get("joint_prob") or 0.0)
            structured_risk = float(structured.get("risk_score") or 0.0)
            structured_band = dict(structured.get("onset_band") or prediction.get("onset_band") or {})
            structured_band_width = float(
                structured.get("band_width")
                or (float(structured_band.get("right_ratio", 0.0)) - float(structured_band.get("left_ratio", 0.0)))
                or 0.0
            )
            structured_onset_ratio = float(structured.get("onset_ratio", prediction.get("onset_ratio", 0.5)) or 0.5)
            structured_verb = str(structured.get("verb_label") or "").strip()
            structured_noun_id = structured.get("noun_object_id")
            structured_noun_unknown = bool(structured.get("noun_is_unknown"))
            structured_noun_exists = bool(structured.get("noun_exists", False))
            noun_exists_prob = float(prediction.get("noun_exists_prob", 0.0) or 0.0)
            fused_onset_ratio = float(structured_onset_ratio)
            fused_onset_band = dict(structured_band)
            handtrack_conf = float(track_prior.get("confidence", 0.0) or 0.0)
            if track_prior and runtime_constraints.get("clamp_onset_ratio") is None:
                sem_weight = max(
                    1e-6,
                    0.55 * float(prediction.get("onset_confidence", 0.0) or 0.0)
                    + 0.45 * structured_joint_conf,
                )
                track_weight = max(1e-6, handtrack_conf)
                total_weight = sem_weight + track_weight
                fused_onset_ratio = self._bounded01(
                    (
                        sem_weight * float(structured_onset_ratio)
                        + track_weight * float(track_prior.get("onset_ratio", structured_onset_ratio) or structured_onset_ratio)
                    )
                    / float(max(1e-6, total_weight)),
                    structured_onset_ratio,
                )
                track_band = dict(track_prior.get("onset_band") or {})
                if track_band:
                    fused_onset_band = {
                        "center_ratio": float(fused_onset_ratio),
                        "left_ratio": self._bounded01(
                            (
                                sem_weight * float(structured_band.get("left_ratio", fused_onset_ratio) or fused_onset_ratio)
                                + track_weight * float(track_band.get("left_ratio", fused_onset_ratio) or fused_onset_ratio)
                            )
                            / float(max(1e-6, total_weight)),
                            fused_onset_ratio,
                        ),
                        "right_ratio": self._bounded01(
                            (
                                sem_weight * float(structured_band.get("right_ratio", fused_onset_ratio) or fused_onset_ratio)
                                + track_weight * float(track_band.get("right_ratio", fused_onset_ratio) or fused_onset_ratio)
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
                    structured_band_width = float(
                        fused_onset_band["right_ratio"] - fused_onset_band["left_ratio"]
                    )
            self._log(
                "hoi_semantic_redecode",
                event_id=event_id,
                hand=hand_key,
                clamp_fields=list(runtime_constraints.get("locked_fields") or []),
                best_onset_ratio=structured_onset_ratio,
                best_verb=structured_verb,
                best_noun_id=structured_noun_id,
                best_noun_exists=structured_noun_exists,
                joint_confidence=structured_joint_conf,
                risk_score=structured_risk,
            )

            onset_state = get_field_state(hand_data, "functional_contact_onset")
            if onset_state.get("status") != "confirmed":
                onset_frame = int(round(start + fused_onset_ratio * segment_len))
                onset_frame = max(start, min(end, onset_frame))
                onset_conf = (
                    0.45 * float(prediction.get("onset_confidence", 0.0) or 0.0)
                    + 0.35 * structured_joint_conf
                    + 0.20 * handtrack_conf
                )
                review_recommended = self._semantic_review_recommended(
                    "onset",
                    onset_conf,
                    band_width=structured_band_width,
                    risk_score=structured_risk,
                )
                reason = (
                    "System suggestion for the most likely onset inside the current hand segment."
                    if track_prior
                    else "System suggestion for the most likely onset inside the current segment."
                )
                self._suggest_hand_field(
                    hand_data,
                    "functional_contact_onset",
                    onset_frame,
                    source="semantic_adapter_onset",
                    confidence=onset_conf,
                    reason=reason,
                    safe_to_apply=True,
                    review_recommended=review_recommended,
                    meta={
                        "onset_band": dict(fused_onset_band),
                        "joint_confidence": structured_joint_conf,
                        "risk_score": structured_risk,
                        "handtrack_prior": dict(track_prior or {}),
                        "runtime_constraints": dict(prediction.get("runtime_constraints") or {}),
                    },
                )
                if hand_data.get("functional_contact_onset") is None or not review_recommended:
                    apply_field_suggestion(
                        hand_data,
                        "functional_contact_onset",
                        source="semantic_adapter_onset",
                        as_status="suggested",
                    )

            verb_candidates = list(prediction.get("verb_candidates") or [])
            if verb_candidates:
                top_label = structured_verb or str((verb_candidates[0] or {}).get("label") or "").strip()
                top_score = 0.0
                for row in verb_candidates:
                    if str(row.get("label") or "").strip() == top_label:
                        top_score = float(row.get("score") or 0.0)
                        break
                verb_state = get_field_state(hand_data, "verb")
                if verb_state.get("status") != "confirmed" and top_label:
                    verb_conf = 0.55 * float(top_score) + 0.45 * structured_joint_conf
                    review_recommended = self._semantic_review_recommended(
                        "verb",
                        verb_conf,
                        risk_score=structured_risk,
                    )
                    self._suggest_hand_field(
                        hand_data,
                        "verb",
                        top_label,
                        source="semantic_adapter_verb",
                        confidence=verb_conf,
                        reason="Structured semantic decoding refined the current verb suggestion using onset-aware temporal context.",
                        safe_to_apply=True,
                        review_recommended=review_recommended,
                        meta={
                            "joint_confidence": structured_joint_conf,
                            "risk_score": structured_risk,
                            "runtime_constraints": dict(prediction.get("runtime_constraints") or {}),
                        },
                    )
                    if not str(hand_data.get("verb") or "").strip() or not review_recommended:
                        apply_field_suggestion(
                            hand_data,
                            "verb",
                            source="semantic_adapter_verb",
                            as_status="suggested",
                        )

            noun_state = get_field_state(hand_data, "noun_object_id")
            noun_required = self._noun_required_for_verb(structured_verb or hand_data.get("verb"))
            noun_exists_threshold = float(
                prediction.get("noun_exists_threshold", self._semantic_review_thresholds.get("noun_exists", 0.62)) or 0.62
            )
            explicit_no_noun_lock = self._hand_has_explicit_no_noun_lock(
                hand_data,
                verb_name=(structured_verb or hand_data.get("verb")),
            )
            if explicit_no_noun_lock:
                clear_field_suggestion(hand_data, "noun_object_id")
            noun_should_exist = noun_required or structured_noun_exists or noun_exists_prob >= noun_exists_threshold
            if noun_state.get("status") != "confirmed" and noun_should_exist:
                if structured_noun_id is not None and not structured_noun_unknown:
                    noun_conf = structured_joint_conf
                    review_recommended = self._semantic_review_recommended(
                        "noun",
                        noun_conf,
                        exists_prob=noun_exists_prob,
                        risk_score=structured_risk,
                    )
                    noun_name = self._object_name_for_id(
                        int(structured_noun_id),
                        default_for_none="",
                        fallback=f"Object {int(structured_noun_id)}",
                    )
                    grounding_candidate = self._best_grounding_candidate_for_noun(
                        hand_key,
                        hand_data,
                        structured_noun_id,
                    )
                    source_decision = self._estimate_noun_source_decision(
                        hand_key,
                        hand_data,
                        semantic_noun_id=structured_noun_id,
                        semantic_confidence=noun_conf,
                        grounding_candidate=grounding_candidate,
                    )
                    previous_source_decision = dict(
                        hand_data.get("_noun_source_decision") or {}
                    )
                    hand_data["_noun_source_decision"] = dict(source_decision or {})
                    detector_preferred = str(
                        source_decision.get("preferred_family") or ""
                    ).strip().lower() == "detector_grounding"
                    decision_signature = (
                        str(source_decision.get("preferred_source") or ""),
                        int(source_decision.get("semantic_noun_id", -1) or -1),
                        int(source_decision.get("detector_noun_id", -1) or -1),
                        round(float(source_decision.get("score_margin", 0.0) or 0.0), 4),
                    )
                    previous_signature = (
                        str(previous_source_decision.get("preferred_source") or ""),
                        int(previous_source_decision.get("semantic_noun_id", -1) or -1),
                        int(previous_source_decision.get("detector_noun_id", -1) or -1),
                        round(float(previous_source_decision.get("score_margin", 0.0) or 0.0), 4),
                    )
                    if decision_signature != previous_signature:
                        self._log(
                            "hoi_noun_source_decision",
                            event_id=event_id,
                            hand=hand_key,
                            semantic_noun_id=source_decision.get("semantic_noun_id"),
                            detector_noun_id=source_decision.get("detector_noun_id"),
                            preferred_source=source_decision.get("preferred_source"),
                            preferred_family=source_decision.get("preferred_family"),
                            score_margin=source_decision.get("score_margin"),
                            semantic_source_acceptance_est=source_decision.get("source_a_acceptance"),
                            detector_source_acceptance_est=source_decision.get("source_b_acceptance"),
                            semantic_source_score=source_decision.get("source_a_score"),
                            detector_source_score=source_decision.get("source_b_score"),
                            semantic_source_support=source_decision.get("source_a_support"),
                            detector_source_support=source_decision.get("source_b_support"),
                            decision_basis=source_decision.get("decision_basis"),
                        )
                    current_suggestion = get_field_suggestion(hand_data, "noun_object_id")
                    current_state = get_field_state(hand_data, "noun_object_id")
                    current_source = str(current_suggestion.get("source") or "").strip().lower()
                    if not current_source:
                        current_source = str(current_state.get("source") or "").strip().lower()
                    current_status = str(current_state.get("status") or "").strip().lower()
                    semantic_suggested_value = bool(
                        current_status == "suggested"
                        and current_source.startswith("semantic_adapter")
                    )
                    if detector_preferred and (
                        self._hand_noun_object_id(hand_data) is None or semantic_suggested_value
                    ):
                        if semantic_suggested_value:
                            self._clear_hand_field(
                                hand_data,
                                "noun_object_id",
                                source="detector_grounding_preferred",
                            )
                        if current_source.startswith("semantic_adapter"):
                            clear_field_suggestion(hand_data, "noun_object_id")
                    else:
                        self._suggest_hand_field(
                            hand_data,
                            "noun_object_id",
                            int(structured_noun_id),
                            source="semantic_adapter_noun",
                            confidence=noun_conf,
                            reason="Structured semantic decoding refined the noun suggestion under the current verb-noun ontology.",
                            safe_to_apply=True,
                            review_recommended=review_recommended,
                            meta={
                                "display_value": noun_name,
                                "noun_exists_prob": noun_exists_prob,
                                "joint_confidence": structured_joint_conf,
                                "risk_score": structured_risk,
                                "noun_is_unknown": False,
                                "grounding_candidate": dict(grounding_candidate or {}),
                                "source_decision": dict(source_decision or {}),
                                "runtime_constraints": dict(prediction.get("runtime_constraints") or {}),
                            },
                        )
                        if self._hand_noun_object_id(hand_data) is None or not review_recommended:
                            apply_field_suggestion(
                                hand_data,
                                "noun_object_id",
                                source="semantic_adapter_noun",
                                as_status="suggested",
                            )
                elif structured_noun_unknown:
                    hand_data["_noun_source_decision"] = {}
                    self._log(
                        "hoi_semantic_unknown_noun",
                        event_id=event_id,
                        hand=hand_key,
                        verb=structured_verb or hand_data.get("verb"),
                        noun_exists_prob=noun_exists_prob,
                        joint_confidence=structured_joint_conf,
                        risk_score=structured_risk,
                    )
            elif (
                noun_state.get("status") != "confirmed"
                and not noun_should_exist
                and self._hand_noun_object_id(hand_data) is None
                and self._verb_allows_no_noun(structured_verb or hand_data.get("verb"))
            ):
                hand_data["_noun_source_decision"] = {}
                no_noun_conf = self._bounded01(
                    0.55 * (1.0 - noun_exists_prob) + 0.45 * structured_joint_conf,
                    0.0,
                )
                risk_limit = float(
                    self._semantic_review_thresholds.get("risk_score", 0.34) or 0.34
                )
                noun_conf_limit = float(
                    self._semantic_review_thresholds.get("noun_confidence", 0.80)
                    or 0.80
                )
                review_recommended = bool(
                    structured_risk >= risk_limit or no_noun_conf < noun_conf_limit
                )
                self._suggest_hand_field(
                    hand_data,
                    "noun_object_id",
                    None,
                    source="semantic_adapter_no_noun",
                    confidence=no_noun_conf,
                    reason="Structured semantic decoding predicts that the current verb can be completed without a noun/object under the ontology.",
                    safe_to_apply=True,
                    review_recommended=review_recommended,
                    meta={
                        "explicit_empty": True,
                        "display_value": str(NO_NOUN_TOKEN or "No noun / object"),
                        "noun_exists_prob": noun_exists_prob,
                        "joint_confidence": structured_joint_conf,
                        "risk_score": structured_risk,
                        "runtime_constraints": dict(
                            prediction.get("runtime_constraints") or {}
                        ),
                    },
                )
                apply_field_suggestion(
                    hand_data,
                    "noun_object_id",
                    source="semantic_adapter_no_noun",
                    as_status="suggested",
                )
            elif not noun_should_exist and self._hand_noun_object_id(hand_data) is None:
                hand_data["_noun_source_decision"] = {}
                clear_field_suggestion(hand_data, "noun_object_id")

            if self.selected_event_id == int(event_id) and hand_key in self.event_draft:
                self.event_draft[hand_key] = copy.deepcopy(hand_data)
        self._refresh_selected_event_runtime_views(
            event_id,
            refresh_boxes=True,
            refresh_focus=False,
        )

    def _ensure_hand_annotation_state(self, hand_data: Optional[dict]) -> dict:
        if not isinstance(hand_data, dict):
            hand_data = {}
        hand_data = ensure_hand_annotation_state(hand_data)
        return self._sync_hand_alias_fields(hand_data)

    def _hydrate_hand_annotation_state(
        self, hand_data: Optional[dict], default_source: str = "loaded_annotation"
    ) -> dict:
        if not isinstance(hand_data, dict):
            hand_data = {}
        hand_data = hydrate_existing_field_state(hand_data, default_source=default_source)
        return self._sync_hand_alias_fields(hand_data)

    def _set_pending_field_source(self, field_name: str, source: str) -> None:
        field_name = str(field_name or "").strip()
        if field_name:
            self._pending_field_sources[field_name] = str(source or "").strip() or "manual_ui"

    def _consume_pending_field_source(self, field_name: str, default: str) -> str:
        field_name = str(field_name or "").strip()
        if not field_name:
            return default
        source = self._pending_field_sources.pop(field_name, None)
        return str(source or default).strip() or default

    def _set_hand_field_state(
        self,
        hand_data: Optional[dict],
        field_name: str,
        *,
        source: str,
        value: Any = _NO_FIELD_VALUE,
        status: str = "confirmed",
        note: str = "",
    ) -> None:
        if not isinstance(hand_data, dict):
            return
        kwargs = {
            "source": source,
            "status": status,
            "note": note,
        }
        if value is not _NO_FIELD_VALUE:
            kwargs["value"] = value
        set_field_confirmation(hand_data, field_name, **kwargs)

    def _suggest_hand_field(
        self,
        hand_data: Optional[dict],
        field_name: str,
        value: Any,
        *,
        source: str,
        confidence: Optional[float] = None,
        reason: str = "",
        safe_to_apply: bool = True,
        review_recommended: bool = True,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(hand_data, dict):
            return
        if self._hand_field_locked_for_automation(hand_data, field_name):
            clear_field_suggestion(hand_data, field_name)
            return
        set_field_suggestion(
            hand_data,
            field_name,
            value,
            source=source,
            confidence=confidence,
            reason=reason,
            safe_to_apply=safe_to_apply,
            review_recommended=review_recommended,
            meta=meta,
        )

    def _clear_hand_runtime_suggestions(self, hand_data: Optional[dict]) -> None:
        if not isinstance(hand_data, dict):
            return
        for field_name in (
            "verb",
            "noun_object_id",
            "functional_contact_onset",
            "interaction_start",
            "interaction_end",
        ):
            try:
                clear_field_suggestion(hand_data, field_name)
            except Exception:
                pass
        hand_data.pop("_noun_source_decision", None)

    def _clear_all_runtime_suggestions(self) -> None:
        for event in list(getattr(self, "events", []) or []):
            if not isinstance(event, dict):
                continue
            hoi_data = dict(event.get("hoi_data", {}) or {})
            for actor in self.actors_config:
                hand_key = actor["id"]
                hand_data = hoi_data.get(hand_key)
                if isinstance(hand_data, dict):
                    self._clear_hand_runtime_suggestions(hand_data)
        for hand_data in list((getattr(self, "event_draft", {}) or {}).values()):
            if isinstance(hand_data, dict):
                self._clear_hand_runtime_suggestions(hand_data)

    def _clear_hand_field(self, hand_data: Optional[dict], field_name: str, source: str = "manual_clear") -> None:
        if not isinstance(hand_data, dict):
            return
        if (
            self._hand_field_locked_for_automation(hand_data, field_name)
            and self._field_source_family(source) != "human_manual"
        ):
            return
        clear_field_value(hand_data, field_name, source=source)

    def _box_source(self, box: Optional[dict], default: str = "unknown_box") -> str:
        if not isinstance(box, dict):
            return default
        return str(box.get("source") or default).strip() or default

    def _is_synthetic_hand_box(self, box: Optional[dict]) -> bool:
        if not isinstance(box, dict):
            return False
        if not self._normalize_hand_label(box.get("label")):
            return False
        source = self._box_source(box, default="unknown_box")
        return bool(box.get("synthetic")) or source.startswith("handtrack_once")

    def _friendly_hand_box_source(self, box: Optional[dict]) -> str:
        source = self._box_source(box, default="unknown_box")
        if source.startswith("handtrack_once"):
            return "tracked hand support"
        if source.startswith("materialized_handtrack"):
            return "editable hand box"
        if source.startswith("mediapipe_tasks"):
            return "detected hand support"
        if source.startswith("mediapipe"):
            return "detected hand support"
        if source.startswith("manual_"):
            return "manual hand box"
        if source.startswith("hands_xml") or source.startswith("xml"):
            return "imported hand box"
        return source.replace("_", " ")

    def _selected_hand_support_status(self, hand_key: str) -> Tuple[str, str]:
        hand_key = str(hand_key or "").strip()
        if not hand_key:
            return "Hand support: select an actor to inspect hand boxes.", ""
        frame = int(getattr(self.player, "current_frame", 0) or 0)
        box = dict((getattr(self, "current_hands", {}) or {}).get(hand_key) or {})
        handtrack_status = dict(getattr(self, "_handtrack_status", {}) or {})
        if box:
            source_text = self._friendly_hand_box_source(box)
            synthetic = self._is_synthetic_hand_box(box)
            detail = f"Current frame {frame}: a hand box is visible for {self._get_actor_full_label(hand_key)}."
            if synthetic:
                detail += " It is a tracked overlay. In Edit Boxes mode, click it to create an editable hand box."
            else:
                detail += f" Source: {source_text}."
            return f"Hand support: {detail}", detail
        if bool(handtrack_status.get("building")):
            detail = f"Current frame {frame}: hand support is still preparing for this clip."
            return f"Hand support: {detail}", detail
        track = self._handtrack_track(hand_key)
        frame_map = dict(track.get("frame_map") or {})
        if frame_map:
            detail = f"Current frame {frame}: no hand box is available for {self._get_actor_full_label(hand_key)} on this frame."
            return f"Hand support: {detail}", detail
        error_text = str(handtrack_status.get("error") or "").strip()
        if error_text:
            detail = f"Current frame {frame}: clip-level hand support is unavailable for this clip."
            return f"Hand support: {detail}", detail
        if self._detection_assist_enabled():
            detail = (
                f"Current frame {frame}: no hand box is visible. Use the assist controls for a quick check, "
                "or create a hand box manually if needed."
            )
        else:
            detail = (
                f"Current frame {frame}: no hand box is visible. Create a hand box manually if needed."
            )
        return f"Hand support: {detail}", detail

    def _compute_sparse_evidence_state(
        self,
        hand_data: Optional[dict],
        *,
        persist: bool = False,
    ) -> dict:
        if not isinstance(hand_data, dict):
            return {}
        self._sync_hand_alias_fields(hand_data)
        point_defs = (
            ("start", "interaction_start", "Start"),
            ("onset", "functional_contact_onset", "Onset"),
            ("end", "interaction_end", "End"),
        )
        role_defs = (
            ("noun", "noun_object_id", "Noun"),
            ("instrument", "instrument_object_id", "Instrument"),
        )
        states = {}
        for role_def in role_defs:
            role_slug, object_key, role_label = role_def
            obj_id = hand_data.get(object_key)
            obj_name = self._object_name_for_id(obj_id, default_for_none="", fallback="")
            for point_slug, time_key, time_label in point_defs:
                slot = f"{role_slug}_{point_slug}"
                frame = hand_data.get(time_key)
                try:
                    frame_int = int(frame) if frame is not None else None
                except Exception:
                    frame_int = None
                obj_id = (resolve_object(hand_data,self._assembly_data(),frame_int)
                          if role_slug=='noun' and frame_int is not None else hand_data.get(object_key))
                obj_name = self._object_name_for_id(obj_id, default_for_none="", fallback="")
                base = {
                    "slot": slot,
                    "role": role_slug,
                    "role_label": role_label,
                    "time_key": point_slug,
                    "time_label": time_label,
                    "frame": frame_int,
                    "object_id": obj_id,
                    "object_name": obj_name,
                }
                if obj_id is None or frame is None:
                    states[slot] = {
                        **base,
                        "status": "blocked",
                        "source": "missing_context",
                        "note": "Sparse evidence becomes active once both the object link and keyframe exist.",
                    }
                    continue
                if frame_int is None:
                    states[slot] = {
                        **base,
                        "status": "blocked",
                        "source": "invalid_frame",
                        "note": "The keyframe is not valid yet.",
                    }
                    continue
                match = self._best_matching_object_box_on_frame(frame_int, obj_id)
                if match:
                    states[slot] = {
                        **base,
                        "status": "confirmed",
                        "source": self._box_source(match, default="box_observed"),
                        "note": f"{role_label} evidence observed on the {time_label.lower()} keyframe.",
                    }
                else:
                    states[slot] = {
                        **base,
                        "status": "missing",
                        "source": "missing_evidence",
                        "note": f"No {role_slug} box was found on the {time_label.lower()} keyframe.",
                    }
        if persist:
            hand_data["_sparse_evidence_state"] = copy.deepcopy(states)
        return states

    def _sparse_evidence_summary(
        self,
        hand_data: Optional[dict],
        *,
        persist: bool = False,
    ) -> dict:
        state = self._compute_sparse_evidence_state(hand_data, persist=persist)
        expected = 0
        confirmed = 0
        missing = 0
        blocked = 0
        for row in list(state.values()):
            status = str((row or {}).get("status") or "").strip().lower()
            if status == "blocked":
                blocked += 1
                continue
            expected += 1
            if status == "confirmed":
                confirmed += 1
            elif status == "missing":
                missing += 1
        return {
            "expected": expected,
            "confirmed": confirmed,
            "missing": missing,
            "blocked": blocked,
            "state": state,
        }

    def _box_center_xy(self, box: Optional[dict]) -> Tuple[float, float]:
        if not isinstance(box, dict):
            return 0.0, 0.0
        try:
            x1 = float(box.get("x1", 0.0) or 0.0)
            y1 = float(box.get("y1", 0.0) or 0.0)
            x2 = float(box.get("x2", 0.0) or 0.0)
            y2 = float(box.get("y2", 0.0) or 0.0)
        except Exception:
            return 0.0, 0.0
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5

    def _box_diag(self, box: Optional[dict]) -> float:
        if not isinstance(box, dict):
            return 0.0
        try:
            x1 = float(box.get("x1", 0.0) or 0.0)
            y1 = float(box.get("y1", 0.0) or 0.0)
            x2 = float(box.get("x2", 0.0) or 0.0)
            y2 = float(box.get("y2", 0.0) or 0.0)
        except Exception:
            return 0.0
        return float(max(0.0, math.hypot(x2 - x1, y2 - y1)))

    def _box_iou(self, box_a: Optional[dict], box_b: Optional[dict]) -> float:
        if not isinstance(box_a, dict) or not isinstance(box_b, dict):
            return 0.0
        try:
            ax1 = float(box_a.get("x1", 0.0) or 0.0)
            ay1 = float(box_a.get("y1", 0.0) or 0.0)
            ax2 = float(box_a.get("x2", 0.0) or 0.0)
            ay2 = float(box_a.get("y2", 0.0) or 0.0)
            bx1 = float(box_b.get("x1", 0.0) or 0.0)
            by1 = float(box_b.get("y1", 0.0) or 0.0)
            bx2 = float(box_b.get("x2", 0.0) or 0.0)
            by2 = float(box_b.get("y2", 0.0) or 0.0)
        except Exception:
            return 0.0
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

    def _box_edge_gap(self, box_a: Optional[dict], box_b: Optional[dict]) -> float:
        if not isinstance(box_a, dict) or not isinstance(box_b, dict):
            return float("inf")
        try:
            ax1 = float(box_a.get("x1", 0.0) or 0.0)
            ay1 = float(box_a.get("y1", 0.0) or 0.0)
            ax2 = float(box_a.get("x2", 0.0) or 0.0)
            ay2 = float(box_a.get("y2", 0.0) or 0.0)
            bx1 = float(box_b.get("x1", 0.0) or 0.0)
            by1 = float(box_b.get("y1", 0.0) or 0.0)
            bx2 = float(box_b.get("x2", 0.0) or 0.0)
            by2 = float(box_b.get("y2", 0.0) or 0.0)
        except Exception:
            return float("inf")
        dx = max(0.0, ax1 - bx2, bx1 - ax2)
        dy = max(0.0, ay1 - by2, by1 - ay2)
        return float(math.hypot(dx, dy))

    def _box_confidence_score(self, box: Optional[dict]) -> float:
        if not isinstance(box, dict):
            return 0.0
        conf = box.get("confidence")
        if conf is not None:
            return self._bounded01(conf, 0.0)
        source = str(box.get("source") or "").strip().lower()
        if source.startswith("manual_box"):
            return 0.85
        if source.startswith("manual"):
            return 0.80
        if source.startswith("yolo_import"):
            return 0.65
        if source.startswith("import"):
            return 0.60
        if source.startswith("yolo"):
            return 0.55
        return 0.50

    def _hand_object_proximity_score(
        self,
        hand_box: Optional[dict],
        object_box: Optional[dict],
    ) -> float:
        if not isinstance(hand_box, dict) or not isinstance(object_box, dict):
            return 0.0
        frame_w = float(getattr(self.player, "_frame_w", 0) or 0.0)
        frame_h = float(getattr(self.player, "_frame_h", 0) or 0.0)
        frame_diag = float(max(1.0, math.hypot(frame_w, frame_h)))
        hand_diag = float(max(1.0, self._box_diag(hand_box)))
        object_diag = float(max(1.0, self._box_diag(object_box)))
        hx, hy = self._box_center_xy(hand_box)
        ox, oy = self._box_center_xy(object_box)
        center_dist = float(math.hypot(ox - hx, oy - hy))
        center_scale = max(24.0, 0.95 * hand_diag + 0.55 * object_diag)
        center_score = self._bounded01(
            1.0 - center_dist / float(max(1.0, center_scale)),
            0.0,
        )
        gap = self._box_edge_gap(hand_box, object_box)
        gap_score = self._bounded01(
            1.0 - gap / float(max(1.0, 0.18 * frame_diag)),
            0.0,
        )
        iou_score = self._bounded01(self._box_iou(hand_box, object_box) * 3.0, 0.0)
        return self._bounded01(
            max(gap_score, 0.60 * center_score + 0.40 * iou_score),
            0.0,
        )

    def _frame_hand_box(self, hand_key: str, frame: int) -> Dict[str, Any]:
        if f"{hand_key}:{int(frame)}" in getattr(self, "_suppressed_hand_boxes", []):
            return {}
        try:
            frame = int(frame)
        except Exception:
            return {}
        for box in list(self.bboxes.get(frame, []) or []):
            if self._normalize_hand_label(box.get("label")) == str(hand_key):
                return dict(box)
        if self._manual_mode_enabled():
            return {}
        track = self._handtrack_track(str(hand_key))
        row = dict((track.get("frame_map") or {}).get(frame) or {})
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
            "locked": False,
            "synthetic": True,
        }

    def _frame_boxes_with_cached_hands(self, frame: int) -> List[dict]:
        merged: List[dict] = [
            dict(box)
            for box in list(self.bboxes.get(frame, []) or [])
            if isinstance(box, dict)
        ]
        existing_hands = {
            self._normalize_hand_label(box.get("label"))
            for box in list(merged or [])
            if isinstance(box, dict)
        }
        for actor in list(self.actors_config or []):
            hand_key = str((actor or {}).get("id") or "").strip()
            if not hand_key or hand_key in existing_hands:
                continue
            hand_box = self._frame_hand_box(hand_key, frame)
            if hand_box:
                merged.append(dict(hand_box))
        return merged

    def _event_object_candidate_frames(
        self,
        hand_key: str,
        hand_data: Optional[dict],
    ) -> List[int]:
        if not isinstance(hand_data, dict):
            return []
        start = hand_data.get("interaction_start")
        onset = hand_data.get("functional_contact_onset")
        end = hand_data.get("interaction_end")
        try:
            start = int(start) if start is not None else None
            onset = int(onset) if onset is not None else None
            end = int(end) if end is not None else None
        except Exception:
            start = onset = end = None
        if start is None and onset is None and end is None:
            return []
        if start is not None and end is not None and end < start:
            start, end = end, start
        frames: List[int] = []

        def _add(frame_value: Optional[int]) -> None:
            if frame_value is None:
                return
            try:
                frame_int = int(frame_value)
            except Exception:
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
        track = self._handtrack_track(str(hand_key))
        motion_peak_frame = track.get("motion_peak_frame")
        try:
            motion_peak_frame = (
                int(motion_peak_frame)
                if motion_peak_frame is not None
                else None
            )
        except Exception:
            motion_peak_frame = None
        _add(motion_peak_frame)
        if motion_peak_frame is not None:
            _add(motion_peak_frame - 1)
            _add(motion_peak_frame + 1)
        return frames

    def _collect_event_object_candidates(
        self,
        hand_key: str,
        hand_data: Optional[dict],
    ) -> List[dict]:
        if not isinstance(hand_data, dict):
            return []
        onset_frame = hand_data.get("functional_contact_onset")
        try:
            onset_frame = int(onset_frame) if onset_frame is not None else None
        except Exception:
            onset_frame = None
        frames = self._event_object_candidate_frames(hand_key, hand_data)
        if not frames:
            return []

        track = self._handtrack_track(str(hand_key))
        frame_map = dict(track.get("frame_map") or {})
        motion_peak_frame = track.get("motion_peak_frame")
        try:
            motion_peak_frame = (
                int(motion_peak_frame)
                if motion_peak_frame is not None
                else None
            )
        except Exception:
            motion_peak_frame = None
        max_motion = max(
            [1e-6]
            + [
                float((frame_map.get(int(frame)) or {}).get("motion", 0.0) or 0.0)
                for frame in frames
            ]
        )

        candidates: Dict[int, Dict[str, Any]] = {}
        for frame in frames:
            hand_box = self._frame_hand_box(str(hand_key), int(frame))
            track_row = dict(frame_map.get(int(frame)) or {})
            motion_value = float(track_row.get("motion", 0.0) or 0.0)
            motion_weight = self._bounded01(motion_value / float(max_motion), 0.0)
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
            for box in list(self.bboxes.get(frame, []) or []):
                if self._normalize_hand_label(box.get("label")):
                    continue
                object_id = box.get("id")
                try:
                    object_id = int(object_id)
                except Exception:
                    continue
                row = candidates.setdefault(
                    object_id,
                    {
                        "object_id": object_id,
                        "object_name": self._object_name_for_id(
                            object_id,
                            default_for_none="",
                            fallback=f"Object {object_id}",
                        ),
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
                yolo_conf = self._box_confidence_score(box)
                proximity = self._hand_object_proximity_score(hand_box, box)
                motion_alignment = max(peak_weight, motion_weight) * max(0.15, proximity)
                support_score = 0.40 + 0.35 * proximity + 0.25 * yolo_conf
                onset_support_score = onset_weight * max(
                    0.20,
                    0.60 * proximity + 0.40 * yolo_conf,
                )
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
                row["hand_proximity_max"] = max(
                    float(row.get("hand_proximity_max", 0.0) or 0.0),
                    float(proximity),
                )
                row["yolo_confidence_max"] = max(
                    float(row.get("yolo_confidence_max", 0.0) or 0.0),
                    float(yolo_conf),
                )
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
        max_support_score = max(
            [1e-6]
            + [float(item.get("support_score", 0.0) or 0.0) for item in rows]
        )
        max_onset_support_score = max(
            [1e-6]
            + [float(item.get("onset_support_score", 0.0) or 0.0) for item in rows]
        )
        max_motion_support_score = max(
            [1e-6]
            + [float(item.get("motion_support_score", 0.0) or 0.0) for item in rows]
        )
        cleaned_rows: List[dict] = []
        for item in rows:
            frame_count = max(1, int(item.pop("_frame_count", 0) or 0))
            hand_proximity_mean = float(item.pop("_hand_proximity_sum", 0.0) or 0.0) / float(frame_count)
            yolo_confidence_mean = float(item.pop("_confidence_sum", 0.0) or 0.0) / float(frame_count)
            support_norm = self._bounded01(
                float(item.get("support_score", 0.0) or 0.0) / float(max_support_score),
                0.0,
            )
            onset_norm = self._bounded01(
                float(item.get("onset_support_score", 0.0) or 0.0)
                / float(max_onset_support_score),
                0.0,
            )
            motion_norm = self._bounded01(
                float(item.get("motion_support_score", 0.0) or 0.0)
                / float(max_motion_support_score),
                0.0,
            )
            item["hand_proximity_mean"] = float(hand_proximity_mean)
            item["yolo_confidence_mean"] = float(yolo_confidence_mean)
            item["hand_conditioned"] = bool(
                float(item.get("hand_proximity_max", 0.0) or 0.0) > 0.0
                or motion_norm > 0.0
            )
            item["candidate_score"] = self._bounded01(
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
            item["candidate_gap"] = float(
                max(
                    0.0,
                    float(item.get("candidate_score", 0.0) or 0.0) - next_score,
                )
            )
        return cleaned_rows

    def _refresh_sparse_evidence_snapshots(self) -> None:
        for event in list(self.events or []):
            hoi_data = event.get("hoi_data", {}) or {}
            if not isinstance(hoi_data, dict):
                continue
            for actor in self.actors_config:
                hand_key = actor["id"]
                hand_data = hoi_data.get(hand_key)
                if isinstance(hand_data, dict):
                    self._compute_sparse_evidence_state(hand_data, persist=True)
        for hand_key, hand_data in list((getattr(self, "event_draft", {}) or {}).items()):
            if isinstance(hand_data, dict):
                self._compute_sparse_evidence_state(hand_data, persist=True)

    def _sync_event_videomae_suggestions(
        self, event_id: int, event: Optional[dict], candidates: List[dict],
        *, target_hand=None,
    ) -> None:
        # Global clip scores are candidates, not labels for every actor.
        if not event or not candidates or not target_hand:
            return
        top = dict(candidates[0] or {})
        label = str(top.get("label") or "").strip()
        if not label:
            return
        rows = [event.get("hoi_data", {}).get(target_hand)]
        if self.selected_event_id == event_id:
            rows.append(self.event_draft.get(target_hand))
        for hand_data in rows:
            if not isinstance(hand_data, dict):
                continue
            self._suggest_hand_field(
                hand_data, "verb", label,
                source="videomae_top1", confidence=top.get("score"),
                reason="Clip-level candidate; verify the selected hand before applying.",
                safe_to_apply=False, review_recommended=True,
            )


    def _blank_hand_data(self) -> dict:
        hand_data = {
            "verb": "",
            "target_object_id": None,
            "noun_object_id": None,
            "instrument_object_id": None,
            "tool_object_id": None,
            "anomaly_label": _DEFAULT_ANOMALY_LABEL,
            "has_anomaly": False,
            "interaction_start": None,
            "functional_contact_onset": None,
            "interaction_end": None,
        }
        return self._ensure_hand_annotation_state(hand_data)

    def _on_hoi_timeline_select(self, event_id: int, hand_key: str):
        self._set_selected_event(event_id, hand_key)
        if bool(getattr(self, "_advanced_inspector_visible", True)):
            self._focus_inspector_tab("event")
        else:
            self._focus_inline_editor_field(self._inline_primary_field_name())
        self._refresh_inline_primary_focus(delay_ms=40)

    def _workspace_has_annotation_state(self) -> bool:
        return bool(
            self._assembly_data().get('states')
            or
            list(getattr(self, "events", []) or [])
            or list(getattr(self, "raw_boxes", []) or [])
            or str(getattr(self, "current_annotation_path", "") or "").strip()
        )

    def _confirm_workspace_reset(self, *, target_video: str = "", action_label: str = "reset the workspace") -> bool:
        if not self._workspace_has_annotation_state():
            return True
        unsaved = bool(self._hoi_has_unsaved_changes())
        lines = [
            "The current annotation workspace will be cleared.",
            "Loaded study assets and assist settings will be kept.",
        ]
        if target_video:
            lines.append(f"Next video: {os.path.basename(str(target_video or '').strip())}")
        if unsaved:
            lines.append("Unsaved event or box edits will be lost.")
        lines.append("")
        lines.append(f"Do you want to {action_label}?")
        reply = QMessageBox.question(
            self,
            "Reset Workspace",
            "\n".join(lines),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def _prepare_handtrack_for_video_switch(self, target_video_path: str) -> None:
        target_key = self._normalized_video_path(target_video_path)
        if (
            target_key
            and str(getattr(self, "_videomae_precomputed_cache_key", "") or "")
            and str(getattr(self, "_videomae_precomputed_cache_key", "") or "") != target_key
        ):
            self._clear_precomputed_videomae_cache()
        building_key = self._normalized_video_path(
            str((getattr(self, "_handtrack_status", {}) or {}).get("video_path") or "")
        )
        if (
            bool((getattr(self, "_handtrack_status", {}) or {}).get("building"))
            and building_key
            and target_key
            and building_key != target_key
        ):
            self._log(
                "hoi_handtrack_build_superseded",
                previous_video=str((self._handtrack_status or {}).get("video_path") or ""),
                next_video=str(target_video_path or ""),
            )
            self._handtrack_status.update(
                {
                    "ready": False,
                    "building": False,
                    "video_path": str(target_video_path or ""),
                    "error": "",
                }
            )
            self._handtrack_worker = None

    def _reset_annotation_workspace_state(
        self,
        *,
        reason: str,
        keep_current_video: bool = True,
        start_new_clip_session: bool = False,
    ) -> None:
        current_frame = int(getattr(self.player, "current_frame", 0) or 0)
        current_video = str(getattr(self, "video_path", "") or "").strip()
        previous_event_count = int(len(getattr(self, "events", []) or []))
        previous_box_count = int(len(getattr(self, "raw_boxes", []) or []))
        previous_annotation_path = str(getattr(self, "current_annotation_path", "") or "").strip()
        had_video = bool(getattr(self.player, "cap", None))

        self._log(
            "hoi_workspace_reset",
            reason=reason,
            keep_current_video=bool(keep_current_video and had_video),
            previous_event_count=previous_event_count,
            previous_box_count=previous_box_count,
            previous_annotation_path=previous_annotation_path,
            had_video=had_video,
        )

        if start_new_clip_session:
            self._flush_live_operation_logs(warn_user=False)

        self._annotation_provenance = {}
        self.frame_review = {}
        self.shared_assembly = empty_timeline()
        self._assembly_default_reference = False
        self.events.clear()
        self.raw_boxes = []
        self._suppressed_hand_boxes = []
        self.bboxes = {}
        self._validation_highlights = {}
        self._relation_highlights = {}
        self._dismissed_query_ids.clear()
        self._clear_relation_highlight()
        self._clear_undo_history()
        self.current_annotation_path = ""

        if getattr(self, "btn_validation", None) is not None:
            self.btn_validation.blockSignals(True)
            self.btn_validation.setChecked(False)
            self.btn_validation.blockSignals(False)
        self.validation_enabled = False
        self.validation_session_active = False
        self.validation_modified = False
        self.validation_change_count = 0
        self._set_validation_ui_state(False)
        if getattr(self, "validation_op_logger", None) is not None:
            self.validation_op_logger.clear()

        self._rebuild_bboxes_from_raw()
        if keep_current_video and had_video:
            self._refresh_boxes_for_frame(current_frame)
            self._update_overlay(current_frame)
            self._set_frame_controls(current_frame)
        self._refresh_events()
        self._after_events_loaded()
        self._update_status_label()
        self._update_incomplete_indicator()
        self._mark_query_calibration_dirty()
        self._mark_hoi_saved()

        if start_new_clip_session:
            if getattr(self, "op_logger", None) is not None:
                self.op_logger.clear()
            if getattr(self, "validation_op_logger", None) is not None:
                self.validation_op_logger.clear()
            self._begin_clip_logging_session(current_video if keep_current_video and had_video else "")
            if keep_current_video and had_video:
                self._log(
                    "hoi_clip_session_start",
                    path=current_video,
                    reset_workspace=True,
                    reason=reason,
                )
                self._log_annotation_ready_state("hoi_workspace_reset")

    def _new_trial_reset_workspace(self) -> None:
        if not self._confirm_workspace_reset(action_label="start a new trial"):
            return
        self._reset_annotation_workspace_state(
            reason="manual_new_trial",
            keep_current_video=True,
            start_new_clip_session=True,
        )
        QMessageBox.information(
            self,
            "New Trial",
            "Workspace reset.\n\nLoaded models, ontology, vocabularies, and adapter weights were kept.",
        )

    def _normalize_hoi_timing_triplet(
        self,
        start: Any,
        end: Any,
        onset: Any,
    ) -> Tuple[int, int, int]:
        new_start = int(start)
        new_end = int(end)
        if new_end < new_start:
            new_start, new_end = new_end, new_start
        new_onset = int(onset)
        new_onset = max(new_start, min(new_onset, new_end))
        return new_start, new_end, new_onset

    def _on_hoi_timeline_update(
        self, event_id: int, hand_key: str, start: int, end: int, onset: int
    ):
        ev = self._find_event_by_id(event_id)
        if not ev:
            return
        self._push_undo()
        h = ev.get("hoi_data", {}).setdefault(hand_key, self._blank_hand_data())
        self._ensure_hand_annotation_state(h)
        before_hand = copy.deepcopy(h)
        prev_start = self._field_snapshot(h, "interaction_start")
        prev_onset = self._field_snapshot(h, "functional_contact_onset")
        prev_end = self._field_snapshot(h, "interaction_end")
        new_start, new_end, new_onset = self._normalize_hoi_timing_triplet(
            start,
            end,
            onset,
        )
        start_changed = prev_start.get("value") != new_start
        onset_changed = prev_onset.get("value") != new_onset
        end_changed = prev_end.get("value") != new_end
        if not (start_changed or onset_changed or end_changed):
            if self.selected_event_id == event_id:
                self.event_draft[hand_key] = copy.deepcopy(h)
                self._update_status_label()
                self._refresh_inline_primary_focus(delay_ms=20)
            return
        onset_state_config = self._timeline_onset_state_for_mode()
        h["interaction_start"] = new_start
        h["interaction_end"] = new_end
        h["functional_contact_onset"] = new_onset
        if onset_changed:h["_onset_manually_adjusted"]=True
        self._set_hand_field_state(
            h, "interaction_start", source="manual_timeline", status="confirmed"
        )
        self._set_hand_field_state(
            h, "interaction_end", source="manual_timeline", status="confirmed"
        )
        preserved_onset_status = str(prev_onset.get("status") or "").strip().lower()
        preserved_onset_source = str(prev_onset.get("source") or "").strip()
        preserved_onset_note = str(
            (get_field_state(h, "functional_contact_onset") or {}).get("note") or ""
        ).strip()
        onset_source = onset_state_config.get("source")
        onset_status = onset_state_config.get("status")
        onset_note = onset_state_config.get("note")
        if not onset_changed and preserved_onset_status:
            onset_status = preserved_onset_status
            onset_source = preserved_onset_source or onset_source
            onset_note = preserved_onset_note or onset_note
        self._set_hand_field_state(
            h,
            "functional_contact_onset",
            source=onset_source,
            status=onset_status,
            note=onset_note,
        )
        self._sync_event_frames(ev)
        if self.selected_event_id == event_id:
            self.event_draft[hand_key] = copy.deepcopy(h)
            self._update_status_label()
            self._refresh_inline_primary_focus(delay_ms=40)
        self._bump_query_state_revision()
        self._refresh_events()
        self.hoi_timeline.refresh()
        self._invalidate_videomae_candidates(event_id)
        self._set_semantic_reinfer_hint(
            event_id,
            hand_key,
            h,
            reason="timeline_update",
            edited_fields=[
                "interaction_start",
                "functional_contact_onset",
                "interaction_end",
            ],
        )
        self._record_semantic_feedback(
            ev,
            hand_key,
            reason="timeline_update",
            before_hand=before_hand,
            edited_fields=[
                "interaction_start",
                "functional_contact_onset",
                "interaction_end",
            ],
            supervision_kind="edited",
        )
        if self.selected_event_id == event_id:
            self._update_action_top5_display(event_id)
            self._queue_action_label_refresh(event_id, delay_ms=450)
            self._refresh_semantic_suggestions_for_event(event_id, ev)
        self._log(
            "hoi_event_edit_frames",
            event_id=event_id,
            hand=hand_key,
            previous_start=prev_start.get("value"),
            start=start,
            previous_onset=prev_onset.get("value"),
            onset=onset,
            previous_end=prev_end.get("value"),
            end=end,
            previous_onset_status=prev_onset.get("status"),
            previous_onset_source=prev_onset.get("source"),
            new_onset_source=onset_source,
            new_onset_status=onset_status,
            onset_changed=bool(onset_changed),
            rework_after_automation=bool(
                ((prev_onset.get("status") or prev_onset.get("value") is not None) and self._is_automation_source(prev_onset.get("source")))
                or ((prev_start.get("status") or prev_start.get("value") is not None) and self._is_automation_source(prev_start.get("source")))
                or ((prev_end.get("status") or prev_end.get("value") is not None) and self._is_automation_source(prev_end.get("source")))
            ),
        )

    def _on_hoi_timeline_create(
        self, hand_key: str, start: int, end: int, onset: int
    ) -> Optional[int]:
        if not self.player.cap:
            QMessageBox.information(
                self, "Info", "Load a video before creating HOI events."
            )
            return None
        new_start, new_end, new_onset = self._normalize_hoi_timing_triplet(
            start,
            end,
            onset,
        )
        self._push_undo()
        new_event = {
            "event_id": self.event_id_counter,
            "frames": [new_start, new_end],
            "hoi_data": {actor["id"]: self._blank_hand_data() for actor in self.actors_config},
        }
        hand_data = new_event["hoi_data"][hand_key]
        onset_state_config = self._timeline_onset_state_for_mode()
        hand_data.update(self._current_hand_meta())
        hand_data["interaction_start"] = new_start
        hand_data["interaction_end"] = new_end
        hand_data["functional_contact_onset"] = new_onset
        self._set_hand_field_state(
            hand_data, "interaction_start", source="manual_timeline", status="confirmed"
        )
        self._set_hand_field_state(
            hand_data, "interaction_end", source="manual_timeline", status="confirmed"
        )
        self._set_hand_field_state(
            hand_data,
            "functional_contact_onset",
            source=onset_state_config.get("source"),
            status=onset_state_config.get("status"),
            note=onset_state_config.get("note"),
        )
        if str(hand_data.get("verb") or "").strip():
            self._set_hand_field_state(
                hand_data, "verb", source="manual_create", status="confirmed"
            )
        if self._hand_noun_object_id(hand_data) is not None:
            self._set_hand_field_state(
                hand_data,
                "noun_object_id",
                source="manual_create",
                status="confirmed",
            )
        self._sync_event_frames(new_event)
        self.events.append(new_event)
        self.event_id_counter += 1
        self._bump_query_state_revision()
        self._refresh_events()
        self.hoi_timeline.refresh()
        created_fields = [
            "interaction_start",
            "functional_contact_onset",
            "interaction_end",
        ]
        if str(hand_data.get("verb") or "").strip():
            created_fields.append("verb")
        if self._hand_noun_object_id(hand_data) is not None:
            created_fields.append("noun_object_id")
        self._set_semantic_reinfer_hint(
            new_event["event_id"],
            hand_key,
            hand_data,
            reason="timeline_create",
            edited_fields=created_fields,
        )
        self._record_semantic_feedback(
            new_event,
            hand_key,
            reason="timeline_create",
            edited_fields=created_fields,
            supervision_kind="edited",
        )
        self._set_selected_event(new_event["event_id"], hand_key)
        if not self._manual_mode_enabled():
            def _focus_created_onset() -> None:
                try:
                    if (
                        int(self.selected_event_id) == int(new_event["event_id"])
                        and str(self.selected_hand_label or "") == str(hand_key)
                    ):
                        self._focus_inline_editor_field("functional_contact_onset")
                except Exception:
                    pass

            QTimer.singleShot(60, _focus_created_onset)
        self._log(
            "hoi_event_create",
            event_id=new_event["event_id"],
            hand=hand_key,
            start=new_start,
            onset=new_onset,
            end=new_end,
            start_source="manual_timeline",
            onset_source=onset_state_config.get("source"),
            end_source="manual_timeline",
        )
        return new_event["event_id"]

    def _jump_to_selected_start(self):
        self._jump_to_selected_keyframe("interaction_start")

    def _jump_to_selected_onset(self):
        self._jump_to_selected_keyframe("functional_contact_onset")

    def _jump_to_selected_end(self):
        self._jump_to_selected_keyframe("interaction_end")

    def _jump_to_selected_keyframe(self, key: str):
        if self.selected_event_id is None or not self.selected_hand_label:
            return
        ev = self._find_event_by_id(self.selected_event_id)
        if not ev:
            return
        hand_data = ev.get("hoi_data", {}).get(self.selected_hand_label, {})
        frame = hand_data.get(key)
        if frame is None:
            return
        try:
            self.player.seek(int(frame))
            current_frame = int(getattr(self.player, "current_frame", frame) or frame)
            self._refresh_boxes_for_frame(current_frame)
            self._update_overlay(current_frame)
            self._set_frame_controls(current_frame)
        except Exception:
            pass

    def _on_hoi_timeline_hover(self, frame: int):
        if not self.player.cap:
            return
        if getattr(self.player, "is_playing", False):
            return
        if frame is None or frame < 0:
            target = int(getattr(self.player, "current_frame", 0))
            try:
                self.player.seek(target, preview_only=True)
            except Exception:
                return
            self._refresh_boxes_for_frame(target)
            self._update_overlay(target)
            return
        try:
            target = max(self.player.crop_start, min(frame, self.player.crop_end))
        except Exception:
            target = int(frame)
        self.player.seek(target, preview_only=True)
        self._refresh_boxes_for_frame(target)
        self._update_overlay(target)

    def _on_hoi_meta_changed(self):
        self._sync_action_panel_selection()
        if not self.selected_hand_label:
            return

        prev = copy.deepcopy(self.event_draft.get(self.selected_hand_label, {}) or {})
        current_ui_verb = str(self.combo_verb.currentText() or "").strip()
        current_ui_noun = self.combo_target.currentData()
        prev_noun = self._hand_noun_object_id(prev)
        prev_instrument = self._hand_instrument_object_id(prev)
        current_ui_instrument = self._combo_object_id(
            getattr(self, "combo_instrument", None),
            fallback=prev_instrument,
        )
        current_ui_anomaly = self._normalize_anomaly_label(
            getattr(getattr(self, "combo_anomaly", None), "currentText", lambda: "")()
        )
        prev_anomaly = self._normalize_anomaly_label(prev.get("anomaly_label"))
        should_push_undo = bool(
            current_ui_verb != str(prev.get("verb") or "").strip()
            or current_ui_noun != prev_noun
            or current_ui_instrument != prev_instrument
            or current_ui_anomaly != prev_anomaly
        )
        if should_push_undo:
            self._push_undo()
        self._save_ui_to_hand_draft(self.selected_hand_label)
        hand_data = self.event_draft.get(self.selected_hand_label, {}) or {}
        self._ensure_hand_annotation_state(hand_data)
        changed_fields = set()
        changed_metadata = []
        meta_fields = [("verb", "manual_ui")]
        meta_fields.append(("noun_object_id", "manual_link"))
        for field_name, default_source in meta_fields:
            before_snapshot = self._field_snapshot(prev, field_name)
            old_value = prev.get(field_name)
            new_value = hand_data.get(field_name)
            if old_value == new_value:
                continue
            changed_fields.add(field_name)
            if field_name == "verb":
                is_empty = not str(new_value or "").strip()
            else:
                is_empty = new_value is None
            if is_empty:
                clear_source = self._consume_pending_field_source(
                    field_name, "manual_clear"
                )
                if (
                    field_name == "noun_object_id"
                    and str(hand_data.get("verb") or "").strip()
                    and self._verb_allows_no_noun(hand_data.get("verb"))
                ):
                    confirm_source = (
                        clear_source
                        if clear_source not in ("manual_clear", "manual_link")
                        else "manual_no_noun"
                    )
                    self._set_hand_no_noun_confirmation(
                        hand_data,
                        source=confirm_source,
                        status="confirmed",
                    )
                else:
                    self._clear_hand_field(
                        hand_data,
                        field_name,
                        source=clear_source,
                    )
            else:
                self._set_hand_field_state(
                    hand_data,
                    field_name,
                    source=self._consume_pending_field_source(field_name, default_source),
                    status="confirmed",
                )
            after_snapshot = self._field_snapshot(hand_data, field_name)
            self._log(
                "hoi_manual_meta_change",
                event_id=self.selected_event_id,
                hand=self.selected_hand_label,
                field=field_name,
                old_value=before_snapshot.get("value"),
                new_value=after_snapshot.get("value"),
                old_status=before_snapshot.get("status"),
                new_status=after_snapshot.get("status"),
                old_source=before_snapshot.get("source"),
                new_source=after_snapshot.get("source"),
                old_source_family=before_snapshot.get("source_family"),
                new_source_family=after_snapshot.get("source_family"),
                rework_after_automation=bool(
                    (before_snapshot.get("status") or before_snapshot.get("value") is not None)
                    and self._is_automation_source(before_snapshot.get("source"))
                ),
            )
        for field_name, old_value, new_value in (
            (
                "instrument_object_id",
                prev_instrument,
                self._hand_instrument_object_id(hand_data),
            ),
            (
                "anomaly_label",
                prev_anomaly,
                self._normalize_anomaly_label(hand_data.get("anomaly_label")),
            ),
        ):
            if old_value == new_value:
                continue
            changed_metadata.append(field_name)
            self._log(
                "hoi_manual_meta_change",
                event_id=self.selected_event_id,
                hand=self.selected_hand_label,
                field=field_name,
                old_value=old_value,
                new_value=new_value,
                new_source="manual_ui",
            )
        if "verb" in changed_fields and self._hand_noun_object_id(hand_data) is None:
            noun_state = get_field_state(hand_data, "noun_object_id")
            if (
                self._noun_required_for_verb(hand_data.get("verb"))
                and str(noun_state.get("status") or "").strip().lower() == "confirmed"
            ):
                before_snapshot = self._field_snapshot(hand_data, "noun_object_id")
                self._clear_hand_field(
                    hand_data,
                    "noun_object_id",
                    source="verb_requires_noun",
                )
                changed_fields.add("noun_object_id")
                after_snapshot = self._field_snapshot(hand_data, "noun_object_id")
                self._log(
                    "hoi_manual_meta_change",
                    event_id=self.selected_event_id,
                    hand=self.selected_hand_label,
                    field="noun_object_id",
                    old_value=before_snapshot.get("value"),
                    new_value=after_snapshot.get("value"),
                    old_status=before_snapshot.get("status"),
                    new_status=after_snapshot.get("status"),
                    old_source=before_snapshot.get("source"),
                    new_source=after_snapshot.get("source"),
                    old_source_family=before_snapshot.get("source_family"),
                    new_source_family=after_snapshot.get("source_family"),
                    rework_after_automation=bool(
                        (before_snapshot.get("status") or before_snapshot.get("value") is not None)
                        and self._is_automation_source(before_snapshot.get("source"))
                    ),
                )
        if prev != hand_data:
            self._mark_validation_change()
            self._bump_query_state_revision()
        if self.selected_event_id is not None:
            self._apply_draft_to_selected_event()
            if "verb" in changed_fields:
                event = self._find_event_by_id(self.selected_event_id)
                if event is not None:
                    self._invalidate_videomae_candidates(self.selected_event_id)
                    event.pop("videomae_meta", None)
                    event.pop("videomae_local_top5", None)
                    event.pop("videomae_local_meta", None)
            self._refresh_events()
            self._update_hoi_titles()
            if getattr(self, "hoi_timeline", None):
                self.hoi_timeline.refresh()
            if "verb" in changed_fields:
                self._update_action_top5_display(self.selected_event_id)
                self._queue_action_label_refresh(
                    self.selected_event_id, delay_ms=120, force=True
                )
            if changed_fields:
                event = self._find_event_by_id(self.selected_event_id)
                if event is not None and self.selected_hand_label:
                    self._set_semantic_reinfer_hint(
                        self.selected_event_id,
                        self.selected_hand_label,
                        hand_data,
                        reason="manual_meta_edit",
                        edited_fields=sorted(changed_fields),
                    )
                    self._record_semantic_feedback(
                        event,
                        self.selected_hand_label,
                        reason="manual_meta_edit",
                        before_hand=prev,
                        edited_fields=sorted(changed_fields),
                        supervision_kind="edited",
                    )
                    self._refresh_semantic_suggestions_for_event(self.selected_event_id, event)
        self._update_status_label()

    def _after_events_loaded(self):
        self.selected_event_id = None
        self.selected_hand_label = None
        self._selected_edit_box = None
        self._semantic_reinfer_hints.clear()
        self._reset_event_draft()
        self._reset_query_session_metrics()
        self._clear_videomae_action_runtime(clear_event_scores=True)
        self._update_action_top5_display(None)
        self._bump_query_state_revision()
        self._update_next_best_query_panel()
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.set_selected(None, None)
            self._update_hoi_titles()
            self.hoi_timeline.refresh()

        self._incomplete_last_key = None
        self._incomplete_last_order = None
        self._update_incomplete_indicator()
        if getattr(self, "chk_auto_jump_incomplete", None) and self.chk_auto_jump_incomplete.isChecked():
            QTimer.singleShot(0, self._jump_first_incomplete)

    def _on_verb_added(self, lb):
        """lb may be a LabelDef (from LabelPanel) or a name string."""
        missing = self._missing_requirements(for_relation=False)
        if missing:
            QMessageBox.information(
                self,
                "Info",
                f"Please load required data before adding verbs: {', '.join(missing)}",
            )
            return
        max_id = max([v.id for v in self.verbs], default=-1)
        if isinstance(lb, LabelDef):
            name = lb.name
        else:
            name = str(lb)
        color_name = (
            getattr(lb, "color_name", None) if isinstance(lb, LabelDef) else None
        )
        if not color_name:
            color_name = "Auto"
        new_lb = LabelDef(name=name, id=max_id + 1, color_name=color_name)
        self.verbs.append(new_lb)
        self._renumber_verbs()
        self._update_verb_combo()
        self._log("hoi_verb_add", name=new_lb.name, id=new_lb.id)

    def _on_verb_removed(self, idx: int):
        removed = None
        if 0 <= idx < len(self.verbs):
            removed = self.verbs.pop(idx)
        self._renumber_verbs()
        self.current_verb_idx = min(self.current_verb_idx, len(self.verbs) - 1)
        self._update_verb_combo()
        if removed:
            self._log("hoi_verb_remove", name=removed.name, id=removed.id)

    def _on_verb_renamed(self, idx: int, new: str):
        old = None
        if 0 <= idx < len(self.verbs):
            old = self.verbs[idx].name
            self.verbs[idx].name = new
        self._renumber_verbs()
        self._update_verb_combo()
        if old is not None:
            self._replace_verb_in_hoi_data(self.event_draft, old, new)
            for ev in self.events:
                self._replace_verb_in_hoi_data(ev.get("hoi_data", {}), old, new)
            self._refresh_events()
            self._update_hoi_titles()
            if getattr(self, "hoi_timeline", None):
                self.hoi_timeline.refresh()
            self._log("hoi_verb_rename", old=old, new=new)

    def _on_verb_selected(self, idx: int):
        if idx < 0 or idx >= len(self.verbs) or self._action_panel_sync:
            return
        self._sync_action_panel_selection(self.verbs[idx].name)

    def _apply_verb_choice(self, verb_name: str, source: str = "manual"):
        verb_name = str(verb_name or "").strip()
        if not verb_name:
            return
        current = self.combo_verb.currentText().strip()
        self._set_pending_field_source("verb", source)
        if current != verb_name:
            self.combo_verb.setCurrentText(verb_name)
        else:
            self._sync_action_panel_selection(verb_name)
            self._on_hoi_meta_changed()
        self._log(
            "hoi_apply_verb_choice",
            source=source,
            verb=verb_name,
            event_id=self.selected_event_id,
            hand=self.selected_hand_label,
        )

    def _sync_action_panel_selection(self, verb_name: str = None):
        if self._action_panel_sync:
            return
        verb_name = str(verb_name if verb_name is not None else self.combo_verb.currentText()).strip()
        self._action_panel_sync = True
        try:
            if hasattr(self, "label_panel") and self.label_panel:
                if verb_name and self.label_panel.index_of_label(verb_name) >= 0:
                    self.label_panel.select_label_by_name(verb_name)
                else:
                    self.label_panel.clear_selection()
        finally:
            self._action_panel_sync = False

    def _on_verb_library_item_clicked(self, item: QListWidgetItem):
        if item is None:
            return
        verb_name = str(item.data(Qt.UserRole) or item.text() or "").strip()
        if not verb_name or verb_name == "__OTHER_LABEL__":
            return
        self._apply_verb_choice(verb_name, source="verb_library")

    def _renumber_verbs(self):
        """Ensure ids are monotonically increasing and new ids append after max."""
        cur_max = 0
        for v in self.verbs:
            cur_max = max(cur_max, getattr(v, "id", 0))
        # ensure unique increasing ids
        seen = set()
        for v in self.verbs:
            if v.id in seen:
                cur_max += 1
                v.id = cur_max
            seen.add(v.id)
        # advance next id in UI
        try:
            self.label_panel.id_spin.setValue(cur_max + 1)
        except Exception:
            pass

    def _normalize_hand_label(self, label: str) -> Optional[str]:
        if label is None:
            return None
        norm = str(label).strip().lower().replace(" ", "_")
        for actor in self.actors_config:
            aid = actor["id"]
            if norm == aid.lower():
                return aid
            if norm in (
                actor.get("short", "").lower(),
                actor.get("label", "").lower(),
            ):
                return aid
        # Fallbacks for very common terms
        if norm in ("left", "l_hand", "left_hand"):
            for actor in self.actors_config:
                if "left" in actor["id"].lower():
                    return actor["id"]
            return self.actors_config[0]["id"]
        if norm in ("right", "r_hand", "right_hand"):
            for actor in self.actors_config:
                if "right" in actor["id"].lower():
                    return actor["id"]
            return self.actors_config[1]["id"] if len(self.actors_config) > 1 else self.actors_config[0]["id"]
        return None

    def _is_hand_label(self, label: str) -> bool:
        if not label:
            return False
        l_str = str(label).strip()
        return any(actor["id"] == l_str for actor in self.actors_config)

    def _color_for_index(self, idx: int) -> str:
        """Pick a non-gray preset color cycling through the palette."""
        palette = [k for k in PRESET_COLORS.keys() if k.lower() != "gray"] or list(
            PRESET_COLORS.keys()
        )
        if not palette:
            return "gray"
        return palette[idx % len(palette)]

    def _verb_color(self, name: str) -> str:
        """Return a QColor-friendly string for a verb name."""
        for v in self.verbs:
            if v.name == name:
                return color_from_key(getattr(v, "color_name", "gray")).name()
        return "#f08c00"

    def _missing_requirements(self, for_relation: bool = False) -> List[str]:
        """
        New check chain: only checks Video and Hands.
        Does not check classes or object bboxes.
        """
        missing = []
        if not self.player.cap:
            missing.append("video")

        # Only check for hands data when creating a relation
        if for_relation:
            # Check if any hand data exists
            has_hand = any(
                b
                for b in self.raw_boxes
                if self._is_hand_label(b.get("label"))
            )
            if not has_hand:
                missing.append("hand bboxes (XML)")

        return missing

    # ---------- undo/redo ----------
    def _snapshot_state(self) -> dict:
        """Capture a deep copy of event + bbox state for undo/redo."""
        return {
            "frame_review": copy.deepcopy(getattr(self,"frame_review",{})),
            "shared_assembly": copy.deepcopy(self._assembly_data()),
            "assembly_default_reference": getattr(self,"_assembly_default_reference",False),
            "events": copy.deepcopy(self.events),
            "event_id_counter": self.event_id_counter,
            "event_draft": copy.deepcopy(self.event_draft),
            "selected_hand_label": self.selected_hand_label,
            "selected_event_id": self.selected_event_id,
            "raw_boxes": copy.deepcopy(self.raw_boxes),
            "suppressed_hand_boxes": list(getattr(self, "_suppressed_hand_boxes", [])),
            "box_id_counter": self.box_id_counter,
            "global_object_map": copy.deepcopy(self.global_object_map),
            "id_to_category": copy.deepcopy(self.id_to_category),
            "object_id_counter": self.object_id_counter,
            "class_map": copy.deepcopy(self.class_map),
            "target_selected": self.combo_target.currentData(),
            "instrument_selected": getattr(
                getattr(self, "combo_instrument", None),
                "currentData",
                lambda: None,
            )(),
            "anomaly_selected": getattr(
                getattr(self, "combo_anomaly", None),
                "currentText",
                lambda: _DEFAULT_ANOMALY_LABEL,
            )(),
            "current_frame": int(getattr(self.player, "current_frame", 0)),
        }

    def _restore_state(self, state: dict):
        """Restore a snapshot created by _snapshot_state."""
        self.frame_review = copy.deepcopy(state.get("frame_review",{}))
        self.shared_assembly = validate_timeline(state.get("shared_assembly", {"schema":"shared-assembly-1","states":[]}))
        self._assembly_default_reference = state.get("assembly_default_reference",False)
        self.events = copy.deepcopy(state.get("events", []))
        self.event_id_counter = state.get("event_id_counter", 0)
        self.event_draft = copy.deepcopy(state.get("event_draft", {}))
        self.selected_hand_label = state.get("selected_hand_label", None)
        self.selected_event_id = state.get("selected_event_id", None)
        self.raw_boxes = copy.deepcopy(state.get("raw_boxes", []))
        self._suppressed_hand_boxes = list(state.get("suppressed_hand_boxes", []))
        self.box_id_counter = state.get("box_id_counter", 1)
        self.global_object_map = copy.deepcopy(state.get("global_object_map", {}))
        self.id_to_category = copy.deepcopy(state.get("id_to_category", {}))
        self.object_id_counter = state.get("object_id_counter", 0)
        self.class_map = copy.deepcopy(state.get("class_map", {}))

        self._rebuild_bboxes_from_raw()

        tgt_sel = state.get("target_selected")
        inst_sel = state.get("instrument_selected")
        self._rebuild_object_combos(tgt_sel, inst_sel)
        if getattr(self, "combo_anomaly", None) is not None:
            self._populate_anomaly_combo(
                self.combo_anomaly,
                state.get("anomaly_selected", _DEFAULT_ANOMALY_LABEL),
            )

        frame = state.get("current_frame", self.player.current_frame)
        if self.player.cap:
            self.player.seek(frame)
        self._refresh_boxes_for_frame(frame)
        self._set_frame_controls(frame)

        for aid, chk in self.actor_controls.items():
            chk.blockSignals(True)
            chk.setChecked(self.selected_hand_label == aid)
            chk.blockSignals(False)

        if self.selected_hand_label:
            self._load_hand_draft_to_ui(self.selected_hand_label)
        else:
            self._update_status_label()
        self.list_objects.setEnabled(self.selected_hand_label is not None)
        for ev in self.events:
            self._sync_event_frames(ev)
        self._refresh_events()
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.set_selected(
                self.selected_event_id, self.selected_hand_label
            )
            self.hoi_timeline.set_current_frame(frame)
            self.hoi_timeline.refresh()

        self._update_action_top5_display(self.selected_event_id)
        self._bump_query_state_revision()
        self._update_next_best_query_panel()
        self._update_incomplete_indicator()

    def _push_undo(self):
        if self._undo_block:
            return
        self._mark_validation_change()
        self._hoi_undo_stack.append(self._snapshot_state())
        if len(self._hoi_undo_stack) > self._undo_limit:
            self._hoi_undo_stack.pop(0)
        self._hoi_redo_stack.clear()

    def _mark_validation_change(self) -> None:
        if not bool(getattr(self, "validation_enabled", False)):
            return
        self.validation_modified = True
        self.validation_change_count += 1

    def _hoi_undo(self):
        if not self._hoi_undo_stack:
            return
        self._undo_block = True
        try:
            current = self._snapshot_state()
            state = self._hoi_undo_stack.pop()
            self._hoi_redo_stack.append(current)
            self._restore_state(state)
            self._log(
                "hoi_undo",
                selected_event_id=self.selected_event_id,
                selected_hand=self.selected_hand_label or "",
                undo_depth=len(self._hoi_undo_stack),
                redo_depth=len(self._hoi_redo_stack),
            )
        finally:
            self._undo_block = False

    def _hoi_redo(self):
        if not self._hoi_redo_stack:
            return
        self._undo_block = True
        try:
            current = self._snapshot_state()
            state = self._hoi_redo_stack.pop()
            self._hoi_undo_stack.append(current)
            self._restore_state(state)
            self._log(
                "hoi_redo",
                selected_event_id=self.selected_event_id,
                selected_hand=self.selected_hand_label or "",
                undo_depth=len(self._hoi_undo_stack),
                redo_depth=len(self._hoi_redo_stack),
            )
        finally:
            self._undo_block = False

    def _clear_undo_history(self):
        self._hoi_undo_stack.clear()
        self._hoi_redo_stack.clear()

    def set_logging_policy(
        self, oplog_enabled: bool, validation_summary_enabled: bool
    ) -> None:
        if getattr(self, "op_logger", None) is not None:
            self.op_logger.enabled = bool(oplog_enabled)
        if getattr(self, "validation_op_logger", None) is not None:
            self.validation_op_logger.enabled = bool(oplog_enabled)
        self.validation_summary_enabled = bool(validation_summary_enabled)

    def _flush_ops_log_safely(
        self,
        logger,
        log_path: str,
        context: str,
        warn_user: bool = True,
    ) -> None:
        if logger is None or not log_path:
            return
        should_write = bool(getattr(logger, "enabled", False))
        if should_write and hasattr(logger, "has_rows") and not logger.has_rows():
            return
        try:
            logger.flush(log_path)
            if should_write:
                print(f"[LOG] {log_path}")
        except Exception as ex:
            print(f"[LOG][ERROR] {context} ops log write failed: {log_path} ({ex})")
            if warn_user:
                QMessageBox.warning(
                    self,
                    "Operation log",
                    f"Operation log write failed, but annotation save already succeeded.\n\nTarget: {log_path}\n\n{ex}",
                )

    def _begin_clip_logging_session(self, video_path: str = "") -> None:
        self._clip_session_index = int(getattr(self, "_clip_session_index", 0) or 0) + 1
        session_base = str(
            getattr(getattr(self, "op_logger", None), "session_id", "")
            or getattr(getattr(self, "validation_op_logger", None), "session_id", "")
            or "session"
        ).strip()
        video_base = os.path.splitext(
            os.path.basename(str(video_path or self.video_path or "").strip())
        )[0]
        video_base = re.sub(r"[^A-Za-z0-9_.-]+", "_", video_base or "clip").strip("_") or "clip"
        self._clip_session_id = f"{session_base}_{video_base}_{self._clip_session_index:03d}"
        self._annotation_ready_signature = ""
        self._annotation_readiness_phase = ""

    def _reset_clip_logging_for_new_video(self, next_video_path: str) -> None:
        next_norm = self._normalized_video_path(next_video_path)
        current_norm = self._normalized_video_path(self.video_path)
        if not current_norm and not getattr(self, "_clip_session_id", ""):
            if getattr(self, "op_logger", None) is not None:
                self.op_logger.clear()
            if getattr(self, "validation_op_logger", None) is not None:
                self.validation_op_logger.clear()
        if current_norm and next_norm and current_norm != next_norm:
            try:
                self._flush_live_operation_logs(warn_user=False)
            except Exception:
                pass
            if getattr(self, "op_logger", None) is not None:
                self.op_logger.clear()
            if getattr(self, "validation_op_logger", None) is not None:
                self.validation_op_logger.clear()
        if not getattr(self, "_clip_session_id", "") or current_norm != next_norm:
            self._begin_clip_logging_session(next_video_path)

    @staticmethod
    def _field_source_family(source: Any) -> str:
        text = str(source or "").strip().lower()
        if not text:
            return "unknown"
        if text.startswith(
            (
                "manual",
                "inline_",
                "verb_library",
                "object_pick",
                "noun_only_mode",
                "query_confirm",
                "validation",
                "human",
            )
        ):
            return "human_manual"
        if text.startswith("semantic_adapter") or text.startswith("videomae"):
            return "semantic_model"
        if text.startswith("handtrack_once") or text.startswith("hand_conditioned"):
            return "hand_conditioned"
        if "completion" in text:
            return "completion"
        if "yolo" in text or "detector" in text:
            return "detector"
        if text in {"query_controller", "query_apply"}:
            return "controller"
        return "other"

    @classmethod
    def _is_automation_source(cls, source: Any) -> bool:
        return cls._field_source_family(source) in {
            "semantic_model",
            "hand_conditioned",
            "completion",
            "detector",
            "controller",
        }

    def _field_snapshot(self, hand_data: Optional[dict], field_name: str) -> Dict[str, Any]:
        if not isinstance(hand_data, dict):
            return {
                "value": None,
                "status": "",
                "source": "",
                "source_family": "unknown",
            }
        state = get_field_state(hand_data, field_name)
        source = str(state.get("source") or "").strip()
        return {
            "value": hand_data.get(field_name),
            "status": str(state.get("status") or "").strip(),
            "source": source,
            "source_family": self._field_source_family(source),
        }

    def _semantic_feedback_fields(self) -> Tuple[str, ...]:
        return (
            "interaction_start",
            "functional_contact_onset",
            "interaction_end",
            "verb",
            "noun_object_id",
        )

    def _event_hand_runtime_summary(
        self,
        event: Optional[dict],
        *,
        focus_hand: str = "",
    ) -> Dict[str, Dict[str, Any]]:
        if not isinstance(event, dict):
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        hoi_data = dict(event.get("hoi_data") or {})
        for actor in list(self.actors_config or []):
            hand_key = str((actor or {}).get("id") or "").strip()
            if not hand_key:
                continue
            hand_data = dict(hoi_data.get(hand_key) or {})
            out[hand_key] = {
                "selected": bool(hand_key == str(focus_hand or "").strip()),
                "interaction_start": hand_data.get("interaction_start"),
                "functional_contact_onset": hand_data.get("functional_contact_onset"),
                "interaction_end": hand_data.get("interaction_end"),
                "verb": str(hand_data.get("verb") or "").strip(),
                "noun_object_id": self._hand_noun_object_id(hand_data),
                "instrument_object_id": self._hand_instrument_object_id(hand_data),
                "tool_object_id": self._hand_instrument_object_id(hand_data),
                "anomaly_label": self._normalize_anomaly_label(
                    hand_data.get("anomaly_label")
                ),
                "has_anomaly": self._is_anomalous_label(
                    hand_data.get("anomaly_label")
                ),
                "has_segment": bool(
                    hand_data.get("interaction_start") is not None
                    or hand_data.get("functional_contact_onset") is not None
                    or hand_data.get("interaction_end") is not None
                ),
            }
        return out

    def _cross_hand_context_for_event(
        self,
        event: Optional[dict],
        hand_key: str,
        hand_data: Optional[dict],
    ) -> Dict[str, Any]:
        if not isinstance(event, dict) or not isinstance(hand_data, dict):
            return {}
        try:
            start = int(hand_data.get("interaction_start")) if hand_data.get("interaction_start") is not None else None
            end = int(hand_data.get("interaction_end")) if hand_data.get("interaction_end") is not None else None
        except Exception:
            start = end = None
        if start is None or end is None:
            return {}
        if end < start:
            start, end = end, start
        segment_len = max(1, int(end) - int(start))
        rows: List[Dict[str, Any]] = []
        for actor in list(self.actors_config or []):
            other_key = str((actor or {}).get("id") or "").strip()
            if not other_key or other_key == str(hand_key or "").strip():
                continue
            other = dict(((event.get("hoi_data") or {}).get(other_key) or {}))
            onset = other.get("functional_contact_onset")
            try:
                onset = int(onset) if onset is not None else None
            except Exception:
                onset = None
            if onset is None or onset < start or onset > end:
                continue
            onset_state = get_field_state(other, "functional_contact_onset")
            onset_status = str(onset_state.get("status") or "").strip().lower()
            if onset_status not in {"confirmed", "suggested"}:
                continue
            onset_ratio = self._bounded01(
                (int(onset) - int(start)) / float(segment_len),
                0.5,
            )
            exclude_weight = 0.16 if onset_status == "confirmed" else 0.08
            rows.append(
                {
                    "hand": other_key,
                    "onset_frame": int(onset),
                    "onset_ratio": float(onset_ratio),
                    "status": onset_status,
                    "exclude_weight": float(exclude_weight),
                }
            )
        rows.sort(key=lambda row: float(row.get("exclude_weight", 0.0)), reverse=True)
        return {
            "other_hands": rows,
            "exclusive_count": int(len(rows)),
            "primary_exclusion": dict(rows[0]) if rows else {},
        }

    def _semantic_reinfer_hint_key(self, event_id: Any, hand_key: Any) -> str:
        try:
            event_text = str(int(event_id))
        except Exception:
            event_text = str(event_id or "")
        return f"{event_text}::{str(hand_key or '').strip()}"

    def _set_semantic_reinfer_hint(
        self,
        event_id: Any,
        hand_key: str,
        hand_data: Optional[dict],
        *,
        reason: str,
        edited_fields: Optional[Sequence[str]] = None,
    ) -> None:
        hand_key = str(hand_key or "").strip()
        if event_id is None or not hand_key or not isinstance(hand_data, dict):
            return
        edited = {
            self._canonical_query_field_name(name)
            for name in list(edited_fields or [])
            if self._canonical_query_field_name(name)
        }
        onset_anchor_ratio = None
        onset_anchor_half_width = None
        onset_value = hand_data.get("functional_contact_onset")
        start = hand_data.get("interaction_start")
        end = hand_data.get("interaction_end")
        onset_state = get_field_state(hand_data, "functional_contact_onset")
        onset_status = str(onset_state.get("status") or "").strip().lower()
        if (
            "functional_contact_onset" not in edited
            and onset_value is not None
            and onset_status != "confirmed"
        ):
            try:
                start = int(start) if start is not None else None
                end = int(end) if end is not None else None
                onset_value = int(onset_value)
            except Exception:
                start = end = onset_value = None
            if start is not None and end is not None and onset_value is not None:
                if end < start:
                    start, end = end, start
                segment_len = max(1, int(end) - int(start))
                onset_anchor_ratio = self._bounded01(
                    (int(onset_value) - int(start)) / float(segment_len),
                    0.5,
                )
                onset_anchor_half_width = (
                    0.08
                    if onset_status == "suggested"
                    else 0.12
                )
        self._semantic_reinfer_hints[self._semantic_reinfer_hint_key(event_id, hand_key)] = {
            "reason": str(reason or "").strip(),
            "edited_fields": sorted(str(name) for name in edited if name),
            "onset_anchor_ratio": onset_anchor_ratio,
            "onset_anchor_half_width": onset_anchor_half_width,
        }

    def _consume_semantic_reinfer_hint(
        self,
        event_id: Any,
        hand_key: str,
    ) -> Dict[str, Any]:
        key = self._semantic_reinfer_hint_key(event_id, hand_key)
        return dict(self._semantic_reinfer_hints.pop(key, {}) or {})

    def _current_annotation_ready_state(self) -> Dict[str, Any]:
        video_loaded = bool(self.video_path and getattr(self.player, "cap", None))
        nouns_loaded = bool(len(getattr(self, "global_object_map", {}) or {}) > 0)
        verbs_loaded = bool(len(getattr(self, "verbs", []) or []) > 0)
        ontology_loaded = bool(
            len(getattr(getattr(self, "hoi_ontology", None), "relations", {}) or {}) > 0
        )
        yolo_loaded = bool(getattr(self, "yolo_model", None) is not None and self.yolo_weights_path)
        videomae_weights_loaded = bool(str(getattr(self, "videomae_weights_path", "") or "").strip())
        videomae_verb_list_loaded = bool(str(getattr(self, "videomae_verb_list_path", "") or "").strip())
        videomae_ready = bool(self._videomae_ready())
        videomae_cache_loaded = bool(self._videomae_precomputed_ready())
        videomae_source_ready = bool(videomae_ready or videomae_cache_loaded)
        handtrack_status = dict(getattr(self, "_handtrack_status", {}) or {})
        handtrack_building = bool(handtrack_status.get("building"))
        handtrack_ready = bool(handtrack_status.get("ready"))
        handtrack_backend = str(handtrack_status.get("backend") or "").strip()
        semantic_status = self._semantic_adapter_runtime_status()
        assist_mode = self._experiment_mode_key()
        base_ready = bool(video_loaded and nouns_loaded and verbs_loaded and ontology_loaded)
        mode_ready = bool(base_ready and videomae_source_ready) if assist_mode == "full_assist" else base_ready
        required_missing = [
            name
            for name, present in (
                ("video", video_loaded),
                ("nouns", nouns_loaded),
                ("verbs", verbs_loaded),
                ("ontology", ontology_loaded),
                ("videomae", (assist_mode != "full_assist") or videomae_source_ready),
            )
            if not present
        ]
        if mode_ready:
            clip_phase = "ready"
            clip_label = "Ready" if assist_mode != "full_assist" else "Ready to annotate"
            detail_parts = (
                ["Video and required lists are loaded."]
                if assist_mode != "full_assist"
                else ["Current mode requirements satisfied."]
            )
            if assist_mode == "full_assist":
                if videomae_cache_loaded and not videomae_ready:
                    detail_parts.append("Using precomputed clip features.")
                elif videomae_ready:
                    detail_parts.append("Assist features are ready for this clip.")
                if handtrack_building:
                    detail_parts.append("Hand support is still preparing in the background.")
                elif handtrack_ready:
                    detail_parts.append("Hand support is available for this clip.")
                if semantic_status.get("training_running"):
                    detail_parts.append("Personalized assistance is updating in the background.")
                elif semantic_status.get("model_available"):
                    detail_parts.append("Personalized assistance is available.")
        elif video_loaded:
            clip_phase = "preparing"
            clip_label = "Preparing" if assist_mode != "full_assist" else "Preparing clip"
            if required_missing:
                missing_labels = {
                    "video": "video",
                    "nouns": "noun list",
                    "verbs": "verb list",
                    "ontology": "ontology",
                    "videomae": "assist features",
                }
                pretty_missing = [missing_labels.get(name, name) for name in required_missing]
                detail_parts = ["Waiting for: " + ", ".join(pretty_missing) + "."]
            else:
                detail_parts = (
                    ["Loading the current clip."]
                    if assist_mode != "full_assist"
                    else ["Waiting for the current clip session to finish preparing."]
                )
            if assist_mode == "full_assist" and not videomae_source_ready:
                detail_parts.append("Full Assist is waiting for the assist model or precomputed clip features.")
            if assist_mode == "full_assist":
                if handtrack_building:
                    detail_parts.append("Hand support is still preparing in the background.")
                if semantic_status.get("training_running"):
                    detail_parts.append("Personalized assistance is updating in the background.")
        else:
            clip_phase = "idle"
            clip_label = "Load clip"
            detail_parts = (
                ["Load a video, noun list, verb list, and ontology."]
                if assist_mode != "full_assist"
                else ["Load a video and the required study assets before timing starts."]
            )
        clip_detail = " ".join(part for part in detail_parts if str(part or "").strip())
        return {
            "assist_mode": assist_mode,
            "video_loaded": video_loaded,
            "nouns_loaded": nouns_loaded,
            "verbs_loaded": verbs_loaded,
            "ontology_loaded": ontology_loaded,
            "yolo_loaded": yolo_loaded,
            "videomae_weights_loaded": videomae_weights_loaded,
            "videomae_verb_list_loaded": videomae_verb_list_loaded,
            "videomae_ready": videomae_ready,
            "videomae_cache_loaded": videomae_cache_loaded,
            "videomae_source_ready": videomae_source_ready,
            "handtrack_building": handtrack_building,
            "handtrack_ready": handtrack_ready,
            "handtrack_backend": handtrack_backend,
            "semantic_training_running": bool(semantic_status.get("training_running")),
            "semantic_model_available": bool(semantic_status.get("model_available")),
            "semantic_model_path": str(semantic_status.get("active_model_path") or ""),
            "semantic_model_sample_count": semantic_status.get("model_sample_count"),
            "semantic_feedback_rows": int(semantic_status.get("feedback_rows", 0) or 0),
            "semantic_feedback_pending": int(semantic_status.get("feedback_pending", 0) or 0),
            "semantic_participant_code": str(semantic_status.get("participant_code") or ""),
            "semantic_workspace": str(semantic_status.get("workspace") or ""),
            "base_ready": base_ready,
            "mode_ready": mode_ready,
            "required_missing": required_missing,
            "clip_phase": clip_phase,
            "clip_label": clip_label,
            "clip_detail": clip_detail,
        }

    def _update_clip_readiness_ui(self, state: Optional[Dict[str, Any]] = None) -> None:
        chip = getattr(self, "lbl_clip_readiness_chip", None)
        if chip is None:
            return
        state = dict(state or self._current_annotation_ready_state() or {})
        phase = str(state.get("clip_phase") or "").strip().lower()
        tone = {
            "ready": "ok",
            "preparing": "warn",
            "idle": "neutral",
        }.get(phase, "neutral")
        label = str(state.get("clip_label") or "Load clip").strip() or "Load clip"
        detail = str(state.get("clip_detail") or "").strip()
        self._set_status_chip(chip, label, tone)
        chip.setToolTip(detail or label)

    def _log_annotation_ready_state(self, reason: str) -> None:
        state = self._current_annotation_ready_state()
        self._update_clip_readiness_ui(state)
        signature = json.dumps(state, sort_keys=True, ensure_ascii=True)
        if signature == str(getattr(self, "_annotation_ready_signature", "") or ""):
            return
        previous_phase = str(getattr(self, "_annotation_readiness_phase", "") or "")
        self._annotation_ready_signature = signature
        self._annotation_readiness_phase = str(state.get("clip_phase") or "")
        self._log("hoi_annotation_ready_state", reason=reason, **state)
        current_phase = str(state.get("clip_phase") or "")
        if current_phase != previous_phase:
            if current_phase == "preparing":
                self._log("hoi_clip_preparing", reason=reason, **state)
            elif current_phase == "ready":
                self._log("hoi_clip_ready_to_annotate", reason=reason, **state)
        if bool(state.get("mode_ready")):
            self._log("hoi_annotation_ready", reason=reason, **state)

    def _maybe_warn_full_assist_semantic_unavailable(self, reason: str) -> None:
        state = self._current_annotation_ready_state()
        if str(state.get("assist_mode") or "") != "full_assist":
            self._full_assist_warning_signature = ""
            return
        if not bool(state.get("video_loaded")):
            return
        if bool(state.get("videomae_source_ready")):
            self._full_assist_warning_signature = ""
            return
        warning_key = json.dumps(
            {
                "video": self._normalized_video_path(self.video_path),
                "mode": state.get("assist_mode"),
                "missing": list(state.get("required_missing") or []),
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        if warning_key == str(getattr(self, "_full_assist_warning_signature", "") or ""):
            return
        self._full_assist_warning_signature = warning_key
        QMessageBox.warning(
            self,
            "Full Assist Not Ready",
            "Full Assist is selected, but the assist model or precomputed clip features are not available yet.\n\n"
            "Basic timing support can still run, but action suggestions are not fully active.\n\n"
            "Load one of the following before relying on Full Assist:\n"
            "- Assist model weights\n"
            "- Precomputed clip features (.videomae_cache.npz)",
        )

    @staticmethod
    def _classify_log_event(event: str) -> Dict[str, Any]:
        event_name = str(event or "").strip()
        mapping = {
            "hoi_load_video": (True, "setup", "load_video"),
            "hoi_load_targets": (True, "setup", "load_nouns"),
            "hoi_load_verbs": (True, "setup", "load_verbs"),
            "hoi_load_classes": (True, "setup", "load_class_map"),
            "hoi_load_ontology": (True, "setup", "load_ontology"),
            "hoi_load_yolo_model": (True, "setup", "load_yolo_model"),
            "hoi_load_videomae": (True, "setup", "load_videomae"),
            "hoi_load_videomae_cache": (True, "setup", "load_videomae_cache"),
            "hoi_clip_session_start": (True, "setup", "clip_session_start"),
            "hoi_experiment_mode_changed": (True, "setup", "change_mode"),
            "hoi_participant_code_changed": (True, "setup", "change_participant"),
            "hoi_workspace_reset": (True, "setup", "reset_workspace"),
            "hoi_annotation_ready_state": (False, "setup", "annotation_ready_state"),
            "hoi_annotation_ready": (False, "setup", "annotation_ready"),
            "hoi_clip_preparing": (False, "setup", "clip_preparing"),
            "hoi_clip_ready_to_annotate": (False, "setup", "clip_ready_to_annotate"),
            "hoi_event_create": (True, "annotation", "event_create"),
            "hoi_event_edit_frames": (True, "annotation", "event_edit_frames"),
            "hoi_manual_meta_change": (True, "annotation", "manual_field_change"),
            "hoi_apply_verb_choice": (True, "annotation", "choose_verb"),
            "hoi_apply_noun_choice": (True, "annotation", "choose_noun"),
            "hoi_inline_confirm_field": (True, "annotation", "confirm_field"),
            "hoi_inline_object_candidate_pick": (True, "annotation", "choose_object_candidate"),
            "hoi_select_object": (True, "annotation", "select_object"),
            "hoi_query_focus": (True, "query", "focus"),
            "hoi_query_apply": (True, "query", "apply"),
            "hoi_query_reject": (True, "query", "reject"),
            "hoi_undo": (True, "history", "undo"),
            "hoi_redo": (True, "history", "redo"),
            "hoi_select_hand": (True, "navigation", "select_hand"),
            "hoi_jump_to_frame": (True, "navigation", "jump_to_frame"),
            "hoi_seek_slider": (True, "navigation", "seek_slider"),
            "hoi_seek_relative": (True, "navigation", "seek_relative"),
            "hoi_crop_set": (True, "navigation", "set_crop"),
            "hoi_play": (True, "navigation", "play"),
            "hoi_pause": (True, "navigation", "pause"),
            "hoi_stop": (True, "navigation", "stop"),
            "hoi_incomplete_jump": (True, "navigation", "jump_incomplete"),
            "hoi_edit_boxes_toggle": (True, "bbox", "toggle_edit_boxes"),
            "hoi_frame_swap_hands": (True, "bbox", "swap_hands"),
            "hoi_validation_on": (True, "review", "validation_on"),
            "hoi_validation_off": (True, "review", "validation_off"),
            "hoi_validation_cancel": (True, "review", "validation_cancel"),
            "hoi_save_annotations": (True, "save", "save_annotations"),
        }
        if event_name in mapping:
            is_user_operation, family, kind = mapping[event_name]
            return {
                "is_user_operation": is_user_operation,
                "operation_family": family,
                "operation_kind": kind,
            }
        if event_name.startswith("hoi_box_"):
            return {
                "is_user_operation": True,
                "operation_family": "bbox",
                "operation_kind": event_name.replace("hoi_", "", 1),
            }
        if event_name.startswith("hoi_detect_"):
            return {
                "is_user_operation": True,
                "operation_family": "bbox",
                "operation_kind": event_name.replace("hoi_", "", 1),
            }
        return {
            "is_user_operation": False,
            "operation_family": "",
            "operation_kind": "",
        }

    def _log(self, event: str, **fields):
        # Keep the log focused on user-driven actions.
        auto_events = {"frame_advanced"}
        if event in auto_events:
            return
        path_field_tokens = ("path", "file", "video", "model")
        for key, value in list(fields.items()):
            key_text = str(key or "").lower()
            if not any(token in key_text for token in path_field_tokens):
                continue
            if isinstance(value, str):
                fields[key] = self._portable_path_token(value)
        classification = self._classify_log_event(event)
        fields.setdefault("clip_session_id", getattr(self, "_clip_session_id", "") or "")
        fields.setdefault("clip_session_index", int(getattr(self, "_clip_session_index", 0) or 0))
        fields.setdefault("is_user_operation", classification.get("is_user_operation", False))
        fields.setdefault("operation_family", classification.get("operation_family", ""))
        fields.setdefault("operation_kind", classification.get("operation_kind", ""))
        fields.setdefault(
            "annotation_ready_for_mode",
            bool(self._current_annotation_ready_state().get("mode_ready")),
        )
        fields.setdefault("assist_mode", self._experiment_mode_key())
        fields.setdefault("annotator_id", self._current_annotator_id())
        fields.setdefault("selected_event_id", self.selected_event_id)
        fields.setdefault("selected_hand", self.selected_hand_label or "")
        try:
            base = os.path.splitext(os.path.basename(self.video_path or ""))[0]
        except Exception:
            base = ""
        if base:
            fields.setdefault("video_id", base)
        if getattr(self, "op_logger", None):
            try:
                frame = getattr(self.player, "current_frame", None)
            except Exception:
                frame = None
            if frame is not None:
                fields.setdefault("frame", frame)
            if self.validation_enabled and self.validator_name:
                fields.setdefault("validator", self.validator_name)
            self.op_logger.log(event, **fields)
        if self.validation_enabled and getattr(self, "validation_op_logger", None):
            self.validation_op_logger.log(event, **fields)

    def _bump_bbox_revision(self) -> int:
        self._bbox_revision = int(getattr(self, "_bbox_revision", 0) or 0) + 1
        self._refresh_completion_indicators(update_status=True)
        return self._bbox_revision

    def _refresh_completion_indicators(
        self,
        *,
        refresh_evidence: bool = True,
        update_status: bool = False,
    ) -> None:
        """Refresh UI elements whose state depends on event completeness."""
        if refresh_evidence and hasattr(self, "events") and hasattr(self, "actors_config"):
            self._refresh_sparse_evidence_snapshots()
        if hasattr(self, "lbl_incomplete"):
            self._update_incomplete_indicator()
        if getattr(self, "hoi_timeline", None):
            self._update_hoi_titles()
            self.hoi_timeline.refresh()
        if update_status:
            if bool(getattr(self, "_inline_editor_sync", False)):
                self._completion_status_refresh_pending = True
            else:
                self._update_status_label()

    @staticmethod
    def _is_box_locked(box: Optional[dict]) -> bool:
        return bool((box or {}).get("locked"))

    def _raw_box_matches(self, raw_box: dict, box: dict, frame: Optional[int] = None) -> bool:
        if not raw_box or not box:
            return False
        target_frame = frame
        if target_frame is None:
            target_frame = box.get("frame")
        if target_frame is None and box.get("orig_frame") is not None:
            try:
                target_frame = int(box.get("orig_frame")) + int(self.start_offset)
            except Exception:
                target_frame = None
        if target_frame is not None:
            try:
                if int(raw_box.get("orig_frame", 0)) != int(target_frame) - int(self.start_offset):
                    return False
            except Exception:
                return False
        if box.get("id") is not None and raw_box.get("id") != box.get("id"):
            return False
        if box.get("label") is not None and str(raw_box.get("label")) != str(box.get("label")):
            return False
        return self._coords_close(raw_box, box)

    def _matching_raw_boxes(self, box: dict, frame: Optional[int] = None) -> List[dict]:
        return [
            rb
            for rb in list(self.raw_boxes or [])
            if self._raw_box_matches(rb, box, frame=frame)
        ]

    def _box_locked_for_action(self, box: Optional[dict], frame: Optional[int] = None) -> bool:
        if self._is_box_locked(box):
            return True
        return any(self._is_box_locked(rb) for rb in self._matching_raw_boxes(box or {}, frame=frame))

    def _set_box_lock_state(
        self,
        box: dict,
        *,
        locked: bool,
        frame: Optional[int] = None,
        interactive: bool = True,
    ) -> bool:
        matches = self._matching_raw_boxes(box or {}, frame=frame)
        if not matches:
            return False
        target_locked = bool(locked)
        if all(self._is_box_locked(rb) == target_locked for rb in matches):
            return False
        self._push_undo()
        for rb in matches:
            rb["locked"] = target_locked
        self._rebuild_bboxes_from_raw()
        self._bump_bbox_revision()
        self._bump_query_state_revision()
        refresh_frame = frame if frame is not None else getattr(self.player, "current_frame", 0)
        self._refresh_boxes_for_frame(int(refresh_frame))
        self._log(
            "hoi_box_lock" if target_locked else "hoi_box_unlock",
            box_id=box.get("id"),
            label=box.get("label"),
            frame=refresh_frame,
        )
        if interactive:
            QMessageBox.information(
                self,
                "Box Lock",
                "Box locked." if target_locked else "Box unlocked.",
            )
        return True

    def _live_ops_log_base(self) -> str:
        annotation_path = str(getattr(self, "current_annotation_path", "") or "").strip()
        if annotation_path:
            return os.path.splitext(annotation_path)[0]
        return ""

    def _sync_live_edits_for_export(self) -> None:
        if self.selected_event_id is not None:
            self._save_ui_to_hand_draft(self.selected_hand_label)
            self._apply_draft_to_selected_event()

    def _recovery_snapshot_dir(self) -> str:
        path = os.path.join(self._repo_root_dir(), "runtime_artifacts", "recovery")
        os.makedirs(path, exist_ok=True)
        return path

    def _recovery_snapshot_path(self, reason: str = "recovery") -> str:
        base = self._safe_filename_token(
            self._default_annotation_basename(),
            fallback="hoi_recovery",
        )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reason_token = self._safe_filename_token(reason, fallback="recovery")
        return os.path.join(
            self._recovery_snapshot_dir(),
            f"{base}_{stamp}_{reason_token}.recovery.json",
        )

    def _save_recovery_snapshot(self, *, reason: str = "recovery", error: str = "") -> str:
        if not self._workspace_has_annotation_state():
            return ""
        try:
            self._sync_live_edits_for_export()
            payload = self._build_payload_v2()
            payload["_recovery"] = {
                "reason": str(reason or "recovery"),
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "current_annotation_path": str(getattr(self, "current_annotation_path", "") or "").strip(),
                "error": str(error or "").strip(),
            }
            fp = self._recovery_snapshot_path(reason=reason)
            atomic_json(fp, payload)
            try:
                graph = build_hoi_event_graph(
                    self.events,
                    video_path=self.video_path,
                    annotation_path=fp,
                    actors_config=self.actors_config,
                    assembly_timeline=self._assembly_data(),
                )
                save_event_graph_sidecar(fp, graph)
            except Exception:
                pass
            try:
                self._flush_live_operation_logs(warn_user=False)
            except Exception:
                pass
            self._log(
                "hoi_recovery_snapshot_saved",
                path=fp,
                reason=str(reason or "recovery"),
                error=str(error or "").strip(),
            )
            return fp
        except Exception as ex:
            try:
                self._log(
                    "hoi_recovery_snapshot_failed",
                    reason=str(reason or "recovery"),
                    error=str(ex),
                )
            except Exception:
                pass
            return ""

    def _flush_live_operation_logs(self, warn_user: bool = False) -> None:
        log_base = self._live_ops_log_base()
        if not log_base:
            return
        logger = getattr(self, "op_logger", None)
        if logger is not None and getattr(logger, "enabled", False) and (
            getattr(logger, "is_dirty", lambda: True)() or not os.path.isfile(log_base + ".ops.log.csv")
        ):
            self._flush_ops_log_safely(
                logger,
                log_base + ".ops.log.csv",
                context="HOI live flush",
                warn_user=warn_user,
            )
        validation_logger = getattr(self, "validation_op_logger", None)
        if (
            validation_logger is not None
            and getattr(validation_logger, "enabled", False)
            and self.validation_session_active
            and (
                getattr(validation_logger, "is_dirty", lambda: True)()
                or not os.path.isfile(log_base + ".validation.ops.log.csv")
            )
        ):
            self._flush_ops_log_safely(
                validation_logger,
                log_base + ".validation.ops.log.csv",
                context="HOI validation live flush",
                warn_user=warn_user,
            )

    def _refresh_inference_action_states(self) -> None:
        detection_enabled = self._detection_assist_enabled() and self._yolo_infer_worker is None
        semantic_enabled = (
            self._semantic_assist_enabled()
            and self._videomae_infer_worker is None
            and self._videomae_batch_progress is None
        )
        for obj in (
            getattr(self, "act_detect_current_frame", None),
            getattr(self, "act_detect_selected_action", None),
            getattr(self, "act_detect_all_actions", None),
            getattr(self, "btn_detect", None),
            getattr(self, "btn_detect_action", None),
            getattr(self, "btn_inline_detect_objects", None),
        ):
            if obj is not None:
                obj.setEnabled(detection_enabled)
        for obj in (
            getattr(self, "act_review_selected_action_label", None),
            getattr(self, "act_auto_apply_action_labels", None),
            getattr(self, "act_batch_apply_action_labels", None),
            getattr(self, "btn_suggest_action_label", None),
        ):
            if obj is not None:
                obj.setEnabled(semantic_enabled)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Left:
            self._seek_relative(-1)
            event.accept()
            return
        if key == Qt.Key_Right:
            self._seek_relative(+1)
            event.accept()
            return
        if key == Qt.Key_Up:
            self._seek_seconds(-1)
            event.accept()
            return
        if key == Qt.Key_Down:
            self._seek_seconds(+1)
            event.accept()
            return
        return super().keyPressEvent(event)

    # ---------- IO ----------
    @staticmethod
    def _normalized_video_path(path: str) -> str:
        raw = str(path or "").strip()
        if not raw:
            return ""
        try:
            return os.path.normcase(os.path.abspath(raw))
        except Exception:
            return os.path.normcase(raw)

    def _apply_video_session(
        self,
        path: str,
        start: int = None,
        end: int = None,
        frame: int = None,
        log_event: str = "hoi_load_video",
    ) -> bool:
        target_path = str(path or "").strip()
        if not target_path:
            return False
        target_norm = self._normalized_video_path(target_path)
        current_norm = self._normalized_video_path(self.video_path)
        reuse_loaded = bool(self.player.cap) and bool(current_norm) and current_norm == target_norm
        if not reuse_loaded:
            if not self.player.load(target_path):
                return False

        total_frames = int(getattr(self.player, "frame_count", 0) or 0)
        if total_frames <= 0:
            return False
        self._reset_clip_logging_for_new_video(target_path)
        start_frame = 0 if start is None else int(start)
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = total_frames - 1 if end is None else int(end)
        end_frame = max(start_frame, min(end_frame, total_frames - 1))
        current_frame = start_frame if frame is None else int(frame)
        current_frame = max(start_frame, min(current_frame, end_frame))

        self.player.set_crop(start_frame, end_frame)
        if int(getattr(self.player, "current_frame", start_frame)) != current_frame:
            self.player.seek(current_frame)

        self.video_path = target_path
        self.start_offset = start_frame
        self.end_frame = end_frame
        self._reset_query_session_metrics()

        self.spin_start_offset.blockSignals(True)
        self.spin_start_offset.setMaximum(max(0, total_frames - 1))
        self.spin_start_offset.setValue(start_frame)
        self.spin_start_offset.blockSignals(False)

        self.spin_end_frame.blockSignals(True)
        self.spin_end_frame.setMaximum(max(0, total_frames - 1))
        self.spin_end_frame.setValue(end_frame)
        self.spin_end_frame.blockSignals(False)

        self.spin_jump.setMinimum(start_frame)
        self.spin_jump.setMaximum(max(start_frame, end_frame))
        self.spin_jump.setValue(current_frame)

        self.slider.blockSignals(True)
        self.slider.setMinimum(start_frame)
        self.slider.setMaximum(end_frame)
        self.slider.setValue(current_frame)
        self.slider.blockSignals(False)

        self._rebuild_bboxes_from_raw()
        self._refresh_boxes_for_frame(current_frame)
        self._set_frame_controls(current_frame)

        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.set_frame_count(self.player.frame_count)
            self.hoi_timeline.set_current_frame(current_frame)
            self.hoi_timeline.refresh()

        self._update_play_pause_button()
        self._prepare_handtrack_for_video_switch(target_path)
        if not self._manual_mode_enabled():
            try:
                self._start_handtrack_precompute(target_path)
            except Exception as ex:
                self._log("hoi_handtrack_build_schedule_failed", error=str(ex))
        if log_event:
            self._log(
                log_event,
                path=target_path,
                start=start_frame,
                end=end_frame,
                frame=current_frame,
                frames=self.player.frame_count,
            )
            self._log("hoi_clip_session_start", path=target_path)
        self._log_annotation_ready_state(log_event or "apply_video_session")
        return True

    def _load_video(self):
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Load Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        if not fp:
            return
        target_norm = self._normalized_video_path(fp)
        current_norm = self._normalized_video_path(self.video_path)
        if current_norm == target_norm and self._workspace_has_annotation_state():
            self._log("hoi_video_already_open_edits_retained", path=fp)
            return
        # Probe the destination lock before clearing the current workspace.
        from pathlib import Path
        from PyQt5.QtCore import QLockFile
        if getattr(self, '_autosave_reserved', None) != Path(fp).resolve():
            directory = Path(fp).resolve().parent / '.impact-recovery'
            directory.mkdir(exist_ok=True)
            probe = QLockFile(str(directory / (Path(fp).name + '.json.lock')))
            probe.setStaleLockTime(0)
            if not probe.tryLock(0):
                QMessageBox.warning(self, 'Video already open', 'Close the other editor using this video before loading it.')
                return
            probe.unlock()
        if current_norm and target_norm and current_norm != target_norm and self._workspace_has_annotation_state():
            if not self._confirm_save_before_loading_video(fp):
                return
            self._reserve_autosave(fp)
            self._reset_annotation_workspace_state(
                reason="load_new_video_auto",
                keep_current_video=True,
                start_new_clip_session=False,
            )
        self._update_clip_readiness_ui(
            {
                "clip_phase": "preparing",
                "clip_label": "Preparing clip",
                "clip_detail": "Loading video and resolving the current clip session.",
            }
        )
        QApplication.processEvents()
        if not self._apply_video_session(fp, log_event="hoi_load_video"):
            QMessageBox.warning(self, "Error", "Failed to load video.")
            self._log_annotation_ready_state("hoi_load_video_failed")
            return
        self.current_annotation_path = ""
        self._finish_autosave()
        self._autosave_path = None
        self._autosave_baseline = None
        self._task_resume_path = ""
        self._task_trial_id = ""
        try:
            task_loaded = self._auto_load_task_bundle(fp)
        except Exception as exc:
            QMessageBox.warning(self, "Task loading failed", str(exc))
            return
        if not task_loaded:
            self._auto_load_local_assets_for_video(fp)
            self._open_autosave()
        self._log_annotation_ready_state("hoi_load_video_assets")
        self._maybe_warn_full_assist_semantic_unavailable("hoi_load_video_assets")
        self._mark_query_calibration_dirty()
        self._update_onboarding_banner()

    def _load_bboxes(self):
        """
        Phase 1 Smart Loader (Corrected):
        Saves filtered YOLO boxes into self.raw_boxes to prevent overwriting.
        """
        if not self._guard_experiment_mode("detection"):
            return
        # --- 1. Dependency Check ---
        if not self.player.cap:
            QMessageBox.warning(
                self, "Missing prerequisite", "Please load a video first (resolution needed)."
            )
            return
        if not self.global_object_map:
            QMessageBox.warning(
                self,
                "Missing prerequisite",
                "Please import the noun/object list first.",
            )
            return
        if not self.class_map:
            QMessageBox.warning(
                self,
                "Missing prerequisite",
                "Please load a class map first (data.yaml).",
            )
            return

        dir_path = QFileDialog.getExistingDirectory(
            self, "Select YOLO labels directory"
        )
        if not dir_path:
            return

        # --- 2. Build Smart Map ---
        category_to_default_id = self._build_category_to_default_id()

        if not category_to_default_id:
            QMessageBox.warning(
                self,
                "Error",
                "Cannot build the default object map. Check the noun/object list.",
            )
            return

        # --- 3. Parse Loop ---
        loaded_count = 0
        w, h = self.player._frame_w, self.player._frame_h
        new_raw_boxes = []

        import os

        files = sorted([f for f in os.listdir(dir_path) if f.endswith(".txt")])

        for fname in files:
            # Get raw frame index (without offset)
            frame_idx = self._frame_from_filename(fname)

            path = os.path.join(dir_path, fname)
            with open(path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                    except Exception:
                        continue

                    cls_name = self._norm_category(self.class_map.get(cls_id, ""))
                    if not cls_name:
                        continue

                    # Match ID
                    matched_uid = category_to_default_id.get(cls_name)

                    if matched_uid is not None:
                        # Calc absolute coordinates
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        x2 = (cx + bw / 2) * w
                        y2 = (cy + bh / 2) * h

                        # Construct raw_box dict
                        # Must include orig_frame for _rebuild to calculate offset correctly
                        new_raw_boxes.append(
                            {
                                "id": matched_uid,  # Forced merge to global ID
                                "orig_frame": frame_idx,
                                "label": self.class_map.get(cls_id, ""),
                                "class_id": cls_id,
                                "source": "yolo_import",
                                "locked": False,
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                            }
                        )
                        loaded_count += 1

        # --- 4. Update Memory ---
        # Append new data to raw_boxes
        # Strategy: Keep hands (Left_hand/Right_hand), clear other old objects, add new objects
        has_existing_objects = any(
            b
            for b in self.raw_boxes
            if not self._is_hand_label(b.get("label"))
        )
        if new_raw_boxes or has_existing_objects:
            self._push_undo()
        self.raw_boxes = [
            b
            for b in self.raw_boxes
            if self._is_hand_label(b.get("label")) or self._is_box_locked(b)
        ]
        self.raw_boxes.extend(new_raw_boxes)

        # Rebuild bboxes index
        self._rebuild_bboxes_from_raw()
        self._bump_bbox_revision()
        self._bump_query_state_revision()

        self._refresh_boxes_for_frame(self.player.current_frame)
        self._log("hoi_load_bboxes", path=dir_path, count=loaded_count)
        QMessageBox.information(
            self,
            "Done",
            f"Loaded {loaded_count} detection boxes.\nOld objects cleared, hands preserved.",
        )

    def _load_yaml(self):
        """Load data.yaml to get class_id -> class_name mapping"""
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Load Class Map (data.yaml)",
            "",
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if not fp:
            return
        self._load_yaml_from_path(fp, notify_user=True)

    def _load_yaml_from_path(
        self,
        fp: str,
        *,
        notify_user: bool = True,
        auto_discovered: bool = False,
    ) -> bool:
        if not self._guard_asset_mutation(
            "class map",
            path=fp,
            notify_user=notify_user,
            auto_discovered=auto_discovered,
            full_assist_only=True,
        ):
            return False
        try:
            self.class_map = self._parse_yaml_names(fp)
            self._log(
                "hoi_load_classes",
                path=fp,
                count=len(self.class_map),
                auto_discovered=bool(auto_discovered),
            )
            added = self._sync_library_with_class_map(notify_user=False)
            if notify_user:
                QMessageBox.information(
                    self,
                    "Loaded",
                    (
                        f"Loaded {len(self.class_map)} classes from yaml.\n"
                        f"Matched {int(added)} labels to the current noun/object library.\n"
                        "Classes outside the loaded noun list were ignored."
                    ),
                )
            return True
        except Exception as ex:
            if notify_user:
                QMessageBox.warning(self, "Error", f"Failed to parse data.yaml:\n{ex}")
            else:
                self._log("hoi_auto_load_failed", asset="class_map", path=fp, error=str(ex))
            return False

    def _sync_library_with_class_map(self, *, notify_user: bool = True) -> int:
        """Restrict class-map sync to categories already present in the noun/object library."""
        if not self.class_map:
            return 0

        existing_by_norm = {}
        for name in self.global_object_map.keys():
            norm_name = self._norm_category(name)
            if norm_name in ("left_hand", "right_hand"):
                continue
            existing_by_norm.setdefault(norm_name, name)

        if not existing_by_norm:
            if notify_user:
                QMessageBox.information(
                    self,
                    "Sync Complete",
                    "Loaded the YOLO class map, but skipped library sync because no noun/object list is loaded yet.",
                )
            return 0

        matched_count = 0
        for cid, class_name in self.class_map.items():
            norm_name = self._norm_category(class_name)
            if norm_name in ("left_hand", "right_hand"):
                continue

            existing_match = existing_by_norm.get(norm_name)
            if not existing_match:
                continue
            self.id_to_category[existing_match] = existing_match
            matched_count += 1

        if matched_count > 0 and notify_user:
            QMessageBox.information(
                self,
                "Sync Complete",
                f"Matched {matched_count} yaml classes to the current noun/object library.",
            )
        return int(matched_count)

    def _load_verbs_txt(self):
        """Import verb list txt with duplication check"""
        fp, _ = QFileDialog.getOpenFileName(
            self, "Import Verb List", "", "Text Files (*.txt);;All Files (*)"
        )
        if not fp:
            return
        self._load_verbs_from_path(fp, notify_user=True)

    def _load_verbs_from_path(
        self,
        fp: str,
        *,
        notify_user: bool = True,
        auto_discovered: bool = False,
    ) -> int:
        if not self._guard_asset_mutation(
            "verb list",
            path=fp,
            notify_user=notify_user,
            auto_discovered=auto_discovered,
        ):
            return 0
        try:
            lines = [
                ln.strip()
                for ln in open(fp, "r", encoding="utf-8").read().splitlines()
                if ln.strip()
            ]

            # Calc max ID for new assignment
            max_id = max([v.id for v in self.verbs], default=-1)
            added_count = 0

            for idx, ln in enumerate(lines):
                parts = ln.split()
                if not parts:
                    continue
                name = self._canonical_label_name(parts[0])
                if not name:
                    continue

                # Check for duplicates (case-insensitive)
                if any(v.name.lower() == name.lower() for v in self.verbs):
                    continue

                max_id += 1
                color_name = "Auto"
                self.verbs.append(LabelDef(name=name, id=max_id, color_name=color_name))
                added_count += 1

            self._renumber_verbs()
            self._update_verb_combo()
            self.label_panel.refresh()

            if notify_user and added_count > 0:
                QMessageBox.information(
                    self,
                    "Loaded",
                    f"Successfully imported {added_count} new verbs (duplicates skipped).",
                )
            elif notify_user:
                QMessageBox.information(
                    self, "Info", "No new verbs imported (all exist)."
                )
            self._update_onboarding_banner()
            self._log("hoi_load_verbs", path=fp, added=added_count, auto_discovered=bool(auto_discovered))
            self._log_annotation_ready_state("hoi_load_verbs")
            return int(added_count)

        except Exception as ex:
            if notify_user:
                QMessageBox.warning(
                    self,
                    "Verb List",
                    f"Failed to load verbs from:\n{fp}\n\n{ex}",
                )
            else:
                self._log("hoi_auto_load_failed", asset="verbs", path=fp, error=str(ex))
            return 0

    def _load_yolo_model(self):
        if not self._guard_experiment_mode("detection"):
            return
        """Load Ultralytics YOLOv11 .pt weights for current-frame detection."""
        fp, _ = QFileDialog.getOpenFileName(
            self, "Load YOLO Model", "", "YOLO Weights (*.pt *.pth);;All Files (*)"
        )
        if not fp:
            return
        self._load_yolo_model_from_path(fp, notify_user=True)

    def _load_yolo_model_from_path(
        self,
        fp: str,
        *,
        notify_user: bool = True,
        auto_discovered: bool = False,
    ) -> bool:
        if not self._guard_asset_mutation(
            "YOLO model",
            path=fp,
            notify_user=notify_user,
            auto_discovered=auto_discovered,
            full_assist_only=True,
        ):
            return False
        try:
            from ultralytics import YOLO
        except Exception as ex:
            if notify_user:
                QMessageBox.warning(
                    self, "Missing package", f"Ultralytics is not available:\n{ex}"
                )
            else:
                self._log("hoi_auto_load_failed", asset="yolo_model", path=fp, error=str(ex))
            return False
        try:
            self.yolo_model = YOLO(fp)
            self.yolo_weights_path = fp
            if notify_user:
                QMessageBox.information(
                    self, "Loaded", f"YOLO model loaded:\n{os.path.basename(fp)}"
                )
            self._log("hoi_load_yolo_model", path=fp, auto_discovered=bool(auto_discovered))
            self._log_annotation_ready_state("hoi_load_yolo_model")
            return True
        except Exception as ex:
            if notify_user:
                QMessageBox.warning(
                    self,
                    "YOLO Model",
                    f"Failed to load YOLO model:\n{fp}\n\n{ex}",
                )
            else:
                self._log("hoi_auto_load_failed", asset="yolo_model", path=fp, error=str(ex))
            return False

    def _load_semantic_adapter(self):
        if not self._guard_experiment_mode("semantic"):
            return
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Load Semantic Adapter Base",
            "",
            "Adapter Weights (*.pt *.pth);;All Files (*)",
        )
        if not fp:
            return
        if self._load_semantic_adapter_from_path(
            fp,
            notify_user=True,
            auto_discovered=False,
            record_as_base=True,
        ):
            self._ensure_semantic_adapter_loaded()
            if self.selected_event_id is not None:
                event = self._find_event_by_id(self.selected_event_id)
                if event is not None:
                    self._refresh_semantic_suggestions_for_event(self.selected_event_id, event)
                    self._update_status_label()
                    self._update_next_best_query_panel()

    def _load_videomae_model(self):
        if not self._guard_experiment_mode("semantic"):
            return
        fp, _ = QFileDialog.getOpenFileName(
            self, "Load Frozen Video Encoder", "", "Weight Files (*.pt *.pth *.ckpt *.bin *.safetensors);;All Files (*)"
        )
        if not fp:
            return
        self._store_videomae_weights_from_path(fp, notify_user=True)

    def _load_videomae_cache(self):
        if not self._guard_experiment_mode("semantic"):
            return
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Load Precomputed Encoder Cache",
            "",
            "NumPy Cache (*.npz);;All Files (*)",
        )
        if not fp:
            return
        self._load_precomputed_videomae_cache(fp, notify_user=True)

    def _load_videomae_verb_list(self):
        if not self._guard_experiment_mode("semantic"):
            return
        fp, _ = QFileDialog.getOpenFileName(
            self, "Load Optional Verb-Prior Labels", "", "YAML/JSON Files (*.yaml *.json);;Text Files (*.txt);;All Files (*)"
        )
        if not fp:
            return
        self._store_videomae_verb_list_from_path(fp, notify_user=True)

    def _store_videomae_weights_from_path(
        self,
        fp: str,
        *,
        notify_user: bool = True,
        auto_discovered: bool = False,
    ) -> bool:
        if not self._guard_asset_mutation(
            "VideoMAE weights",
            path=fp,
            notify_user=notify_user,
            auto_discovered=auto_discovered,
            full_assist_only=True,
        ):
            return False
        self.videomae_weights_path = fp
        self._videomae_loaded_key = ""
        self._clear_videomae_action_runtime(clear_event_scores=False)
        return self._init_videomae(
            notify_user=notify_user,
            auto_discovered=auto_discovered,
        )

    def _store_videomae_verb_list_from_path(
        self,
        fp: str,
        *,
        notify_user: bool = True,
        auto_discovered: bool = False,
    ) -> bool:
        if not self._guard_asset_mutation(
            "VideoMAE verb labels",
            path=fp,
            notify_user=notify_user,
            auto_discovered=auto_discovered,
            full_assist_only=True,
        ):
            return False
        self.videomae_verb_list_path = fp
        self._videomae_loaded_key = ""
        self._clear_videomae_action_runtime(clear_event_scores=False)
        if self.videomae_weights_path:
            return self._init_videomae(notify_user=notify_user, auto_discovered=auto_discovered)
        if notify_user:
            QMessageBox.information(self, "Info", "Verb list stored. Please load model weights to initialize.")
        self._log("hoi_store_videomae_verbs", path=fp, auto_discovered=bool(auto_discovered))
        self._log_annotation_ready_state("hoi_store_videomae_verbs")
        return True

    def _videomae_model_key(self) -> str:
        weights = os.path.abspath(str(self.videomae_weights_path or "").strip())
        verbs = os.path.abspath(str(self.videomae_verb_list_path or "").strip())
        if not weights:
            return ""
        return f"{weights}|{verbs}"

    def _videomae_ready(self) -> bool:
        return bool(
            self.video_path
            and self.videomae_handler.model is not None
            and self.videomae_handler.processor is not None
            and self._videomae_loaded_key
            and self._videomae_loaded_key == self._videomae_model_key()
        )

    def _clear_videomae_action_runtime(self, clear_event_scores: bool = False):
        try:
            self._videomae_auto_timer.stop()
        except Exception:
            pass
        self._videomae_auto_event_id = None
        self._videomae_auto_force_refresh = False
        self._videomae_action_cache.clear()
        self._videomae_feature_cache.clear()
        self._videomae_local_action_cache.clear()
        self._videomae_local_feature_cache.clear()
        self._videomae_event_signatures.clear()
        if clear_event_scores:
            for event in self.events:
                if isinstance(event, dict):
                    event.pop("videomae_top5", None)
                    event.pop("videomae_local_top5", None)
                    event.pop("videomae_local_meta", None)
                    for actor in self.actors_config:
                        hand_data = event.get("hoi_data", {}).get(actor["id"], {}) or {}
                        self._ensure_hand_annotation_state(hand_data)
                        suggestion = hand_data.get("_field_suggestions", {}).get("verb", {})
                        source = str((suggestion or {}).get("source") or "").strip()
                        if source.startswith("videomae"):
                            clear_field_suggestion(hand_data, "verb")
            for actor in self.actors_config:
                draft = self.event_draft.get(actor["id"], {}) or {}
                self._ensure_hand_annotation_state(draft)
                suggestion = draft.get("_field_suggestions", {}).get("verb", {})
                source = str((suggestion or {}).get("source") or "").strip()
                if source.startswith("videomae"):
                    clear_field_suggestion(draft, "verb")

    def _normalize_videomae_candidates(self, candidates) -> List[dict]:
        rows: List[dict] = []
        for cand in list(candidates or []):
            if not isinstance(cand, dict):
                continue
            label = str(cand.get("label") or "").strip()
            if not label:
                continue
            score = cand.get("score")
            try:
                score = None if score is None else float(score)
            except Exception:
                score = None
            rows.append({"label": label, "score": score})
        return rows

    def _primary_onset_context_for_event(self, event: Optional[dict]) -> dict:
        if not isinstance(event, dict):
            return {}
        ordered_hands: List[str] = []
        if (
            self.selected_event_id is not None
            and int(event.get("event_id", -1)) == int(self.selected_event_id)
            and self.selected_hand_label
        ):
            ordered_hands.append(str(self.selected_hand_label))
        for actor in self.actors_config:
            hand_key = str(actor["id"])
            if hand_key not in ordered_hands:
                ordered_hands.append(hand_key)

        for hand_key in ordered_hands:
            hand_data = event.get("hoi_data", {}).get(hand_key, {}) or {}
            try:
                start = (
                    int(hand_data.get("interaction_start"))
                    if hand_data.get("interaction_start") is not None
                    else None
                )
                end = (
                    int(hand_data.get("interaction_end"))
                    if hand_data.get("interaction_end") is not None
                    else None
                )
                onset = (
                    int(hand_data.get("functional_contact_onset"))
                    if hand_data.get("functional_contact_onset") is not None
                    else None
                )
            except Exception:
                start = end = onset = None
            if start is None or end is None:
                continue
            onset_state = get_field_state(hand_data, "functional_contact_onset")
            track_prior = self._handtrack_segment_prior(hand_key, start, end)
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
                "hand": hand_key,
                "start_frame": int(start),
                "end_frame": int(end),
                "onset_frame": onset_for_context,
                "onset_status": onset_status,
                "onset_band": band,
                "onset_source": onset_source,
                "handtrack_prior": dict(track_prior or {}),
            }
        return {}

    def _videomae_signature_for_event(self, event: Optional[dict]) -> str:
        if not event or (
            not self._videomae_ready() and not self._videomae_precomputed_ready()
        ):
            return ""
        start, end = self._compute_event_frames(event)
        if start is None or end is None:
            return ""
        onset_context = self._primary_onset_context_for_event(event)
        payload = {
            "video_path": os.path.abspath(str(self.video_path or "").strip()),
            "start": int(start),
            "end": int(end),
            "onset_frame": onset_context.get("onset_frame"),
            "onset_status": onset_context.get("onset_status"),
            "onset_band": onset_context.get("onset_band"),
            "model": self._videomae_loaded_key
            or str(getattr(self, "_videomae_precomputed_cache_path", "") or ""),
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def _cached_videomae_candidates(self, event_id: int, event: Optional[dict]):
        signature = self._videomae_signature_for_event(event)
        if not signature:
            return [], ""
        rows: List[dict] = []
        if self._videomae_event_signatures.get(int(event_id)) == signature and event:
            rows = self._normalize_videomae_candidates(event.get("videomae_top5"))
        if not rows:
            rows = self._normalize_videomae_candidates(
                self._videomae_action_cache.get(signature)
            )
        return rows, signature

    def _store_videomae_candidates(
        self,
        event_id: int,
        event: Optional[dict],
        signature: str,
        candidates,
    ) -> List[dict]:
        clean = self._normalize_videomae_candidates(candidates)
        stored = [dict(row) for row in clean]
        if event is not None:
            event["videomae_top5"] = [dict(row) for row in stored]
        if signature and stored:
            self._videomae_action_cache[signature] = [dict(row) for row in stored]
            self._videomae_event_signatures[int(event_id)] = signature
        return stored

    def _invalidate_videomae_candidates(self, event_id: Optional[int]) -> None:
        if event_id is None:
            return
        try:
            key = int(event_id)
        except Exception:
            return
        self._videomae_event_signatures.pop(key, None)

    def _queue_action_label_refresh(
        self,
        event_id: Optional[int] = None,
        delay_ms: Optional[int] = None,
        force: bool = False,
    ) -> None:
        if not self._semantic_assist_enabled():
            return
        if event_id is None:
            event_id = self.selected_event_id
        if event_id is None:
            return
        try:
            event_id = int(event_id)
        except Exception:
            return
        event = self._find_event_by_id(event_id)
        if not event:
            return
        if not self._videomae_ready() and not self._videomae_precomputed_ready():
            if self.selected_event_id == event_id:
                self._update_action_top5_display(event_id)
            return
        if not force:
            cached, signature = self._cached_videomae_candidates(event_id, event)
            cached_local = self._cached_videomae_local_candidates(event_id, event)
            has_segment_feature = bool(self._videomae_feature_cache.get(signature))
            has_local_feature = bool(self._videomae_local_feature_cache.get(signature))
            local_context = self._primary_local_onset_context_for_event(event)
            has_local_ready = (
                not bool(local_context) or bool(cached_local) or bool(has_local_feature)
            )
            if cached or (has_segment_feature and has_local_ready):
                if self.selected_event_id == event_id:
                    self._update_action_top5_display(event_id)
                return
        self._videomae_auto_event_id = event_id
        self._videomae_auto_force_refresh = bool(force)
        wait_ms = self._videomae_auto_timer.interval() if delay_ms is None else max(0, int(delay_ms))
        self._videomae_auto_timer.start(wait_ms)

    def _run_pending_action_label_refresh(self) -> None:
        event_id = self._videomae_auto_event_id
        force = bool(self._videomae_auto_force_refresh)
        self._videomae_auto_event_id = None
        self._videomae_auto_force_refresh = False
        if event_id is None or self.selected_event_id != event_id:
            return
        self._detect_action_label(
            event_id,
            interactive=False,
            auto_apply=False,
            use_cache=not force,
            force=force,
            report_errors=False,
        )

    def _init_videomae(self, *, notify_user: bool = True, auto_discovered: bool = False) -> bool:
        self._clear_videomae_action_runtime(clear_event_scores=False)
        success, msg = self.videomae_handler.load_model(
            self.videomae_weights_path, self.videomae_verb_list_path
        )
        if success:
            self._videomae_loaded_key = self._videomae_model_key()
            if notify_user:
                QMessageBox.information(self, "Success", f"VideoMAE V2 initialized:\n{msg}")
            self._log(
                "hoi_load_videomae",
                weights=self.videomae_weights_path,
                verbs=self.videomae_verb_list_path,
                auto_discovered=bool(auto_discovered),
            )
            self._log_annotation_ready_state("hoi_load_videomae")
            if self.selected_event_id is not None:
                self._queue_action_label_refresh(self.selected_event_id, delay_ms=120)
            return True
        else:
            self._videomae_loaded_key = ""
            if notify_user:
                QMessageBox.warning(
                    self,
                    "VideoMAE",
                    (
                        "Failed to load VideoMAE.\n\n"
                        f"Weights: {self.videomae_weights_path or '(not set)'}\n"
                        f"Verb list: {self.videomae_verb_list_path or '(not set)'}\n\n"
                        f"{msg}"
                    ),
                )
            else:
                self._log(
                    "hoi_auto_load_failed",
                    asset="videomae",
                    weights=self.videomae_weights_path,
                    verbs=self.videomae_verb_list_path,
                    error=str(msg),
                )
            return False

    def _videomae_hand_edit_snapshot(self, hand_data):
        # Candidate metadata is deliberately excluded; human field state is not.
        if not isinstance(hand_data, dict):
            return None
        self._ensure_hand_annotation_state(hand_data)
        keys = ("verb", "noun_object_id", "target_object_id",
                "instrument_object_id", "tool_object_id", "interaction_start",
                "functional_contact_onset", "interaction_end", "_field_state")
        return copy.deepcopy({key: hand_data.get(key) for key in keys})

    def _videomae_request_target_is_current(self, request, event):
        hand = request.get("target_hand")
        if not hand or not request.get("target_snapshot"):
            return False
        if request.get("video_path") != self.video_path:
            return False
        # A late result must not open a dialog or edit a different selection.
        if (self.selected_event_id != request.get("event_id")
                or self.selected_hand_label != hand):
            return False
        if request.get("signature") != self._videomae_signature_for_event(event):
            return False
        return (
            request["target_snapshot"] == self._videomae_hand_edit_snapshot(
                event.get("hoi_data", {}).get(hand))
            and request.get("draft_snapshot") == self._videomae_hand_edit_snapshot(
                self.event_draft.get(hand))
        )

    def _build_videomae_request(
        self,
        event_id,
        *,
        interactive=True,
        auto_apply=True,
        use_cache=True,
        force=False,
        report_errors=False,
        target_hand=None,
    ) -> Optional[Dict[str, Any]]:
        notify_user = bool(interactive or report_errors)
        event = self._find_event_by_id(event_id)
        if not event:
            if notify_user:
                QMessageBox.warning(self, "Warning", "Please select an action segment first.")
            return None

        start, end = self._compute_event_frames(event)
        if start is None or end is None:
            if notify_user:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Action segment must have start and end frames.",
                )
            return None

        signature = self._videomae_signature_for_event(event)
        onset_context = self._primary_onset_context_for_event(event)
        local_onset_context = self._primary_local_onset_context_for_event(event)
        candidates: List[dict] = []
        local_candidates: List[dict] = []
        cached_segment_feature: List[float] = []
        cached_local_segment_feature: List[float] = []
        cached_segment_meta: Dict[str, Any] = {}
        cached_local_meta: Dict[str, Any] = {}
        if use_cache and not force:
            candidates, signature = self._cached_videomae_candidates(int(event_id), event)
            local_candidates = self._cached_videomae_local_candidates(int(event_id), event)
            cached_segment_feature = list(self._videomae_feature_cache.get(signature) or [])
            cached_local_segment_feature = list(
                self._videomae_local_feature_cache.get(signature) or []
            )
            if (
                not candidates
                or not local_candidates
                or not cached_segment_feature
                or (
                    bool(local_onset_context)
                    and not cached_local_segment_feature
                )
            ):
                precomputed = self._precomputed_videomae_summary_for_event(event)
                if not candidates:
                    candidates = [dict(row) for row in list(precomputed.get("candidates") or [])]
                if not local_candidates:
                    local_candidates = [dict(row) for row in list(precomputed.get("local_candidates") or [])]
                if not cached_segment_feature:
                    cached_segment_feature = list(precomputed.get("segment_feature") or [])
                if not cached_local_segment_feature:
                    cached_local_segment_feature = list(precomputed.get("local_segment_feature") or [])
                cached_segment_meta = dict(precomputed.get("segment_meta") or {})
                cached_local_meta = dict(precomputed.get("local_meta") or {})

        need_global = not bool(candidates) and not bool(cached_segment_feature)
        need_local = (
            bool(local_onset_context)
            and not bool(local_candidates)
            and not bool(cached_local_segment_feature)
        )
        if need_global and not self._videomae_ready():
            if notify_user:
                QMessageBox.warning(
                    self,
                    "Detection Error",
                    "Load the assist model or precomputed clip features first.",
                )
            return None

        if (self.selected_event_id != int(event_id)
                or target_hand != self.selected_hand_label
                or target_hand not in event.get("hoi_data", {})
                or target_hand not in self.event_draft):
            target_hand = None
        target_snapshot = self._videomae_hand_edit_snapshot(
            event.get("hoi_data", {}).get(target_hand)) if target_hand else None
        draft_snapshot = self._videomae_hand_edit_snapshot(
            self.event_draft.get(target_hand)) if target_hand else None

        return {
            "target_hand": target_hand,
            "target_snapshot": target_snapshot,
            "draft_snapshot": draft_snapshot,
            "event_id": int(event_id),
            "interactive": bool(interactive),
            "auto_apply": bool(auto_apply),
            "use_cache": bool(use_cache),
            "force": bool(force),
            "report_errors": bool(report_errors),
            "notify_user": bool(notify_user),
            "video_path": self.video_path,
            "start": int(start),
            "end": int(end),
            "signature": str(signature or ""),
            "onset_context": dict(onset_context or {}),
            "local_onset_context": dict(local_onset_context or {}),
            "cached_candidates": [dict(row) for row in list(candidates or [])],
            "cached_local_candidates": [dict(row) for row in list(local_candidates or [])],
            "cached_segment_feature": list(cached_segment_feature or []),
            "cached_local_segment_feature": list(cached_local_segment_feature or []),
            "cached_segment_meta": dict(cached_segment_meta or {}),
            "cached_local_meta": dict(cached_local_meta or {}),
            "need_global": bool(need_global),
            "need_local": bool(need_local),
        }

    def _start_videomae_inference(self, request: Dict[str, Any]) -> bool:
        if not request:
            return False
        if self._videomae_infer_worker is not None:
            self._videomae_pending_request = dict(request)
            return True
        self._videomae_infer_request = dict(request)
        self._videomae_infer_worker = VideoMAEInferenceWorker(
            self.videomae_handler,
            request,
        )
        self._videomae_infer_worker.finished.connect(self._on_videomae_inference_finished)
        self._refresh_inference_action_states()
        self._log(
            "hoi_videomae_async_start",
            event_id=request.get("event_id"),
            interactive=bool(request.get("interactive")),
            auto_apply=bool(request.get("auto_apply")),
        )
        self._videomae_infer_worker.start()
        return True

    def _apply_videomae_result(self, payload) -> bool:
        request = dict((payload or {}).get("request") or self._videomae_infer_request or {})
        if not request:
            return False
        notify_user = bool(request.get("notify_user"))
        success = bool((payload or {}).get("success"))
        if not success:
            err = str((payload or {}).get("error") or "VideoMAE inference failed.")
            self._log(
                "hoi_videomae_async_failed",
                event_id=request.get("event_id"),
                error=err,
            )
            if notify_user:
                QMessageBox.warning(self, "Detection Error", err)
            return False

        event_id = int(request.get("event_id"))
        event = self._find_event_by_id(event_id)
        if not event:
            return False

        signature = str(request.get("signature") or "")
        candidates = [dict(row) for row in list(request.get("cached_candidates") or [])]
        local_candidates = [dict(row) for row in list(request.get("cached_local_candidates") or [])]
        if not candidates:
            candidates = self._normalize_videomae_candidates((payload or {}).get("candidates"))
        if not local_candidates:
            local_candidates = self._normalize_videomae_candidates(
                (payload or {}).get("local_candidates")
            )

        predicted_segment_feature = list(request.get("cached_segment_feature") or [])
        if not predicted_segment_feature:
            predicted_segment_feature = self._normalize_dense_feature(
                ((payload or {}).get("segment_meta") or {}).get("segment_feature"),
                length=int(getattr(self, "_semantic_videomae_feature_dim", 0) or 0),
            )
        predicted_local_segment_feature = list(request.get("cached_local_segment_feature") or [])
        if not predicted_local_segment_feature:
            predicted_local_segment_feature = self._normalize_dense_feature(
                ((payload or {}).get("local_meta") or {}).get("segment_feature"),
                length=int(getattr(self, "_semantic_videomae_local_feature_dim", 0) or 0),
            )
        predicted_segment_meta = dict(
            (payload or {}).get("segment_meta") or request.get("cached_segment_meta") or {}
        )
        predicted_local_meta = dict(
            (payload or {}).get("local_meta") or request.get("cached_local_meta") or {}
        )
        current_signature = self._videomae_signature_for_event(event)
        event_is_current = (bool(signature) and current_signature == signature
                            and request.get("video_path") == self.video_path)
        event_for_store = event if event_is_current else None

        if candidates:
            candidates = self._store_videomae_candidates(
                event_id,
                event_for_store,
                signature,
                candidates,
            )
        if signature and predicted_segment_feature:
            self._videomae_feature_cache[signature] = list(predicted_segment_feature)
        if signature and local_candidates:
            self._videomae_local_action_cache[signature] = [dict(row) for row in local_candidates]
        if signature and predicted_local_segment_feature:
            self._videomae_local_feature_cache[signature] = list(predicted_local_segment_feature)
        if event_for_store is not None:
            onset_context = dict(request.get("onset_context") or {})
            prev_meta = dict(event.get("videomae_meta") or {})
            event["videomae_meta"] = {
                "sampling_mode": str(
                    predicted_segment_meta.get("sampling_mode")
                    or prev_meta.get("sampling_mode")
                    or ""
                ).strip(),
                "indices": list(
                    predicted_segment_meta.get("indices") or prev_meta.get("indices") or []
                ),
                "onset_band": dict(onset_context.get("onset_band") or {}),
            }
            if local_candidates:
                prev_local_meta = dict(event.get("videomae_local_meta") or {})
                event["videomae_local_top5"] = [dict(row) for row in local_candidates]
                event["videomae_local_meta"] = {
                    "sampling_mode": str(
                        predicted_local_meta.get("sampling_mode")
                        or prev_local_meta.get("sampling_mode")
                        or "onset_local_window"
                    ).strip(),
                    "indices": list(
                        predicted_local_meta.get("indices")
                        or prev_local_meta.get("indices")
                        or []
                    ),
                    "window": dict(request.get("local_onset_context") or {}),
                }
            if candidates and self._videomae_request_target_is_current(request, event):
                self._sync_event_videomae_suggestions(
                    event_id, event, candidates, target_hand=request["target_hand"])
            # Do not trigger the all-hand semantic autofill from a clip suggestion.
            if candidates and self.selected_event_id == event_id:
                self._update_action_top5_display(event_id)

        if not candidates:
            if notify_user and bool(getattr(self.videomae_handler, "enable_verb_prior", False)):
                QMessageBox.information(self, "Info", "No labels predicted.")
            elif notify_user and predicted_segment_feature:
                QMessageBox.information(
                    self,
                    "Info",
                    "Encoder features refreshed. Direct verb labels are unavailable in feature-only mode.",
                )
            return bool(predicted_segment_feature or predicted_local_segment_feature)

        if event_for_store is None or not self._videomae_request_target_is_current(request, event):
            return True

        interactive = bool(request.get("interactive"))
        auto_apply = bool(request.get("auto_apply"))
        selected = None
        if interactive:
            dlg = ActionLabelSelector(candidates, self)
            if dlg.exec_() == QDialog.Accepted:
                selected = dlg.get_selected_label()
            else:
                return True
        elif auto_apply and candidates:
            selected = candidates[0]["label"]

        if not selected:
            return True

        verb_id = -1
        for v in self.verbs:
            if v.name.lower() == selected.lower():
                verb_id = v.id
                break

        if verb_id == -1:
            if notify_user:
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Verb '{selected}' not found in the current project verb list.",
                )
            return True

        # Recheck after the modal dialog (nested Qt event loop).
        if not self._videomae_request_target_is_current(request, event):
            self._log("hoi_skip_stale_verb_suggestion", event_id=event_id)
            return True
        hid = request["target_hand"]
        hand_data = event["hoi_data"][hid]
        draft = self.event_draft[hid]
        if not interactive and any(
            get_field_state(row, "verb").get("status") in ("confirmed", "locked")
            for row in (hand_data, draft)
        ):
            self._log("hoi_skip_locked_verb_autofill", event_id=event_id,
                      hand=hid, suggested=selected)
            return True

        self._push_undo()
        source_name = "videomae_selected" if interactive else "videomae_top1_auto"
        target_status = "confirmed" if interactive else "suggested"
        # event['verb_id'] belongs to the legacy event-wide action, not both hands.
        for row in (hand_data, draft):
            row["verb"] = selected
            self._set_hand_field_state(row, "verb", source=source_name,
                                       status=target_status)
        self._log("hoi_apply_scoped_verb_suggestion", event_id=event_id,
                  hand=hid, verb=selected, source=source_name)

        self._refresh_events()
        if hasattr(self, "hoi_timeline") and self.hoi_timeline:
            self.hoi_timeline.refresh()
        if self.selected_event_id == event_id:
            if self.selected_hand_label:
                self._load_hand_draft_to_ui(self.selected_hand_label)
            else:
                self._sync_action_panel_selection(selected)
        return True

    def _finish_videomae_batch(self, canceled: bool = False) -> None:
        progress = self._videomae_batch_progress
        self._videomae_batch_progress = None
        self._videomae_batch_requests = []
        if progress is not None:
            try:
                progress.close()
                progress.deleteLater()
            except Exception:
                pass
        self._refresh_inference_action_states()
        if canceled:
            QMessageBox.information(self, "Canceled", "Batch action detection canceled.")
        else:
            QMessageBox.information(self, "Finished", "Verb candidates refreshed. Select a hand to review and apply a candidate.")

    def _run_next_videomae_batch_request(self) -> None:
        progress = self._videomae_batch_progress
        if progress is None:
            return
        if progress.wasCanceled():
            self._finish_videomae_batch(canceled=True)
            return
        total = int(progress.maximum())
        processed = max(0, total - len(self._videomae_batch_requests))
        progress.setValue(processed)
        if not self._videomae_batch_requests:
            progress.setValue(total)
            self._finish_videomae_batch(canceled=False)
            return
        next_event_id = self._videomae_batch_requests.pop(0)
        request = self._build_videomae_request(
            next_event_id,
            interactive=False,
            auto_apply=False,
            use_cache=False,
            force=True,
            report_errors=False,
        )
        if not request:
            QTimer.singleShot(0, self._run_next_videomae_batch_request)
            return
        self._start_videomae_inference(request)

    def _on_videomae_inference_finished(self, payload) -> None:
        worker = self._videomae_infer_worker
        self._videomae_infer_worker = None
        self._videomae_infer_request = {}
        try:
            self._apply_videomae_result(payload)
        finally:
            self._refresh_inference_action_states()
            if worker is not None:
                worker.deleteLater()
            if self._videomae_batch_progress is not None:
                QTimer.singleShot(0, self._run_next_videomae_batch_request)
                return
            pending = self._videomae_pending_request
            self._videomae_pending_request = None
            if pending:
                QTimer.singleShot(
                    0,
                    lambda req=dict(pending): self._start_videomae_inference(req),
                )

    def _detect_selected_action_label(self):
        if not self._guard_experiment_mode("semantic"):
            return
        if self.selected_event_id is None:
            QMessageBox.warning(self, "Warning", "Please select an action segment first.")
            return
        self._detect_action_label(
            self.selected_event_id,
            interactive=True,
            auto_apply=True,
            use_cache=True,
            force=False,
            report_errors=True,
        )

    def _refresh_selected_action_label(self):
        if not self._guard_experiment_mode("semantic"):
            return
        if self.selected_event_id is None:
            QMessageBox.warning(self, "Warning", "Please select an action segment first.")
            return
        self._detect_action_label(
            self.selected_event_id,
            interactive=False,
            auto_apply=False,
            use_cache=False,
            force=True,
            report_errors=True,
        )

    def _detect_action_label(
        self,
        event_id,
        interactive=True,
        auto_apply=True,
        use_cache=True,
        force=False,
        report_errors=False,
    ):
        request = self._build_videomae_request(
            event_id,
            interactive=interactive,
            auto_apply=auto_apply,
            use_cache=use_cache,
            force=force,
            report_errors=report_errors,
            target_hand=(self.selected_hand_label
                         if self.selected_event_id == event_id else None),
        )
        if not request:
            return False
        if not request.get("need_global") and not request.get("need_local"):
            return self._apply_videomae_result(
                {
                    "success": True,
                    "request": request,
                    "candidates": request.get("cached_candidates") or [],
                    "local_candidates": request.get("cached_local_candidates") or [],
                    "segment_meta": {},
                    "local_meta": {},
                }
            )
        self._start_videomae_inference(request)
        return True

    def _update_action_top5_display(self, event_id):
        if not hasattr(self, "label_panel") or not self.label_panel:
            return
        if not self._semantic_assist_enabled():
            self.label_panel.clear_candidate_priority()
            self._sync_action_panel_selection()
            self._update_next_best_query_panel()
            return

        if event_id is None:
            self.label_panel.clear_candidate_priority()
            self._sync_action_panel_selection()
            self._update_next_best_query_panel()
            return

        event = self._find_event_by_id(event_id)
        candidates, _signature = self._cached_videomae_candidates(int(event_id), event)
        if not event or not candidates:
            self.label_panel.clear_candidate_priority()
            self._sync_action_panel_selection()
            self._update_next_best_query_panel()
            return

        raw_candidates = [
            (cand.get("label"), cand.get("score"))
            for cand in candidates
            if cand.get("label")
        ]
        visible_candidates = [
            (name, score)
            for name, score in raw_candidates
            if self.label_panel.index_of_label(name) >= 0
        ]
        if visible_candidates:
            self.label_panel.set_candidate_priority(visible_candidates)
        else:
            self.label_panel.clear_candidate_priority()
        self._sync_action_panel_selection()
        self._update_next_best_query_panel()

    def _detect_all_action_labels(self):
        if not self._guard_experiment_mode("semantic"):
            return
        if not self.events:
            QMessageBox.information(self, "Info", "No actions to detect.")
            return
        if not self._videomae_ready():
            QMessageBox.information(
                self,
                "Info",
                "Load frozen video-encoder weights before batch verb ranking.",
            )
            return
        if not bool(getattr(self.videomae_handler, "enable_verb_prior", False)):
            QMessageBox.information(
                self,
                "Info",
                "Batch verb ranking requires encoder-side verb labels. The current encoder load is feature-only mode.",
            )
            return
        if self._videomae_infer_worker is not None or self._videomae_batch_progress is not None:
            QMessageBox.information(
                self,
                "Assist Inference",
                "An assist job is already running. Wait for it to finish first.",
            )
            return
        self._videomae_pending_request = None
        self._videomae_batch_requests = [int(event.get("event_id")) for event in list(self.events or [])]
        self._videomae_batch_progress = QProgressDialog(
            "Computing verb rankings for all events...",
            "Cancel",
            0,
            len(self._videomae_batch_requests),
            self,
        )
        self._videomae_batch_progress.setWindowModality(Qt.WindowModal)
        self._videomae_batch_progress.setValue(0)
        self._videomae_batch_progress.show()
        self._run_next_videomae_batch_request()

    def _ask_existing_policy(self):
        """Ask how to handle existing boxes on the current frame."""
        box = QMessageBox(self)
        box.setWindowTitle("Existing boxes")
        box.setText(
            "This frame already has object boxes. How do you want to apply YOLO results?"
        )
        append_btn = box.addButton("Append", QMessageBox.AcceptRole)
        replace_btn = box.addButton("Replace", QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        remember = QCheckBox("Remember my choice")
        box.setCheckBox(remember)
        box.exec_()
        clicked = box.clickedButton()
        policy = None
        if clicked == append_btn:
            policy = "append"
        elif clicked == replace_btn:
            policy = "replace"
        if policy and remember.isChecked():
            self.yolo_existing_policy = policy
        return policy

    def _read_video_frames_bgr(self, frame_indices) -> Dict[int, Any]:
        out: Dict[int, Any] = {}
        indices = []
        for value in list(frame_indices or []):
            try:
                indices.append(int(value))
            except Exception:
                continue
        if not indices or not self.video_path:
            return out
        cap = cv2.VideoCapture(self.video_path)
        try:
            if not cap.isOpened():
                return out
            for frame_idx in sorted(set(indices)):
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                    ok, frame_bgr = cap.read()
                except Exception:
                    ok, frame_bgr = False, None
                if ok and frame_bgr is not None:
                    out[int(frame_idx)] = frame_bgr.copy()
        finally:
            try:
                cap.release()
            except Exception:
                pass
        return out

    def _frame_has_raw_hand_label(
        self,
        frame_idx: int,
        hand_key: Optional[str],
    ) -> bool:
        norm_hand = self._normalize_hand_label(hand_key)
        if not norm_hand:
            return False
        frame_idx = int(frame_idx)
        for rb in list(getattr(self, "raw_boxes", []) or []):
            target_frame = int(rb.get("orig_frame", 0) or 0) + int(self.start_offset)
            if target_frame != frame_idx:
                continue
            if self._normalize_hand_label(rb.get("label")) == norm_hand:
                return True
        return False

    @staticmethod
    def _box_confidence_value(box: Optional[dict]) -> float:
        if not isinstance(box, dict):
            return 0.0
        for key in ("confidence", "detection_confidence", "handedness_score"):
            try:
                value = float(box.get(key))
            except Exception:
                continue
            if math.isfinite(value):
                if value > 1.0:
                    value = value / 100.0
                return max(0.0, min(1.0, value))
        return 0.0

    @staticmethod
    def _protected_hand_box_source(box: Optional[dict]) -> bool:
        source = str((box or {}).get("source") or "").strip().lower()
        return (
            source.startswith("manual_")
            or source == "loaded_annotation"
            or source.startswith("hands_xml")
        )

    def _hand_initial_box_score(self, box: Optional[dict]) -> float:
        source = str((box or {}).get("source") or "").strip().lower()
        conf = self._box_confidence_value(box)
        if "mediapipe" in source:
            return 0.74 + 0.20 * conf
        if "yolo" in source:
            return 0.60 + 0.25 * conf
        if "hands_xml" in source:
            return 0.68 + 0.05 * conf
        return 0.50 + 0.20 * conf

    def _apply_hand_box_candidates(
        self,
        frame_idx: int,
        candidates: Sequence[dict],
        *,
        push_undo: bool = True,
        refresh_view: bool = True,
    ) -> int:
        """Apply model hand boxes while preserving locked/manual edits."""
        frame_idx = int(frame_idx)
        candidate_by_hand: Dict[str, dict] = {}
        for candidate in list(candidates or []):
            if not isinstance(candidate, dict):
                continue
            norm_label = self._normalize_hand_label(candidate.get("label"))
            if not norm_label:
                continue
            next_candidate = dict(candidate)
            next_candidate["label"] = norm_label
            prev = candidate_by_hand.get(norm_label)
            if prev is None or self._hand_initial_box_score(next_candidate) > self._hand_initial_box_score(prev):
                candidate_by_hand[norm_label] = next_candidate
        if not candidate_by_hand:
            return 0

        existing_by_hand: Dict[str, List[dict]] = {key: [] for key in candidate_by_hand}
        keep = []
        for rb in list(self.raw_boxes or []):
            target_frame = int(rb.get("orig_frame", 0) or 0) + int(self.start_offset)
            norm = self._normalize_hand_label(rb.get("label"))
            if target_frame == frame_idx and norm in candidate_by_hand:
                existing_by_hand.setdefault(norm, []).append(rb)
                continue
            keep.append(rb)

        applied_candidates = []
        preserved_count = 0
        replaced_count = 0
        for norm, candidate in candidate_by_hand.items():
            existing = list(existing_by_hand.get(norm) or [])
            if any(self._is_box_locked(box) or self._protected_hand_box_source(box) for box in existing):
                keep.extend(existing)
                preserved_count += len(existing)
                continue
            if existing:
                best_existing = max(
                    self._hand_initial_box_score(box) for box in existing
                )
                if best_existing >= self._hand_initial_box_score(candidate):
                    keep.extend(existing)
                    preserved_count += len(existing)
                    continue
                replaced_count += len(existing)
            applied_candidates.append(candidate)

        if not applied_candidates and replaced_count <= 0:
            if refresh_view:
                self._update_status_label()
            return 0
        if push_undo:
            self._push_undo()
        self.raw_boxes = keep
        for candidate in applied_candidates:
            new_rb = {
                "id": self.box_id_counter,
                "orig_frame": frame_idx - self.start_offset,
                "label": candidate.get("label"),
                "source": candidate.get("source") or "model_hand",
                "confidence": self._box_confidence_value(candidate),
                "locked": False,
                "x1": candidate.get("x1"),
                "y1": candidate.get("y1"),
                "x2": candidate.get("x2"),
                "y2": candidate.get("y2"),
            }
            new_rb["model_score"] = self._hand_initial_box_score(new_rb)
            self.raw_boxes.append(new_rb)
            self.box_id_counter += 1

        self._rebuild_bboxes_from_raw()
        self._bump_bbox_revision()
        self._bump_query_state_revision()
        if refresh_view:
            self._refresh_boxes_for_frame(frame_idx)
            self._update_status_label()
        self._log(
            "hoi_apply_hand_candidates",
            frame=frame_idx,
            applied=len(applied_candidates),
            preserved=preserved_count,
            replaced=replaced_count,
        )
        return len(applied_candidates)

    def _ensure_hand_boxes_for_detection_frames(
        self,
        frame_map: Dict[int, Any],
        frame_to_hand_keys: Dict[int, Sequence[str]],
    ) -> int:
        """Materialize MediaPipe hand boxes before object detection on action frames."""
        if not isinstance(frame_map, dict) or not frame_map:
            return 0
        detected_total = 0
        undo_pushed = False
        checked_frames = 0
        for frame_idx in sorted(frame_map.keys()):
            try:
                frame_idx_int = int(frame_idx)
            except Exception:
                continue
            frame_bgr = frame_map.get(frame_idx)
            if frame_bgr is None:
                frame_bgr = frame_map.get(frame_idx_int)
            if frame_bgr is None:
                continue
            hand_keys = frame_to_hand_keys.get(frame_idx_int) or []
            required_hands: List[str] = []
            for value in list(hand_keys or []):
                norm_hand = self._normalize_hand_label(value)
                if norm_hand and norm_hand not in required_hands:
                    required_hands.append(norm_hand)
            if not required_hands:
                continue
            if all(
                self._frame_has_raw_hand_label(frame_idx_int, hand_key)
                for hand_key in required_hands
            ):
                continue
            checked_frames += 1
            detected = int(
                self._detect_current_frame_hands(
                    frame_bgr,
                    frame_idx=frame_idx_int,
                    push_undo=not undo_pushed,
                    target_hand_keys=required_hands,
                    refresh_view=False,
                )
                or 0
            )
            if detected > 0:
                detected_total += detected
                undo_pushed = True
        if detected_total > 0:
            try:
                current_frame = int(getattr(self.player, "current_frame", 0) or 0)
            except Exception:
                current_frame = 0
            if current_frame in frame_map:
                self._refresh_boxes_for_frame(current_frame)
            self._log(
                "hoi_auto_detect_hands_keyframes",
                frames=checked_frames,
                count=detected_total,
            )
        return int(detected_total)

    def _frame_has_selected_hand_box(
        self,
        frame_idx: int,
        hand_key: Optional[str] = None,
    ) -> bool:
        frame_idx = int(frame_idx)
        norm_hand = self._normalize_hand_label(hand_key or self.selected_hand_label)
        if not norm_hand:
            return False
        return bool(self._frame_hand_box(norm_hand, frame_idx))

    def _frame_has_object_boxes(self, frame_idx: int) -> bool:
        frame_idx = int(frame_idx)
        for rb in list(getattr(self, "raw_boxes", []) or []):
            target_frame = int(rb.get("orig_frame", 0) or 0) + int(self.start_offset)
            if target_frame == frame_idx and not self._is_hand_label(rb.get("label")):
                return True
        return False

    def _preferred_event_detection_ids(self, event: Optional[dict]) -> set:
        preferred_ids: set = set()
        if not isinstance(event, dict):
            return preferred_ids
        for actor in list(self.actors_config or []):
            hand_key = str((actor or {}).get("id") or "").strip()
            if not hand_key:
                continue
            hand_data = ((event.get("hoi_data", {}) or {}).get(hand_key) or {})
            noun_id = self._hand_noun_object_id(hand_data)
            if noun_id is None:
                noun_id = self._strong_semantic_noun_suggestion_id(hand_data)
            instrument_id = self._hand_instrument_object_id(hand_data)
            try:
                if noun_id is not None:
                    preferred_ids.add(int(noun_id))
            except Exception:
                pass
            try:
                if instrument_id is not None:
                    preferred_ids.add(int(instrument_id))
            except Exception:
                continue
        return preferred_ids

    def _preferred_click_detection_ids(self) -> Optional[set]:
        hand_data = self._selected_hand_data()
        if bool(getattr(self, "rad_draw_inst", None) and self.rad_draw_inst.isChecked()):
            preferred_id = self._hand_instrument_object_id(hand_data)
            if preferred_id is None:
                preferred_id = getattr(
                    getattr(self, "combo_instrument", None),
                    "currentData",
                    lambda: None,
                )()
        else:
            preferred_id = self._hand_noun_object_id(hand_data)
            if preferred_id is None and isinstance(hand_data, dict):
                preferred_id = self._strong_semantic_noun_suggestion_id(hand_data)
            if preferred_id is None:
                preferred_id = getattr(self.combo_target, "currentData", lambda: None)()
        try:
            return {int(preferred_id)} if preferred_id is not None else None
        except Exception:
            return None

    def _select_best_click_detection(
        self,
        detections: Sequence[dict],
        click_point: Optional[Sequence[float]],
        *,
        prefer_non_hand: bool = False,
        include_related_hands: bool = False,
    ) -> List[dict]:
        if not click_point:
            return list(detections or [])
        try:
            px = float(click_point[0])
            py = float(click_point[1])
        except Exception:
            return list(detections or [])

        ranked: List[Tuple[Tuple[float, float, float, float], dict, bool]] = []
        for det in list(detections or []):
            try:
                x1, y1, x2, y2 = [float(v) for v in list(det.get("xyxy") or [0, 0, 0, 0])]
            except Exception:
                continue
            margin = 18.0
            contains = (
                (x1 - margin) <= px <= (x2 + margin)
                and (y1 - margin) <= py <= (y2 + margin)
            )
            if not contains:
                continue
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            center_dist = math.hypot(cx - px, cy - py)
            area = max(1.0, (x2 - x1) * (y2 - y1))
            confidence = float(det.get("confidence") or 0.0)
            is_hand = bool(self._normalize_hand_label(det.get("class_name")))
            ranked.append(
                ((center_dist, area, -confidence, -max(0.0, x2 - x1)), det, is_hand)
            )
        if not ranked:
            return []
        ranked.sort(key=lambda item: item[0])
        if prefer_non_hand:
            object_ranked = [item for item in ranked if not item[2]]
            if object_ranked:
                selected = [dict(object_ranked[0][1])]
                if include_related_hands:
                    selected.extend(dict(item[1]) for item in ranked if item[2])
                return selected
        return [dict(ranked[0][1])]

    def _on_video_canvas_click(
        self,
        frame_idx: int,
        image_x: Optional[float] = None,
        image_y: Optional[float] = None,
    ) -> None:
        if self.selected_event_id is None or not str(getattr(self, "selected_hand_label", "") or "").strip():
            return
        if image_x is None or image_y is None:
            return
        clicked_box = self._find_clicked_object_box(frame_idx, image_x, image_y)
        if isinstance(clicked_box, dict):
            try:
                object_id = int(clicked_box.get("id"))
            except Exception:
                object_id = None
            if object_id is not None:
                self._select_object_item_in_list(object_id, best_bbox=clicked_box)
                self._on_object_selection()
                return
        if not self._detection_assist_enabled():
            return
        if not self.class_map or not self.global_object_map:
            return
        hand_data = self._selected_hand_data()
        preferred_ids = self._preferred_click_detection_ids()
        primary_field = self._inline_primary_field_name(hand_data)
        if preferred_ids is None and primary_field != "noun_object_id":
            return
        if self.yolo_model is None:
            self._load_yolo_model()
            if self.yolo_model is None:
                return
        if self._yolo_inference_busy():
            return

        frame_idx = int(frame_idx)
        frame_bgr = None
        if frame_idx == int(getattr(self.player, "current_frame", 0)):
            frame_bgr = self.player.get_current_frame_bgr()
        if frame_bgr is None:
            frame_bgr = self._read_video_frames_bgr([frame_idx]).get(frame_idx)
        if frame_bgr is None:
            return

        push_undo_for_yolo = True
        if not self._frame_has_raw_hand_label(frame_idx, self.selected_hand_label):
            detected_hands = int(
                self._detect_current_frame_hands(
                    frame_bgr,
                    frame_idx=frame_idx,
                    push_undo=True,
                    target_hand_keys=[self.selected_hand_label],
                )
                or 0
            )
            if detected_hands > 0:
                push_undo_for_yolo = False

        self._start_yolo_inference(
            [
                {
                    "frame_idx": frame_idx,
                    "frame_bgr": frame_bgr,
                    "include_ids": None if preferred_ids is None else list(preferred_ids),
                    "replace_existing": False,
                    "click_point": [float(image_x), float(image_y)],
                    "auto_select": True,
                }
            ],
            context={
                "mode": "click_target",
                "push_undo": bool(push_undo_for_yolo),
                "notify_empty": True,
                "notify_errors": True,
                "auto_select_object": True,
                "auto_assign_selected_hand": True,
                "log_event": "hoi_detect_yolo_click",
                "log_fields": {
                    "frame": frame_idx,
                    "click_x": round(float(image_x), 1),
                    "click_y": round(float(image_y), 1),
                    "event_id": int(self.selected_event_id),
                    "hand": str(self.selected_hand_label or ""),
                    "filtered": bool(preferred_ids),
                    "primary_field": primary_field,
                },
            },
        )

    def _yolo_inference_busy(self) -> bool:
        if self._yolo_infer_worker is None:
            return False
        QMessageBox.information(
            self,
            "YOLO Inference",
            "A YOLO inference job is already running. Wait for it to finish first.",
        )
        return True

    def _start_yolo_inference(self, requests, *, context: Optional[Dict[str, Any]] = None) -> bool:
        if self._yolo_inference_busy():
            return False
        request_rows = [dict(row or {}) for row in list(requests or []) if row]
        if not request_rows:
            return False
        self._yolo_infer_context = dict(context or {})
        self._yolo_infer_worker = YoloInferenceWorker(
            self.yolo_model,
            request_rows,
            self.yolo_conf,
            self.yolo_iou,
            class_map=self.class_map,
        )
        self._yolo_infer_worker.finished.connect(self._on_yolo_inference_finished)
        self._refresh_inference_action_states()
        self._log(
            "hoi_detect_yolo_async_start",
            mode=str(self._yolo_infer_context.get("mode") or ""),
            frames=len(request_rows),
        )
        self._yolo_infer_worker.start()
        return True

    def _apply_yolo_inference_results(self, results, context: Dict[str, Any]) -> None:
        category_to_default_id = self._build_category_to_default_id()
        if not category_to_default_id:
            QMessageBox.warning(
                self,
                "Error",
                "Cannot map detections without the noun/object list.",
            )
            return

        result_rows = [dict(row or {}) for row in list(results or [])]
        will_change = False
        for row in result_rows:
            frame_idx = int(row.get("frame_idx") or 0)
            detections = list(row.get("detections") or [])
            if detections:
                will_change = True
                break
            if not bool(row.get("replace_existing")):
                continue
            existing_unlocked = any(
                (rb.get("orig_frame", 0) + self.start_offset) == frame_idx
                and not self._is_hand_label(rb.get("label"))
                and not self._is_box_locked(rb)
                for rb in list(self.raw_boxes or [])
            )
            if existing_unlocked:
                will_change = True
                break

        if will_change and bool(context.get("push_undo", True)):
            self._push_undo()

        w = float(getattr(self.player, "_frame_w", 0) or 0)
        h = float(getattr(self.player, "_frame_h", 0) or 0)
        total_added = 0
        total_skipped = 0
        total_locked_preserved = 0
        total_replaced = 0
        total_hand_added = 0
        onset_projected = 0
        any_change = False
        auto_select_candidates: List[Tuple[int, dict]] = []
        click_target_mode = False

        for row in result_rows:
            frame_idx = int(row.get("frame_idx") or 0)
            include_ids_raw = row.get("include_ids")
            include_ids = None if include_ids_raw is None else set(include_ids_raw or [])
            replace_existing = bool(row.get("replace_existing"))
            click_point = row.get("click_point")
            if click_point:
                click_target_mode = True

            if replace_existing:
                kept = []
                for rb in list(self.raw_boxes or []):
                    target_frame = int(rb.get("orig_frame", 0) or 0) + int(self.start_offset)
                    if target_frame == frame_idx and not self._is_hand_label(rb.get("label")):
                        if self._is_box_locked(rb):
                            total_locked_preserved += 1
                            kept.append(rb)
                        else:
                            any_change = True
                            total_replaced += 1
                        continue
                    kept.append(rb)
                self.raw_boxes = kept

            new_boxes = []
            detections = list(row.get("detections") or [])
            if click_point:
                detections = self._select_best_click_detection(
                    detections,
                    click_point,
                    prefer_non_hand=True,
                    include_related_hands=True,
                )
            hand_candidates = []
            for det in detections:
                cls_id = int(det.get("class_id"))
                class_name = det.get("class_name")
                if not class_name:
                    total_skipped += 1
                    continue
                hand_label = self._normalize_hand_label(class_name)
                if hand_label:
                    x1, y1, x2, y2 = [
                        float(v) for v in list(det.get("xyxy") or [0, 0, 0, 0])
                    ]
                    x1 = max(0.0, min(w, x1))
                    y1 = max(0.0, min(h, y1))
                    x2 = max(0.0, min(w, x2))
                    y2 = max(0.0, min(h, y2))
                    hand_candidates.append(
                        {
                            "label": hand_label,
                            "source": "yolo_detect",
                            "confidence": det.get("confidence"),
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        }
                    )
                    continue
                norm_name = self._norm_category(class_name, for_detection=True)
                if not norm_name:
                    total_skipped += 1
                    continue
                uid = category_to_default_id.get(norm_name)
                result_label = str(class_name)
                if include_ids is not None:
                    matched_uid = None
                    for req_id in include_ids:
                        req_names = [
                            name
                            for name, val in self.global_object_map.items()
                            if val == req_id
                        ]
                        for req_name in req_names:
                            if self._norm_category(req_name) == norm_name:
                                matched_uid = req_id
                                result_label = str(req_name)
                                break
                        if matched_uid is not None:
                            break
                    uid = matched_uid
                if uid is None:
                    total_skipped += 1
                    continue
                if include_ids is not None and uid not in include_ids:
                    total_skipped += 1
                    continue
                x1, y1, x2, y2 = [float(v) for v in list(det.get("xyxy") or [0, 0, 0, 0])]
                x1 = max(0.0, min(w, x1))
                y1 = max(0.0, min(h, y1))
                x2 = max(0.0, min(w, x2))
                y2 = max(0.0, min(h, y2))
                new_boxes.append(
                    {
                        "id": uid,
                        "orig_frame": frame_idx - self.start_offset,
                        "label": result_label,
                        "class_id": cls_id,
                        "source": "yolo_detect",
                        "confidence": det.get("confidence"),
                        "locked": False,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )
            if hand_candidates:
                applied_hands = self._apply_hand_box_candidates(
                    frame_idx,
                    hand_candidates,
                    push_undo=False,
                    refresh_view=False,
                )
                if applied_hands > 0:
                    any_change = True
                    total_hand_added += int(applied_hands)
            if new_boxes:
                self.raw_boxes.extend(new_boxes)
                any_change = True
                total_added += len(new_boxes)
                if bool(row.get("auto_select")):
                    try:
                        auto_select_candidates.append(
                            (int(new_boxes[0].get("id")), dict(new_boxes[0]))
                        )
                    except Exception:
                        pass

        mode = str(context.get("mode") or "").strip()
        projection_event_ids: List[int] = []
        requested_event_id = (context.get("log_fields") or {}).get("event_id")
        try:
            requested_event_id = (
                int(requested_event_id) if requested_event_id is not None else None
            )
        except Exception:
            requested_event_id = None
        if mode == "action_keyframes" and requested_event_id is not None:
            projection_event_ids = [int(requested_event_id)]
        elif mode == "all_action_keyframes":
            for event in list(getattr(self, "events", []) or []):
                try:
                    projection_event_ids.append(int((event or {}).get("id")))
                except Exception:
                    continue

        if any_change:
            self._rebuild_bboxes_from_raw()
            self._bump_bbox_revision()
            self._bump_query_state_revision()

        if projection_event_ids:
            needs_projection = any(
                self._ensure_event_onset_grounding(event_id, dry_run=True) > 0
                for event_id in projection_event_ids
            )
            if (
                needs_projection
                and not any_change
                and bool(context.get("push_undo", True))
            ):
                self._push_undo()
            if needs_projection:
                for event_id in projection_event_ids:
                    onset_projected += self._ensure_event_onset_grounding(event_id)
                if onset_projected > 0:
                    any_change = True

        if any_change:
            self._refresh_boxes_for_frame(int(getattr(self.player, "current_frame", 0)))
            if mode == "action_keyframes" and self.selected_event_id is not None:
                if requested_event_id is None or requested_event_id == int(self.selected_event_id):
                    self._focus_visual_support_for_selected_event()
            if bool(context.get("auto_select_object")) and auto_select_candidates:
                object_id, best_bbox = auto_select_candidates[0]
                self._select_object_item_in_list(object_id, best_bbox=best_bbox)
                if bool(context.get("auto_assign_selected_hand")):
                    self._on_object_selection()

        log_event = str(context.get("log_event") or "hoi_detect_yolo")
        log_fields = dict(context.get("log_fields") or {})
        log_fields.update(
            {
                "added": total_added,
                "skipped": total_skipped,
                "locked_preserved": total_locked_preserved,
                "replaced": total_replaced,
                "hand_added": total_hand_added,
                "onset_projected": onset_projected,
                "frames": len(result_rows),
            }
        )
        self._log(log_event, **log_fields)

        notify_empty = bool(context.get("notify_empty"))
        if notify_empty and (not any_change or (click_target_mode and total_added <= 0)):
            if click_target_mode:
                QMessageBox.information(
                    self,
                    "Info",
                    "No object detection matched the clicked target point on this frame.",
                )
            else:
                QMessageBox.information(self, "Info", "No detections found.")

    def _on_yolo_inference_finished(self, payload) -> None:
        worker = self._yolo_infer_worker
        context = dict(self._yolo_infer_context or {})
        self._yolo_infer_worker = None
        self._yolo_infer_context = {}
        try:
            success = bool((payload or {}).get("success"))
            if not success:
                error = str((payload or {}).get("error") or "YOLO inference failed.")
                self._log(
                    "hoi_detect_yolo_async_failed",
                    mode=str(context.get("mode") or ""),
                    error=error,
                )
                if bool(context.get("notify_errors", True)):
                    QMessageBox.warning(self, "Error", f"YOLO inference failed:\n{error}")
                return
            self._apply_yolo_inference_results(
                (payload or {}).get("results") or [],
                context,
            )
            self._log(
                "hoi_detect_yolo_async_finished",
                mode=str(context.get("mode") or ""),
                frames=int((payload or {}).get("frames") or 0),
            )
        finally:
            self._refresh_inference_action_states()
            if worker is not None:
                worker.deleteLater()

    def _detect_current_frame_yolo(
        self,
        frame_idx=None,
        include_ids=None,
        override_policy=None,
        push_undo=True,
        frame_bgr=None,
    ):
        """Run YOLOv11 detection on one frame and optionally filter to selected object ids."""
        if not self.player.cap:
            QMessageBox.warning(self, "Missing prerequisite", "Please load a video first.")
            return
        if not self.class_map:
            QMessageBox.warning(
                self, "Missing prerequisite", "Please load a class map first (data.yaml)."
            )
            return
        if not self.global_object_map:
            QMessageBox.warning(
                self,
                "Missing prerequisite",
                "Please import the noun/object list first.",
            )
            return
        if self.yolo_model is None:
            self._load_yolo_model()
            if self.yolo_model is None:
                return

        if frame_idx is None:
            frame_idx = int(self.player.current_frame)
        else:
            frame_idx = int(frame_idx)
        if frame_bgr is None:
            if frame_idx == int(self.player.current_frame):
                frame_bgr = self.player.get_current_frame_bgr()
            else:
                frame_bgr = self._read_video_frames_bgr([frame_idx]).get(frame_idx)
        if frame_bgr is None:
            QMessageBox.warning(self, "Error", "No video frame available.")
            return

        existing = [
            b
            for b in self.bboxes.get(frame_idx, [])
            if not self._is_hand_label(b.get("label"))
        ]
        replace_existing = False
        if existing:
            policy = override_policy or self.yolo_existing_policy or self._ask_existing_policy()
            if not policy:
                return
            replace_existing = policy == "replace"

        self._start_yolo_inference(
            [
                {
                    "frame_idx": int(frame_idx),
                    "frame_bgr": frame_bgr,
                    "include_ids": None if include_ids is None else list(include_ids or []),
                    "replace_existing": bool(replace_existing),
                }
            ],
            context={
                "mode": "single_frame",
                "push_undo": bool(push_undo),
                "notify_empty": True,
                "notify_errors": True,
                "log_event": "hoi_detect_yolo",
                "log_fields": {
                    "frame": int(frame_idx),
                },
            },
        )

    def _ensure_mp_hands(self):
        """Initialize MediaPipe hand detection with solutions->tasks fallback."""
        if self.mp_hands_error:
            return None
        if self.mp_hands is not None:
            return self.mp_hands
        try:
            import mediapipe as mp
        except Exception as ex:
            self.mp_hands_error = str(ex)
            QMessageBox.warning(
                self, "Missing package", f"MediaPipe is not available:\n{ex}"
            )
            return None
        try:
            solutions_hands = getattr(getattr(mp, "solutions", None), "hands", None)
            if solutions_hands is not None:
                detector = solutions_hands.Hands(
                    static_image_mode=True,
                    max_num_hands=self.mp_hands_max,
                    min_detection_confidence=self.mp_hands_conf,
                    min_tracking_confidence=self.mp_hands_track_conf,
                )
                self.mp_hands = {
                    "backend": "solutions",
                    "detector": detector,
                    "mp": mp,
                }
                self.mp_hands_backend = "solutions"
                return self.mp_hands
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_path = _mediapipe_hand_landmarker_model_file(self.video_path)
            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self.mp_hands_max,
                min_hand_detection_confidence=self.mp_hands_conf,
                min_tracking_confidence=self.mp_hands_track_conf,
            )
            detector = vision.HandLandmarker.create_from_options(options)
            self.mp_hands = {
                "backend": "tasks",
                "detector": detector,
                "mp": mp,
                "model_path": model_path,
            }
            self.mp_hands_backend = "tasks"
            return self.mp_hands
        except Exception as ex:
            self.mp_hands_error = str(ex)
            QMessageBox.warning(
                self,
                "MediaPipe Error",
                "Failed to initialize MediaPipe hand detection.\n\n"
                f"{ex}",
            )
            return None

    def _precompute_persistent_hand_tracks(self):
        if not self.player.cap or not self.video_path:
            QMessageBox.warning(
                self,
                "Missing prerequisite",
                "Please load a video first.",
            )
            return
        if self._handtrack_status.get("building"):
            QMessageBox.information(
                self,
                "Hand Tracks",
                "Persistent hand-track precomputation is already running.",
            )
            return
        self._start_handtrack_precompute(self.video_path, force=True)

    def _extract_mp_solution_hand_detections(self, results, width: int, height: int) -> List[Dict[str, Any]]:
        if not results or not getattr(results, "multi_hand_landmarks", None):
            return []
        handedness: List[Dict[str, Any]] = []
        if getattr(results, "multi_handedness", None):
            for hd in list(results.multi_handedness or []):
                label = ""
                score = 0.0
                try:
                    label = str(hd.classification[0].label or "").strip().lower()
                except Exception:
                    label = ""
                try:
                    score = float(hd.classification[0].score or 0.0)
                except Exception:
                    score = 0.0
                handedness.append({"label": label, "score": score})
        while len(handedness) < len(list(results.multi_hand_landmarks or [])):
            handedness.append({"label": "", "score": 0.0})
        detections: List[Dict[str, Any]] = []
        for idx, hand_lms in enumerate(list(results.multi_hand_landmarks or [])):
            xs = [float(lm.x) for lm in list(hand_lms.landmark or [])]
            ys = [float(lm.y) for lm in list(hand_lms.landmark or [])]
            if not xs or not ys:
                continue
            x1 = max(0.0, min(float(width), min(xs) * float(width)))
            y1 = max(0.0, min(float(height), min(ys) * float(height)))
            x2 = max(0.0, min(float(width), max(xs) * float(width)))
            y2 = max(0.0, min(float(height), max(ys) * float(height)))
            hand_info = dict(handedness[idx] or {}) if idx < len(handedness) else {}
            detections.append(
                {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "cx": float(0.5 * (x1 + x2)),
                    "cy": float(0.5 * (y1 + y2)),
                    "area": float(max(1.0, max(0.0, x2 - x1) * max(0.0, y2 - y1))),
                    "handedness": _mediapipe_anatomical_label(hand_info.get("label")),
                    "handedness_score": float(hand_info.get("score") or 0.0),
                    "detection_confidence": float(hand_info.get("score") or 0.0),
                }
            )
        return detections

    def _detect_current_frame_hands(
        self,
        frame_bgr,
        frame_idx=None,
        push_undo=True,
        target_hand_keys: Optional[Sequence[str]] = None,
        refresh_view: bool = True,
    ):
        """Run MediaPipe Hands on one frame and replace hand boxes for that frame."""
        mp_bundle = self._ensure_mp_hands()
        if mp_bundle is None:
            return 0
        if frame_bgr is None:
            return 0
        target_hands: set = set()
        for value in list(target_hand_keys or []):
            norm_hand = self._normalize_hand_label(value)
            if norm_hand:
                target_hands.add(norm_hand)
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        backend = str((mp_bundle or {}).get("backend") or "solutions").strip().lower()
        detector = (mp_bundle or {}).get("detector")
        mp = (mp_bundle or {}).get("mp")
        try:
            if backend == "tasks":
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = detector.detect(mp_image)
                detections = _extract_hand_detections_tasks(results, w, h)
            else:
                results = detector.process(rgb)
                detections = self._extract_mp_solution_hand_detections(results, w, h)
        except Exception as ex:
            if not self.mp_hands_error:
                self.mp_hands_error = str(ex)
                QMessageBox.warning(
                    self, "MediaPipe Error", f"Hand detection failed:\n{ex}"
                )
            return 0
        if not detections:
            if refresh_view:
                self._update_status_label()
            return 0

        hand_candidates = []
        used = set()
        candidate_source = (
            "mediapipe_hands" if backend == "solutions" else "mediapipe_tasks_hands"
        )
        for det in list(detections or []):
            x1 = float(det.get("x1", 0.0) or 0.0)
            y1 = float(det.get("y1", 0.0) or 0.0)
            x2 = float(det.get("x2", 0.0) or 0.0)
            y2 = float(det.get("y2", 0.0) or 0.0)
            a1 = self.actors_config[0]["id"]
            a2 = self.actors_config[1]["id"] if len(self.actors_config) > 1 else a1

            label = None
            h_low = str(det.get("handedness") or "").strip().lower()
            if h_low.startswith("left"):
                label = self._normalize_hand_label("left")
            elif h_low.startswith("right"):
                label = self._normalize_hand_label("right")
            if label is None or label in used:
                continue
            if self.mp_hands_swap:
                label = a2 if label == a1 else a1
            if label in used:
                other = a2 if label == a1 else a1
                if other not in used:
                    label = other
                else:
                    continue
            if label in used and all(a["id"] in used for a in self.actors_config[:2]):
                continue
            norm_label = self._normalize_hand_label(label)
            if target_hands and norm_label not in target_hands:
                continue
            if norm_label:
                label = norm_label
            used.add(label)
            confidence = self._box_confidence_value(det)
            hand_candidates.append(
                {
                    "label": label,
                    "source": candidate_source,
                    "confidence": confidence,
                    "detection_confidence": confidence,
                    "handedness_score": self._box_confidence_value(
                        {"handedness_score": det.get("handedness_score")}
                    ),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

        if not hand_candidates:
            return 0

        if frame_idx is None:
            frame_idx = int(self.player.current_frame)
        applied_count = self._apply_hand_box_candidates(
            frame_idx,
            hand_candidates,
            push_undo=push_undo,
            refresh_view=refresh_view,
        )
        self._log(
            "hoi_detect_hands",
            frame=frame_idx,
            count=applied_count,
            backend=backend,
        )
        return applied_count

    def _detect_selected_action(self):
        if not self._guard_experiment_mode("detection"):
            return
        if self.selected_event_id is None:
            QMessageBox.information(
                self,
                "Object Detection",
                "Select an action first, then use Detect Action.",
            )
            return
        self._detect_action_active_items(self.selected_event_id)

    def _detect_action_active_items(self, event_id: int):
        if not self._guard_experiment_mode("detection"):
            return
        if not self.player.cap:
            QMessageBox.warning(self, "Missing prerequisite", "Please load a video first.")
            return
        if self._yolo_inference_busy():
            return
        ev = self._find_event_by_id(event_id)
        if not ev:
            return
        keyframes = set()
        frame_to_hand_keys: Dict[int, set] = {}
        active_ids = set()
        for actor in self.actors_config:
            hand_key = actor["id"]
            hand_data = ev.get("hoi_data", {}).get(hand_key, {}) or {}
            for key in (
                "interaction_start",
                "functional_contact_onset",
                "interaction_end",
            ):
                val = hand_data.get(key)
                if val is not None:
                    try:
                        frame_idx = int(val)
                    except Exception:
                        continue
                    keyframes.add(frame_idx)
                    frame_to_hand_keys.setdefault(frame_idx, set()).add(hand_key)
            target_id = self._hand_noun_object_id(hand_data)
            try:
                if target_id is not None:
                    active_ids.add(int(target_id))
            except Exception:
                pass
            instrument_id = self._hand_instrument_object_id(hand_data)
            try:
                if instrument_id is not None:
                    active_ids.add(int(instrument_id))
            except Exception:
                pass

        if not keyframes:
            QMessageBox.warning(
                self,
                "Object Detection",
                "The selected action does not have start, onset, or end frames yet.",
            )
            return

        frame_map = self._read_video_frames_bgr(sorted(keyframes))
        if not frame_map:
            QMessageBox.warning(
                self,
                "Object Detection",
                "Could not read the selected action keyframes from the current video.",
            )
            return
        hand_auto_detected = self._ensure_hand_boxes_for_detection_frames(
            frame_map,
            frame_to_hand_keys,
        )

        include_ids = set(active_ids)
        if not include_ids:
            include_ids = set(self._preferred_event_detection_ids(ev) or [])
        if not self.class_map:
            QMessageBox.warning(
                self, "Missing prerequisite", "Please load a class map first (data.yaml)."
            )
            return
        if not self.global_object_map:
            QMessageBox.warning(
                self,
                "Missing prerequisite",
                "Please import the noun/object list first.",
            )
            return
        if self.yolo_model is None:
            self._load_yolo_model()
            if self.yolo_model is None:
                return
        if not include_ids:
            reply = QMessageBox.question(
                self,
                "Object Detection",
                "This action has no assigned noun object.\nDetect all object categories on its keyframes instead?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                return
            include_ids = None

        batch_policy = "append"
        for fidx in keyframes:
            existing = [
                b
                for b in self.bboxes.get(int(fidx), [])
                if not self._is_hand_label(b.get("label"))
            ]
            if existing:
                batch_policy = self.yolo_existing_policy or self._ask_existing_policy()
                if not batch_policy:
                    return
                break

        requests = []
        for fidx in sorted(keyframes):
            frame_bgr = frame_map.get(int(fidx))
            if frame_bgr is None:
                continue
            requests.append(
                {
                    "frame_idx": int(fidx),
                    "frame_bgr": frame_bgr,
                    "include_ids": None if include_ids is None else list(include_ids or []),
                    "replace_existing": batch_policy == "replace",
                }
            )
        if not requests:
            QMessageBox.warning(
                self,
                "Object Detection",
                "Could not read the selected action keyframes from the current video.",
            )
            return
        self._start_yolo_inference(
            requests,
            context={
                "mode": "action_keyframes",
                "push_undo": not bool(hand_auto_detected),
                "notify_empty": False,
                "notify_errors": True,
                "log_event": "hoi_detect_action_keyframes",
                "log_fields": {
                    "event_id": event_id,
                    "filtered": bool(include_ids),
                    "hand_auto_detected": int(hand_auto_detected),
                },
            },
        )

    def _detect_all_actions(self):
        if not self._guard_experiment_mode("detection"):
            return
        if not self.events:
            QMessageBox.information(self, "Object Detection", "No actions are available.")
            return
        if not self.player.cap:
            QMessageBox.warning(self, "Missing prerequisite", "Please load a video first.")
            return
        if not self.class_map:
            QMessageBox.warning(
                self, "Missing prerequisite", "Please load a class map first (data.yaml)."
            )
            return
        if not self.global_object_map:
            QMessageBox.warning(
                self,
                "Missing prerequisite",
                "Please import the noun/object list first.",
            )
            return
        if self.yolo_model is None:
            self._load_yolo_model()
            if self.yolo_model is None:
                return
        if self._yolo_inference_busy():
            return

        frame_to_active_ids = {}
        frame_to_hand_keys: Dict[int, set] = {}
        for ev in self.events:
            for actor in self.actors_config:
                hand_key = actor["id"]
                hand_data = ev.get("hoi_data", {}).get(hand_key, {}) or {}
                keyframes = set()
                for key in (
                    "interaction_start",
                    "functional_contact_onset",
                    "interaction_end",
                ):
                    val = hand_data.get(key)
                    if val is not None:
                        try:
                            keyframes.add(int(val))
                        except Exception:
                            continue
                if not keyframes:
                    continue
                active_ids = set()
                target_id = self._hand_noun_object_id(hand_data)
                try:
                    if target_id is not None:
                        active_ids.add(int(target_id))
                except Exception:
                    pass
                instrument_id = self._hand_instrument_object_id(hand_data)
                try:
                    if instrument_id is not None:
                        active_ids.add(int(instrument_id))
                except Exception:
                    pass
                for fidx in keyframes:
                    frame_to_hand_keys.setdefault(int(fidx), set()).add(hand_key)
                    frame_to_active_ids.setdefault(int(fidx), set()).update(active_ids)

        if not frame_to_active_ids:
            QMessageBox.information(
                self,
                "Object Detection",
                "No action keyframes with assigned noun objects were found.",
            )
            return

        batch_policy = "append"
        for fidx in frame_to_active_ids.keys():
            existing = [
                b
                for b in self.bboxes.get(int(fidx), [])
                if not self._is_hand_label(b.get("label"))
            ]
            if existing:
                batch_policy = self.yolo_existing_policy or self._ask_existing_policy()
                if not batch_policy:
                    return
                break

        frame_map = self._read_video_frames_bgr(sorted(frame_to_active_ids.keys()))
        requests = []
        for fidx, active_ids in sorted(frame_to_active_ids.items()):
            frame_bgr = frame_map.get(int(fidx))
            if frame_bgr is None:
                continue
            requests.append(
                {
                    "frame_idx": int(fidx),
                    "frame_bgr": frame_bgr,
                    "include_ids": list(active_ids or []) if active_ids else None,
                    "replace_existing": batch_policy == "replace",
                }
            )
        if not requests:
            QMessageBox.warning(
                self,
                "Object Detection",
                "Could not read any action keyframes from the current video.",
            )
            return
        hand_auto_detected = self._ensure_hand_boxes_for_detection_frames(
            frame_map,
            frame_to_hand_keys,
        )
        self._start_yolo_inference(
            requests,
            context={
                "mode": "all_action_keyframes",
                "push_undo": not bool(hand_auto_detected),
                "notify_empty": False,
                "notify_errors": True,
                "log_event": "hoi_detect_all_action_keyframes",
                "log_fields": {
                    "hand_auto_detected": int(hand_auto_detected),
                },
            },
        )

    def _incremental_train_yolo(self):
        """Main entry point for YOLO fine-tuning on current annotations."""
        if not self._guard_experiment_mode("detection"):
            return
        if not self.yolo_weights_path or not self.video_path:
            QMessageBox.warning(self, "Error", "Please load both a video and a YOLO model first.")
            return

        if not self.class_map:
             QMessageBox.warning(self, "Error", "Please load a class map (data.yaml) first.")
             return

        # 1. Ask for config
        dlg = YoloTrainDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        config = dlg.get_config()

        # 2. Prepare dataset
        try:
            yaml_path = self._prepare_yolo_dataset(config)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to prepare dataset:\n{e}")
            return

        # 3. Start training
        self._log("hoi_yolo_train_start", weights=self.yolo_weights_path, config=config)
        self.train_worker = YoloTrainWorker(self.yolo_weights_path, yaml_path, config)
        self.train_worker.progress.connect(self._on_train_progress)
        self.train_worker.finished.connect(self._on_train_finished)
        
        self.train_progress_dlg = QProgressDialog("Initializing training...", "Cancel", 0, 0, self)
        self.train_progress_dlg.setWindowTitle("YOLO Training")
        self.train_progress_dlg.setWindowModality(Qt.WindowModal)
        self.train_progress_dlg.setMinimumDuration(0)
        self.train_progress_dlg.canceled.connect(self.train_worker.terminate) # Simple kill if canceled
        self.train_progress_dlg.show()
        
        self.train_worker.start()

    def _prepare_yolo_dataset(self, config):
        """Harvest frames and boxes from raw_boxes to create a YOLO dataset."""
        import shutil
        import random
        
        base_dir = config["output_dir"]
        os.makedirs(base_dir, exist_ok=True)
        
        # YOLO structure
        img_train = os.path.join(base_dir, "images", "train")
        img_val = os.path.join(base_dir, "images", "val")
        lbl_train = os.path.join(base_dir, "labels", "train")
        lbl_val = os.path.join(base_dir, "labels", "val")
        
        for d in [img_train, img_val, lbl_train, lbl_val]:
            if os.path.exists(d):
                try:
                    shutil.rmtree(d)
                except Exception:
                    pass
            os.makedirs(d, exist_ok=True)

        # Reverse class map: normalized_name -> cid
        name_to_cid = {self._norm_category(v): k for k, v in self.class_map.items()}
        
        # Collect frames to export (matching annotated objects with class_map)
        frame_to_boxes = {}
        for b in self.raw_boxes:
            if self._is_hand_label(b.get("label")):
                continue
            
            lbl = self._norm_category(b.get("label", ""))
            if lbl not in name_to_cid:
                continue
            
            f = b.get("orig_frame")
            if f not in frame_to_boxes:
                frame_to_boxes[f] = []
            frame_to_boxes[f].append(b)
        
        if not frame_to_boxes:
            raise Exception("No annotated objects found in your library that match classes in data.yaml.\n"
                            "Check if names match exactly (ignoring case/underscores).")

        # Export frames
        cap = cv2.VideoCapture(self.video_path)
        frame_indices = list(frame_to_boxes.keys())
        random.shuffle(frame_indices)
        
        split_idx = int(len(frame_indices) * config["train_split"])
        train_frames = frame_indices[:split_idx]
        
        for fidx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret: continue
            
            h, w = frame.shape[:2]
            subset = "train" if fidx in train_frames else "val"
            
            img_filename = f"frame_{fidx}.jpg"
            img_path = os.path.join(base_dir, "images", subset, img_filename)
            lbl_path = os.path.join(base_dir, "labels", subset, f"frame_{fidx}.txt")
            
            cv2.imwrite(img_path, frame)
            
            with open(lbl_path, "w") as f_lbl:
                for b in frame_to_boxes[fidx]:
                    lbl = self._norm_category(b.get("label", ""))
                    cid = name_to_cid[lbl]
                    # Normalized YOLO: xc yc w h
                    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
                    bw = (bx2 - bx1) / w
                    bh = (by2 - by1) / h
                    bxc = (bx1 + bx2) / (2 * w)
                    byc = (by1 + by2) / (2 * h)
                    f_lbl.write(f"{cid} {bxc:.6f} {byc:.6f} {bw:.6f} {bh:.6f}\n")
        
        cap.release()
        
        # Generate dataset.yaml
        yaml_content = {
            "path": os.path.abspath(base_dir),
            "train": "images/train",
            "val": "images/val",
            "names": {int(k): v for k, v in self.class_map.items()}
        }
        yaml_path = os.path.join(base_dir, "dataset.yaml")
        with open(yaml_path, "w") as f_yaml:
            yaml.dump(yaml_content, f_yaml, sort_keys=False)
            
        return yaml_path

    def _on_train_progress(self, msg):
        if hasattr(self, "train_progress_dlg") and self.train_progress_dlg:
            self.train_progress_dlg.setLabelText(msg)

    def _on_train_finished(self, success, message):
        if hasattr(self, "train_progress_dlg") and self.train_progress_dlg:
            self.train_progress_dlg.close()
            self.train_progress_dlg.deleteLater()
            self.train_progress_dlg = None
        
        # Give a small buffer for thread to fully unwind
        QTimer.singleShot(100, lambda: self._show_train_result(success, message))

    def _show_train_result(self, success, message):
        if success:
            self._log("hoi_yolo_train_finished", success=True)
            QMessageBox.information(self, "Training Finished", message)
        else:
            self._log("hoi_yolo_train_finished", success=False, error=message)
            QMessageBox.critical(self, "Training Error", f"Training failed:\n{message}")

    def _confirm_close_request(self, prompt_parent=None) -> bool:
        if bool(getattr(self, "_close_request_approved", False)):
            return True
        if (
            getattr(self, '_correction_process', None) is not None
            or self._yolo_infer_worker is not None
            or self._videomae_infer_worker is not None
            or self._videomae_batch_progress is not None
            or self._handtrack_worker is not None
        ):
            QMessageBox.information(
                prompt_parent or self,
                "Inference Running",
                "Wait for the current background job to finish before closing the window.",
            )
            return False
        ok_incomplete, _ = self._check_incomplete_hoi(context="close")
        if not ok_incomplete:
            return False
        if self._hoi_has_unsaved_changes():
            reply = QMessageBox.question(
                prompt_parent or self,
                "Unsaved Changes",
                (
                    "Current HOI annotations have unsaved changes.\n\n"
                    "Save before closing?"
                ),
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if reply == QMessageBox.Cancel:
                return False
            if reply == QMessageBox.Save:
                self._save_annotations_json()
                if self._hoi_has_unsaved_changes():
                    return False
        self._close_request_approved = True
        return True

    def _finalize_close_request(self) -> None:
        if bool(getattr(self, "_close_request_finalized", False)):
            return
        self._close_request_finalized = True
        self._flush_live_operation_logs(warn_user=False)
        try:
            self._oplog_flush_timer.stop()
        except Exception:
            pass
        if callable(self._on_close):
            try:
                self._on_close()
            except Exception:
                pass

    def _detect_current_frame_combined(self):
        """Run YOLO object detection and MediaPipe hand detection on the current frame."""
        if not self._guard_experiment_mode("detection"):
            return
        if not self.player.cap:
            QMessageBox.warning(self, "Missing prerequisite", "Please load a video first.")
            return
        frame_bgr = self.player.get_current_frame_bgr()
        if frame_bgr is None:
            QMessageBox.warning(self, "Error", "No video frame available.")
            return

        # Objects (YOLO)
        if self.class_map and self.global_object_map:
            self._detect_current_frame_yolo(frame_bgr=frame_bgr)
        else:
            QMessageBox.information(
                self,
                "Info",
                "Object detection skipped. Load a class map and noun/object list first.",
            )

        # Hands (MediaPipe)
        self._detect_current_frame_hands(frame_bgr)

    def _save_annotations_json(self):
        """Save HOI annotations with optional integrity checks."""
        try:
            # persist any in-progress edits before validation/export
            self._sync_live_edits_for_export()

            # 1. Basic check
            if not self.events and not self.raw_boxes:
                QMessageBox.information(self, "Info", "No HOI data to save.")
                return

            if not self._require_participant_code_for_study_save():
                self._log("hoi_save_annotations_cancelled", reason="missing_participant_code")
                return

            # 2. Generate filename
            default_name = getattr(self, "_task_resume_path", "") or f"{self._default_annotation_basename()}.json"
            fp, _ = QFileDialog.getSaveFileName(
                self,
                "Save HOI Annotations",
                default_name,
                "JSON Files (*.json);;All Files (*)",
            )
            if not fp:
                return

            from pathlib import Path
            if getattr(self, '_task_resume_path', ''):
                expected = Path(self._task_resume_path).resolve()
                if Path(fp).resolve() != expected:
                    QMessageBox.warning(self, 'Save in task folder',
                        'Save this task as reviewed.json in its video folder so reopening restores your work.')
                    return
            elif Path(fp).name in ('annotations.json', 'review.json'):
                QMessageBox.warning(self, 'Preserve source annotations',
                    'Choose reviewed.json or another output name. Source annotations are kept unchanged.')
                return

            ok_incomplete, _ = self._check_incomplete_hoi(context="save")
            if not ok_incomplete:
                self._log("hoi_save_annotations_cancelled", reason="incomplete_check")
                return

            # 4. Build payload
            payload = self._build_payload_v2()
            incomplete_issue_count = len(self._collect_incomplete_hoi())
            is_valid_save = incomplete_issue_count == 0

            # 5. Write file
            self._finish_autosave()
            atomic_json(fp, payload)

            try:
                graph = build_hoi_event_graph(
                    self.events,
                    video_path=self.video_path,
                    annotation_path=fp,
                    actors_config=self.actors_config,
                    assembly_timeline=self._assembly_data(),
                )
                graph_path = save_event_graph_sidecar(fp, graph)
                self._log(
                    "hoi_save_event_graph",
                    path=graph_path,
                    anchors=len(graph.get("onset_anchors", []) or []),
                    locked_regions=len(graph.get("locked_regions", []) or []),
                    consistency_issues=len(graph.get("consistency_flags", []) or []),
                    sparse_evidence_expected=(
                        (graph.get("stats", {}) or {}).get(
                            "sparse_evidence_expected_count", 0
                        )
                    ),
                    sparse_evidence_missing=(
                        (graph.get("stats", {}) or {}).get(
                            "sparse_evidence_missing_count", 0
                        )
                    ),
                    field_state_counts=(graph.get("stats", {}) or {}).get(
                        "field_state_counts", {}
                    ),
                    field_source_counts=(graph.get("stats", {}) or {}).get(
                        "field_source_counts", {}
                    ),
                )
            except Exception as graph_ex:
                self._log(
                    "hoi_save_event_graph_failed",
                    path=fp,
                    error=str(graph_ex),
                )

            self._log(
                "hoi_save_annotations",
                path=fp,
                events=len(self.events),
                boxes=len(self.raw_boxes),
                is_valid_save=bool(is_valid_save),
                incomplete_issue_count=int(incomplete_issue_count),
            )
            log_base = os.path.splitext(fp)[0]
            if log_base:
                self._flush_ops_log_safely(
                    getattr(self, "op_logger", None),
                    log_base + ".ops.log.csv",
                    context="HOI save",
                )
            if (
                log_base
                and getattr(self, "validation_op_logger", None)
                and self.validation_op_logger.enabled
                and self.validation_session_active
            ):
                self._flush_ops_log_safely(
                    self.validation_op_logger,
                    log_base + ".validation.ops.log.csv",
                    context="HOI validation save",
                )
            self._save_validation_summary(fp, payload)
            self.current_annotation_path = fp
            self._ensure_semantic_adapter_loaded()
            self._mark_query_calibration_dirty()
            self._mark_hoi_saved()
            if getattr(self, '_autosave_path', None):
                self._autosave_baseline = self._autosave_snapshot()
                for recovery in (self._autosave_path, self._autosave_path.with_name(self._autosave_path.name + '.bak')):
                    recovery.unlink(missing_ok=True)
            QMessageBox.information(
                self, "Saved", f"Successfully saved to:\n{os.path.basename(fp)}"
            )

        except Exception as ex:
            self._log(
                "hoi_save_annotations_failed",
                path=locals().get("fp", ""),
                error=str(ex),
            )
            import traceback

            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Save failed:\n{str(ex)}")

    def _collect_incomplete_hoi(self) -> List[dict]:
        from core.completion_review import collect_review_issues
        return self._frame_review_issues() + self._policy_review_issues() + collect_review_issues(
            self.events, self.actors_config,
            lambda data, hand: self._hand_completion_state(data, hand_key=hand),
            object_ids=set(self.global_object_map.values()),
            frame_count=getattr(self.player, "frame_count", None),
        )

    def _check_incomplete_hoi(self, context: str = "save") -> tuple:
        issues = self._collect_incomplete_hoi()
        if not issues:
            self._log(
                "hoi_incomplete_check",
                context=context,
                has_issues=False,
                proceed=True,
                count=0,
            )
            return True, False
        preview = []
        for entry in issues[:4]:
            hand_short = self._get_actor_short_label(entry["hand"])
            missing = ", ".join(entry["missing"])
            preview.append(f"Event {entry['event_id']} {hand_short}: {missing}")
        if len(issues) > 4:
            preview.append(f"... and {len(issues) - 4} more")
        msg = (
            "Incomplete HOI events detected.\n"
            + "Review missing/invalid fields, object references, unconfirmed suggestions or keyframe boxes.\n\n"
            + "\n".join(preview)
            + "\n\nContinue anyway?"
        )
        reply = QMessageBox.question(
            self, "Incomplete annotations", msg, QMessageBox.Yes | QMessageBox.No
        )
        proceed = reply == QMessageBox.Yes
        self._log(
            "hoi_incomplete_check",
            context=context,
            has_issues=True,
            proceed=proceed,
            count=len(issues),
            issue_preview=preview,
        )
        return proceed, True

    def _update_incomplete_indicator(self):
        issues = self._collect_incomplete_hoi()
        self._incomplete_issues = issues
        if not issues:
            self.lbl_incomplete.setText("Incomplete: none")
            self.lbl_incomplete.setToolTip("No field or box issues found. Semantic and anomaly review is still required.")
            self._set_status_chip(getattr(self, "lbl_incomplete_chip", None), "Incomplete 0", "ok")
            if getattr(self, "lbl_review_status", None):
                self.lbl_review_status.setText("No field or box issues found. Semantic and anomaly review is still required.")
                self.lbl_review_status.setToolTip("No field or box issues found. Semantic and anomaly review is still required.")
            self._set_status_card_tone(
                getattr(self, "review_status_card", None),
                "active" if bool(getattr(self, "validation_enabled", False)) else "ok",
            )
            self.btn_incomplete_prev.setEnabled(False)
            self.btn_incomplete_next.setEnabled(False)
            if getattr(self, "btn_review_prev", None):
                self.btn_review_prev.setEnabled(False)
            if getattr(self, "btn_review_next", None):
                self.btn_review_next.setEnabled(False)
            self._incomplete_idx = -1
            self._update_inspector_tab_labels()
            return
        self._incomplete_idx = min(
            self._incomplete_idx if self._incomplete_idx >= 0 else -1, len(issues) - 1
        )
        self.lbl_incomplete.setText(f"Incomplete: {len(issues)}")
        tooltip = []
        for entry in issues[:6]:
            hand = self._get_actor_short_label(entry["hand"])
            missing = ", ".join(entry["missing"])
            tooltip.append(f"Event {entry['event_id']} {hand}: {missing}")
        if len(issues) > 6:
            tooltip.append("...")
        tooltip_text = "\n".join(tooltip)
        self.lbl_incomplete.setToolTip(tooltip_text)
        self._set_status_chip(getattr(self, "lbl_incomplete_chip", None), f"Incomplete {len(issues)}", "warn")
        if getattr(self, "lbl_review_status", None):
            self.lbl_review_status.setText(tooltip_text)
            self.lbl_review_status.setToolTip(tooltip_text)
        self._set_status_card_tone(getattr(self, "review_status_card", None), "warn")
        self.btn_incomplete_prev.setEnabled(True)
        self.btn_incomplete_next.setEnabled(True)
        if getattr(self, "btn_review_prev", None):
            self.btn_review_prev.setEnabled(True)
        if getattr(self, "btn_review_next", None):
            self.btn_review_next.setEnabled(True)
        self._update_inspector_tab_labels()

    def _jump_first_incomplete(self):
        self._incomplete_last_key = None
        self._incomplete_last_order = None
        self._jump_incomplete(1)

    def _jump_incomplete(self, direction: int):
        from core.completion_review import issue_key, issue_order, next_issue_index
        # Flush the active editor before rebuilding the queue, so fixes disappear.
        if self.selected_event_id is not None and self.selected_hand_label:
            self._save_ui_to_hand_draft(self.selected_hand_label)
            self._apply_draft_to_selected_event()
        self._update_incomplete_indicator()
        if not self._incomplete_issues:
            return
        index = next_issue_index(
            self._incomplete_issues, getattr(self, "_incomplete_last_key", None),
            getattr(self, "_incomplete_last_order", None), direction)
        issue = self._incomplete_issues[index]
        self._incomplete_idx = index
        self._incomplete_last_key = issue_key(issue)
        self._incomplete_last_order = issue_order(issue)
        frame = int(issue.get("frame") or 0)
        event_id, hand = issue.get("event_id"), issue.get("hand")
        # Merely visiting an issue must not trigger semantic autofill.
        previous_guard = getattr(self, "_review_navigation_active", False)
        self._review_navigation_active = True
        try:
            if event_id is not None and self._is_hand_label(hand):
                self._set_selected_event(event_id, hand)
        finally:
            self._review_navigation_active = previous_guard
        if self.player.cap:
            self.player.seek(frame)
        self._refresh_boxes_for_frame(frame)
        self._set_frame_controls(frame)
        self._focus_inspector_tab("review")
        self._focus_inline_editor_field(issue.get("field") or "verb")
        if getattr(self, "lbl_review_status", None):
            self.lbl_review_status.setText(
                f"Issue {index + 1}/{len(self._incomplete_issues)} | Event {event_id} | "
                f"{hand} | Frame {frame}: " + "; ".join(issue["missing"]))
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.set_current_frame(frame)
            self.hoi_timeline.refresh()
        self._log("hoi_incomplete_jump", direction=direction, target=frame,
                  event_id=event_id, hand=hand, field=issue.get("field"),
                  reason=issue.get("code"))

    def _save_validation_summary(self, annotations_path: str, payload: dict):
        if not bool(getattr(self, "validation_summary_enabled", True)):
            return
        if not self.validation_session_active:
            return
        graph_stats = {}
        try:
            graph_stats = dict(
                (
                    build_hoi_event_graph(
                        self.events,
                        video_path=self.video_path,
                        annotation_path=annotations_path,
                        actors_config=self.actors_config,
                        assembly_timeline=self._assembly_data(),
                    ).get("stats", {})
                    or {}
                )
            )
        except Exception:
            graph_stats = {}
        summary = {
            "version": "HOI_VALIDATION_V1",
            "session_id": (
                getattr(getattr(self, "validation_op_logger", None), "session_id", "")
                or getattr(getattr(self, "op_logger", None), "session_id", "")
                or ""
            ),
            "editor": self.validator_name,
            "assist_mode": self._experiment_mode_key(),
            "validation_started_at": self.validation_started_at,
            "validation_saved_at": datetime.now().isoformat(timespec="seconds"),
            "video_path": self.video_path,
            "annotation_path": annotations_path,
            "events": len(self.events),
            "boxes": len(self.raw_boxes),
            "modified": bool(self.validation_modified),
            "change_count": int(self.validation_change_count),
            "query_session_metrics": dict(getattr(self, "_query_metrics", {}) or {}),
            "safe_execution_metrics": dict(getattr(self, "_safe_execution_metrics", {}) or {}),
            "validation_log_summary": self._summarize_query_logger(
                getattr(self, "validation_op_logger", None)
            ),
            "ops_log_summary": self._summarize_query_logger(
                getattr(self, "op_logger", None)
            ),
            "graph_stats": graph_stats,
            "final_state": payload,
        }
        log_path = os.path.splitext(annotations_path)[0] + ".validation.json"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=True)
            print(f"[LOG] {log_path}")
            self._log("hoi_validation_summary_saved", path=log_path)
        except Exception as ex:
            print(f"[LOG][ERROR] HOI validation summary write failed: {log_path} ({ex})")
            QMessageBox.warning(
                self,
                "Validation log",
                f"Failed to write validation log:\n{log_path}\n\n{ex}",
            )

    def _summarize_query_logger(self, logger) -> dict:
        if logger is None or not hasattr(logger, "rows"):
            return {
                "session_id": "",
                "row_count": 0,
                "query_row_count": 0,
                "counts": {},
                "distributions": {},
                "latency_ms": {},
            }
        try:
            rows = list(logger.rows() or [])
        except Exception:
            rows = []
        query_events = {
            "hoi_query_present",
            "hoi_query_focus",
            "hoi_query_apply",
            "hoi_query_reject",
            "hoi_graph_propagate",
        }
        safe_execution_events = {
            "hoi_safe_execution_block",
            "hoi_safe_execution_violation",
            "hoi_safe_execution_rollback",
        }
        query_rows = [
            dict(row)
            for row in rows
            if str((row or {}).get("event") or "").strip() in (query_events | safe_execution_events)
        ]
        def _safe_bool(value: Any) -> bool:
            text = str(value or "").strip().lower()
            if text in {"1", "true", "yes", "y", "t"}:
                return True
            if text in {"0", "false", "no", "n", "f", ""}:
                return False
            return bool(value)

        def _row_source(row: Dict[str, Any]) -> str:
            source = str(
                row.get("source")
                or row.get("suggested_source")
                or "unknown"
            ).strip()
            return source or "unknown"

        def _row_hand_conditioned(row: Dict[str, Any]) -> bool:
            if _safe_bool(row.get("hand_conditioned")):
                return True
            source = _row_source(row).lower()
            query_type = str(row.get("query_type") or "").strip().lower()
            return (
                source.startswith("handtrack_once")
                or source.startswith("hand_conditioned")
                or "hand_conditioned" in query_type
            )

        counts = {
            "presented": 0,
            "focused": 0,
            "applied": 0,
            "accepted": 0,
            "propagated": 0,
            "rejected": 0,
            "safe_blocked": 0,
            "safe_violations": 0,
            "safe_rollbacks": 0,
            "hand_conditioned_presented": 0,
            "hand_conditioned_focused": 0,
            "hand_conditioned_applied": 0,
            "hand_conditioned_accepted": 0,
            "hand_conditioned_propagated": 0,
            "hand_conditioned_rejected": 0,
            "semantic_primary_query_rows": 0,
            "detector_grounding_primary_query_rows": 0,
        }
        by_query_type: Dict[str, int] = defaultdict(int)
        by_action_kind: Dict[str, int] = defaultdict(int)
        by_interaction_form: Dict[str, int] = defaultdict(int)
        by_authority: Dict[str, int] = defaultdict(int)
        by_resolve_kind: Dict[str, int] = defaultdict(int)
        by_suggested_source: Dict[str, int] = defaultdict(int)
        by_hand_conditioned: Dict[str, int] = defaultdict(int)
        by_noun_primary_source: Dict[str, int] = defaultdict(int)
        by_noun_primary_family: Dict[str, int] = defaultdict(int)
        latencies = {"focus": [], "apply": [], "reject": []}
        for row in query_rows:
            event_name = str(row.get("event") or "").strip()
            query_type = str(row.get("query_type") or "").strip() or "unknown"
            action_kind = str(row.get("action_kind") or "").strip() or "unknown"
            interaction_form = (
                str(row.get("interaction_form") or "").strip() or "unknown"
            )
            authority_level = (
                str(row.get("authority_level") or "").strip() or "unknown"
            )
            source = _row_source(row)
            hand_conditioned = _row_hand_conditioned(row)
            noun_primary_source = str(row.get("noun_primary_source") or "").strip() or "unknown"
            noun_primary_family = str(row.get("noun_primary_family") or "").strip() or "unknown"
            by_query_type[query_type] += 1
            by_action_kind[action_kind] += 1
            by_interaction_form[interaction_form] += 1
            by_authority[authority_level] += 1
            by_suggested_source[source] += 1
            by_hand_conditioned["true" if hand_conditioned else "false"] += 1
            if noun_primary_source != "unknown":
                by_noun_primary_source[noun_primary_source] += 1
            if noun_primary_family != "unknown":
                by_noun_primary_family[noun_primary_family] += 1
                if noun_primary_family == "semantic":
                    counts["semantic_primary_query_rows"] += 1
                elif noun_primary_family == "detector_grounding":
                    counts["detector_grounding_primary_query_rows"] += 1
            if event_name == "hoi_query_present":
                counts["presented"] += 1
                if hand_conditioned:
                    counts["hand_conditioned_presented"] += 1
            elif event_name == "hoi_query_focus":
                counts["focused"] += 1
                if hand_conditioned:
                    counts["hand_conditioned_focused"] += 1
            elif event_name == "hoi_query_apply":
                counts["applied"] += 1
                if hand_conditioned:
                    counts["hand_conditioned_applied"] += 1
                resolve_kind = str(row.get("resolve_kind") or "").strip() or "unknown"
                by_resolve_kind[resolve_kind] += 1
                if resolve_kind in ("human_accept_suggestion", "human_confirm_current"):
                    counts["accepted"] += 1
                    if hand_conditioned:
                        counts["hand_conditioned_accepted"] += 1
            elif event_name == "hoi_query_reject":
                counts["rejected"] += 1
                if hand_conditioned:
                    counts["hand_conditioned_rejected"] += 1
            elif event_name == "hoi_graph_propagate":
                counts["propagated"] += 1
                if hand_conditioned:
                    counts["hand_conditioned_propagated"] += 1
            elif event_name == "hoi_safe_execution_block":
                counts["safe_blocked"] += 1
            elif event_name == "hoi_safe_execution_violation":
                counts["safe_violations"] += 1
            elif event_name == "hoi_safe_execution_rollback":
                counts["safe_rollbacks"] += 1
            try:
                latency_ms = float(row.get("query_latency_ms"))
            except Exception:
                latency_ms = 0.0
            if latency_ms > 0:
                if event_name == "hoi_query_focus":
                    latencies["focus"].append(latency_ms)
                elif event_name == "hoi_query_apply":
                    latencies["apply"].append(latency_ms)
                elif event_name == "hoi_query_reject":
                    latencies["reject"].append(latency_ms)
        latency_summary = {}
        for key, values in latencies.items():
            latency_summary[f"{key}_mean"] = round(
                (sum(values) / len(values)) if values else 0.0, 2
            )
        acceptance_values = []
        empirical_cost_values = []
        calibrated_reliability_values = []
        noun_source_margin_values = []
        for row in query_rows:
            try:
                acceptance_prob = float(row.get("acceptance_prob_est"))
            except Exception:
                acceptance_prob = 0.0
            try:
                empirical_cost_ms = float(row.get("empirical_cost_ms"))
            except Exception:
                empirical_cost_ms = 0.0
            try:
                calibrated_reliability = float(row.get("calibrated_reliability"))
            except Exception:
                calibrated_reliability = 0.0
            try:
                noun_source_margin = float(row.get("noun_source_margin"))
            except Exception:
                noun_source_margin = 0.0
            if acceptance_prob > 0:
                acceptance_values.append(acceptance_prob)
            if empirical_cost_ms > 0:
                empirical_cost_values.append(empirical_cost_ms)
            if calibrated_reliability > 0:
                calibrated_reliability_values.append(calibrated_reliability)
            if noun_source_margin > 0:
                noun_source_margin_values.append(noun_source_margin)
        if acceptance_values:
            latency_summary["acceptance_prob_mean"] = round(
                sum(acceptance_values) / len(acceptance_values), 4
            )
        if empirical_cost_values:
            latency_summary["empirical_cost_ms_mean"] = round(
                sum(empirical_cost_values) / len(empirical_cost_values), 2
            )
        if calibrated_reliability_values:
            latency_summary["calibrated_reliability_mean"] = round(
                sum(calibrated_reliability_values) / len(calibrated_reliability_values),
                4,
            )
        if noun_source_margin_values:
            latency_summary["noun_source_margin_mean"] = round(
                sum(noun_source_margin_values) / len(noun_source_margin_values),
                4,
            )
        return {
            "session_id": getattr(logger, "session_id", "") or "",
            "row_count": int(len(rows)),
            "query_row_count": int(len(query_rows)),
            "counts": counts,
            "distributions": {
                "query_type": dict(sorted(by_query_type.items())),
                "action_kind": dict(sorted(by_action_kind.items())),
                "interaction_form": dict(sorted(by_interaction_form.items())),
                "authority_level": dict(sorted(by_authority.items())),
                "resolve_kind": dict(sorted(by_resolve_kind.items())),
                "suggested_source": dict(sorted(by_suggested_source.items())),
                "hand_conditioned": dict(sorted(by_hand_conditioned.items())),
                "noun_primary_source": dict(sorted(by_noun_primary_source.items())),
                "noun_primary_family": dict(sorted(by_noun_primary_family.items())),
            },
            "latency_ms": latency_summary,
        }

    def _load_annotations_json(self):
        """
        Load HOI annotations in the current unified per-hand format only.
        """
        fp, _ = QFileDialog.getOpenFileName(
            self, "Load HOI annotations", "", "JSON Files (*.json);;All Files (*)"
        )
        if not fp:
            return
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = normalize_noun_aliases(json.load(f))
        except Exception as ex:
            QMessageBox.warning(self, "Error", f"Failed to read JSON:\n{ex}")
            return
        if not isinstance(data, dict):
            QMessageBox.warning(self, "Error", "Invalid JSON: root must be an object.")
            return
        # Frozen Qwen proposals require their lossless human-review schema.
        # Delegate before legacy loading clears state or fills unknown Onset.
        if data.get("schema") == "IMPACT-PREANNOTATION-REVIEW-1.0":
            import os
            import sys
            import subprocess
            entry = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "preannotation_review", "review_preannotations.py")
            try:
                if not os.path.isfile(entry):
                    raise FileNotFoundError("Install the complete preannotation review update")
                subprocess.Popen([sys.executable, entry, os.path.abspath(fp)],env=dict(os.environ,IMPACT_SKIP_AUTOMATIC_TRACKS="1" if getattr(self,"skip_auto_tracking_import",False) else "0"))
            except Exception as ex:
                QMessageBox.warning(self, "Preannotation review", str(ex))
            return
        self._clear_undo_history()

        if isinstance(data, dict) and "tracks" in data and "hoi_events" in data:
            self._load_annotations_v2(data, annotation_path=fp)
            self.current_annotation_path = fp
            self._mark_query_calibration_dirty()
            self._log(
                "hoi_load_annotations",
                path=fp,
                events=len(self.events),
                boxes=len(self.raw_boxes),
            )
            self._mark_hoi_saved()
            return

        QMessageBox.warning(
            self,
            "Unsupported Format",
            "This build only supports the current HOI annotation format with 'tracks' and 'hoi_events'.",
        )

    def _load_annotations_v2(self, data: dict, annotation_path: str = ""):
        """Load unified per-hand event format."""
        from core.annotation_migration import adapt_annotation
        archive = None
        if data.get('schema') != 'hoi-annotation' and annotation_path and os.path.isfile(annotation_path):
            from core.annotation_archive import archive_source
            archive = archive_source(annotation_path)
        data = adapt_annotation(data, _PROJECT_PROFILE)
        self._annotation_provenance = copy.deepcopy(data.get('provenance', {}))
        if archive:self._annotation_provenance['original_archive'] = archive
        data = normalize_noun_aliases(data)
        self._clear_undo_history()
        default_actors_config = [
            {"id": "Left_hand", "label": "Left Hand", "short": "L"},
            {"id": "Right_hand", "label": "Right Hand", "short": "R"},
        ]

        def _sorted_int_strings(keys):
            sortable = []
            for key in keys:
                try:
                    sortable.append((int(key), key))
                except Exception:
                    continue
            sortable.sort(key=lambda item: item[0])
            return [raw_key for _num, raw_key in sortable]

        # Validate shared state before clearing existing annotations.
        shared = validate_timeline(data.get("shared_assembly", {"schema":"shared-assembly-1","states":[]}))
        if any(x["frame"] >= int(data.get("frame_count", self.player.frame_count)) for x in shared["states"]+shared['merges']):
            raise ValueError("Shared assembly state is outside the video")
        if any(str(x['object_id']) not in data.get('object_library',{}) for x in shared['states']):
            raise ValueError('Assembly state references an unknown object ID')
        if 'editor_view' in data:
            self._restore_editor_view(data['editor_view'])
        # Reset state
        self.frame_review = copy.deepcopy(data.get("frame_review",{}))
        self.shared_assembly = shared
        self._assembly_default_reference = data.get("assembly_default_reference",False)
        self.events.clear()
        self.event_id_counter = 0
        self.raw_boxes = []
        self._suppressed_hand_boxes = list((data.get("editor_geometry") or {}).get("suppressed_hand_boxes", []))
        self.bboxes = {}
        self._clear_relation_highlight()
        saved_video_path = str(data.get("video_path", "") or "").strip()
        if saved_video_path:
            resolved_video_path = ""
            if os.path.isabs(saved_video_path):
                resolved_video_path = saved_video_path
            elif annotation_path:
                candidate = os.path.join(
                    os.path.dirname(os.path.abspath(annotation_path)),
                    saved_video_path,
                )
                if os.path.isfile(candidate):
                    resolved_video_path = candidate
            if resolved_video_path or not self.video_path:
                self.video_path = resolved_video_path or saved_video_path
        self._noun_only_mode = True
        self.hoi_ontology = HOIOntology.from_dict(data.get("hoi_ontology"))
        self.hoi_ontology_path = str((data.get("hoi_ontology") or {}).get("source_path") or "")

        loaded_actors = data.get("actors_config")
        if isinstance(loaded_actors, list) and loaded_actors:
            self.actors_config = copy.deepcopy(loaded_actors)
        else:
            self.actors_config = copy.deepcopy(default_actors_config)
        self._reinit_actor_system()
        self._apply_noun_only_mode_ui()

        # Restore libraries
        self.global_object_map.clear()
        self.id_to_category.clear()
        self.class_map.clear()
        self._known_anomaly_labels = set(_ANOMALY_LABEL_OPTIONS)
        self.object_id_counter = 0
        for combo in (getattr(self, "combo_target", None), getattr(self, "combo_instrument", None)):
            if isinstance(combo, QComboBox):
                combo.blockSignals(True)
                combo.clear()
                combo.addItem(
                    "" if combo is getattr(self, "combo_instrument", None) else "None",
                    None,
                )
                if combo is getattr(self, "combo_instrument", None):
                    combo.setEditText("")
                combo.blockSignals(False)
        self.verbs.clear()
        self._update_verb_combo()
        self.label_panel.refresh()

        obj_lib = data.get("object_library", {})
        obj_class_map = {}
        max_obj_id = -1
        for id_str in _sorted_int_strings(obj_lib.keys()):
            try:
                uid = int(id_str)
            except Exception:
                continue
            info = obj_lib.get(id_str, {}) or {}
            label = (
                info.get("label")
                or info.get("name")
                or info.get("category")
                or f"obj_{uid}"
            )
            class_id = info.get("class_id")
            label = self._canonical_label_name(label)
            if not label:
                label = f"obj_{uid}"
            if class_id is not None:
                try:
                    class_id = int(class_id)
                except Exception:
                    pass
                obj_class_map[uid] = class_id
                if (
                    label
                    and class_id not in self.class_map
                    and str(class_id) not in self.class_map
                ):
                    self.class_map[class_id] = label
            self.global_object_map[label] = uid
            self.id_to_category[label] = label
            max_obj_id = max(max_obj_id, uid)
            display_text = f"[{uid}] {label}"
            for combo in (getattr(self, "combo_target", None), getattr(self, "combo_instrument", None)):
                if isinstance(combo, QComboBox) and self._object_label_is_selectable(label):
                    combo.addItem(display_text, uid)
        retired = [int(uid) for uid in self._annotation_provenance.get('assembly_identity_map',{})]
        self.object_id_counter = max([max_obj_id,*retired]) + 1
        self.box_id_counter = max(self.object_id_counter + 1, 1)

        if "verb_library" in data:
            verb_lib = data.get("verb_library", {}) or {}
            for vid_str in _sorted_int_strings(verb_lib.keys()):
                try:
                    vid = int(vid_str)
                except Exception:
                    continue
                name = verb_lib.get(vid_str, "")
                name = self._canonical_label_name(name)
                if not name:
                    continue
                color = self._color_for_index(vid)
                self.verbs.append(LabelDef(name=name, id=vid, color_name=color))
            self._renumber_verbs()
            self._update_verb_combo()
            self.label_panel.refresh()
        # Tracks -> raw boxes
        tracks = data.get("tracks", {})
        track_obj_id = {}

        def _label_for_id(uid):
            for name, id_val in self.global_object_map.items():
                if id_val == uid:
                    return name
            return None

        for track_id, info in tracks.items():
            info = info or {}
            category = str(info.get("category", "")).strip()
            obj_id = info.get("object_id", None)
            track_class_id = info.get("class_id")
            if track_class_id is None and obj_id is not None:
                track_class_id = obj_class_map.get(obj_id)
            track_obj_id[track_id] = obj_id
            boxes = list(info.get("boxes", []) or []) + list(info.get("inactive_boxes", []) or [])

            tid_lower = str(track_id).lower()
            cat_lower = category.lower()

            label = None
            for actor in self.actors_config:
                aid = actor["id"]
                aid_lower = aid.lower()
                if (cat_lower == aid_lower) or (aid_lower in tid_lower):
                    label = aid
                    break

            if not label:
                label = _label_for_id(obj_id) or category or f"obj_{obj_id}"

            is_actor = any(actor["id"] == label for actor in self.actors_config)

            for entry in boxes:
                frame = entry.get("frame", None)
                bbox = entry.get("bbox", None)
                if frame is None or not bbox or len(bbox) != 4:
                    continue
                try:
                    f_idx = int(frame)
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                except Exception:
                    continue
                box_class_id = entry.get("class_id", track_class_id)
                if box_class_id is not None:
                    try:
                        box_class_id = int(box_class_id)
                    except Exception:
                        pass
                if is_actor or obj_id is None:
                    bid = self.box_id_counter
                    self.box_id_counter += 1
                else:
                    bid = obj_id
                if not is_actor and box_class_id is not None and label:
                    if (
                        box_class_id not in self.class_map
                        and str(box_class_id) not in self.class_map
                    ):
                        self.class_map[box_class_id] = label
                if getattr(self,"skip_auto_tracking_import",False) and not (entry.get("human_verified") or entry.get("locked") or str(entry.get("source","")).startswith(("manual","human"))):continue
                new_rb = {
                    "id": bid,
                    "orig_frame": f_idx - self.start_offset,
                    "label": label,
                    "source": entry.get("source","loaded_annotation"),
                    "human_verified": bool(entry.get("human_verified",False)),
                    "locked": bool(entry.get("locked")),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
                if not is_actor and box_class_id is not None:
                    new_rb["class_id"] = box_class_id
                self.raw_boxes.append(new_rb)

        self._rebuild_bboxes_from_raw()
        self._refresh_boxes_for_frame(self.player.current_frame)

        # Events -> internal event entries
        events = data.get("hoi_events", {})

        def _event_to_entry(side_key: str, event: dict):
            s = event.get("start_frame")
            o = event.get("contact_onset_frame")
            e = event.get("end_frame")
            links = event.get("links", {}) or {}
            interaction = event.get("interaction", {}) or {}
            verb = event.get("verb") or ""
            annotation_state = event.get("annotation_state", {}) or {}

            def _safe_int(val, fallback):
                try:
                    return int(val)
                except Exception:
                    return fallback

            def _object_id_for_label(label):
                text = str(label or "").strip()
                if not text:
                    return None
                if text in self.global_object_map:
                    return self.global_object_map.get(text)
                norm = self._norm_category(text)
                for name, uid in self.global_object_map.items():
                    if self._norm_category(name) == norm:
                        return uid
                return None

            target_tid = links.get("target_track_id")
            target_id = event.get("noun_object_id", event.get("target_object_id"))
            if target_id is None:target_id = track_obj_id.get(target_tid)
            if target_id is None:
                target_id = _object_id_for_label(
                    interaction.get("target") or interaction.get("noun")
                )
            target_id = _safe_int(target_id, None)

            instrument_tid = links.get("tool_track_id") or links.get("instrument_track_id")
            instrument_id = event.get("instrument_object_id", event.get("tool_object_id"))
            if instrument_id is None:instrument_id = track_obj_id.get(instrument_tid)
            if instrument_id is None:
                instrument_id = _object_id_for_label(
                    interaction.get("instrument") or interaction.get("tool")
                )
            instrument_id = _safe_int(instrument_id, None)

            anomaly_label = self._remember_anomaly_label(display_value(event, _PROJECT_PROFILE))

            start = _safe_int(s, 0)
            end = _safe_int(e, start)
            if end < start:
                start, end = end, start

            empty = {
                "verb": "",
                "target_object_id": None,
                "noun_object_id": None,
                "instrument_object_id": None,
                "tool_object_id": None,
                "anomaly_label": _DEFAULT_ANOMALY_LABEL,
                "has_anomaly": False,
                "interaction_start": None,
                "functional_contact_onset": None,
                "interaction_end": None,
            }
            hoi_data = {actor["id"]: dict(empty) for actor in self.actors_config}
            # Correct matching
            hand_key = None
            for actor in self.actors_config:
                if actor["id"].lower() == side_key:
                    hand_key = actor["id"]
                    break

            if hand_key:
                hoi_data[hand_key] = self._ensure_hand_annotation_state(
                    {
                        "_event_extra": {k: copy.deepcopy(event[k]) for k in (
                            "event_uid", "machine_event_uid", "anomaly_evidence", "migration_review",
                            "local_outcome", "retry_of_event_id", "onset_review_state", "onset_reason"
                        ) if k in event},
                        "_source_event_id": event.get("event_id"),
                        "required_review_frames": list(event.get("required_review_frames",[])),
                        "_onset_manually_adjusted": bool(event.get("onset_manually_adjusted",False)),
                        "shared_assembly_ref": bool(event.get("shared_assembly_ref",False)),
                        "shared_assembly_id": event.get("shared_assembly_id"),
                        "verb": verb,
                        "target_object_id": target_id,
                        "noun_object_id": target_id,
                        "instrument_object_id": instrument_id,
                        "tool_object_id": instrument_id,
                        "anomaly_label": anomaly_label,
                        "has_anomaly": self._is_anomalous_label(anomaly_label),
                        "interaction_start": s,
                        "functional_contact_onset": o,
                        "interaction_end": e,
                        "_field_state": annotation_state.get("field_state", {}),
                        "_field_suggestions": annotation_state.get("field_suggestions", {}),
                        "_sparse_evidence_state": annotation_state.get(
                            "sparse_evidence_state", {}
                        ),
                    }
                )
            return {
                "event_id": self.event_id_counter,
                "frames": [start, end],
                "hoi_data": hoi_data,
            }

        for actor in self.actors_config:
            side_key = actor["id"].lower()
            for event in events.get(side_key, []) or []:
                self.events.append(_event_to_entry(side_key, event))
                self.event_id_counter += 1

        for event in self.events:
            hoi_data = event.get("hoi_data", {}) or {}
            for actor in self.actors_config:
                hand_key = actor["id"]
                if hand_key in hoi_data and isinstance(hoi_data[hand_key], dict):
                    self._hydrate_hand_annotation_state(
                        hoi_data[hand_key], default_source="loaded_annotation"
                    )
                    self._compute_sparse_evidence_state(hoi_data[hand_key], persist=True)

        self._refresh_anomaly_combos(_DEFAULT_ANOMALY_LABEL)
        self._ensure_verbs_cover_events()
        self._refresh_events()
        self._after_events_loaded()

        video_id = data.get("video_id", "")
        QMessageBox.information(
            self,
            "Loaded",
            f"Loaded {len(self.events)} events.\nVideo ID: {video_id or 'unknown'}",
        )

    def _parse_yolo(self, path: str, w: int, h: int) -> List[dict]:
        """
        Supports:
        - Single txt with lines: frame cls xc yc w h [conf]
        - Or directory of per-frame txt (frame_XXXX.txt) with lines: cls xc yc w h [conf]
          coords assumed normalized if <=1, else treated as absolute center/size in px.
        """
        data: List[dict] = []
        if os.path.isdir(path):
            files = sorted([f for f in os.listdir(path) if f.lower().endswith(".txt")])
            for fname in files:
                frame_id = self._frame_from_filename(fname)
                full = os.path.join(path, fname)
                self._parse_yolo_file(full, w, h, data, frame_override=frame_id)
        else:
            self._parse_yolo_file(path, w, h, data, frame_override=None)
        return data

    def _frame_from_filename(self, fname: str) -> int:
        """Extract frame number from filename (e.g., frame_000123.txt -> 123)"""
        name = os.path.splitext(fname)[0]
        # expect frame_000123
        digits = "".join(ch for ch in name if ch.isdigit())
        try:
            return int(digits)
        except Exception:
            return 0

    def _parse_yolo_file(
        self, path: str, w: int, h: int, data: List[dict], frame_override: int = None
    ):
        """Parse individual YOLO text file."""
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                if len(parts) < 5:
                    continue
                idx = 0
                frame = None
                has_frame_token = len(parts) >= 6 and self._looks_number(parts[0])
                if has_frame_token:
                    try:
                        frame = int(float(parts[0]))
                        idx = 1
                    except Exception:
                        frame = None
                        idx = 0
                if frame is None:
                    frame = frame_override if frame_override is not None else 0
                cls = parts[idx]
                coords = parts[idx + 1 : idx + 5]
                if len(coords) < 4:
                    continue
                xc, yc, bw, bh = map(float, coords[:4])
                conf = None
                conf_idx = idx + 5
                if len(parts) > conf_idx and self._looks_number(parts[conf_idx]):
                    try:
                        conf = float(parts[conf_idx])
                    except Exception:
                        conf = None
                normed = max(xc, yc, bw, bh) <= 1.0
                if normed:
                    xc *= w
                    yc *= h
                    bw *= w
                    bh *= h
                x1 = max(0.0, xc - bw * 0.5)
                y1 = max(0.0, yc - bh * 0.5)
                x2 = min(float(w), xc + bw * 0.5)
                y2 = min(float(h), yc + bh * 0.5)
                box = {
                    "id": self.box_id_counter,
                    "orig_frame": frame,
                    "label": self._cls_name(cls),
                    "class_id": self._cls_id(cls),
                    "confidence": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
                self.box_id_counter += 1
                data.append(box)

    @staticmethod
    def _looks_number(txt: str) -> bool:
        try:
            float(txt)
            return True
        except Exception:
            return False

    def _cls_name(self, cls_token):
        try:
            cid = int(float(cls_token))
            if cid in self.class_map:
                return self.class_map[cid]
        except Exception:
            pass
        return str(cls_token)

    def _cls_id(self, cls_token):
        try:
            return int(float(cls_token))
        except Exception:
            return None

    def _build_category_to_default_id(self) -> dict:
        category_to_default_id = {}
        for obj_name, uid in self.global_object_map.items():
            category = self._norm_category(self.id_to_category.get(obj_name, obj_name))
            if not category:
                continue
            if (
                category not in category_to_default_id
                or uid < category_to_default_id[category]
            ):
                category_to_default_id[category] = uid
        return category_to_default_id

    def _norm_category(self, text: str, *, for_detection: bool = False) -> str:
        if not text:
            return ""
        cleaned = str(text).strip().lower()
        cleaned = cleaned.replace("-", "_").replace(" ", "_")
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        from core.annotation_migration import canonical_noun
        cleaned = canonical_noun(cleaned, _PROJECT_PROFILE)
        instance = re.fullmatch(r'(.*)_(\d+)',cleaned)
        if instance and instance[1] in _PROJECT_PROFILE.get('noun_aliases',{}).values():cleaned=instance[1]
        cleaned = _apply_task_specific_category_mapping(cleaned, for_detection=for_detection)
        return cleaned

    def _canonical_label_name(self, text: Any) -> str:
        from core.annotation_migration import canonical_noun
        return canonical_noun(str(text or "").strip("\ufeff").strip().lower(), _PROJECT_PROFILE)

    def _portable_path_token(self, path: Any) -> str:
        text = str(path or "").strip()
        if not text:
            return ""
        return text.replace("\\", "/").rstrip("/").split("/")[-1]

    def _parse_yaml_names(self, path: str) -> Dict[int, str]:
        """
        Minimal parser for data.yaml names mapping. Supports both:
        names: {0: 'label', 1: 'label2'} and flat mappings.
        """
        out: Dict = {}
        in_names = False
        pending_lines = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                pending_lines.append(line)
                if line.startswith("names:"):
                    # Try to parse inline content
                    content = line[len("names:") :].strip()
                    if content.startswith("[") and content.endswith("]"):
                        items = content[1:-1].split(",")
                        for idx, itm in enumerate(items):
                            out[idx] = itm.strip().strip("'\"")
                        return out
                    if content.startswith("{") and content.endswith("}"):
                        pairs = content[1:-1].split(",")
                        for p in pairs:
                            if ":" in p:
                                k, v = p.split(":", 1)
                                try:
                                    out[int(k.strip())] = v.strip().strip("'\"")
                                except Exception:
                                    pass
                        return out
                    in_names = True
                    continue
                if in_names:
                    if ":" in line:
                        key_txt, val_txt = line.split(":", 1)
                        key_txt = key_txt.strip()
                        val_txt = val_txt.strip().strip("'\"")
                        try:
                            idx = int(key_txt)
                            out[idx] = val_txt
                        except Exception:
                            continue
                    else:
                        in_names = False
        if not out:
            for line in pending_lines:
                if ":" not in line:
                    continue
                key_txt, val_txt = line.split(":", 1)
                key_txt = key_txt.strip()
                val_txt = val_txt.strip().strip("'\"")
                if not key_txt.isdigit():
                    continue
                try:
                    idx = int(key_txt)
                    out[idx] = val_txt
                except Exception:
                    continue
        return out

    def _replace_verb_in_hoi_data(self, hoi_data: dict, old: str, new: str) -> None:
        """Update verb string in a hoi_data dict for both hands."""
        if not hoi_data or not old or old == new:
            return
        for actor in self.actors_config:
            hand_key = actor["id"]
            h = hoi_data.get(hand_key, {})
            if h.get("verb") == old:
                h["verb"] = new

    def _collect_event_verbs(self) -> List[str]:
        """Return unique verb strings referenced by any hand in current events."""
        seen = []
        for ev in self.events:
            hoi_data = ev.get("hoi_data", {}) or {}
            for actor in self.actors_config:
                hand_key = actor["id"]
                verb = (hoi_data.get(hand_key, {}) or {}).get("verb") or ""
                if verb and verb not in seen:
                    seen.append(verb)
        return seen

    def _ensure_verbs_cover_events(self) -> None:
        """
        Ensure the verb list includes every verb referenced in events.
        Used when loading valid files or events with verbs but no explicit library.
        """
        name_to_id = {v.name: v.id for v in self.verbs}
        cur_max = max(name_to_id.values(), default=-1)
        added = False
        for verb in self._collect_event_verbs():
            if verb in name_to_id:
                continue
            cur_max += 1
            name_to_id[verb] = cur_max
            self.verbs.append(
                LabelDef(
                    name=verb, id=cur_max, color_name=self._color_for_index(cur_max)
                )
            )
            added = True
        if added:
            self._renumber_verbs()
            self._update_verb_combo()
            self.label_panel.refresh()

    def _build_payload_v2(self) -> dict:
        """[Fix] Build payload with STRICT filtering for empty hands."""

        self._refresh_sparse_evidence_snapshots()
        self.events.sort(key=lambda x: x.get("frames", [0, 0])[0])

        base = os.path.splitext(os.path.basename(self.video_path or "annotation"))[0]

        name_to_id = {
            self._canonical_label_name(v.name): v.id
            for v in self.verbs
            if self._canonical_label_name(v.name)
        }
        cur_max_vid = max(name_to_id.values(), default=-1)
        for verb_name in self._collect_event_verbs():
            verb_name = self._canonical_label_name(verb_name)
            if not verb_name:
                continue
            if verb_name in name_to_id:
                continue
            cur_max_vid += 1
            name_to_id[verb_name] = cur_max_vid
        verb_library = {
            str(v_id): name
            for name, v_id in sorted(name_to_id.items(), key=lambda x: x[1])
        }

        object_library = {}
        for name, uid in sorted(self.global_object_map.items(), key=lambda x: x[1]):
            export_name = self._canonical_label_name(name)
            category = self.id_to_category.get(name, export_name)
            if any(s['object_id']==uid and len(s['components'])>1 for s in self._assembly_data().get('states',[])):
                category = 'assembly'
            entry = {"label": export_name, "category": category}
            cid = self._class_id_for_object(uid, name)
            if cid is not None:
                entry["class_id"] = cid
            object_library[str(uid)] = entry

        if not object_library:
            for rb in self.raw_boxes:
                label = rb.get("label")
                if self._is_hand_label(label):
                    continue
                uid = rb.get("id")
                if uid is None or str(uid) in object_library:
                    continue
                export_label = self._canonical_label_name(label)
                entry = {"label": export_label, "category": export_label}
                cid = self._class_id_for_object(uid, label)
                if cid is not None:
                    entry["class_id"] = cid
                object_library[str(uid)] = entry

        def label_for_id(uid):
            for name, id_val in self.global_object_map.items():
                if id_val == uid:
                    return self._canonical_label_name(name)
            return None

        # --------------------------------------

        # Build per-actor events
        events = {actor["id"].lower(): [] for actor in self.actors_config}
        counts = {actor["id"]: 0 for actor in self.actors_config}

        for i, event in enumerate(self.events):
            global_start, global_end = event.get("frames", [0, 0])

            for actor in self.actors_config:
                hand_key = actor["id"]
                side_key = hand_key.lower()
                h_data = event.get("hoi_data", {}).get(hand_key, {})
                self._ensure_hand_annotation_state(h_data)

                from core.annotation_migration import canonical_verb
                verb = canonical_verb(h_data.get("verb", ""), _PROJECT_PROFILE)
                target = resolve_object(h_data, self._assembly_data(), int(h_data.get("interaction_start") if h_data.get("interaction_start") is not None else event.get("frames",[0])[0]))
                instrument = self._hand_instrument_object_id(h_data)
                anomaly_label = self._normalize_anomaly_label(h_data.get("anomaly_label"))
                has_anomaly = self._is_anomalous_label(anomaly_label)
                s = h_data.get("interaction_start")
                o = h_data.get("functional_contact_onset")
                e = h_data.get("interaction_end")

                has_verb = bool(verb and verb.strip())
                has_objects = (target is not None and not h_data.get('shared_assembly_ref')) or instrument is not None
                has_timestamps = (s is not None) or (o is not None) or (e is not None)
                has_info = has_verb or has_objects or has_timestamps or has_anomaly

                if not has_info:
                    continue

                counts[hand_key] += 1
                prefix = self._get_actor_short_label(hand_key)
                event_id = h_data.get("_source_event_id") or f"{prefix}_{counts[hand_key]:03d}"

                final_start = s if s is not None else global_start
                final_onset = o
                final_end = e if e is not None else global_end

                event_entry = {
                    **copy.deepcopy(h_data.get("_event_extra", {})),
                    "event_id": event_id,
                    "start_frame": final_start,
                    "contact_onset_frame": final_onset,
                    "end_frame": final_end,
                    "noun_object_id": target,
                    "target_object_id": target,
                    "instrument_object_id": instrument,
                    "tool_object_id": instrument,
                    **export_review(anomaly_label, _PROJECT_PROFILE),
                    "has_anomaly": bool(has_anomaly),
                    "annotation_state": {
                        "field_state": self._export_field_state_aliases(
                            h_data.get("_field_state", {}) or {}
                        ),
                        "field_suggestions": self._export_field_suggestion_aliases(
                            h_data.get("_field_suggestions", {}) or {}
                        ),
                        "sparse_evidence_state": self._export_sparse_evidence_state(
                            h_data.get("_sparse_evidence_state", {}) or {}
                        ),
                    },
                }

                if final_onset is not None:
                    event_entry.pop('onset_review_state',None)
                    event_entry.pop('onset_reason',None)
                if has_verb:
                    event_entry["verb"] = verb
                else:
                    event_entry["verb"] = ""

                interaction = {}
                assembly_id = reference_id(h_data,self._assembly_data(),final_start)
                target_label = (noun_at(self._assembly_data(),final_start,assembly_id) if h_data.get("shared_assembly_ref") else label_for_id(target))
                event_entry["shared_assembly_ref"] = bool(h_data.get("shared_assembly_ref",False))
                event_entry["shared_assembly_id"] = assembly_id if h_data.get("shared_assembly_ref") else None
                event_entry["onset_manually_adjusted"] = bool(h_data.get("_onset_manually_adjusted",False))
                event_entry["required_review_frames"] = list(h_data.get("required_review_frames",[]))
                if target_label:
                    interaction["target"] = target_label
                    interaction["noun"] = target_label
                instrument_label = label_for_id(instrument)
                if instrument_label:
                    interaction["instrument"] = instrument_label
                    interaction["tool"] = instrument_label
                event_entry["interaction"] = interaction

                links = {
                    "subject_track_id": f"T_{hand_key.upper()}",
                    "target_track_id": f"T_OBJ_{target}" if target is not None else None,
                    "tool_track_id": (
                        f"T_OBJ_{instrument}" if instrument is not None else None
                    ),
                }
                event_entry["links"] = links
                if h_data.get('shared_assembly_ref'):
                    links['target_track_segments'] = track_segments(self._assembly_data(),assembly_id,final_start,final_end)

                events[side_key].append(event_entry)

        tracks = {}

        def collect_hand_boxes(label):
            out = []
            for rb in self.raw_boxes:
                if rb.get("label") == label:
                    entry = {
                        "frame": int(rb["orig_frame"]) + int(self.start_offset),
                        "bbox": [rb["x1"], rb["y1"], rb["x2"], rb["y2"]],
                        "human_verified": bool(rb.get("human_verified",False)),
                        "source": rb.get("source","unknown"),
                    }
                    if self._is_box_locked(rb):
                        entry["locked"] = True
                    out.append(entry)
            out.sort(key=lambda x: x["frame"])
            return out

        for actor in self.actors_config:
            hand_key = actor["id"]
            side_key = hand_key.lower()
            track_id = f"T_{hand_key.upper()}"
            explicit_boxes = collect_hand_boxes(hand_key)
            support_track = self._handtrack_track(hand_key)
            tracks[track_id] = {
                "category": side_key,
                "object_id": None,
                "boxes": explicit_boxes,
                "boxes_source": "raw_boxes",
            }
            if support_track:
                tracks[track_id]["hand_support_available"] = True
                tracks[track_id]["hand_support_source"] = "handtrack_once"
                tracks[track_id]["hand_support_frame_count"] = int(
                    support_track.get("frame_count", 0) or 0
                )
                tracks[track_id]["hand_support_coverage"] = float(
                    support_track.get("coverage", 0.0) or 0.0
                )
                if not explicit_boxes:
                    tracks[track_id]["note"] = (
                        "No explicit editable hand boxes were saved for this actor; "
                        "clip-level hand support is stored in payload.handtrack_once."
                    )

        for uid_str, info in object_library.items():
            uid = int(uid_str)
            label = info["label"]
            cat = info["category"]
            cid = info.get("class_id")
            b_list = []
            for rb in self.raw_boxes:
                if rb.get("id") == uid and not self._normalize_hand_label(rb.get("label")):
                    ent = {
                        "frame": int(rb["orig_frame"]) + int(self.start_offset),
                        "bbox": [rb["x1"], rb["y1"], rb["x2"], rb["y2"]],
                        "human_verified": bool(rb.get("human_verified",False)),
                        "source": rb.get("source","unknown"),
                    }
                    if cid is not None:
                        ent["class_id"] = cid
                    if self._is_box_locked(rb):
                        ent["locked"] = True
                    b_list.append(ent)
            b_list.sort(key=lambda x: x["frame"])
            tracks[f"T_OBJ_{uid}"] = {
                "category": cat,
                "object_id": uid,
                "boxes": b_list,
            }
            merge = merge_at(self._assembly_data(), uid)
            if merge:
                tracks[f'T_OBJ_{uid}']['boxes'] = [b for b in b_list if b['frame'] < merge['frame']]
                tracks[f'T_OBJ_{uid}']['inactive_boxes'] = [b for b in b_list if b['frame'] >= merge['frame']]
                tracks[f'T_OBJ_{uid}']['active_end_frame'] = merge['frame'] - 1
                tracks[f'T_OBJ_{uid}']['merged_into_object_id'] = merge['target_object_id']
            if cid is not None:
                tracks[f"T_OBJ_{uid}"]["class_id"] = cid

        handtrack_payload = self._sanitize_handtrack_payload_for_save(
            getattr(self, "_handtrack_cache", {}) or {}
        )
        handtrack_summary = {}
        for actor in self.actors_config:
            track = self._handtrack_track(actor["id"])
            if not track:
                continue
            handtrack_summary[str(actor["id"])] = {
                "coverage": float(track.get("coverage", 0.0) or 0.0),
                "motion_peak_frame": (
                    None
                    if track.get("motion_peak_frame") is None
                    else int(track.get("motion_peak_frame"))
                ),
                "motion_peak_score": float(track.get("motion_peak_score", 0.0) or 0.0),
                "frame_count": int(track.get("frame_count", 0) or 0),
            }

        payload = {
            "schema": "hoi-annotation",
            "video_id": getattr(self, "_task_trial_id", "") or base,
            "video_path": self._portable_path_token(self.video_path),
            "participant_code": self._normalized_participant_code(),
            "experiment_mode": self._experiment_mode_key(),
            "fps": self.player.frame_rate,
            "frame_size": [self.player._frame_w, self.player._frame_h],
            "frame_count": self.player.frame_count,
            "bbox_mode": "xyxy",
            "bbox_normalized": False,
            "object_library": object_library,
            "verb_library": verb_library,
            "noun_only_mode": bool(self._noun_only_mode),
            "hoi_ontology": self.hoi_ontology.to_dict(),
            "actors_config": self.actors_config,
            "tracks": tracks,
            "hoi_events": events,
            "handtrack_once": {
                **dict(handtrack_payload or {}),
                "ready": bool(handtrack_summary),
                "cache_file": self._portable_path_token(
                    self._handtrack_status.get("cache_file")
                ),
                "tracks_summary": handtrack_summary,
            },
        }
        payload['provenance'] = copy.deepcopy(getattr(self, '_annotation_provenance', {}))
        payload["frame_review"] = copy.deepcopy(getattr(self,"frame_review",{}))
        payload["shared_assembly"] = validate_timeline(self._assembly_data())
        payload["assembly_default_reference"] = getattr(self,"_assembly_default_reference",False)
        payload["editor_geometry"] = {"suppressed_hand_boxes": list(getattr(self, "_suppressed_hand_boxes", []))}
        payload['editor_view'] = dict(start_frame=int(self.start_offset), end_frame=self.end_frame)
        return payload

    # ---------- UI refresh ----------
    def _set_frame_controls(self, frame: int):
        self._refresh_assembly_caption()
        self._refresh_frame_review_status()
        super()._set_frame_controls(frame)
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.set_current_frame(frame)

    def _on_frame_advanced(self, frame: int):
        self._refresh_boxes_for_frame(frame, skip_events=True, lightweight=True)
        self._set_frame_controls(frame)

    def _on_player_playback_state_changed(self, _playing: bool):
        self._update_play_pause_button()

    def _update_play_pause_button(self):
        playing = bool(getattr(self.player, "is_playing", False))
        self.btn_play.setIcon(
            self.style().standardIcon(
                QStyle.SP_MediaPause if playing else QStyle.SP_MediaPlay
            )
        )
        self.btn_play.setToolTip("Pause" if playing else "Play")

    def _event_visible_object_ids(self, frame=None):
        """Only selected event entities may appear on the canvas; ID 0 is valid."""
        ids = set()
        if self.selected_event_id is None:
            return ids
        draft = getattr(self, "event_draft", {}) or {}
        for data in draft.values():
            if not isinstance(data, dict):
                continue
            f = self._assembly_frame() if frame is None else frame
            for value in (resolve_object(data,self._assembly_data(),f), self._hand_instrument_object_id(data)):
                if value is not None:
                    ids.add(str(value))
        return ids

    def _event_box_visible(self, box):
        if self._normalize_hand_label(box.get("label")):
            return True
        frame = int(box.get('orig_frame',self._assembly_frame()-self.start_offset)) + int(self.start_offset)
        uid = box.get('id')
        return uid is not None and active_id(self._assembly_data(),uid,frame)==uid and str(uid) in self._event_visible_object_ids(frame)

    def _refresh_boxes_for_frame(
        self, frame: int, skip_events: bool = False, lightweight: bool = False
    ):
        """
        Refresh logic: prepare boxes, update list, update player overlay, refresh relations.
        """
        self._update_overlay(frame)
        relation_highlights = getattr(self, "_relation_highlights", {}) or {}
        validation_highlights = getattr(self, "_validation_highlights", {}) or {}
        highlight_ids = {
            **dict(relation_highlights.get("by_id", {}) or {}),
            **dict(validation_highlights.get("by_id", {}) or {}),
        }
        highlight_labels = {
            **dict(relation_highlights.get("by_label", {}) or {}),
            **dict(validation_highlights.get("by_label", {}) or {}),
        }
        boxes = [b for b in self._frame_boxes_with_cached_hands(frame) if self._event_box_visible(b)]
        editable_boxes = [
            dict(box)
            for box in list(self.bboxes.get(frame, []) or [])
            if isinstance(box, dict) and self._event_box_visible(box)
        ]
        self.current_hands = {actor["id"]: None for actor in self.actors_config}
        visible_hand_filter = ""
        if self.selected_event_id is not None and str(getattr(self, "selected_hand_label", "") or "").strip():
            visible_hand_filter = str(self.selected_hand_label or "").strip()

        self.list_objects.blockSignals(True)
        self.list_objects.clear()

        display_boxes = []
        attention = self._current_visual_attention_targets()
        attention_hand = str(attention.get("hand_key") or "").strip()
        attention_target_id = attention.get("target_id")
        proposal_styles: Dict[int, Dict[str, Any]] = {}
        if attention_hand:
            proposal_styles = self._current_frame_object_proposal_styles(
                frame,
                attention_hand,
                self._selected_hand_data(),
            )
        noun_focus_mode = False
        selected_hand_data = self._selected_hand_data()
        selected_edit_box = (
            dict(getattr(self, "_selected_edit_box", {}) or {})
            if bool(getattr(self, "chk_edit_boxes", None) and self.chk_edit_boxes.isChecked())
            else {}
        )
        if attention_target_id is not None and self._hand_noun_object_id(selected_hand_data) is not None:
            noun_focus_mode = True
        try:
            attention_target_id = (
                int(attention_target_id) if attention_target_id is not None else None
            )
        except Exception:
            attention_target_id = None

        for b in boxes:
            lbl = str(b.get("label"))
            locked = self._is_box_locked(b)
            norm_hand = self._normalize_hand_label(lbl)
            selected_for_edit = self._same_box_identity(b, selected_edit_box)
            object_id = None
            synthetic = False
            if norm_hand:
                self.current_hands[norm_hand] = b
                hand_short = self._get_actor_short_label(norm_hand)
                synthetic = self._is_synthetic_hand_box(b)
                item_txt = f"[Hand] {hand_short}"
                draw_label = hand_short
                color = "#38BDF8" if synthetic else "#2563EB"
            else:
                uid = b.get("id")
                name = self._object_name_for_id(uid, fallback=f"ID_{uid}")
                explicit_label = str(b.get("label") or "").strip()
                if explicit_label:
                    if name.startswith("ID_") or self._norm_category(explicit_label) != self._norm_category(name):
                        name = explicit_label
                cid = b.get("class_id")
                if cid is None:
                    cid = self._class_id_for_label(b.get("label") or name)
                cid_txt = str(cid) if cid is not None else "_"
                item_txt = f"[obj:{uid} | cls:{cid_txt}] {name}"
                draw_label = f"{name} [ID {uid}]"
                color = self._class_color(cid)
            if locked:
                item_txt = f"[LOCK] {item_txt}"

            from core.frame_review import valid_record
            entity="H:"+norm_hand if norm_hand else "O:"+str(b.get("id"))
            reviewed=valid_record(getattr(self,"frame_review",{}).get(str(frame),{}).get(entity),self._boxes_for_review(entity,frame))
            item = QListWidgetItem(("✓ " if reviewed else "○ ")+item_txt)
            item.setForeground(QColor("#15803d" if reviewed else "#a16207"))
            item.setToolTip("Human verified" if reviewed else "Not verified for this geometry")
            item.setData(Qt.UserRole, b)
            self.list_objects.addItem(item)
            thick = False
            if b.get("id") in highlight_ids:
                color = highlight_ids[b.get("id")]
                thick = True
            elif norm_hand and norm_hand in highlight_labels:
                color = highlight_labels[norm_hand]
                thick = True
            if not thick:
                if norm_hand and attention_hand and norm_hand == attention_hand:
                    color = "#2563EB"
                    thick = True
                elif not norm_hand and attention_target_id is not None:
                    try:
                        if int(b.get("id")) == attention_target_id:
                            color = "#22C55E"
                            thick = True
                    except Exception:
                        pass
            if not norm_hand:
                try:
                    object_id = int(b.get("id"))
                except Exception:
                    object_id = None
                proposal_style = (
                    dict(proposal_styles.get(object_id) or {})
                    if object_id is not None
                    else {}
                )
                if proposal_style and attention_target_id != object_id:
                    color = str(proposal_style.get("color") or "#F59E0B")
                    thick = bool(proposal_style.get("thick", True))
                if selected_for_edit:
                    color = "#F97316"
                    thick = True
                dimmed = bool(
                    noun_focus_mode
                    and object_id is not None
                    and attention_target_id is not None
                    and object_id != attention_target_id
                    and not proposal_style
                    and not selected_for_edit
                )
            else:
                other_hand = bool(visible_hand_filter and norm_hand != visible_hand_filter)
                if other_hand and not thick and not selected_for_edit:
                    color = "#64748B"
                dimmed = bool(other_hand and not selected_for_edit)
            alpha = 220
            label_alpha = 140
            if dimmed:
                alpha = 72
                label_alpha = 70
            elif selected_for_edit:
                alpha = 255
                label_alpha = 210
            overlay_label = draw_label
            if dimmed and not norm_hand:
                overlay_label = ""

            display_boxes.append(
                {
                    "id": b.get("id"),
                    "frame": frame,
                    "orig_frame": b.get("orig_frame"),
                    "x1": b["x1"],
                    "y1": b["y1"],
                    "x2": b["x2"],
                    "y2": b["y2"],
                    "label": overlay_label,
                    "color": color,
                    "alpha": alpha,
                    "label_alpha": label_alpha,
                    "thick": thick,
                    "dashed": bool((not norm_hand) and object_id is not None and bool(proposal_styles.get(object_id)) and attention_target_id != object_id),
                    "selected": bool(selected_for_edit),
                    "locked": False,
                    "synthetic": bool(synthetic),
                }
            )

        self.list_objects.blockSignals(False)
        if selected_edit_box:
            selected_hand_key = self._normalize_hand_label(selected_edit_box.get("label"))
            if selected_hand_key:
                self._select_box_item_in_list(
                    hand_key=selected_hand_key,
                    best_bbox=selected_edit_box,
                )
            else:
                self._select_box_item_in_list(
                    object_id=selected_edit_box.get("id"),
                    best_bbox=selected_edit_box,
                )

        if hasattr(self.player, "set_overlay_boxes"):
            self.player.set_overlay_boxes(display_boxes)
        elif hasattr(self.player, "set_boxes"):
            self.player.set_boxes(display_boxes)
        else:
            print("Warning: VideoPlayer missing set_overlay_boxes method")

        if hasattr(self.player, "set_edit_context"):
            if self.chk_edit_boxes.isChecked():
                edit_boxes = []
                for b in editable_boxes:
                    edit_boxes.append(
                        {
                            "id": b.get("id"),
                            "frame": frame,
                            "orig_frame": b.get("orig_frame"),
                            "x1": b.get("x1"),
                            "y1": b.get("y1"),
                            "x2": b.get("x2"),
                            "y2": b.get("y2"),
                            "label": b.get("label"),
                            "class_id": b.get("class_id"),
                            "locked": False,
                        }
                    )
                self.player.set_edit_context(
                    edit_boxes,
                    on_change=self._on_box_edited,
                    on_select=self._on_edit_box_selected,
                    on_materialize=self._materialize_synthetic_hand_box_for_edit,
                    label_resolver=self._resolve_label_and_id,
                    label_suggestions=self._label_suggestions(),
                    auto_label_fetcher=self._get_auto_draw_label,
                    allow_add=True,
                    allow_edit=True,
                    selected_box=selected_edit_box or None,
                )
            else:
                self.player.set_edit_context(
                    [],
                    on_change=self._on_box_edited,
                    on_select=self._on_edit_box_selected,
                    on_materialize=self._materialize_synthetic_hand_box_for_edit,
                    label_resolver=self._resolve_label_and_id,
                    label_suggestions=self._label_suggestions(),
                    auto_label_fetcher=self._get_auto_draw_label,
                    allow_add=True,
                    allow_edit=False,
                    selected_box=None,
                )

        if not lightweight:
            self._update_inspector_tab_labels()

        if not skip_events:
            self._refresh_events(refresh_boxes=False)

    def _get_auto_draw_label(self):
        self._pending_draw_box_role = None
        if not self.selected_hand_label:
            return None
        frame_idx = int(getattr(self.player, "current_frame", 0) or 0)
        selected_hand = self._normalize_hand_label(self.selected_hand_label)
        hand_data = self.event_draft.get(self.selected_hand_label) or {}

        if not self._frame_has_selected_hand_box(frame_idx, selected_hand):
            options = []
            hand_short = self._get_actor_short_label(selected_hand)
            hand_choice = f"Hand ({hand_short})"
            options.append(hand_choice)
            target_obj_id = self._hand_noun_object_id(hand_data)
            target_name = self._object_name_for_id(
                target_obj_id,
                default_for_none="",
                fallback="",
            )
            object_choice = (
                f"Object ({target_name})" if target_name else "Object (choose label)"
            )
            options.append(object_choice)
            instrument_obj_id = self._hand_instrument_object_id(hand_data)
            instrument_name = self._object_name_for_id(
                instrument_obj_id,
                default_for_none="",
                fallback="",
            )
            instrument_choice = (
                f"Instrument ({instrument_name})"
                if instrument_name
                else "Instrument (choose label)"
            )
            options.append(instrument_choice)
            choice, ok = QInputDialog.getItem(
                self,
                "New Box",
                "No hand box is available on this frame.\nCreate which kind of box?",
                options,
                0,
                False,
            )
            if not ok:
                return ("", None)
            if choice == hand_choice:
                self._pending_draw_box_role = "hand"
                if getattr(self, "rad_draw_target", None) is not None:
                    self.rad_draw_target.setChecked(True)
                return selected_hand, None
            if choice == instrument_choice:
                self._pending_draw_box_role = "instrument"
                if instrument_name:
                    return instrument_name, self._class_id_for_label(instrument_name)
                return None
            self._pending_draw_box_role = "noun"
            if target_name:
                return target_name, self._class_id_for_label(target_name)
            return None

        hand_data = self.event_draft.get(self.selected_hand_label) or {}
        if bool(getattr(self, "rad_draw_inst", None) and self.rad_draw_inst.isChecked()):
            self._pending_draw_box_role = "instrument"
            target_obj_id = self._hand_instrument_object_id(hand_data)
        elif bool(getattr(self, "rad_draw_target", None) and self.rad_draw_target.isChecked()):
            self._pending_draw_box_role = "noun"
            target_obj_id = self._hand_noun_object_id(hand_data)
        else:
            target_obj_id = None
        if target_obj_id is None:
            if bool(getattr(self, "rad_draw_none", None) and self.rad_draw_none.isChecked()):
                return None
            if (
                getattr(self, "rad_draw_target", None) is not None
                and not self.rad_draw_target.isChecked()
                and getattr(self, "rad_draw_inst", None) is not None
                and not self.rad_draw_inst.isChecked()
            ):
                return None
            return None
        for name, uid in self.global_object_map.items():
            if uid == target_obj_id:
                return name, self._class_id_for_label(name)
        return None

    def _refresh_events(self, refresh_boxes: bool = True):
        """Refresh event views and derived review/query state."""
        frame = self.player.current_frame
        for ev in self.events:
            self._sync_event_frames(ev)
        self._refresh_sparse_evidence_snapshots()
        self._update_overlay(frame)
        relation_highlights = getattr(self, "_relation_highlights", {}) or {}
        has_relation_highlights = bool(
            relation_highlights.get("by_id") or relation_highlights.get("by_label")
        )
        if refresh_boxes and (
            self.validation_enabled or has_relation_highlights
        ):
            self._refresh_boxes_for_frame(frame, skip_events=True)
        self._refresh_completion_indicators(refresh_evidence=False)
        self._update_next_best_query_panel()

    def _update_overlay(self, frame: int):
        """
        Draw active HOI links on top of the current frame.
        """
        hand_color = "#3b82f6"
        target_color = "#22c55e"
        instrument_color = "#f59e0b"
        display_rels = []

        # 1. Valid HOIs from committed events
        for ev in self.events:
            if ev.get("event_id") != self.selected_event_id:
                continue
            for actor in self.actors_config:
                hand_key = actor["id"]
                h_data = ev["hoi_data"][hand_key]
                s = h_data.get("interaction_start")
                e = h_data.get("interaction_end")

                if s is not None and e is not None and s <= frame <= e:
                    draw_item = dict(h_data)
                    draw_item["hand_id"] = hand_key
                    draw_item["event_id"] = ev.get("event_id")
                    display_rels.append(draw_item)

        # 2. Preview from current Draft
        if hasattr(self, "event_draft"):
            for actor in self.actors_config:
                hand_key = actor["id"]
                h_data = self.event_draft[hand_key]
                if self.selected_event_id is not None and (self._hand_noun_object_id(h_data) is not None or self._hand_instrument_object_id(h_data) is not None):
                    draw_item = dict(h_data)
                    draw_item["hand_id"] = hand_key
                    draw_item["event_id"] = self.selected_event_id
                    draw_item["_draft_preview"] = True
                    display_rels.append(draw_item)

        overlay_data = []
        highlight_ids = {}
        highlight_labels = {}
        current_boxes = {
            (("hand",self._normalize_hand_label(b.get("label"))) if self._normalize_hand_label(b.get("label")) else b["id"]): b
            for b in self._frame_boxes_with_cached_hands(frame)
            if isinstance(b, dict) and "id" in b
        }

        for r in display_rels:
            hand_label = r.get("hand_id")
            hand_box = None
            for b in current_boxes.values():
                if self._normalize_hand_label(b.get("label")) == hand_label:
                    hand_box = b
                    break

            if not hand_box:
                continue

            target_id = resolve_object(r, self._assembly_data(), frame)
            instrument_id = r.get("instrument_object_id", r.get("tool_object_id"))
            verb = r.get("verb", "unknown")
            try:
                relation_selected = int(r.get("event_id")) == int(self.selected_event_id)
            except Exception:
                relation_selected = bool(r.get("_draft_preview"))
            if str(hand_label or "") != str(getattr(self, "selected_hand_label", "") or ""):
                relation_selected = False
            highlight_relation = bool(
                relation_selected or self.selected_event_id is None
            )
            relation_width = 5 if relation_selected else 3

            if target_id is not None:
                if target_id in current_boxes:
                    overlay_data.append(
                        {
                            "box_a": hand_box,
                            "box_b": current_boxes[target_id],
                            "label": verb,
                            "color": target_color,
                            "width": relation_width,
                        }
                    )
                    if highlight_relation:
                        highlight_labels[hand_label] = hand_color
                        highlight_ids[target_id] = target_color
            if instrument_id is not None:
                if instrument_id in current_boxes:
                    overlay_data.append(
                        {
                            "box_a": hand_box,
                            "box_b": current_boxes[instrument_id],
                            "label": "instrument",
                            "color": instrument_color,
                            "width": relation_width,
                        }
                    )
                    if highlight_relation:
                        highlight_labels[hand_label] = hand_color
                        highlight_ids[instrument_id] = instrument_color

        self._relation_highlights = {
            "by_id": highlight_ids,
            "by_label": highlight_labels,
        }

        if self.validation_enabled:
            self._validation_highlights = {
                "by_id": dict(highlight_ids),
                "by_label": dict(highlight_labels),
            }
        else:
            self._validation_highlights = {}

        overlay_lines = []
        for rel in overlay_data:
            try:
                box_a = rel["box_a"]
                box_b = rel["box_b"]
                x1 = (float(box_a["x1"]) + float(box_a["x2"])) / 2.0
                y1 = (float(box_a["y1"]) + float(box_a["y2"])) / 2.0
                x2 = (float(box_b["x1"]) + float(box_b["x2"])) / 2.0
                y2 = (float(box_b["y1"]) + float(box_b["y2"])) / 2.0
            except Exception:
                continue
            overlay_lines.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "label": rel.get("label", ""),
                    "color": rel.get("color", ""),
                    "width": rel.get("width", 3),
                }
            )
        self.player.set_overlay_relations(overlay_lines)

    def _stop(self):
        try:
            self.player.stop()
        except Exception:
            pass
        self._update_play_pause_button()
        self._set_frame_controls(self.player.current_frame)
        self._refresh_boxes_for_frame(self.player.current_frame)
        self._log("hoi_stop")

    def _toggle_play_pause(self):
        if getattr(self.player, "is_playing", False):
            self._pause()
        else:
            self._play()

    def _play(self):
        try:
            self.player.play()
        except Exception:
            pass
        self._update_play_pause_button()
        self._log("hoi_play")

    def _pause(self):
        try:
            self.player.pause()
        except Exception:
            pass
        self._update_play_pause_button()
        self._log("hoi_pause")

    def _seek_relative(self, delta: int):
        if not self.player.cap:
            return
        target = max(
            self.player.crop_start,
            min(self.player.crop_end, self.player.current_frame + delta),
        )
        self.player.seek(target)
        self._refresh_boxes_for_frame(target)
        self._set_frame_controls(target)
        self._log("hoi_seek_relative", delta=delta, target=target)

    def _seek_seconds(self, direction: int):
        """Seek by +/-1 second based on video FPS."""
        if not self.player.cap:
            return
        fps = max(1, int(round(self.player.frame_rate or 30)))
        self._seek_relative(direction * fps)

    def _jump_to_spin(self):
        frame = int(self.spin_jump.value())
        if self.player.cap:
            self.player.seek(frame)
            self._refresh_boxes_for_frame(frame)
            self.slider.blockSignals(True)
            self.slider.setValue(frame)
            self.slider.blockSignals(False)
            if getattr(self, "hoi_timeline", None):
                self.hoi_timeline.set_current_frame(frame)
            self._log("hoi_jump_to_frame", target=frame)

    def _on_slider_pressed(self):
        self._slider_scrubbing = True

    def _on_slider_changed(self, val: int):
        if self.player.cap:
            if self._slider_scrubbing or self.slider.isSliderDown():
                self.player.seek(val, preview_only=True)
                self._refresh_boxes_for_frame(val, skip_events=True, lightweight=True)
                if getattr(self, "hoi_timeline", None):
                    self.hoi_timeline.set_current_frame(val)
                return
            self.player.seek(val)
            self._refresh_boxes_for_frame(val)
            if getattr(self, "hoi_timeline", None):
                self.hoi_timeline.set_current_frame(val)
            self._log("hoi_seek_slider", target=val)

    def _on_slider_released(self):
        if not self.player.cap:
            return
        self._slider_scrubbing = False
        val = int(self.slider.value())
        self.player.seek(val)
        self._refresh_boxes_for_frame(val)
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.set_current_frame(val)
        self._log("hoi_seek_slider", target=val)

    def _on_offset_changed(self):
        self.start_offset = int(self.spin_start_offset.value())
        end_val = int(self.spin_end_frame.value())
        self.end_frame = end_val if end_val >= self.start_offset else None
        self._rebuild_bboxes_from_raw()
        self._refresh_boxes_for_frame(self.player.current_frame)
        self._log("hoi_crop_set", start=self.start_offset, end=self.end_frame)

    def _set_edit_boxes_enabled(self, on: bool):
        checkbox = getattr(self, "chk_edit_boxes", None)
        if checkbox is None:
            return
        desired = bool(on)
        if checkbox.isChecked() == desired:
            if getattr(self, "btn_inline_edit_boxes", None) is not None:
                self.btn_inline_edit_boxes.blockSignals(True)
                self.btn_inline_edit_boxes.setChecked(desired)
                self.btn_inline_edit_boxes.blockSignals(False)
            self._update_inline_edit_boxes_button_state()
            return
        checkbox.setChecked(desired)

    def _current_edit_boxes_mode_label(self) -> str:
        checkbox = getattr(self, "chk_edit_boxes", None)
        if checkbox is None or not checkbox.isChecked():
            return "OFF"
        if bool(getattr(self, "rad_draw_inst", None) and self.rad_draw_inst.isChecked()):
            return "ON: Instrument"
        if bool(getattr(self, "rad_draw_target", None) and self.rad_draw_target.isChecked()):
            return "ON: Noun"
        return "ON: Manual"

    def _update_inline_edit_boxes_button_state(self, *_args) -> None:
        btn = getattr(self, "btn_inline_edit_boxes", None)
        checkbox = getattr(self, "chk_edit_boxes", None)
        if btn is None:
            return
        mode_label = self._current_edit_boxes_mode_label()
        btn.blockSignals(True)
        if checkbox is not None:
            btn.setChecked(bool(checkbox.isChecked()))
        btn.setText(f"Edit Boxes [{mode_label}]")
        btn.setToolTip(
            "Edit box mode.\n"
            "- OFF: canvas drag pans the video\n"
            "- ON: Manual = Ctrl+drag creates an unlabeled/manual box\n"
            "- ON: Instrument = Ctrl+drag creates a box that inherits the selected event instrument\n"
            "- ON: Noun = Ctrl+drag creates a box that inherits the selected event noun\n"
            "Shortcut: Ctrl+B"
        )
        btn.blockSignals(False)

    def _toggle_edit_boxes_shortcut(self):
        checkbox = getattr(self, "chk_edit_boxes", None)
        if checkbox is None:
            return
        self._set_edit_boxes_enabled(not checkbox.isChecked())

    def _on_edit_boxes_toggled(self, on: bool):
        if getattr(self, "btn_inline_edit_boxes", None) is not None:
            self.btn_inline_edit_boxes.blockSignals(True)
            self.btn_inline_edit_boxes.setChecked(bool(on))
            self.btn_inline_edit_boxes.blockSignals(False)
        if not on:
            self._selected_edit_box = None
        for widget in (
            getattr(self, "rad_draw_none", None),
            getattr(self, "rad_draw_inst", None),
            getattr(self, "rad_draw_target", None),
        ):
            if widget is not None:
                widget.setEnabled(bool(on))
        if not on and getattr(self, "rad_draw_none", None) is not None:
            self.rad_draw_none.setChecked(True)
        self._update_draw_mode_visibility()
        self._update_inline_edit_boxes_button_state()
        self._refresh_boxes_for_frame(self.player.current_frame)
        self._log("hoi_edit_boxes_toggle", on=bool(on))

    def _set_validation_ui_state(self, on: bool):
        if getattr(self, "lbl_validation", None):
            if on:
                self.lbl_validation.setStyleSheet("color: #12b76a; font-weight: 600;")
            else:
                self.lbl_validation.setStyleSheet("")
        self._set_status_chip(
            getattr(self, "lbl_validation_chip", None),
            "Validation On" if on else "Validation Off",
            "ok" if on else "neutral",
        )
        if getattr(self, "btn_validation", None):
            if on:
                self.btn_validation.setToolTip("Return to annotation mode")
            else:
                self.btn_validation.setToolTip("Toggle validation on/off")
        self._update_incomplete_indicator()

    def _on_validation_toggled(self, on: bool):
        if on:
            name, ok = QInputDialog.getText(self, "Validation", "Editor name:")
            if not ok or not name.strip():
                self.btn_validation.blockSignals(True)
                self.btn_validation.setChecked(False)
                self.btn_validation.blockSignals(False)
                self._set_validation_ui_state(False)
                self._log("hoi_validation_cancel")
                return
            self.validator_name = name.strip()
            self.validation_enabled = True
            self.validation_session_active = True
            self.validation_started_at = datetime.now().isoformat(timespec="seconds")
            self.validation_modified = False
            self.validation_change_count = 0
            if getattr(self, "validation_op_logger", None):
                self.validation_op_logger.enabled = self.op_logger.enabled
                self.validation_op_logger.clear()
            self._set_validation_ui_state(True)
            self._refresh_boxes_for_frame(self.player.current_frame)
            self._log("hoi_validation_on", validator=self.validator_name)
        else:
            self.validation_enabled = False
            self.validation_session_active = False
            self._set_validation_ui_state(False)
            self._refresh_boxes_for_frame(self.player.current_frame)
            self._log("hoi_validation_off")

    def _experiment_mode_key(self) -> str:
        combo = getattr(self, "combo_experiment_mode", None)
        if combo is not None:
            return str(combo.currentData() or getattr(self, "_experiment_mode", "full_assist"))
        return str(getattr(self, "_experiment_mode", "full_assist") or "full_assist")

    def _manual_mode_enabled(self) -> bool:
        return self._experiment_mode_key() == "manual"

    def _detection_assist_enabled(self) -> bool:
        return self._experiment_mode_key() in ("assist", "full_assist")

    def _semantic_assist_enabled(self) -> bool:
        return self._experiment_mode_key() == "full_assist"

    def _timeline_onset_state_for_mode(self) -> Dict[str, str]:
        if self._manual_mode_enabled():
            return {
                "source": "manual_timeline",
                "status": "confirmed",
                "note": "Manual mode treats timeline onset edits as confirmed until the user changes them again.",
            }
        return {
            "source": "manual_timeline_provisional",
            "status": "suggested",
            "note": "Timeline edits create a provisional onset anchor that should be confirmed or refined during review.",
        }

    def _guard_experiment_mode(self, feature: str) -> bool:
        if feature == "detection" and self._detection_assist_enabled():
            return True
        if feature == "semantic" and self._semantic_assist_enabled():
            return True
        if feature == "semantic":
            QMessageBox.information(
                self,
                "Experiment Mode",
                "Switch to Full Assist to use action suggestions.",
            )
        else:
            QMessageBox.information(
                self,
                "Experiment Mode",
                "Switch to Full Assist to use imported or detected assistance.",
            )
        return False

    def _guard_asset_mutation(
        self,
        asset_label: str,
        *,
        path: str = "",
        notify_user: bool = True,
        auto_discovered: bool = False,
        full_assist_only: bool = False,
    ) -> bool:
        asset_text = str(asset_label or "asset").strip() or "asset"
        asset_path = str(path or "").strip()
        if full_assist_only and self._manual_mode_enabled():
            message = f"{asset_text} is disabled in Manual mode."
            self._log(
                "hoi_asset_change_blocked",
                asset=asset_text,
                path=asset_path,
                reason="manual_mode_block",
                auto_discovered=bool(auto_discovered),
                mode=self._experiment_mode_key(),
            )
            if notify_user:
                QMessageBox.information(self, "Experiment Mode", message)
            return False
        if bool(getattr(self, "_user_study_mode", True)) and self._workspace_has_annotation_state():
            message = (
                f"Finish or reset the current clip before changing {asset_text}.\n\n"
                "User-study mode keeps clip assets fixed once annotation has started."
            )
            self._log(
                "hoi_asset_change_blocked",
                asset=asset_text,
                path=asset_path,
                reason="study_session_locked",
                auto_discovered=bool(auto_discovered),
                mode=self._experiment_mode_key(),
            )
            if notify_user:
                QMessageBox.warning(self, "Study Session Locked", message)
            return False
        return True

    def _apply_experiment_mode_ui(self) -> None:
        mode = self._experiment_mode_key()
        self._experiment_mode = mode
        detection_enabled = self._detection_assist_enabled()
        semantic_enabled = self._semantic_assist_enabled()

        for obj in (
            getattr(self, "act_import_yolo_boxes", None),
            getattr(self, "act_load_yolo_model", None),
            getattr(self, "act_detect_current_frame", None),
            getattr(self, "act_detect_selected_action", None),
            getattr(self, "act_detect_all_actions", None),
            getattr(self, "act_load_hands_xml", None),
            getattr(self, "act_incremental_train_yolo", None),
            getattr(self, "act_swap_hands", None),
            getattr(self, "btn_detect", None),
            getattr(self, "btn_detect_action", None),
            getattr(self, "btn_object_tools", None),
        ):
            if obj is not None:
                obj.setEnabled(detection_enabled)

        for obj in (
            getattr(self, "act_load_videomae_model", None),
            getattr(self, "act_load_videomae_cache", None),
            getattr(self, "act_load_videomae_verb_list", None),
            getattr(self, "act_review_selected_action_label", None),
            getattr(self, "act_auto_apply_action_labels", None),
            getattr(self, "act_batch_apply_action_labels", None),
            getattr(self, "btn_suggest_action_label", None),
        ):
            if obj is not None:
                obj.setEnabled(semantic_enabled)

        if getattr(self, "btn_suggest_action_label", None) is not None:
            self.btn_suggest_action_label.setVisible(semantic_enabled)
        if getattr(self, "act_batch_apply_action_labels", None) is not None:
            self.act_batch_apply_action_labels.setVisible(semantic_enabled)
        if getattr(self, "next_query_card", None) is not None:
            self.next_query_card.setVisible(semantic_enabled)
        if getattr(self, "lbl_hand_support_status", None) is not None:
            self.lbl_hand_support_status.setVisible(not self._manual_mode_enabled())

        if not semantic_enabled:
            try:
                self._videomae_auto_timer.stop()
            except Exception:
                pass
            self._update_action_top5_display(None)
        else:
            self._update_action_top5_display(self.selected_event_id)
            if self.selected_event_id is not None:
                self._queue_action_label_refresh(self.selected_event_id, delay_ms=120)
        self._refresh_inference_action_states()
        self._apply_user_study_ui_profile()
        self._apply_inline_editor_mode_visibility()

    def _confirm_save_before_mode_switch(self, previous_mode: str, next_mode: str) -> bool:
        if not self._workspace_has_annotation_state():
            return True
        if not self._hoi_has_unsaved_changes():
            return True

        mode_labels = {
            "manual": "Manual",
            "full_assist": "Full Assist",
        }
        previous_label = mode_labels.get(str(previous_mode or "").strip(), str(previous_mode or "").strip() or "Current")
        next_label = mode_labels.get(str(next_mode or "").strip(), str(next_mode or "").strip() or "New")

        reply = QMessageBox.question(
            self,
            "Save Before Switching Mode",
            (
                f"You are switching from {previous_label} to {next_label}.\n\n"
                "The current HOI annotation workspace has unsaved changes.\n"
                "Save before switching mode?"
            ),
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if reply == QMessageBox.Cancel:
            return False
        if reply == QMessageBox.Save:
            self._save_annotations_json()
            return not self._hoi_has_unsaved_changes()
        return True

    def _confirm_save_before_loading_video(self, target_video: str) -> bool:
        if not self._workspace_has_annotation_state():
            return True
        if not self._hoi_has_unsaved_changes():
            return True

        target_name = os.path.basename(str(target_video or "").strip()) or "new video"
        reply = QMessageBox.question(
            self,
            "Save Before Loading Video",
            (
                f"You are loading {target_name}.\n\n"
                "The current HOI annotation workspace has unsaved changes.\n"
                "Save before loading the new video?"
            ),
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if reply == QMessageBox.Cancel:
            return False
        if reply == QMessageBox.Save:
            self._save_annotations_json()
            return not self._hoi_has_unsaved_changes()
        return True

    def _on_experiment_mode_changed(self, _index: int) -> None:
        if bool(getattr(self, "_experiment_mode_change_guard", False)):
            return
        previous_mode = str(getattr(self, "_experiment_mode", "full_assist") or "full_assist")
        next_mode = self._experiment_mode_key()
        if next_mode != previous_mode:
            if not self._confirm_save_before_mode_switch(previous_mode, next_mode):
                combo = getattr(self, "combo_experiment_mode", None)
                if combo is not None:
                    idx = combo.findData(previous_mode)
                    if idx >= 0:
                        self._experiment_mode_change_guard = True
                        try:
                            combo.setCurrentIndex(idx)
                        finally:
                            self._experiment_mode_change_guard = False
                return
        self._apply_experiment_mode_ui()
        if not self._semantic_assist_enabled():
            self._clear_all_runtime_suggestions()
            if self.selected_event_id is not None:
                event = self._find_event_by_id(self.selected_event_id)
                if event is not None:
                    self._refresh_semantic_suggestions_for_event(self.selected_event_id, event)
                    self._update_status_label()
        self._update_onboarding_banner()
        self._log("hoi_experiment_mode_changed", mode=self._experiment_mode_key())
        self._log_annotation_ready_state("hoi_experiment_mode_changed")
        self._maybe_warn_full_assist_semantic_unavailable("hoi_experiment_mode_changed")

    def _hoi_state_signature(self) -> str:
        payload = {
            "frame_review": getattr(self,"frame_review",{}),
            "shared_assembly": self._assembly_data(),
            "assembly_default_reference": getattr(self,"_assembly_default_reference",False),
            "events": self.events,
            "raw_boxes": self.raw_boxes,
            "event_draft": getattr(self, "event_draft", None),
            "object_map": self.global_object_map,
            "class_map": self.class_map,
            "start_offset": self.start_offset,
            "end_frame": self.end_frame,
        }
        try:
            blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
        except Exception:
            blob = str(
                (
                    len(self.events),
                    len(self.raw_boxes),
                    self.start_offset,
                    self.end_frame,
                )
            )
        return hashlib.md5(blob.encode("utf-8")).hexdigest()

    def _mark_hoi_saved(self) -> None:
        self._hoi_saved_signature = self._hoi_state_signature()

    def _hoi_has_unsaved_changes(self) -> bool:
        saved = getattr(self, "_hoi_saved_signature", None)
        current = self._hoi_state_signature()
        if saved is None:
            return bool(self.events or self.raw_boxes)
        return current != saved

    def _on_object_selection(self):
        if not self.selected_hand_label:
            self._clear_relation_highlight()
            self._update_overlay(self.player.current_frame)
            return

        items = self.list_objects.selectedItems()
        if not items:
            return
        obj_box = items[0].data(Qt.UserRole)
        if not obj_box:
            return
        if self._is_hand_label(obj_box.get("label")):
            return
        if bool(getattr(self, "chk_edit_boxes", None) and self.chk_edit_boxes.isChecked()):
            self._selected_edit_box = dict(obj_box)
        obj_id = obj_box["id"]

        hand_data = self.event_draft.get(self.selected_hand_label)
        if not hand_data:
            return
        if bool(getattr(self, "rad_draw_inst", None) and self.rad_draw_inst.isChecked()):
            prev_instrument = self._hand_instrument_object_id(hand_data)
            if prev_instrument == obj_id:
                self._select_object_item_in_list(obj_id, best_bbox=obj_box)
                return
            self._push_undo()
            hand_data["instrument_object_id"] = obj_id
            hand_data["tool_object_id"] = obj_id
            self._load_hand_draft_to_ui(self.selected_hand_label)
            self._update_overlay(self.player.current_frame)
            self._apply_draft_to_selected_event()
            self._bump_query_state_revision()
            self._refresh_events()
            self._update_hoi_titles()
            if getattr(self, "hoi_timeline", None):
                self.hoi_timeline.refresh()
            self._log(
                "hoi_select_instrument_object",
                hand=self.selected_hand_label,
                event_id=self.selected_event_id,
                previous_instrument=prev_instrument,
                instrument_object_id=self._hand_instrument_object_id(hand_data),
            )
            return
        before_hand = copy.deepcopy(hand_data)
        prev_noun_snapshot = self._field_snapshot(hand_data, "noun_object_id")
        prev_tar = self._hand_noun_object_id(hand_data)
        if (
            prev_tar == obj_id
            and str(prev_noun_snapshot.get("status") or "").strip().lower() == "confirmed"
        ):
            self._select_object_item_in_list(obj_id, best_bbox=obj_box)
            return

        self._push_undo()
        hand_data["noun_object_id"] = obj_id
        hand_data["target_object_id"] = obj_id
        self._bind_assembly_object(hand_data,obj_id)
        self._set_hand_field_state(
            hand_data,
            "noun_object_id",
            source="object_pick",
            status="confirmed",
        )

        self._load_hand_draft_to_ui(self.selected_hand_label)
        self._update_overlay(self.player.current_frame)
        self._apply_draft_to_selected_event()
        self._bump_query_state_revision()
        self._refresh_events()
        self._update_hoi_titles()
        self.hoi_timeline.refresh()
        if prev_tar != obj_id:
            new_noun_snapshot = self._field_snapshot(hand_data, "noun_object_id")
            self._log(
                "hoi_select_object",
                hand=self.selected_hand_label,
                event_id=self.selected_event_id,
                previous_target=prev_tar,
                target=self._hand_noun_object_id(hand_data),
                noun_rework_after_automation=bool(
                    (prev_noun_snapshot.get("status") or prev_noun_snapshot.get("value") is not None)
                    and self._is_automation_source(prev_noun_snapshot.get("source"))
                ),
                rework_after_automation=bool(
                    (prev_noun_snapshot.get("status") or prev_noun_snapshot.get("value") is not None)
                    and self._is_automation_source(prev_noun_snapshot.get("source"))
                ),
                previous_target_source=prev_noun_snapshot.get("source"),
                new_target_source=new_noun_snapshot.get("source"),
            )
            event = self._find_event_by_id(self.selected_event_id)
            if event is not None and self.selected_hand_label:
                self._set_semantic_reinfer_hint(
                    self.selected_event_id,
                    self.selected_hand_label,
                    hand_data,
                    reason="noun_object_pick",
                    edited_fields=["noun_object_id"],
                )
                self._record_semantic_feedback(
                    event,
                    self.selected_hand_label,
                    reason="noun_object_pick",
                    before_hand=before_hand,
                    edited_fields=["noun_object_id"],
                    supervision_kind="edited",
                )
                self._refresh_semantic_suggestions_for_event(self.selected_event_id, event)

    def _assign_drawn_object_to_current_hand(
        self,
        object_id: Any,
        role: Optional[str],
        best_bbox: Optional[dict] = None,
    ) -> None:
        role = str(role or "").strip().lower()
        if role not in {"noun", "instrument"}:
            return
        if self.selected_event_id is None or not self.selected_hand_label:
            return
        hand_data = self.event_draft.get(self.selected_hand_label)
        if not isinstance(hand_data, dict):
            return
        try:
            object_id = int(object_id)
        except Exception:
            return
        if role == "instrument":
            hand_data["instrument_object_id"] = object_id
            hand_data["tool_object_id"] = object_id
        else:
            hand_data["noun_object_id"] = object_id
            hand_data["target_object_id"] = object_id
            self._bind_assembly_object(hand_data,object_id)
            self._set_hand_field_state(
                hand_data,
                "noun_object_id",
                source="manual_box_add",
                status="confirmed",
            )
        self._load_hand_draft_to_ui(self.selected_hand_label)
        self._apply_draft_to_selected_event()
        self._bump_query_state_revision()
        self._refresh_events(refresh_boxes=False)
        self._update_hoi_titles()
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.refresh()
        self._select_object_item_in_list(object_id, best_bbox=best_bbox)
        self._update_status_label()
        self._log(
            "hoi_assign_drawn_object",
            event_id=self.selected_event_id,
            hand=self.selected_hand_label,
            object_role=role,
            object_id=object_id,
        )

    def _select_hand(self, hand_label: str, on: bool):
        """[Step 2] Switch hand context: save old data -> switch -> load new data."""

        # 1. Mutex Logic & UI State Management
        if on:
            # Save previous hand state
            if self.selected_hand_label and self.selected_hand_label != hand_label:
                self._save_ui_to_hand_draft(self.selected_hand_label)
                self._apply_draft_to_selected_event()

            # Set new state
            self.selected_hand_label = hand_label

            # Enforce Mutex UI: uncheck other actors
            for aid, chk in self.actor_controls.items():
                if aid != hand_label:
                    chk.blockSignals(True)
                    chk.setChecked(False)
                    chk.blockSignals(False)

            # Load new hand data to UI
            self._load_hand_draft_to_ui(hand_label)
            self._log("hoi_select_hand", hand=hand_label, on=True)
        else:
            # Uncheck current hand
            if self.selected_hand_label == hand_label:
                self._save_ui_to_hand_draft(hand_label)  # Save final state
                self._apply_draft_to_selected_event()
                self.selected_hand_label = None
                self._update_status_label()
                self._log("hoi_select_hand", hand=hand_label, on=False)

        self.list_objects.setEnabled(self.selected_hand_label is not None)
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.set_selected(
                self.selected_event_id, self.selected_hand_label
            )
            self._update_hoi_titles()
        self._update_overlay(self.player.current_frame)

    def _clear_relation_highlight(self):
        self.selected_event_id = None
        self.selected_hand_label = None
        self._reset_event_draft()
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.set_selected(None, None)
            self._update_hoi_titles()
        self._update_overlay(self.player.current_frame)

    def _on_box_edited(self, box_id: int, new_box: dict):
        """Apply edits to the raw_boxes by id and refresh views."""
        action = new_box.get("_action")
        previous_box_id = box_id
        target_frame = new_box.get("frame")
        target_orig = new_box.get("orig_frame")
        if target_orig is None and target_frame is not None:
            try:
                target_orig = int(target_frame) - int(self.start_offset)
            except Exception:
                target_orig = None

        selected_snapshot = (
            dict(getattr(self, "_selected_edit_box", {}) or {})
            if isinstance(getattr(self, "_selected_edit_box", None), dict)
            else {}
        )

        def _coords_close_to(box: dict, ref: dict) -> bool:
            if not isinstance(box, dict) or not isinstance(ref, dict):
                return False
            try:
                return all(
                    abs(float(box.get(key, 0.0) or 0.0) - float(ref.get(key, 0.0) or 0.0))
                    <= 2.0
                    for key in ("x1", "y1", "x2", "y2")
                )
            except Exception:
                return False

        def _matches(rb):
            if box_id is not None and rb.get("id") != box_id:
                return False
            if target_orig is not None and rb.get("orig_frame") != target_orig:
                return False
            rb_hand = self._normalize_hand_label(rb.get("label"))
            new_hand = self._normalize_hand_label(new_box.get("label"))
            selected_hand = self._normalize_hand_label(selected_snapshot.get("label"))
            if rb_hand or new_hand or selected_hand:
                expected_hand = new_hand or selected_hand
                if not expected_hand or rb_hand != expected_hand:
                    return False
            elif selected_snapshot:
                same_selected_id = selected_snapshot.get("id") == rb.get("id")
                same_selected_frame = True
                selected_orig = selected_snapshot.get("orig_frame")
                if selected_orig is not None:
                    try:
                        same_selected_frame = int(selected_orig) == int(rb.get("orig_frame"))
                    except Exception:
                        same_selected_frame = False
                if same_selected_id and same_selected_frame and not _coords_close_to(rb, selected_snapshot):
                    return False
            elif action == "delete" and not _coords_close_to(rb, new_box):
                return False
            return True

        # Explicit human edits override legacy locks; locks still protect automated updates.

        if action == "delete":
            if any(_matches(rb) for rb in self.raw_boxes):
                self._push_undo()
                deleted_hand = self._normalize_hand_label(new_box.get("label"))
                if deleted_hand and target_orig is not None:
                    suppressed = list(getattr(self, "_suppressed_hand_boxes", []))
                    key = f"{deleted_hand}:{int(target_orig) + int(self.start_offset)}"
                    if key not in suppressed:
                        suppressed.append(key)
                    self._suppressed_hand_boxes = suppressed

                self.raw_boxes = [rb for rb in self.raw_boxes if not _matches(rb)]
                if self._same_box_identity(getattr(self, "_selected_edit_box", None), new_box):
                    self._selected_edit_box = None
                self._rebuild_bboxes_from_raw()
                self._bump_bbox_revision()
                self._bump_query_state_revision()
                self._refresh_boxes_for_frame(self.player.current_frame)
                self._log(
                    "hoi_box_delete", box_id=box_id, frame=self.player.current_frame
                )
            return

        label_txt = new_box.get("label")
        new_class_id = new_box.get("class_id", None)
        if label_txt is not None:
            lbl_res, cid_res, _ = self._resolve_label_and_id(label_txt)
            if lbl_res:
                label_txt = lbl_res
            if cid_res is not None:
                new_class_id = cid_res
        is_hand_box = bool(self._is_hand_label(label_txt))
        if is_hand_box:
            new_class_id = None
        resolved_object_id = None if is_hand_box else self._default_object_id_for_label(label_txt)

        if action == "add":
            draw_role = getattr(self, "_pending_draw_box_role", None)
            self._pending_draw_box_role = None
            if not label_txt:
                return

            hand_data = self._selected_hand_data() or {}
            obj_id = None
            if not is_hand_box and draw_role == "noun":
                obj_id = self._hand_noun_object_id(hand_data)
            elif not is_hand_box and draw_role == "instrument":
                obj_id = self._hand_instrument_object_id(hand_data)
            if not is_hand_box and obj_id is None:
                explicit_id = new_box.get("object_id")
                if explicit_id is None:
                    raw_label = str(new_box.get("label") or "").strip()
                    id_match = re.match(r"^(?:\[(\d+)\]|(\d+))$", raw_label)
                    if id_match:
                        explicit_id = int(id_match.group(1) or id_match.group(2))
                if explicit_id is not None and self._object_name_for_id(explicit_id, default_for_none="", fallback=""):
                    obj_id = int(explicit_id)
            # A manually entered class name creates a new instance by default.

            if target_orig is None:
                try:
                    target_orig = int(self.player.current_frame) - int(
                        self.start_offset
                    )
                except Exception:
                    target_orig = 0

            is_new_object = False
            if obj_id is None:
                if is_hand_box:
                    obj_id = self.box_id_counter
                    self.box_id_counter += 1
                else:
                    obj_id = self.object_id_counter
                    is_new_object = True

            self._push_undo()
            if (not is_hand_box) and new_class_id is not None and label_txt:
                if (
                    new_class_id not in self.class_map
                    and str(new_class_id) not in self.class_map
                ):
                    self.class_map[new_class_id] = label_txt
            if is_new_object and not is_hand_box:
                self.object_id_counter += 1
                self._register_object_entry(obj_id, label_txt)

            new_rb = {
                "id": obj_id,
                "orig_frame": target_orig,
                "label": label_txt,
                "source": "manual_box_add",
                "locked": False,
                "x1": new_box.get("x1"),
                "y1": new_box.get("y1"),
                "x2": new_box.get("x2"),
                "y2": new_box.get("y2"),
            }
            if new_class_id is not None:
                new_rb["class_id"] = new_class_id

            if is_hand_box:
                key = f"{self._normalize_hand_label(label_txt)}:{int(target_orig) + int(self.start_offset)}"
                self._suppressed_hand_boxes = [k for k in getattr(self, "_suppressed_hand_boxes", []) if k != key]
            self.raw_boxes.append(new_rb)
            self._selected_edit_box = dict(new_rb)
            self._rebuild_bboxes_from_raw()
            self._bump_bbox_revision()
            self._bump_query_state_revision()
            self._refresh_boxes_for_frame(self.player.current_frame)
            self._log(
                "hoi_box_add",
                box_id=obj_id,
                class_id=new_class_id,
                label=label_txt,
                frame=self.player.current_frame,
            )
            if not is_hand_box:
                self._assign_drawn_object_to_current_hand(
                    obj_id,
                    draw_role,
                    best_bbox=new_rb,
                )
            return

        changed = False
        if any(_matches(rb) for rb in self.raw_boxes):
            self._push_undo()
        if (not is_hand_box) and new_class_id is not None and label_txt:
            if (
                new_class_id not in self.class_map
                and str(new_class_id) not in self.class_map
            ):
                self.class_map[new_class_id] = label_txt
        for rb in self.raw_boxes:
            if _matches(rb):
                for key in ("x1", "y1", "x2", "y2"):
                    if key in new_box and new_box[key] is not None:
                        rb[key] = new_box[key]
                if label_txt is not None:
                    rb["label"] = label_txt
                if is_hand_box:
                    rb.pop("class_id", None)
                elif new_class_id is not None:
                    rb["class_id"] = new_class_id
                # Keep instance ID during geometry/category edits.
                # Explicit Change ID / Propagate is the only reassignment operation.
                rb["source"] = "manual_box_edit"
                self._selected_edit_box = dict(rb)
                changed = True
        if changed:
            self._rebuild_bboxes_from_raw()
            self._bump_bbox_revision()
            self._bump_query_state_revision()
            self._refresh_boxes_for_frame(self.player.current_frame)
            self._sync_selected_hand_noun_after_box_relabel(
                previous_box_id,
                self._selected_edit_box,
                source="manual_box_edit",
            )
            self._log(
                "hoi_box_edit",
                box_id=box_id,
                class_id=new_class_id,
                label=label_txt,
                frame=self.player.current_frame,
            )

    def _resolve_label_and_id(self, text: str):
        """
        Resolve user-entered box text to (label, class_id, known_flag).
        Numeric input is interpreted as noun/object id first, then falls back to
        detector class id for compatibility.
        """
        if text is None:
            return "", None, False
        stripped = str(text).strip()
        normalized_hand = self._normalize_hand_label(stripped)
        if normalized_hand:
            return normalized_hand, None, True
        bracket_match = re.match(r"^\[(\d+)\]\s*(.+)?$", stripped)
        if bracket_match:
            stripped = bracket_match.group(1).strip()
        try:
            noun_id = int(stripped)
        except Exception:
            noun_id = None
        if noun_id is not None:
            noun_label = self._object_name_for_id(noun_id, default_for_none="", fallback="")
            if noun_label:
                return noun_label, self._class_id_for_label(noun_label), True
        lowered = self._norm_category(stripped)
        if lowered:
            for obj_name, uid in self.global_object_map.items():
                if self._norm_category(obj_name) == lowered:
                    return obj_name, self._class_id_for_label(obj_name), True
        try:
            cid = int(stripped)
            if cid in self.class_map:
                return self.class_map[cid], cid, True
            if str(cid) in self.class_map:
                return self.class_map[str(cid)], cid, True
            for rb in self.raw_boxes:
                if rb.get("class_id") == cid and rb.get("label"):
                    return str(rb.get("label")), cid, True
            return stripped, cid, False
        except Exception:
            pass
        lowered = stripped.lower()
        for k, v in self.class_map.items():
            try:
                if str(v).lower() == lowered:
                    try:
                        cid_val = int(k)
                    except Exception:
                        cid_val = k
                    return v, cid_val, True
            except Exception:
                continue
        for rb in self.raw_boxes:
            if str(rb.get("label", "")).lower() == lowered:
                return rb.get("label"), rb.get("class_id"), True
        return stripped, None, False

    def _normalize_optional_object_id(self, value: Any) -> Optional[Any]:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return value

    def _is_empty_combo_text(self, text: Any) -> bool:
        stripped = str(text or "").strip()
        if not stripped:
            return True
        return stripped.lower() in {
            "none",
            "choose noun...",
            "choose instrument...",
            "search instrument / tool...",
        }

    def _object_id_from_combo_text(self, text: Any) -> Optional[Any]:
        if self._is_empty_combo_text(text):
            return None
        stripped = str(text or "").strip()
        bracket_match = re.match(r"^\[(\d+)\]", stripped)
        if bracket_match:
            try:
                object_id = int(bracket_match.group(1))
            except Exception:
                object_id = None
            if object_id is not None and self._object_name_for_id(
                object_id,
                default_for_none="",
                fallback="",
            ):
                return object_id
        label, _class_id, _known = self._resolve_label_and_id(stripped)
        normalized_label = self._norm_category(label)
        if not normalized_label:
            return None
        for obj_name, object_id in self.global_object_map.items():
            if self._norm_category(obj_name) == normalized_label:
                return self._normalize_optional_object_id(object_id)
        return None

    def _combo_object_id(
        self,
        combo: Optional[QComboBox],
        *,
        fallback: Optional[Any] = None,
    ) -> Optional[Any]:
        if not isinstance(combo, QComboBox):
            return fallback
        text = str(combo.currentText() or "").strip()
        if self._is_empty_combo_text(text):
            return None
        current_index = combo.currentIndex()
        selected_text = ""
        if current_index >= 0:
            selected_text = str(combo.itemText(current_index) or "").strip()
        data = combo.currentData()
        if data is not None and text == selected_text:
            return self._normalize_optional_object_id(data)
        resolved_id = self._object_id_from_combo_text(text)
        if resolved_id is not None:
            return resolved_id
        return fallback

    def _class_id_for_label(self, label: str):
        if not label:
            return None
        target = str(label).lower().strip()
        for key, val in self.class_map.items():
            try:
                if str(val).lower().strip() == target:
                    try:
                        return int(key)
                    except Exception:
                        return key
            except Exception:
                continue
        return None

    def _class_id_for_object(self, uid, label=None):
        for rb in self.raw_boxes:
            if rb.get("id") == uid and rb.get("label") not in (
                self.actors_config[0]["id"],
                self.actors_config[1]["id"] if len(self.actors_config) > 1 else self.actors_config[0]["id"],
            ):
                cid = rb.get("class_id")
                if cid is not None:
                    return cid
        if label:
            return self._class_id_for_label(label)
        return None

    def _class_color(self, class_id):
        if class_id is None:
            return "#00ffff"
        try:
            idx = abs(int(class_id))
        except Exception:
            idx = abs(hash(str(class_id)))
        color_key = self._color_for_index(idx)
        return color_from_key(color_key).name()

    def _object_name_for_id(
        self, uid, default_for_none: str = "_", fallback: str = None
    ) -> str:
        if uid is None:
            return default_for_none
        for name, id_val in self.global_object_map.items():
            if id_val == uid:
                return name
        if fallback is not None:
            return fallback
        return str(uid)

    def _object_label_is_selectable(self, label):
        category = re.sub(r"_\d+$", "", str(label or "").strip())
        return category.casefold() != 'hand' and not self._normalize_hand_label(category)

    def _register_object_entry(self, uid, label: str):
        """Ensure a new object id is represented in the registry and noun combo."""
        if uid is None:
            return
        for name, id_val in self.global_object_map.items():
            if id_val == uid:
                return
        base = (label or f"obj_{uid}").strip()
        if not base:
            base = f"obj_{uid}"
        name = base
        if name in self.global_object_map:
            idx = 1
            while f"{base}_{idx}" in self.global_object_map:
                idx += 1
            name = f"{base}_{idx}"
        self.global_object_map[name] = uid
        self.id_to_category[name] = base
        display_text = f"[{uid}] {name}"
        if self._object_label_is_selectable(name) and self.combo_target.findData(uid) == -1:
            self.combo_target.addItem(display_text, uid)
        inst_combo = getattr(self, "combo_instrument", None)
        if isinstance(inst_combo, QComboBox) and self._object_label_is_selectable(name) and inst_combo.findData(uid) == -1:
            inst_combo.addItem(display_text, uid)
        decorate_combo(self.combo_target, "noun", self.global_object_map)
        decorate_combo(inst_combo, "noun", self.global_object_map)
        if uid >= self.object_id_counter:
            self.object_id_counter = uid + 1

    def _rebuild_object_combos(self, tgt_selected=None, inst_selected=None):
        """Rebuild the noun/object combo from the current object library."""
        combo_specs = [
            (getattr(self, "combo_target", None), tgt_selected),
            (getattr(self, "combo_instrument", None), inst_selected),
        ]
        for combo, selected in combo_specs:
            if not isinstance(combo, QComboBox):
                continue
            was_blocked = combo.blockSignals(True)
            try:
                combo.clear()
                empty_text = (
                    ""
                    if combo is getattr(self, "combo_instrument", None)
                    else "None"
                )
                combo.addItem(empty_text, None)
                for name, uid in sorted(self.global_object_map.items(), key=lambda x: x[1]):
                    if not self._object_label_is_selectable(name):
                        continue
                    display_text = f"[{uid}] {name}"
                    combo.addItem(display_text, uid)
                if selected is not None:
                    idx = combo.findData(selected)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
                elif combo is getattr(self, "combo_instrument", None):
                    combo.setCurrentIndex(0)
                    combo.setEditText("")
            finally:
                combo.blockSignals(was_blocked)
        self._update_inline_event_editor()

    def _label_suggestions(self):
        suggestions = []
        for actor in list(getattr(self, "actors_config", []) or []):
            actor_id = str((actor or {}).get("id") or "").strip()
            actor_label = str((actor or {}).get("label") or "").strip()
            actor_short = str((actor or {}).get("short") or "").strip()
            for item in (actor_id, actor_label, actor_short):
                if item:
                    suggestions.append(item)
        for name, uid in sorted((getattr(self, "global_object_map", {}) or {}).items(), key=lambda x: x[1]):
            if name:
                suggestions.append(str(name))
        for value in (getattr(self, "class_map", {}) or {}).values():
            text = str(value or "").strip()
            if text:
                suggestions.append(text)
        for rb in list(getattr(self, "raw_boxes", []) or []):
            text = str((rb or {}).get("label") or "").strip()
            if text:
                suggestions.append(text)
        seen = set()
        ordered = []
        for item in suggestions:
            key = str(item or "").strip().lower()
            if item and key not in seen:
                seen.add(key)
                ordered.append(item)
        return ordered

    def showEvent(self, event):
        super().showEvent(event)
        if bool(getattr(self, "_quick_start_auto_open_pending", False)):
            QTimer.singleShot(180, self._maybe_open_quick_start_on_startup)

    def closeEvent(self, e):
        if not self._confirm_close_request(prompt_parent=self):
            e.ignore()
            return
        try:
            self._stop_autosave()
        except Exception as exc:
            self._close_request_approved = False
            QMessageBox.warning(self, 'Save still pending', str(exc))
            e.ignore()
            return
        self._finalize_close_request()
        return super().closeEvent(e)

    def _rebuild_bboxes_from_raw(self):
        self.bboxes = {}
        for b in self.raw_boxes:
            target_frame = b.get("orig_frame", 0) + self.start_offset
            if self.end_frame is not None and target_frame > self.end_frame:
                continue
            mapped = dict(b)
            mapped["frame"] = target_frame
            self.bboxes.setdefault(target_frame, []).append(mapped)

    def _load_hands_xml(self):
        """[Fix] Load hand bbox XML with Auto-Alignment for clipped videos."""
        if not self._guard_experiment_mode("detection"):
            return
        fp, _ = QFileDialog.getOpenFileName(
            self, "Load hand bbox XML", "", "XML Files (*.xml);;All Files (*)"
        )
        if not fp:
            return

        if not self.player.cap:
            QMessageBox.information(self, "Info", "Please load a video first.")
            return

        try:
            parsed = self._parse_cvat_xml(fp)

            if not parsed:
                QMessageBox.warning(self, "Parse Error", "No valid hand boxes found.")
                return

            min_xml_frame = min(p["orig_frame"] for p in parsed)
            max_xml_frame = max(p["orig_frame"] for p in parsed)
            video_frame_count = self.player.frame_count
            offset_needed = 0

            if min_xml_frame > 0:
                msg = (
                    f"XML data starts at frame {min_xml_frame} (End: {max_xml_frame}).\n"
                    f"Video length is {video_frame_count} frames.\n\n"
                    f"Do you want to AUTO-ALIGN the data?\n"
                    f"(Shift frames by -{min_xml_frame}, so Frame {min_xml_frame} becomes Frame 0)"
                )

                reply = QMessageBox.question(
                    self, "Auto Align?", msg, QMessageBox.Yes | QMessageBox.No
                )

                if reply == QMessageBox.Yes:
                    offset_needed = min_xml_frame
                    for p in parsed:
                        p["orig_frame"] -= offset_needed

                    max_xml_frame -= offset_needed

            final_max_frame = max_xml_frame + self.start_offset

            if final_max_frame >= video_frame_count:
                QMessageBox.warning(
                    self,
                    "Warning",
                    f"Hand bbox frames exceed video length!\n\n"
                    f"Max Data Frame: {final_max_frame}\n"
                    f"Video Length: {video_frame_count}\n\n"
                    "If you chose 'No' to alignment, try loading again and verify offsets.",
                )

        except Exception as ex:
            QMessageBox.critical(self, "Error", f"Failed to parse XML:\n{ex}")
            return

        self._push_undo()

        self.raw_boxes = [
            b
            for b in self.raw_boxes
            if (not self._is_hand_label(b.get("label"))) or self._is_box_locked(b)
        ]

        self.raw_boxes.extend(parsed)
        self._rebuild_bboxes_from_raw()
        self._bump_bbox_revision()
        self._bump_query_state_revision()
        self._refresh_boxes_for_frame(self.player.current_frame)

        success_msg = (
            f"Loaded {len(parsed)} hand boxes.\nOriginal Start: {min_xml_frame}"
        )
        if offset_needed > 0:
            success_msg += f"\nAuto-Aligned: shifted by -{offset_needed} frames."

        QMessageBox.information(self, "Success", success_msg)
        self._log(
            "hoi_load_hands_xml",
            path=fp,
            count=len(parsed),
            offset_shifted=offset_needed,
        )

    def _parse_cvat_xml(self, path: str) -> List[dict]:
        """
        [Updated] Robust parser for CVAT XML.
        - Supports 'CVAT for Images' (your current format).
        - Supports 'CVAT for Video' track XML.
        - Auto-corrects labels: "left"->"Left_hand", "right"->"Right_hand".
        - Extracts frame numbers from filenames like "...-00135.jpg".
        """
        import xml.etree.ElementTree as ET
        import re

        try:
            tree = ET.parse(path)
            root = tree.getroot()
        except Exception as e:
            raise ValueError(f"Invalid XML file: {e}")

        out = []

        def normalize_label(l_str):
            if not l_str:
                return None
            l_lower = str(l_str).lower().strip()
            if l_lower in ["left", "left_hand", "l_hand", "left hand"]:
                return self.actors_config[0]["id"]
            if l_lower in ["right", "right_hand", "r_hand", "right hand"]:
                return self.actors_config[1]["id"] if len(self.actors_config) > 1 else self.actors_config[0]["id"]
            return None

        def extract_frame_from_name(name_str):
            matches = re.findall(r"\d+", name_str)
            if matches:
                return int(matches[-1])
            return 0

        images = root.findall("image")
        if images:
            for img in images:
                name = img.attrib.get("name", "")
                img_id = img.attrib.get("id")

                try:
                    frame = extract_frame_from_name(name)
                except Exception:
                    frame = int(img_id) if img_id else 0

                for box in img.findall("box"):
                    raw_label = box.attrib.get("label")
                    label = normalize_label(raw_label)

                    if label is None:
                        continue

                    try:
                        out.append(
                            {
                                "id": self.box_id_counter,
                                "orig_frame": frame,
                                "label": label,
                                "source": "hands_xml",
                                "locked": False,
                                "x1": float(box.attrib.get("xtl")),
                                "y1": float(box.attrib.get("ytl")),
                                "x2": float(box.attrib.get("xbr")),
                                "y2": float(box.attrib.get("ybr")),
                            }
                        )
                        self.box_id_counter += 1
                    except (ValueError, TypeError):
                        continue

            if out:
                return out

        tracks = root.findall("track")
        if tracks:
            for track in tracks:
                raw_label = track.attrib.get("label")
                label = normalize_label(raw_label)
                if label is None:
                    continue

                for box in track.findall("box"):
                    if box.attrib.get("outside") == "1":
                        continue
                    try:
                        out.append(
                            {
                                "id": self.box_id_counter,
                                "orig_frame": int(box.attrib.get("frame")),
                                "label": label,
                                "source": "hands_xml",
                                "locked": False,
                                "x1": float(box.attrib.get("xtl")),
                                "y1": float(box.attrib.get("ytl")),
                                "x2": float(box.attrib.get("xbr")),
                                "y2": float(box.attrib.get("ybr")),
                            }
                        )
                        self.box_id_counter += 1
                    except Exception:
                        continue

        return out

    def _validate_integrity(self, relation: dict) -> list:
        """
        Check BBox existence for Start/Onset/End frames in a Relation.
        Returns error list: [{'msg': '...', 'frame': 123}, ...]
        """
        errors = []

        check_points = {
            "Start": relation.get("interaction_start"),
            "Onset": relation.get("functional_contact_onset"),
            "End": relation.get("interaction_end"),
        }

        targets = [
            ("Noun", relation.get("noun_object_id", relation.get("target_object_id"))),
            (
                "Instrument",
                relation.get("instrument_object_id", relation.get("tool_object_id")),
            ),
        ]

        for time_label, frame in check_points.items():
            if frame is None:
                continue

            frame_boxes = self.bboxes.get(frame, [])
            existing_ids = set(b.get("id") for b in frame_boxes)

            for role, uid in targets:
                if uid is None:
                    continue

                if uid not in existing_ids:
                    name = self._object_name_for_id(
                        uid, default_for_none="", fallback=""
                    )
                    role_key = str(role or "").strip().lower()
                    time_key = str(time_label or "").strip().lower()

                    errors.append(
                        {
                            "msg": f"Frame {frame} ({time_label}) missing {role}: [{uid}] {name}",
                            "frame": frame,
                            "role": role,
                            "role_key": role_key,
                            "time_label": time_label,
                            "time_key": time_key,
                            "object_id": uid,
                            "object_name": name,
                            "slot": f"{role_key}_{time_key}",
                        }
                    )
        return errors

    def _import_hoi_ontology(self):
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Import Verb-Noun Ontology CSV",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not fp:
            return
        self._load_ontology_from_path(fp, notify_user=True)

    def _load_ontology_from_path(
        self,
        fp: str,
        *,
        notify_user: bool = True,
        auto_discovered: bool = False,
    ) -> bool:
        if not self._guard_asset_mutation(
            "verb-noun ontology",
            path=fp,
            notify_user=notify_user,
            auto_discovered=auto_discovered,
        ):
            return False
        try:
            ontology = HOIOntology.from_csv(fp)
        except Exception as ex:
            if notify_user:
                QMessageBox.critical(self, "Ontology Import", f"Failed to import ontology:\n{ex}")
            else:
                self._log("hoi_auto_load_failed", asset="ontology", path=fp, error=str(ex))
            return False
        self.hoi_ontology = ontology
        self.hoi_ontology_path = fp
        allowed_edges = sum(len(v or []) for v in ontology.relations.values())
        self._log(
            "hoi_load_ontology",
            path=fp,
            verbs=len(ontology.relations),
            allowed_edges=allowed_edges,
            format_hint=ontology.format_hint,
            auto_discovered=bool(auto_discovered),
        )
        self._log_annotation_ready_state("hoi_load_ontology")
        self._bump_query_state_revision()
        self._update_next_best_query_panel()
        self._refresh_semantic_suggestions_for_event(self.selected_event_id)
        if notify_user:
            QMessageBox.information(
                self,
                "Ontology Loaded",
                f"Loaded ontology for {len(ontology.relations)} verbs with {allowed_edges} allowed verb-noun relations.",
            )
        return True

    def _import_targets(self):
        """Import noun/object list from txt."""
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Import Noun List",
            "",
            "Text Files (*.txt)",
        )
        if not fp:
            return
        self._load_targets_from_path(fp, notify_user=True)

    def _load_targets_from_path(
        self,
        fp: str,
        *,
        notify_user: bool = True,
        auto_discovered: bool = False,
    ) -> int:
        if not self._guard_asset_mutation(
            "noun list",
            path=fp,
            notify_user=notify_user,
            auto_discovered=auto_discovered,
        ):
            return 0
        target_names = self._read_entity_names_from_txt(fp)
        self._loaded_target_names = list(target_names or [])
        self._loaded_target_norms = {
            self._norm_category(name)
            for name in list(target_names or [])
            if self._norm_category(name)
        }
        self.combo_target.clear()
        self.combo_target.addItem("None", None)
        loaded = self._load_entities_into_combo(fp, self.combo_target, notify_user=notify_user)
        self._prune_object_library_to_loaded_targets(notify_user=False)
        self._rebuild_object_combos()
        self._log("hoi_load_targets", path=fp, count=loaded, auto_discovered=bool(auto_discovered))
        self._log_annotation_ready_state("hoi_load_targets")
        return int(loaded)

    def _read_entity_names_from_txt(self, fp: str) -> List[str]:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                return [
                    self._canonical_label_name(line)
                    for line in f
                    if self._canonical_label_name(line)
                ]
        except Exception:
            return []

    def _prune_object_library_to_loaded_targets(self, *, notify_user: bool = True) -> Dict[str, int]:
        allowed_norms = {
            str(name).strip()
            for name in list(getattr(self, "_loaded_target_norms", set()) or set())
            if str(name).strip()
        }
        if not allowed_norms:
            if notify_user:
                QMessageBox.information(
                    self,
                    "Cleanup Skipped",
                    "Load the noun/object list first. Cleanup only prunes labels against the current noun list.",
                )
            return {"removed_labels": 0, "removed_boxes": 0}

        referenced_ids = {s['object_id'] for s in self._assembly_data().get('states',[])}
        for event in list(getattr(self, "events", []) or []):
            hoi_data = dict((event or {}).get("hoi_data") or {})
            for actor in list(getattr(self, "actors_config", []) or []):
                hand_key = str((actor or {}).get("id") or "").strip()
                if not hand_key:
                    continue
                hand_data = dict(hoi_data.get(hand_key) or {})
                target_id = self._hand_noun_object_id(hand_data)
                if target_id is None:
                    continue
                try:
                    referenced_ids.add(int(target_id))
                except Exception:
                    continue

        removed_names: List[str] = []
        for name, uid in list((getattr(self, "global_object_map", {}) or {}).items()):
            category = self._norm_category(self.id_to_category.get(name, name))
            if category in allowed_norms:
                continue
            try:
                uid_int = int(uid)
            except Exception:
                uid_int = None
            if uid_int is not None and uid_int in referenced_ids:
                continue
            removed_names.append(str(name))

        for name in removed_names:
            self.global_object_map.pop(name, None)
            self.id_to_category.pop(name, None)

        removed_boxes = 0
        if getattr(self, "raw_boxes", None):
            kept_boxes = []
            for rb in list(self.raw_boxes or []):
                if self._is_hand_label(rb.get("label")):
                    kept_boxes.append(rb)
                    continue
                category = self._norm_category(
                    self.id_to_category.get(rb.get("label"), rb.get("label"))
                )
                try:
                    box_uid = int(rb.get("id"))
                except Exception:
                    box_uid = None
                source_family = self._field_source_family(self._box_source(rb))
                keep_box = bool(
                    category in allowed_norms
                    or (box_uid is not None and box_uid in referenced_ids)
                    or self._is_box_locked(rb)
                    or source_family == "human_manual"
                )
                if keep_box:
                    kept_boxes.append(rb)
                else:
                    removed_boxes += 1
            if removed_boxes:
                self.raw_boxes = kept_boxes
                self._rebuild_bboxes_from_raw()
                self._bump_bbox_revision()
                self._bump_query_state_revision()
                self._refresh_boxes_for_frame(int(getattr(self.player, "current_frame", 0)))

        if removed_names:
            self._rebuild_object_combos(tgt_selected=self.combo_target.currentData())

        self._log(
            "hoi_prune_object_library",
            removed_labels=len(removed_names),
            removed_boxes=int(removed_boxes),
            referenced_ids=len(referenced_ids),
        )
        if notify_user:
            QMessageBox.information(
                self,
                "Cleanup Complete",
                (
                    f"Removed {len(removed_names)} unused object labels and "
                    f"{int(removed_boxes)} unlocked detector boxes outside the current noun list."
                ),
            )
        return {
            "removed_labels": int(len(removed_names)),
            "removed_boxes": int(removed_boxes),
        }

    def _load_entities_into_combo(self, fp, combo, *, notify_user: bool = True):
        """
        Universal entity loader: reads TXT, assigns global IDs, populates combo box.
        """
        try:
            loaded_count = 0
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    category = self._canonical_label_name(line)
                    if not category:
                        continue
                    
                    entity_name = category

                    if entity_name in self.global_object_map:
                        uid = self.global_object_map[entity_name]
                    else:
                        uid = self.object_id_counter
                        self.global_object_map[entity_name] = uid
                        self.object_id_counter += 1

                    self.id_to_category[entity_name] = category

                    display_text = f"[{uid}] {entity_name}"

                    if combo.findData(uid) == -1:
                        combo.addItem(display_text, uid)
                        loaded_count += 1

            if notify_user:
                QMessageBox.information(
                    self,
                    "Success",
                    f"Loaded {loaded_count} entities to {combo.objectName() or 'list'}",
                )
            self._update_onboarding_banner()
            return loaded_count
        except Exception as e:
            if notify_user:
                QMessageBox.critical(self, "Error", f"Failed: {e}")
            else:
                self._log("hoi_auto_load_failed", asset="entities", path=fp, error=str(e))
            return 0

    def _on_object_list_menu(self, pos):
        """Triggered when right-clicking the Objects list."""
        item = self.list_objects.itemAt(pos)
        if not item:
            return

        box_data = item.data(Qt.UserRole)
        if not box_data:
            return

        menu = QMenu(self)

        if not self._is_hand_label(box_data.get("label")):
            action_change = menu.addAction("Change ID / Propagate...")
            action_change.triggered.connect(lambda: self._dialog_change_id(item))

        if self._is_box_locked(box_data):
            action_lock = menu.addAction("Unlock Box")
            action_lock.triggered.connect(
                lambda: self._set_box_lock_state(
                    box_data,
                    locked=False,
                    frame=int(getattr(self.player, "current_frame", 0)),
                )
            )
        else:
            action_lock = menu.addAction("Lock Box")
            action_lock.triggered.connect(
                lambda: self._set_box_lock_state(
                    box_data,
                    locked=True,
                    frame=int(getattr(self.player, "current_frame", 0)),
                )
            )

        action_delete = menu.addAction("Delete Box")
        action_delete.triggered.connect(lambda: self._delete_box(item))

        menu.exec_(self.list_objects.mapToGlobal(pos))

    def _dialog_change_id(self, item):
        """[Fix] Dialog to change ID to ANY target (removes category restriction)."""
        box = item.data(Qt.UserRole)
        if self._box_locked_for_action(box, frame=int(getattr(self.player, "current_frame", 0))):
            QMessageBox.information(
                self,
                "Box Locked",
                "Unlock this box before changing its ID.",
            )
            return
        current_uid = box.get("id")
        if any(s['object_id']==current_uid for s in self._assembly_data().get('states',[])):
            QMessageBox.information(self,'Shared object ID','This ID is referenced by the shared timeline. Select a different shared object in Shared assembly instead of renaming its track ID.')
            return

        curr_name = self._object_name_for_id(current_uid, fallback="Unknown")

        candidates = []
        for name, uid in self.global_object_map.items():
            candidates.append((name, uid))

        candidates.sort(key=lambda x: x[1])

        if not candidates:
            QMessageBox.warning(self, "Error", "Entity Library is empty.")
            return

        from PyQt5.QtWidgets import (
            QDialog,
            QVBoxLayout,
            QComboBox,
            QCheckBox,
            QDialogButtonBox,
        )

        d = QDialog(self)
        d.setWindowTitle(f"Change ID for {curr_name}")
        d.resize(400, 150)
        layout = QVBoxLayout(d)

        layout.addWidget(QLabel(f"Current: [{current_uid}] {curr_name}"))
        layout.addWidget(QLabel("Change to:"))

        combo = QComboBox()
        current_idx = 0
        for i, (name, uid) in enumerate(candidates):
            combo.addItem(f"[{uid}] {name}", uid)
            if uid == current_uid:
                current_idx = i
        combo.setCurrentIndex(current_idx)
        layout.addWidget(combo)

        chk_propagate = QCheckBox("Apply to subsequent frames (Auto-Track)")
        chk_propagate.setChecked(True)
        chk_propagate.setToolTip(
            "Use IoU overlap to apply changes to subsequent frames."
        )
        layout.addWidget(chk_propagate)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(d.accept)
        btns.rejected.connect(d.reject)
        layout.addWidget(btns)

        if d.exec_() == QDialog.Accepted:
            new_uid = combo.currentData()
            if new_uid == current_uid:
                return

            new_label = next(
                (n for n, u in self.global_object_map.items() if u == new_uid), ""
            )

            self._push_undo()

            self._update_raw_box_id(
                self.player.current_frame, box, current_uid, new_uid, new_label
            )

            box["id"] = new_uid
            box["label"] = new_label

            count = 1
            if chk_propagate.isChecked():
                count += self._propagate_id(box, new_uid, new_label)

            self._rebuild_bboxes_from_raw()
            self._bump_query_state_revision()
            self._refresh_boxes_for_frame(self.player.current_frame)
            self._sync_selected_hand_noun_after_box_relabel(
                current_uid,
                box,
                source="manual_box_change_id",
            )

            self._log(
                "hoi_box_change_id",
                old_id=current_uid,
                new_id=new_uid,
                new_label=new_label,
                propagated=bool(chk_propagate.isChecked()),
                count=count,
            )
            QMessageBox.information(
                self,
                "Updated",
                f"Updated ID to [{new_uid}] {new_label}\nApplied to {count} frames.",
            )

    def _propagate_id(self, source_box, new_uid, new_label=None):
        """
        [Fix] Propagate ID change based on IoU only (ignoring category mismatch).
        Allows correcting misclassified objects across frames.
        """
        start_frame = source_box["frame"]

        curr_frame = start_frame
        curr_box_geo = [
            source_box["x1"],
            source_box["y1"],
            source_box["x2"],
            source_box["y2"],
        ]
        modified_count = 0

        while True:
            curr_frame += 1
            if curr_frame >= self.player.frame_count:
                break

            next_boxes = self.bboxes.get(curr_frame, [])
            if not next_boxes:
                break

            best_match = None
            max_iou = 0.0

            for nb in next_boxes:
                if self._is_hand_label(nb.get("label")):
                    continue
                if self._box_locked_for_action(nb, frame=curr_frame):
                    continue

                def get_iou(boxA, boxB):
                    xA = max(boxA[0], boxB["x1"])
                    yA = max(boxA[1], boxB["y1"])
                    xB = min(boxA[2], boxB["x2"])
                    yB = min(boxA[3], boxB["y2"])
                    interArea = max(0, xB - xA) * max(0, yB - yA)
                    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                    boxBArea = (boxB["x2"] - boxB["x1"]) * (boxB["y2"] - boxB["y1"])
                    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

                iou = get_iou(curr_box_geo, nb)

                if iou > 0.3:
                    if iou > max_iou:
                        max_iou = iou
                        best_match = nb

            if best_match:
                old_id = best_match.get("id")

                best_match["id"] = new_uid
                if new_label:
                    best_match["label"] = new_label

                self._update_raw_box_id(
                    curr_frame, best_match, old_id, new_uid, new_label
                )

                curr_box_geo = [
                    best_match["x1"],
                    best_match["y1"],
                    best_match["x2"],
                    best_match["y2"],
                ]
                modified_count += 1
            else:
                break

        return modified_count

    @staticmethod
    def _coords_close(a: dict, b: dict, tol: float = 1.0) -> bool:
        for key in ("x1", "y1", "x2", "y2"):
            if abs(float(a.get(key, 0.0)) - float(b.get(key, 0.0))) > tol:
                return False
        return True

    def _update_raw_box_id(
        self, frame: int, box: dict, old_id: int, new_id: int, new_label: str = None
    ):
        """[Fix] Update raw box ID and LABEL/CLASS to keep consistency."""
        target_frame = int(frame) - self.start_offset
        for rb in self.raw_boxes:
            if rb.get("orig_frame") != target_frame:
                continue
            if old_id is not None and rb.get("id") != old_id:
                continue
            if not self._coords_close(rb, box):
                continue
            if self._is_box_locked(rb):
                continue

            rb["id"] = new_id
            if new_label:
                rb["label"] = new_label
                cid = self._class_id_for_label(new_label)
                if cid is not None:
                    rb["class_id"] = cid
            rb["source"] = "manual_box_edit"

    def _delete_box(self, item):
        """Delete specific box from the current frame."""
        box = item.data(Qt.UserRole)
        frame = self.player.current_frame

        if frame in self.bboxes and box in self.bboxes[frame]:
            self._push_undo()
            deleted_hand = self._normalize_hand_label(box.get("label"))
            if deleted_hand:
                suppressed = list(getattr(self, "_suppressed_hand_boxes", []))
                key = f"{deleted_hand}:{int(frame)}"
                if key not in suppressed:
                    suppressed.append(key)
                self._suppressed_hand_boxes = suppressed
            target_frame = int(frame) - self.start_offset
            self.raw_boxes = [
                rb
                for rb in self.raw_boxes
                if not (
                    rb.get("orig_frame") == target_frame
                    and rb.get("id") == box.get("id")
                    and rb.get("label") == box.get("label")
                    and self._coords_close(rb, box)
                )
            ]
            self._rebuild_bboxes_from_raw()
            self._bump_bbox_revision()
            self._bump_query_state_revision()
            self._refresh_boxes_for_frame(frame)
            self._log("hoi_box_delete", box_id=box.get("id"), frame=frame)

    def _save_hands_xml(self):
        """Export current hand BBoxes (including modifications) to CVAT XML format."""
        if not self.raw_boxes:
            QMessageBox.information(self, "Info", "No hand data to save.")
            return

        default_name = "hands_modified.xml"
        fp, _ = QFileDialog.getSaveFileName(
            self, "Save Hands XML", default_name, "XML Files (*.xml)"
        )
        if not fp:
            return

        try:
            import xml.etree.ElementTree as ET

            root = ET.Element("annotations")
            ET.SubElement(root, "meta")

            tracks = {actor["id"]: [] for actor in self.actors_config}

            for b in self.raw_boxes:
                lbl = b.get("label")
                if lbl in tracks:
                    tracks[lbl].append(b)

            for label, boxes in tracks.items():
                if not boxes:
                    continue

                boxes.sort(key=lambda x: x["orig_frame"])

                track_node = ET.SubElement(
                    root, "track", id=str(self.box_id_counter), label=label
                )
                self.box_id_counter += 1

                for b in boxes:
                    ET.SubElement(
                        track_node,
                        "box",
                        frame=str(b["orig_frame"]),
                        xtl=f"{b['x1']:.2f}",
                        ytl=f"{b['y1']:.2f}",
                        xbr=f"{b['x2']:.2f}",
                        ybr=f"{b['y2']:.2f}",
                        outside="0",
                        occluded="0",
                        keyframe="1",
                    )

            tree = ET.ElementTree(root)
            tree.write(fp, encoding="utf-8", xml_declaration=True)
            self._log("hoi_save_hands_xml", path=fp, count=len(self.raw_boxes))
            QMessageBox.information(self, "Saved", f"Hand XML saved to:\n{fp}")

        except Exception as ex:
            QMessageBox.critical(self, "Error", f"Failed to save XML:\n{ex}")

    def _reset_event_draft(self):
        """[Step 1] Initialize/reset event draft structure."""
        self.event_draft = {}
        for actor in self.actors_config:
            self.event_draft[actor["id"]] = self._blank_hand_data()
            uid = self._assembly_default_id(self._assembly_frame())
            if uid is not None:
                self._set_assembly_reference(self.event_draft[actor['id']],uid)
        if hasattr(self, "lbl_event_status"):
            self.lbl_event_status.setText("No event selected.")

    def _save_ui_to_hand_draft(self, hand_label: str):
        """[Restore] Save UI to draft using COMBO BOX."""
        if not hand_label or hand_label not in self.event_draft:
            return
        hand_data = self.event_draft[hand_label]
        self._ensure_hand_annotation_state(hand_data)
        existing_verb = str(hand_data.get("verb") or "").strip()
        existing_target_id = self._hand_noun_object_id(hand_data)
        existing_instrument_id = self._hand_instrument_object_id(hand_data)
        existing_anomaly_label = self._normalize_anomaly_label(
            hand_data.get("anomaly_label")
        )

        main_verb = str(
            getattr(getattr(self, "combo_verb", None), "currentText", lambda: "")() or ""
        ).strip()
        inline_verb = str(
            getattr(getattr(self, "combo_inline_verb", None), "currentText", lambda: "")() or ""
        ).strip()
        if main_verb:
            verb = main_verb
        elif inline_verb and inline_verb != "Choose verb...":
            verb = inline_verb
        else:
            verb = existing_verb

        try:
            main_target_id = getattr(getattr(self, "combo_target", None), "currentData", lambda: None)()
        except Exception:
            main_target_id = None
        try:
            inline_target_id = getattr(
                getattr(self, "combo_inline_noun", None), "currentData", lambda: None
            )()
        except Exception:
            inline_target_id = None
        if main_target_id is not None:
            target_id = main_target_id
        elif inline_target_id is not None:
            target_id = inline_target_id
        else:
            target_id = existing_target_id

        instrument_combo = getattr(self, "combo_instrument", None)
        inline_instrument_combo = getattr(self, "combo_inline_instrument", None)
        try:
            main_instrument_id = self._combo_object_id(
                instrument_combo,
                fallback=existing_instrument_id,
            )
        except Exception:
            main_instrument_id = existing_instrument_id
        try:
            inline_instrument_id = self._combo_object_id(
                inline_instrument_combo,
                fallback=existing_instrument_id,
            )
        except Exception:
            inline_instrument_id = existing_instrument_id
        inline_instrument_active = False
        if isinstance(inline_instrument_combo, QComboBox):
            inline_instrument_active = bool(inline_instrument_combo.hasFocus())
            try:
                editor = inline_instrument_combo.lineEdit()
                inline_instrument_active = inline_instrument_active or bool(
                    editor is not None and editor.hasFocus()
                )
            except Exception:
                pass
        if inline_instrument_active:
            instrument_id = inline_instrument_id
        elif isinstance(instrument_combo, QComboBox):
            instrument_id = main_instrument_id
        elif isinstance(inline_instrument_combo, QComboBox):
            instrument_id = inline_instrument_id
        else:
            instrument_id = existing_instrument_id

        anomaly_combo = getattr(self, "combo_anomaly", None)
        inline_anomaly_combo = getattr(self, "combo_inline_anomaly", None)
        try:
            main_anomaly_label = (
                anomaly_combo.currentText()
                if isinstance(anomaly_combo, QComboBox)
                else existing_anomaly_label
            )
        except Exception:
            main_anomaly_label = existing_anomaly_label
        try:
            inline_anomaly_label = (
                inline_anomaly_combo.currentText()
                if isinstance(inline_anomaly_combo, QComboBox)
                else existing_anomaly_label
            )
        except Exception:
            inline_anomaly_label = existing_anomaly_label
        inline_anomaly_active = False
        if isinstance(inline_anomaly_combo, QComboBox):
            inline_anomaly_active = bool(inline_anomaly_combo.hasFocus())
            try:
                editor = inline_anomaly_combo.lineEdit()
                inline_anomaly_active = inline_anomaly_active or bool(
                    editor is not None and editor.hasFocus()
                )
            except Exception:
                pass
        if inline_anomaly_active:
            anomaly_label = self._remember_anomaly_label(inline_anomaly_label)
        elif isinstance(anomaly_combo, QComboBox):
            anomaly_label = self._remember_anomaly_label(main_anomaly_label)
        else:
            anomaly_label = self._remember_anomaly_label(inline_anomaly_label)

        if verb.casefold()=="hold" and (not existing_verb.strip() or hand_data.get("functional_contact_onset") is None) and not onset_resolved(hand_data) and not hand_data.get("_onset_manually_adjusted") and type(hand_data.get("interaction_start")) is int:
            hand_data["functional_contact_onset"]=hand_data["interaction_start"]
        hand_data["verb"] = verb
        if hand_data.get('shared_assembly_ref'):
            target_id = reference_id(hand_data,self._assembly_data(),self._assembly_frame())
        hand_data["target_object_id"] = target_id
        hand_data["noun_object_id"] = target_id
        hand_data["instrument_object_id"] = instrument_id
        hand_data["tool_object_id"] = instrument_id
        hand_data["anomaly_label"] = anomaly_label
        hand_data["has_anomaly"] = self._is_anomalous_label(anomaly_label)

    def _rebuild_actor_checkboxes(self):
        # Clear old
        while self.actor_layout.count():
            item = self.actor_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
                item.widget().deleteLater()
        self.actor_controls.clear()

        # Re-add
        for actor in self.actors_config:
            aid = actor["id"]
            short = actor.get("short", aid[:1])
            chk = QCheckBox(short)
            chk.setToolTip(actor.get("label", aid))
            chk.toggled.connect(lambda on, a=aid: self._select_hand(a, on))
            self.actor_layout.addWidget(chk)
            self.actor_controls[aid] = chk

    def _on_configure_actors(self):
        """Show dialog to manage actors."""
        actions = ["Add Actor", "Remove Actor", "Rename Actor"]
        action, ok = QInputDialog.getItem(
            self, "Configure Actors", "Select action:", actions, 0, False
        )
        if not ok:
            return

        if action == "Add Actor":
            name, ok = QInputDialog.getText(self, "Add Actor", "Actor ID (e.g. hand_3):")
            if ok and name:
                self.actors_config.append(
                    {
                        "id": name,
                        "label": name.replace("_", " ").title(),
                        "short": name[:1].upper(),
                    }
                )
                self._reinit_actor_system()
        elif action == "Remove Actor":
            ids = [a["id"] for a in self.actors_config]
            name, ok = QInputDialog.getItem(
                self, "Remove Actor", "Select actor to remove:", ids, 0, False
            )
            if ok and len(self.actors_config) > 1:
                self.actors_config = [a for a in self.actors_config if a["id"] != name]
                self._reinit_actor_system()
        elif action == "Rename Actor":
            ids = [a["id"] for a in self.actors_config]
            old_name, ok = QInputDialog.getItem(
                self, "Rename Actor", "Select actor to rename:", ids, 0, False
            )
            if ok:
                new_name, ok = QInputDialog.getText(
                    self, "Rename Actor", f"New ID for {old_name}:", QLineEdit.Normal, old_name
                )
                if ok and new_name and new_name != old_name:
                    for a in self.actors_config:
                        if a["id"] == old_name:
                            a["id"] = new_name
                            a["label"] = new_name.replace("_", " ").title()
                            a["short"] = new_name[:1].upper()
                    self._reinit_actor_system()

    def _reinit_actor_system(self):
        """Update UI and Timeline after actor config change."""
        actor_ids = [actor["id"] for actor in self.actors_config]
        self.current_hands = {actor_id: None for actor_id in actor_ids}
        if self.selected_hand_label not in actor_ids:
            self.selected_hand_label = None
        self._rebuild_actor_checkboxes()
        self._reset_event_draft()
        if hasattr(self, "hoi_timeline"):
            self.hoi_timeline.actors_config = self.actors_config
            self.hoi_timeline._reinit_rows()
            self.hoi_timeline.set_selected(self.selected_event_id, self.selected_hand_label)
            self.hoi_timeline.refresh()
        self._update_status_label()

    def _swap_frame_hands(self):
        """Swap boxes for the first two actors on current frame."""
        if len(self.actors_config) < 2:
            return

        a1 = self.actors_config[0]["id"]
        a2 = self.actors_config[1]["id"]

        if not self.player.cap:
            QMessageBox.information(
                self, "Info", "Load a video before swapping frame hands."
            )
            return
        frame_idx = int(self.player.current_frame)
        target_orig = frame_idx - int(self.start_offset)

        actor_ids = [a1, a2]
        hand_boxes = [
            rb
            for rb in self.raw_boxes
            if rb.get("orig_frame") == target_orig
            and rb.get("label") in actor_ids
        ]
        if not hand_boxes:
            QMessageBox.information(
                self, "Info", f"No boxes for {a1}/{a2} found on the current frame."
            )
            return

        self._push_undo()
        for rb in self.raw_boxes:
            if rb.get("orig_frame") != target_orig:
                continue
            if self._is_box_locked(rb):
                continue
            lbl = rb.get("label")
            if lbl == a1:
                rb["label"] = a2
            elif lbl == a2:
                rb["label"] = a1

        self._rebuild_bboxes_from_raw()
        self._bump_bbox_revision()
        self._refresh_boxes_for_frame(frame_idx)
        self._log("hoi_frame_swap_hands", frame=frame_idx)
        QMessageBox.information(
            self, "Swapped", f"Swapped {a1}/{a2} boxes on this frame."
        )

    def _load_hand_draft_to_ui(self, hand_label: str):
        """[Restore] Load draft to UI using COMBO BOX."""
        if not hand_label or hand_label not in self.event_draft:
            return

        hand_data = self.event_draft[hand_label]
        self._ensure_hand_annotation_state(hand_data)

        self.combo_target.setEnabled(not hand_data.get('shared_assembly_ref',False))
        if getattr(self,'combo_inline_noun',None) is not None:
            self.combo_inline_noun.setEnabled(not hand_data.get('shared_assembly_ref',False))
        controls = [getattr(self, name, None) for name in
                    ('combo_verb', 'combo_target', 'combo_instrument', 'combo_anomaly')]
        blocked = [(control, control.blockSignals(True)) for control in controls if control is not None]

        try:
            verb = hand_data.get("verb", "")

            idx = self.combo_verb.findText(verb)
            if idx >= 0:
                self.combo_verb.setCurrentIndex(idx)
            else:
                self.combo_verb.setCurrentText(verb)

            target_id = self._hand_noun_object_id(hand_data)
            idx_tar = self.combo_target.findData(target_id)
            if idx_tar >= 0:
                self.combo_target.setCurrentIndex(idx_tar)
            else:
                self.combo_target.setCurrentIndex(0)

            instrument_id = self._hand_instrument_object_id(hand_data)
            if getattr(self, "combo_instrument", None) is not None:
                idx_inst = self.combo_instrument.findData(instrument_id)
                if idx_inst >= 0:
                    self.combo_instrument.setCurrentIndex(idx_inst)
                else:
                    self.combo_instrument.setCurrentIndex(0)
                if instrument_id is None:
                    self.combo_instrument.setEditText("")

            anomaly_label = self._normalize_anomaly_label(hand_data.get("anomaly_label"))
            if getattr(self, "combo_anomaly", None) is not None:
                self._populate_anomaly_combo(self.combo_anomaly, anomaly_label)

        finally:
            for control, previous in blocked:
                control.blockSignals(previous)

        self._sync_action_panel_selection(hand_data.get("verb", ""))
        self._update_status_label()

    def _update_status_label(self):
        """Refresh status bar showing the selected HOI event and hand details."""
        if not hasattr(self, "lbl_event_status"):
            return
        self._update_inspector_tab_labels()
        if self.selected_event_id is None:
            self.lbl_event_title.setText("No event selected")
            self.lbl_event_frames.setText("Frames: -")
            self.lbl_event_meta.setText("Verb: -   Noun: -   Instrument: -   Anomaly: -")
            self.lbl_event_status.setText("No HOI event selected.")
            if hasattr(self, "lbl_hand_support_status"):
                self.lbl_hand_support_status.setVisible(not self._manual_mode_enabled())
                self.lbl_hand_support_status.setText("Hand support: load and select an event to inspect hand boxes.")
                self.lbl_hand_support_status.setToolTip("")
            self._set_status_chip(getattr(self, "lbl_event_actor_chip", None), "Idle", "neutral")
            self._set_status_chip(getattr(self, "lbl_event_health_chip", None), "Waiting", "neutral")
            self._set_status_card_tone(getattr(self, "event_status_card", None), "neutral")
            for btn in (
                getattr(self, "btn_jump_start_chip", None),
                getattr(self, "btn_jump_onset_chip", None),
                getattr(self, "btn_jump_end_chip", None),
            ):
                if btn is not None:
                    btn.setEnabled(False)
            self._update_onboarding_banner()
            self._update_inline_event_editor()
            return

        hand = self.selected_hand_label
        if hand:
            h_data = self.event_draft.get(hand, {})
            self._ensure_hand_annotation_state(h_data)
            manual_mode = self._manual_mode_enabled()
            s = h_data.get("interaction_start")
            o = h_data.get("functional_contact_onset")
            e = h_data.get("interaction_end")
            s_txt = str(s) if s is not None else "-"
            o_txt = str(o) if o is not None else "-"
            e_txt = str(e) if e is not None else "-"
            verb = h_data.get("verb") or "-"
            target_id = self._hand_noun_object_id(h_data)
            target_name = "-"
            for name, id_val in self.global_object_map.items():
                if target_id is not None and id_val == target_id:
                    target_name = name
            instrument_id = self._hand_instrument_object_id(h_data)
            instrument_name = "-"
            for name, id_val in self.global_object_map.items():
                if instrument_id is not None and id_val == instrument_id:
                    instrument_name = name
            anomaly_label = self._normalize_anomaly_label(h_data.get("anomaly_label"))

            actor_label = self._get_actor_full_label(hand)
            missing = []
            completion_state = self._hand_completion_state(h_data, hand_key=hand)
            missing = list(completion_state.get("missing") or [])
            evidence_summary = self._sparse_evidence_summary(h_data)
            evidence_expected = int(evidence_summary.get("expected", 0) or 0)
            evidence_confirmed = int(evidence_summary.get("confirmed", 0) or 0)
            evidence_missing = int(evidence_summary.get("missing", 0) or 0)
            suggested_fields = list(completion_state.get("suggested_fields") or [])
            if missing:
                tone = "warn"
                health_text = f"Missing {len(missing)}"
            elif suggested_fields:
                tone = "warn"
                health_text = (
                    f"Pending {len(suggested_fields)}"
                    if manual_mode
                    else f"Review {len(suggested_fields)}"
                )
            else:
                tone = "ok"
                health_text = "Complete"

            self.lbl_event_title.setText(
                f"Event {self.selected_event_id}" + ("  Completed" if not missing and not suggested_fields else "")
            )
            self.lbl_event_frames.setText(f"Frames  Start {s_txt}   Onset {o_txt}   End {e_txt}")
            self.lbl_event_meta.setText(
                f"Verb: {verb}   Noun: {target_name}   "
                f"Instrument: {instrument_name}   Anomaly: {anomaly_label}"
            )
            if missing:
                status_text = (
                    "Fill: " + " -> ".join(missing)
                    if manual_mode
                    else "Missing: " + ", ".join(missing)
                )
            elif suggested_fields:
                labels = [self._display_field_label(name) for name in suggested_fields]
                status_text = (
                    "Pending manual check: " + ", ".join(labels)
                    if manual_mode
                    else "Suggested, needs confirmation: " + ", ".join(labels)
                )
            else:
                if evidence_expected > 0:
                    status_text = (
                        f"Event completed. Evidence {evidence_confirmed}/{evidence_expected} grounded."
                        if manual_mode
                        else f"Event completed and ready for review. Sparse evidence {evidence_confirmed}/{evidence_expected} grounded."
                    )
                else:
                    status_text = (
                        "Event completed."
                        if manual_mode
                        else "Event completed and ready for review."
                    )
            self.lbl_event_status.setText(status_text)
            support_text, support_tooltip = self._selected_hand_support_status(hand)
            if hasattr(self, "lbl_hand_support_status"):
                self.lbl_hand_support_status.setVisible(not self._manual_mode_enabled())
                self.lbl_hand_support_status.setText(support_text)
                self.lbl_hand_support_status.setToolTip(support_tooltip)
            self._set_status_chip(getattr(self, "lbl_event_actor_chip", None), actor_label, "neutral")
            self._set_status_chip(getattr(self, "lbl_event_health_chip", None), health_text, tone)
            self._set_status_card_tone(
                getattr(self, "event_status_card", None),
                "active" if not missing and not suggested_fields else "warn",
            )
            if getattr(self, "btn_jump_start_chip", None):
                self.btn_jump_start_chip.setEnabled(s is not None)
            if getattr(self, "btn_jump_onset_chip", None):
                self.btn_jump_onset_chip.setEnabled(o is not None)
            if getattr(self, "btn_jump_end_chip", None):
                self.btn_jump_end_chip.setEnabled(e is not None)
        else:
            self.lbl_event_title.setText(f"Event {self.selected_event_id}")
            self.lbl_event_frames.setText("Frames: -")
            self.lbl_event_meta.setText("Select an actor row to edit its metadata.")
            self.lbl_event_status.setText(f"Event {self.selected_event_id} selected.")
            if hasattr(self, "lbl_hand_support_status"):
                self.lbl_hand_support_status.setVisible(not self._manual_mode_enabled())
                self.lbl_hand_support_status.setText(
                    "Hand support: select Left Hand or Right Hand to inspect the current-frame hand box state."
                )
                self.lbl_hand_support_status.setToolTip("")
            self._set_status_chip(getattr(self, "lbl_event_actor_chip", None), "No actor", "warn")
            self._set_status_chip(getattr(self, "lbl_event_health_chip", None), "Pending", "warn")
            self._set_status_card_tone(getattr(self, "event_status_card", None), "warn")
            for btn in (
                getattr(self, "btn_jump_start_chip", None),
                getattr(self, "btn_jump_onset_chip", None),
                getattr(self, "btn_jump_end_chip", None),
            ):
                if btn is not None:
                    btn.setEnabled(False)
        self._update_onboarding_banner()
        self._update_inline_event_editor()
