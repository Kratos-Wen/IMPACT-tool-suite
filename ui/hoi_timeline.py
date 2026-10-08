from typing import Callable, List, Optional, Dict
from PyQt5.QtCore import Qt, QSize, QRect, QPoint
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPolygon, QFontMetrics
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QFrame,
)
from ui.timeline import BaseTimelineRow
from utils.constants import MIN_VIEW_SPAN, DEFAULT_VIEW_SPAN


class HOITimelineRow(BaseTimelineRow):
    """Single hand timeline row with onset marker editing."""

    def __init__(
        self,
        hand_key: str,
        title: str,
        get_segments: Callable[[], List[Dict]],
        get_color: Callable[[str], QColor],
        on_select: Callable[[int, str], None],
        on_update: Callable[[int, str, int, int, int], None],
        on_create: Callable[[str, int, int, int], Optional[int]],
        on_delete: Callable[[int, str], None],
        on_activate: Optional[Callable[[int, str], None]],
        on_hover: Callable[[int], None],
        get_frame_count: Callable[[], int],
        get_view_start: Callable[[], int],
        get_view_span: Callable[[], int],
        get_fps: Callable[[], int],
        get_gutter: Callable[[], int],
        parent=None,
    ):
        super().__init__(
            get_frame_count, get_view_start, get_view_span, get_fps, get_gutter, parent
        )
        self.hand_key = hand_key
        self.title = title
        self._get_segments = get_segments
        self._get_color = get_color
        self._on_select = on_select
        self._on_update = on_update
        self._on_create = on_create
        self._on_delete = on_delete
        self._on_activate = on_activate
        self._on_hover = on_hover

        self.setMouseTracking(True)
        self.setMinimumHeight(48)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._selected_event_id: Optional[int] = None
        self._dragging = False
        self._mode: Optional[str] = (
            None  # "create"|"move"|"resize_left"|"resize_right"|"move_onset"
        )
        self._active_event_id: Optional[int] = None
        self._active_interval: Optional[tuple] = None
        self._active_onset: Optional[int] = None
        self._preview_interval: Optional[tuple] = None
        self._preview_onset: Optional[int] = None
        self._create_anchor: Optional[int] = None
        self._grab_offset = 0

    def sizeHint(self) -> QSize:
        return QSize(900, 48)

    def _row_font(self, delta: float = 0.0, weight: int = QFont.Normal) -> QFont:
        font = QFont(self.font())
        size = font.pointSizeF()
        if size <= 0:
            size = 9.0
        font.setPointSizeF(max(7.5, size + delta))
        font.setWeight(weight)
        return font

    def set_selected(self, event_id: Optional[int]):
        if self._selected_event_id != event_id:
            self._selected_event_id = event_id
            self.update()

    def set_title(self, title: str):
        if title != self.title:
            self.title = title
            self.update()

    def _segments(self) -> List[Dict]:
        segs = []
        for seg in self._get_segments() or []:
            try:
                s = int(seg.get("start"))
                e = int(seg.get("end"))
            except Exception:
                continue
            if e < s:
                s, e = e, s
            onset = seg.get("onset")
            try:
                onset = int(onset) if onset is not None else s
            except Exception:
                onset = s
            onset = max(s, min(onset, e))
            segs.append(
                {
                    "event_id": seg.get("event_id"),
                    "start": s,
                    "end": e,
                    "onset": onset,
                    "verb": seg.get("verb") or "",
                    "color": seg.get("color"),
                }
            )
        return sorted(segs, key=lambda x: (x["start"], x.get("event_id", -1)))

    def _segment_at(self, frame: int) -> Optional[Dict]:
        for seg in self._segments():
            if seg["start"] <= frame <= seg["end"]:
                return seg
        return None

    def _hit_segment(self, x: int, y: Optional[int] = None) -> tuple:
        tol = 5
        for seg in self._segments():
            x1 = self.frame_to_x(seg["start"])
            x2 = self.frame_to_x(seg["end"])
            if x1 - tol <= x <= x2 + tol:
                onset_x = self.frame_to_x(seg["onset"])
                rect_top = 6
                onset_zone = y is not None and y <= rect_top + 10
                if abs(x - onset_x) <= tol and (
                    onset_zone or seg["onset"] not in (seg["start"], seg["end"])
                ):
                    return seg, "onset"
                if abs(x - x1) <= tol:
                    return seg, "left"
                if abs(x - x2) <= tol:
                    return seg, "right"
                if x1 < x < x2:
                    return seg, "center"
        return None, "none"

    def _snap_onset_to_segment_edge(self, frame: int, start: int, end: int, x: int) -> int:
        tol = 8
        try:
            if abs(int(x) - self.frame_to_x(int(start))) <= tol:
                return int(start)
            end_x = self.frame_to_x(int(end))
            if abs(int(x) - end_x) <= tol:
                return int(end)
        except Exception:
            pass
        return max(int(start), min(int(frame), int(end)))

    def _overlaps(self, start: int, end: int, exclude_id: Optional[int]) -> bool:
        for seg in self._segments():
            if exclude_id is not None and seg.get("event_id") == exclude_id:
                continue
            if not (end < seg["start"] or start > seg["end"]):
                return True
        return False

    def paintEvent(self, _e):
        p = QPainter(self)
        row_selected = self._selected_event_id is not None
        accent = QColor("#2563EB")
        row_bg = QColor("#EFF6FF") if row_selected else QColor("#F8FAFC")
        gutter_bg = QColor("#DBEAFE") if row_selected else QColor("#EEF2F6")
        p.fillRect(self.rect(), row_bg)
        p.fillRect(QRect(0, 0, self.get_gutter(), self.height()), gutter_bg)
        if row_selected:
            p.fillRect(QRect(0, 0, 4, self.height()), accent)

        start = self.get_vs()
        span = self.get_span()
        end = start + span
        fps = max(1, self.get_fps())

        self._draw_time_grid(p, start, end, fps)
        self._draw_gutter_title(p, self.title)

        for seg in self._segments():
            s, e, onset = seg["start"], seg["end"], seg["onset"]
            if e < start or s > end:
                continue
            s_vis = max(s, start)
            e_vis = min(e, end)
            x1 = self.frame_to_x(s_vis)
            x2 = self.frame_to_x(e_vis)
            rect = QRect(x1, 6, max(4, x2 - x1), self.height() - 12)
            color = seg.get("color") or self._get_color(seg.get("verb", ""))
            if isinstance(color, str):
                color = QColor(color)
            sel = seg.get("event_id") == self._selected_event_id
            fill = QColor(color).lighter(112 if sel else 108)
            outline = accent if sel else color.darker(140)
            p.setBrush(QBrush(fill))
            p.setPen(QPen(outline, 3 if sel else 1))
            p.drawRoundedRect(rect, 5, 5)

            if sel:
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(255, 255, 255, 120), 1))
                p.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 4, 4)

            if seg.get("verb"):
                p.setPen(QPen(QColor("#101828") if sel else QColor("#344054")))
                p.setFont(self._row_font(-1.0, QFont.DemiBold if sel else QFont.Medium))
                text = str(seg.get("verb"))
                elided = p.fontMetrics().elidedText(
                    text, Qt.ElideRight, max(0, rect.width() - 8)
                )
                p.drawText(
                    rect.adjusted(4, 2, -4, -2), Qt.AlignLeft | Qt.AlignVCenter, elided
                )

            if onset is not None:
                onset_x = self.frame_to_x(onset)
                marker_color = accent if sel else color.darker(160)
                p.setPen(QPen(marker_color, 2))
                p.drawLine(onset_x, rect.top(), onset_x, rect.bottom())
                tri = QPolygon(
                    [
                        QPoint(onset_x, rect.top() - 2),
                        QPoint(onset_x - 4, rect.top() + 6),
                        QPoint(onset_x + 4, rect.top() + 6),
                    ]
                )
                p.setBrush(QBrush(marker_color))
                p.drawPolygon(tri)

        if self._preview_interval is not None:
            s, e = self._preview_interval
            s_vis = max(s, start)
            e_vis = min(e, end)
            if s_vis <= e_vis:
                x1 = self.frame_to_x(s_vis)
                x2 = self.frame_to_x(e_vis)
                rect = QRect(x1, 6, max(4, x2 - x1), self.height() - 12)
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(70, 90, 120), 2, Qt.DashLine))
                p.drawRoundedRect(rect, 4, 4)
                if self._preview_onset is not None:
                    onset_x = self.frame_to_x(self._preview_onset)
                    p.setPen(QPen(QColor(70, 90, 120), 2))
                    p.drawLine(onset_x, rect.top(), onset_x, rect.bottom())

        self._draw_current_frame_marker(p, start, end)

        if self._hover_frame is not None and start <= self._hover_frame <= end:
            label = f"F {self._hover_frame} | {self._hover_frame / fps:.2f}s"
            self._draw_hover_marker(p, start, end, fps, label)

        p.setPen(QPen(QColor("#D0D5DD"), 1))
        p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)

    def mouseMoveEvent(self, e):
        g = self.get_gutter()
        if not self._dragging and e.x() < g:
            self._hover_frame = None
            self._on_hover(-1)
            self.setCursor(Qt.ArrowCursor)
            self.update()
            return
        f = self.x_to_frame(e.x())
        self._hover_frame = f
        self._on_hover(f)

        if not self._dragging:
            seg, where = self._hit_segment(e.x(), e.y())
            if where in ("left", "right"):
                self.setCursor(Qt.SizeHorCursor)
            elif where == "center":
                self.setCursor(Qt.OpenHandCursor)
            elif where == "onset":
                self.setCursor(Qt.SizeHorCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            self.update()
            return

        if self._mode == "create":
            start = self._create_anchor
            if start is None:
                start = self._preview_interval[0] if self._preview_interval else f
            s = int(start)
            e_ = max(s, int(f))
            if not self._overlaps(s, e_, None):
                self._preview_interval = (s, e_)
                self._preview_onset = s + ((e_ - s) // 2)
            self.update()
            return

        if self._mode == "move_onset" and self._active_interval:
            s, e_ = self._active_interval
            onset = self._snap_onset_to_segment_edge(f, s, e_, e.x())
            self._preview_interval = (s, e_)
            self._preview_onset = onset
            self.update()
            return

        if self._active_interval is None:
            return

        s0, e0 = self._active_interval
        if self._mode == "resize_left":
            s = min(f, e0)
            e_ = e0
            if not self._overlaps(s, e_, self._active_event_id):
                self._preview_interval = (s, e_)
                if self._active_onset is not None:
                    self._preview_onset = max(s, min(self._active_onset, e_))
        elif self._mode == "resize_right":
            s = s0
            e_ = max(f, s0)
            if not self._overlaps(s, e_, self._active_event_id):
                self._preview_interval = (s, e_)
                if self._active_onset is not None:
                    self._preview_onset = max(s, min(self._active_onset, e_))
        elif self._mode == "move":
            length = e0 - s0
            target_s = f - self._grab_offset
            s = max(0, min(target_s, self.get_fc() - 1 - length))
            e_ = s + length
            if not self._overlaps(s, e_, self._active_event_id):
                self._preview_interval = (s, e_)
                if self._active_onset is not None:
                    offset = self._active_onset - s0
                    self._preview_onset = max(s, min(s + offset, e_))
        self.update()

    def leaveEvent(self, e):
        if not self._dragging and self._hover_frame is not None:
            self._hover_frame = None
            self._on_hover(-1)
            self.setCursor(Qt.ArrowCursor)
            self.update()
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            seg, _ = self._hit_segment(e.x(), e.y())
            if seg and self._on_delete:
                self._on_delete(seg.get("event_id"), self.hand_key)
            return
        if e.button() != Qt.LeftButton:
            return
        g = self.get_gutter()
        if e.x() < g:
            return
        seg, where = self._hit_segment(e.x(), e.y())
        f = self.x_to_frame(e.x())
        if seg:
            self._on_select(seg.get("event_id"), self.hand_key)
            self._dragging = True
            self._active_event_id = seg.get("event_id")
            self._active_interval = (seg["start"], seg["end"])
            self._active_onset = seg.get("onset")
            if where == "left":
                self._mode = "resize_left"
            elif where == "right":
                self._mode = "resize_right"
            elif where == "onset":
                self._mode = "move_onset"
            else:
                self._mode = "move"
                self._grab_offset = f - seg["start"]
            self._preview_interval = self._active_interval
            self._preview_onset = self._active_onset
            self.update()
            return

        # create new segment
        if self._overlaps(f, f, None):
            return
        self._dragging = True
        self._mode = "create"
        self._create_anchor = f
        self._preview_interval = (f, f)
        self._preview_onset = f
        self.update()

    def mouseReleaseEvent(self, _e):
        if not self._dragging:
            return
        if self._mode == "create" and self._preview_interval:
            s, e = self._preview_interval
            onset = self._preview_onset if self._preview_onset is not None else s
            created_id = None
            if self._on_create:
                created_id = self._on_create(self.hand_key, s, e, onset)
            if created_id is not None:
                self.set_selected(created_id)
        elif self._active_event_id is not None and self._preview_interval:
            s, e = self._preview_interval
            onset = self._preview_onset if self._preview_onset is not None else s
            if self._on_update:
                self._on_update(self._active_event_id, self.hand_key, s, e, onset)

        self._dragging = False
        self._mode = None
        self._active_event_id = None
        self._active_interval = None
        self._active_onset = None
        self._preview_interval = None
        self._preview_onset = None
        self._create_anchor = None
        self.update()

    def mouseDoubleClickEvent(self, e):
        if e.button() != Qt.LeftButton:
            super().mouseDoubleClickEvent(e)
            return
        if e.x() < self.get_gutter():
            return
        seg, _where = self._hit_segment(e.x(), e.y())
        if not seg:
            return
        event_id = seg.get("event_id")
        if event_id is None:
            return
        self._on_select(event_id, self.hand_key)
        if self._on_activate:
            self._on_activate(event_id, self.hand_key)
        e.accept()
        return


class HOITimeline(QWidget):
    """Multi-actor HOI timeline with dynamic row generation."""

    def __init__(
        self,
        get_segments_for_hand: Callable[[str], List[Dict]],
        get_color_for_verb: Callable[[str], QColor],
        on_select: Callable[[int, str], None],
        on_update: Callable[[int, str, int, int, int], None],
        on_create: Callable[[str, int, int, int], Optional[int]],
        on_delete: Callable[[int, str], None],
        on_hover: Callable[[int], None],
        get_frame_count: Callable[[], int],
        get_fps: Callable[[], int],
        on_activate: Optional[Callable[[int, str], None]] = None,
        get_title_for_hand: Optional[Callable[[str], str]] = None,
        actors_config: List[Dict] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._get_segments = get_segments_for_hand
        self._get_color = get_color_for_verb
        self._on_select = on_select
        self._on_update = on_update
        self._on_create = on_create
        self._on_delete = on_delete
        self._on_activate = on_activate
        self._on_hover = on_hover
        self._get_fc = get_frame_count
        self._get_fps = get_fps
        self._get_title = get_title_for_hand
        self._view_start = 0
        self._view_span: Optional[int] = None
        self._gutter_px = 118
        self._block_view_signal = False
        self._user_span_override = False
        self._selected_event_id: Optional[int] = None
        self._selected_hand: Optional[str] = None
        
        self.actors_config = actors_config or [
            {"id": "Left_hand", "short": "L"},
            {"id": "Right_hand", "short": "R"},
        ]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.rows_frame = QFrame(self)
        self.rows_frame.setFrameShape(QFrame.Box)
        self.rows_frame.setFrameShadow(QFrame.Plain)
        self.rows_layout = QVBoxLayout(self.rows_frame)
        self.rows_layout.setContentsMargins(6, 6, 6, 6)
        self.rows_layout.setSpacing(4)
        
        self.actor_rows: Dict[str, HOITimelineRow] = {}
        self._build_rows()
        
        layout.addWidget(self.rows_frame)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)
        self.lbl_timeline_caption = QLabel("Timeline")
        self.lbl_timeline_caption.setStyleSheet("color: #344054; font-weight: 600;")
        controls.addWidget(self.lbl_timeline_caption, 0)
        lbl_start = QLabel("Start")
        lbl_start.setStyleSheet("color: #667085;")
        controls.addWidget(lbl_start, 0)
        self.slider_view = QSlider(Qt.Horizontal, self)
        self.slider_view.valueChanged.connect(self._on_view_start_changed)
        controls.addWidget(self.slider_view, 2)
        lbl_span = QLabel("Span")
        lbl_span.setStyleSheet("color: #667085;")
        controls.addWidget(lbl_span, 0)
        self.slider_span = QSlider(Qt.Horizontal, self)
        self.slider_span.valueChanged.connect(self._on_view_span_changed)
        controls.addWidget(self.slider_span, 3)
        controls.addSpacing(6)
        self.lbl_view_summary = QLabel("Frames --")
        self.lbl_view_summary.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_view_summary.setStyleSheet("color: #667085; font-weight: 600;")
        controls.addWidget(self.lbl_view_summary, 0)
        layout.addLayout(controls)

        self._init_sliders()
        self.update_titles()

    def _reinit_rows(self):
        """Rebuild rows based on updated actors_config."""
        self._build_rows()
        self.update_titles()
        self.update()

    def _build_rows(self):
        # Clear existing rows if any
        for row in self.actor_rows.values():
            self.rows_layout.removeWidget(row)
            row.deleteLater()
        self.actor_rows.clear()

        for actor in self.actors_config:
            aid = actor["id"]
            short = actor.get("short", aid[:1])
            row = HOITimelineRow(
                aid,
                short,
                lambda a=aid: self._get_segments(a),
                self._get_color,
                self._on_select,
                self._on_update,
                self._on_create,
                self._on_delete,
                self._on_activate,
                self._on_hover,
                self._get_fc,
                self.get_view_start,
                self.get_view_span,
                self._get_fps,
                self.get_gutter,
                self,
            )
            self.actor_rows[aid] = row
            self.rows_layout.addWidget(row)

    def sizeHint(self) -> QSize:
        row_count = len(self.actor_rows)
        return QSize(900, 30 + row_count * 40)

    def get_view_start(self) -> int:
        return self._view_start

    def get_view_span(self) -> int:
        if self._view_span is None:
            return max(1, int(self._get_fc()))
        return max(1, int(self._view_span))

    def get_gutter(self) -> int:
        return self._gutter_px

    def _compute_gutter_px(self, titles: List[str]):
        font = QFont(self.font())
        if font.pointSizeF() <= 0:
            font.setPointSizeF(9.0)
        fm = QFontMetrics(font)
        max_w = 0
        for t in titles:
            for part in str(t or "").splitlines():
                part = part.strip()
                if not part:
                    continue
                try:
                    w = fm.horizontalAdvance(part)
                except AttributeError:
                    w = fm.width(part)
                max_w = max(max_w, w)
        self._gutter_px = max(112, min(180, max_w + 20))

    def update_titles(self):
        titles = []
        for aid, row in self.actor_rows.items():
            title = aid
            if self._get_title:
                title = self._get_title(aid) or aid
            row.set_title(title)
            titles.append(title)
        
        self._compute_gutter_px(titles)
        for row in self.actor_rows.values():
            row.update()

    def set_frame_count(self, fc: int):
        fc = max(1, int(fc))
        if not self._user_span_override:
            self._view_span = min(fc, DEFAULT_VIEW_SPAN)
        else:
            if self._view_span is None:
                self._view_span = min(fc, DEFAULT_VIEW_SPAN)
            else:
                self._view_span = min(self._view_span, fc)
        self._view_start = min(self._view_start, max(0, fc - self._view_span))
        self._init_sliders()
        self.refresh()

    def _update_view_summary(self):
        if not hasattr(self, "lbl_view_summary"):
            return
        fc = max(1, int(self._get_fc()))
        fps = max(1, int(self._get_fps()))
        span = max(1, int(self._view_span or fc))
        start = max(0, int(self._view_start))
        end = max(start, min(fc - 1, start + span - 1))
        self.lbl_view_summary.setText(f"F {start}-{end} | {span / fps:.1f}s")

    def _init_sliders(self):
        fc = max(1, int(self._get_fc()))
        self._block_view_signal = True
        self.slider_span.blockSignals(True)
        self.slider_span.setMinimum(0)
        self.slider_span.setMaximum(100)

        def span_to_val(span):
            if fc <= MIN_VIEW_SPAN:
                return 100
            span = max(MIN_VIEW_SPAN, min(span, fc))
            return int(round(100 * (span - MIN_VIEW_SPAN) / max(1, fc - MIN_VIEW_SPAN)))

        if self._view_span is None:
            self._view_span = min(fc, DEFAULT_VIEW_SPAN)
        if fc <= MIN_VIEW_SPAN:
            self._view_span = fc
        self.slider_span.setValue(span_to_val(self._view_span))
        self.slider_span.blockSignals(False)

        self._refresh_view_slider()
        self._update_view_summary()
        self._block_view_signal = False

    def _refresh_view_slider(self):
        fc = max(1, int(self._get_fc()))
        span = max(1, int(self._view_span or fc))
        max_start = max(0, fc - span)
        self.slider_view.blockSignals(True)
        self.slider_view.setMinimum(0)
        self.slider_view.setMaximum(max_start)
        self._view_start = min(self._view_start, max_start)
        self.slider_view.setValue(self._view_start)
        self.slider_view.blockSignals(False)
        for row in self.actor_rows.values():
            row.update()
        self._update_view_summary()

    def _on_view_start_changed(self, v: int):
        self._view_start = int(v)
        for row in self.actor_rows.values():
            row.update()
        self._update_view_summary()

    def _on_view_span_changed(self, val: int):
        fc = max(1, int(self._get_fc()))
        if fc <= MIN_VIEW_SPAN:
            new_span = fc
        else:
            new_span = int(round(MIN_VIEW_SPAN + (fc - MIN_VIEW_SPAN) * val / 100.0))
            new_span = max(MIN_VIEW_SPAN, min(new_span, fc))
        if self._view_start + new_span > fc:
            self._view_start = max(0, fc - new_span)
        self._view_span = new_span
        if not self._block_view_signal:
            self._user_span_override = True
        self._refresh_view_slider()
        self._update_view_summary()

    def _ensure_visible(self, frame: int):
        fc = max(1, int(self._get_fc()))
        span = max(1, int(self._view_span or fc))
        start = self._view_start
        end = start + span - 1
        if frame < start or frame > end:
            new_start = max(0, min(fc - span, frame - span // 2))
            self._view_start = new_start
            self._refresh_view_slider()

    def set_selected(self, event_id: Optional[int], hand_key: Optional[str]):
        self._selected_event_id = event_id
        self._selected_hand = hand_key
        for aid, row in self.actor_rows.items():
            row.set_selected(event_id if hand_key == aid else None)

    def set_current_frame(self, frame: int, follow: bool = True):
        if follow:
            self._ensure_visible(frame)
        for row in self.actor_rows.values():
            row.set_current_frame(frame)

    def refresh(self):
        for row in self.actor_rows.values():
            row.update()
