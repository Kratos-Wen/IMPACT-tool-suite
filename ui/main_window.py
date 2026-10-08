from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QSizePolicy, QVBoxLayout, QWidget

from ui.placeholder import PlaceholderPane
from utils.op_logger import OperationLogger
from utils.optional_deps import (
    MissingOptionalDependency,
    format_missing_dependency_message,
    import_optional_module,
)
from utils.shortcut_settings import load_logging_policy, save_logging_policy


class MainWindow(QWidget):
    """Standalone host for IMPACT HOI."""

    def __init__(self, logger: OperationLogger = None):
        super().__init__()
        self._app_title = "IMPACT HOI"
        self.setWindowTitle(self._app_title)
        self.setGeometry(80, 60, 1360, 860)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        try:
            screen = QApplication.primaryScreen()
            geom = screen.availableGeometry() if screen else None
            if geom:
                min_w = min(1100, max(900, geom.width() - 180))
                min_h = min(760, max(620, geom.height() - 180))
                self.setMinimumSize(min_w, min_h)
                target_w = max(min_w, int(geom.width() * 0.94))
                target_h = max(min_h, int(geom.height() * 0.93))
                self.resize(max(min_w, target_w), max(min_h, target_h))
            else:
                self.setMinimumSize(800, 600)
        except Exception:
            self.setMinimumSize(800, 600)
        try:
            self.setWindowFlag(Qt.MSWindowsFixedSizeDialogHint, False)
        except Exception:
            pass

        self.op_logger = logger or OperationLogger(False)
        cli_forced_ops_enabled = bool(getattr(self.op_logger, "enabled", False))
        initial_logging = load_logging_policy(
            default_ops_enabled=cli_forced_ops_enabled,
            default_validation_summary_enabled=True,
        )
        self._validation_summary_enabled = bool(
            initial_logging.get("validation_summary_enabled", True)
        )
        self.op_logger.enabled = bool(
            cli_forced_ops_enabled or initial_logging.get("ops_csv_enabled", False)
        )
        self.hoi_window = None
        self.placeholder = None

        root = QVBoxLayout(self)

        try:
            module = import_optional_module(
                "ui.hoi_window",
                feature_name="IMPACT HOI",
                install_hint=(
                    "Install the optional HOI dependencies only if you need this task, "
                    "for example: pip install opencv-python ultralytics mediapipe torch torchvision torchaudio"
                ),
            )
            HOIWindow = getattr(module, "HOIWindow")
            self.hoi_window = HOIWindow(
                self,
                on_close=None,
                logger=self.op_logger,
            )
            root.addWidget(self.hoi_window)
            self._apply_shortcuts()
            self._set_logging_policy(
                bool(getattr(self.op_logger, "enabled", False)),
                bool(self._validation_summary_enabled),
            )
        except MissingOptionalDependency as ex:
            self.placeholder = PlaceholderPane(
                "IMPACT HOI",
                format_missing_dependency_message(ex),
            )
            root.addWidget(self.placeholder)
        except Exception as ex:
            self.placeholder = PlaceholderPane(
                "IMPACT HOI",
                f"Failed to load the task window:\n{ex}",
            )
            root.addWidget(self.placeholder)

    def _apply_shortcuts(self, bindings=None) -> None:
        try:
            if self.hoi_window is not None:
                self.hoi_window.apply_shortcut_settings(bindings)
        except Exception:
            pass

    def _set_logging_policy(
        self, oplog_enabled: bool, validation_summary_enabled: bool
    ) -> None:
        self.op_logger.enabled = bool(oplog_enabled)
        self._validation_summary_enabled = bool(validation_summary_enabled)
        try:
            ok_save, path_or_err = save_logging_policy(
                {
                    "ops_csv_enabled": bool(oplog_enabled),
                    "validation_summary_enabled": bool(validation_summary_enabled),
                }
            )
            if not ok_save:
                print(f"[LOG][ERROR] Failed to save logging policy: {path_or_err}")
        except Exception:
            pass
        try:
            if self.hoi_window is not None:
                self.hoi_window.set_logging_policy(
                    bool(oplog_enabled), bool(validation_summary_enabled)
                )
        except Exception:
            pass

    def closeEvent(self, event):
        try:
            if self.hoi_window is not None:
                if not self.hoi_window._confirm_close_request(prompt_parent=self):
                    event.ignore()
                    return
                self.hoi_window._finalize_close_request()
        except Exception:
            event.ignore()
            return
        return super().closeEvent(event)

