#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified 60-run protocol runner (P300×2 + SSVEP×2) for clinical operators."""

from __future__ import annotations

import sys
from pathlib import Path
import subprocess

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment_protocol.protocol_runner import ProtocolConfig, ProtocolRunner  # noqa: E402


class ProtocolRunnerWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hybrid protocol runner (P300×2 + SSVEP×2)")
        self.setMinimumWidth(640)

        self._runner: ProtocolRunner | None = None
        self._stimulus_proc: subprocess.Popen[str] | None = None

        root = QVBoxLayout(self)
        form = QFormLayout()

        self.ed_subject = QLineEdit("subject_001")
        self.ed_output = QLineEdit(str((_ROOT / "experiment_runs").resolve()))
        self.ed_com = QLineEdit("")

        self.spin_p300 = QSpinBox()
        self.spin_p300.setRange(1, 200)
        self.spin_p300.setValue(15)

        self.spin_ssvep = QSpinBox()
        self.spin_ssvep.setRange(1, 200)
        self.spin_ssvep.setValue(15)

        # Stimulator (PsychoPy) auto-mode parameters
        self.spin_inter_trial = QDoubleSpinBox()
        self.spin_inter_trial.setRange(0.0, 30.0)
        self.spin_inter_trial.setDecimals(1)
        self.spin_inter_trial.setSingleStep(0.5)
        self.spin_inter_trial.setValue(1.0)

        self.spin_sequences = QSpinBox()
        self.spin_sequences.setRange(1, 50)
        self.spin_sequences.setValue(12)

        self.spin_plan_trials = QSpinBox()
        self.spin_plan_trials.setRange(1, 50)
        self.spin_plan_trials.setValue(15)

        self.spin_plan_target = QSpinBox()
        self.spin_plan_target.setRange(0, 8)
        self.spin_plan_target.setValue(4)

        self.spin_plan_target_epochs = QSpinBox()
        self.spin_plan_target_epochs.setRange(1, 50)
        self.spin_plan_target_epochs.setValue(12)

        self.chk_run_stimulus = QCheckBox(
            "Запускать экранную стимуляцию (PsychoPy, случайная цель, без клика START)"
        )
        self.chk_run_stimulus.setChecked(True)
        self.chk_run_stimulus.setToolTip(
            "Поднимает `python run_app.py --auto-random-protocol --no-analyzer`.\n"
            "Можно задать первые trial с фиксированной целью для набора шаблона (см. run_app.py --auto-calib-*).\n"
            "Нужен Python с установленным psychopy.\n"
            "Маркеры trial_start/target идут в LSL как и при ручном START."
        )

        form.addRow("Subject ID:", self.ed_subject)
        form.addRow("Output root:", self.ed_output)
        form.addRow("COM port (Migalka):", self.ed_com)
        form.addRow("P300 trials per mode:", self.spin_p300)
        form.addRow("SSVEP blocks per mode:", self.spin_ssvep)
        form.addRow("Stimulus inter-trial pause (s):", self.spin_inter_trial)
        form.addRow("Stimulus sequences (per trial):", self.spin_sequences)
        form.addRow("AUC plan trials:", self.spin_plan_trials)
        form.addRow("AUC plan target tile (0..8):", self.spin_plan_target)
        form.addRow("Template target epochs:", self.spin_plan_target_epochs)
        form.addRow("", self.chk_run_stimulus)

        root.addLayout(form)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Start protocol")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        root.addLayout(btn_row)

        self.lbl_status = QLabel("Idle. Запустите EEG/LSL и стимулятор, затем Start.")
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._on_tick)

        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)

    def _on_start(self) -> None:
        subject = self.ed_subject.text().strip()
        out_root = self.ed_output.text().strip()
        com = self.ed_com.text().strip()
        if not subject:
            QMessageBox.warning(self, "Subject", "Введите Subject ID")
            return
        if not out_root:
            QMessageBox.warning(self, "Output", "Введите Output root")
            return
        if not com:
            QMessageBox.warning(self, "COM", "Введите COM порт мигалки (например, COM3 или /dev/tty.usbmodem...)")
            return

        if self.chk_run_stimulus.isChecked():
            run_py = sys.executable
            run_script = _ROOT / "run_app.py"
            if run_script.exists():
                self._stimulus_proc = subprocess.Popen(
                    [
                        run_py,
                        str(run_script),
                        "--auto-random-protocol",
                        "--no-analyzer",
                        "--inter-trial-s",
                        str(float(self.spin_inter_trial.value())),
                        "--sequences",
                        str(int(self.spin_sequences.value())),
                        "--auto-max-trials",
                        str(int(self.spin_p300.value()) * 2),
                        "--auto-plan-trials",
                        str(int(self.spin_plan_trials.value())),
                        "--auto-plan-target-tile-id",
                        str(int(self.spin_plan_target.value())),
                        "--auto-plan-target-repeats",
                        "0",
                        "--auto-plan-target-epochs",
                        str(int(self.spin_plan_target_epochs.value())),
                    ],
                    cwd=str(_ROOT),
                )

        cfg = ProtocolConfig(
            output_root=Path(out_root),
            subject_id=subject,
            com_port=com,
            p300_trials_per_mode=int(self.spin_p300.value()),
            ssvep_blocks_per_mode=int(self.spin_ssvep.value()),
        )
        self._runner = ProtocolRunner(cfg)
        self._runner.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._timer.start()
        self.lbl_status.setText("Started. Preflight…")

    def _on_stop(self) -> None:
        if self._stimulus_proc is not None:
            try:
                if self._stimulus_proc.poll() is None:
                    self._stimulus_proc.terminate()
            except Exception:
                pass
            self._stimulus_proc = None
        if self._runner is None:
            return
        self._runner.stop(reason="user_stop")
        self._runner = None
        self._timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Stopped.")

    def _on_tick(self) -> None:
        if self._runner is None:
            return
        self._runner.tick()
        self.lbl_status.setText(self._runner.status_text)
        # After P300 stage ends, stop the PsychoPy stimulator so it won't keep generating extra trials
        if self._stimulus_proc is not None and self._stimulus_proc.poll() is None:
            if self._runner.state in ("ssvep_continuous", "ssvep_burst", "finalize", "stopped"):
                try:
                    self._stimulus_proc.terminate()
                except Exception:
                    pass
                self._stimulus_proc = None
        if self._runner.state in ("stopped",):
            self._timer.stop()
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = ProtocolRunnerWidget()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

