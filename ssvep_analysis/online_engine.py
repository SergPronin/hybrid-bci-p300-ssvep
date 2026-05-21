from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ssvep_analysis.burst_gate import BurstGate, BurstGateConfig

# Reuse MSI runtime helpers (CoreCLR/pythonnet) used by ssvep_analyzer GUI.
from scripts import test_msi_exec as tme


@dataclass(frozen=True)
class SSVEPParams:
    fs_hz: float = 250.0
    window_sec: float = 2.0
    freqs_hz: Tuple[float, ...] = ()
    mode: str = "continuous"  # "continuous" | "burst"
    # Индексы каналов EEG с 0; пустой кортеж = все каналы потока
    roi_channels_0idx: Tuple[int, ...] = ()


@dataclass(frozen=True)
class SSVEPDecision:
    winner_1based: Optional[int]
    winner_0idx: Optional[int]
    coef: Optional[list]
    mode: str
    classify_allowed: bool
    debug: Dict[str, Any]


class SSVEPOnlineEngine:
    """Headless SSVEP engine: buffer EEG window, (optionally) burst-gate, run MSIExec."""

    def __init__(self) -> None:
        self.params = SSVEPParams()
        self._burst_gate = BurstGate(BurstGateConfig(window_sec=float(self.params.window_sec)))
        self._msi = None
        self._models = None
        self._n_samples = 0
        self._buf_t: list[float] = []
        self._buf_x: list[np.ndarray] = []  # each: (n_ch,)

    def reset(self, *, params: Optional[SSVEPParams] = None) -> None:
        if params is not None:
            self.params = params
        self._burst_gate = BurstGate(BurstGateConfig(window_sec=float(self.params.window_sec)))
        n_lamps = len(self.params.freqs_hz)
        if n_lamps > 0:
            self._burst_gate.set_active_lamps(n_lamps)
        self._n_samples = int(round(float(self.params.fs_hz) * float(self.params.window_sec)))
        self._buf_t = []
        self._buf_x = []
        if self._msi is not None:
            self._apply_templates()

    def ensure_msi_ready(self) -> None:
        if self._msi is not None:
            return
        msi, _msi_res, _dotnet_root = tme.load_msi_runtime()
        self._msi = msi
        self._apply_templates()

    def _apply_templates(self) -> None:
        assert self._msi is not None
        freqs = [float(x) for x in self.params.freqs_hz]
        np_models = tme.generate_model_signals(freqs, float(self.params.fs_hz), float(self.params.window_sec))
        model_list = tme.build_model_signal_list(self._msi, np_models, verbose=False)
        self._msi.ModelSignal = model_list
        self._models = np_models

    def ingest_eeg_chunk(self, *, times: Sequence[float], samples: np.ndarray) -> None:
        arr = np.asarray(samples, dtype=np.float64)
        if arr.size == 0:
            return
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            arr = arr.reshape(arr.shape[0], -1)
        ts = list(float(t) for t in times)
        if not ts:
            return
        n = min(len(ts), int(arr.shape[0]))
        for i in range(n):
            self._buf_t.append(float(ts[i]))
            self._buf_x.append(np.asarray(arr[i], dtype=np.float64).ravel())
        self._trim()

    def ingest_migalka_marker(self, *, lsl_time: float, value: object) -> None:
        """For burst mode gating: feed LSL markers 100+i|on/off."""
        self._burst_gate.ingest_marker(float(lsl_time), value)

    def _trim(self) -> None:
        nmax = int(max(1, self._n_samples))
        if len(self._buf_x) > nmax:
            drop = len(self._buf_x) - nmax
            self._buf_x = self._buf_x[drop:]
            self._buf_t = self._buf_t[drop:]

    def can_classify(self) -> bool:
        if len(self._buf_x) < int(self._n_samples):
            return False
        if str(self.params.mode) != "burst":
            return True
        if not self._buf_t:
            return False
        allowed, _reason = self._burst_gate.classify_allowed(
            np.asarray(self._buf_t, dtype=np.float64)
        )
        return bool(allowed)

    def classify(self) -> SSVEPDecision:
        self.ensure_msi_ready()
        assert self._msi is not None
        allowed = self.can_classify()
        if not allowed:
            return SSVEPDecision(
                winner_1based=None,
                winner_0idx=None,
                coef=None,
                mode=str(self.params.mode),
                classify_allowed=False,
                debug={"note": "buffer_not_ready_or_gate_closed", "buf_len": len(self._buf_x), "n_samples": self._n_samples},
            )
        # MSI expects (channels, samples)
        X = np.stack(self._buf_x, axis=0)  # (samples, channels)
        sig_np = np.ascontiguousarray(X.T, dtype=np.float64)
        roi = [int(c) for c in self.params.roi_channels_0idx]
        ch_idx = (
            [c for c in roi if 0 <= c < int(sig_np.shape[0])]
            if roi
            else list(range(int(sig_np.shape[0])))
        )
        if not ch_idx:
            return SSVEPDecision(
                winner_1based=None,
                winner_0idx=None,
                coef=None,
                mode=str(self.params.mode),
                classify_allowed=False,
                debug={"note": "no_valid_roi_channels", "roi": list(roi), "n_ch": int(sig_np.shape[0])},
            )
        sig_np = np.ascontiguousarray(sig_np[ch_idx, :], dtype=np.float64)
        sig_managed = tme.numpy_to_double_matrix2d(sig_np, verbose=False)
        t0 = time.perf_counter()
        winner = int(self._msi.MSIExec(sig_managed))
        dt_ms = float((time.perf_counter() - t0) * 1000.0)
        coef = None
        try:
            # could be scalar or list-like
            c = self._msi.Coef
            if c is not None:
                cnt = getattr(c, "Count", None)
                if cnt is not None:
                    coef = [float(c[i]) for i in range(int(cnt))]
                else:
                    coef = [float(c)]
        except Exception:
            coef = None
        w0 = winner - 1 if winner > 0 else None
        return SSVEPDecision(
            winner_1based=winner,
            winner_0idx=w0,
            coef=coef,
            mode=str(self.params.mode),
            classify_allowed=True,
            debug={"msi_exec_ms": dt_ms, "buf_shape": [int(sig_np.shape[0]), int(sig_np.shape[1])]},
        )

