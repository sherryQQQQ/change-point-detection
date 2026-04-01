"""
RegimeModel — connects CP detection → segment fitting → PMF prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, List, Optional


class RegimeModel:
    """
    Connects change-point detection → segment fitting → PMF prediction.

    Parameters
    ----------
    predictor : UnifiedPredictor instance
    pmf_fn    : callable(Z_piece, dt_piece, mu, m, t, ...) -> np.ndarray
    alarm_threshold : float
        CP probability threshold used to split data into segments.
    """

    def __init__(self, predictor, pmf_fn: Callable, alarm_threshold: float = 0.1):
        self.predictor = predictor
        self.pmf_fn = pmf_fn
        self.alarm_threshold = alarm_threshold
        self._segments: List[pd.DataFrame] = []
        self._pmfs: List[np.ndarray] = []
        self._regime_info: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def fit(self, data: pd.DataFrame, mu: float, m: int = 1,
            t: float = 1.0, **pmf_kwargs) -> List[Dict[str, Any]]:
        """
        Run predictor, detect change-point boundaries, compute PMF per segment.

        Returns list of dicts: {segment, cp_time, pmf, Z_piece, dt_piece}
        """
        results = self.predictor.predict(data)
        inner = results.get('predictor')

        # Extract per-point CP probabilities
        cp_probs = np.zeros(len(data))
        if inner is not None and hasattr(inner, 'results'):
            r = inner.results
            for key in ('changepoint_prob', 'changepoint_probs'):
                if key in r and r[key] is not None:
                    raw = np.asarray(r[key]).flatten()
                    # align to data length
                    n = min(len(raw), len(data))
                    cp_probs[:n] = raw[:n]
                    break

        # Find change-point indices
        cp_indices = list(np.where(cp_probs > self.alarm_threshold)[0])
        boundaries = [0] + cp_indices + [len(data)]
        segments = [data.iloc[s:e].copy()
                    for s, e in zip(boundaries[:-1], boundaries[1:]) if e > s]

        self._segments = segments
        self._pmfs = []
        self._regime_info = []

        for seg in segments:
            vals = seg['value'].values.astype(float)
            times = (seg['time'].values.astype(float)
                     if 'time' in seg.columns else np.arange(len(seg), dtype=float))

            if len(times) < 2:
                Z_piece = np.array([np.mean(vals)])
                dt_piece = np.array([1.0])
            else:
                dt_piece = np.diff(times, prepend=times[0])
                dt_piece[0] = times[1] - times[0]
                Z_piece = np.maximum(vals, 1e-6)

            dt_piece = np.where(np.isnan(dt_piece) | (dt_piece <= 0), 0.05, dt_piece)
            t_seg = min(t, dt_piece.sum() * 0.99)

            try:
                pmf = self.pmf_fn(Z_piece, dt_piece, mu, m, t=t_seg, **pmf_kwargs)
            except Exception as e:
                print(f"  PMF failed for segment (len={len(seg)}): {e}")
                pmf = None

            self._pmfs.append(pmf)
            self._regime_info.append({
                'segment': seg,
                'cp_time': float(seg['time'].iloc[0]) if 'time' in seg.columns else 0.0,
                'pmf': pmf,
                'Z_piece': Z_piece,
                'dt_piece': dt_piece,
            })

        return self._regime_info

    # ------------------------------------------------------------------
    def get_pmfs(self) -> List[Optional[np.ndarray]]:
        return self._pmfs

    def get_segments(self) -> List[pd.DataFrame]:
        return self._segments

    # ------------------------------------------------------------------
    def plot_regimes(self, figsize=(14, 5)):
        if not self._segments:
            print("Call fit() first.")
            return

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        ax_ts = axes[0]
        colors = plt.cm.tab10.colors
        for k, seg in enumerate(self._segments):
            t_col = seg['time'] if 'time' in seg.columns else np.arange(len(seg))
            ax_ts.plot(t_col, seg['value'], color=colors[k % len(colors)],
                       label=f'Regime {k+1}', lw=1.5)
        ax_ts.set_title('Detected Regimes')
        ax_ts.set_xlabel('Time')
        ax_ts.set_ylabel('Value')
        ax_ts.legend()
        ax_ts.grid(True, alpha=0.3)

        ax_pmf = axes[1]
        for k, pmf in enumerate(self._pmfs):
            if pmf is not None and len(pmf):
                ax_pmf.plot(np.arange(len(pmf)), pmf,
                            color=colors[k % len(colors)],
                            label=f'Regime {k+1}', lw=1.5)
        ax_pmf.set_title('Predicted PMFs per Regime')
        ax_pmf.set_xlabel('Queue Length')
        ax_pmf.set_ylabel('Probability')
        ax_pmf.legend()
        ax_pmf.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
