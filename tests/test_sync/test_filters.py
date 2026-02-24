"""
Project: PitchAura
File Created: 2026-02-23
Author: Xingnan Zhu
File Name: test_filters.py
Description:
    Tests for sync.smooth(): moving average and Kalman filter.
    Verifies noise reduction, output shape invariants, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch_aura.sync.filters import smooth
from pitch_aura.types import FrameRecord, FrameSequence, PitchSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _noisy_seq(
    n_frames: int = 50,
    noise_std: float = 2.0,
    seed: int = 42,
    frame_rate: float = 25.0,
) -> FrameSequence:
    """Sequence with a single player on a linear path plus Gaussian noise."""
    rng = np.random.default_rng(seed)
    pitch = PitchSpec(length=105.0, width=68.0, origin="center")
    dt = 1.0 / frame_rate
    frames = []
    for i in range(n_frames):
        t = i * dt
        true_pos = np.array([float(i) * 0.5, 0.0])
        noisy_pos = true_pos + rng.normal(0.0, noise_std, 2)
        frames.append(FrameRecord(
            timestamp=t,
            period=1,
            ball_position=np.array([0.0, 0.0]),
            player_ids=["p1"],
            team_ids=["home"],
            positions=np.array([noisy_pos]),
            velocities=np.zeros((1, 2)),
            is_goalkeeper=np.zeros(1, dtype=bool),
        ))
    return FrameSequence(frames=frames, frame_rate=frame_rate, pitch=pitch,
                         home_team_id="home", away_team_id="away")


def _constant_seq(n_frames: int = 10) -> FrameSequence:
    """Sequence where all players are stationary (no noise)."""
    pitch = PitchSpec()
    frames = [
        FrameRecord(
            timestamp=i * 0.04,
            period=1,
            ball_position=np.array([0.0, 0.0]),
            player_ids=["p1", "p2"],
            team_ids=["home", "away"],
            positions=np.array([[10.0, 5.0], [-10.0, -5.0]]),
            velocities=np.zeros((2, 2)),
            is_goalkeeper=np.zeros(2, dtype=bool),
        )
        for i in range(n_frames)
    ]
    return FrameSequence(frames=frames, frame_rate=25.0, pitch=pitch,
                         home_team_id="home", away_team_id="away")


# ---------------------------------------------------------------------------
# Shared behaviour (both methods)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["moving_average", "kalman"])
class TestSmoothCommon:
    def test_returns_frame_sequence(self, method: str):
        seq = _constant_seq()
        result = smooth(seq, method=method)
        assert isinstance(result, FrameSequence)

    def test_frame_count_preserved(self, method: str):
        seq = _constant_seq(10)
        result = smooth(seq, method=method)
        assert len(result) == 10

    def test_player_count_preserved(self, method: str):
        seq = _constant_seq(10)
        result = smooth(seq, method=method)
        for frame in result.frames:
            assert frame.n_players == 2

    def test_metadata_preserved(self, method: str):
        seq = _constant_seq()
        result = smooth(seq, method=method)
        assert result.frame_rate == seq.frame_rate
        assert result.home_team_id == seq.home_team_id
        assert result.pitch.length == seq.pitch.length

    def test_timestamps_unchanged(self, method: str):
        seq = _constant_seq(8)
        result = smooth(seq, method=method)
        orig_ts = seq.timestamps
        new_ts = result.timestamps
        np.testing.assert_allclose(new_ts, orig_ts, atol=1e-12)

    def test_stationary_player_unchanged(self, method: str):
        """Smoothing a stationary trajectory should not move positions much."""
        seq = _constant_seq(20)
        result = smooth(seq, method=method)
        for frame in result.frames:
            np.testing.assert_allclose(frame.positions[0], [10.0, 5.0], atol=0.1)

    def test_velocities_populated(self, method: str):
        seq = _noisy_seq(20)
        result = smooth(seq, method=method)
        for frame in result.frames:
            assert frame.velocities is not None

    def test_empty_sequence_returns_empty(self, method: str):
        pitch = PitchSpec()
        empty = FrameSequence(frames=[], frame_rate=25.0, pitch=pitch,
                              home_team_id="home", away_team_id="away")
        result = smooth(empty, method=method)
        assert len(result) == 0

    def test_invalid_method_raises(self, method: str):
        seq = _constant_seq()
        with pytest.raises(ValueError, match="method"):
            smooth(seq, method="fft_magic")


# ---------------------------------------------------------------------------
# Moving average specific
# ---------------------------------------------------------------------------

class TestMovingAverage:
    def test_reduces_noise(self):
        """After smoothing, the std of position residuals should decrease."""
        seq = _noisy_seq(n_frames=100, noise_std=3.0)
        result = smooth(seq, method="moving_average", window=7)
        # Extract position x of player p1 from both
        orig_x = np.array([f.positions[0, 0] for f in seq.frames])
        smooth_x = np.array([f.positions[0, 0] for f in result.frames])
        true_x = np.arange(100) * 0.5
        assert np.std(smooth_x - true_x) < np.std(orig_x - true_x)

    def test_window_1_is_identity(self):
        """Window=1 averages nothing — positions identical to input."""
        seq = _noisy_seq(10)
        result = smooth(seq, method="moving_average", window=1)
        for i in range(len(seq)):
            np.testing.assert_allclose(
                result[i].positions, seq[i].positions, atol=1e-10
            )


# ---------------------------------------------------------------------------
# Kalman filter specific
# ---------------------------------------------------------------------------

class TestKalmanFilter:
    def test_reduces_noise(self):
        """Kalman filter should reduce position noise."""
        seq = _noisy_seq(n_frames=80, noise_std=3.0)
        result = smooth(seq, method="kalman", process_noise=1.0, measurement_noise=4.0)
        orig_x = np.array([f.positions[0, 0] for f in seq.frames])
        smooth_x = np.array([f.positions[0, 0] for f in result.frames])
        true_x = np.arange(80) * 0.5
        assert np.std(smooth_x - true_x) < np.std(orig_x - true_x)

    def test_high_measurement_noise_smooths_more(self):
        """Higher r → trust measurements less → smoother output."""
        seq = _noisy_seq(60, noise_std=2.0)
        result_low_r = smooth(seq, method="kalman", measurement_noise=0.1)
        result_high_r = smooth(seq, method="kalman", measurement_noise=50.0)
        x_low  = np.array([f.positions[0, 0] for f in result_low_r.frames])
        x_high = np.array([f.positions[0, 0] for f in result_high_r.frames])
        # High r → smaller frame-to-frame jitter (smoother trajectory)
        assert np.std(np.diff(x_high)) < np.std(np.diff(x_low))
