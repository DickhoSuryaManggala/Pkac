import numpy as np
import pytest

pytest.importorskip("cv2", reason="butuh opencv (opencv-python-headless)")
from vehicle_lighting.lighting import apply_lighting, mean_luminance, auto_gamma, clahe_enhance


def _img():
    rng = np.random.default_rng(0)
    return rng.integers(40, 220, (64, 64, 3), dtype=np.uint8)


def test_identity():
    x = _img()
    assert np.abs(apply_lighting(x, 1.0, 1.0).astype(int) - x.astype(int)).max() <= 1


def test_monotonic_darker():
    x = _img()
    l = [mean_luminance(apply_lighting(x, e, 1.0)) for e in (1.0, 0.4, 0.12)]
    assert l[0] > l[1] > l[2]


def test_enhance_brightens_dark_and_keeps_shape():
    dark = apply_lighting(_img(), 0.12, 1.4)
    assert mean_luminance(auto_gamma(dark)) > mean_luminance(dark)
    assert clahe_enhance(dark).shape == dark.shape
