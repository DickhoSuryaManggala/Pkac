import pandas as pd
import pytest

pytest.importorskip("cv2", reason="butuh opencv (opencv-python-headless)")
from vehicle_lighting.data import group_split, check_leakage


def test_group_split_no_leak_and_all_splits_present():
    groups = {"a": [f"a/{i}" for i in range(20)], "b": [f"b/{i}" for i in range(20)]}
    mp = group_split(groups, {"train": .7, "val": .15, "test": .15}, seed=1)
    df = pd.DataFrame({"group_id": list(mp), "split": list(mp.values())})
    assert check_leakage(df) == []
    assert set(df.split) == {"train", "val", "test"}


def test_detect_leakage():
    df = pd.DataFrame({"group_id": ["g1", "g1"], "split": ["train", "test"]})
    assert check_leakage(df)
