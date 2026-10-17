import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bdd100k_to_dataset as bdd


def test_mapping():
    assert bdd.CLASS_MAP == {"car": "mobil", "motor": "sepeda_motor", "bus": "bus", "truck": "truk"}
    assert bdd.TOD_MAP == {"daytime": "normal", "dawn/dusk": "dim", "night": "low"}


def test_group_of():
    assert bdd.group_of("0000f77c-6257be58.jpg") == "0000f77c"
    assert bdd.group_of("abcdef__0000f77c-6257be58__000.jpg") == "abcdef__0000f77c"


def test_reservoir_bounds():
    rng = random.Random(0)
    r = bdd.Reservoir(5, rng)
    for i in range(100):
        r.add(i)
    assert len(r.items) == 5
    r2 = bdd.Reservoir(10, rng)
    for i in range(3):
        r2.add(i)
    assert sorted(r2.items) == [0, 1, 2]


def test_split_groups_no_leak_and_all_splits():
    groups = [f"v{i}" for i in range(60)]
    group_classes = {g: {"mobil"} for g in groups}
    fracs = {"train": 0.7, "val": 0.15, "test": 0.15}
    assign = bdd.split_groups(groups, group_classes, fracs, seed=1)
    assert set(assign) == set(groups)
    assert set(assign.values()) == {"train", "val", "test"}
    per_split = {s: sum(1 for v in assign.values() if v == s) for s in fracs}
    assert per_split["train"] > per_split["test"]


def test_split_groups_respects_class_balance():
    groups = [f"v{i}" for i in range(40)]
    group_classes = {g: {"bus" if i % 2 else "mobil"} for i, g in enumerate(groups)}
    assign = bdd.split_groups(groups, group_classes, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=7)
    for cls in ("mobil", "bus"):
        counts = {s: sum(1 for g, c in group_classes.items() if cls in c and assign[g] == s)
                  for s in ("train", "val", "test")}
        assert counts["train"] > 0 and counts["val"] > 0 and counts["test"] > 0


class _Bucket:
    def __init__(self, items):
        self.items = items


def test_select_final_tags_correct_lighting_and_caps():
    buckets = {}
    for cls in bdd.CLASSES:
        for lt in bdd.LIGHTINGS:
            buckets[(cls, lt)] = _Bucket([(f"img_{cls}_{lt}_{i}", "grp", 0, 0, 10, 10)
                                          for i in range(4000)])
    final = bdd.select_final(buckets, per_class=1000, rng=random.Random(0))
    from collections import Counter
    counts = Counter((r[0], r[1]) for r in final)
    for cls in bdd.CLASSES:
        total = sum(counts[(cls, lt)] for lt in bdd.LIGHTINGS)
        assert total == 1000
        assert counts[(cls, "dim")] > 0 and counts[(cls, "normal")] > 0 and counts[(cls, "low")] > 0
        assert all(counts[(cls, lt)] <= 1000 for lt in bdd.LIGHTINGS)
