from __future__ import annotations

from aegis.data.embodiments import get_embodiment, list_embodiments
from aegis.eval.learned import recommend_auto, recommend_learned


class TestEmbodiments:
    def test_two_entries(self) -> None:
        assert set(list_embodiments()) == {"franka", "ur5e"}
        assert get_embodiment("franka")["validated"] is True
        assert get_embodiment("ur5e")["validated"] is False

    def test_unknown_raises(self) -> None:
        try:
            get_embodiment("humanoid")
        except KeyError:
            return
        raise AssertionError("unknown embodiment must raise")


class TestLearned:
    def test_placeholder_falls_back(self) -> None:
        summary = {"episodes": 3, "successes": 3, "fails": 0, "safety_violations": 0,
                   "recovery_events": 0, "truncated": 0}
        features = {"success_rate": 1.0, "violations_per_ep": 0.0, "recoveries_per_ep": 0.0,
                    "truncated_rate": 0.0}
        structured, model = recommend_auto(summary, features)
        assert model == "heuristic_v1" and structured["label"] == "ready"

    def test_trained_model_scores(self, tmp_path) -> None:
        import json

        mdl = {"trained": True, "bias": 0.0,
               "weights": {"success_rate": 2.0, "violations_per_ep": -0.4,
                           "recoveries_per_ep": 0.0, "truncated_rate": 0.0},
               "thresholds": [-0.5, 0.5], "labels": ["a", "b", "c"]}
        p = tmp_path / "m.json"
        p.write_text(json.dumps(mdl), encoding="utf-8")
        import aegis.eval.learned as L

        old = L.MODEL_PATH
        L.MODEL_PATH = p
        try:
            out = recommend_learned({"episodes": 1}, {"success_rate": 1.0, "violations_per_ep": 0.0,
                                                     "recoveries_per_ep": 0.0, "truncated_rate": 0.0})
            assert out is not None and out["label"] == "c"
        finally:
            L.MODEL_PATH = old
