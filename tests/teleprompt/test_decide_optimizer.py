from typing import Annotated, Literal

import pytest

import dspy
from dspy.experimental import Choice, Decide, DecideOptimizer, Noul, Score
from dspy.utils.callback import BaseCallback
from dspy.utils.dummies import DummyLM


class Client:
    """Deterministic provider boundary; inputs select fixed raw distributions."""

    def __call__(self, state, questions):
        answers = {}
        for name, question in questions.items():
            if question["type"] == "noul":
                answers[name] = {"noul": state["p"]}
            elif question["type"] == "score":
                answers[name] = {"confidence": 0.61, "probabilities": state["p"]}
            else:
                answers[name] = {"choice": "2", "confidence": 0.73, "probabilities": state["p"]}
        return answers


def examples(*rows):
    return [dspy.Example(p=p, answer=answer).with_inputs("p") for p, answer in rows]


def fit(student, trainset, metric=None, **kwargs):
    metric = metric or (lambda example, prediction: prediction.answer == example.answer)
    with dspy.context(system_one=Client()):
        return DecideOptimizer(metric=metric, num_threads=2, **kwargs).compile(student, trainset=trainset)


@pytest.mark.parametrize("rich", [False, True])
def test_threshold_fitting_preserves_student_traces_and_state(tmp_path, rich):
    student = Decide(
        dspy.Signature({"p": (float, dspy.InputField()), "answer": (Noul if rich else bool, dspy.OutputField())})
    )
    data = examples((0.6, False), (0.7, True), (0.9, True))

    def metric(example, prediction):
        return bool(prediction.answer) == example.answer

    optimized = fit(student, data, metric)
    assert optimized.thresholds["answer"] == 0.65
    assert student.thresholds == {"answer": 0.5}
    assert optimized.signature is student.signature
    assert optimized.named_predictors() == []
    assert optimized.named_parameters() == [("self", optimized)]
    assert optimized._compiled

    optimized.save(tmp_path / "state.json")
    restored = student.deepcopy()
    restored.load(tmp_path / "state.json")
    assert restored.thresholds == optimized.thresholds
    assert restored.reset_copy().thresholds == optimized.thresholds
    with dspy.context(system_one=Client(), trace=[]):
        assert bool(restored(p=0.65).answer) is True
        assert bool(restored(p=0.649).answer) is False
        assert [entry[0] for entry in dspy.settings.trace] == [restored, restored]


@pytest.mark.parametrize("p,target,threshold", [(0, True, 0), (0.99, False, 1)])
def test_threshold_search_includes_endpoints(p, target, threshold):
    optimized = fit(Decide("p: float -> answer: bool"), examples((p, target)))
    assert optimized.thresholds["answer"] == threshold


@pytest.mark.parametrize("rich", [False, True])
def test_score_fits_numeric_values_not_cuts(tmp_path, rich):
    rating = Score[(-2, "bad"), (3, "fair"), (10, "great")]
    student = Decide(
        dspy.Signature(
            {
                "p": (dict[int, float], dspy.InputField()),
                "answer": (rating if rich else Annotated[float, rating], dspy.OutputField()),
            }
        )
    )
    data = examples(
        ({0: 1, 1: 0, 2: 0}, -2),
        ({0: 0, 1: 0, 2: 1}, 10),
        ({0: 0.1, 1: 0.3, 2: 0.6}, 7.6),
    )

    def metric(example, prediction):
        return -abs(float(prediction.answer) - example.answer)

    optimized = fit(student, data, metric, grid_size=13)
    assert optimized.weights["answer"] == pytest.approx([-2, 6, 10])
    assert student.weights["answer"] == [-2, 3, 10]
    assert rating.options == ((-2, "bad"), (3, "fair"), (10, "great"))
    optimized.save(tmp_path / "score.json")
    restored = student.deepcopy()
    restored.load(tmp_path / "score.json")
    with dspy.context(system_one=Client()):
        result = restored(p={0: 0.2, 1: 0.7, 2: 0.1}).answer
        assert float(result) == pytest.approx(4.8)
        if rich:
            assert result.probabilities == {0: 0.2, 1: 0.7, 2: 0.1}
            assert result.confidence == 0.61


def test_score_endpoint_values_can_move_inward():
    rating = Score[(-2, "low"), (10, "high")]
    student = Decide(
        dspy.Signature(
            {
                "p": (dict[int, float], dspy.InputField()),
                "answer": (rating, dspy.OutputField()),
            }
        )
    )
    optimized = fit(
        student,
        examples(({0: 1, 1: 0}, 0), ({0: 0, 1: 1}, 8)),
        lambda example, prediction: -abs(float(prediction.answer) - example.answer),
        grid_size=13,
    )
    assert optimized.weights["answer"] == pytest.approx([0, 8])


@pytest.mark.parametrize("rich", [False, True])
def test_choice_fits_multipliers_and_keeps_raw_evidence(tmp_path, rich):
    label = Choice[(2, "first"), ("other", "second")]
    student = Decide(
        dspy.Signature(
            {
                "p": (dict[str, float], dspy.InputField()),
                "answer": (label if rich else Literal[2, "other"], dspy.OutputField()),
            }
        )
    )
    student.weights["answer"] = {}  # Omitted multipliers have effective weight 1.
    data = examples(({"2": 0.7, "other": 0.3}, "other"), ({"2": 0.97, "other": 0.03}, 2))

    def metric(example, prediction):
        return (prediction.answer.value if rich else prediction.answer) == example.answer

    optimized = fit(student, data, metric)
    assert optimized.weights["answer"] == {"2": 0.25}
    assert student.weights["answer"] == {}
    optimized.save(tmp_path / "choice.json")
    restored = student.deepcopy()
    restored.load(tmp_path / "choice.json")
    with dspy.context(system_one=Client()):
        result = restored(p={"2": 0.7, "other": 0.3}).answer
        assert (result.value if rich else result) == "other"
        if rich:
            assert result.probabilities == {"2": 0.7, "other": 0.3}
            assert result.confidence == 0.73


def test_sweeps_revisit_interacting_fields():
    student = Decide("p: float -> first: bool, second: bool")
    objective = {(True, True): 0, (False, True): -1, (True, False): 1, (False, False): 2}

    def metric(example, prediction):
        return objective[prediction.first, prediction.second]

    one = fit(student, examples((0.5, None)), metric, grid_size=3, max_rounds=1)
    two = fit(student, examples((0.5, None)), metric, grid_size=3, max_rounds=2)
    assert one.thresholds == {"first": 0.5, "second": 1}
    assert two.thresholds == {"first": 1, "second": 1}


def test_sub_display_precision_improvements_are_not_rounded_away():
    student = Decide("p: float -> answer: bool")

    def metric(example, prediction):
        return 0.000001 if prediction.answer else 0

    optimized = fit(student, examples((0.1, True)), metric)
    assert optimized.thresholds["answer"] == 0


def test_mixed_program_optimizes_final_metric_and_leaves_predict_and_frozen_module_alone():
    class Frozen(dspy.Module):
        def __init__(self):
            self.decide = Decide("p: float -> answer: bool")
            self._compiled = True

    class Program(dspy.Module):
        def __init__(self):
            decision = Decide("p: float -> answer: bool")
            self.steps = [decision, decision]
            self.frozen = Frozen()
            self.predict = dspy.Predict("p: float -> answer")

        def forward(self, p):
            return dspy.Prediction(answer=self.predict(p=p).answer if self.steps[0](p=p).answer else "skip")

    student = Program()
    student.predict.demos = [dspy.Example(p=0.8, answer="keep").with_inputs("p")]
    with dspy.context(lm=DummyLM([{"answer": "keep"}] * 100)):
        optimized = fit(student, examples((0.6, "skip"), (0.7, "keep")))
    assert optimized.steps[0].thresholds == {"answer": 0.65}
    assert optimized.steps[0] is optimized.steps[1]
    assert optimized.steps[0] is not student.steps[0]
    assert optimized.frozen.decide.thresholds == {"answer": 0.5}
    assert optimized.predict.dump_state() == student.predict.dump_state()


def test_ties_keep_non_grid_parameters_and_stop_after_one_sweep():
    student = Decide("p: float -> answer: bool")
    student.thresholds["answer"] = 0.731
    calls = []

    def metric(example, prediction):
        calls.append(prediction)
        return 1

    optimized = fit(student, examples((0.8, True)), metric, grid_size=3, max_rounds=8)
    assert optimized.thresholds == {"answer": 0.731}
    assert len(calls) == 4  # Baseline plus three candidates; no unproductive second sweep.


@pytest.mark.parametrize("kwargs", [{"grid_size": 1}, {"grid_size": 2.5}, {"max_rounds": 0}, {"max_rounds": True}])
def test_invalid_search_configuration(kwargs):
    with pytest.raises(ValueError):
        DecideOptimizer(metric=lambda e, p: 0, **kwargs)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_metrics_fail(score):
    with pytest.raises(ValueError, match="finite metric"):
        fit(Decide("p: float -> answer: bool"), examples((0.5, True)), lambda e, p: score)


def test_invalid_student_empty_data_and_metric_failure():
    student = Decide("p: float -> answer: bool")
    with pytest.raises(ValueError, match="trainset"):
        fit(student, [])
    with pytest.raises(ValueError, match="discoverable Decide"):
        fit(dspy.Predict("p -> answer"), examples((0.5, True)))
    student.thresholds["answer"] = -1
    with pytest.raises(ValueError, match="Threshold"):
        fit(student, examples((0.5, True)))
    student.thresholds["answer"] = 0.5

    def broken_metric(example, prediction):
        raise RuntimeError("metric failed")

    with pytest.raises(Exception, match="Execution cancelled"):
        fit(student, examples((0.5, True)), broken_metric)
    assert student.thresholds == {"answer": 0.5}


def test_compile_uses_teleprompter_callbacks():
    class Callback(BaseCallback):
        def __init__(self):
            self.events = []

        def on_compile_start(self, call_id, instance, inputs):
            self.events.append("start")

        def on_compile_end(self, call_id, outputs, exception=None):
            self.events.append("end")

    callback = Callback()
    with dspy.context(callbacks=[callback]):
        fit(Decide("p: float -> answer: bool"), examples((0.5, True)), grid_size=2)
    assert callback.events == ["start", "end"]
