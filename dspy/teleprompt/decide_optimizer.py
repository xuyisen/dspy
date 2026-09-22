"""Numeric coordinate search for experimental Decide parameters."""

import math

from dspy.adapters.types.decision import Choice
from dspy.evaluate.evaluate import Evaluate
from dspy.predict.decide import Decide
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.utils.annotation import experimental


@experimental
class DecideOptimizer(Teleprompter):
    """Fit Decide parameters against a whole-program metric, without rewriting.

    Each coordinate sweep searches Boolean thresholds, Score numeric values
    (not cut points), and Choice probability multipliers. Only strict metric
    improvements are accepted; ties preserve the current configuration.
    This is bounded greedy search, not a guarantee of a global optimum.

    Args:
        metric: Per-example numeric metric to maximize, as in dspy.Evaluate.
        max_rounds: Maximum coordinate sweeps; stop early after a sweep without improvement.
        grid_size: Number of evenly spaced candidates, including both endpoints,
            for thresholds and Score values within the declared rubric range.
            Choice multipliers use 0.25, 0.5, 1, 2, and 4. Existing values are
            always retained unless a candidate improves the metric.
        num_threads: Evaluation concurrency, as in dspy.Evaluate.

    Provider caching may reuse identical requests. Changed intermediate values
    can cause new requests, and ordinary predictors still execute. No traces,
    prompts, demonstrations, or routing policies are fitted.
    """

    def __init__(self, *, metric, max_rounds=3, grid_size=21, num_threads=None):
        super().__init__()
        if type(max_rounds) is not int or max_rounds < 1:
            raise ValueError("max_rounds must be a positive integer.")
        if type(grid_size) is not int or grid_size < 2:
            raise ValueError("grid_size must be an integer of at least 2.")
        self.metric = metric
        self.max_rounds = max_rounds
        self.grid_size = grid_size
        self.num_threads = num_threads

    def compile(self, student, *, trainset):
        """Return an independently fitted copy; leave the student unchanged.

        Use a separate held-out set to assess generalization. Invalid parameters,
        program/metric errors, and nonfinite metric values fail compilation.
        """
        if not trainset:
            raise ValueError("trainset must contain at least one example.")
        program = student.deepcopy()
        decisions = [param for _, param in program.named_parameters() if isinstance(param, Decide)]
        if not decisions:
            raise ValueError("The student must contain at least one discoverable Decide parameter.")
        for decision in decisions:
            decision._validate_parameters(decision._output_types(decision.signature))
        evaluator = Evaluate(devset=trainset, metric=self.metric, num_threads=self.num_threads, max_errors=1)

        def evaluate():
            # Evaluate.score is rounded for display. Search must retain small improvements.
            scores = [float(score) for _, _, score in evaluator(program).results]
            if not all(math.isfinite(score) for score in scores):
                raise ValueError("DecideOptimizer requires finite metric values.")
            return math.fsum(score / len(scores) for score in scores)

        best_score = evaluate()
        for _ in range(self.max_rounds):
            previous_score = best_score
            for decision in decisions:
                for container, key, candidates in self._coordinates(decision):
                    best_value = container[key]
                    for candidate in candidates:
                        if candidate == best_value:
                            continue
                        container[key] = candidate
                        try:
                            score = evaluate()
                            if score > best_score:
                                best_score, best_value = score, candidate
                        finally:
                            container[key] = best_value
            if best_score == previous_score:
                break
        program._compiled = True
        return program

    def _coordinates(self, decision):
        fractions = [i / (self.grid_size - 1) for i in range(self.grid_size)]
        for name in decision.thresholds:
            yield decision.thresholds, name, fractions
        types = decision._output_types(decision.signature)
        for name, kind in types.items():
            if name not in decision.weights:
                continue
            if issubclass(kind, Choice):
                for label, _ in kind.options:
                    weights = decision.weights[name]
                    candidates = [{**weights, str(label): value} for value in (0.25, 0.5, 1.0, 2.0, 4.0)]
                    yield decision.weights, name, candidates
            else:
                low, high = kind.options[0][0], kind.options[-1][0]
                grid = [low * (1 - fraction) + high * fraction for fraction in fractions]
                for index in range(len(kind.options)):
                    weights = decision.weights[name]
                    candidates = [
                        [*weights[:index], value, *weights[index + 1 :]]
                        for value in grid
                        if (index == 0 or weights[index - 1] < value)
                        and (index == len(weights) - 1 or value < weights[index + 1])
                    ]
                    yield decision.weights, name, candidates
