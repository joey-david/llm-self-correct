import math

from src.uq.metrics import accuracy, prr, roc_auc


def test_prr_improves_when_uncertain_samples_are_bad():
    scores = [0.9, 0.8, 0.1, 0.05]
    qualities = [0.0, 0.0, 1.0, 1.0]
    value = prr(scores, qualities, top_fraction=0.5)
    assert value > 0.0


def test_roc_auc_simple():
    scores = [0.9, 0.1, 0.8, 0.2]
    labels = [1, 0, 1, 0]
    assert math.isclose(roc_auc(scores, labels), 1.0)


def test_accuracy_strips_whitespace():
    preds = [" answer", "no"]
    refs = ["answer", "yes"]
    assert accuracy(preds, refs) == 0.5
