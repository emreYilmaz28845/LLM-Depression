import numpy as np
import pytest

from scripts import model_selector_analysis as m


def test_ola_trusts_locally_better_model_and_ties_to_text():
    xsel = np.array([[1., 0.], [1., .1], [-1., 0.], [-1., -.1]])
    ysel = np.array([1, 1, 0, 0])
    pt_sel = np.array([0, 0, 0, 0])  # text wrong near +x, right near -x
    pa_sel = np.array([1, 1, 1, 1])  # A+T right near +x, wrong near -x
    xq = np.array([[1., .05], [-1., -.05]])
    pt, pa = np.array([0, 0]), np.array([1, 1])
    pred, ct, ca = m.ola(xsel, ysel, pt_sel, pa_sel, xq, pt, pa, k=2)
    assert pred.tolist() == [1, 0]
    # equal competence -> text
    pred, _, _ = m.ola(xsel, ysel, pt_sel, pt_sel, xq, pt, pa, k=2)
    assert pred.tolist() == [0, 0]


def test_lca_uses_neighbours_of_predicted_class_only():
    xsel = np.array([[1., 0.], [1., .1], [-1., 0.], [-1., -.1]])
    ysel = np.array([1, 1, 0, 0])
    pt_sel = np.array([1, 1, 1, 1])
    pa_sel = np.array([0, 0, 0, 0])
    xq = np.array([[1., .05]])
    # text predicts 1: class-1 neighbours, text right on both -> 1.0
    # A+T predicts 0: class-0 neighbours, A+T right on both -> 1.0; tie -> text
    pred, ct, ca = m.lca(xsel, ysel, pt_sel, pa_sel, xq, np.array([1]), np.array([0]), k=2)
    assert ct.tolist() == [1.] and ca.tolist() == [1.] and pred.tolist() == [1]


def test_vote3_majority():
    assert m.vote3(np.array([1, 0, 1]), np.array([0, 0, 1]), np.array([1, 1, 0])).tolist() == [1, 0, 1]


def test_stack_rejects_single_class():
    with pytest.raises(ValueError):
        m.stack(np.zeros((4, 3)), np.ones(4, int), np.zeros((1, 3)))


def test_stack_learns_sign():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(60, 3)); y = (z[:, 0] > 0).astype(int)
    pred, coef = m.stack(z, y, np.array([[2., 0., 0.], [-2., 0., 0.]]))
    assert pred.tolist() == [1, 0] and coef[0] > 0


def test_neighbours_capped_by_selection_size():
    assert m.neighbours(np.eye(3) + .1, np.ones((1, 3)), 7).shape == (1, 3)
