import numpy as np

from scripts import seed_ensemble_analysis as m


def test_soft_vote_uses_mean_margin():
    z = np.array([[3., -1.], [-1., -1.], [-1., 1.5]])
    assert m.soft_vote(z).tolist() == [1, 0]


def test_agreement_counts_majority():
    z = np.array([[1., 1., -1.], [1., -1., -1.], [1., 1., -1.]])
    assert m.agreement(z).tolist() == [3, 2, 3]


def test_seed_agree_prefers_unanimous_modality_and_ties_to_text():
    # subject 0: text 2/3 positive (vote 1), A+T 3/3 negative -> A+T (0)
    # subject 1: text 3/3 positive, A+T 2/3 negative -> text (1)
    # subject 2: both 2/3, disagree -> text
    zt = np.array([[1., 1., 1.], [1., 1., -1.], [-1., 1., 1.]])
    za = np.array([[-1., -1., 1.], [-1., -1., -1.], [-1., 1., -1.]])
    assert m.seed_agree(zt, za).tolist() == [0, 1, 1]


def test_seed_agree_equals_text_when_ensembles_agree():
    zt = np.array([[1., -1.], [1., -1.], [-1., -1.]])
    za = np.array([[1., -1.], [1., -1.], [1., -1.]])
    assert m.seed_agree(zt, za).tolist() == m.soft_vote(zt).tolist()
