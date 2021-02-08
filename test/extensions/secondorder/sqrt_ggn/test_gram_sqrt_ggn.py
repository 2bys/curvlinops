"""Tests for ``lowrank.extensions.secondorder.sqrt_ggn.gram_sqrt_ggn.py``."""

from test.implementation.autograd import AutogradExtensions
from test.implementation.backpack import BackpackExtensions
from test.problem import make_test_problems
from test.settings import SETTINGS
from test.utils import check_sizes_and_values, remove_zeros

import pytest

PROBLEMS = make_test_problems(SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_gram_sqrt_ggn_spectrum(problem):
    """Compare spectrum of full GGN with GGN Gram matrix."""
    problem.set_up()

    ggn_mat = AutogradExtensions(problem).ggn()
    gram_mat = BackpackExtensions(problem).gram_sqrt_ggn()

    ggn_evals, _ = ggn_mat.symeig()
    gram_evals, _ = gram_mat.symeig()

    rtol, atol = 1e-5, 1e-6
    filtered_ggn_evals = remove_zeros(ggn_evals, rtol=rtol, atol=atol)
    filtered_gram_evals = remove_zeros(gram_evals, rtol=rtol, atol=atol)

    rtol, atol = 1e-5, 1e-7
    check_sizes_and_values(
        filtered_ggn_evals, filtered_gram_evals, rtol=rtol, atol=atol
    )
    problem.tear_down()


MC_SAMPLES = [1, 5, 10]
MC_SAMPLES_IDS = [f"mc_samples={s}" for s in MC_SAMPLES]


@pytest.mark.parametrize("mc_samples", MC_SAMPLES, ids=MC_SAMPLES_IDS)
@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_gram_sqrt_ggn_mc_spectrum(problem, mc_samples):
    """Compare spectrum of full GGNMC with GGNMC Gram matrix."""
    problem.set_up()
    ggn_mat = BackpackExtensions(problem).ggn_mc(mc_samples)
    problem.tear_down()

    # need another set_up to reset the random seed for the MC samples
    problem.set_up()
    gram_mat = BackpackExtensions(problem).gram_sqrt_ggn_mc(mc_samples)
    problem.tear_down()

    ggn_evals, _ = ggn_mat.symeig()
    gram_evals, _ = gram_mat.symeig()

    rtol, atol = 1e-5, 1e-6
    filtered_ggn_evals = remove_zeros(ggn_evals, rtol=rtol, atol=atol)
    filtered_gram_evals = remove_zeros(gram_evals, rtol=rtol, atol=atol)

    rtol, atol = 1e-5, 1e-7
    check_sizes_and_values(
        filtered_ggn_evals, filtered_gram_evals, rtol=rtol, atol=atol
    )