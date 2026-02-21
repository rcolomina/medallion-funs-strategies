"""
Tests for the GaussianHMM implementation.

Covers: initialization, emission computation, forward/backward algorithms,
Baum-Welch (EM), Viterbi decoding, and predict_proba.
"""

import numpy as np
import pytest

from renaissance_trading.hmm import GaussianHMM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_observations():
    """Two-regime data that is easy to separate."""
    rng = np.random.RandomState(0)
    low = rng.normal(-1.0, 0.2, 200)
    high = rng.normal(1.0, 0.2, 200)
    # Alternate blocks: low, high, low, high
    return np.concatenate([low[:100], high[:100], low[100:], high[100:]])


@pytest.fixture
def two_state_model():
    """Pre-configured 2-state model (not yet fitted)."""
    return GaussianHMM(n_states=2, n_iter=100, tol=1e-8)


@pytest.fixture
def fitted_model(simple_observations, two_state_model):
    """A 2-state model fitted on the simple two-regime data."""
    two_state_model.fit(simple_observations)
    return two_state_model


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_params(self):
        m = GaussianHMM()
        assert m.n_states == 3
        assert m.n_iter == 100
        assert m.tol == 1e-6

    def test_custom_params(self):
        m = GaussianHMM(n_states=5, n_iter=50, tol=1e-4, verbose=True)
        assert m.n_states == 5
        assert m.n_iter == 50
        assert m.verbose is True


# ---------------------------------------------------------------------------
# Parameter initialization
# ---------------------------------------------------------------------------

class TestInitParams:
    def test_pi_sums_to_one(self, two_state_model):
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        two_state_model._init_params(obs)
        assert pytest.approx(two_state_model.pi.sum()) == 1.0

    def test_transition_rows_sum_to_one(self, two_state_model):
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        two_state_model._init_params(obs)
        row_sums = two_state_model.A.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    def test_variances_positive(self, two_state_model):
        obs = np.array([1.0, 1.0, 1.0, 1.0])  # zero variance input
        two_state_model._init_params(obs)
        assert np.all(two_state_model.variances > 0)

    def test_diagonal_dominance(self, two_state_model):
        obs = np.arange(10, dtype=float)
        two_state_model._init_params(obs)
        for i in range(two_state_model.n_states):
            assert two_state_model.A[i, i] > 0.5


# ---------------------------------------------------------------------------
# Gaussian PDF
# ---------------------------------------------------------------------------

class TestGaussianPDF:
    def test_standard_normal_at_zero(self):
        # N(0|0,1) = 1/sqrt(2*pi) ~ 0.3989
        pdf = GaussianHMM._gaussian_pdf(0.0, 0.0, 1.0)
        assert pytest.approx(pdf, rel=1e-4) == 1.0 / np.sqrt(2 * np.pi)

    def test_symmetry(self):
        pdf_pos = GaussianHMM._gaussian_pdf(1.0, 0.0, 1.0)
        pdf_neg = GaussianHMM._gaussian_pdf(-1.0, 0.0, 1.0)
        assert pytest.approx(pdf_pos) == pdf_neg

    def test_vectorized(self):
        x = np.array([-1.0, 0.0, 1.0])
        result = GaussianHMM._gaussian_pdf(x, 0.0, 1.0)
        assert result.shape == (3,)
        assert np.all(result > 0)


# ---------------------------------------------------------------------------
# Emission matrix
# ---------------------------------------------------------------------------

class TestEmissionMatrix:
    def test_shape(self, two_state_model):
        obs = np.array([0.0, 1.0, 2.0])
        two_state_model._init_params(obs)
        B = two_state_model._compute_emission_matrix(obs)
        assert B.shape == (3, 2)

    def test_positive(self, two_state_model):
        obs = np.array([0.0, 1.0, 2.0])
        two_state_model._init_params(obs)
        B = two_state_model._compute_emission_matrix(obs)
        assert np.all(B > 0)


# ---------------------------------------------------------------------------
# Forward algorithm
# ---------------------------------------------------------------------------

class TestForward:
    def test_alpha_shape(self, fitted_model, simple_observations):
        B = fitted_model._compute_emission_matrix(simple_observations)
        alpha, scaling = fitted_model._forward(simple_observations, B)
        assert alpha.shape == (len(simple_observations), fitted_model.n_states)
        assert scaling.shape == (len(simple_observations),)

    def test_alpha_rows_sum_to_one(self, fitted_model, simple_observations):
        """Scaled forward variables should sum to ~1 at each time step."""
        B = fitted_model._compute_emission_matrix(simple_observations)
        alpha, _ = fitted_model._forward(simple_observations, B)
        row_sums = alpha.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_scaling_positive(self, fitted_model, simple_observations):
        B = fitted_model._compute_emission_matrix(simple_observations)
        _, scaling = fitted_model._forward(simple_observations, B)
        assert np.all(scaling > 0)


# ---------------------------------------------------------------------------
# Backward algorithm
# ---------------------------------------------------------------------------

class TestBackward:
    def test_beta_shape(self, fitted_model, simple_observations):
        B = fitted_model._compute_emission_matrix(simple_observations)
        _, scaling = fitted_model._forward(simple_observations, B)
        beta = fitted_model._backward(simple_observations, B, scaling)
        assert beta.shape == (len(simple_observations), fitted_model.n_states)

    def test_beta_nonnegative(self, fitted_model, simple_observations):
        B = fitted_model._compute_emission_matrix(simple_observations)
        _, scaling = fitted_model._forward(simple_observations, B)
        beta = fitted_model._backward(simple_observations, B, scaling)
        assert np.all(beta >= 0)


# ---------------------------------------------------------------------------
# E-step (gamma, xi)
# ---------------------------------------------------------------------------

class TestEStep:
    def test_gamma_sums_to_one(self, fitted_model, simple_observations):
        B = fitted_model._compute_emission_matrix(simple_observations)
        alpha, scaling = fitted_model._forward(simple_observations, B)
        beta = fitted_model._backward(simple_observations, B, scaling)
        gamma, _ = fitted_model._e_step(
            simple_observations, B, alpha, beta, scaling
        )
        row_sums = gamma.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_xi_shape(self, fitted_model, simple_observations):
        B = fitted_model._compute_emission_matrix(simple_observations)
        alpha, scaling = fitted_model._forward(simple_observations, B)
        beta = fitted_model._backward(simple_observations, B, scaling)
        _, xi = fitted_model._e_step(
            simple_observations, B, alpha, beta, scaling
        )
        T = len(simple_observations)
        N = fitted_model.n_states
        assert xi.shape == (T - 1, N, N)

    def test_xi_sums_to_one(self, fitted_model, simple_observations):
        """Each xi[t] should sum to ~1 over all (i,j)."""
        B = fitted_model._compute_emission_matrix(simple_observations)
        alpha, scaling = fitted_model._forward(simple_observations, B)
        beta = fitted_model._backward(simple_observations, B, scaling)
        _, xi = fitted_model._e_step(
            simple_observations, B, alpha, beta, scaling
        )
        for t in range(xi.shape[0]):
            assert pytest.approx(xi[t].sum(), abs=1e-6) == 1.0


# ---------------------------------------------------------------------------
# Baum-Welch (fit)
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_returns_self(self, two_state_model, simple_observations):
        result = two_state_model.fit(simple_observations)
        assert result is two_state_model

    def test_log_likelihood_exists(self, fitted_model):
        assert hasattr(fitted_model, "log_likelihood_")
        assert np.isfinite(fitted_model.log_likelihood_)

    def test_transition_matrix_valid(self, fitted_model):
        A = fitted_model.A
        assert A.shape == (2, 2)
        np.testing.assert_allclose(A.sum(axis=1), 1.0, atol=1e-10)
        assert np.all(A >= 0)

    def test_recovers_two_means(self, fitted_model):
        """Model should discover means near -1 and +1."""
        sorted_means = np.sort(fitted_model.means)
        assert sorted_means[0] < -0.5
        assert sorted_means[1] > 0.5

    def test_log_likelihood_nondecreasing(self, simple_observations):
        """EM theorem: log-likelihood must not decrease between iterations."""
        model = GaussianHMM(n_states=2, n_iter=5, tol=0)
        obs = np.asarray(simple_observations, dtype=np.float64)
        model._init_params(obs)

        prev_ll = -np.inf
        for _ in range(5):
            B = model._compute_emission_matrix(obs)
            alpha, scaling = model._forward(obs, B)
            ll = np.sum(np.log(np.maximum(scaling, 1e-300)))
            assert ll >= prev_ll - 1e-6  # allow tiny float rounding
            prev_ll = ll
            beta = model._backward(obs, B, scaling)
            gamma, xi = model._e_step(obs, B, alpha, beta, scaling)
            model._m_step(obs, gamma, xi)


# ---------------------------------------------------------------------------
# Viterbi decode
# ---------------------------------------------------------------------------

class TestDecode:
    def test_output_shape(self, fitted_model, simple_observations):
        states = fitted_model.decode(simple_observations)
        assert states.shape == (len(simple_observations),)

    def test_states_in_range(self, fitted_model, simple_observations):
        states = fitted_model.decode(simple_observations)
        assert np.all(states >= 0)
        assert np.all(states < fitted_model.n_states)

    def test_recovers_block_structure(self, fitted_model, simple_observations):
        """
        Data is [low]*100 + [high]*100 + [low]*100 + [high]*100.
        The decoded states should mostly change at indices ~100, ~200, ~300.
        """
        states = fitted_model.decode(simple_observations)
        # First 50 should be same state, last 50 should be same state
        assert len(set(states[:50])) == 1
        assert len(set(states[350:])) == 1
        # And they should differ
        assert states[50] != states[150]


# ---------------------------------------------------------------------------
# Predict proba
# ---------------------------------------------------------------------------

class TestPredictProba:
    def test_output_shape(self, fitted_model, simple_observations):
        gamma = fitted_model.predict_proba(simple_observations)
        assert gamma.shape == (len(simple_observations), fitted_model.n_states)

    def test_rows_sum_to_one(self, fitted_model, simple_observations):
        gamma = fitted_model.predict_proba(simple_observations)
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-6)

    def test_confident_in_clear_regimes(self, fitted_model, simple_observations):
        """In the middle of a clear block, model should be >90% confident."""
        gamma = fitted_model.predict_proba(simple_observations)
        # Middle of first low block (index ~50)
        assert gamma[50].max() > 0.9
        # Middle of first high block (index ~150)
        assert gamma[150].max() > 0.9


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_state(self):
        model = GaussianHMM(n_states=1, n_iter=10)
        obs = np.random.randn(50)
        model.fit(obs)
        states = model.decode(obs)
        assert np.all(states == 0)

    def test_short_sequence(self):
        model = GaussianHMM(n_states=2, n_iter=10)
        obs = np.array([0.0, 1.0, 0.0])
        model.fit(obs)
        states = model.decode(obs)
        assert len(states) == 3

    def test_constant_input(self):
        model = GaussianHMM(n_states=2, n_iter=10)
        obs = np.ones(50)
        model.fit(obs)
        assert np.all(np.isfinite(model.means))
        assert np.all(model.variances > 0)
