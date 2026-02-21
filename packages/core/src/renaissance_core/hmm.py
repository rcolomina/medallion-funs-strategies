"""
Hidden Markov Model with Gaussian emissions — implemented from scratch.

Implements:
    - Forward algorithm (alpha)
    - Backward algorithm (beta)
    - Baum-Welch (EM) for parameter learning
    - Viterbi for most likely state decoding
    - Log-space computation for numerical stability

Parameters lambda = (A, B, pi) where:
    A  = transition matrix (n_states x n_states)
    B  = emission params: means mu and variances sigma^2 per state
    pi = initial state distribution
"""

import numpy as np


class GaussianHMM:
    """
    Hidden Markov Model with Gaussian emissions.

    Parameters
    ----------
    n_states : int
        Number of hidden states.
    n_iter : int
        Maximum number of Baum-Welch iterations.
    tol : float
        Convergence tolerance for log-likelihood.
    verbose : bool
        Print progress during fitting.
    """

    def __init__(self, n_states=3, n_iter=100, tol=1e-6, verbose=False):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose

    def _init_params(self, observations):
        """Initialize parameters using quantile-based heuristic."""
        N = self.n_states

        # pi: uniform initial distribution
        self.pi = np.ones(N) / N

        # A: slight preference for staying in current state
        if N == 1:
            self.A = np.ones((1, 1))
        else:
            self.A = np.full((N, N), 0.1 / (N - 1))
            np.fill_diagonal(self.A, 0.9)
            self.A /= self.A.sum(axis=1, keepdims=True)

        # B: Initialize means by splitting data into quantiles
        sorted_obs = np.sort(observations)
        quantiles = np.array_split(sorted_obs, N)
        self.means = np.array([q.mean() for q in quantiles])
        self.variances = np.array([max(q.var(), 1e-6) for q in quantiles])

    @staticmethod
    def _gaussian_pdf(x, mean, var):
        """
        Gaussian emission probability:
        b_i(o_t) = N(o_t | mu_i, sigma_i^2)
        """
        return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(
            -0.5 * (x - mean) ** 2 / var
        )

    def _compute_emission_matrix(self, observations):
        """
        Compute B matrix: B[t, j] = b_j(o_t).
        Shape: (T, N)
        """
        T = len(observations)
        N = self.n_states
        B = np.zeros((T, N))
        for j in range(N):
            B[:, j] = self._gaussian_pdf(
                observations, self.means[j], self.variances[j]
            )
        # Floor to avoid log(0)
        B = np.maximum(B, 1e-300)
        return B

    def _forward(self, observations, B):
        """
        Forward algorithm with scaling for numerical stability.

        alpha_t(i) = P(o_1, ..., o_t, q_t = S_i | lambda)

        Returns
        -------
        alpha_hat : ndarray (T, N)
            Scaled forward variables.
        scaling : ndarray (T,)
            Scaling factors.
        """
        T = len(observations)
        N = self.n_states
        alpha_hat = np.zeros((T, N))
        scaling = np.zeros(T)

        # Initialization: alpha_1(i) = pi_i * b_i(o_1)
        alpha_hat[0] = self.pi * B[0]
        scaling[0] = alpha_hat[0].sum()
        if scaling[0] == 0:
            scaling[0] = 1e-300
        alpha_hat[0] /= scaling[0]

        # Induction: alpha_t(j) = [sum_i alpha_{t-1}(i) * a_ij] * b_j(o_t)
        for t in range(1, T):
            for j in range(N):
                alpha_hat[t, j] = np.sum(alpha_hat[t - 1] * self.A[:, j]) * B[t, j]
            scaling[t] = alpha_hat[t].sum()
            if scaling[t] == 0:
                scaling[t] = 1e-300
            alpha_hat[t] /= scaling[t]

        return alpha_hat, scaling

    def _backward(self, observations, B, scaling):
        """
        Backward algorithm with scaling.

        beta_t(i) = P(o_{t+1}, ..., o_T | q_t = S_i, lambda)

        Returns
        -------
        beta_hat : ndarray (T, N)
            Scaled backward variables.
        """
        T = len(observations)
        N = self.n_states
        beta_hat = np.zeros((T, N))

        # Initialization: beta_T(i) = 1 (scaled)
        beta_hat[T - 1] = 1.0 / scaling[T - 1]

        # Induction: beta_t(i) = sum_j a_ij * b_j(o_{t+1}) * beta_{t+1}(j)
        for t in range(T - 2, -1, -1):
            for i in range(N):
                beta_hat[t, i] = np.sum(
                    self.A[i, :] * B[t + 1, :] * beta_hat[t + 1, :]
                )
            if scaling[t] > 0:
                beta_hat[t] /= scaling[t]

        return beta_hat

    def _e_step(self, observations, B, alpha_hat, beta_hat, scaling):
        """
        E-Step of Baum-Welch: compute gamma and xi.

        gamma_t(i) = P(q_t = S_i | O, lambda)
        xi_t(i,j)  = P(q_t = S_i, q_{t+1} = S_j | O, lambda)
        """
        T = len(observations)
        N = self.n_states

        # gamma_t(i) using scaled variables
        gamma = alpha_hat * beta_hat * scaling[:, np.newaxis]
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum = np.maximum(gamma_sum, 1e-300)
        gamma = gamma / gamma_sum

        # xi_t(i,j)
        xi = np.zeros((T - 1, N, N))
        for t in range(T - 1):
            for i in range(N):
                for j in range(N):
                    xi[t, i, j] = (
                        alpha_hat[t, i]
                        * self.A[i, j]
                        * B[t + 1, j]
                        * beta_hat[t + 1, j]
                    )
            xi_sum = xi[t].sum()
            if xi_sum > 0:
                xi[t] /= xi_sum

        return gamma, xi

    def _m_step(self, observations, gamma, xi):
        """
        M-Step of Baum-Welch: re-estimate parameters.

        pi_i   = gamma_1(i)
        a_ij   = sum_t xi_t(i,j) / sum_t gamma_t(i)
        mu_i   = sum_t gamma_t(i)*o_t / sum_t gamma_t(i)
        var_i  = sum_t gamma_t(i)*(o_t - mu_i)^2 / sum_t gamma_t(i)
        """
        N = self.n_states

        # Update pi
        self.pi = gamma[0] / gamma[0].sum()

        # Update A
        for i in range(N):
            denom = gamma[:-1, i].sum()
            if denom > 0:
                for j in range(N):
                    self.A[i, j] = xi[:, i, j].sum() / denom
            else:
                self.A[i] = 1.0 / N
        self.A /= self.A.sum(axis=1, keepdims=True)

        # Update emission parameters
        for i in range(N):
            weight_sum = gamma[:, i].sum()
            if weight_sum > 0:
                self.means[i] = np.sum(gamma[:, i] * observations) / weight_sum
                diff = observations - self.means[i]
                self.variances[i] = np.sum(gamma[:, i] * diff**2) / weight_sum
                self.variances[i] = max(self.variances[i], 1e-6)

    def fit(self, observations):
        """
        Baum-Welch algorithm (EM) to learn HMM parameters.

        Iterates E-step and M-step until convergence.
        Guarantees P(O|lambda) is non-decreasing (by EM theorem).

        Parameters
        ----------
        observations : array-like
            1-D sequence of observations.

        Returns
        -------
        self
        """
        observations = np.asarray(observations, dtype=np.float64)
        self._init_params(observations)

        prev_log_likelihood = -np.inf

        for iteration in range(self.n_iter):
            B = self._compute_emission_matrix(observations)
            alpha_hat, scaling = self._forward(observations, B)

            # Log-likelihood: log P(O|lambda) = sum_t log(c_t)
            log_likelihood = np.sum(np.log(np.maximum(scaling, 1e-300)))

            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                if self.verbose:
                    print(f"  Converged at iteration {iteration}")
                break
            prev_log_likelihood = log_likelihood

            if self.verbose and iteration % 10 == 0:
                print(
                    f"  Iter {iteration:3d}: log-likelihood = {log_likelihood:.4f}"
                )

            beta_hat = self._backward(observations, B, scaling)
            gamma, xi = self._e_step(
                observations, B, alpha_hat, beta_hat, scaling
            )
            self._m_step(observations, gamma, xi)

        self.log_likelihood_ = log_likelihood
        return self

    def decode(self, observations):
        """
        Viterbi algorithm — find the most likely hidden state sequence.

        Uses log-space for numerical stability:
            delta_t(j) = max_i [delta_{t-1}(i) + log a_ij] + log b_j(o_t)

        Parameters
        ----------
        observations : array-like
            1-D sequence of observations.

        Returns
        -------
        states : ndarray of int
            Most likely state sequence.
        """
        observations = np.asarray(observations, dtype=np.float64)
        T = len(observations)
        N = self.n_states
        B = self._compute_emission_matrix(observations)

        log_A = np.log(np.maximum(self.A, 1e-300))
        log_B = np.log(np.maximum(B, 1e-300))
        log_pi = np.log(np.maximum(self.pi, 1e-300))

        delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)

        # Initialization
        delta[0] = log_pi + log_B[0]

        # Recursion
        for t in range(1, T):
            for j in range(N):
                candidates = delta[t - 1] + log_A[:, j]
                psi[t, j] = np.argmax(candidates)
                delta[t, j] = candidates[psi[t, j]] + log_B[t, j]

        # Backtracking
        states = np.zeros(T, dtype=int)
        states[T - 1] = np.argmax(delta[T - 1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def predict_proba(self, observations):
        """
        Return gamma_t(i) — posterior state probabilities at each time step.

        Parameters
        ----------
        observations : array-like
            1-D sequence of observations.

        Returns
        -------
        gamma : ndarray (T, N)
            Posterior probability of each state at each time step.
        """
        observations = np.asarray(observations, dtype=np.float64)
        B = self._compute_emission_matrix(observations)
        alpha_hat, scaling = self._forward(observations, B)
        beta_hat = self._backward(observations, B, scaling)
        gamma, _ = self._e_step(observations, B, alpha_hat, beta_hat, scaling)
        return gamma
