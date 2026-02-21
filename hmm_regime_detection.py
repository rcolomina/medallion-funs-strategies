"""
Hidden Markov Model for Market Regime Detection
================================================
Full implementation from scratch + hmmlearn comparison.

This implements the core mathematics that Renaissance Technologies
built upon — Hidden Markov Models with the Baum-Welch algorithm
for detecting hidden market regimes from observable price data.

Author: Claude (for Ruben)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


# =============================================================================
# PART 1: HMM FROM SCRATCH — Full Baum-Welch Implementation
# =============================================================================

class GaussianHMM:
    """
    Hidden Markov Model with Gaussian emissions.
    
    Implements:
        - Forward algorithm (α)
        - Backward algorithm (β)
        - Baum-Welch (EM) for parameter learning
        - Viterbi for most likely state decoding
        - Log-space computation for numerical stability
    
    Parameters λ = (A, B, π) where:
        A  = transition matrix (n_states × n_states)
        B  = emission params: means μ and variances σ² per state
        π  = initial state distribution
    """
    
    def __init__(self, n_states=3, n_iter=100, tol=1e-6, verbose=False):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        
    def _init_params(self, observations):
        """Initialize parameters using k-means-style heuristic."""
        N = self.n_states
        T = len(observations)
        
        # π: uniform initial distribution
        self.pi = np.ones(N) / N
        
        # A: slight preference for staying in current state
        self.A = np.full((N, N), 0.1 / (N - 1))
        np.fill_diagonal(self.A, 0.9)
        # Normalize rows
        self.A /= self.A.sum(axis=1, keepdims=True)
        
        # B: Initialize means by splitting data into quantiles
        sorted_obs = np.sort(observations)
        quantiles = np.array_split(sorted_obs, N)
        self.means = np.array([q.mean() for q in quantiles])
        self.variances = np.array([max(q.var(), 1e-6) for q in quantiles])
        
    def _gaussian_pdf(self, x, mean, var):
        """
        Gaussian emission probability:
        b_i(o_t) = N(o_t | μ_i, σ_i²) = (1/√(2πσ²)) · exp(-(x-μ)²/(2σ²))
        """
        return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * (x - mean)**2 / var)
    
    def _compute_emission_matrix(self, observations):
        """
        Compute B matrix: B[t, j] = b_j(o_t)
        Shape: (T, N)
        """
        T = len(observations)
        N = self.n_states
        B = np.zeros((T, N))
        for j in range(N):
            B[:, j] = self._gaussian_pdf(observations, self.means[j], self.variances[j])
        # Floor to avoid log(0)
        B = np.maximum(B, 1e-300)
        return B
    
    # ----- FORWARD ALGORITHM -----
    def _forward(self, observations, B):
        """
        Forward algorithm with scaling for numerical stability.
        
        α_t(i) = P(o_1, ..., o_t, q_t = S_i | λ)
        
        With scaling:
            α̃_t(i) = α_t(i) / c_t
            where c_t = Σ_i α_t(i)  [scaling factor]
        
        Returns:
            alpha_hat: scaled forward variables (T, N)
            scaling:   scaling factors (T,)
        """
        T = len(observations)
        N = self.n_states
        alpha_hat = np.zeros((T, N))
        scaling = np.zeros(T)
        
        # Initialization: α_1(i) = π_i · b_i(o_1)
        alpha_hat[0] = self.pi * B[0]
        scaling[0] = alpha_hat[0].sum()
        if scaling[0] == 0:
            scaling[0] = 1e-300
        alpha_hat[0] /= scaling[0]
        
        # Induction: α_t(j) = [Σ_i α_{t-1}(i) · a_ij] · b_j(o_t)
        for t in range(1, T):
            for j in range(N):
                alpha_hat[t, j] = np.sum(alpha_hat[t-1] * self.A[:, j]) * B[t, j]
            scaling[t] = alpha_hat[t].sum()
            if scaling[t] == 0:
                scaling[t] = 1e-300
            alpha_hat[t] /= scaling[t]
        
        return alpha_hat, scaling
    
    # ----- BACKWARD ALGORITHM -----
    def _backward(self, observations, B, scaling):
        """
        Backward algorithm with scaling.
        
        β_t(i) = P(o_{t+1}, ..., o_T | q_t = S_i, λ)
        
        Returns:
            beta_hat: scaled backward variables (T, N)
        """
        T = len(observations)
        N = self.n_states
        beta_hat = np.zeros((T, N))
        
        # Initialization: β_T(i) = 1 (scaled)
        beta_hat[T-1] = 1.0 / scaling[T-1]
        
        # Induction: β_t(i) = Σ_j a_ij · b_j(o_{t+1}) · β_{t+1}(j)
        for t in range(T-2, -1, -1):
            for i in range(N):
                beta_hat[t, i] = np.sum(
                    self.A[i, :] * B[t+1, :] * beta_hat[t+1, :]
                )
            if scaling[t] > 0:
                beta_hat[t] /= scaling[t]
        
        return beta_hat
    
    # ----- E-STEP: Compute γ and ξ -----
    def _e_step(self, observations, B, alpha_hat, beta_hat, scaling):
        """
        E-Step of Baum-Welch.
        
        γ_t(i) = P(q_t = S_i | O, λ)
               = α_t(i) · β_t(i) / P(O|λ)
               
        ξ_t(i,j) = P(q_t = S_i, q_{t+1} = S_j | O, λ)
                  = α_t(i) · a_ij · b_j(o_{t+1}) · β_{t+1}(j) / P(O|λ)
        """
        T = len(observations)
        N = self.n_states
        
        # γ_t(i) — using scaled variables
        gamma = alpha_hat * beta_hat * scaling[:, np.newaxis]
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum = np.maximum(gamma_sum, 1e-300)
        gamma = gamma / gamma_sum
        
        # ξ_t(i,j)
        xi = np.zeros((T-1, N, N))
        for t in range(T-1):
            for i in range(N):
                for j in range(N):
                    xi[t, i, j] = (alpha_hat[t, i] * self.A[i, j] * 
                                   B[t+1, j] * beta_hat[t+1, j])
            xi_sum = xi[t].sum()
            if xi_sum > 0:
                xi[t] /= xi_sum
        
        return gamma, xi
    
    # ----- M-STEP: Re-estimate parameters -----
    def _m_step(self, observations, gamma, xi):
        """
        M-Step of Baum-Welch.
        
        π̂_i = γ_1(i)
        
        â_ij = Σ_t ξ_t(i,j) / Σ_t γ_t(i)
        
        μ̂_i = Σ_t γ_t(i)·o_t / Σ_t γ_t(i)
        
        σ̂²_i = Σ_t γ_t(i)·(o_t - μ̂_i)² / Σ_t γ_t(i)
        """
        T = len(observations)
        N = self.n_states
        
        # Update π: initial state probabilities
        self.pi = gamma[0] / gamma[0].sum()
        
        # Update A: transition matrix
        for i in range(N):
            denom = gamma[:-1, i].sum()
            if denom > 0:
                for j in range(N):
                    self.A[i, j] = xi[:, i, j].sum() / denom
            else:
                self.A[i] = 1.0 / N
        # Ensure rows sum to 1
        self.A /= self.A.sum(axis=1, keepdims=True)
        
        # Update emission parameters (Gaussian)
        for i in range(N):
            weight_sum = gamma[:, i].sum()
            if weight_sum > 0:
                # μ̂_i = weighted mean
                self.means[i] = np.sum(gamma[:, i] * observations) / weight_sum
                # σ̂²_i = weighted variance
                diff = observations - self.means[i]
                self.variances[i] = np.sum(gamma[:, i] * diff**2) / weight_sum
                self.variances[i] = max(self.variances[i], 1e-6)  # floor
    
    # ----- BAUM-WELCH: Full EM -----
    def fit(self, observations):
        """
        Baum-Welch algorithm (EM) to learn HMM parameters.
        
        Iterates E-step and M-step until convergence.
        Guarantees P(O|λ) is non-decreasing (by EM theorem).
        """
        observations = np.asarray(observations, dtype=np.float64)
        self._init_params(observations)
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.n_iter):
            # Compute emission matrix
            B = self._compute_emission_matrix(observations)
            
            # Forward pass
            alpha_hat, scaling = self._forward(observations, B)
            
            # Log-likelihood: log P(O|λ) = Σ_t log(c_t)
            log_likelihood = np.sum(np.log(np.maximum(scaling, 1e-300)))
            
            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                if self.verbose:
                    print(f"  Converged at iteration {iteration}")
                break
            prev_log_likelihood = log_likelihood
            
            if self.verbose and iteration % 10 == 0:
                print(f"  Iter {iteration:3d}: log-likelihood = {log_likelihood:.4f}")
            
            # Backward pass
            beta_hat = self._backward(observations, B, scaling)
            
            # E-step
            gamma, xi = self._e_step(observations, B, alpha_hat, beta_hat, scaling)
            
            # M-step
            self._m_step(observations, gamma, xi)
        
        self.log_likelihood_ = log_likelihood
        return self
    
    # ----- VITERBI: Most likely state sequence -----
    def decode(self, observations):
        """
        Viterbi algorithm — find the most likely hidden state sequence.
        
        Q* = argmax_Q P(Q | O, λ)
        
        Uses log-space for numerical stability:
            δ_t(j) = max_i [δ_{t-1}(i) + log a_ij] + log b_j(o_t)
        """
        observations = np.asarray(observations, dtype=np.float64)
        T = len(observations)
        N = self.n_states
        B = self._compute_emission_matrix(observations)
        
        # Log-space computation
        log_A = np.log(np.maximum(self.A, 1e-300))
        log_B = np.log(np.maximum(B, 1e-300))
        log_pi = np.log(np.maximum(self.pi, 1e-300))
        
        # δ and ψ matrices
        delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)
        
        # Initialization
        delta[0] = log_pi + log_B[0]
        
        # Recursion
        for t in range(1, T):
            for j in range(N):
                candidates = delta[t-1] + log_A[:, j]
                psi[t, j] = np.argmax(candidates)
                delta[t, j] = candidates[psi[t, j]] + log_B[t, j]
        
        # Backtracking
        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states
    
    def predict_proba(self, observations):
        """Return γ_t(i) — posterior state probabilities at each time step."""
        observations = np.asarray(observations, dtype=np.float64)
        B = self._compute_emission_matrix(observations)
        alpha_hat, scaling = self._forward(observations, B)
        beta_hat = self._backward(observations, B, scaling)
        gamma, _ = self._e_step(observations, B, alpha_hat, beta_hat, scaling)
        return gamma


# =============================================================================
# PART 2: GENERATE SYNTHETIC MARKET DATA WITH KNOWN REGIMES
# =============================================================================

def generate_regime_data(n_samples=2000, seed=42):
    """
    Generate synthetic market returns with 3 known regimes.
    
    This lets us validate the HMM can recover the true hidden states.
    
    Regimes:
        0 = Bull:     μ=+0.08% daily, σ=0.8%   (high return, low vol)
        1 = Sideways:  μ=+0.01% daily, σ=1.2%   (flat, medium vol)
        2 = Bear:      μ=-0.10% daily, σ=2.0%   (negative return, high vol)
    """
    np.random.seed(seed)
    
    # True regime parameters
    true_means = [0.0008, 0.0001, -0.001]
    true_stds = [0.008, 0.012, 0.020]
    
    # True transition matrix
    true_A = np.array([
        [0.95, 0.04, 0.01],  # Bull → Bull (sticky)
        [0.10, 0.80, 0.10],  # Sideways transitions freely
        [0.05, 0.10, 0.85],  # Bear → Bear (sticky)
    ])
    
    # Generate regime sequence using Markov chain
    states = np.zeros(n_samples, dtype=int)
    states[0] = 0  # Start in bull
    for t in range(1, n_samples):
        states[t] = np.random.choice(3, p=true_A[states[t-1]])
    
    # Generate returns from regime-specific distributions
    returns = np.zeros(n_samples)
    for t in range(n_samples):
        s = states[t]
        returns[t] = np.random.normal(true_means[s], true_stds[s])
    
    # Convert to price series
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create dates
    dates = pd.date_range('2015-01-01', periods=n_samples, freq='B')
    
    return returns, prices, states, dates


# =============================================================================
# PART 3: RUN THE FULL PIPELINE
# =============================================================================

def main():
    print("=" * 70)
    print("HIDDEN MARKOV MODEL FOR MARKET REGIME DETECTION")
    print("Mathematical implementation from scratch")
    print("=" * 70)
    
    # --- Step 1: Generate synthetic data with known regimes ---
    print("\n[1] Generating synthetic market data with 3 regimes...")
    returns, prices, true_states, dates = generate_regime_data(n_samples=2000)
    print(f"    Generated {len(returns)} daily returns")
    print(f"    True regime distribution: {np.bincount(true_states)}")
    
    # --- Step 2: Fit our from-scratch HMM ---
    print("\n[2] Fitting HMM from scratch (Baum-Welch)...")
    model = GaussianHMM(n_states=3, n_iter=200, tol=1e-8, verbose=True)
    model.fit(returns)
    
    # Sort states by mean (so state 0 = bear, state 2 = bull)
    state_order = np.argsort(model.means)
    
    print(f"\n    Learned parameters:")
    regime_names = ['Bear', 'Sideways', 'Bull']
    for idx, s in enumerate(state_order):
        print(f"    State {idx} ({regime_names[idx]}): "
              f"μ = {model.means[s]*100:.4f}%/day, "
              f"σ = {np.sqrt(model.variances[s])*100:.4f}%/day")
    
    print(f"\n    Transition matrix (reordered):")
    A_reordered = model.A[state_order][:, state_order]
    for i, name in enumerate(regime_names):
        row = '  '.join(f'{x:.3f}' for x in A_reordered[i])
        print(f"      {name:>8s}: [{row}]")
    
    # --- Step 3: Decode states with Viterbi ---
    print("\n[3] Decoding states with Viterbi algorithm...")
    predicted_states_raw = model.decode(returns)
    
    # Remap to sorted order
    remap = {old: new for new, old in enumerate(state_order)}
    predicted_states = np.array([remap[s] for s in predicted_states_raw])
    
    # Remap true states for comparison (true: 0=bull, 1=side, 2=bear)
    # Our convention: 0=bear, 1=side, 2=bull
    true_remapped = np.array([{0: 2, 1: 1, 2: 0}[s] for s in true_states])
    
    accuracy = np.mean(predicted_states == true_remapped)
    print(f"    State recovery accuracy: {accuracy:.1%}")
    
    # --- Step 4: Get posterior probabilities ---
    print("\n[4] Computing posterior state probabilities γ_t(i)...")
    gamma = model.predict_proba(returns)
    # Reorder gamma columns
    gamma_reordered = gamma[:, state_order]
    
    # --- Step 5: Comparison with hmmlearn ---
    print("\n[5] Comparing with hmmlearn library...")
    try:
        from hmmlearn.hmm import GaussianHMM as HMMLearnGaussian
        
        lib_model = HMMLearnGaussian(
            n_components=3, covariance_type='diag',
            n_iter=200, tol=1e-8, random_state=42
        )
        lib_model.fit(returns.reshape(-1, 1))
        lib_states = lib_model.predict(returns.reshape(-1, 1))
        
        lib_order = np.argsort(lib_model.means_.flatten())
        print(f"    hmmlearn learned parameters:")
        for idx, s in enumerate(lib_order):
            mu = float(lib_model.means_[s].flatten()[0])
            sig = float(np.sqrt(lib_model.covars_[s].flatten()[0]))
            print(f"    State {idx} ({regime_names[idx]}): "
                  f"μ = {mu*100:.4f}%/day, σ = {sig*100:.4f}%/day")
    except ImportError:
        print("    hmmlearn not available, skipping comparison")
    
    # --- Step 6: Kelly Criterion position sizing per regime ---
    print("\n[6] Kelly Criterion position sizing by regime:")
    print("    f* = μ_i / σ²_i  (optimal fraction of capital)")
    for idx, s in enumerate(state_order):
        kelly = model.means[s] / model.variances[s]
        print(f"    {regime_names[idx]:>8s}: Kelly f* = {kelly:.2f} "
              f"({'LONG' if kelly > 0 else 'SHORT/FLAT'})")
    
    # --- Step 7: Visualizations ---
    print("\n[7] Creating visualizations...")
    
    fig, axes = plt.subplots(5, 1, figsize=(16, 20), 
                              gridspec_kw={'height_ratios': [2, 1.5, 1.5, 1.5, 1.5]})
    fig.suptitle('Hidden Markov Model — Market Regime Detection\n'
                 '(Baum-Welch algorithm from scratch)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    colors = {'Bear': '#d62728', 'Sideways': '#ff7f0e', 'Bull': '#2ca02c'}
    color_list = [colors[regime_names[s]] for s in predicted_states]
    
    # Plot 1: Price with regime coloring
    ax = axes[0]
    for i in range(len(dates) - 1):
        ax.plot(dates[i:i+2], prices[i:i+2], color=color_list[i], linewidth=0.8)
    ax.set_ylabel('Price', fontsize=11)
    ax.set_title('Synthetic Price Series Colored by Detected Regime', fontsize=12)
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=colors[r], lw=3, label=r) 
                       for r in regime_names]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Returns with regime background
    ax = axes[1]
    ax.bar(dates, returns * 100, color=color_list, width=1.5, alpha=0.7)
    ax.set_ylabel('Daily Return (%)', fontsize=11)
    ax.set_title('Daily Returns Colored by Regime', fontsize=12)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Posterior probabilities γ_t(i)
    ax = axes[2]
    for idx, name in enumerate(regime_names):
        ax.fill_between(dates, gamma_reordered[:, idx], 
                        alpha=0.5, label=f'P({name})', 
                        color=colors[name])
    ax.set_ylabel('Posterior Probability', fontsize=11)
    ax.set_title('γ_t(i) — State Posterior Probabilities (from Forward-Backward)', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: True vs Predicted states
    ax = axes[3]
    ax.plot(dates, true_remapped, 'k-', alpha=0.3, label='True', linewidth=1)
    ax.plot(dates, predicted_states, 'b-', alpha=0.6, label='Predicted (Viterbi)', linewidth=0.8)
    ax.set_ylabel('State', fontsize=11)
    ax.set_title(f'True vs Predicted States (Accuracy: {accuracy:.1%})', fontsize=12)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(regime_names)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Emission distributions per regime
    ax = axes[4]
    x_range = np.linspace(-0.06, 0.06, 500)
    for idx, s in enumerate(state_order):
        pdf = norm.pdf(x_range, model.means[s], np.sqrt(model.variances[s]))
        ax.fill_between(x_range * 100, pdf, alpha=0.4, 
                        color=colors[regime_names[idx]],
                        label=f'{regime_names[idx]}: μ={model.means[s]*100:.3f}%, '
                              f'σ={np.sqrt(model.variances[s])*100:.3f}%')
    ax.set_xlabel('Daily Return (%)', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title('Learned Emission Distributions b_i(o_t) per Regime', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=30)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('/home/claude/hmm_regime_detection.png', dpi=150, bbox_inches='tight')
    print("    Saved: hmm_regime_detection.png")
    
    # --- Bonus: Transition diagram ---
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    ax2.set_title('Learned Transition Matrix A\n'
                   'a_ij = P(q_{t+1} = S_j | q_t = S_i)', fontsize=14)
    im = ax2.imshow(A_reordered, cmap='YlOrRd', vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f'{A_reordered[i, j]:.3f}', 
                     ha='center', va='center', fontsize=14,
                     color='white' if A_reordered[i, j] > 0.5 else 'black')
    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))
    ax2.set_xticklabels(regime_names, fontsize=12)
    ax2.set_yticklabels(regime_names, fontsize=12)
    ax2.set_xlabel('To State (j)', fontsize=12)
    ax2.set_ylabel('From State (i)', fontsize=12)
    plt.colorbar(im, ax=ax2, label='Transition Probability')
    plt.tight_layout()
    plt.savefig('/home/claude/hmm_transition_matrix.png', dpi=150, bbox_inches='tight')
    print("    Saved: hmm_transition_matrix.png")
    
    # --- Summary statistics ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Model: {model.n_states}-state Gaussian HMM")
    print(f"  Data: {len(returns)} observations")
    print(f"  Log-likelihood: {model.log_likelihood_:.2f}")
    print(f"  State recovery accuracy: {accuracy:.1%}")
    print(f"\n  Regime statistics:")
    for idx, name in enumerate(regime_names):
        mask = predicted_states == idx
        n_days = mask.sum()
        pct = n_days / len(returns) * 100
        mean_ret = returns[mask].mean() * 100 * 252  # Annualized
        vol = returns[mask].std() * 100 * np.sqrt(252)  # Annualized
        sharpe = (mean_ret / vol) if vol > 0 else 0
        print(f"    {name:>8s}: {n_days:4d} days ({pct:5.1f}%), "
              f"Ann. Return: {mean_ret:+6.1f}%, "
              f"Ann. Vol: {vol:5.1f}%, "
              f"Sharpe: {sharpe:+.2f}")
    
    print("\n  Key insight (à la Renaissance):")
    print("  The model doesn't need to be right every day.")
    print("  It just needs to be right >50.75% of the time,")
    print("  millions of trades, with leverage and cost control.")
    print("=" * 70)
    
    plt.close('all')


if __name__ == '__main__':
    main()
