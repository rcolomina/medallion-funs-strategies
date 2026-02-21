"""
CLI entry point — runs the full HMM regime detection pipeline.
"""

import numpy as np
from scipy.stats import norm

from renaissance_trading.hmm import GaussianHMM
from renaissance_trading.data import generate_regime_data


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

    print("\n    Learned parameters:")
    regime_names = ["Bear", "Sideways", "Bull"]
    for idx, s in enumerate(state_order):
        print(
            f"    State {idx} ({regime_names[idx]}): "
            f"mu = {model.means[s] * 100:.4f}%/day, "
            f"sigma = {np.sqrt(model.variances[s]) * 100:.4f}%/day"
        )

    print("\n    Transition matrix (reordered):")
    A_reordered = model.A[state_order][:, state_order]
    for i, name in enumerate(regime_names):
        row = "  ".join(f"{x:.3f}" for x in A_reordered[i])
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
    print("\n[4] Computing posterior state probabilities gamma_t(i)...")
    gamma = model.predict_proba(returns)
    gamma_reordered = gamma[:, state_order]

    # --- Step 5: Kelly Criterion position sizing per regime ---
    print("\n[5] Kelly Criterion position sizing by regime:")
    print("    f* = mu_i / sigma^2_i  (optimal fraction of capital)")
    for idx, s in enumerate(state_order):
        kelly = model.means[s] / model.variances[s]
        print(
            f"    {regime_names[idx]:>8s}: Kelly f* = {kelly:.2f} "
            f"({'LONG' if kelly > 0 else 'SHORT/FLAT'})"
        )

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
        print(
            f"    {name:>8s}: {n_days:4d} days ({pct:5.1f}%), "
            f"Ann. Return: {mean_ret:+6.1f}%, "
            f"Ann. Vol: {vol:5.1f}%, "
            f"Sharpe: {sharpe:+.2f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
