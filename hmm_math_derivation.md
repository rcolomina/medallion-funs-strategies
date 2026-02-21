# Hidden Markov Models for Market Regime Detection
## Full Mathematical Derivation & the Baum-Welch Algorithm

*As used by Renaissance Technologies' Medallion Fund*

---

## 1. Markov Chains — The Foundation

### 1.1 Definition

A **Markov chain** is a stochastic process {X₁, X₂, ..., Xₜ} satisfying the **Markov property**:

```
P(Xₜ₊₁ = sⱼ | Xₜ = sᵢ, Xₜ₋₁ = sₖ, ...) = P(Xₜ₊₁ = sⱼ | Xₜ = sᵢ)
```

The future state depends **only** on the current state — not the full history. This is the "memoryless" property.

### 1.2 Transition Matrix

For a system with N states {s₁, s₂, ..., sₙ}, define the **transition probability matrix** A:

```
aᵢⱼ = P(Xₜ₊₁ = sⱼ | Xₜ = sᵢ)
```

Constraints:
- aᵢⱼ ≥ 0 for all i, j
- Σⱼ aᵢⱼ = 1 for all i (rows sum to 1)

**Market example** with 3 regimes (Bull, Sideways, Bear):

```
         Bull   Side   Bear
A = Bull [ 0.7   0.2   0.1 ]
    Side [ 0.3   0.5   0.2 ]
    Bear [ 0.2   0.3   0.5 ]
```

Reading: If we're in a Bull state, there's 70% chance of staying Bull, 20% of transitioning to Sideways, 10% to Bear.

---

## 2. Hidden Markov Models (HMMs)

### 2.1 The Key Insight

In financial markets, we **cannot directly observe** the regime (state). We only see **emissions** — returns, prices, volumes. The states are *hidden*.

An HMM is defined by the tuple **λ = (A, B, π)** where:

**A** = Transition matrix (N × N) — probability of moving between hidden states

```
aᵢⱼ = P(qₜ₊₁ = Sⱼ | qₜ = Sᵢ)
```

**B** = Emission probabilities — probability of observing output oₜ given hidden state Sᵢ

For continuous observations (like returns), we typically use Gaussian emissions:

```
bᵢ(oₜ) = N(oₜ | μᵢ, σᵢ²) = (1 / √(2πσᵢ²)) · exp(-(oₜ - μᵢ)² / (2σᵢ²))
```

Each hidden state i has its own mean μᵢ and variance σᵢ².

**π** = Initial state distribution (1 × N)

```
πᵢ = P(q₁ = Sᵢ)
```

### 2.2 The Three Fundamental HMM Problems

| Problem | Question | Algorithm |
|---------|----------|-----------|
| **Evaluation** | Given λ and observations O, what is P(O\|λ)? | Forward algorithm |
| **Decoding** | Given λ and O, what is the most likely state sequence? | Viterbi algorithm |
| **Learning** | Given O, what λ maximizes P(O\|λ)? | Baum-Welch (EM) |

Renaissance primarily cared about **Problem 3** (learning the model from data) and **Problem 2** (decoding the current regime).

---

## 3. The Forward Algorithm

### 3.1 Definition

Define the **forward variable**:

```
αₜ(i) = P(o₁, o₂, ..., oₜ, qₜ = Sᵢ | λ)
```

This is the probability of observing the partial sequence o₁...oₜ AND being in state Sᵢ at time t.

### 3.2 Derivation

**Initialization** (t = 1):

```
α₁(i) = πᵢ · bᵢ(o₁)    for i = 1, 2, ..., N
```

The probability of starting in state i AND emitting the first observation.

**Induction** (t = 2, ..., T):

```
αₜ(j) = [Σᵢ₌₁ᴺ αₜ₋₁(i) · aᵢⱼ] · bⱼ(oₜ)
```

**Derivation of the induction step:**

```
αₜ(j) = P(o₁, ..., oₜ, qₜ = Sⱼ | λ)

       = Σᵢ P(o₁, ..., oₜ, qₜ₋₁ = Sᵢ, qₜ = Sⱼ | λ)        [marginalize over previous state]

       = Σᵢ P(oₜ | qₜ = Sⱼ, o₁..oₜ₋₁, qₜ₋₁ = Sᵢ, λ)        [chain rule]
           · P(qₜ = Sⱼ | o₁..oₜ₋₁, qₜ₋₁ = Sᵢ, λ)
           · P(o₁..oₜ₋₁, qₜ₋₁ = Sᵢ | λ)

       = Σᵢ bⱼ(oₜ) · aᵢⱼ · αₜ₋₁(i)                           [by HMM assumptions]

       = bⱼ(oₜ) · Σᵢ αₜ₋₁(i) · aᵢⱼ
```

**Termination:**

```
P(O | λ) = Σᵢ₌₁ᴺ αₜ(i)
```

**Complexity:** O(N²T) instead of the naive O(Nᵀ) — this is what makes HMMs tractable.

---

## 4. The Backward Algorithm

### 4.1 Definition

Define the **backward variable**:

```
βₜ(i) = P(oₜ₊₁, oₜ₊₂, ..., oₜ | qₜ = Sᵢ, λ)
```

The probability of observing the *future* partial sequence given current state Sᵢ.

### 4.2 Derivation

**Initialization** (t = T):

```
βₜ(i) = 1    for all i
```

(No future observations to account for.)

**Induction** (t = T-1, T-2, ..., 1):

```
βₜ(i) = Σⱼ₌₁ᴺ aᵢⱼ · bⱼ(oₜ₊₁) · βₜ₊₁(j)
```

**Derivation:**

```
βₜ(i) = P(oₜ₊₁, ..., oₜ | qₜ = Sᵢ, λ)

       = Σⱼ P(oₜ₊₁, ..., oₜ, qₜ₊₁ = Sⱼ | qₜ = Sᵢ, λ)

       = Σⱼ P(oₜ₊₂, ..., oₜ | qₜ₊₁ = Sⱼ, λ)
           · P(oₜ₊₁ | qₜ₊₁ = Sⱼ, λ)
           · P(qₜ₊₁ = Sⱼ | qₜ = Sᵢ, λ)

       = Σⱼ βₜ₊₁(j) · bⱼ(oₜ₊₁) · aᵢⱼ
```

---

## 5. The Baum-Welch Algorithm (Expectation-Maximization)

This is the core algorithm Renaissance Technologies built upon. It learns the optimal model parameters λ = (A, B, π) from observed data.

### 5.1 Key Quantities

**Gamma** — probability of being in state i at time t:

```
γₜ(i) = P(qₜ = Sᵢ | O, λ) = αₜ(i) · βₜ(i) / P(O | λ)
```

where P(O | λ) = Σᵢ αₜ(i) · βₜ(i)  [normalization constant]

**Xi** — probability of transitioning from state i to state j at time t:

```
ξₜ(i, j) = P(qₜ = Sᵢ, qₜ₊₁ = Sⱼ | O, λ)

          = αₜ(i) · aᵢⱼ · bⱼ(oₜ₊₁) · βₜ₊₁(j) / P(O | λ)
```

**Derivation of ξₜ(i, j):**

```
ξₜ(i, j) = P(qₜ = Sᵢ, qₜ₊₁ = Sⱼ, O | λ) / P(O | λ)

Numerator = P(o₁..oₜ, qₜ = Sᵢ | λ)                    [= αₜ(i)]
          · P(qₜ₊₁ = Sⱼ | qₜ = Sᵢ, λ)                 [= aᵢⱼ]
          · P(oₜ₊₁ | qₜ₊₁ = Sⱼ, λ)                     [= bⱼ(oₜ₊₁)]
          · P(oₜ₊₂..oₜ | qₜ₊₁ = Sⱼ, λ)                 [= βₜ₊₁(j)]

          = αₜ(i) · aᵢⱼ · bⱼ(oₜ₊₁) · βₜ₊₁(j)
```

Note: γₜ(i) = Σⱼ ξₜ(i, j)

### 5.2 The E-Step and M-Step

The Baum-Welch algorithm iterates between:

**E-Step:** Compute γₜ(i) and ξₜ(i, j) using current parameters and the forward-backward algorithm.

**M-Step:** Re-estimate parameters:

**Initial state distribution:**

```
π̂ᵢ = γ₁(i)
```

(Expected frequency of being in state i at t=1)

**Transition probabilities:**

```
âᵢⱼ = Σₜ₌₁ᵀ⁻¹ ξₜ(i, j) / Σₜ₌₁ᵀ⁻¹ γₜ(i)
```

(Expected number of transitions from i→j divided by expected number of times in state i)

**Emission parameters (Gaussian):**

Mean:
```
μ̂ᵢ = Σₜ₌₁ᵀ γₜ(i) · oₜ / Σₜ₌₁ᵀ γₜ(i)
```

Variance:
```
σ̂ᵢ² = Σₜ₌₁ᵀ γₜ(i) · (oₜ - μ̂ᵢ)² / Σₜ₌₁ᵀ γₜ(i)
```

### 5.3 Convergence

The algorithm guarantees that P(O | λ) is **non-decreasing** at each iteration (this follows from the EM convergence theorem). It converges to a local maximum of the likelihood.

**Proof sketch:** By Jensen's inequality applied to the auxiliary function Q(λ, λ_old):

```
Q(λ, λ_old) = Σ_all_states P(Q | O, λ_old) · log P(O, Q | λ)
```

One can show Q(λ_new, λ_old) ≥ Q(λ_old, λ_old) implies P(O | λ_new) ≥ P(O | λ_old).

### 5.4 Numerical Stability — Log-Space Computation

In practice, α and β values become extremely small for long sequences (products of many probabilities). Renaissance would have used **scaling** or **log-space** computation:

```
log αₜ(j) = log bⱼ(oₜ) + log_sum_exp[log αₜ₋₁(i) + log aᵢⱼ for i = 1..N]
```

where log_sum_exp(x₁, ..., xₙ) = max(xᵢ) + log(Σ exp(xᵢ - max(xᵢ)))

---

## 6. The Viterbi Algorithm (Most Likely State Sequence)

### 6.1 Purpose

Find the single most likely sequence of hidden states:

```
Q* = argmax_Q P(Q | O, λ)
```

This tells you: "Given the returns we observed, what is the most likely sequence of market regimes?"

### 6.2 Algorithm (Dynamic Programming)

Define:

```
δₜ(j) = max_{q₁..qₜ₋₁} P(q₁, ..., qₜ₋₁, qₜ = Sⱼ, o₁, ..., oₜ | λ)
```

**Initialization:**
```
δ₁(i) = πᵢ · bᵢ(o₁)
ψ₁(i) = 0
```

**Recursion:**
```
δₜ(j) = max_i [δₜ₋₁(i) · aᵢⱼ] · bⱼ(oₜ)
ψₜ(j) = argmax_i [δₜ₋₁(i) · aᵢⱼ]
```

**Termination:**
```
P* = max_i δₜ(i)
qₜ* = argmax_i δₜ(i)
```

**Backtracking:**
```
qₜ* = ψₜ₊₁(qₜ₊₁*)    for t = T-1, ..., 1
```

---

## 7. Connection to Kelly Criterion (Bet Sizing)

Once you know the current regime (via Viterbi or γ), you can size positions optimally.

The **Kelly fraction** for a bet with edge p (win probability) and odds b:

```
f* = (bp - q) / b    where q = 1 - p
```

For continuous returns with Gaussian distribution in regime i:

```
f* = μᵢ / σᵢ²
```

This is exactly what Berlekamp implemented at Renaissance — using the regime-conditional return distributions to determine optimal position sizes.

---

## 8. Multivariate Extension

Real markets require **multivariate** HMMs where observations are vectors (multiple assets, features):

```
oₜ = [return_asset1, return_asset2, volume, volatility, ...]ᵀ ∈ ℝᵈ
```

Emission distribution becomes multivariate Gaussian:

```
bᵢ(oₜ) = N(oₜ | μᵢ, Σᵢ)
        = (2π)^(-d/2) |Σᵢ|^(-1/2) exp(-½ (oₜ - μᵢ)ᵀ Σᵢ⁻¹ (oₜ - μᵢ))
```

where Σᵢ is the d × d covariance matrix for state i.

The M-step updates become:

```
μ̂ᵢ = Σₜ γₜ(i) · oₜ / Σₜ γₜ(i)

Σ̂ᵢ = Σₜ γₜ(i) · (oₜ - μ̂ᵢ)(oₜ - μ̂ᵢ)ᵀ / Σₜ γₜ(i)
```

This is what Henry Laufer pushed for — a **single cross-asset model** where correlations between assets help identify the hidden regime.

---

## Summary of Computational Complexity

| Algorithm | Time | Space |
|-----------|------|-------|
| Forward | O(N²T) | O(NT) |
| Backward | O(N²T) | O(NT) |
| Baum-Welch (per iteration) | O(N²T) | O(NT) |
| Viterbi | O(N²T) | O(NT) |

Where N = number of hidden states, T = length of observation sequence.
