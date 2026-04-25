"""Auto-checker helpers for Chapter 6, Autoencoders."""

import numpy as np

_OK   = "✅"
_FAIL = "❌"
_NONE = "⬜"
_TOL  = 1e-3


# ---------------------------------------------------------------------------
# Linear autoencoder: encode then decode using PCA components
# ---------------------------------------------------------------------------

def check_linear_autoencode(fn, X_c, V):
    """Check that fn returns (Z, X_hat) where Z = X_c @ V.T and X_hat = Z @ V."""
    got = fn(X_c, V)
    n, d = X_c.shape
    k = V.shape[0]
    expected_Z = X_c @ V.T
    expected_X_hat = expected_Z @ V

    if got is None:
        print(f"  {_NONE} linear_autoencode: not implemented yet (expected Z {expected_Z.shape}, X_hat {expected_X_hat.shape})")
        return

    if not (isinstance(got, tuple) and len(got) == 2):
        print(f"  {_FAIL} linear_autoencode: should return a tuple (Z, X_hat), got {type(got).__name__}")
        return

    Z, X_hat = got
    Z = np.asarray(Z, dtype=float)
    X_hat = np.asarray(X_hat, dtype=float)

    if Z.shape != (n, k):
        print(f"  {_FAIL} linear_autoencode: Z has shape {Z.shape} (expected {(n, k)})")
        print(f"       Hint: encode with Z = X_c @ V.T  (V has shape (k, d), so V.T has shape (d, k)).")
        return

    if X_hat.shape != (n, d):
        print(f"  {_FAIL} linear_autoencode: X_hat has shape {X_hat.shape} (expected {(n, d)})")
        print(f"       Hint: decode with X_hat = Z @ V  (Z has shape (n, k), V has shape (k, d)).")
        return

    z_match = np.allclose(Z, expected_Z, atol=_TOL) or np.allclose(Z, -expected_Z, atol=_TOL)
    if not z_match:
        if np.allclose(Z, X_c @ V, atol=_TOL):
            print(f"  {_FAIL} linear_autoencode: Z used V instead of V.T")
            print(f"       Hint: V has shape (k, d); to project X_c (n, d) into the k-dim latent, multiply by V.T.")
            return
        print(f"  {_FAIL} linear_autoencode: Z values do not match X_c @ V.T")
        return

    if not np.allclose(X_hat, expected_X_hat, atol=_TOL):
        if np.allclose(X_hat, Z @ V.T, atol=_TOL):
            print(f"  {_FAIL} linear_autoencode: X_hat used V.T instead of V")
            print(f"       Hint: to map back from the latent (n, k) to the data space (n, d), multiply by V (not V.T).")
            return
        print(f"  {_FAIL} linear_autoencode: X_hat values do not match Z @ V")
        return

    err = float(((X_c - X_hat) ** 2).mean())
    print(f"  {_OK} linear_autoencode: Z {Z.shape}, X_hat {X_hat.shape}, MSE = {err:.4f}")


# ---------------------------------------------------------------------------
# KL divergence  KL( N(mu, sigma^2) || N(0, I) )  in closed form
# ---------------------------------------------------------------------------

def check_kl_divergence(fn, mu, log_var):
    """Check that fn returns the per-sample KL of N(mu, exp(log_var)) against N(0, I)."""
    got = fn(mu, log_var)
    expected = -0.5 * np.sum(1 + log_var - mu ** 2 - np.exp(log_var), axis=1)

    if got is None:
        print(f"  {_NONE} kl_divergence: not implemented yet (expected vector of length {mu.shape[0]})")
        return

    got = np.asarray(got, dtype=float)

    if got.ndim == 0 or got.shape == ():
        print(f"  {_FAIL} kl_divergence: returned a scalar (expected one KL value per row of mu)")
        print(f"       Hint: sum across the latent axis only, not across samples too.")
        return

    if got.shape != expected.shape:
        print(f"  {_FAIL} kl_divergence: shape {got.shape} (expected {expected.shape})")
        print(f"       Hint: take np.sum(..., axis=1) over the latent dimension; keep the sample axis.")
        return

    if np.allclose(got, expected, atol=_TOL):
        print(f"  {_OK} kl_divergence: mean KL = {got.mean():.4f}")
        return

    # Common sign error: returned the negation
    if np.allclose(got, -expected, atol=_TOL):
        print(f"  {_FAIL} kl_divergence: sign is flipped (got the negative ELBO term, not the KL)")
        print(f"       Hint: KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var)). Watch the leading minus sign.")
        return

    # Forgot to exponentiate log_var (used log_var as if it were variance directly)
    expected_no_exp = -0.5 * np.sum(1 + log_var - mu ** 2 - log_var, axis=1)
    if np.allclose(got, expected_no_exp, atol=_TOL):
        print(f"  {_FAIL} kl_divergence: did not exponentiate log_var")
        print(f"       Hint: variance is exp(log_var). The formula has -exp(log_var), not -log_var.")
        return

    # Used variance directly as if it were sigma (forgot to square)
    sigma = np.exp(0.5 * log_var)
    expected_sigma = -0.5 * np.sum(1 + log_var - mu ** 2 - sigma, axis=1)
    if np.allclose(got, expected_sigma, atol=_TOL):
        print(f"  {_FAIL} kl_divergence: used sigma instead of sigma^2 (variance)")
        print(f"       Hint: the formula needs the variance term, which is exp(log_var), not exp(0.5 * log_var).")
        return

    print(f"  {_FAIL} kl_divergence: values do not match")
    print(f"       Hint: KL = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var), axis=1).")


# ---------------------------------------------------------------------------
# Reparameterization trick:  z = mu + exp(0.5 * log_var) * epsilon
# ---------------------------------------------------------------------------

def check_reparameterize(fn, mu, log_var, rng_seed=0):
    """Check that fn returns z = mu + sigma * eps with sigma = exp(0.5 * log_var)."""
    rng_a = np.random.default_rng(rng_seed)
    rng_b = np.random.default_rng(rng_seed)
    got = fn(mu, log_var, rng_a)

    if got is None:
        print(f"  {_NONE} reparameterize: not implemented yet (expected array of shape {mu.shape})")
        return

    got = np.asarray(got, dtype=float)
    if got.shape != mu.shape:
        print(f"  {_FAIL} reparameterize: shape {got.shape} (expected {mu.shape})")
        print(f"       Hint: z has the same shape as mu; sample one epsilon per element.")
        return

    sigma = np.exp(0.5 * log_var)
    expected_eps = rng_b.standard_normal(mu.shape)
    expected = mu + sigma * expected_eps

    if np.allclose(got, expected, atol=_TOL):
        print(f"  {_OK} reparameterize: z {got.shape}, empirical std vs target = {got.std(axis=0).mean():.3f} vs {sigma.mean():.3f}")
        return

    # Forgot the 0.5 factor: used exp(log_var) as sigma instead of sqrt(exp(log_var))
    sigma_naive = np.exp(log_var)
    if np.allclose(got, mu + sigma_naive * expected_eps, atol=_TOL):
        print(f"  {_FAIL} reparameterize: used exp(log_var) as sigma; that is the variance, not the standard deviation")
        print(f"       Hint: sigma = exp(0.5 * log_var)  (or equivalently np.sqrt(np.exp(log_var))).")
        return

    # Returned mu only (forgot the noise)
    if np.allclose(got, mu, atol=_TOL):
        print(f"  {_FAIL} reparameterize: returned mu unchanged (forgot to add the noise term)")
        print(f"       Hint: z = mu + sigma * epsilon, where epsilon ~ N(0, I).")
        return

    # Sampled epsilon but did not scale by sigma
    if np.allclose(got, mu + expected_eps, atol=_TOL):
        print(f"  {_FAIL} reparameterize: forgot to scale epsilon by sigma")
        print(f"       Hint: multiply epsilon by sigma = exp(0.5 * log_var) before adding to mu.")
        return

    # Sample is plausible but not bit-identical — likely used a different RNG.
    # Check empirical std as a softer pass.
    empirical_std = (got - mu).std(axis=0)
    if np.allclose(empirical_std.mean(), sigma.mean(), atol=0.1):
        print(f"  {_OK} reparameterize: shape and empirical std match (different RNG draws are fine)")
        return

    print(f"  {_FAIL} reparameterize: values do not match the expected formula")
    print(f"       Hint: sigma = np.exp(0.5 * log_var); eps = rng.standard_normal(mu.shape); return mu + sigma * eps.")
