"""Offline generator for lecture-06 precomputed embeddings.

Trains tiny AE and VAE models on the 1000-sample MNIST subset from lecture-05
and writes four CSVs that the student notebook visualizes.

Usage (from repo root):
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements-precompute.txt
    python content/unsupervised-deep-learning/lecture-06/generate_precomputed.py

Outputs (alongside this script):
    embeddings_k2.csv      - 1000 x {label, ae_z, vae_mu, vae_logvar} for k=2
    bottleneck_sweep.csv   - per-sample 2D-projected embeddings + recon error at k in {2, 8, 32}
    active_units_k16.csv   - per-dim mean KL for the k=16 VAE
    latent_walk.csv        - 11 x 784 decoded waypoints, digit 0 -> digit 1
    kl_annealing.csv       - per-epoch (beta, recon, kl) for constant vs annealed VAE
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
LECTURE_05 = HERE.parent / "lecture-05"
SEED = 0
DEVICE = torch.device("cpu")


def load_mnist():
    df = pd.read_csv(LECTURE_05 / "mnist_digits.csv")
    y = df["label"].to_numpy()
    X = df.drop("label", axis=1).to_numpy(dtype=np.float32) / 255.0
    return X, y


class AE(nn.Module):
    def __init__(self, d, k, h=256):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, k))
        self.dec = nn.Sequential(nn.Linear(k, h), nn.ReLU(), nn.Linear(h, d), nn.Sigmoid())

    def forward(self, x):
        z = self.enc(x)
        return z, self.dec(z)


class VAE(nn.Module):
    def __init__(self, d, k, h=256):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU())
        self.mu = nn.Linear(h, k)
        self.logvar = nn.Linear(h, k)
        self.dec = nn.Sequential(nn.Linear(k, h), nn.ReLU(), nn.Linear(h, d), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z, self.dec(z)


def train_ae(X, k, epochs=80, batch=128, lr=1e-3):
    torch.manual_seed(SEED)
    model = AE(X.shape[1], k).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(X).to(DEVICE)
    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = Xt[idx]
            _, xh = model(xb)
            loss = F.mse_loss(xh, xb, reduction="mean")
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def train_vae(X, k, epochs=120, batch=128, lr=1e-3, beta=1.0):
    torch.manual_seed(SEED)
    model = VAE(X.shape[1], k).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(X).to(DEVICE)
    d = X.shape[1]
    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = Xt[idx]
            mu, logvar, z, xh = model(xb)
            recon = F.mse_loss(xh, xb, reduction="sum") / xb.shape[0]
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
            loss = recon + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def train_vae_logged(X, k, epochs=120, batch=128, lr=1e-3, beta_schedule=None, beta_target=1.0):
    """Train a VAE while logging per-epoch (beta, recon, kl).

    beta_schedule: callable epoch -> beta. If None, uses constant beta = beta_target.
    Returns (model, log) where log is a list of dicts.
    """
    if beta_schedule is None:
        beta_schedule = lambda _: beta_target
    torch.manual_seed(SEED)
    model = VAE(X.shape[1], k).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(X).to(DEVICE)
    n = X.shape[0]
    log = []
    for e in range(epochs):
        beta = float(beta_schedule(e))
        perm = torch.randperm(n)
        recon_sum = 0.0
        kl_sum = 0.0
        n_batches = 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = Xt[idx]
            mu, logvar, z, xh = model(xb)
            recon = F.mse_loss(xh, xb, reduction="sum") / xb.shape[0]
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
            loss = recon + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
            recon_sum += float(recon.detach())
            kl_sum += float(kl.detach())
            n_batches += 1
        log.append({
            "epoch": e,
            "beta": beta,
            "recon": recon_sum / n_batches,
            "kl": kl_sum / n_batches,
        })
    return model, log


def linear_warmup(n_warmup, target=1.0):
    """Beta ramps linearly from 0 to target over n_warmup epochs, then stays at target."""
    def schedule(e):
        if e >= n_warmup:
            return target
        return target * e / max(1, n_warmup)
    return schedule


@torch.no_grad()
def ae_embed_and_recon(model, X):
    Xt = torch.from_numpy(X)
    z, xh = model(Xt)
    err = ((Xt - xh) ** 2).mean(dim=1).numpy()
    return z.numpy(), err


@torch.no_grad()
def vae_embed(model, X):
    Xt = torch.from_numpy(X)
    mu, logvar = model.encode(Xt)
    return mu.numpy(), logvar.numpy()


@torch.no_grad()
def vae_recon_err(model, X):
    Xt = torch.from_numpy(X)
    mu, logvar, z, xh = model(Xt)
    return ((Xt - xh) ** 2).mean(dim=1).numpy()


def project_2d(Z):
    """PCA-project to 2D for visualization when k > 2."""
    Zc = Z - Z.mean(axis=0)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    return Zc @ Vt[:2].T


def main():
    print("Loading MNIST subset...")
    X, y = load_mnist()
    print(f"  X={X.shape}, y={y.shape}")

    # ---- 1. embeddings_k2.csv (AE vs VAE side by side) ----
    print("Training AE k=2...")
    ae2 = train_ae(X, k=2)
    ae2_z, _ = ae_embed_and_recon(ae2, X)
    print("Training VAE k=2...")
    vae2 = train_vae(X, k=2)
    vae2_mu, vae2_lv = vae_embed(vae2, X)

    df_k2 = pd.DataFrame({
        "label": y,
        "ae_z1": ae2_z[:, 0], "ae_z2": ae2_z[:, 1],
        "vae_mu1": vae2_mu[:, 0], "vae_mu2": vae2_mu[:, 1],
        "vae_logvar1": vae2_lv[:, 0], "vae_logvar2": vae2_lv[:, 1],
    })
    df_k2.to_csv(HERE / "embeddings_k2.csv", index=False)
    print(f"  wrote embeddings_k2.csv ({df_k2.shape})")

    # ---- 2. bottleneck_sweep.csv ----
    rows = {"label": y}
    for k in [2, 8, 32]:
        print(f"Training AE k={k} for sweep...")
        m_ae = ae2 if k == 2 else train_ae(X, k=k)
        z_ae, err_ae = ae_embed_and_recon(m_ae, X)
        z_ae_2d = z_ae if k == 2 else project_2d(z_ae)
        rows[f"ae_k{k}_z1"] = z_ae_2d[:, 0]
        rows[f"ae_k{k}_z2"] = z_ae_2d[:, 1]
        rows[f"ae_k{k}_err"] = err_ae

        print(f"Training VAE k={k} for sweep...")
        m_vae = vae2 if k == 2 else train_vae(X, k=k)
        mu_v, _ = vae_embed(m_vae, X)
        mu_2d = mu_v if k == 2 else project_2d(mu_v)
        err_v = vae_recon_err(m_vae, X)
        rows[f"vae_k{k}_z1"] = mu_2d[:, 0]
        rows[f"vae_k{k}_z2"] = mu_2d[:, 1]
        rows[f"vae_k{k}_err"] = err_v

    df_sweep = pd.DataFrame(rows)
    df_sweep.to_csv(HERE / "bottleneck_sweep.csv", index=False)
    print(f"  wrote bottleneck_sweep.csv ({df_sweep.shape})")

    # ---- 3. active_units_k16.csv ----
    print("Training VAE k=16 for active units...")
    vae16 = train_vae(X, k=16)
    mu16, lv16 = vae_embed(vae16, X)
    # per-dim KL averaged across the dataset
    per_dim_kl = -0.5 * (1 + lv16 - mu16 ** 2 - np.exp(lv16))
    df_au = pd.DataFrame({"dim": np.arange(16), "mean_kl": per_dim_kl.mean(axis=0)})
    df_au.to_csv(HERE / "active_units_k16.csv", index=False)
    print(f"  wrote active_units_k16.csv ({df_au.shape})")

    # ---- 4. latent_walk.csv ----
    # Use the k=16 VAE: pick one digit-0 sample and one digit-1 sample,
    # encode each to its mu, lerp 11 waypoints, decode each.
    print("Generating latent walk 0 -> 1...")
    rng = np.random.default_rng(SEED)
    i0 = rng.choice(np.where(y == 0)[0])
    i1 = rng.choice(np.where(y == 1)[0])
    with torch.no_grad():
        mu_pair = vae16.encode(torch.from_numpy(X[[i0, i1]]))[0].numpy()
    ts = np.linspace(0.0, 1.0, 11)
    walk_z = np.stack([(1 - t) * mu_pair[0] + t * mu_pair[1] for t in ts])
    with torch.no_grad():
        decoded = vae16.dec(torch.from_numpy(walk_z.astype(np.float32))).numpy()
    df_walk = pd.DataFrame(decoded, columns=[f"px{i}" for i in range(decoded.shape[1])])
    df_walk.insert(0, "t", ts)
    df_walk.to_csv(HERE / "latent_walk.csv", index=False)
    print(f"  wrote latent_walk.csv ({df_walk.shape})")

    # ---- 5. kl_annealing.csv (+ kl_annealing_active.csv) ----
    # Two runs side by side at higher beta and larger k so the collapse
    # vs rescue contrast is unambiguous: constant beta = 4 from epoch zero
    # vs linear warmup 0 -> 4 over the first 40 of 150 epochs.
    ANNEAL_K = 32
    ANNEAL_EPOCHS = 150
    ANNEAL_WARMUP = 40
    ANNEAL_BETA = 4.0

    print(f"Training VAE k={ANNEAL_K} with constant beta = {ANNEAL_BETA} (logged)...")
    model_const, log_const = train_vae_logged(
        X, k=ANNEAL_K, epochs=ANNEAL_EPOCHS, beta_target=ANNEAL_BETA
    )
    print(f"Training VAE k={ANNEAL_K} with linear-warmup beta 0 -> {ANNEAL_BETA} over {ANNEAL_WARMUP} (logged)...")
    model_anneal, log_anneal = train_vae_logged(
        X, k=ANNEAL_K, epochs=ANNEAL_EPOCHS,
        beta_schedule=linear_warmup(ANNEAL_WARMUP, target=ANNEAL_BETA),
    )
    df_anneal = pd.DataFrame({
        "epoch": [r["epoch"] for r in log_const],
        "beta_const": [r["beta"] for r in log_const],
        "beta_anneal": [r["beta"] for r in log_anneal],
        "recon_const": [r["recon"] for r in log_const],
        "recon_anneal": [r["recon"] for r in log_anneal],
        "kl_const": [r["kl"] for r in log_const],
        "kl_anneal": [r["kl"] for r in log_anneal],
    })
    df_anneal.to_csv(HERE / "kl_annealing.csv", index=False)
    print(f"  wrote kl_annealing.csv ({df_anneal.shape})")

    # Final per-dim mean KL for each of the two runs, so the notebook can
    # show "constant killed N dims, annealed left M alive" side by side.
    mu_c, lv_c = vae_embed(model_const, X)
    mu_a, lv_a = vae_embed(model_anneal, X)
    per_dim_const  = (-0.5 * (1 + lv_c - mu_c ** 2 - np.exp(lv_c))).mean(axis=0)
    per_dim_anneal = (-0.5 * (1 + lv_a - mu_a ** 2 - np.exp(lv_a))).mean(axis=0)
    df_active_anneal = pd.DataFrame({
        "dim": np.arange(ANNEAL_K),
        "mean_kl_const":  per_dim_const,
        "mean_kl_anneal": per_dim_anneal,
    })
    df_active_anneal.to_csv(HERE / "kl_annealing_active.csv", index=False)
    print(f"  wrote kl_annealing_active.csv ({df_active_anneal.shape})")

    # ---- 6. bottleneck_curves.csv ----
    # Per-epoch (recon, kl) for several VAE bottleneck sizes, one column
    # pair per k. Replaces the old AE-vs-VAE scatter + recon-vs-k summary
    # with training curves that show how capacity reshapes both losses.
    CURVE_KS = [2, 4, 8, 16, 32]
    CURVE_EPOCHS = 80
    curve_logs = {}
    for kk in CURVE_KS:
        print(f"Training VAE k={kk} for bottleneck curves...")
        _, log = train_vae_logged(X, k=kk, epochs=CURVE_EPOCHS)
        curve_logs[kk] = log

    df_curves = pd.DataFrame({"epoch": np.arange(CURVE_EPOCHS)})
    for kk in CURVE_KS:
        df_curves[f"k{kk}_recon"] = [r["recon"] for r in curve_logs[kk]]
        df_curves[f"k{kk}_kl"]    = [r["kl"]    for r in curve_logs[kk]]
    df_curves.to_csv(HERE / "bottleneck_curves.csv", index=False)
    print(f"  wrote bottleneck_curves.csv ({df_curves.shape})")

    print("Done.")


if __name__ == "__main__":
    main()
