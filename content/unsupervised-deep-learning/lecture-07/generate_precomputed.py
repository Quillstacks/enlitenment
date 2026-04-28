"""Offline generator for lecture-07 precomputed assets.

Trains the small models that the chapter-7 'evens-only / odds-as-OOD' setup
needs and writes the CSVs that the student notebook visualizes.

The recipe:
- Train set : ~5000 MNIST samples drawn from digits {0, 2, 4, 6, 8}.
- Test set  : ~1000 evens (held-out, in-distribution) + ~1000 odds (out-of-
  distribution by construction, the model never sees these labels).
- Models    : a small MLP classifier with dropout p=0.3 (used for both the
              calibration trap and the MC-dropout passes), a VAE on evens
              (for reconstruction-based anomaly detection), and one classifier
              with a confidence head (for the confidence-head section).
- Outputs   : CSVs alongside this script, one per visualization.

Usage (from repo root):
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements-precompute.txt
    python content/unsupervised-deep-learning/lecture-07/generate_precomputed.py

Outputs (alongside this script):
    mnist_evens_odds.npz       - cached train/test arrays (regenerate by deleting)
    test_images.csv            - test-set pixels (idx, label, parity, pixel_0..783)
    clf_predictions.csv        - main classifier softmax + logits on test set
    mc_dropout_p03.csv         - T=20 MC-dropout passes, dropout 0.3
    vae_recon_errors.csv       - per-sample VAE reconstruction error
    vae_recon_examples.csv     - 50 evens + 50 odds (orig, recon) pairs for a grid
    vae_latent_codes.csv       - VAE latent means (k=16) for the test set
    confidence_head_preds.csv  - main classifier prediction + scalar confidence
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SEED = 0
DEVICE = torch.device("cpu")

EVEN_LABELS = (0, 2, 4, 6, 8)
ODD_LABELS  = (1, 3, 5, 7, 9)

N_TRAIN_PER_CLASS = 1000   # 5000 evens for training
N_TEST_PER_CLASS  = 200    # 1000 evens + 1000 odds for testing

LATENT_K          = 16
T_MC_DROPOUT      = 20


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_mnist_evens_odds():
    """Load MNIST, build (train_evens, test_evens, test_odds), cache to npz."""
    cache = HERE / "mnist_evens_odds.npz"
    if cache.exists():
        d = np.load(cache)
        return d["X_train"], d["y_train"], d["X_test"], d["y_test"]

    print("Fetching MNIST via torchvision (cached on first run)...")
    from torchvision import datasets
    raw = datasets.MNIST(root=str(HERE / "_mnist_raw"), train=True, download=True)
    X_all = raw.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_all = raw.targets.numpy().astype(int)

    rng = np.random.default_rng(SEED)
    X_train_parts, y_train_parts = [], []
    X_test_parts,  y_test_parts  = [], []

    for d in range(10):
        idx = np.where(y_all == d)[0]
        rng.shuffle(idx)
        if d in EVEN_LABELS:
            X_train_parts.append(X_all[idx[:N_TRAIN_PER_CLASS]])
            y_train_parts.append(y_all[idx[:N_TRAIN_PER_CLASS]])
            test_slice = idx[N_TRAIN_PER_CLASS:N_TRAIN_PER_CLASS + N_TEST_PER_CLASS]
        else:
            test_slice = idx[:N_TEST_PER_CLASS]
        X_test_parts.append(X_all[test_slice])
        y_test_parts.append(y_all[test_slice])

    X_train = np.concatenate(X_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    X_test  = np.concatenate(X_test_parts,  axis=0)
    y_test  = np.concatenate(y_test_parts,  axis=0)

    perm_tr = rng.permutation(len(X_train))
    perm_te = rng.permutation(len(X_test))
    X_train, y_train = X_train[perm_tr], y_train[perm_tr]
    X_test,  y_test  = X_test[perm_te],  y_test[perm_te]

    np.savez(cache, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f"Cached MNIST splits to {cache.name}")
    return X_train, y_train, X_test, y_test


def even_label_to_index(y):
    """Map {0,2,4,6,8} to {0,1,2,3,4}. Other labels become -1."""
    out = np.full_like(y, -1)
    for i, e in enumerate(EVEN_LABELS):
        out[y == e] = i
    return out


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Classifier(nn.Module):
    """Small MLP classifier with dropout. n_out controls 5- vs 6-class heads."""

    def __init__(self, n_out=5, p_drop=0.3):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_out)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        h = self.drop(F.relu(self.fc1(x)))
        h = self.drop(F.relu(self.fc2(h)))
        return self.fc3(h)  # logits


class VAE(nn.Module):
    """Same shape as lecture-06's VAE."""

    def __init__(self, k=LATENT_K, h=256):
        super().__init__()
        self.enc    = nn.Sequential(nn.Linear(784, h), nn.ReLU())
        self.mu     = nn.Linear(h, k)
        self.logvar = nn.Linear(h, k)
        self.dec    = nn.Sequential(nn.Linear(k, h), nn.ReLU(),
                                    nn.Linear(h, 784), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z, self.dec(z)


class ConfidenceClassifier(nn.Module):
    """Classifier with an extra scalar confidence head, DeVries & Taylor 2018.

    Light dropout (p=0.1) on the trunk: heavier dropout shifts the cnf logit
    between train and eval and saturates `c`, while no regularization at all
    lets the cls head fit MNIST evens to ~0 training loss, which gives the
    confidence head no incentive to drop `c` on hard inputs. p=0.1 plus
    weight decay (in the optimizer) and Gaussian input noise during training
    is enough to keep NLL nontrivial without breaking `c` calibration.
    """

    def __init__(self, n_out=5, p_drop=0.2):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.cls = nn.Linear(128, n_out)
        self.cnf = nn.Linear(128, 1)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        h = self.drop(F.relu(self.fc1(x)))
        h = self.drop(F.relu(self.fc2(h)))
        return self.cls(h), torch.sigmoid(self.cnf(h)).squeeze(-1)


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def _batches(n, batch, rng):
    idx = np.arange(n); rng.shuffle(idx)
    for s in range(0, n, batch):
        yield idx[s:s + batch]


def train_classifier(model, X, y_idx, epochs=25, batch=128, lr=1e-3, tag=""):
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y_idx).long()
    rng = np.random.default_rng(SEED)
    model.train()
    for ep in range(epochs):
        total = 0.0
        for sl in _batches(len(X), batch, rng):
            opt.zero_grad()
            logits = model(Xt[sl])
            loss = F.cross_entropy(logits, yt[sl])
            loss.backward(); opt.step()
            total += float(loss) * len(sl)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  [{tag}] epoch {ep+1:3d}/{epochs}  loss={total/len(X):.4f}")
    model.eval()
    return model


def train_vae(model, X, epochs=120, batch=128, lr=1e-3, beta=1.0):
    """Train a VAE with BCE reconstruction. Sigmoid decoder + [0,1] pixels match
    a Bernoulli observation model, so BCE is the correct likelihood (MSE on the
    same head produces blurry reconstructions and a flatter in-dist / OOD gap).
    """
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(X)
    rng = np.random.default_rng(SEED)
    model.train()
    for ep in range(epochs):
        total_recon = total_kl = 0.0
        for sl in _batches(len(X), batch, rng):
            xb = Xt[sl]
            opt.zero_grad()
            mu, logvar, _, x_hat = model(xb)
            recon = F.binary_cross_entropy(x_hat, xb, reduction="sum") / len(xb)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
            loss = recon + beta * kl
            loss.backward(); opt.step()
            total_recon += float(recon) * len(xb)
            total_kl    += float(kl)    * len(xb)
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"  [vae] epoch {ep+1:3d}/{epochs}  "
                  f"recon={total_recon/len(X):.3f}  kl={total_kl/len(X):.3f}")
    model.eval()
    return model


def make_synthetic_ood_batch(X_pool, n, rng):
    """n synthetic OOD inputs: thirds-mix of pixel-shuffled evens, mixup of two
    evens, and heavily rotated evens (60-300 deg).

    The rotated set is the most useful proxy: it preserves digit-like shape but
    breaks the trained orientation, which is the closest 'digit but not from
    training' signal we can synthesise without OOD labels.
    """
    from scipy.ndimage import rotate
    X_pool = np.asarray(X_pool, dtype=np.float32)
    n_each = n // 3
    n_rot  = n - 2 * n_each

    base = X_pool[rng.integers(0, len(X_pool), size=n_each)].copy()
    for i in range(n_each):
        rng.shuffle(base[i])

    a = X_pool[rng.integers(0, len(X_pool), size=n_each)]
    b = X_pool[rng.integers(0, len(X_pool), size=n_each)]
    alpha = rng.uniform(0.3, 0.7, size=(n_each, 1)).astype(np.float32)
    mixed = alpha * a + (1 - alpha) * b

    src = X_pool[rng.integers(0, len(X_pool), size=n_rot)].reshape(-1, 28, 28)
    angles = rng.uniform(60, 300, size=n_rot)
    rotated = np.stack([
        rotate(im, ang, reshape=False, order=1, mode='constant')
        for im, ang in zip(src, angles)
    ]).reshape(-1, 784).astype(np.float32).clip(0.0, 1.0)

    return np.concatenate([base, mixed, rotated], axis=0).astype(np.float32)


def train_confidence_classifier(model, X, y_idx, epochs=80, batch=128, lr=1e-3,
                                lam=0.2, budget=0.5, weight_decay=1e-3,
                                lam_floor=0.15, alpha_ood=0.5,
                                tag="confhead"):
    """DeVries & Taylor 2018 confidence head + outlier exposure.

    Loss per batch:
        NLL on hint-blended p' = c * p + (1-c) * y_onehot     # in-distribution
      + lam * mean(-log(c))                                    # confidence penalty
      + alpha_ood * mean(-log(1 - c_ood))                      # push c -> 0 on synth OOD

    Without the OOD term the head learns to be confident on everything trained-
    distribution looks like, including odd digits the model never saw - the gap
    between in-dist and OOD c stays small. The synthetic OOD batch gives the
    head a real `this is not a clean training digit` gradient signal that
    partially generalises to true OOD.

    No dropout and no input noise: each was tried but introduced a train/eval
    distribution shift on the cnf head that collapsed c to 0 at eval time.
    """
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y_idx).long()
    n_cls = int(y_idx.max() + 1)
    rng = np.random.default_rng(SEED)
    model.train()
    lam_now = float(lam)
    for ep in range(epochs):
        total = 0.0
        last_conf_pen = last_mean_c_id = last_mean_c_ood = 0.0
        for sl in _batches(len(X), batch, rng):
            opt.zero_grad()
            xb = Xt[sl]
            logits, c_id = model(xb)
            p = F.softmax(logits, dim=1)
            yh = F.one_hot(yt[sl], n_cls).float()
            c_ = c_id.unsqueeze(1).clamp(1e-6, 1 - 1e-6)
            p_blend = c_ * p + (1 - c_) * yh
            nll = -torch.log((p_blend * yh).sum(dim=1) + 1e-12).mean()
            conf_pen = -torch.log(c_id.clamp(1e-6, 1.0)).mean()

            ood_np = make_synthetic_ood_batch(X, n=len(sl), rng=rng)
            _, c_ood = model(torch.from_numpy(ood_np))
            ood_loss = -torch.log((1.0 - c_ood).clamp(1e-6, 1.0)).mean()

            loss = nll + lam_now * conf_pen + alpha_ood * ood_loss
            loss.backward(); opt.step()
            total += float(loss) * len(sl)
            cp = float(conf_pen)
            if cp < budget: lam_now /= 1.01
            else:           lam_now *= 1.01
            lam_now = max(lam_floor, lam_now)
            last_conf_pen   = cp
            last_mean_c_id  = float(c_id.mean())
            last_mean_c_ood = float(c_ood.mean())
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  [{tag}] epoch {ep+1:3d}/{epochs}  "
                  f"loss={total/len(X):.4f}  mean_c_id={last_mean_c_id:.3f}  "
                  f"mean_c_ood={last_mean_c_ood:.3f}  lam={lam_now:.3f}")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Asset generators
# ---------------------------------------------------------------------------

def write_main_predictions(model, X_test, y_test, out_path):
    """Single deterministic pass: top-1 softmax + max-logit + raw logits + label."""
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test)).numpy()
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    parity = np.where(np.isin(y_test, EVEN_LABELS), "even", "odd")
    df = pd.DataFrame({
        "idx":          np.arange(len(X_test)),
        "label":        y_test,
        "parity":       parity,
        "top1_idx":     probs.argmax(axis=1),
        "top1_label":   np.array(EVEN_LABELS)[probs.argmax(axis=1)],
        "top1_softmax": probs.max(axis=1),
        "max_logit":    logits.max(axis=1),
    })
    for i, e in enumerate(EVEN_LABELS):
        df[f"logit_{e}"] = logits[:, i]
        df[f"prob_{e}"]  = probs[:, i]
    df.to_csv(out_path, index=False)
    print(f"  wrote {out_path.name}  ({len(df)} rows)")


def write_mc_dropout_passes(model, X_test, y_test, T, out_path):
    """T forward passes with dropout enabled, store per-pass softmax probs."""
    Xt = torch.from_numpy(X_test)
    parity = np.where(np.isin(y_test, EVEN_LABELS), "even", "odd")
    cols = {"idx": np.arange(len(X_test)), "label": y_test, "parity": parity}
    model.train()  # enable dropout
    with torch.no_grad():
        for t in range(T):
            torch.manual_seed(SEED + 100 + t)
            logits = model(Xt).numpy()
            p = np.exp(logits - logits.max(axis=1, keepdims=True))
            p = p / p.sum(axis=1, keepdims=True)
            for i, e in enumerate(EVEN_LABELS):
                cols[f"t{t:02d}_c{e}"] = p[:, i]
    model.eval()
    pd.DataFrame(cols).to_csv(out_path, index=False)
    print(f"  wrote {out_path.name}  ({len(X_test)} rows, T={T})")


def write_vae_outputs(vae, X_test, y_test, errors_path, examples_path, latent_path):
    """Reconstruction errors, a sample of (orig, recon) pairs, and latent codes.

    Decode from `mu` directly at eval. Going through `reparameterize` would
    add stochastic noise to every reconstruction and dilute the in-dist / OOD
    contrast we want to read off recon error.
    """
    Xt = torch.from_numpy(X_test)
    with torch.no_grad():
        mu_t, _ = vae.encode(Xt)
        x_hat_t = vae.dec(mu_t)
    mu      = mu_t.numpy()
    x_hat   = x_hat_t.numpy()
    recon_e = ((X_test - x_hat) ** 2).mean(axis=1)
    parity  = np.where(np.isin(y_test, EVEN_LABELS), "even", "odd")

    pd.DataFrame({"idx": np.arange(len(X_test)), "label": y_test, "parity": parity,
                  "recon_error": recon_e}).to_csv(errors_path, index=False)
    print(f"  wrote {errors_path.name}  ({len(X_test)} rows)")

    rng = np.random.default_rng(SEED + 3)
    even_idx = rng.choice(np.where(parity == "even")[0], size=50, replace=False)
    odd_idx  = rng.choice(np.where(parity == "odd")[0],  size=50, replace=False)
    sample_idx = np.concatenate([even_idx, odd_idx])
    rows = []
    for i in sample_idx:
        row = {"idx": int(i), "label": int(y_test[i]), "parity": parity[i],
               "recon_error": float(recon_e[i])}
        for j in range(784): row[f"orig_{j}"]  = float(X_test[i, j])
        for j in range(784): row[f"recon_{j}"] = float(x_hat[i, j])
        rows.append(row)
    pd.DataFrame(rows).to_csv(examples_path, index=False)
    print(f"  wrote {examples_path.name}  ({len(rows)} rows)")

    cols = {"idx": np.arange(len(X_test)), "label": y_test, "parity": parity}
    for k in range(LATENT_K):
        cols[f"mu_{k}"] = mu[:, k]
    pd.DataFrame(cols).to_csv(latent_path, index=False)
    print(f"  wrote {latent_path.name}  ({len(X_test)} rows, k={LATENT_K})")


def write_confidence_head_preds(model, X_test, y_test, out_path):
    """Single deterministic eval pass.

    With outlier exposure during training the head no longer saturates at
    `c = 1` on a single eval-mode pass, so we do not need MC-Dropout averaging
    to desaturate. Eval-time MC averaging here actually hurts: it reintroduces
    a train/eval distribution shift on the cnf head and collapses `c`.
    """
    model.eval()
    Xt = torch.from_numpy(X_test)
    with torch.no_grad():
        logits, c = model(Xt)
    logits = logits.numpy(); c = c.numpy()
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    parity = np.where(np.isin(y_test, EVEN_LABELS), "even", "odd")
    df = pd.DataFrame({
        "idx":          np.arange(len(X_test)),
        "label":        y_test,
        "parity":       parity,
        "top1_idx":     probs.argmax(axis=1),
        "top1_label":   np.array(EVEN_LABELS)[probs.argmax(axis=1)],
        "top1_softmax": probs.max(axis=1),
        "confidence":   c,
    })
    df.to_csv(out_path, index=False)
    print(f"  wrote {out_path.name}  ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    print("=" * 72)
    print("Chapter 7 precomputed-asset generator")
    print("=" * 72)

    X_train, y_train, X_test, y_test = load_mnist_evens_odds()
    y_train_idx = even_label_to_index(y_train)
    print(f"Train evens : {X_train.shape}    Test (mixed): {X_test.shape}")

    # 0. Test-set pixels: every visualization that wants to show an example
    # image merges on `idx` against this file.
    parity = np.where(np.isin(y_test, EVEN_LABELS), "even", "odd")
    img_cols = {"idx": np.arange(len(X_test)), "label": y_test, "parity": parity}
    for j in range(784):
        img_cols[f"pixel_{j}"] = X_test[:, j]
    pd.DataFrame(img_cols).to_csv(HERE / "test_images.csv", index=False)
    print(f"  wrote test_images.csv  ({len(X_test)} rows)")

    # 1. Main classifier (dropout p=0.3): used for the calibration trap and
    # the MC-dropout passes.
    print("\n-- training main classifier (dropout p=0.3) --")
    main_clf = Classifier(n_out=5, p_drop=0.3)
    train_classifier(main_clf, X_train, y_train_idx, tag="clf-p03")
    write_main_predictions(main_clf, X_test, y_test, HERE / "clf_predictions.csv")
    write_mc_dropout_passes(main_clf, X_test, y_test, T_MC_DROPOUT,
                            HERE / "mc_dropout_p03.csv")

    # 2. VAE on evens.
    print("\n-- training VAE on evens --")
    vae = VAE(k=LATENT_K)
    train_vae(vae, X_train)
    write_vae_outputs(vae, X_test, y_test,
                      errors_path  =HERE / "vae_recon_errors.csv",
                      examples_path=HERE / "vae_recon_examples.csv",
                      latent_path  =HERE / "vae_latent_codes.csv")

    # 3. Confidence-head classifier (DeVries & Taylor + outlier exposure).
    # No dropout: the OOD-exposure recipe is sensitive to train/eval shift.
    print("\n-- training classifier with confidence head --")
    conf_clf = ConfidenceClassifier(n_out=5, p_drop=0.0)
    train_confidence_classifier(conf_clf, X_train, y_train_idx, tag="confhead")
    write_confidence_head_preds(conf_clf, X_test, y_test,
                                HERE / "confidence_head_preds.csv")

    print("\nDone. CSVs written next to this script.")


if __name__ == "__main__":
    main()
