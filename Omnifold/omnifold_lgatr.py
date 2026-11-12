"""
L-GATr-slim implementation of OmniFold.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import Optional
from tqdm import tqdm

from lgatr import LGATrSlim


# ---------------------------------------------------------------------------
# 1. Convert EventObjects → (vectors, scalars) tensors
# ---------------------------------------------------------------------------

def events_to_fourmomenta(events, device="cpu"):
    """
    Convert EventObjects to L-GATr-slim input tensors.

    Particle ordering: [j1, j2, electron, muon, MET]  →  N_particles = 5

    Returns
    -------
    vectors : torch.Tensor  (N_events, 5, 1, 4)   four-momenta [E, px, py, pz]
    scalars : torch.Tensor  (N_events, 5, 4)       one-hot particle type
    """
    import awkward as ak

    def to_epxpypz(obj):
        return np.stack([
            np.asarray(ak.to_numpy(obj.energy)),
            np.asarray(ak.to_numpy(obj.px)),
            np.asarray(ak.to_numpy(obj.py)),
            np.asarray(ak.to_numpy(obj.pz)),
        ], axis=-1)  # (N, 4)

    j1 = to_epxpypz(events.jets[:, 0])
    j2 = to_epxpypz(events.jets[:, 1])
    el = to_epxpypz(events.electrons[:, 0])
    mu = to_epxpypz(events.muons[:, 0])

    met_pt  = np.asarray(ak.to_numpy(events.met.pt)).squeeze()
    met_phi = np.asarray(ak.to_numpy(events.met.phi)).squeeze()
    met_p4  = np.stack([
        met_pt,
        met_pt * np.cos(met_phi),
        met_pt * np.sin(met_phi),
        np.zeros_like(met_pt),
    ], axis=-1)


    p4 = np.stack([j1, j2, el, mu, met_p4], axis=1)  # (N, 5, 4)
    vectors = torch.tensor(p4, dtype=torch.float32).unsqueeze(2)  # (N, 5, 1, 4)

    # one-hot particle type: [is_jet, is_electron, is_muon, is_met]
    n = len(j1)
    type_onehot = np.zeros((n, 5, 4), dtype=np.float32)
    type_onehot[:, 0, 0] = 1  # j1
    type_onehot[:, 1, 0] = 1  # j2
    type_onehot[:, 2, 1] = 1  # electron
    type_onehot[:, 3, 2] = 1  # muon
    type_onehot[:, 4, 3] = 1  # MET
    scalars = torch.tensor(type_onehot, dtype=torch.float32)

    return vectors.to(device), scalars.to(device)


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------

class LGATrSlimClassifier(nn.Module):
    """
    Per-event binary classifier built on LGATrSlim.
    Mean-pools particle scalar outputs, then applies a linear head.
    """
    def __init__(
        self,
        num_blocks=4,
        hidden_v_channels=16,
        hidden_s_channels=32,
        num_heads=4,
        in_s_channels=4,
    ):
        super().__init__()
        self.net = LGATrSlim(
            num_blocks=num_blocks,
            in_v_channels=1,
            out_v_channels=1,
            hidden_v_channels=hidden_v_channels,
            in_s_channels=in_s_channels,
            out_s_channels=hidden_s_channels,
            hidden_s_channels=hidden_s_channels,
            num_heads=num_heads,
        )
        self.head = nn.Linear(hidden_s_channels, 1)

    def forward(self, vectors, scalars):
        # vectors: (B, N, 1, 4),  scalars: (B, N, C_s)
        _, out_s = self.net(vectors=vectors, scalars=scalars)
        pooled = out_s.mean(dim=1)           # (B, hidden_s_channels)
        return self.head(pooled).squeeze(-1) # (B,)


# ---------------------------------------------------------------------------
# 3. Weighted BCE loss  (mirrors omnifold.py weighted_binary_crossentropy)
# ---------------------------------------------------------------------------

def _weighted_bce(logits, labels, weights):
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    return (loss * weights).sum() / weights.sum()


# ---------------------------------------------------------------------------
# 4. Train one classifier step
# ---------------------------------------------------------------------------

def _train_step(
    model,
    v_sim, s_sim, w_sim,       # simulation
    v_data, s_data, w_data,    # data (or reweighted gen in step 2)
    n_epochs, batch_size, lr, device,
    verbose=0,
):
    n_sim  = v_sim.shape[0]
    n_data = v_data.shape[0]

    # Normalise total weights of each class to be equal
    w_sim_t  = torch.tensor(
        w_sim  / w_sim.sum()  * n_sim,  dtype=torch.float32, device=device)
    w_data_t = torch.tensor(
        w_data / w_data.sum() * n_data, dtype=torch.float32, device=device)

    v_all = torch.cat([v_sim,  v_data],  dim=0)
    s_all = torch.cat([s_sim,  s_data],  dim=0)
    y_all = torch.cat([torch.zeros(n_sim, device=device),
                       torch.ones(n_data,  device=device)])
    w_all = torch.cat([w_sim_t, w_data_t])

    loader = DataLoader(
        TensorDataset(v_all, s_all, y_all, w_all),
        batch_size=batch_size, shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, train_accs = [], []
    model.train()
    for epoch_idx in range(n_epochs):
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch_idx+1}/{n_epochs}", leave=False)
        for v_b, s_b, y_b, w_b in pbar:
            optimizer.zero_grad()
            logits = model(v_b, s_b)
            loss   = _weighted_bce(logits, y_b, w_b)
            loss.backward()
            optimizer.step()
            epoch_loss    += loss.item() * len(y_b)
            epoch_correct += ((logits > 0) == y_b.bool()).sum().item()
            epoch_total   += len(y_b)
            pbar.set_postfix(loss=f"{epoch_loss/epoch_total:.4f}",
                            acc=f"{epoch_correct/epoch_total:.4f}")
        train_losses.append(epoch_loss / epoch_total)
        train_accs.append(epoch_correct / epoch_total)
        if verbose > 0:
            print(f"  epoch {epoch_idx+1}/{n_epochs}  loss={train_losses[-1]:.4f}  acc={train_accs[-1]:.4f}")
    return train_losses, train_accs


# ---------------------------------------------------------------------------
# 5. Reweight  (mirrors omnifold.reweight)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _reweight(model, vectors, scalars, batch_size, device):
    model.eval()
    all_w = []
    for v_b, s_b in DataLoader(TensorDataset(vectors, scalars), batch_size=batch_size):
        probs = torch.sigmoid(model(v_b.to(device), s_b.to(device))).cpu().numpy()
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        all_w.append(probs / (1 - probs))
    return np.concatenate(all_w)


# ---------------------------------------------------------------------------
# 6. Loss plot  (mirrors _save_loss_plot from omnifold_ydf)
# ---------------------------------------------------------------------------

def _save_loss_plot_lgatr(train_losses, train_accs, report_dir, i, step):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss.plot(epochs, train_losses, label="train")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title(f"Loss — Iter {i + 1}, Step {step}")
    ax_loss.legend()

    ax_acc.plot(epochs, train_accs, label="train")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title(f"Accuracy — Iter {i + 1}, Step {step}")
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(
        os.path.join(report_dir, f"loss_iter{i}_step{step}.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. omnifold_lgatr  —  drop-in replacement for omnifold_ydf
# ---------------------------------------------------------------------------

def omnifold_lgatr(
    theta0: np.ndarray,
    theta_unknown_S: np.ndarray,
    iterations: int,
    prior_gen_events,
    prior_reco_events,
    data_reco_events,
    feature_names: Optional[list] = None,   # kept for API compatibility, unused
    verbose: int = 0,
    save_report_path: Optional[str] = None,
    # L-GATr-slim hyperparameters
    num_blocks: int = 4,
    hidden_v_channels: int = 16,
    hidden_s_channels: int = 32,
    num_heads: int = 4,
    n_epochs: int = 20,
    batch_size: int = 1024,
    lr: float = 1e-3,
    device: str = "cpu",
) -> np.ndarray:
    """
    OmniFold with L-GATr-slim classifiers.

    Parameters
    ----------
    theta0 : np.ndarray  shape (N, 2, F)
        theta0[:, 0] = gen features, theta0[:, 1] = reco features.
        Used only to determine N (number of prior events).
    theta_unknown_S : np.ndarray  shape (M, F)
        Used only to determine M (number of data events).
    prior_gen_events  : EventObjects  matched gen  (N events, same order as theta0)
    prior_reco_events : EventObjects  matched reco (N events, same order as theta0)
    data_reco_events  : EventObjects  observed data reco (M events)

    Returns
    -------
    weights : np.ndarray  shape (iterations, 2, N)
        weights[i, 0] = step-1 reco weights after iteration i+1
        weights[i, 1] = step-2 gen  weights after iteration i+1
    """
    n_prior = len(theta0)
    weights = np.empty((iterations, 2, n_prior))

    model_kwargs = dict(
        num_blocks=num_blocks,
        hidden_v_channels=hidden_v_channels,
        hidden_s_channels=hidden_s_channels,
        num_heads=num_heads,
    )

    report_dir = None
    if save_report_path is not None:
        report_dir = os.path.join(save_report_path, "lgatr_reports")
        os.makedirs(report_dir, exist_ok=True)

    # Pre-compute four-momenta tensors once (expensive for large datasets)
    if verbose > 0:
        print("Preparing four-momenta tensors...")
    v_prior_reco, s_prior_reco = events_to_fourmomenta(prior_reco_events, device=device)
    v_data_reco,  s_data_reco  = events_to_fourmomenta(data_reco_events,  device=device)
    v_prior_gen,  s_prior_gen  = events_to_fourmomenta(prior_gen_events,  device=device)

    # Initial weights: all ones (same as omnifold_ydf)
    weights_push = np.ones(n_prior, dtype=np.float32)  # step-2 output, feeds step-1

    for i in range(iterations):
        if verbose > 0:
            print(f"\nITERATION: {i + 1}\n")

        # ---- Step 1: classify reco sim (reweighted by weights_push) vs data ----
        if verbose > 0:
            print("STEP 1")

        n_data = v_data_reco.shape[0]
        model_step1 = LGATrSlimClassifier(**model_kwargs).to(device)
        losses1, accs1 = _train_step(
            model=model_step1,
            v_sim=v_prior_reco, s_sim=s_prior_reco,
            w_sim=weights_push,
            v_data=v_data_reco,  s_data=s_data_reco,
            w_data=np.ones(n_data, dtype=np.float32),
            n_epochs=n_epochs, batch_size=batch_size, lr=lr, device=device,
            verbose=verbose,
        )

        # weights_pull = weights_push * f/(1-f)  (same accumulation as YDF step 1)
        weights_pull = weights_push * _reweight(
            model_step1, v_prior_reco, s_prior_reco, batch_size, device)
        weights[i, 0] = weights_pull

        if report_dir is not None:
            _save_loss_plot_lgatr(losses1, accs1, report_dir, i, step=1)

        # ---- Step 2: classify gen vs reweighted gen ----
        if verbose > 0:
            print("STEP 2")

        model_step2 = LGATrSlimClassifier(**model_kwargs).to(device)
        losses2, accs2 = _train_step(
            model=model_step2,
            v_sim=v_prior_gen,  s_sim=s_prior_gen,
            w_sim=np.ones(n_prior, dtype=np.float32),
            v_data=v_prior_gen, s_data=s_prior_gen,   # same events, different weights
            w_data=weights_pull,
            n_epochs=n_epochs, batch_size=batch_size, lr=lr, device=device,
            verbose=verbose
        )

        weights_push = _reweight(
            model_step2, v_prior_gen, s_prior_gen, batch_size, device)
        weights[i, 1] = weights_push

        if report_dir is not None:
            _save_loss_plot_lgatr(losses2, accs2, report_dir, i, step=2)

    return weights
