import wandb
import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Any, Mapping

def init_omnifold_run(
    iterations: int,
    theta0: np.ndarray,
    theta_unknown_S: np.ndarray,
    extra_config: Optional[Dict[str, Any]] = None,
) -> None:
    config: Dict[str, Any] = {
        "iterations":  iterations,
        "features":    theta0.shape[-1],
        "n_sim":       len(theta0),
        "n_data":      len(theta_unknown_S),
    }
    if extra_config:
        config.update(extra_config)

    wandb.init(
        entity="rikvanrhee-nikhef",
        project="hww-omnifold",
        config=config,
    )


class WandbOmniFoldCallback(tf.keras.callbacks.Callback):
    def __init__(self, iteration: int, step: int) -> None:
        super().__init__()
        self.iteration = iteration
        self.step = step

    def on_epoch_end(self, epoch: int, logs: Optional[Mapping[str, Any]] = None) -> None:
        logs = logs or {}
        wandb.log({
            f"iter{self.iteration+1}_step{self.step}_loss":     logs.get("loss"),
            f"iter{self.iteration+1}_step{self.step}_val_loss": logs.get("val_loss"),
            f"iter{self.iteration+1}_step{self.step}_acc":      logs.get("accuracy"),
            "iteration": self.iteration + 1,
            "step":      self.step,
            "epoch":     epoch,
        })


def log_weights(
    iteration: int,
    weights_pull: np.ndarray,
    weights_push: np.ndarray,
) -> None:
    wandb.log({
        f"iter{iteration+1}_pull_weights_mean": weights_pull.mean(),
        f"iter{iteration+1}_pull_weights_std":  weights_pull.std(),
        f"iter{iteration+1}_push_weights_mean": weights_push.mean(),
        f"iter{iteration+1}_push_weights_std":  weights_push.std(),
        "iteration": iteration + 1,
    })