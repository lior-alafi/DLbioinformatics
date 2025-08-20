# utils/experiment_tracker.py
import os, csv, datetime
import matplotlib.pyplot as plt
import torch

class ExperimentTracker:
    """
    Handles experiment logging, saving best models, generating CSVs and plots.

    Usage:
        tracker = ExperimentTracker(
            out_dir="experiments_out",
            exp_name="transformer_test",
            hyperparams={"embed_dim": 128, "hidden": 64, "layers": 4, "heads": 4, "alpha": 0.1, "beta": 0.0, "gamma": 1.5},
            track_best_by="pearson"
        )

        # Inside training loop:
        tracker.log_epoch(epoch, train_loss, val_loss, pearson, spearman, model)

        # After training:
        tracker.export()
    """

    def __init__(self, out_dir, exp_name, hyperparams, track_best_by="pearson"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.exp_name = f"{exp_name}_{self._ts()}"
        self.hyperparams = hyperparams
        self.track_best_by = track_best_by

        self.history = {"epoch": [], "train_loss": [], "val_loss": [], "pearson": [], "spearman": []}
        self.best = {"epoch": -1, "val_loss": float("inf"), "pearson": -999.0, "spearman": -999.0, "state_dict": None}

    def _ts(self):
        return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def log_epoch(self, epoch, train_loss, val_loss, pearson, spearman, model):
        # Save history
        self.history["epoch"].append(epoch + 1)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["pearson"].append(pearson)
        self.history["spearman"].append(spearman)

        # Update best
        criterion_value = pearson if self.track_best_by == "pearson" else -val_loss
        best_value = self.best["pearson"] if self.track_best_by == "pearson" else -self.best["val_loss"]

        if criterion_value > best_value:
            self.best.update({
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "pearson": pearson,
                "spearman": spearman,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}
            })

    def _save_model(self):
        if self.best["state_dict"] is not None:
            path = os.path.join(self.out_dir, f"{self.exp_name}_BEST.pt")
            torch.save(self.best["state_dict"], path)
            print(f"[{self.exp_name}] Saved best model → {path}")

    def _save_history_csv(self):
        path = os.path.join(self.out_dir, f"{self.exp_name}_history.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "pearson", "spearman"])
            for e, tl, vl, p, s in zip(
                self.history["epoch"], self.history["train_loss"], self.history["val_loss"], self.history["pearson"], self.history["spearman"]
            ):
                writer.writerow([e, f"{tl:.6f}", f"{vl:.6f}", f"{p:.6f}", f"{s:.6f}"])
        print(f"[{self.exp_name}] Saved per-epoch CSV → {path}")

    def _append_summary_csv(self):
        path = os.path.join(self.out_dir, "summary_all_experiments.csv")
        exists = os.path.isfile(path)
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow([
                    "experiment_id", *self.hyperparams.keys(),
                    "best_epoch", "best_val_loss", "best_pearson", "best_spearman"
                ])
            writer.writerow([
                self.exp_name, *self.hyperparams.values(),
                self.best["epoch"], f"{self.best['val_loss']:.6f}",
                f"{self.best['pearson']:.6f}", f"{self.best['spearman']:.6f}"
            ])
        print(f"[{self.exp_name}] Appended to summary CSV → {path}")

    def _save_plot(self):
        fig = plt.figure(figsize=(10, 6))
        plt.plot(self.history["epoch"], self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["epoch"], self.history["val_loss"], label="Val Loss")
        plt.plot(self.history["epoch"], self.history["pearson"], label="Val Pearson")
        plt.plot(self.history["epoch"], self.history["spearman"], label="Val Spearman")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(self.exp_name)
        plt.legend()
        path = os.path.join(self.out_dir, f"{self.exp_name}_plot.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[{self.exp_name}] Saved plot → {path}")

    def export(self):
        """Save model, CSV history, summary, and plot."""
        self._save_model()
        self._save_history_csv()
        self._append_summary_csv()
        self._save_plot()
