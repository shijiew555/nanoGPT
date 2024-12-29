import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from lightning import Trainer
import lightning as L
from torch.nn import functional as F
from torch.optim import Adam, SGD
from model import GPT, GPTConfig
from lightning.pytorch.profilers import SimpleProfiler


class GenerateDataset(Dataset):
    def __init__(self, num_samples, sequence_length, p_heads):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.p_heads = p_heads

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sequence = torch.bernoulli(
            torch.full((self.sequence_length,), self.p_heads)
        ).long()
        return sequence


class CoinFlipModel(L.LightningModule):
    def __init__(self, model, optimizer_type):
        super().__init__()
        self.model = model
        self.optimizer_type = optimizer_type
        self.training_times = []
        self.forward_time = []

    def training_step(self, batch, batch_idx):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        start_time = time.time()
        logits, loss = self.model(inputs, targets)
        elapsed_time = time.time() - start_time
        self.forward_time.append(elapsed_time)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        if self.optimizer_type == "SGD":
            return SGD(self.model.parameters(), lr=5e-4)
        elif self.optimizer_type == "Adam":
            return Adam(self.model.parameters(), lr=5e-4)

    def optimizer_zero_grad(self, *args, **kwargs):
        start_time = time.time()
        super().optimizer_zero_grad(*args, **kwargs)
        elapsed_time = time.time() - start_time
        self.training_times.append(elapsed_time)


def train_model(
    optimizer_type, num_samples=1000, sequence_length=10, p_heads=0.666
) -> list:
    dataset = GenerateDataset(num_samples, sequence_length, p_heads)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    config = GPTConfig(
        vocab_size=2,
        block_size=sequence_length,
        n_layer=1,
        n_head=2,
        n_embd=16,
    )
    model = GPT(config)
    pl_model = CoinFlipModel(model, optimizer_type)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        trainer = Trainer(
            max_epochs=1,
            accelerator="gpu",
            devices=device,
            profiler=SimpleProfiler(),
            enable_progress_bar=False,
        )
    else:
        trainer = Trainer(
            max_epochs=1,
            devices=1,
            profiler=SimpleProfiler(),
            enable_progress_bar=False,
        )

    trainer.fit(model=pl_model, train_dataloaders=dataloader)
    optimizer_step_times = trainer.profiler.recorded_durations['[LightningModule]CoinFlipModel.optimizer_step'][1:]
    trainer.profiler.recorded_durations['forward_time'] = pl_model.forward_time
    
    record_time(optimizer_type, trainer.profiler.recorded_durations)
    
    return optimizer_step_times

def record_time(optimizer_type, recorded_durations):
    # Data for the table
    rows = [
        ["Operations", "Total Time (Seconds)"],
        ["Batch Load", sum(recorded_durations['[_TrainingEpochLoop].train_dataloader_next'][1:])],
        ["Forward Pass", sum(recorded_durations['forward_time'][1:])],
        ["Backward Pass", sum(recorded_durations['[Strategy]SingleDeviceStrategy.backward'][1:])],
        ["Optimizer Step", sum(recorded_durations['[LightningModule]CoinFlipModel.optimizer_step'][1:])],
    ]
    
    plt.figure(figsize=(4, 4))

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Hide the default axes
    ax.axis("tight")
    ax.axis("off")

    # Create the table
    table = ax.table(
        cellText=rows,       # Data for the table cells
        loc="center",        # Table location
        cellLoc="center"
    )
    
    # Save the plot
    table_path = os.path.join(f"profiled_{optimizer_type}_times.png")
    plt.savefig(table_path)
    plt.close()

def bootstrap(data, n_bootstraps=1000):
    return [
        np.mean(np.random.choice(data, len(data), replace=True))
        for _ in range(n_bootstraps)
    ]

def bootstrap_pairwise_diff(sgd_times, adam_times, n_bootstrap=1000):
    assert len(sgd_times) == len(
        adam_times
    ), "SGD and Adam must have the same number of samples."

    # pairwise differences
    pairwise_diff = np.array(sgd_times) - np.array(adam_times)

    # sampling of differences
    bootstrap_diffs = [
        np.mean(np.random.choice(pairwise_diff, size=len(pairwise_diff), replace=True))
        for _ in range(n_bootstrap)
    ]

    return pairwise_diff, bootstrap_diffs

def plot_time_distributions(sgd_times, adam_times):

    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    data = {
        "SGD": sgd_times[1:],
        "Adam": adam_times[1:],
    }
    sns.boxplot(data=data)
    plt.title("Optimizer Step Time Distribution for SGD vs Adam Optimizer")
    plt.ylabel("Time (s)")
    plt.xlabel("Optimizer")

    # plt.tight_layout()
    boxplot_path = os.path.join("boxplot.png")
    plt.savefig(boxplot_path)
    plt.close()

    sgd_bootstrap = bootstrap(sgd_times)
    adam_bootstrap = bootstrap(adam_times)

    plt.figure(figsize=(12, 6))
    sns.histplot(sgd_bootstrap, kde=True, label="SGD Mean Distribution", color="blue")
    sns.histplot(
        adam_bootstrap, kde=True, label="Adam Mean Distribution", color="orange"
    )
    plt.legend()
    plt.title("Bootstrap Mean Distributions")
    plt.xlabel("Mean Optimizer Step Time (s)")
    plt.ylabel("Frequency")
    histplot_path = os.path.join("bootstrap.png")
    plt.savefig(histplot_path)
    plt.close()

    pairwise_diff, bootstrap_diffs = bootstrap_pairwise_diff(sgd_times, adam_times)

    # bootstrap distribution of differences
    plt.figure(figsize=(12, 6))
    sns.histplot(bootstrap_diffs, kde=True, color="purple")
    plt.axvline(
        np.mean(pairwise_diff), color="red", linestyle="--", label="Mean Difference"
    )
    plt.legend()
    plt.title("Bootstrap Distribution of Pairwise Differences (SGD - Adam)")
    plt.xlabel("SGD Step Time - Adam Step Time (s)")
    plt.ylabel("Frequency")
    histplot_path = os.path.join("timediff.png")
    plt.savefig(histplot_path)
    plt.close()


def main() -> None:

    sgd_times = train_model("SGD")
    adam_times = train_model("Adam")

    print(f"sgd_times: {sgd_times}")
    print(f"adam_times: {adam_times}")

    plot_time_distributions(sgd_times, adam_times)


if __name__ == "__main__":
    main()