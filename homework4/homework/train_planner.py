import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from .datasets.road_dataset import load_data
from .metrics import PlannerMetric
from .models import load_model, save_model

"""
Usage:
    python3 -m homework.train_planner --your_args here
"""
def train_planner(
    model_name: str = "cnn_planner",
    train_data_path: str = "drive_data/train",
    val_data_path: str = "drive_data/val",
    num_epoch: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
        except Exception as e:
            print(f"CUDA initialization failed: {e}. Using CPU.")
            device = torch.device("cpu")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    torch.manual_seed(seed)

    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data(train_data_path, shuffle=True, batch_size=batch_size, num_workers=2, transform_pipeline="default")
    val_data = load_data(val_data_path, shuffle=False, transform_pipeline="default")

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    planner_train = PlannerMetric()
    planner_val = PlannerMetric()

    for epoch in range(num_epoch):
        planner_train.reset()
        planner_val.reset()

        model.train()
        for batch in train_data:
            img = batch["image"].to(device)
            label = batch["waypoints"].to(device)
            label_mask = batch["waypoints_mask"].to(device)
            out = model(img)
            loss = torch.mean((out - label) ** 2 * label_mask.unsqueeze(-1))  # Apply mask
            planner_train.add(out, label, label_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            for batch in val_data:
                img = batch["image"].to(device)
                label = batch["waypoints"].to(device)
                label_mask = batch["waypoints_mask"].to(device)
                out = model(img)
                planner_val.add(out, label, label_mask)

        scheduler.step()

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            train_metrics = planner_train.compute()
            val_metrics = planner_val.compute()
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_longitudinal_error={train_metrics['longitudinal_error']:.4f} "
                f"train_lateral_error={train_metrics['lateral_error']:.4f} "
                f"val_longitudinal_error={val_metrics['longitudinal_error']:.4f} "
                f"val_lateral_error={val_metrics['lateral_error']:.4f}"
            )

    save_model(model)

if __name__ == "__main__":
    # Define the command-line arguments
    parser = argparse.ArgumentParser(description="Train planner models")
    parser.add_argument(
        "--model_name",
        type=str,
        default="cnn_planner",
        help="Model to train. Options: mlp_planner, transformer_planner, cnn_planner, or all",
    )
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=100,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training",
    )
    args = parser.parse_args()

    # Define the models to train
    models_to_train = []
    if args.model_name == "all":
        models_to_train = ["mlp_planner", "transformer_planner", "cnn_planner"]
    else:
        models_to_train = [args.model_name]

    # Train each model
    for model_name in models_to_train:
        print(f"Training model: {model_name}")
        train_planner(
            model_name=model_name,
            num_epoch=args.num_epoch,
            lr=args.lr,
            batch_size=args.batch_size,
        )
        print(f"Finished training: {model_name}")

