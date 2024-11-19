"""
Usage:
    python3 -m homework.train_planner --model_name <model_name>
"""
import numpy as np
import torch

from .datasets.road_dataset import load_data
from .metrics import PlannerMetric
from .models import load_model, save_model


def train_planner(
    model_name: str,
    num_epoch: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 2024,
    **kwargs,
):
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Load the model
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    # Load data
    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2, transform_pipeline="default")
    val_data = load_data("drive_data/val", shuffle=False, transform_pipeline="default")

    # Create loss function, optimizer, and metrics
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    planner_train = PlannerMetric()
    planner_val = PlannerMetric()

    # Training loop
    for epoch in range(num_epoch):
        planner_train.reset()
        planner_val.reset()

        # Training phase
        model.train()
        for batch in train_data:
            label = batch["waypoints"].to(device)
            label_mask = batch["waypoints_mask"].to(device)

            if model_name in ["mlp_planner", "transformer_planner"]:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                out = model(track_left, track_right)  # For MLP and Transformer models
            elif model_name == "cnn_planner":
                img = batch["image"].to(device)
                out = model(img)  # For CNN model
            else:
                raise ValueError(f"Unsupported model_name: {model_name}")

            # Loss function with lateral error weighting
            lateral_loss_weight = 2.0
            lateral_error = torch.abs(out[..., 0] - label[..., 0]) * label_mask
            longitudinal_error = torch.abs(out[..., 1] - label[..., 1]) * label_mask
            loss = (lateral_loss_weight * lateral_error + longitudinal_error).mean()

            planner_train.add(out, label, label_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        with torch.inference_mode():
            for batch in val_data:
                label = batch["waypoints"].to(device)
                label_mask = batch["waypoints_mask"].to(device)

                if model_name in ["mlp_planner", "transformer_planner"]:
                    track_left = batch["track_left"].to(device)
                    track_right = batch["track_right"].to(device)
                    out = model(track_left, track_right)
                elif model_name == "cnn_planner":
                    img = batch["image"].to(device)
                    out = model(img)
                else:
                    raise ValueError(f"Unsupported model_name: {model_name}")

                planner_val.add(out, label, label_mask)

        # Compute and log metrics
        train_metrics = planner_train.compute()
        val_metrics = planner_val.compute()

        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"train_longitudinal_error={train_metrics['longitudinal_error']:.4f} "
            f"train_lateral_error={train_metrics['lateral_error']:.4f} "
            f"val_longitudinal_error={val_metrics['longitudinal_error']:.4f} "
            f"val_lateral_error={val_metrics['lateral_error']:.4f}"
        )

    # Save model
    save_model(model)
    print(f"Model {model_name} saved successfully!")


if __name__ == "__main__":
    import argparse

    # Argument parsing
    parser = argparse.ArgumentParser(description="Train a planner model.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model to train (mlp_planner, transformer_planner, cnn_planner)",
    )
    parser.add_argument("--num_epoch", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    args = parser.parse_args()

    # Train the model
    train_planner(
        model_name=args.model_name,
        num_epoch=args.num_epoch,
        lr=args.lr,
        batch_size=args.batch_size,
    )
