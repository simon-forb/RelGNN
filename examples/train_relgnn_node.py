import argparse
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from atomic_routes import get_atomic_routes
from relbench.base import EntityTask, TaskType
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relgnn_model import RelGNN_Model
from text_embedder import GloveTextEmbedding
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from tabmodel_for_relbench.f1_simplified import (
    SimplifiedDriverDNF,
    SimplifiedDriverTop3,
    SimplifiedF1Dataset,
)
from tabmodel_for_relbench.utils import print_binary_classification_metrics


def parse_args():
    p = argparse.ArgumentParser()
    # Optimization
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--weight_decay", type=float, default=0.0)

    # Sampling
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--num_neighbors", type=int, default=128)
    p.add_argument("--subgraph_type", type=str, default="directional")
    p.add_argument("--temporal_strategy", type=str, default="uniform")
    p.add_argument("--max_steps_per_epoch", type=int, default=2000)

    # Model (RelGNN)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--num_model_layers", type=int, default=2)
    p.add_argument("--aggr", type=str, default="sum")
    p.add_argument("--norm", type=str, default="layer_norm")
    p.add_argument("--num_heads", type=int, default=4)

    # Misc
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--cache_dir",
        type=str,
        default=os.path.expanduser("~/.cache/relbench_examples"),
    )
    p.add_argument(
        "--save_path",
        type=str,
        default="checkpoints/f1_simplified_driver-top3_relgnn.pth",
    )
    return p.parse_args()


@torch.no_grad()
def predict_proba(
    model: torch.nn.Module,
    loader: NeighborLoader,
    task: EntityTask,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    pred_list: list[Tensor] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = batch.to(device)
        logits = model(batch, task.entity_table)  # [batch, 1]
        probs = torch.sigmoid(logits).view(-1)
        pred_list.append(probs.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    # Load dataset and task.
    dataset = SimplifiedF1Dataset(cache_dir=None)
    # task = DriverTop3TaskSimplifiedSchema(dataset)
    task = SimplifiedDriverDNF(dataset)

    assert task.task_type == TaskType.BINARY_CLASSIFICATION, (
        f"This script currently assumes binary classification, got {task.task_type}."
    )

    # Build DB/graph from your simplified DB.
    # Your make_db() uses the full original DB (upto_test_timestamp=False), so we do the same here.
    db = dataset.get_db(upto_test_timestamp=False)

    # stype inference + caching (same pattern as RelGNN/RelBench examples)
    stypes_cache_path = Path(f"{args.cache_dir}/f1_simplified/stypes.json")
    try:
        with open(stypes_cache_path, "r") as f:
            col_to_stype_dict = json.load(f)
        for table, col_to_stype in col_to_stype_dict.items():
            for col, stype_str in col_to_stype.items():
                col_to_stype[col] = stype(stype_str)

        missing_tables = set(db.table_dict.keys()) - set(col_to_stype_dict.keys())
        if missing_tables:
            col_to_stype_dict = get_stype_proposal(db)
            stypes_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stypes_cache_path, "w") as f:
                json.dump(col_to_stype_dict, f, indent=2, default=str)
    except FileNotFoundError:
        col_to_stype_dict = get_stype_proposal(db)
        stypes_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stypes_cache_path, "w") as f:
            json.dump(col_to_stype_dict, f, indent=2, default=str)

    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=device), batch_size=256
        ),
        cache_dir=f"{args.cache_dir}/f1_simplified/materialized",
    )

    # Neighbor loaders for train/val/test using YOUR task tables.
    loader_dict: Dict[str, NeighborLoader] = {}
    for split in ["train", "val", "test"]:
        table = task.get_table(split)
        table_input = get_node_train_table_input(table=table, task=task)

        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=[
                int(args.num_neighbors / (2**i)) for i in range(args.num_layers)
            ],
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            subgraph_type=args.subgraph_type,
            batch_size=args.batch_size,
            temporal_strategy=args.temporal_strategy,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )

    atomic_routes = get_atomic_routes(data.edge_types)

    model = RelGNN_Model(
        data=data,
        col_stats_dict=col_stats_dict,
        out_channels=1,
        num_model_layers=args.num_model_layers,
        channels=args.channels,
        aggr=args.aggr,
        norm=args.norm,
        num_heads=args.num_heads,
        atomic_routes=atomic_routes,
        shallow_list=[],
        id_awareness=False,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = BCEWithLogitsLoss()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = -float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch in tqdm(loader_dict["train"], desc=f"epoch {epoch}/{args.epochs}"):
            batch = batch.to(device)

            # RelBench node-table transform places labels at:
            # batch[task.entity_table].y for seed nodes
            y = batch[task.entity_table].y.float().view(-1, 1)

            logits = model(batch, task.entity_table)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss)
            steps += 1
            if steps >= args.max_steps_per_epoch:
                break

        # Evaluate on val
        val_pred = predict_proba(model, loader_dict["val"], task, device)
        val_metrics = task.evaluate(
            val_pred, target_table=task.get_table("val", mask_input_cols=False)
        )
        # DriverTop3Task uses "roc_auc" in RelBench; keep this generic:
        val_score = float(val_metrics.get("roc_auc", list(val_metrics.values())[0]))

        improved = val_score > best_val
        if improved:
            best_val = val_score
            torch.save(model.state_dict(), save_path)

        print(
            f"epoch={epoch} train_loss={total_loss / max(steps, 1):.4f} "
            f"val_score={val_score:.6f} best={best_val:.6f} saved={improved}"
        )

    # Final test with best checkpoint
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_pred = predict_proba(model, loader_dict["test"], task, device)
    test_table = task.get_table("test", mask_input_cols=False)
    test_metrics = task.evaluate(test_pred, target_table=test_table)
    print(f"Test metrics: {test_metrics}")

    ground_truth = test_table.df[task.target_col].to_numpy()
    pred_proba = np.stack([1.0 - test_pred, test_pred], axis=1)
    print_binary_classification_metrics(ground_truth, pred_proba)


if __name__ == "__main__":
    main()
