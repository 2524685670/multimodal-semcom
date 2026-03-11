"""
核心训练脚本
阶段一训练脚本：对齐训练
训练 SemanticUnitAggregator + AlignmentModule
CLIPFeatureExtractor 全程冻结

用法:
    python pipeline/train_stage1.py [--config configs/default.yaml] [--use_dummy]
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import argparse
import yaml
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# 将项目根目录加入 path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.dataset import get_dataloader
from models.feature_extractor import CLIPFeatureExtractor
from models.semantic_unit import SemanticUnitAggregator
from models.alignment import AlignmentModule
from models.task_head import RetrievalTaskHead
from utils.visualization import plot_similarity_matrix, plot_training_curves

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def train_one_epoch(
    feature_extractor: CLIPFeatureExtractor,
    aggregator: SemanticUnitAggregator,
    alignment: AlignmentModule,
    dataloader,
    optimizer,
    loss_weights: dict,
    device: torch.device,
    epoch: int,
) -> dict:
    """训练一个 epoch，返回平均 loss 字典"""
    aggregator.train()
    alignment.train()

    total_loss = 0.0
    total_global = 0.0
    total_unit = 0.0
    total_div = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        text_ids = batch["text_input_ids"].to(device)
        text_mask = batch["text_attention_mask"].to(device)

        # 特征提取 (冻结, 无梯度)
        with torch.no_grad():
            features = feature_extractor(images, text_ids, text_mask)

        # 语义单元聚合
        units = aggregator(
            features["visual_tokens"],
            features["text_tokens"],
            text_attention_mask=text_mask,
        )

        # 对齐损失
        align_out = alignment(
            units["visual_units"],
            units["text_units"],
            features["visual_cls"],
            features["text_cls"],
        )

        # 加权总损失
        loss = (
            loss_weights["global_align"] * align_out["loss_global"]
            + loss_weights["unit_align"] * align_out["loss_unit"]
            + loss_weights["diversity"] * align_out["loss_diversity"]
        )

        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        nn.utils.clip_grad_norm_(
            list(aggregator.parameters()) + list(alignment.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        total_loss += loss.item()
        total_global += align_out["loss_global"].item()
        total_unit += align_out["loss_unit"].item()
        total_div += align_out["loss_diversity"].item()
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "global": f"{align_out['loss_global'].item():.4f}",
            "unit": f"{align_out['loss_unit'].item():.4f}",
        })

    return {
        "total_loss": total_loss / max(num_batches, 1),
        "loss_global": total_global / max(num_batches, 1),
        "loss_unit": total_unit / max(num_batches, 1),
        "loss_diversity": total_div / max(num_batches, 1),
    }

@torch.no_grad()
def validate(
    feature_extractor: CLIPFeatureExtractor,
    aggregator: SemanticUnitAggregator,
    alignment: AlignmentModule,
    dataloader,
    device: torch.device,
) -> dict:
    """在验证集上计算检索指标 + 收集可视化数据"""
    aggregator.eval()
    alignment.eval()

    all_v_global = []
    all_t_global = []
    sample_sim_matrices = []    # 保存前 5 个样本的 sim_matrix 用于可视化

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        images = batch["image"].to(device)
        text_ids = batch["text_input_ids"].to(device)
        text_mask = batch["text_attention_mask"].to(device)

        features = feature_extractor(images, text_ids, text_mask)
        units = aggregator(
            features["visual_tokens"],
            features["text_tokens"],
            text_attention_mask=text_mask,
        )
        align_out = alignment(
            units["visual_units"],
            units["text_units"],
            features["visual_cls"],
            features["text_cls"],
        )

        all_v_global.append(align_out["v_global"].cpu())
        all_t_global.append(align_out["t_global"].cpu())

        if len(sample_sim_matrices) < 5:
            # 保存 batch 中前几个样本的 sim_matrix
            for i in range(min(align_out["sim_matrix"].size(0),
                               5 - len(sample_sim_matrices))):
                sample_sim_matrices.append(
                    align_out["sim_matrix"][i].cpu()
                )

    all_v_global = torch.cat(all_v_global, dim=0)   # [N, proj_dim]
    all_t_global = torch.cat(all_t_global, dim=0)

    # 检索评估
    retrieval_results = RetrievalTaskHead.evaluate_retrieval(
        all_v_global, all_t_global
    )

    return {
        "retrieval": retrieval_results,
        "sim_matrices": sample_sim_matrices,
    }

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Alignment Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--use_dummy", action="store_true",
                        help="使用 DummyDataset 跑通逻辑")
    parser.add_argument("--device", type=str, default=None,
                        help="指定设备，默认自动检测")
    args = parser.parse_args()

    # ---- 配置 ----
    config = load_config(args.config)
    model_cfg = config["model"]
    train_cfg = config["training"]["stage1"]
    output_cfg = config["output"]

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[Train] 使用设备: {device}")

    # ---- 目录创建 ----
    os.makedirs(output_cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(output_cfg["log_dir"], exist_ok=True)
    os.makedirs(output_cfg["result_dir"], exist_ok=True)

    # ---- 数据 ----
    use_dummy = args.use_dummy or not os.path.isdir(config["data"]["data_root"])
    if use_dummy:
        print("[Train] 使用 DummyDataset")

    train_loader = get_dataloader(config, split="train", use_dummy=use_dummy)
    val_loader = get_dataloader(config, split="val", use_dummy=use_dummy, shuffle=False)
    print(f"[Train] 训练集大小: {len(train_loader.dataset)}, "
          f"验证集大小: {len(val_loader.dataset)}")

    # ---- 模型 ----
    feature_extractor = CLIPFeatureExtractor(model_cfg["clip_model"]).to(device)

    aggregator = SemanticUnitAggregator(
        visual_input_dim=model_cfg["visual_hidden_dim"],
        text_input_dim=model_cfg["text_hidden_dim"],
        unit_dim=model_cfg["unit_dim"],
        num_visual_units=model_cfg["num_visual_units"],
        num_text_units=model_cfg["num_text_units"],
        num_layers=model_cfg["num_attention_layers"],
        num_heads=model_cfg["num_attention_heads"],
    ).to(device)

    alignment_module = AlignmentModule(
        unit_dim=model_cfg["unit_dim"],
        proj_dim=model_cfg["proj_dim"],
        visual_hidden_dim=model_cfg["visual_hidden_dim"],
        text_hidden_dim=model_cfg["text_hidden_dim"],
    ).to(device)

    # ---- 优化器 ----
    trainable_params = (
        list(aggregator.parameters()) + list(alignment_module.parameters())
    )
    optimizer = AdamW(
        trainable_params,
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"], eta_min=1e-6)

    # ---- 训练日志 ----
    log = {
        "total_loss": [],
        "loss_global": [],
        "loss_unit": [],
        "loss_diversity": [],
        "val_i2t_r1": [],
        "val_t2i_r1": [],
        "val_rsum": [],
    }

    best_rsum = 0.0

    # ---- 训练循环 ----
    print(f"\n{'='*60}")
    print(f" 阶段一训练开始 | Epochs: {train_cfg['epochs']}")
    print(f"{'='*60}\n")

    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time.time()

        # 训练
        train_metrics = train_one_epoch(
            feature_extractor, aggregator, alignment_module,
            train_loader, optimizer, train_cfg["loss_weights"],
            device, epoch,
        )

        # 验证
        val_results = validate(
            feature_extractor, aggregator, alignment_module,
            val_loader, device,
        )
        retrieval = val_results["retrieval"]

        scheduler.step()

        # 记录日志
        log["total_loss"].append(train_metrics["total_loss"])
        log["loss_global"].append(train_metrics["loss_global"])
        log["loss_unit"].append(train_metrics["loss_unit"])
        log["loss_diversity"].append(train_metrics["loss_diversity"])
        log["val_i2t_r1"].append(retrieval["i2t_r1"])
        log["val_t2i_r1"].append(retrieval["t2i_r1"])
        log["val_rsum"].append(retrieval["rsum"])

        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{train_cfg['epochs']} | "
            f"Loss: {train_metrics['total_loss']:.4f} "
            f"(G:{train_metrics['loss_global']:.4f} "
            f"U:{train_metrics['loss_unit']:.4f} "
            f"D:{train_metrics['loss_diversity']:.4f}) | "
            f"I2T-R@1: {retrieval['i2t_r1']:.1f}% "
            f"T2I-R@1: {retrieval['t2i_r1']:.1f}% "
            f"RSum: {retrieval['rsum']:.1f} | "
            f"Time: {elapsed:.1f}s"
        )

        # 保存最优模型
        if retrieval["rsum"] > best_rsum:
            best_rsum = retrieval["rsum"]
            ckpt = {
                "epoch": epoch,
                "aggregator": aggregator.state_dict(),
                "alignment": alignment_module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rsum": best_rsum,
                "config": config,
            }
            save_path = os.path.join(output_cfg["checkpoint_dir"], "stage1_best.pt")
            torch.save(ckpt, save_path)
            print(f"  -> 最优模型已保存 (RSum={best_rsum:.1f})")

        # 每 5 个 epoch 保存一次热力图
        if epoch % 5 == 0 or epoch == 1:
            for i, sim_mat in enumerate(val_results["sim_matrices"][:3]):
                plot_similarity_matrix(
                    sim_mat,
                    save_path=os.path.join(
                        output_cfg["result_dir"],
                        f"sim_matrix_epoch{epoch}_sample{i}.png"
                    ),
                    title=f"Epoch {epoch} - Sample {i}",
                )

    # ---- 训练结束 ----
    print(f"\n{'='*60}")
    print(f" 阶段一训练完成 | 最优 RSum: {best_rsum:.1f}")
    print(f"{'='*60}")

    # 保存最终模型
    final_ckpt = {
        "epoch": train_cfg["epochs"],
        "aggregator": aggregator.state_dict(),
        "alignment": alignment_module.state_dict(),
        "config": config,
    }
    torch.save(
        final_ckpt,
        os.path.join(output_cfg["checkpoint_dir"], "stage1_final.pt"),
    )

    # 保存训练曲线
    plot_training_curves(
        log,
        save_path=os.path.join(output_cfg["result_dir"], "stage1_training_curves.png"),
        title="Stage 1: Alignment Training",
    )

    print("[Train] 所有文件已保存。")

if __name__ == "__main__":
    main()