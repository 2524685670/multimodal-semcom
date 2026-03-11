"""
阶段一验收测试
测试内容:
1. 配置加载
2. 数据集创建
3. 模型前向传播 shape 正确 + loss 有限
4. 梯度流正确（CLIP冻结，aggregator/alignment有梯度）
5. (训练后) 检索指标非随机 + 相似度矩阵有结构

用法:
    python tests/test_stage1.py [--config configs/default.yaml] [--trained]
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import argparse
import yaml
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def test_config(config_path: str):
    """测试 0: 配置文件加载"""
    print("=" * 50)
    print("测试 0: 配置文件加载")
    config = load_config(config_path)
    assert "data" in config, "缺少 data 配置"
    assert "model" in config, "缺少 model 配置"
    assert "training" in config, "缺少 training 配置"
    assert "output" in config, "缺少 output 配置"
    print("  ✅ 配置文件加载成功")
    return config

def test_dataset(config: dict):
    """测试 1: DummyDataset 创建与 DataLoader 迭代"""
    print("=" * 50)
    print("测试 1: DummyDataset")
    from data.dataset import get_dataloader

    loader = get_dataloader(config, split="train", use_dummy=True)
    batch = next(iter(loader))

    B = config["training"]["stage1"]["batch_size"]
    img_size = config["data"]["image_size"]
    max_len = config["data"].get("max_text_length", 77)

    assert batch["image"].shape == (B, 3, img_size, img_size), \
        f"image shape 错误: {batch['image'].shape}"
    assert batch["text_input_ids"].shape == (B, max_len), \
        f"text_input_ids shape 错误: {batch['text_input_ids'].shape}"
    assert batch["text_attention_mask"].shape == (B, max_len), \
        f"text_attention_mask shape 错误: {batch['text_attention_mask'].shape}"

    print(f"  ✅ DataLoader 迭代正常, batch shapes 正确")
    return batch

def test_forward(config: dict, device: torch.device):
    """测试 2: 完整前向传播"""
    print("=" * 50)
    print("测试 2: 完整前向传播 (shape + loss 有限性)")

    from data.dataset import DummyDataset
    from models.feature_extractor import CLIPFeatureExtractor
    from models.semantic_unit import SemanticUnitAggregator
    from models.alignment import AlignmentModule

    model_cfg = config["model"]
    B = 4  # 小 batch 用于测试

    # 构造小 batch
    dataset = DummyDataset(size=B, image_size=config["data"]["image_size"])
    images = torch.stack([dataset[i]["image"] for i in range(B)]).to(device)
    text_ids = torch.stack([dataset[i]["text_input_ids"] for i in range(B)]).to(device)
    text_mask = torch.stack([dataset[i]["text_attention_mask"] for i in range(B)]).to(device)

    # 手动构造 batch (因为 DummyDataset 返回单样本)
    images = torch.stack([dataset[i]["image"] for i in range(B)]).to(device)
    text_ids = torch.stack([dataset[i]["text_input_ids"] for i in range(B)]).to(device)
    text_mask = torch.stack([dataset[i]["text_attention_mask"] for i in range(B)]).to(device)

    # 初始化模型
    fe = CLIPFeatureExtractor(model_cfg["clip_model"]).to(device)
    agg = SemanticUnitAggregator(
        visual_input_dim=model_cfg["visual_hidden_dim"],
        text_input_dim=model_cfg["text_hidden_dim"],
        unit_dim=model_cfg["unit_dim"],
        num_visual_units=model_cfg["num_visual_units"],
        num_text_units=model_cfg["num_text_units"],
        num_layers=model_cfg["num_attention_layers"],
        num_heads=model_cfg["num_attention_heads"],
    ).to(device)
    align = AlignmentModule(
        unit_dim=model_cfg["unit_dim"],
        proj_dim=model_cfg["proj_dim"],
        visual_hidden_dim=model_cfg["visual_hidden_dim"],
        text_hidden_dim=model_cfg["text_hidden_dim"],
    ).to(device)

    # 前向传播
    with torch.no_grad():
        features = fe(images, text_ids, text_mask)

    Nv = model_cfg["num_visual_units"]
    Nt = model_cfg["num_text_units"]
    proj_dim = model_cfg["proj_dim"]

    # 检查特征提取输出
    assert features["visual_tokens"].shape == (B, 49, model_cfg["visual_hidden_dim"]), \
        f"visual_tokens shape 错误: {features['visual_tokens'].shape}"
    assert features["text_tokens"].shape[0] == B and \
           features["text_tokens"].shape[2] == model_cfg["text_hidden_dim"], \
        f"text_tokens shape 错误: {features['text_tokens'].shape}"
    print("  ✅ 特征提取 shape 正确")

    # 检查聚合输出
    units = agg(features["visual_tokens"], features["text_tokens"],
                text_attention_mask=text_mask)
    assert units["visual_units"].shape == (B, Nv, model_cfg["unit_dim"]), \
        f"visual_units shape 错误: {units['visual_units'].shape}"
    assert units["text_units"].shape == (B, Nt, model_cfg["unit_dim"]), \
        f"text_units shape 错误: {units['text_units'].shape}"
    print("  ✅ 语义单元聚合 shape 正确")

    # 检查对齐输出
    align_out = align(
        units["visual_units"], units["text_units"],
        features["visual_cls"], features["text_cls"],
    )
    assert align_out["v_proj"].shape == (B, Nv, proj_dim)
    assert align_out["t_proj"].shape == (B, Nt, proj_dim)
    assert align_out["v_global"].shape == (B, proj_dim)
    assert align_out["t_global"].shape == (B, proj_dim)
    assert align_out["sim_matrix"].shape == (B, Nv, Nt)
    print("  ✅ 对齐模块输出 shape 正确")

    # 检查 loss 有限
    for loss_name in ["loss_global", "loss_unit", "loss_diversity"]:
        loss_val = align_out[loss_name]
        assert torch.isfinite(loss_val), f"{loss_name} 不是有限数: {loss_val}"
    print(f"  ✅ 所有损失值有限: "
          f"global={align_out['loss_global'].item():.4f}, "
          f"unit={align_out['loss_unit'].item():.4f}, "
          f"div={align_out['loss_diversity'].item():.4f}")

    return fe, agg, align

def test_gradient_flow(config: dict, device: torch.device):
    """测试 3: 梯度流检查"""
    print("=" * 50)
    print("测试 3: 梯度流检查")

    from data.dataset import DummyDataset
    from models.feature_extractor import CLIPFeatureExtractor
    from models.semantic_unit import SemanticUnitAggregator
    from models.alignment import AlignmentModule

    model_cfg = config["model"]
    B = 4

    dataset = DummyDataset(size=B, image_size=config["data"]["image_size"])
    images = torch.stack([dataset[i]["image"] for i in range(B)]).to(device)
    text_ids = torch.stack([dataset[i]["text_input_ids"] for i in range(B)]).to(device)
    text_mask = torch.stack([dataset[i]["text_attention_mask"] for i in range(B)]).to(device)

    fe = CLIPFeatureExtractor(model_cfg["clip_model"]).to(device)
    agg = SemanticUnitAggregator(
        visual_input_dim=model_cfg["visual_hidden_dim"],
        text_input_dim=model_cfg["text_hidden_dim"],
        unit_dim=model_cfg["unit_dim"],
        num_visual_units=model_cfg["num_visual_units"],
        num_text_units=model_cfg["num_text_units"],
        num_layers=model_cfg["num_attention_layers"],
        num_heads=model_cfg["num_attention_heads"],
    ).to(device)
    align = AlignmentModule(
        unit_dim=model_cfg["unit_dim"],
        proj_dim=model_cfg["proj_dim"],
        visual_hidden_dim=model_cfg["visual_hidden_dim"],
        text_hidden_dim=model_cfg["text_hidden_dim"],
    ).to(device)

    # 前向 + 反向
    with torch.no_grad():
        features = fe(images, text_ids, text_mask)

    units = agg(features["visual_tokens"], features["text_tokens"],
                text_attention_mask=text_mask)
    align_out = align(units["visual_units"], units["text_units"],
                      features["visual_cls"], features["text_cls"])

    loss = align_out["loss_global"] + align_out["loss_unit"] + align_out["loss_diversity"]
    loss.backward()

    # CLIP 冻结检查
    clip_has_grad = False
    for p in fe.parameters():
        if p.grad is not None:
            clip_has_grad = True
            break
    assert not clip_has_grad, "CLIP 参数不应该有梯度！"
    print("  ✅ CLIP 特征提取器参数无梯度（正确冻结）")

    # Aggregator 梯度检查
    agg_has_grad = False
    for name, p in agg.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            agg_has_grad = True
            break
    assert agg_has_grad, "Aggregator 参数应该有梯度！"
    print("  ✅ Aggregator 参数有梯度（正确训练）")

    # Alignment 梯度检查
    align_has_grad = False
    for name, p in align.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            align_has_grad = True
            break
    assert align_has_grad, "AlignmentModule 参数应该有梯度！"
    print("  ✅ AlignmentModule 参数有梯度（正确训练）")

def test_trained_model(config: dict, device: torch.device):
    """测试 4: (训练后) 加载 checkpoint 并验证"""
    print("=" * 50)
    print("测试 4: 训练后模型验证")

    checkpoint_path = os.path.join(config["output"]["checkpoint_dir"], "stage1_best.pt")
    if not os.path.exists(checkpoint_path):
        print(f"  ⚠️  未找到 checkpoint: {checkpoint_path}")
        print("  ⚠️  跳过训练后测试（请先运行 train_stage1.py）")
        return

    from data.dataset import get_dataloader
    from models.feature_extractor import CLIPFeatureExtractor
    from models.semantic_unit import SemanticUnitAggregator
    from models.alignment import AlignmentModule
    from models.task_head import RetrievalTaskHead

    model_cfg = config["model"]

    # 加载 checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"  加载 checkpoint: epoch={ckpt['epoch']}, rsum={ckpt['rsum']:.1f}")

    fe = CLIPFeatureExtractor(model_cfg["clip_model"]).to(device)
    agg = SemanticUnitAggregator(
        visual_input_dim=model_cfg["visual_hidden_dim"],
        text_input_dim=model_cfg["text_hidden_dim"],
        unit_dim=model_cfg["unit_dim"],
        num_visual_units=model_cfg["num_visual_units"],
        num_text_units=model_cfg["num_text_units"],
        num_layers=model_cfg["num_attention_layers"],
        num_heads=model_cfg["num_attention_heads"],
    ).to(device)
    align = AlignmentModule(
        unit_dim=model_cfg["unit_dim"],
        proj_dim=model_cfg["proj_dim"],
        visual_hidden_dim=model_cfg["visual_hidden_dim"],
        text_hidden_dim=model_cfg["text_hidden_dim"],
    ).to(device)

    agg.load_state_dict(ckpt["aggregator"])
    align.load_state_dict(ckpt["alignment"])
    agg.eval()
    align.eval()

    # 在验证集上测试
    use_dummy = not os.path.isdir(config["data"]["data_root"])
    val_loader = get_dataloader(config, split="val", use_dummy=use_dummy, shuffle=False)

    all_v_global = []
    all_t_global = []
    all_sim_matrices = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            text_ids = batch["text_input_ids"].to(device)
            text_mask = batch["text_attention_mask"].to(device)

            features = fe(images, text_ids, text_mask)
            units = agg(features["visual_tokens"], features["text_tokens"],
                        text_attention_mask=text_mask)
            align_out = align(units["visual_units"], units["text_units"],
                              features["visual_cls"], features["text_cls"])

            all_v_global.append(align_out["v_global"].cpu())
            all_t_global.append(align_out["t_global"].cpu())
            all_sim_matrices.append(align_out["sim_matrix"].cpu())

    all_v_global = torch.cat(all_v_global)
    all_t_global = torch.cat(all_t_global)

    # 检索指标
    results = RetrievalTaskHead.evaluate_retrieval(all_v_global, all_t_global)
    print(f"  检索指标:")
    print(f"    I2T R@1={results['i2t_r1']:.1f}% R@5={results['i2t_r5']:.1f}% "
          f"R@10={results['i2t_r10']:.1f}%")
    print(f"    T2I R@1={results['t2i_r1']:.1f}% R@5={results['t2i_r5']:.1f}% "
          f"R@10={results['t2i_r10']:.1f}%")
    print(f"    RSum={results['rsum']:.1f}")

    # 宽松验收：R@10 > 0 即可 (不要求高指标)
    assert results["i2t_r10"] > 0 or results["t2i_r10"] > 0, \
        "R@10 应该 > 0（检索结果不应完全随机）"
    print("  ✅ 检索指标非零（通过）")

    # 相似度矩阵多样性检查
    first_sim = all_sim_matrices[0][0]  # 第一个样本
    sim_std = first_sim.std().item()
    print(f"  相似度矩阵 std={sim_std:.4f}")
    assert sim_std > 0.01, f"相似度矩阵 std 过小({sim_std:.4f})，可能塌缩"
    print("  ✅ 相似度矩阵有结构（未塌缩）")

def main():
    parser = argparse.ArgumentParser(description="Stage 1 Tests")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--trained", action="store_true",
                        help="是否运行训练后测试")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("\n" + "=" * 60)
    print(" 阶段一验收测试")
    print("=" * 60)

    passed = 0
    total = 0

    try:
        # 测试 0: 配置
        total += 1
        config = test_config(args.config)
        passed += 1
    except Exception as e:
        print(f"  ❌ 测试 0 失败: {e}")
        return

    try:
        # 测试 1: 数据
        total += 1
        test_dataset(config)
        passed += 1
    except Exception as e:
        print(f"  ❌ 测试 1 失败: {e}")

    try:
        # 测试 2: 前向传播
        total += 1
        test_forward(config, device)
        passed += 1
    except Exception as e:
        print(f"  ❌ 测试 2 失败: {e}")
        import traceback; traceback.print_exc()

    try:
        # 测试 3: 梯度流
        total += 1
        test_gradient_flow(config, device)
        passed += 1
    except Exception as e:
        print(f"  ❌ 测试 3 失败: {e}")
        import traceback; traceback.print_exc()

    if args.trained:
        try:
            # 测试 4: 训练后验证
            total += 1
            test_trained_model(config, device)
            passed += 1
        except Exception as e:
            print(f"  ❌ 测试 4 失败: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "=" * 60)
    if passed == total:
        print(f" ✅ Stage 1 PASSED ({passed}/{total})")
    else:
        print(f" ❌ Stage 1 FAILED ({passed}/{total} passed)")
    print("=" * 60)

if __name__ == "__main__":
    main()