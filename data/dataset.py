"""
数据集加载模块
包含:
- DummyDataset: 随机张量模拟数据，用于逻辑验证
- Flickr30kDataset: 真实数据加载 (数据可用时使用)
- get_dataloader: DataLoader 工厂函数
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Tuple

class DummyDataset(Dataset):
    """
    随机张量数据集，用于在真实数据不可用时跑通全链路逻辑。
    模拟 Flickr30k 的数据格式：
    - image: [3, 224, 224] 随机像素值
    - text_input_ids: [77] 随机 token id
    - text_attention_mask: [77] 前 random_len 个为1，其余为0
    """

    def __init__(self, size: int = 1000, image_size: int = 224,
                 max_text_length: int = 77, seed: int = 42):
        super().__init__()
        self.size = size
        self.image_size = image_size
        self.max_text_length = max_text_length

        # 固定随机种子保证可复现
        gen = torch.Generator().manual_seed(seed)

        # 预生成所有数据 (小规模，可全部放内存)
        self.images = torch.randn(size, 3, image_size, image_size, generator=gen)
        self.text_input_ids = torch.randint(
            0, 49408, (size, max_text_length), generator=gen
        )  # CLIP vocab size = 49408

        # 随机有效文本长度 (5~max_text_length)
        self.text_lengths = torch.randint(
            5, max_text_length, (size,), generator=gen
        )
        self.text_attention_masks = torch.zeros(size, max_text_length, dtype=torch.long)
        for i in range(size):
            self.text_attention_masks[i, : self.text_lengths[i]] = 1

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "image": self.images[idx],                           # [3, 224, 224]
            "text_input_ids": self.text_input_ids[idx],          # [77]
            "text_attention_mask": self.text_attention_masks[idx],# [77]
            "image_id": idx,
            "caption": f"dummy caption {idx}",
        }

class Flickr30kDataset(Dataset):
    """
    Flickr30k 数据集加载器
    预期目录结构:
      data_root/
        images/           # 所有图片
        results_20130124.token  # caption 文件 (每图5条)
    
    MVP 阶段每张图只取第 1 条 caption。
    """

    def __init__(self, data_root: str, split: str = "train",
                 train_size: int = 5000, val_size: int = 500,
                 image_size: int = 224, max_text_length: int = 77,
                 clip_model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.max_text_length = max_text_length

        # 尝试加载 CLIP processor
        try:
            from transformers import CLIPProcessor
            self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        except Exception as e:
            print(f"[WARN] 无法加载 CLIPProcessor: {e}, 将使用基础预处理")
            self.processor = None

        # 加载 caption 并按 image_id 排序
        self.samples = self._load_annotations()

        # 划分 train / val
        if split == "train":
            self.samples = self.samples[:train_size]
        elif split == "val":
            self.samples = self.samples[train_size: train_size + val_size]
        else:
            raise ValueError(f"split 必须为 'train' 或 'val', 收到 '{split}'")

        print(f"[Flickr30k] split={split}, 样本数={len(self.samples)}")

    def _load_annotations(self):
        """解析 Flickr30k caption 文件，每张图取第 1 条 caption"""
        caption_file = os.path.join(self.data_root, "results_20130124.token")
        image_dir = os.path.join(self.data_root, "images")

        samples = []
        seen_images = set()

        if not os.path.exists(caption_file):
            print(f"[WARN] Caption 文件不存在: {caption_file}")
            return samples

        with open(caption_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                img_caption_id = parts[0]       # e.g. "1000092795.jpg#0"
                caption = parts[1]
                img_name = img_caption_id.split("#")[0]
                caption_idx = int(img_caption_id.split("#")[1])

                # 只取第 1 条 caption
                if caption_idx != 0:
                    continue
                if img_name in seen_images:
                    continue
                seen_images.add(img_name)

                img_path = os.path.join(image_dir, img_name)
                if os.path.exists(img_path):
                    samples.append({
                        "image_path": img_path,
                        "caption": caption,
                        "image_id": len(samples),
                    })

        # 按 image_id 排序，保证划分一致
        samples.sort(key=lambda x: x["image_id"])
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        from PIL import Image
        image = Image.open(sample["image_path"]).convert("RGB")
        caption = sample["caption"]

        if self.processor is not None:
            encoded = self.processor(
                text=[caption],
                images=image,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_text_length,
            )
            return {
                "image": encoded["pixel_values"].squeeze(0),        # [3, 224, 224]
                "text_input_ids": encoded["input_ids"].squeeze(0),  # [77]
                "text_attention_mask": encoded["attention_mask"].squeeze(0),
                "image_id": sample["image_id"],
                "caption": caption,
            }
        else:
            # 基础预处理回退方案
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                ),
            ])
            image_tensor = transform(image)
            # 简单分词 (不推荐，仅兜底)
            text_ids = torch.zeros(self.max_text_length, dtype=torch.long)
            text_mask = torch.zeros(self.max_text_length, dtype=torch.long)
            tokens = caption.split()[:self.max_text_length - 2]
            text_mask[: len(tokens) + 2] = 1
            return {
                "image": image_tensor,
                "text_input_ids": text_ids,
                "text_attention_mask": text_mask,
                "image_id": sample["image_id"],
                "caption": caption,
            }

def get_dataloader(
    config: dict,
    split: str = "train",
    use_dummy: bool = False,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """
    DataLoader 工厂函数
    
    Args:
        config: 完整配置字典
        split: "train" 或 "val"
        use_dummy: 是否使用 DummyDataset
        shuffle: 是否打乱，默认 train=True, val=False
    
    Returns:
        DataLoader 实例
    """
    data_cfg = config["data"]
    train_cfg = config["training"]["stage1"]

    if shuffle is None:
        shuffle = (split == "train")

    if use_dummy:
        dataset = DummyDataset(
            size=data_cfg["train_size"] if split == "train" else data_cfg["val_size"],
            image_size=data_cfg["image_size"],
            max_text_length=data_cfg.get("max_text_length", 77),
        )
    else:
        data_root = data_cfg["data_root"]
        if not os.path.isdir(data_root):
            print(f"[WARN] 数据目录 {data_root} 不存在，回退到 DummyDataset")
            dataset = DummyDataset(
                size=data_cfg["train_size"] if split == "train" else data_cfg["val_size"],
                image_size=data_cfg["image_size"],
                max_text_length=data_cfg.get("max_text_length", 77),
            )
        else:
            dataset = Flickr30kDataset(
                data_root=data_root,
                split=split,
                train_size=data_cfg["train_size"],
                val_size=data_cfg["val_size"],
                image_size=data_cfg["image_size"],
                max_text_length=data_cfg.get("max_text_length", 77),
                clip_model_name=config["model"]["clip_model"],
            )

    return DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=shuffle,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=(split == "train"),
    )