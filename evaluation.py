import argparse
import os
import random
import math
import numpy as np
import albumentations as A
import cv2
import timm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from albumentations.pytorch import ToTensorV2
from pathlib import Path
from timm.data import resolve_model_data_config
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List
from PIL import Image
from train import (
    CustomImageFolder,
    drop_broken_samples,
    filter_dataset_to_classnames,
    build_family_mapping,
    build_model,
    CutMax,
    ResizeWithPad,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def evaluate_model(model, dataloader, criterion, dataset_size, family_ids=None):
    model.eval()
    running_loss = 0.0
    running_corrects1 = 0
    running_corrects5 = 0
    running_family_corrects1 = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds1 = torch.max(outputs, 1)
                _, top5 = torch.topk(outputs, k=min(5, outputs.shape[1]), dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects1 += torch.sum(preds1 == labels)
            running_corrects5 += torch.sum(top5.eq(labels.view(-1, 1)).any(dim=1))
            if family_ids is not None:
                family_preds1 = family_ids[preds1]
                family_labels = family_ids[labels]
                running_family_corrects1 += torch.sum(family_preds1 == family_labels)

    denom = max(1, dataset_size)
    epoch_loss = running_loss / denom
    epoch_acc1 = running_corrects1.double() / denom
    epoch_acc5 = running_corrects5.double() / denom
    epoch_family_acc1 = (
        running_family_corrects1.double() / denom if family_ids is not None else 0.0
    )
    return (
        float(epoch_loss),
        float(epoch_acc1),
        float(epoch_acc5),
        float(epoch_family_acc1),
    )


# ---------- main ----------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation-only script")
    parser.add_argument("--model_folder", type=str, required=True, help="Folder containing class_names.txt and checkpoint")
    parser.add_argument("--data_folder", type=str, required=True, help="Evaluation dataset root (class subfolders)")
    parser.add_argument("--network_type", type=str, required=True, help="Architecture used for training")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint; defaults to <model_folder>/trained_model.pth")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args()


def main(args):
    if args.seed is not None:
        print(f"[seed] Setting seed to {args.seed} (deterministic cuDNN)")
        set_seed(args.seed)

    with open(os.path.join(args.model_folder, "class_names.txt"), "r") as f:
        class_names = f.read().splitlines()

    cfg_model = timm.create_model(args.network_type, pretrained=False, num_classes=0)
    data_cfg = resolve_model_data_config(cfg_model)
    _, h, w = data_cfg["input_size"]
    target_size = (w, h)
    norm_mean = list(data_cfg["mean"])
    norm_std = list(data_cfg["std"])

    val_transform = A.Compose(
        [
            A.Lambda(image=CutMax(1024)),
            A.Lambda(image=ResizeWithPad(target_size)),
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2(),
        ]
    )

    dataset = CustomImageFolder(args.data_folder, transform=val_transform)
    dataset.samples, bad = drop_broken_samples(dataset.samples)
    dataset.imgs = dataset.samples
    dataset.targets = [t for _, t in dataset.samples]
    if bad:
        print(f"[warn] dropped {len(bad)} broken images")
        for p, reason, shape in bad:
            print(f"  - {p} | {reason} | shape={shape}")

    dataset = filter_dataset_to_classnames(dataset, class_names)
    family_names, class_idx_to_family, missing_family_delim = build_family_mapping(dataset.classes)
    if missing_family_delim:
        print("[warn] Some class names lack '--' delimiter; using full name as family:")
        for name in missing_family_delim:
            print(f"  - {name}")
    family_ids = torch.tensor(class_idx_to_family, device=device)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Build model exactly as in training (handles DINOv3 sequential head)
    model = build_model(args.network_type, len(class_names))
    model.to(device)

    ckpt_path = args.checkpoint_path or os.path.join(args.model_folder, "trained_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    criterion = nn.CrossEntropyLoss()
    loss, acc1, acc5, fam_acc1 = evaluate_model(
        model, dataloader, criterion, len(dataset), family_ids
    )

    print(
        f"[eval] Loss: {loss:.4f} "
        f"Top1: {acc1:.4f} Top5: {acc5:.4f} "
        f"FamilyTop1: {fam_acc1:.4f} "
        f"(classes={len(class_names)}, families={len(family_names)})"
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)

