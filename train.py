import albumentations as A
import argparse
import csv
import cv2
import math
import numpy as np
import os
import random
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from PIL import Image
from albumentations.pytorch import ToTensorV2
from timm.data import resolve_model_data_config
from pathlib import Path
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from typing import Tuple, List

try:
    from timm.layers import Mlp
except Exception:
    # Older timm versions keep Mlp under timm.models.layers
    from timm.models.layers import Mlp

# Set device
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CHOICES = [
    "resnet50",
    "resnet101",
    "resnet152",
    "efficientnet_b0",
    "efficientnet_b2",
    "efficientnet_v2_s",
    "convnext_small",
    "convnext_base",
    "swin_base_patch4_window7_224",
    "vit_base_patch16_224",
    "vit_large_patch16_dinov3.lvd1689m",
    "vit_huge_plus_patch16_dinov3.lvd1689m",
    "mobilenetv3_large_100",
    "seresnext50_32x4d",
]
DINOV3_MODELS = [
    "vit_large_patch16_dinov3.lvd1689m",
    "vit_huge_plus_patch16_dinov3.lvd1689m",
]


def parse_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Training script")

    # Add arguments
    parser.add_argument(
        "--image_folder",
        type=str,
        default="sample_data/output",
        help="Path to the folder containing the images",
    )
    parser.add_argument(
        "--extra_image_folders",
        type=str,
        default=None,
        help="Comma-separated additional train folders to concatenate with image_folder",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="output",
        help="Path to the folder where the trained model will be saved",
    )
    parser.add_argument(
        "--val_folder",
        type=str,
        default=None,
        help="Optional path to a separate validation folder (skips random split)",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.01,
        help="Fraction of the dataset to be used for testing",
    )
    parser.add_argument(
        "-net",
        "--network_type",
        type=str,
        default="resnet50",
        choices=MODEL_CHOICES,
        help="Network architecture (timm). Examples: "
        + ", ".join(MODEL_CHOICES),
    )
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=0.0001, help="Learning rate"
    )
    parser.add_argument(
        "--backbone_lr_scale",
        type=float,
        default=0.2,
        help="Scale factor for backbone LR when using DINOv3 models",
    )
    parser.add_argument(
        "-e", "--num_epochs", type=int, default=100, help="Number of epochs"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for dataloader"
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=1.0,
        help="If >0, clip total grad norm to this value",
    )
    parser.add_argument(
        "--debug_grad",
        action="store_true",
        help="Print grad norms on the first train step for quick sanity checks",
    )
    parser.add_argument(
        "--top_k_fonts",
        type=int,
        default=3000,
        help="Keep only the top-k classes by image count (<=0 means keep all)",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Run evaluation on the validation set using a checkpoint and exit",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to checkpoint for eval-only mode; defaults to output/<net>/best_model_params.pt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (dataset split, torch/np/random).",
    )

    # Parse the arguments
    args = parser.parse_args()

    return args


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, **kwargs):
        super(CustomImageFolder, self).__init__(root, **kwargs)
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path).convert("RGB")

        if self.transform is not None:
            sample = np.array(sample)  # Convert PIL image to numpy array
            transformed = self.transform(image=sample)  # Apply Albumentations transform
            sample = transformed["image"]  # Extract transformed image

        return sample, target


def drop_broken_samples(samples: List[Tuple[str, int]]):
    """
    Remove images that cannot be opened or have zero width/height.
    Returns (clean_samples, bad_list).
    """
    clean = []
    bad = []
    for path, target in samples:
        try:
            with Image.open(path) as img:
                w, h = img.size
                if w == 0 or h == 0:
                    bad.append((path, "zero-dim", (w, h)))
                    continue
        except Exception as e:
            bad.append((path, str(e), None))
            continue
        clean.append((path, target))
    return clean, bad


class ResizeWithPad:

    def __init__(
        self, new_shape: Tuple[int, int], padding_color: Tuple[int] = (255, 255, 255)
    ) -> None:
        self.new_shape = new_shape
        self.padding_color = padding_color

    def __call__(self, image: np.array, **kwargs) -> np.array:
        """Maintains aspect ratio and resizes with padding.
        Params:
            image: Image to be resized.
            new_shape: Expected (width, height) of new image.
            padding_color: Tuple in BGR of padding color
        Returns:
            image: Resized image with padding
        """
        original_shape = (image.shape[1], image.shape[0])
        ratio = float(max(self.new_shape)) / max(original_shape)
        # Avoid zero-sized dims when original shapes are extremely skinny/long
        new_size = tuple([max(1, int(round(x * ratio))) for x in original_shape])
        image = cv2.resize(image, new_size)
        delta_w = self.new_shape[0] - new_size[0]
        delta_h = self.new_shape[1] - new_size[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=self.padding_color,
        )
        return image


class CutMax:
    """Cuts the image to the maximum size"""

    def __init__(self, max_size: int = 1024) -> None:
        self.max_size = max_size

    def __call__(self, image: np.array, **kwargs) -> np.array:
        """Cuts the image to the maximum size"""
        if image.shape[0] > self.max_size:
            image = image[: self.max_size, :, :]
        if image.shape[1] > self.max_size:
            image = image[:, : self.max_size, :]
        return image


def build_model(network_type: str, num_classes: int) -> nn.Module:
    """
    Create a classification model. For DINOv3 ViTs, drop the default head and
    attach a small MLP head similar to the test notebook example.
    """
    if network_type in DINOV3_MODELS:
        backbone = timm.create_model(
            network_type, pretrained=True, num_classes=0, global_pool="avg"
        )  # remove default head
        # Allow non-multiple-of-patch sizes (e.g., 518) by padding inside patch embed
        if hasattr(backbone, "patch_embed") and hasattr(
            backbone.patch_embed, "dynamic_img_pad"
        ):
            backbone.patch_embed.dynamic_img_pad = True
        in_dim = backbone.num_features
        head = Mlp(
            in_features=in_dim,
            hidden_features=1024,
            out_features=num_classes,
            drop=0.1,
        )
        model = nn.Sequential(backbone, head)
    else:
        model = timm.create_model(
            network_type, pretrained=True, num_classes=num_classes
        )
    return model


def filter_dataset_top_k(dataset: CustomImageFolder, top_k: int):
    """
    Keep only the top-k classes by image count. Reindexes class ids.
    Returns the filtered dataset and the kept class name list (in order).
    """
    if top_k is None or top_k <= 0 or top_k >= len(dataset.classes):
        return dataset, list(dataset.classes)

    old_classes = list(dataset.classes)
    counts = [0 for _ in old_classes]
    for _, target in dataset.samples:
        counts[target] += 1

    sorted_idx = sorted(
        range(len(counts)), key=lambda i: (-counts[i], old_classes[i])
    )
    keep_idx = sorted_idx[:top_k]
    keep_set = set(keep_idx)
    keep_names = [old_classes[i] for i in keep_idx]
    old_to_new = {old: new for new, old in enumerate(keep_idx)}

    new_samples = []
    new_targets = []
    for path, target in dataset.samples:
        if target in keep_set:
            new_t = old_to_new[target]
            new_samples.append((path, new_t))
            new_targets.append(new_t)

    dataset.samples = new_samples
    dataset.imgs = new_samples  # alias used by ImageFolder
    dataset.targets = new_targets
    dataset.classes = keep_names
    dataset.class_to_idx = {name: idx for idx, name in enumerate(keep_names)}
    return dataset, keep_names


def filter_dataset_to_classnames(dataset: CustomImageFolder, keep_names):
    """
    Filter dataset to an existing class name list and reindex targets to match.
    """
    if keep_names is None:
        return dataset

    keep_names = list(keep_names)
    old_classes = list(dataset.classes)
    name_to_old_idx = {name: idx for idx, name in enumerate(old_classes)}
    keep_idx = [name_to_old_idx[n] for n in keep_names if n in name_to_old_idx]
    keep_set = set(keep_idx)
    old_to_new = {name: idx for idx, name in enumerate(keep_names)}

    new_samples = []
    new_targets = []
    for path, target in dataset.samples:
        if target in keep_set:
            name = old_classes[target]
            new_t = old_to_new[name]
            new_samples.append((path, new_t))
            new_targets.append(new_t)

    dataset.samples = new_samples
    dataset.imgs = new_samples
    dataset.targets = new_targets
    dataset.classes = keep_names
    dataset.class_to_idx = {name: idx for idx, name in enumerate(keep_names)}
    return dataset


def build_family_mapping(class_names: List[str]):
    """
    Map each class index to a family index using the prefix before '--'.
    Returns (family_names, class_idx_to_family_idx, missing_delim_names).
    """
    family_to_idx = {}
    family_names = []
    class_idx_to_family = []
    missing = []
    for name in class_names:
        if "--" in name:
            family = name.split("--", 1)[0]
        else:
            family = name
            missing.append(name)
        if family not in family_to_idx:
            family_to_idx[family] = len(family_names)
            family_names.append(family)
        class_idx_to_family.append(family_to_idx[family])
    return family_names, class_idx_to_family, missing


def set_seed(seed: int):
    """
    Set all relevant seeds for reproducibility and enable deterministic cuDNN.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def compute_grad_norm(params) -> float:
    """
    L2 norm of gradients for a parameter iterable.
    """
    total = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total += param_norm.item() ** 2
    return math.sqrt(total)


def evaluate_model(model, dataloader, criterion, dataset_size, family_ids=None):
    """
    Run a single evaluation pass; returns (loss, top1_acc, top5_acc, family_top1_acc).
    """
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


def save_best_predictions(
    model,
    subset,
    raw_dataset,
    transform,
    class_names,
    output_folder,
    epoch,
    acc_top1,
    acc_top5,
    acc_family_top1=None,
    max_samples=100,
):
    """
    On best val improvement, sample up to max_samples images from the validation subset,
    run prediction, save images, and dump a CSV mapping (filename, predicted, target, src_path).
    """
    if not hasattr(subset, "indices"):
        return

    idx_list = list(subset.indices)
    if not idx_list:
        return

    k = min(max_samples, len(idx_list))
    chosen = random.sample(idx_list, k)

    family_suffix = ""
    if acc_family_top1 is not None:
        family_suffix = f"_famt1-{acc_family_top1:.4f}"

    out_dir = os.path.join(
        output_folder,
        "best_samples",
        f"epoch_{epoch}_acc_t1-{acc_top1:.4f}_t5-{acc_top5:.4f}{family_suffix}",
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "predictions.csv")

    model.eval()
    rows = []
    with torch.no_grad():
        for i, idx in enumerate(chosen):
            img, target = raw_dataset[idx]
            src_path = raw_dataset.samples[idx][0]
            arr = np.array(img)
            tensor = transform(image=arr)["image"].unsqueeze(0).to(device)
            with torch.cuda.amp.autocast():
                outputs = model(tensor)
                _, pred = torch.max(outputs, 1)
            pred_idx = int(pred.item())
            target_idx = int(target)
            pred_name = class_names[pred_idx] if pred_idx < len(class_names) else ""
            target_name = class_names[target_idx] if target_idx < len(class_names) else ""

            # Save image as PNG
            filename = f"sample_{i:04d}_pred-{pred_name}_gt-{target_name}.png"
            Image.fromarray(arr).save(os.path.join(out_dir, filename))

            rows.append(
                {
                    "filename": filename,
                    "predicted": pred_name,
                    "target": target_name,
                    "src_path": src_path,
                }
            )

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "predicted", "target", "src_path"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    if args.seed is not None:
        print(f"[seed] Setting seed to {args.seed} (deterministic cuDNN, fixed split)")
        set_seed(args.seed)

    os.makedirs(args.output_folder, exist_ok=True)
    model_output_dir = os.path.join(args.output_folder, args.network_type)
    os.makedirs(model_output_dir, exist_ok=True)

    # Access the arguments
    image_folder = args.image_folder
    val_folder = args.val_folder
    network_type = args.network_type
    is_dinov3 = network_type in DINOV3_MODELS

    # Use timm data_config for the chosen backbone (aligns with HF model card)
    cfg_model = timm.create_model(network_type, pretrained=True, num_classes=0)
    data_cfg = resolve_model_data_config(cfg_model)
    _, h, w = data_cfg["input_size"]
    target_size = (w, h)
    norm_mean = list(data_cfg["mean"])
    norm_std = list(data_cfg["std"])

    # Define a custom transform function to preprocess the images using Albumentations
    train_transform = A.Compose(
        [
            A.Lambda(image=CutMax(1024)),
            A.Lambda(image=ResizeWithPad(target_size)),  # Custom SquarePad
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=(0.8, 1.0),
                rotate_limit=10,
                interpolation=1,
                p=0.7,
            ),
            A.ColorJitter(p=0.2),
            A.ISONoise(p=0.2),
            A.ImageCompression(quality_lower=70, quality_upper=95, p=0.2),
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2(),
        ]
    )

    # Validation: no heavy augmentation, only resize/normalize
    val_transform = A.Compose(
        [
            A.Lambda(image=CutMax(1024)),
            A.Lambda(image=ResizeWithPad(target_size)),
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2(),
        ]
    )

    check_transform = A.Compose(
        [
            A.Lambda(image=CutMax(1024)),
            A.Lambda(image=ResizeWithPad(target_size)),  # Custom SquarePad
            A.ShiftScaleRotate(
                shift_limit_x=0.1,
                shift_limit_y=0.1,
                scale_limit=(0.8, 1.0),
                rotate_limit=10,
                interpolation=1,
                p=0.7,
            ),
            A.ColorJitter(p=0.2),
            A.ISONoise(p=0.2),
            A.ImageCompression(quality_lower=70, quality_upper=95, p=0.2),
        ]
    )

    best_model_params_path = os.path.join(model_output_dir, "best_model_params.pt")

    # Create dataset (train/val) and a raw copy for saving images
    def _load_clean(folder: str, transform):
        ds = CustomImageFolder(folder, transform=transform)
        ds.samples, bad = drop_broken_samples(ds.samples)
        ds.imgs = ds.samples
        ds.targets = [t for _, t in ds.samples]
        return ds, bad

    dataset, bad_train = _load_clean(image_folder, train_transform)
    raw_train_dataset, bad_raw_train = _load_clean(image_folder, None)

    # Merge extra train folders if provided
    extra_folders = []
    if args.extra_image_folders:
        extra_folders = [p for p in args.extra_image_folders.split(",") if p.strip()]

    for extra in extra_folders:
        extra_ds, bad_extra = _load_clean(extra, train_transform)
        extra_raw, bad_extra_raw = _load_clean(extra, None)
        # Ensure class mapping matches
        if extra_ds.class_to_idx != dataset.class_to_idx:
            raise ValueError(
                f"Class mapping mismatch between base folder and extra folder {extra}"
            )
        dataset.samples.extend(extra_ds.samples)
        dataset.imgs = dataset.samples
        dataset.targets = [t for _, t in dataset.samples]

        raw_train_dataset.samples.extend(extra_raw.samples)
        raw_train_dataset.imgs = raw_train_dataset.samples
        raw_train_dataset.targets = [t for _, t in raw_train_dataset.samples]

        bad_train += bad_extra
        bad_raw_train += bad_extra_raw

    if bad_train or bad_raw_train:
        dropped = len(set([p for p, _, _ in bad_train + bad_raw_train]))
        print(f"[warn] dropped {dropped} broken images (see paths below)")
        for p, reason, shape in bad_train + bad_raw_train:
            print(f"  - {p} | {reason} | shape={shape}")

    dataset, kept_class_names = filter_dataset_top_k(dataset, args.top_k_fonts)
    raw_train_dataset = filter_dataset_to_classnames(
        raw_train_dataset, kept_class_names
    )
    class_names = dataset.classes
    family_names, class_idx_to_family, missing_family_delim = build_family_mapping(
        class_names
    )
    if missing_family_delim:
        print(
            "[warn] Some class names lack '--' delimiter; using full name as family:"
        )
        for name in missing_family_delim:
            print(f"  - {name}")
    family_ids = torch.tensor(class_idx_to_family, device=device)
    print(
        f"Dataset after top_k filter: {len(class_names)} classes, {len(dataset)} images "
        f"(top_k_fonts={args.top_k_fonts})"
    )

    raw_dataset = raw_train_dataset
    # Build a validation dataset with light transforms (no augmentation)
    if val_folder:
        val_dataset_full = CustomImageFolder(val_folder, transform=val_transform)
        raw_val_dataset = CustomImageFolder(val_folder, transform=None)

        val_dataset_full.samples, bad_val = drop_broken_samples(val_dataset_full.samples)
        val_dataset_full.imgs = val_dataset_full.samples
        val_dataset_full.targets = [t for _, t in val_dataset_full.samples]

        raw_val_dataset.samples, bad_raw_val = drop_broken_samples(
            raw_val_dataset.samples
        )
        raw_val_dataset.imgs = raw_val_dataset.samples
        raw_val_dataset.targets = [t for _, t in raw_val_dataset.samples]

        if bad_val or bad_raw_val:
            dropped = len(set([p for p, _, _ in bad_val + bad_raw_val]))
            print(f"[warn] dropped {dropped} broken val images (see paths below)")
            for p, reason, shape in bad_val + bad_raw_val:
                print(f"  - {p} | {reason} | shape={shape}")

        val_dataset_full = filter_dataset_to_classnames(
            val_dataset_full, kept_class_names
        )
        raw_val_dataset = filter_dataset_to_classnames(
            raw_val_dataset, kept_class_names
        )

        train_indices = list(range(len(dataset)))
        val_indices = list(range(len(val_dataset_full)))
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
        dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}
        raw_dataset = raw_val_dataset
        print(
            f"Using provided val folder '{val_folder}': "
            f"train={len(train_dataset)}, val={len(val_dataset)}"
        )
    else:
        val_dataset_full = CustomImageFolder(image_folder, transform=val_transform)
        val_dataset_full.samples = list(dataset.samples)
        val_dataset_full.imgs = val_dataset_full.samples
        val_dataset_full.targets = list(dataset.targets)
        val_dataset_full.classes = list(dataset.classes)
        val_dataset_full.class_to_idx = dict(dataset.class_to_idx)

        n = len(dataset)  # total number of examples
        n_test = int(args.test_split * n)  # take ~10% for test
        if n_test <= 0 and n > 1:
            n_test = 1
        if n_test >= n and n > 1:
            n_test = n - 1

        indices = torch.randperm(n)
        val_indices = indices[:n_test].tolist()
        train_indices = indices[n_test:].tolist()

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
        dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}
        print(
            f"Split sizes -> train: {len(train_dataset)}, val: {len(val_dataset)} "
            f"(test_split={args.test_split})"
        )

    check_dataset = CustomImageFolder(image_folder, transform=check_transform)
    # reuse combined samples for visual check
    check_dataset.samples = list(dataset.samples)
    check_dataset.imgs = check_dataset.samples
    check_dataset.targets = list(dataset.targets)
    check_dataset.classes = list(dataset.classes)
    check_dataset.class_to_idx = dict(dataset.class_to_idx)
    Path(os.path.join(model_output_dir, "check")).mkdir(parents=True, exist_ok=True)
    for i, data in zip(range(100), check_dataset):
        img = data[0]
        Image.fromarray(img).save(os.path.join(model_output_dir, "check", f"{i}.png"))

    # Save classnames to a txt file
    with open(os.path.join(model_output_dir, "class_names.txt"), "w") as f:
        for item in class_names:
            f.write(f"{item}\n")
    print(f"Found {len(class_names)} classes.")

    # test_set = torch.utils.data.Subset(dataset, range(n_test))  # take first 10%
    # train_set = torch.utils.data.Subset(dataset, range(n_test, n))  # take the rest
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

    # Create a dataloader for the dataset
    batch_size = args.batch_size
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, num_workers=args.num_workers, batch_size=batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        val_dataset, num_workers=args.num_workers, batch_size=batch_size, shuffle=False
    )
    dataloaders = {"train": train_dataloader, "val": test_dataloader}

    # Build the selected model
    model = build_model(network_type, len(class_names))
    model.to(device)

    # Define the loss function and optimizer
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    backbone_params = None
    head_params = None
    if is_dinov3 and isinstance(model, nn.Sequential) and len(model) >= 2:
        backbone_params = list(model[0].parameters())
        head_params = list(model[1].parameters())
        optimizer = optim.AdamW(
            [
                {
                    "params": backbone_params,
                    "lr": args.learning_rate * args.backbone_lr_scale,
                },
                {"params": head_params, "lr": args.learning_rate},
            ],
            weight_decay=1e-4,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-4
        )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Eval-only mode: load checkpoint, run evaluation, and exit
    if args.eval_only:
        ckpt_path = args.checkpoint_path or best_model_params_path
        if not os.path.exists(ckpt_path):
            print(f"[error] checkpoint not found: {ckpt_path}")
            return
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        val_loss, val_acc1, val_acc5, val_family_acc1 = evaluate_model(
            model, dataloaders["val"], criterion, dataset_sizes["val"], family_ids
        )
        print(
            f"[eval-only] Loss: {val_loss:.4f} "
            f"Top1: {val_acc1:.4f} Top5: {val_acc5:.4f} "
            f"FamilyTop1: {val_family_acc1:.4f}"
        )
        return

    # Decay LR by a factor of 0.1 every 7 epochs
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    # lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.num_epochs, T_mult=1, eta_min=0
    )

    # Create a TensorBoard writer
    writer = SummaryWriter()

    # Initial inference on validation set before training
    init_loss, init_acc1, init_acc5, init_family_acc1 = evaluate_model(
        model, dataloaders["val"], criterion, dataset_sizes["val"], family_ids
    )
    print(
        f"Initial val -> Loss: {init_loss:.4f} "
        f"Top1: {init_acc1:.4f} Top5: {init_acc5:.4f} "
        f"FamilyTop1: {init_family_acc1:.4f}"
    )

    # Training loop
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_score = -float("inf")

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch}/{args.num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and dation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects1 = 0
            running_corrects5 = 0
            running_family_corrects1 = 0

            # Iterate over data.
            progress = tqdm(dataloaders[phase])
            for step_idx, (inputs, labels) in enumerate(progress):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad(set_to_none=True)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    # ⭐️ ⭐️ Autocasting
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds1 = torch.max(outputs, 1)
                        _, top5 = torch.topk(outputs, k=min(5, outputs.shape[1]), dim=1)
                        family_preds1 = family_ids[preds1]
                        family_labels = family_ids[labels]

                    # backward + optimize only if in training phase
                    if phase == "train":
                        scaler.scale(loss).backward()
                        if scaler.is_enabled():
                            scaler.unscale_(optimizer)
                        if args.grad_clip_norm and args.grad_clip_norm > 0:
                            nn.utils.clip_grad_norm_(
                                model.parameters(), args.grad_clip_norm
                            )
                        if args.debug_grad and epoch == 0 and step_idx == 0:
                            if backbone_params and head_params:
                                bnorm = compute_grad_norm(backbone_params)
                                hnorm = compute_grad_norm(head_params)
                                print(
                                    f"[debug] grad_norm backbone={bnorm:.4f} head={hnorm:.4f}"
                                )
                            else:
                                all_params = list(model.parameters())
                                total_norm = compute_grad_norm(all_params)
                                print(f"[debug] grad_norm total={total_norm:.4f}")
                        scaler.step(optimizer)
                        scaler.update()

                # statistics
                batch_loss = float(loss.detach().item())
                running_loss += batch_loss * inputs.size(0)
                running_corrects1 += torch.sum(preds1 == labels.data)
                running_corrects5 += torch.sum(top5.eq(labels.view(-1, 1)).any(dim=1))
                running_family_corrects1 += torch.sum(
                    family_preds1 == family_labels
                )

                # Show step loss every 50 steps on tqdm
                if phase == "train" and (step_idx + 1) % 50 == 0:
                    progress.set_postfix({"step_loss": f"{batch_loss:.4f}"})
            if phase == "train":
                scheduler.step()

            denom = dataset_sizes[phase]
            epoch_loss = running_loss / denom
            epoch_acc1 = running_corrects1.double() / denom
            epoch_acc5 = running_corrects5.double() / denom
            epoch_family_acc1 = running_family_corrects1.double() / denom

            print(
                f"{phase} Loss: {epoch_loss:.4f} "
                f"Top1: {epoch_acc1:.4f} Top5: {epoch_acc5:.4f} "
                f"FamilyTop1: {epoch_family_acc1:.4f}"
            )

            # Write the loss to TensorBoard
            writer.add_scalar(f"{phase}/Loss", epoch_loss, epoch)
            writer.add_scalar(f"{phase}/Top1", epoch_acc1, epoch)
            writer.add_scalar(f"{phase}/Top5", epoch_acc5, epoch)
            writer.add_scalar(f"{phase}/FamilyTop1", epoch_family_acc1, epoch)

            # deep copy the model
            if phase == "val":
                score = float(epoch_acc1 + epoch_acc5)
                if score > best_score:
                    best_score = score
                    best_acc1 = float(epoch_acc1)
                    best_acc5 = float(epoch_acc5)
                    torch.save(model.state_dict(), best_model_params_path)
                    save_best_predictions(
                        model,
                        val_dataset,
                        raw_dataset,
                        val_transform,
                        class_names,
                        model_output_dir,
                        epoch,
                        best_acc1,
                        best_acc5,
                        epoch_family_acc1,
                        max_samples=100,
                    )

        print(
            f"Best val -> Top1: {best_acc1:.4f} Top5: {best_acc5:.4f} "
            f"(score=Top1+Top5={best_score:.4f})"
        )

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))

        print()

    # Save the trained model
    torch.save(
        model.state_dict(), os.path.join(model_output_dir, "trained_model.pth")
    )

    # Close the TensorBoard writer
    writer.close()


if __name__ == "__main__":
    args = parse_args()

    main(args)
