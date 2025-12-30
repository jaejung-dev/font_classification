import albumentations as A
import argparse
import numpy as np
import os
import timm
import torch

from albumentations.pytorch import ToTensorV2
from timm.data import resolve_model_data_config
from train import CutMax, ResizeWithPad
from PIL import Image


def parse_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Inference script")

    # Add arguments
    parser.add_argument(
        "--model_folder",
        type=str,
        default="sample_data/model",
        help="Path where the trained model was saved",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="sample_data/output/Lato-Regular",
        help="Path to images to run inference on",
    )
    parser.add_argument(
        "-net",
        "--network_type",
        type=str,
        default="resnet50",
        help="Type of network architecture",
    )
    args = parser.parse_args()

    return args


def main(args):
    with open(os.path.join(args.model_folder, "class_names.txt"), "r") as f:
        class_names = f.read().splitlines()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_model = timm.create_model(args.network_type, pretrained=False, num_classes=0)
    data_cfg = resolve_model_data_config(cfg_model)
    _, h, w = data_cfg["input_size"]
    target_size = (w, h)
    norm_mean = list(data_cfg["mean"])
    norm_std = list(data_cfg["std"])

    model = timm.create_model(
        args.network_type, pretrained=False, num_classes=len(class_names)
    )
    model.to(device)

    model_path = os.path.join(args.model_folder, "trained_model.pth")
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint)
    model.eval()

    transform = A.Compose(
        [
            A.Lambda(image=CutMax(1024)),
            A.Lambda(image=ResizeWithPad(target_size)),
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2(),
        ]
    )

    for image_file in os.listdir(args.data_folder):
        image_path = os.path.join(args.data_folder, image_file)
        with Image.open(image_path) as im:
            image = np.array(im.convert("RGB"))
        image = transform(image=image)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model(image)
            _, prediction = torch.max(probs, 1)
        prediction_idx = int(prediction.item())
        print(image_file, class_names[prediction_idx])


if __name__ == "__main__":
    args = parse_args()

    main(args)
