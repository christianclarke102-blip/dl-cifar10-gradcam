import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

from configs.config import IMAGE_SIZE, NUM_CLASSES
from src.model import build_model
from src.utils import get_device
from src.gradcam import GradCAM

CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

def denormalize(img_tensor):
    """Tensor CxHxW -> numpy HxWxC in [0,1] for display."""
    img = img_tensor.clone().cpu()
    for c, (m, s) in enumerate(zip(CIFAR10_MEAN, CIFAR10_STD)):
        img[c] = img[c] * s + m
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()

def main():
    device = get_device()
    os.makedirs("outputs/figures", exist_ok=True)

    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_set = datasets.CIFAR10(root="data", train=False, download=True, transform=tf)

    model = build_model(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load("outputs/models/best.pt", map_location=device))
    model.eval()

    # target layer: last conv in ResNet layer4 block
    target_layer = model.layer4[-1].conv2
    cam = GradCAM(model, target_layer)

    # pick a few samples
    sample_indices = [0, 7, 25, 80]
    for i, idx in enumerate(sample_indices, start=1):
        x, y = test_set[idx]
        x_in = x.unsqueeze(0).to(device)

        # predict
        with torch.no_grad():
            logits = model(x_in)
            pred = logits.argmax(dim=1).item()

        heatmap = cam(x_in, class_idx=pred).detach().cpu().numpy() # HxW in [0,1]
        img = denormalize(x)

        # overlay
        plt.figure()
        plt.imshow(img)
        plt.imshow(heatmap, alpha=0.45)
        plt.axis("off")
        plt.title(f"True: {CLASSES[y]} | Pred: {CLASSES[pred]}")
        out_path = f"outputs/figures/gradcam_{i}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"✅ Saved {out_path}")

if __name__ == "__main__":
    main()
