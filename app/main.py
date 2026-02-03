import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from configs.config import IMAGE_SIZE, NUM_CLASSES
from src.model import build_model
from src.gradcam import GradCAM
from src.utils import get_device

CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

def denormalize(img_tensor):
    img = img_tensor.clone().cpu()
    for c, (m, s) in enumerate(zip(CIFAR10_MEAN, CIFAR10_STD)):
        img[c] = img[c] * s + m
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()

st.set_page_config(page_title="CIFAR-10 + Grad-CAM", layout="wide")
st.title("🖼️ CIFAR-10 Classifier (ResNet18) + Grad-CAM")

device = get_device()

@st.cache_resource
def load_model():
    model = build_model(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load("outputs/models/best.pt", map_location=device))
    model.eval()
    return model

model = load_model()

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    x = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))

    st.subheader("Top-3 Predictions")
    top3 = probs.argsort()[-3:][::-1]
    for i, idx in enumerate(top3, 1):
        st.write(f"{i}. **{CLASSES[idx]}** — {probs[idx]*100:.2f}%")

    st.subheader("Grad-CAM (model attention)")
    target_layer = model.layer4[-1].conv2
    cam = GradCAM(model, target_layer)
    heatmap = cam(x, class_idx=pred).detach().cpu().numpy()

    # Make a heatmap image the same size as uploaded image
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(img.size)

    fig = plt.figure()
    plt.imshow(img)
    plt.imshow(heatmap_img, alpha=0.45)
    plt.axis("off")
    st.pyplot(fig)
else:
    st.info("Upload an image to see predictions and a Grad-CAM heatmap.")
