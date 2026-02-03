import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from configs.config import BATCH_SIZE, NUM_WORKERS, IMAGE_SIZE, NUM_CLASSES
from src.data import get_loaders
from src.model import build_model
from src.utils import get_device

CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def main():
    device = get_device()
    _, _, test_loader = get_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, image_size=IMAGE_SIZE)

    model = build_model(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load("outputs/models/best.pt", map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print(classification_report(y_true, y_pred, target_names=CLASSES))

    cm = confusion_matrix(y_true, y_pred)
    os.makedirs("outputs/figures", exist_ok=True)

    plt.figure()
    plt.imshow(cm)
    plt.title("CIFAR-10 Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("outputs/figures/confusion_matrix.png", dpi=200)
    print("✅ Saved outputs/figures/confusion_matrix.png")

if __name__ == "__main__":
    main()
