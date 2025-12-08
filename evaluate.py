# evaluate.py

import torch
from sklearn.metrics import classification_report, confusion_matrix
from config import DEVICE, MODEL_PATH
from dataset import get_dataloaders
from model import CNNModel

def main():
    _, _, test_dl, class_names = get_dataloaders()

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = CNNModel().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
