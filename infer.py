# infer.py

import torch
from PIL import Image
from torchvision import transforms
from config import DEVICE, IMAGE_SIZE, MODEL_PATH
from model import CNNModel

def get_inference_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5]),
    ])

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = CNNModel().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    class_names = checkpoint["class_names"]
    return model, class_names

def predict_image(image_path:"test_image.jpg" ):
    model, class_names = load_model()
    tf = get_inference_transform()

    image = Image.open("test_image.jpg").convert("RGB")
    x = tf(image).unsqueeze(0).to(DEVICE)  # shape: [1, 3, H, W]

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = probs.argmax().item()

    return class_names[idx], probs[idx].item()

# New: prediction directly from a PIL image (for frontend uploads)
def predict_pil_image(image: Image.Image):
    model, class_names = load_model()
    tf = get_inference_transform()

    img = image.convert("RGB")
    x = tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = probs.argmax().item()

    return class_names[idx], probs[idx].item()

if __name__ == "__main__":
    # quick manual test if you want
    label, conf = predict_image_path("test_image.jpg")
    print("Prediction:", label)
    print("Confidence:", conf * 100, "%")
