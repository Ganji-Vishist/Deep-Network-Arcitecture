# utils.py
from PIL import Image
import torchvision.transforms as transforms
import torch

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image

def predict_damage(model, image_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    classes = ['Minor Damage', 'Moderate Damage', 'Severe Damage']
    return classes[predicted.item()]
