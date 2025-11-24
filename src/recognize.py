from PIL import Image
import torch
from torchvision import transforms

# Трансформы
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((45, 45)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Классы модели
classes = ['0','1','2','3','4','5','6','7','8','9','+','-','*','/']

def recognize_image(img, model):
    """
    img: PIL Image
    model: обученная модель
    """
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)
    return classes[pred.item()]
