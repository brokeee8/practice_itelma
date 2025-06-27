import torch
print(torch.cuda.is_available())          # Должно быть True
print(torch.cuda.get_device_name(0))      # Название твоей видеокарты
import torch
import torchvision

print(torch.__version__)           # Должна быть ~2.x
print(torch.cuda.is_available())  # True
print(torchvision.__version__)     # Должна быть примерно ~0.15+
