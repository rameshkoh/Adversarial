import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image

# Load a pre-trained model (e.g., ResNet)
model = models.resnet50(pretrained=True)
model.eval()

# Load an image and preprocess
image = Image.open('path_to_image.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

# Define epsilon
epsilon = 0.01

# Compute the gradient of the loss w.r.t. the input
input_batch.requires_grad = True
output = model(input_batch)
loss = nn.CrossEntropyLoss()(output, torch.tensor([true_label]))

# Backpropagate the error
model.zero_grad()
loss.backward()

# Apply FGSM
data_grad = input_batch.grad.data
perturbed_image = input_batch + epsilon * data_grad.sign()
perturbed_image = torch.clamp(perturbed_image, 0, 1)
