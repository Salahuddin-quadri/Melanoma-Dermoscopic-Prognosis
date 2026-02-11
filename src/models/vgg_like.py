"""
VGG-like reference model for dermoscopic image analysis.
Compatible with the system's model interface (PyTorch, similar to ResNet50Model).
"""
import torch
import torch.nn as nn

class VGGLike(nn.Module):
    def __init__(self, num_classes: int = 1, hidden_dim: int = 256, dropout: float = 0.3, task: str = "classification", multitask: bool = False):
        super().__init__()
        self.task = task
        self.multitask = multitask
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(256 * 28 * 28, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.reg_head = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        if self.multitask:
            cls = self.sigmoid(self.cls_head(x)).squeeze(1)
            reg = self.reg_head(x).squeeze(1)
            return {"cls": cls, "reg": reg}
        else:
            if self.task == "classification":
                return self.sigmoid(self.cls_head(x)).squeeze(1)
            else:
                return self.reg_head(x).squeeze(1)

def create_vgg_like_model(hidden_dim: int = 256, dropout: float = 0.3, task: str = "classification", multitask: bool = False):
    return VGGLike(hidden_dim=hidden_dim, dropout=dropout, task=task, multitask=multitask)
