import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Classifier(nn.Module):
    """
    ResNet-18 model for image classification, adapted for TorchServe
    """
    def __init__(self, num_classes=1000):
        super(ResNet18Classifier, self).__init__()
        # Load pre-trained ResNet-18 model
        self.resnet18 = models.resnet18(pretrained=False)
        # Replace the final fully connected layer if needed
        if num_classes != 1000:
            self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet18(x)