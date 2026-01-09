import torch.nn as nn
import torchvision.models as models
from .cbam import CBAM

class ResEmoteNetCBAM(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.5): # Added dropout_rate
        super().__init__()
        # ... (features and cbam layers remain the same) ...
        self.cbam = CBAM(2048) 
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # New: Add Dropout layer for regularization
        self.dropout = nn.Dropout(p=dropout_rate) 
        
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.pool(x).flatten(1)
        
        # New: Apply dropout before the final classifier
        x = self.dropout(x) 
        
        return self.fc(x)
