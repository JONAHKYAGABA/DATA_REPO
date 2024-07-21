import torch
import torch.nn as nn
import torchvision.models as models

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class VGGMODEL(nn.Module):
    def __init__(self, list_of_classes):
        super(VGGMODEL, self).__init__()
        self.list_of_classes = list_of_classes
        self.num_classes = len(self.list_of_classes)
        
        # Load the VGG19 model with pre-trained weights
        self.vgg19 = models.vgg19(pretrained=True)
        
        # Add SE-Layer to each block
        self.se_layers = nn.ModuleList([
            SELayer(64),
            SELayer(128),
            SELayer(256),
            SELayer(512),
        ])
        
        # Modify the classifier to include dropout and match the number of classes
        self.vgg19.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # Change input size to match custom feature map size
            nn.ReLU(True),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.5),
            nn.Linear(4096, self.num_classes)
        )
    
    def forward(self, x):
        se_layer_indices = [0, 5, 10, 19]  # Indices where SE-Layers should be inserted
        se_layer_counter = 0
        for i, (name, layer) in enumerate(self.vgg19.features._modules.items()):
            x = layer(x)
            if i in se_layer_indices:
                x = self.se_layers[se_layer_counter](x)
                se_layer_counter += 1
        x = self.vgg19.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.vgg19.classifier(x)
        return x
