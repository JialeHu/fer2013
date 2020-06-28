
import torch.nn as nn
import torch.nn.functional as F

class ImageNet(nn.Module):

    def __init__(self):
        super(ImageNet, self).__init__()

        # Kernal
        # 1 input image channel (48x48), 64 output channels, 3x3 square conv
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(512)

        # y = Wx + b
        self.lin1 = nn.Linear(512*6*6, 512*6)
        self.lin1_bn = nn.BatchNorm1d(512*6)
        self.lin2 = nn.Linear(512*6, 512)
        self.lin2_bn = nn.BatchNorm1d(512)
        self.lin3 = nn.Linear(512, 128)
        self.lin3_bn = nn.BatchNorm1d(128)
        self.lin4 = nn.Linear(128, 7)

    def forward(self, x):

        # (Bx1x48x48) -> BN -> Conv1 -> BN -> Relu -> MaxPool -> (Bx64x24x24)
        x = F.max_pool2d(F.dropout2d(F.relu(self.conv1_bn(self.conv1(x)))), (2, 2))
        # -> conv2 -> BN -> Relu -> MaxPool -> (Bx128x12x12)
        x = F.max_pool2d(F.dropout2d(F.relu(self.conv2_bn(self.conv2(x)))), 2)
        # -> conv3 -> BN -> Relu -> MaxPool -> (Bx256x6x6)
        x = F.max_pool2d(F.dropout2d(F.relu(self.conv3_bn(self.conv3(x)))), 2)
        # -> conv4 -> BN -> Relu -> MaxPool -> (Bx512x6x6)
        x = F.dropout2d(F.relu(self.conv4_bn(self.conv4(x))))
        # -> Flatten -> (Bx512*6*6)
        x = x.view(-1, self.num_flat_features(x))
        # -> lin1 -> BN -> Relu -> (Bx512*6)
        x = F.relu(F.dropout(self.lin1_bn(self.lin1(x))))
        # -> lin2 -> BN -> Relu -> (Bx512)
        x = F.relu(F.dropout(self.lin2_bn(self.lin2(x))))

        x = F.relu(F.dropout(self.lin3_bn(self.lin3(x))))
        # -> lin3 -> (Bx7)
        x = F.softmax(self.lin4(x), dim=1)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        
        return num_features

    def num_train_parameters(self):
        # Return Model Stats
        train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return train_params

        