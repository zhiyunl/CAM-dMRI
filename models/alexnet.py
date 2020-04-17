import torch.nn as nn


class AlexNet_BN(nn.Module):
    def __init__(self, num_class):
        super(AlexNet_BN, self).__init__()
        self.conv = nn.Sequential(
            # 1 x ? x ?
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),

            # 32 x ? x ?
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            # 64 x ? x ?
            nn.MaxPool2d(2, 2),
            # 64 x ? / 2 x ? / 2
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(0.2),
            # 128 x ? / 2 x ? / 2
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(0.2),
            # 256 x ? / 2 x ? / 2
            nn.MaxPool2d(2, 2),
            # 256 x ? / 4 x ? / 4
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(0.2),
            # 512 x ? / 4 x ? / 4
            nn.MaxPool2d(2, 2),
            # 512 x ? / 8 x ? / 8
            # newly added
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024, eps=0.001),
            nn.LeakyReLU(0.2)
        )
        # modification
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # same as img_size//8
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(1024, num_class)

    def forward(self, x):
        features = self.conv(x)
        # print(features.shape)
        flatten = self.dropout(self.gap(features).view(features.size(0), -1))
        # print(flatten.shape)
        output = self.classifier(flatten)
        return output


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # TODO remove layer after conv5
            #   add one more conv layer
            # size Nx256x13x13
            nn.Conv2d(256, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = self.gap(x).view(x.size(0), -1)
        # x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(bn=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    """
    if bn:
        return AlexNet_BN(**kwargs)
    else:
        return AlexNet(**kwargs)
