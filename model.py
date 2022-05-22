import torch
import torch.nn as nn


class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride) -> None:
        super(CBL, self).__init__()
        self.padding = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, self.padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.leakyrelu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, residual=True) -> None:
        super(ResidualBlock, self).__init__()
        self.residual = residual
        self.conv1 = CBL(in_channels, in_channels //
                         2, kernel_size=1, stride=1)
        self.conv2 = CBL(in_channels // 2, in_channels,
                         kernel_size=3, stride=1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.residual else self.conv2(self.conv1(x))


class Darknet53(nn.Module):
    def __init__(self, in_channels=3, blocks=[1, 2, 8, 8, 4]):
        super(Darknet53, self).__init__()
        # input_size: (Batch_size, 3, H, W)
        self.in_channels = in_channels
        self.conv1 = CBL(self.in_channels, 32, 3, 1)  # (32, H, W)
        self.conv2 = CBL(32, 64, 3, 2)  # (64, H/2, W/2)
        self.layer1 = self._make_layer(64, 1)  # (64, H/2, W/2)
        self.conv3 = CBL(64, 128, 3, 2)  # (128, H/4, W/4)
        self.layer2 = self._make_layer(128, 2)  # (128, H/4, W/4)
        self.conv4 = CBL(128, 256, 3, 2)  # (256, H/8, W/8)
        self.layer3 = self._make_layer(256, 8)  # (256, H/8, W/8)
        self.conv5 = CBL(256, 512, 3, 2)  # (512, H/16, W/16)
        self.layer4 = self._make_layer(512, 8)  # (512, H/16, W/16)
        self.conv6 = CBL(512, 1024, 3, 2)  # (1024, H/32, W/32)
        self.layer5 = self._make_layer(1024, 4)  # (1024, H/32, W/32)

    def _make_layer(self, in_channels, repeat_times):
        layer = []
        for _ in range(repeat_times):
            layer.append(ResidualBlock(in_channels))
            return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.layer1(x)
        x = self.conv3(x)
        x = self.layer2(x)
        x = self.conv4(x)
        x = self.layer3(x)
        c3 = x
        x = self.conv5(x)
        x = self.layer4(x)
        c4 = x
        x = self.conv6(x)
        c5 = self.layer5(x)
        return c3, c4, c5


class FPN(nn.Module):
    def __init__(self, c3_channels=256, c4_channels=512, c5_channels=1024) -> None:
        super(FPN, self).__init__()
        self.c3_channels = c3_channels
        self.c4_channels = c4_channels
        self.c5_channels = c5_channels
        # --------generate c5------------ # c5 -> (1024, H/32, W/32)
        self.conv1 = nn.Sequential(
            CBL(c5_channels, 512, 1, 1),
            CBL(512, 1024, 3, 1)
        )
        # --------generate c4------------ #c4 -> (512, H/16, W/16)
        self.conv2 = nn.Sequential(
            CBL(1024, 256, 1, 1),
            nn.Upsample(scale_factor=2)
        )
        self.conv3 = nn.Sequential(
            CBL(256 * 3, 256, 1, 1),
            CBL(256, 512, 3, 1)
        )
        # --------generate c3------------ # c3 -> (256, H/8, W/8)
        self.conv4 = nn.Sequential(
            CBL(512, 128, 1, 1),
            nn.Upsample(scale_factor=2)
        )
        self.conv5 = nn.Sequential(
            CBL(128 * 3, 128, 1, 1),
            CBL(128, 256, 3, 1)
        )

    def forward(self, c3, c4, c5):
        c5 = self.conv1(c5)
        c4 = self.conv3(torch.cat((self.conv2(c5), c4), dim=1))
        c3 = self.conv5(torch.cat((self.conv4(c4), c3), dim=1))
        return c3, c4, c5


class DetectionHead(nn.Module):
    def __init__(self, c3_channels=256, c4_channels=512, c5_channels=1024, num_classes=20) -> None:
        super(DetectionHead, self).__init__()
        self.num_classes = num_classes
        self.c3_channels = c3_channels
        self.c4_channels = c4_channels
        self.c5_channels = c5_channels
        self.c3head = nn.Sequential(
            CBL(c3_channels, c3_channels // 2, kernel_size=1, stride=1),
            CBL(c3_channels // 2, c3_channels, kernel_size=3, stride=1),
            CBL(c3_channels, c3_channels // 2, kernel_size=1, stride=1),
            CBL(c3_channels // 2, c3_channels, kernel_size=3, stride=1),
            nn.Conv2d(c3_channels, 3 * (num_classes + 5),
                      kernel_size=1, bias=True)
        )
        self.c4head = nn.Sequential(
            CBL(c4_channels, c4_channels // 2, kernel_size=1, stride=1),
            CBL(c4_channels // 2, c4_channels, kernel_size=3, stride=1),
            CBL(c4_channels, c4_channels // 2, kernel_size=1, stride=1),
            CBL(c4_channels // 2, c4_channels, kernel_size=3, stride=1),
            nn.Conv2d(c4_channels, 3 * (num_classes + 5),
                      kernel_size=1, bias=True)
        )
        self.c5head = nn.Sequential(
            CBL(c5_channels, c5_channels // 2, kernel_size=1, stride=1),
            CBL(c5_channels // 2, c5_channels, kernel_size=3, stride=1),
            CBL(c5_channels, c5_channels // 2, kernel_size=1, stride=1),
            CBL(c5_channels // 2, c5_channels, kernel_size=3, stride=1),
            nn.Conv2d(c5_channels, 3 * (num_classes + 5),
                      kernel_size=1, bias=True)
        )

    def forward(self, c3, c4, c5):
        c3 = self.c3head(c3)
        c4 = self.c4head(c4)
        c5 = self.c5head(c5)
        c3 = c3.reshape(c3.shape[0], 3, self.num_classes + 5,
                        c3.shape[2], c3.shape[3]).permute(0, 1, 3, 4, 2)
        c4 = c4.reshape(c4.shape[0], 3, self.num_classes + 5,
                        c4.shape[2], c4.shape[3]).permute(0, 1, 3, 4, 2)
        c5 = c5.reshape(c5.shape[0], 3, self.num_classes + 5,
                        c5.shape[2], c5.shape[3]).permute(0, 1, 3, 4, 2)
        return c3, c4, c5


class YOLOv3(nn.Module):
    def __init__(self, num_classes=20) -> None:
        super(YOLOv3, self).__init__()
        self.backbone = Darknet53()
        self.fpn = FPN()
        self.head = DetectionHead(num_classes=num_classes)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        c3, c4, c5 = self.fpn(c3, c4, c5)
        c3, c4, c5 = self.head(c3, c4, c5)
        return (c5, c4, c3)


if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert out[0].shape == (2, 3, IMAGE_SIZE //
                            32, IMAGE_SIZE//32, num_classes + 5)
    assert out[1].shape == (2, 3, IMAGE_SIZE //
                            16, IMAGE_SIZE//16, num_classes + 5)
    assert out[2].shape == (2, 3, IMAGE_SIZE //
                            8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")
