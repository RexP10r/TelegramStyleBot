import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features, affine=True),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, device, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        self.device = device

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
        ]

        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features, affine=True),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=True
                ),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_features, out_features, 3, stride=1),
                nn.InstanceNorm2d(out_features, affine=True),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x).to(self.device)


class Discriminator(nn.Module):
    def __init__(
        self, device, input_nc, hidden_n: int = 3, hidden_size: int = 64
    ):
        super(Discriminator, self).__init__()
        self.device = device
        h = hidden_size

        model = [
            nn.utils.spectral_norm(
                nn.Conv2d(input_nc, h, 4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        for _ in range(hidden_n):
            h_next = h * 2
            model += [
                nn.utils.spectral_norm(
                    nn.Conv2d(h, h_next, 4, stride=2, padding=1)
                ),
                nn.InstanceNorm2d(h_next, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            h = h_next

        model += [nn.Conv2d(h, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return (
            F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1).to(self.device)
        )
