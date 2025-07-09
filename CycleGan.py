import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.Conv2d(
                in_features, in_features, 3, padding=1, padding_mode="reflect"
            ),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_features, in_features, 3, padding=1, padding_mode="reflect"
            ),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(
        self,
        device,
        input_nc,
        output_nc,
        n_du_samples=1,
        n_residual_blocks=2,
        hidden_size=96,
    ):
        super(Generator, self).__init__()

        self.device = device

        model = [
            nn.Conv2d(
                input_nc, hidden_size, 7, padding=3, padding_mode="reflect"
            ),
            nn.InstanceNorm2d(hidden_size),
            nn.ReLU(inplace=True),
        ]

        in_features = hidden_size
        out_features = in_features * 2
        for _ in range(n_du_samples):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        out_features = in_features // 2
        for idx in range(n_du_samples):
            upsample_block = [
                nn.Conv2d(
                    in_features,
                    out_features * 4,
                    kernel_size=3,
                    padding=1,
                    padding_mode="reflect",
                ),
                nn.PixelShuffle(2),
                # nn.ConvTranspose2d(
                # in_features,
                # out_features,
                # 3,
                # stride=2,
                # padding=1,
                # output_padding=1
                # ),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]

            model += upsample_block
            in_features = out_features
            out_features = in_features // 2

        model += [
            nn.Conv2d(
                hidden_size, output_nc, 7, padding=3, padding_mode="reflect"
            ),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)
        self.model.apply(self.init_weights)

    def forward(self, x):
        return self.model(x).to(self.device)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Discriminator(nn.Module):
    def __init__(
        self, device, input_nc, hidden_n: int = 2, hidden_size: int = 64
    ):
        super(Discriminator, self).__init__()
        self.device = device
        h = hidden_size

        model = [
            nn.Conv2d(input_nc, h, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        for _ in range(hidden_n):
            h_next = h * 2
            model += [
                nn.Conv2d(h, h_next, 4, stride=2, padding=1),
                nn.InstanceNorm2d(h_next),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            h = h_next

        model += [nn.Conv2d(h, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)
        self.model.apply(self.init_weights)

    def forward(self, x):
        x = self.model(x)
        return (
            F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1).to(self.device)
        )

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
