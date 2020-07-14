import torch
import torch.nn as nn
from sacred import Experiment

ex = Experiment("UNet")

from config import initialise
from utils.tupperware import tupperware

ex = initialise(ex)


def convrelu(
    in_channels, out_channels, kernel=3, padding=1, stride=1, normaliser=nn.BatchNorm2d
):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding, stride=stride),
        normaliser(out_channels),
        nn.ReLU(inplace=True),
    )


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale, normaliser):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1
        )
        self.upsample = nn.PixelShuffle(up_scale)
        self.normaliser = normaliser(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        x = self.normaliser(x)
        x = self.relu(x)
        return x


class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        normaliser = nn.BatchNorm2d

        # _0/_1 represents encoder branch
        self.layer0_0 = nn.Sequential(
            convrelu(3, 32, 3, 1, 1, normaliser=normaliser),
            convrelu(32, 32, 3, 1, 1, normaliser=normaliser),
            nn.MaxPool2d(2, 2),  # size=(N, 32, x.H/2, x.W/2)
        )

        self.layer0_1 = nn.Sequential(
            convrelu(3, 32, 3, 1, 1, normaliser=normaliser),
            convrelu(32, 32, 3, 1, 1, normaliser=normaliser),
        )
        self.maxpool0 = nn.MaxPool2d(2, 2)  # size=(N, 32, x.H/2, x.W/2)

        self.layer1_0 = nn.Sequential(
            convrelu(32, 64, 3, 1, 1, normaliser=normaliser),
            convrelu(64, 64, 3, 1, 1, normaliser=normaliser),
            nn.MaxPool2d(2, 2),
        )  # size=(N, 64, x.H/4, x.W/4)

        self.layer1_1 = nn.Sequential(
            convrelu(32, 64, 3, 1, 1, normaliser=normaliser),
            convrelu(64, 64, 3, 1, 1, normaliser=normaliser),
        )
        self.maxpool1 = nn.MaxPool2d(2, 2)  # size=(N, 64, x.H/4, x.W/4)

        self.layer2_0 = nn.Sequential(
            convrelu(64, 128, 3, 1, 1, normaliser=normaliser),
            convrelu(128, 128, 3, 1, 1, normaliser=normaliser),
            nn.MaxPool2d(2, 2),
        )  # size=(N, 128, x.H/8, x.W/8)

        self.layer2_1 = nn.Sequential(
            convrelu(64, 128, 3, 1, 1, normaliser=normaliser),
            convrelu(128, 128, 3, 1, 1, normaliser=normaliser),
        )
        self.maxpool2 = nn.MaxPool2d(2, 2)  # size=(N, 128, x.H/8, x.W/8)

        self.layer3_0 = nn.Sequential(
            convrelu(128, 256, 3, 1, 1, normaliser=normaliser),
            convrelu(256, 256, 3, 1, 1, normaliser=normaliser),
            nn.MaxPool2d(2, 2),
        )  # size=(N, 256, x.H/16, x.W/16)

        self.layer3_1 = nn.Sequential(
            convrelu(128, 256, 3, 1, 1, normaliser=normaliser),
            convrelu(256, 256, 3, 1, 1, normaliser=normaliser),
        )

        self.layer4_0 = nn.Sequential(
            convrelu(256, 512, 3, 1, 1, normaliser=normaliser),
            convrelu(512, 512, 3, 1, 1, normaliser=normaliser),
        )  # size=(N, 512, x.H/16, x.W/16)

        self.conv_up3 = nn.Sequential(
            convrelu(256 + 256, 256, 3, 1, normaliser=normaliser),
            convrelu(256, 256, 3, 1, 1, normaliser=normaliser),
        )

        self.conv_up2 = nn.Sequential(
            convrelu(128 + 128, 128, 3, 1, normaliser=normaliser),
            convrelu(128, 128, 3, 1, 1, normaliser=normaliser),
        )

        self.conv_up1 = nn.Sequential(
            convrelu(64 + 64, 64, 3, 1, normaliser=normaliser),
            convrelu(64, 64, 3, 1, 1, normaliser=normaliser),
        )

        self.conv_up0 = nn.Sequential(
            convrelu(32 + 32, 32, 3, 1, normaliser=normaliser),
            convrelu(32, 32, 3, 1, 1, normaliser=normaliser),
        )

        self.upsample_16_8 = nn.ConvTranspose2d(
            512, 256, 2, stride=2
        )  # upsample H/16 to H/8
        self.upsample_8_4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upsample_4_2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.upsample_2_1 = nn.ConvTranspose2d(64, 32, 2, stride=2)

        self.get_image = nn.Sequential(nn.Conv2d(32, 3, 1), nn.Tanh())

    def forward(self, img):

        # first encoder branch
        x = self.layer0_0(img)
        x = self.layer1_0(x)
        x = self.layer2_0(x)
        x = self.layer3_0(x)
        layer4_0 = self.layer4_0(x)  # 512 channels

        # second encoder branch
        layer0_1 = self.layer0_1(img)

        layer1_1 = self.maxpool0(layer0_1)
        layer1_1 = self.layer1_1(layer1_1)

        layer2_1 = self.maxpool1(layer1_1)
        layer2_1 = self.layer2_1(layer2_1)

        layer3_1 = self.maxpool2(layer2_1)
        layer3_1 = self.layer3_1(layer3_1)

        x = self.upsample_16_8(layer4_0, output_size=layer3_1.size())
        x = torch.cat([x, layer3_1], dim=1)
        x = self.conv_up3(x)  # size=(N, 256, x.H/8, x.W/8)

        x = self.upsample_8_4(x)  # size=(N, 128, x.H/4, x.W/4)
        x = torch.cat([x, layer2_1], dim=1)
        x = self.conv_up2(x)  # size=(N, 128, x.H/4, x.W/4)

        x = self.upsample_4_2(x)  # size=(N, 64, x.H/2, x.W/2)
        x = torch.cat([x, layer1_1], dim=1)
        x = self.conv_up1(x)  # size=(N, 64, x.H/2, x.W/2)

        x = self.upsample_2_1(x)  # size=(N, 32, x.H, x.W)
        x = torch.cat([x, layer0_1], dim=1)
        x = self.conv_up0(x)  # size=(N, 32, x.H, x.W)

        x = self.get_image(x)

        return x


@ex.automain
def main(_run):
    from torchsummary import summary

    args = tupperware(_run.config)
    model = Unet().to(args.device)

    summary(model, (3, 256, 512))
