import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.model_serialization import load_state_dict
from utils.ops import unpixel_shuffle


from models.gca_net_improved import GCANet_improved, GCANet_improved_deeper
from models.gca_net_atrous import (
    GCANet_atrous,
    GCANet_atrous_corrected,
    GCANet_atrous_stage_2,
    SmoothDilatedResidualAtrousGuidedBlock,
    SmoothDilatedResidualAtrousGuidedBlockStage2,
)

# Deep Guided Filter (DGF) utils
from models.DGF_utils.guided_filter import FastGuidedFilter, ConvGuidedFilter
from models.DGF_utils.adaptive_norm import AdaptiveNorm
from models.DGF_utils.weights_init import weights_init_identity
from models.DGF_utils.eca_module import eca_layer
from models.DGF_utils.SIREN import SIREN
from models.DGF_utils.adaptive_norm import norm_dict


from sacred import Experiment
from config import initialise

ex = Experiment("DGF")
ex = initialise(ex)


def build_lr_net(norm=AdaptiveNorm, layer=5):
    layers = [
        nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(24),
        nn.LeakyReLU(0.2, inplace=True),
    ]

    for l in range(1, layer):
        layers += [
            nn.Conv2d(
                24,
                24,
                kernel_size=3,
                stride=1,
                padding=2 ** l,
                dilation=2 ** l,
                bias=False,
            ),
            norm(24),
            nn.LeakyReLU(0.2, inplace=True),
        ]

    layers += [
        nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(24),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(24, 3, kernel_size=1, stride=1, padding=0, dilation=1),
    ]

    net = nn.Sequential(*layers)

    net.apply(weights_init_identity)

    return net


def build_lr_net_pixelshuffle(
    args, norm=AdaptiveNorm, layer=5, use_eca=False, siren=False
):
    if siren:
        activation = SIREN()
    else:
        activation = nn.LeakyReLU(0.2)

    layers = [
        nn.Conv2d(
            3 * args.pixelshuffle_ratio ** 2,
            48,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False,
        ),
        norm(48),
        activation,
    ]

    for l in range(1, layer):
        if (l % 2 == 0) or (not use_eca):
            layers += [
                nn.Conv2d(
                    48,
                    48,
                    kernel_size=3,
                    stride=1,
                    padding=2 ** l,
                    dilation=2 ** l,
                    bias=False,
                ),
                norm(48),
                activation,
            ]
        else:
            layers += [
                nn.Conv2d(
                    48,
                    48,
                    kernel_size=3,
                    stride=1,
                    padding=2 ** l,
                    dilation=2 ** l,
                    bias=False,
                ),
                norm(48),
                eca_layer(48),
                activation,
            ]

    if use_eca:
        layers += [
            nn.Conv2d(
                48, 48, kernel_size=3, stride=1, padding=1, dilation=1, bias=False
            ),
            norm(48),
            activation,
            eca_layer(48),
            nn.Conv2d(
                48,
                3 * args.pixelshuffle_ratio ** 2,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
            ),
        ]
    else:
        layers += [
            nn.Conv2d(
                48, 48, kernel_size=3, stride=1, padding=1, dilation=1, bias=False
            ),
            norm(48),
            activation,
            nn.Conv2d(
                48,
                3 * args.pixelshuffle_ratio ** 2,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
            ),
        ]

    net = nn.Sequential(*layers)

    net.apply(weights_init_identity)

    return net


class DeepGuidedFilter(nn.Module):
    def __init__(self, radius=1, eps=1e-8):
        super(DeepGuidedFilter, self).__init__()
        self.lr = build_lr_net()
        self.gf = FastGuidedFilter(radius, eps)

    def forward(self, x_lr, x_hr):
        return self.gf(x_lr, self.lr(x_lr), x_hr).clamp(0, 1)

    def init_lr(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        load_state_dict(self.lr, checkpoint["state_dict"])


class DeepGuidedFilterAdvanced(DeepGuidedFilter):
    def __init__(self, radius=1, eps=1e-4):
        super(DeepGuidedFilterAdvanced, self).__init__(radius, eps)

        self.guided_map = nn.Sequential(
            nn.Conv2d(3, 15, 1, bias=False),
            AdaptiveNorm(15),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(15, 3, 1),
        )
        self.guided_map.apply(weights_init_identity)

    def forward(self, x_lr, x_hr):
        return self.gf(self.guided_map(x_lr), self.lr(x_lr), self.guided_map(x_hr))


class DeepGuidedFilterConvGF(nn.Module):
    def __init__(self, radius=1, layer=5):
        super(DeepGuidedFilterConvGF, self).__init__()
        self.lr = build_lr_net(layer=layer)
        self.gf = ConvGuidedFilter(radius, norm=AdaptiveNorm)

    def forward(self, x_lr, x_hr):
        return F.tanh(self.gf(x_lr, self.lr(x_lr), x_hr))

    def init_lr(self, path):
        self.lr.load_state_dict(torch.load(path))


class DeepGuidedFilterGuidedMapConvGF(DeepGuidedFilterConvGF):
    def __init__(self, radius=1, dilation=0, c=16, layer=5):
        super(DeepGuidedFilterGuidedMapConvGF, self).__init__(radius, layer)

        self.guided_map = nn.Sequential(
            nn.Conv2d(3, c, 1, bias=False)
            if dilation == 0
            else nn.Conv2d(3, c, 3, padding=dilation, dilation=dilation, bias=False),
            AdaptiveNorm(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, 3, 1),
        )

        self.downsample = nn.Upsample(
            scale_factor=0.5, mode="bilinear", align_corners=True
        )

    def forward(self, x_hr):
        x_lr = self.downsample(x_hr)
        return F.tanh(
            self.gf(self.guided_map(x_lr), self.lr(x_lr), self.guided_map(x_hr))
        )


class DeepGuidedFilterGuidedMapConvGFPixelShuffle(nn.Module):
    def __init__(self, args, radius=1, dilation=0):
        super(DeepGuidedFilterGuidedMapConvGFPixelShuffle, self).__init__()

        c = args.guided_map_channels

        self.guided_map = nn.Sequential(
            nn.Conv2d(
                3,
                c,
                kernel_size=args.guided_map_kernel_size,
                padding=args.guided_map_kernel_size // 2,
                bias=False,
            )
            if dilation == 0
            else nn.Conv2d(3, c, 3, padding=dilation, dilation=dilation, bias=False),
            AdaptiveNorm(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                c,
                3,
                kernel_size=args.guided_map_kernel_size,
                padding=args.guided_map_kernel_size // 2,
            ),
        )
        self.lr = build_lr_net_pixelshuffle(
            args, layer=args.CAN_layers, siren=args.use_SIREN, use_eca=args.use_ECA
        )
        self.gf = ConvGuidedFilter(radius, norm=AdaptiveNorm)
        self.downsample = nn.Upsample(
            scale_factor=0.5, mode="bilinear", align_corners=True
        )

    def init_lr(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        load_state_dict(self.lr, checkpoint["state_dict"])

    def forward(self, x_hr):
        x_lr = self.downsample(x_hr)
        x_lr_unpixelshuffled = unpixel_shuffle(x_lr, 2)
        y_lr = F.pixel_shuffle(self.lr(x_lr_unpixelshuffled), 2)

        return F.tanh(self.gf(self.guided_map(x_lr), y_lr, self.guided_map(x_hr)))


class DeepGuidedFilterGuidedMapConvGFPixelShuffleGCAImproved(nn.Module):
    def __init__(
        self,
        args,
        radius=1,
        dilation=0,
        use_FFA: bool = False,
        use_deeper_GCAN: bool = False,
    ):
        super(DeepGuidedFilterGuidedMapConvGFPixelShuffleGCAImproved, self).__init__()

        c = args.guided_map_channels

        self.guided_map = nn.Sequential(
            nn.Conv2d(
                3,
                c,
                kernel_size=args.guided_map_kernel_size,
                padding=args.guided_map_kernel_size // 2,
                bias=False,
            )
            if dilation == 0
            else nn.Conv2d(3, c, 3, padding=dilation, dilation=dilation, bias=False),
            AdaptiveNorm(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                c,
                3,
                kernel_size=args.guided_map_kernel_size,
                padding=args.guided_map_kernel_size // 2,
            ),
        )

        self.lr = GCANet_improved(
            in_c=3 * args.pixelshuffle_ratio ** 2,
            out_c=3 * args.pixelshuffle_ratio ** 2,
            use_FFA=use_FFA,
        )

        if use_deeper_GCAN:
            self.lr = GCANet_improved_deeper(
                in_c=3 * args.pixelshuffle_ratio ** 2,
                out_c=3 * args.pixelshuffle_ratio ** 2,
                use_FFA=use_FFA,
            )

        if args.use_ECA:
            self.lr = GCAECANet_improved(
                in_c=3 * args.pixelshuffle_ratio ** 2,
                out_c=3 * args.pixelshuffle_ratio ** 2,
            )

        self.gf = ConvGuidedFilter(radius, norm=AdaptiveNorm)

        self.downsample = nn.Upsample(
            scale_factor=0.5, mode="bilinear", align_corners=True
        )

    def init_lr(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        load_state_dict(self.lr, checkpoint["state_dict"])

    def forward(self, x_hr):
        x_lr = self.downsample(x_hr)
        x_lr_unpixelshuffled = unpixel_shuffle(x_lr, 2)
        y_lr = F.pixel_shuffle(self.lr(x_lr_unpixelshuffled), 2)

        return F.tanh(self.gf(self.guided_map(x_lr), y_lr, self.guided_map(x_hr)))


class DeepGuidedFilterGuidedMapConvGFPixelShuffleGCAAtrous(nn.Module):
    def __init__(self, args, radius=1, dilation=0):
        super().__init__()

        self.args = args
        norm = norm_dict[args.norm_layer]

        c = args.guided_map_channels
        if args.guided_map_is_atrous_residual:
            self.guided_map = SmoothDilatedResidualAtrousGuidedBlock(
                in_channel=3, channel_num=c, args=args
            )
        else:
            self.guided_map = nn.Sequential(
                nn.Conv2d(
                    3,
                    c,
                    kernel_size=args.guided_map_kernel_size,
                    padding=args.guided_map_kernel_size // 2,
                    bias=False,
                )
                if dilation == 0
                else nn.Conv2d(
                    3, c, 3, padding=dilation, dilation=dilation, bias=False
                ),
                norm(c),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(
                    c,
                    3,
                    kernel_size=args.guided_map_kernel_size,
                    padding=args.guided_map_kernel_size // 2,
                ),
            )

        if args.model == "guided-filter-pixelshuffle-gca-atrous":
            self.lr = GCANet_atrous(
                in_c=3 * args.pixelshuffle_ratio ** 2,
                out_c=3 * args.pixelshuffle_ratio ** 2,
                args=args,
            )
        elif args.model == "guided-filter-pixelshuffle-gca-atrous-corrected":
            self.lr = GCANet_atrous_corrected(
                in_c=3 * args.pixelshuffle_ratio ** 2,
                out_c=3 * args.pixelshuffle_ratio ** 2,
                args=args,
            )

        elif args.model == "atrous-guided-filter-pixelshuffle-gca-atrous-corrected":
            self.lr = GCANet_atrous_corrected(
                in_c=3 * args.pixelshuffle_ratio ** 2,
                out_c=3 * args.pixelshuffle_ratio ** 2,
                args=args,
            )

        if args.model == "guided-filter-pixelshuffle-gca-atrous":
            self.gf = ConvGuidedFilter(radius, norm=AdaptiveNorm)
        else:
            self.gf = ConvGuidedFilter(radius, norm=norm)

        self.downsample = nn.Upsample(
            scale_factor=0.5, mode="bilinear", align_corners=True
        )

    def init_lr(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        load_state_dict(self.lr, checkpoint["state_dict"])

    def forward(self, x_hr):
        x_lr = self.downsample(x_hr)
        x_lr_unpixelshuffled = unpixel_shuffle(x_lr, 2)
        y_lr = F.pixel_shuffle(self.lr(x_lr_unpixelshuffled), 2)

        return F.tanh(self.gf(self.guided_map(x_lr), y_lr, self.guided_map(x_hr)))


class DeepGuidedFilterGuidedMapConvGFPixelShuffleGCAAtrousStage2(nn.Module):
    def __init__(self, args, radius=1, dilation=0):
        super().__init__()

        self.args = args
        norm = norm_dict[args.norm_layer]

        c = args.guided_map_channels
        self.guided_map = SmoothDilatedResidualAtrousGuidedBlockStage2(
            in_channel=3 * args.num_ensemble, channel_num=c, args=args
        )

        self.lr = GCANet_atrous_corrected(
            in_c=3 * args.num_ensemble * args.pixelshuffle_ratio ** 2,
            out_c=3 * args.pixelshuffle_ratio ** 2,
            args=args,
        )

        self.gf = ConvGuidedFilter(radius, norm=norm)

        self.downsample = nn.Upsample(
            scale_factor=0.5, mode="bilinear", align_corners=True
        )

    def init_lr(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        load_state_dict(self.lr, checkpoint["state_dict"])

    def forward(self, x_hr):
        x_lr = self.downsample(x_hr)
        x_lr_unpixelshuffled = unpixel_shuffle(x_lr, 2)
        y_lr = F.pixel_shuffle(self.lr(x_lr_unpixelshuffled), 2)

        return F.tanh(self.gf(self.guided_map(x_lr), y_lr, self.guided_map(x_hr)))


@ex.automain
def main(_run):
    from utils.tupperware import tupperware
    from torchsummary import summary
    from utils.model_serialization import load_state_dict

    args = tupperware(_run.config)
    model = DeepGuidedFilterGuidedMapConvGFPixelShuffleGCAAtrousStage2(args)
    # ckpt = torch.load("/Users/Ankivarun/Downloads/model_latest.pth", map_location="cpu")
    # model_state_dict = ckpt["state_dict"]
    # model_state_dict.pop("module.lr.gate.bias", None)
    # model_state_dict.pop("module.lr.gate.weight", None)
    # load_state_dict(model, model_state_dict)

    summary(model, (12, 512, 1024))

    # ckpt = torch.load(
    #     args.ckpt_dir / args.exp_name / "model_latest.pth", map_location="cpu"
    # )
    # load_state_dict(model, ckpt["state_dict"])
    #
    # lr_state = {"state_dict": model.lr.state_dict()}
    # torch.save(lr_state, args.ckpt_dir / args.exp_name / "lr_latest.pth")
    # summary(model, (3, 1024, 2048))
