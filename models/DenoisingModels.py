from models.subNets import *
from models.cbam import *
import torch


class ntire_rdb_gd_rir_ver1(nn.Module):
    def __init__(
        self, input_channel, numforrg=4, numofrdb=16, numofconv=8, numoffilters=64, t=1
    ):
        super(ntire_rdb_gd_rir_ver1, self).__init__()

        self.numforrg = numforrg  # num of rdb units in one residual group
        self.numofrdb = numofrdb  # num of all rdb units
        self.nDenselayer = numofconv
        self.numofkernels = numoffilters
        self.t = t

        self.layer1 = nn.Conv2d(
            input_channel, self.numofkernels, kernel_size=3, stride=1, padding=1
        )
        # self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(
            self.numofkernels, self.numofkernels, kernel_size=4, stride=2, padding=1
        )

        modules = []
        for i in range(self.numofrdb // self.numforrg):
            modules.append(
                GRDB(
                    self.numofkernels,
                    self.nDenselayer,
                    self.numofkernels,
                    self.numforrg,
                )
            )
        self.rglayer = nn.Sequential(*modules)

        self.layer7 = nn.ConvTranspose2d(
            self.numofkernels, self.numofkernels, kernel_size=4, stride=2, padding=1
        )

        # self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(
            self.numofkernels, input_channel, kernel_size=3, stride=1, padding=1
        )
        self.cbam = CBAM(self.numofkernels, 16)

    def forward(self, x):
        out = self.layer1(x)
        # out = self.layer2(out)
        out = self.layer3(out)

        # out = self.rglayer(out)
        for grdb in self.rglayer:
            for i in range(self.t):
                out = grdb(out)

        out = self.layer7(out)
        out = self.cbam(out)

        # out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return out + x


class ntire_rdb_gd_rir_ver2(nn.Module):
    def __init__(
        self,
        input_channel=3,
        numofmodules=2,
        numforrg=4,
        numofrdb=16,
        numofconv=8,
        numoffilters=64,
        t=1,
    ):
        super(ntire_rdb_gd_rir_ver2, self).__init__()

        self.numofmodules = numofmodules  # num of modules to make residual
        self.numforrg = numforrg  # num of rdb units in one residual group
        self.numofrdb = numofrdb  # num of all rdb units
        self.nDenselayer = numofconv
        self.numofkernels = numoffilters
        self.t = t

        self.layer1 = nn.Conv2d(
            input_channel, self.numofkernels, kernel_size=3, stride=1, padding=1
        )
        # self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(
            self.numofkernels, self.numofkernels, kernel_size=4, stride=2, padding=1
        )

        modules = []
        for i in range(self.numofrdb // (self.numofmodules * self.numforrg)):
            modules.append(
                GGRDB(
                    self.numofmodules,
                    self.numofkernels,
                    self.nDenselayer,
                    self.numofkernels,
                    self.numforrg,
                )
            )
        for i in range(
            (self.numofrdb % (self.numofmodules * self.numforrg)) // self.numforrg
        ):
            modules.append(
                GRDB(
                    self.numofkernels,
                    self.nDenselayer,
                    self.numofkernels,
                    self.numforrg,
                )
            )
        self.rglayer = nn.Sequential(*modules)

        self.layer7 = nn.ConvTranspose2d(
            self.numofkernels, self.numofkernels, kernel_size=4, stride=2, padding=1
        )

        # self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(
            self.numofkernels, input_channel, kernel_size=3, stride=1, padding=1
        )
        self.cbam = CBAM(numoffilters, 16)

    def forward(self, x):
        out = self.layer1(x)
        # out = self.layer2(out)
        out = self.layer3(out)

        for grdb in self.rglayer:
            for i in range(self.t):
                out = grdb(out)

        out = self.layer7(out)
        out = self.cbam(out)

        # out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return out + x


class Generator_one2many_gd_rir_old(nn.Module):
    def __init__(
        self, input_channel, numforrg=4, numofrdb=16, numofconv=8, numoffilters=64
    ):
        super(Generator_one2many_gd_rir_old, self).__init__()

        self.numforrg = numforrg  # num of rdb units in one residual group
        self.numofrdb = numofrdb  # num of all rdb units
        self.nDenselayer = numofconv
        self.numofkernels = numoffilters

        self.layer1 = nn.Conv2d(
            input_channel, self.numofkernels, kernel_size=3, stride=1, padding=1
        )
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(
            self.numofkernels, self.numofkernels, kernel_size=4, stride=2, padding=1
        )

        modules = []
        for i in range(self.numofrdb // self.numforrg):
            modules.append(
                GRDB(
                    self.numofkernels,
                    self.nDenselayer,
                    self.numofkernels,
                    self.numforrg,
                )
            )
        self.rglayer = nn.Sequential(*modules)

        self.layer7 = nn.ConvTranspose2d(
            self.numofkernels, self.numofkernels, kernel_size=4, stride=2, padding=1
        )
        self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(
            self.numofkernels, input_channel, kernel_size=3, stride=1, padding=1
        )
        self.cbam = CBAM(self.numofkernels, 16)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.rglayer(out)

        out = self.layer7(out)
        out = self.cbam(out)
        out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return out + x


if __name__ == "__main__":
    from torchsummary import summary

    net = ntire_rdb_gd_rir_ver2(input_channel=12, numofrdb=12)

    summary(net, (12, 128, 256))
