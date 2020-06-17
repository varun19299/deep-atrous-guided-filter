import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    """
    custom weights initialization called on netG and netD
    https://github.com/pytorch/examples/blob/master/dcgan/main.py
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

####################################################################################################################


class make_dense(nn.Module):
    def __init__(self, nChannels, nChannels_, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels_, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)
        self.nChannels = nChannels

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

class make_residual_dense_ver1(nn.Module):
    def __init__(self, nChannels, nChannels_, growthRate, kernel_size=3):
        super(make_residual_dense_ver1, self).__init__()
        self.conv = nn.Conv2d(nChannels_, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)
        self.nChannels_ = nChannels_
        self.nChannels = nChannels
        self.growthrate = growthRate

    def forward(self, x):
        # print('1', x.shape, self.nChannels, self.nChannels_, self.growthrate)
        # print('2', outoflayer.shape)
        # print('3', out.shape, outoflayer.shape)
        # print('4', out.shape)

        outoflayer = F.relu(self.conv(x))
        out = torch.cat((x[:, :self.nChannels, :, :] + outoflayer, x[:, self.nChannels:, :, :]), 1)
        out = torch.cat((out, outoflayer), 1)
        return out

class make_residual_dense_ver2(nn.Module):
    def __init__(self, nChannels, nChannels_, growthRate, kernel_size=3):
        super(make_residual_dense_ver2, self).__init__()
        if nChannels == nChannels_ :
            self.conv = nn.Conv2d(nChannels_, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                                  bias=False)
        else:
            self.conv = nn.Conv2d(nChannels_ + growthRate, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                                  bias=False)

        self.nChannels_ = nChannels_
        self.nChannels = nChannels
        self.growthrate = growthRate

    def forward(self, x):
        # print('1', x.shape, self.nChannels, self.nChannels_, self.growthrate)
        # print('2', outoflayer.shape)
        # print('3', out.shape, outoflayer.shape)
        # print('4', out.shape)

        outoflayer = F.relu(self.conv(x))
        if x.shape[1] == self.nChannels:
            out = torch.cat((x, x + outoflayer), 1)
        else:
            out = torch.cat((x[:, :self.nChannels, :, :], x[:, self.nChannels:self.nChannels + self.growthrate, :, :] + outoflayer, x[:, self.nChannels + self.growthrate:, :, :]), 1)
        out = torch.cat((out, outoflayer), 1)
        return out

class make_dense_LReLU(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense_LReLU, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Residual dense block (RDB) architecture
class RDB(nn.Module):
    """
    https://github.com/lizhengwei1992/ResidualDenseNetwork-Pytorch
    """

    def __init__(self, nChannels, nDenselayer, growthRate):
        """
        :param nChannels: input feature 의 channel 수
        :param nDenselayer: RDB(residual dense block) 에서 Conv 의 개수
        :param growthRate: Conv 의 output layer 의 수
        """
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels, nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)

        ###################kingrdb ver2##############################################
        # self.conv_1x1 = nn.Conv2d(nChannels_ + growthRate, nChannels, kernel_size=1, padding=0, bias=False)
        ###################else######################################################
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        # local residual 구조
        out = out + x
        return out

def RDB_Blocks(channels, size):
    bundle = []
    for i in range(size):
        bundle.append(RDB(channels, nDenselayer=8, growthRate=64))  # RDB(input channels,
    return nn.Sequential(*bundle)

####################################################################################################################
# Group of Residual dense block (GRDB) architecture
class GRDB(nn.Module):
    """
    https://github.com/lizhengwei1992/ResidualDenseNetwork-Pytorch
    """

    def __init__(self, numofkernels, nDenselayer, growthRate, numforrg):
        """
        :param nChannels: input feature 의 channel 수
        :param nDenselayer: RDB(residual dense block) 에서 Conv 의 개수
        :param growthRate: Conv 의 output layer 의 수
        """
        super(GRDB, self).__init__()

        modules = []
        for i in range(numforrg):
            modules.append(RDB(numofkernels, nDenselayer=nDenselayer, growthRate=growthRate))
        self.rdbs = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(numofkernels * numforrg, numofkernels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = x
        outputlist = []
        for rdb in self.rdbs:
            output = rdb(out)
            outputlist.append(output)
            out = output
        concat = torch.cat(outputlist, 1)
        out = x + self.conv_1x1(concat)
        return out

# Group of group of Residual dense block (GRDB) architecture
class GGRDB(nn.Module):
    """
    https://github.com/lizhengwei1992/ResidualDenseNetwork-Pytorch
    """

    def __init__(self, numofmodules, numofkernels, nDenselayer, growthRate, numforrg):
        """
        :param nChannels: input feature 의 channel 수
        :param nDenselayer: RDB(residual dense block) 에서 Conv 의 개수
        :param growthRate: Conv 의 output layer 의 수
        """
        super(GGRDB, self).__init__()

        modules = []
        for i in range(numofmodules):
            modules.append(GRDB(numofkernels, nDenselayer=nDenselayer, growthRate=growthRate, numforrg=numforrg))
        self.grdbs = nn.Sequential(*modules)

    def forward(self, x):
        output = x
        for grdb in self.grdbs:
            output = grdb(output)

        return x + output

####################################################################################################################


class ResidualBlock(nn.Module):
    """
    one_to_many 논문에서 제시된 resunit 구조
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = self.bn1(x)
        residual = self.relu1(residual)
        residual = self.conv1(residual)
        residual = self.bn2(residual)
        residual = self.relu2(residual)
        residual = self.conv2(residual)
        return x + residual


def ResidualBlocks(channels, size):
    bundle = []
    for i in range(size):
        bundle.append(ResidualBlock(channels))
    return nn.Sequential(*bundle)