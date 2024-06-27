import torch.nn as nn
import torch
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

affine_par = True
in_place = True


class VAE(nn.Module):
    def __init__(self, in_channels=344, in_dim=(16, 16), out_dim=(3, 512, 512)):
        super(VAE, self).__init__()
        self.in_channels = in_channels*2
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modalities = out_dim[0]
        self.encoder_channels = 16
        self.split_dim = int(self.in_channels / 2)

        # self.reshape_dim = (int(self.out_dim[1] / 16), int(self.out_dim[2] / 16), int(self.out_dim[3] / 16))
        # self.linear_in_dim = int(16 * (in_dim[0] / 2) * (in_dim[1] / 2) * (in_dim[2] / 2))

        self.reshape_dim = (int(self.out_dim[1] / 16), int(self.out_dim[2] / 16))
        self.linear_in_dim = int(self.encoder_channels * (in_dim[0] / 2) * (in_dim[1] / 2))
        self.linear_vu_dim = self.encoder_channels * self.reshape_dim[0] * self.reshape_dim[1]

        channels_vup2 = int(self.in_channels / 2)
        channels_vup1 = 192   # int(channels_vup2 / 2)
        channels_vup0 = int(channels_vup1 / 2)
        channels_vup = int(channels_vup0 / 2)

        self.VD = nn.Sequential(
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=in_place),
            nn.Conv2d(in_channels=in_channels, out_channels=self.encoder_channels, stride=2, kernel_size=3, padding=1),
        )

        self.linear_mu_logvar = nn.Sequential(
            nn.Linear(self.linear_in_dim, self.in_channels)
        )

        # TODO VU layer here
        self.linear_vu = nn.Sequential(
            nn.Linear(channels_vup2, self.linear_vu_dim)
        )
        self.norm_relu = nn.Sequential(
            nn.InstanceNorm2d(self.encoder_channels),
            nn.ReLU(inplace=in_place),
        )
        self.VU = self.UpBlock2(self.encoder_channels, channels_vup2)

        self.Vup2 = self.UpBlock2(channels_vup2, channels_vup1)
        self.Vblock2 = nn.Sequential(EncoderBlock(channels_vup1, channels_vup1))

        self.Vup1 = self.UpBlock2(channels_vup1, channels_vup0)
        self.Vblock1 = nn.Sequential(EncoderBlock(channels_vup0, channels_vup0))

        self.Vup0 = self.UpBlock2(channels_vup0, channels_vup)
        self.Vblock0 = nn.Sequential(EncoderBlock(channels_vup, channels_vup))

        self.Vend = self.head(channels_vup, self.modalities)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        # print(mu, std)
        return eps.mul(std).add(mu)

    def encode(self, x):
        x = self.VD(x)
        x = x.view(-1, self.linear_in_dim)
        parameter = self.linear_mu_logvar(x)
        mu = parameter[:, :self.in_channels//2]
        logvar = parameter[:, self.in_channels//2:]
        return mu, logvar

    def decode(self, z):
        y = self.linear_vu(z)
        y = y.view(-1, self.encoder_channels, self.reshape_dim[0], self.reshape_dim[1])
        y = self.norm_relu(y)
        y = self.VU(y)
        y = self.Vup2(y)
        y = self.Vblock2(y)
        y = self.Vup1(y)
        y = self.Vblock1(y)
        y = self.Vup0(y)
        y = self.Vblock0(y)
        dec = self.Vend(y)
        return torch.sigmoid(dec)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        recon_x = self.decode(z)
        return z, recon_x, mu, logvar

    def UpBlock2(self, in_channels, out_channels):
        upblock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=in_place),
        )
        return upblock

    def head(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                      stride=1, padding=0)
        )
        return conv


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        return x * self.sigmoid(y)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel = ChannelAttention(in_planes, ratio)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class EncoderBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, multi_grid=1, attention_ksize=7, downsample=None):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), stride=stride, padding=(1, 1),
                               dilation=dilation * multi_grid, bias=False)
        self.norm1 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=in_place)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=1, padding=(1, 1),
                               dilation=dilation * multi_grid, bias=False)
        self.norm2 = nn.InstanceNorm2d(planes)

        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        # self.eca = ECA(k_size=3)
        # self.cbam = CBAM(planes, ratio=16, kernel_size=attention_ksize)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        return out


if __name__ == "__main__":
    model = VAE(in_channels=640, in_dim=(256//16, 256//16), out_dim=(3, 256, 256))
    model.eval()
    input = torch.rand(2, 640, 16, 16)
    z, recon_x, mu, logvar = model(input)
    print(z.shape, recon_x.shape, mu.shape, logvar.shape)

