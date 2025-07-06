import torch
import torch.nn as nn
from tqdm import tqdm

device = torch.device("cuda")


def conv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True),
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True),
        nn.LeakyReLU(0.2, True)
    )


def downconv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True),
        nn.LeakyReLU(0.2, True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        conv(in_planes, out_planes * 4),  # conv before
        nn.PixelShuffle(2),
        nn.LeakyReLU(0.2, True),
        conv(out_planes, out_planes)  # conv after
    )


# 减少计算量的版本
# def upconv(in_planes, out_planes):
#     return nn.Sequential(
#         conv(in_planes, out_planes)
#         nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#         conv(out_planes, out_planes)  # conv after
#     )


def tail_upconv(in_planes, out_planes):
    return nn.Sequential(
        conv(in_planes, out_planes * 4),
        nn.PixelShuffle(2),
        # nn.Sigmoid()  # 使用后全局变白, 不使用完全可以
    )


# 减少计算量的版本
# def tail_upconv(in_planes, out_planes):
#     return nn.Sequential(
#         nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#         nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)
#     )


# 简单UNet, 可能更快更简单的结构就够了
class UNet(nn.Module):
    def __init__(self, in_planes=7, out_planes=1, c=32):
        super(UNet, self).__init__()
        self.down0 = downconv(in_planes, c)  # 1/2
        self.down1 = downconv(c, c * 2)  # 1/4
        self.down2 = downconv(c * 2, c * 4)  # 1/8

        self.up2 = upconv(c * 4, c * 2)  # 1/4
        self.up1 = upconv(c * (2 + 2), c * 2)  # 1/2
        self.up_tail = tail_upconv(c * (2 + 1), out_planes)  # 1/1

    def forward(self, x):
        x0 = self.down0(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)

        x2 = self.up2(x2)
        x1 = self.up1(torch.cat((x2, x1), 1))
        x0 = self.up_tail(torch.cat((x1, x0), 1))

        return x0


if __name__ == '__main__':
    net = UNet()
    # net.load_state_dict(torch.load("mask.pkl"))
    net.eval().cuda().half()


    img0 = torch.rand((1, 3, 544, 960)).cuda().half()
    img1 = torch.rand((1, 3, 544, 960)).cuda().half()
    diff = ((img1 - img0) ** 2).mean(1, keepdim=True)
    diff = diff / (diff.max() + 1e-6)  # normalize

    inp = torch.cat((img0, diff, img1), 1)
    mask = net(inp)
    print(mask.shape)  # B, 1, H, W

    # apply to bidirectional optical flow
    # flow_mask = mask.repeat(1, 2, 1, 1) > 0.5
    # flow01[flow_mask] = 0
    # flow10[flow_mask] = 0

    # speed test
    pbar = tqdm(total=1000)
    for i in range(1000):
        with torch.no_grad():
            res = net(inp)
            pbar.update(1)
