from torch.optim import AdamW
import itertools
from MaskNet import *
from loss import *
from IFNet_HDv3 import IFNet

device = torch.device("cuda")


class Model:
    def __init__(self, local_rank=-1):

        self.rife = IFNet()
        self.maskNet = UNet()

        self.optimG = AdamW(itertools.chain(
            self.maskNet.parameters(),
        ), lr=1e-6, weight_decay=1e-2)

        self.scaler = torch.cuda.amp.GradScaler()
        self.device()

    def train(self):
        self.rife.eval()
        self.maskNet.train()

    def eval(self):
        self.rife.eval()
        self.maskNet.eval()

    def device(self):
        self.rife.to(device)
        self.maskNet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }

        # rife4.18是目前为止验证的在精致区域上表现最好的光流, 得益于在结构中存在足量大分辨率上的计算且训练时间足够久
        # 更学术的方案是使用Exploring-Discontinuity-for-VFI中的dataloader生成dmap, 多光流算法生成mask和dmap算L1 Loss做横评
        # 不直接使用pandatimo的方案来制作masknet是因为他生成的dmap并不包含原图像中的静止区域，
        # 导致网络训练出来只能生成与dmap中标注的字幕/矩形方块等简单静止物体，他的方案不适合光流类补帧
        param = convert(torch.load('rife418.pkl'))
        self.rife.load_state_dict(param, False)
        # self.maskNet.load_state_dict("mask.pkl", True)

    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(self.maskNet.state_dict(), '{}/mask.pkl'.format(path))

    def update(self, imgs, gt, learning_rate=0):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        self.train()

        img0, img1 = imgs[:, :3], imgs[:, 3:]

        p = np.random.uniform(0, 1)
        if p < 0.3:
            img0 = gt
        elif p < 0.6:
            img1 = gt

        with torch.no_grad():
            flow = self.rife(torch.cat((img0, img1), 1))[0][-1]

            mask_gt = (abs(flow[:, :1]) < 0.5) & (abs(flow[:, 1:2]) < 0.5) & (abs(flow[:, 2:3]) < 0.5) & (
                    abs(flow[:, 3:4]) < 0.5)

            mask_gt = mask_gt.float()

        with torch.autocast("cuda"):
            diff = ((img1 - img0) ** 2).mean(1, keepdim=True)
            diff = diff / (diff.max() + 1e-6)  # normalize
            mask_pred = self.maskNet(torch.cat((img0, diff, img1), 1))
            # mask_pred = self.maskNet(torch.cat((img0, img1), 1))

        # bce_loss = F.binary_cross_entropy_with_logits(mask_pred, mask_gt)
        # dice_loss = 1 - (2 * (mask_pred * mask_gt).sum() + 1e-5) / ((mask_pred + mask_gt).sum() + 1e-5)

        loss_l1_global = (mask_pred - mask_gt).abs().mean()  # 只用该loss全黑
        loss_l1_target = (mask_pred - mask_gt)[mask_gt > 0].abs().mean()  # 只用该loss全白

        loss_grad = grad_loss(mask_pred) + grad_loss_2(mask_pred)

        # loss_G = loss_l1_global + loss_l1_target + 1 * loss_grad  # 不建议使用grad loss, 会影响准确度
        loss_G = loss_l1_global + loss_l1_target  # 该loss组合目前收敛最好
        # loss_G = bce_loss + 0.5 * dice_loss  # 不收敛
        self.scaler.scale(loss_G).backward()

        # 不需要梯度裁剪
        # self.scaler.unscale_(self.optimG)
        # torch.nn.utils.clip_grad_norm_(self.maskNet.parameters(), 1.0)

        self.scaler.step(self.optimG)
        self.scaler.update()
        self.optimG.zero_grad()

        return mask_pred, {
            'mask_gt': mask_gt,
            'loss_l1_global': loss_l1_global,
            'loss_l1_target': loss_l1_target,
            'loss_grad': loss_grad,
            'img0': img0,
            "img1": img1,
            'flow': flow,
        }
