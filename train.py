import math
import time
import argparse
from model import Model
from dataset import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from util import *
from torch.utils.data.sampler import RandomSampler

log_path = 'train_log'
device = torch.device("cuda")

def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
    return 3e-4 * mul + 3e-6


def train(model):
    writer = SummaryWriter(log_path + '/train')

    step = 0
    dataset = VimeoDataset('train')
    sampler = RandomSampler(dataset)
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, sampler=sampler,
                            persistent_workers=True)
    args.step_per_epoch = train_data.__len__()
    print('training...')
    time_stamp = time.time()
    for epoch in range(args.epoch):
        for i, data in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            data_gpu, timestep = data
            data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]
            learning_rate = get_learning_rate(step)
            imgs = torch.cat((imgs, imgs.flip(-1)), 0)
            gt = torch.cat((gt, gt.flip(-1)), 0)
            pred, info = model.update(imgs, gt, learning_rate)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step % 20 == 1:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/l1_global', info['loss_l1_global'], step)
                writer.add_scalar('loss/l1_target', info['loss_l1_target'], step)
                writer.add_scalar('loss/grad', info['loss_grad'], step)
            if step % 200 == 1:
                mask = (torch.cat((pred, info['mask_gt']), 3).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(
                    'uint8')

                img0 = (info['img0'].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                img1 = (info['img1'].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')

                flow = info['flow'].permute(0, 2, 3, 1).detach().cpu().numpy()
                for i in range(min(args.batch_size, 5)):
                    imgs = np.concatenate((img0[i], img1[i]), 1)[:, :, ::-1]
                    writer.add_image(str(i) + '/img', imgs, step, dataformats='HWC')
                    writer.add_image(str(i) + '/flow', flow2rgb(flow[i]), step, dataformats='HWC')
                    writer.add_image(str(i) + '/mask', mask[i], step, dataformats='HWC')
                writer.flush()
            if step % 1000 == 1:
                model.save_model(log_path)
            print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_l1:{:.4e}'.format(epoch, i, args.step_per_epoch,
                                                                            data_time_interval, train_time_interval,
                                                                            info['loss_l1_global']))
            step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=44, type=int)  # 实际上20k step收敛的就很好了
    parser.add_argument('--batch_size', default=8, type=int, help='minibatch size')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    args = parser.parse_args()
    model = Model(args.local_rank)

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    def convert(param):
        return {
            k.replace("module.", ""): v
            for k, v in param.items()
            if "module." in k
        }


    model.load_model('log418')  # 从https://github.com/hzwer/Practical-RIFE下载
    train(model)
