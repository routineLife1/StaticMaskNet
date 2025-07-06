import cv2
import os
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class VimeoDataset(Dataset):
    def __init__(self, dataset_name, batch_size=32):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.h = 448
        self.w = 448
        self.load_data()


    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        # 数据集用什么都行, 随便两帧算高分辨率光流, 最好是1080p或者更大的图
        # 建议算完光流得到mask再对mask crop来训练网络

        self.train_data = []
        self.val_data = []
        path = '/atd_12k/train_10k/'
        for d in os.listdir(path):
            img0 = (path + d + '/frame1.jpg')
            img1 = (path + d + '/frame2.jpg')
            img2 = (path + d + '/frame3.jpg')
            self.train_data.append((1, (img0, img1, img2)))
            self.train_data.append((1, (img0, img1, img2)))

        path = '/atd_12k/test_2k_original/'
        for d in os.listdir(path):
            img0 = (path + d + '/frame1.jpg')
            img1 = (path + d + '/frame2.jpg')
            img2 = (path + d + '/frame3.jpg')
            self.train_data.append((1, (img0, img1, img2)))
            self.train_data.append((1, (img0, img1, img2)))

        for l in os.listdir("/vimeo_septuplet/sequences"):
            for l2 in os.listdir(f"/vimeo_septuplet/sequences/{l}"):
                path = f"/vimeo_septuplet/sequences/{l}/{l2}"
                data_tuple = []
                for l3 in sorted(os.listdir(path)):
                    data_tuple.append(f'{path}/{l3}')
                self.train_data.append((0, data_tuple))

        # for l in os.listdir("../SportsSloMo/YOUR_DATA_PATH/SportsSloMo_frames"):
        #     path = '../SportsSloMo/YOUR_DATA_PATH/SportsSloMo_frames/{}/'.format(l)
        #     for i in range(0, len(os.listdir(path)) - 7, 7):
        #         data_tuple = []
        #         for j in range(7):
        #             data_tuple.append('{}frame_{:04d}.jpg'.format(path, i+j))
        #         self.train_data.append((2, data_tuple))
        #
        # with open("data/adobe240fps_folder_{}.txt".format('train')) as f:
        #     data = f.readlines()
        #     for l in data:
        #         l = l.strip('\n')
        #         path = "../Super-SloMo/adobe240fps/extracted/{}".format(l)
        #         interval = 14
        #         for i in range(0, len(os.listdir(path)) - 14, interval):
        #             data_tuple = []
        #             for j in range(14):
        #                 data_tuple.append('{}/{:04d}.jpg'.format(path, i+j+1))
        #             self.train_data.append((2, data_tuple))
        #             self.train_data.append((2, data_tuple))
        #
        #

        if self.dataset_name == 'train':
            self.meta_data = self.train_data
        else:
            self.meta_data = self.val_data
        self.nr_sample = len(self.meta_data)        

    def aug(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1

    def getimg(self, index, training=False):
        data = self.meta_data[index][1]
        datasetid = self.meta_data[index][0]

        if datasetid == 1:
            img0 = cv2.imread(data[0])
            gt = cv2.imread(data[1])
            img1 = cv2.imread(data[2])

            p = np.random.uniform(0, 1)
            if p < 0.3:
                img0 = cv2.resize(img0, (480, 270), interpolation=cv2.INTER_LINEAR)
                img1 = cv2.resize(img1, (480, 270), interpolation=cv2.INTER_LINEAR)
                gt = cv2.resize(gt, (480, 270), interpolation=cv2.INTER_LINEAR)
            elif p < 0.6:
                img0 = cv2.resize(img0, (960, 540), interpolation=cv2.INTER_LINEAR)
                img1 = cv2.resize(img1, (960, 540), interpolation=cv2.INTER_LINEAR)
                gt = cv2.resize(gt, (960, 540), interpolation=cv2.INTER_LINEAR)
            elif p < 0.8:
                img0 = cv2.resize(img0, (1440, 810), interpolation=cv2.INTER_LINEAR)
                img1 = cv2.resize(img1, (1440, 810), interpolation=cv2.INTER_LINEAR)
                gt = cv2.resize(gt, (1440, 810), interpolation=cv2.INTER_LINEAR)
            step = 0.5
        elif datasetid == 2:
            if len(data) == 7:
                ind = [0, 1, 2, 3, 4, 5, 6]
            elif len(data) == 14:
                ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            ind = random.choices(ind, k=3)
            ind.sort()
            while ind[0] == ind[2]:
                ind = random.choices([0, 1, 2, 3, 4, 5, 6], k=3)
                ind.sort()
            img0 = cv2.imread(data[ind[0]])
            gt = cv2.imread(data[ind[1]])
            img1 = cv2.imread(data[ind[2]])
            p = np.random.uniform(0, 1)
            if p < 0.5:
                img0 = cv2.resize(img0, (640, 360), interpolation=cv2.INTER_LINEAR)
                img1 = cv2.resize(img1, (640, 360), interpolation=cv2.INTER_LINEAR)
                gt = cv2.resize(gt, (640, 360), interpolation=cv2.INTER_LINEAR)
            step = (ind[1] - ind[0]) * 1.0 / (ind[2] - ind[0] + 1e-6)
        else:
            ind = [0, 1, 2, 3, 4, 5, 6]
            random.shuffle(ind)
            ind = ind[:3]
            ind.sort()
            img0 = cv2.imread(data[ind[0]])
            gt = cv2.imread(data[ind[1]])
            img1 = cv2.imread(data[ind[2]])
            step = (ind[1] - ind[0]) * 1.0 / (ind[2] - ind[0])
        return img0, gt, img1, step, datasetid
            
    def __getitem__(self, index):
        if self.dataset_name == 'train':
            img0, gt, img1, timestep, datasetid = self.getimg(index, True)
            if np.random.uniform(0, 1) < 0.5 and datasetid == 0:
                p = np.random.choice([1.5, 2.0, 2.5, 4.0])
                h, w = int(256 * p), int(448 * p)
                img0 = cv2.resize(img0, (w, h), interpolation=cv2.INTER_CUBIC)
                img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_CUBIC)
                gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_CUBIC)
            if img0.shape[0] < 448:
                img0 = np.concatenate((img0, img0[:, :, ::-1].copy()), 0)
                img1 = np.concatenate((img1, img1[:, :, ::-1].copy()), 0)
                gt = np.concatenate((gt, gt[:, :, ::-1].copy()), 0)
            while np.abs(img0/255. - img1/255.).mean() < 0.005:
                index = (index + 1) % self.nr_sample
                img0, gt, img1, timestep, datasetid = self.getimg(index, True)
                if np.random.uniform(0, 1) < 0.5 and datasetid == 0:
                    p = np.random.choice([1.5, 2.0])
                    h, w = int(256 * p), int(448 * p)
                    img0 = cv2.resize(img0, (w, h), interpolation=cv2.INTER_CUBIC)
                    img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_CUBIC)
                    gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_CUBIC)
                if img0.shape[0] < 448:
                    img0 = np.concatenate((img0, img0[:, ::-1, :].copy()), 0)
                    img1 = np.concatenate((img1, img1[:, ::-1, :].copy()), 0)
                    gt = np.concatenate((gt, gt[:, ::-1, :].copy()), 0)
            img0, gt, img1 = self.aug(img0, gt, img1, self.h, self.w)
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
            if random.uniform(0, 1) < 0.5:
                tmp = img1
                img1 = img0
                img0 = tmp
                timestep = 1 - timestep
        timestep = torch.tensor(timestep).reshape(1, 1, 1)
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)

        return torch.cat((img0, img1, gt), 0), timestep
    
if __name__ == '__main__':
    ds = DataLoader(VimeoDataset('train'))
    print(f"loaded {ds.__len__()} train data")
