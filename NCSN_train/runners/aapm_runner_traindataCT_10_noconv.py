import numpy as np
import tqdm
from ..losses.dsm import anneal_dsm_score_estimation
from ..losses.sliced_sm import anneal_sliced_score_estimation_vr
import torch.nn.functional as F
import logging
import torch
import os
import glob
import scipy.io as scio
import shutil
#import tensorboardX
import torch.optim as optim
from torchvision.datasets import CIFAR10
# from torchvision.datasets import MNIST, CIFAR10, SVHN
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader  # , Subset
# from ..datasets.celeba import CelebA
# from ..datasets import get_dataset, data_transform, inverse_data_transform
from ..models.cond_refinenet_dilated_noconv import CondRefineNetDilated  ###########
from torchvision.utils import save_image, make_grid
from PIL import Image
import matplotlib.pyplot as plt
# from scipy.misc import imread
import random
# import dicom
import pydicom as dicom
from .multiCTmain import FanBeam
import random
# python separate_ImageNet.py --model ncsn --runner AapmRunnerdata_10C --config aapm_10C.yml --doc AapmCT_10C --image_folder output
__all__ = ['AapmRunnerdata_10C']
fanBeam = FanBeam()
randomData = [60, 120, 240]
class trainset_loader(Dataset):
    def __init__(self):
        # glob.glob()函数，将目录下所有跟通配符模式相同的文件放到一个列表中。
        self.files_A = sorted(glob.glob('./train/' + '*.mat'))
    def __getitem__(self, index):
        file_A = self.files_A[index]
        file_B = file_A
        # 读取.mat，原本读完是dic字典型，【‘data’】转换为了narray
        label_data = scio.loadmat(file_B)['label']
        label_data = label_data.astype(np.float32)
        proj = fanBeam.FP(label_data, 580)
        label_data = proj
#        recon = fanBeam.SIRT(proj)
        data_array = (label_data - np.min(label_data)) / (np.max(label_data) - np.min(label_data))  ### 0-1

        data_array = np.expand_dims(data_array, 2)
        # data_array_10 = data_array.repeat([1,1,10],axis=2)
        data_array_10 = np.tile(data_array, (1, 1, 10))
        data_array_10 = data_array_10.transpose((2, 0, 1))

        return data_array_10

    def __len__(self):
        return len(self.files_A)


class testset_loader(Dataset):
    def __init__(self):
        # glob.glob()函数，将目录下所有跟通配符模式相同的文件放到一个列表中。
        # self.file_path = 'input_' + dose
        # self.files_A = sorted(glob.glob(os.path.join(root, 'train', self.file_path, 'data') + '*.mat'))
        self.files_A = sorted(glob.glob('./train/' + '*.mat'))

    def __getitem__(self, index):
        file_A = self.files_A[index]
        file_B = file_A
        # 读取.mat，原本读完是dic字典型，【‘data’】转换为了narray
        label_data = scio.loadmat(file_B)['label']
        label_data = label_data.astype(np.float32)
        data_array = (label_data - np.min(label_data)) / (np.max(label_data) - np.min(label_data))  ### 0-1
        data_array = np.expand_dims(data_array, 2)
        # data_array_10 = data_array.repeat([1,1,10],axis=2)
        data_array_10 = np.tile(data_array, (1, 1, 10))
        data_array_10 = data_array_10.transpose((2, 0, 1))

        return data_array_10

    def __len__(self):
        return len(self.files_A)


class GetCT(Dataset):

    def __init__(self, root, augment=None):
        super().__init__()
        self.data_names = np.array([root + "/" + x for x in os.listdir(root)])
        # self.data_names = random.shuffle(self.data_names)
        # print(self.data_names)
        self.augment = None

    # 获取医学图像数据
    def __getitem__(self, index):
        # x=loadmat(self.data_names[index])['Img2']

        dataCT = dicom.read_file(self.data_names[index])  # 读取CT文件
        data_array = dataCT.pixel_array.astype(np.float32) * dataCT.RescaleSlope + dataCT.RescaleIntercept
        data_array = (data_array - np.min(data_array)) / (np.max(data_array) - np.min(data_array))  ### 0-1
        data_array = np.expand_dims(data_array, 2)
        # data_array_10 = data_array.repeat([1,1,10],axis=2)
        data_array_10 = np.tile(data_array, (1, 1, 10))
        data_array_10 = data_array_10.transpose((2, 0, 1))

        return data_array_10

    def __len__(self):
        # if type(self.data_names) != 'str':
        # self.data_names = str(self.data_names)
        return len(self.data_names)


class AapmRunnerdata_10C():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    # 设置优化器
    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    def logit_transform(self, image, lam=1e-6):
        image = lam + (1 - 2 * lam) * image
        return torch.log(image) - torch.log1p(-image)

    def train(self):
        # 初始图像预变换
        if self.config.data.random_flip is False:
            tran_transform = test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])
        else:
            tran_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
            test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

        if self.config.data.dataset == 'CIFAR10':
            dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                              transform=tran_transform)
            test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10_test'), train=False, download=True,
                                   transform=test_transform)
        elif self.config.data.dataset == "AAPM":
            # dataset = GetCT(root= "/data/wyy/EASEL/EASEL/quarter_1mm/Ltrain",augment=None)
            print("train")
            #
        dataloader = DataLoader(trainset_loader(), batch_size=self.config.training.batch_size, shuffle=True,
                                num_workers=4)  ######
        test_loader = DataLoader(testset_loader(), batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=4, drop_last=True)  ########

        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels
        # 路径
        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)
        # 保存可视化路径
      #  tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)
        # DSM模型训练
        score = CondRefineNetDilated(self.config).to(self.config.device)
        # 多GPU显卡
        score = torch.nn.DataParallel(score)
        # 优化器
        optimizer = self.get_optimizer(score.parameters())
        # 恢复训练
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint_43000.pth'))
            score.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        step = 43000
        # 前向激活函数，sigma_begin: 1 sigma_end: 0.01 #前向计算激活函数 num_classes: 12 #分类数目
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_classes))).float().to(self.config.device)

        for epoch in range(self.config.training.n_epochs):
            for i, X in enumerate(dataloader):
                print(X.shape)
                step += 1
                score.train()
                X = X.to(self.config.device)
                X = X / 256. * 255. + torch.rand_like(X) / 256.
                if self.config.data.logit_transform:
                    X = self.logit_transform(X)
                # 损失函数
                labels = torch.randint(0, len(sigmas), (X.shape[0],), device=X.device)
                if self.config.training.algo == 'dsm':
                    loss = anneal_dsm_score_estimation(score, X, labels, sigmas, self.config.training.anneal_power)
                elif self.config.training.algo == 'ssm':
                    loss = anneal_sliced_score_estimation_vr(score, X, labels, sigmas,
                                                             n_particles=self.config.training.n_particles)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #tb_logger.add_scalar('loss', loss, global_step=step)
                logging.info("step: {}, loss: {}".format(step, loss.item()))

                if step >= self.config.training.n_iters:
                    return 0
                # 每100 step进行评估，每一个采样频率保存模型,每500步保存一次模型
                if step % 100 == 0:
                    score.eval()
                    try:
                        # test_X, test_y = next(test_iter)
                        test_X = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X = next(test_iter)

                    test_X = test_X.to(self.config.device)
                    test_X = test_X / 256. * 255. + torch.rand_like(test_X) / 256.
                    if self.config.data.logit_transform:
                        test_X = self.logit_transform(test_X)

                    test_labels = torch.randint(0, len(sigmas), (test_X.shape[0],), device=test_X.device)

                    with torch.no_grad():
                        test_dsm_loss = anneal_dsm_score_estimation(score, test_X, test_labels, sigmas,
                                                                    self.config.training.anneal_power)

                    #tb_logger.add_scalar('test_dsm_loss', test_dsm_loss, global_step=step)

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))
