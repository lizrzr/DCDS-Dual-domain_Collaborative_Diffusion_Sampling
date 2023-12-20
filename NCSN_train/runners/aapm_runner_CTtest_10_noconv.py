import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from ..models.cond_refinenet_dilated_noconv import CondRefineNetDilated
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from .multiCTmain import FanBeam
import time
import scipy.io as scio
from scipy.ndimage import zoom
from scipy.optimize import minimize

plt.ion()
savepath = './result/'
__all__ = ['Aapm_Runner_CTtest_10_noconv']


def matrix1_0(row, col, num):
    matrix = np.zeros((row, col), dtype=int)
    matrix[::num, :] = 1
    return matrix


class Aapm_Runner_CTtest_10_noconv():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def write_images(self, x, image_save_path):
        x = np.array(x, dtype=np.uint8)
        cv2.imwrite(image_save_path, x)

    def test(self):
        # 2023.03.06 lizrzr
        AAPM = scio.loadmat('aapm.mat')['save_data']
        # aaa = scio.loadmat('000420.mat')['inputs']

        slice_img = AAPM[:, :, 420]

        slice_img = scio.loadmat('xiaoqiu1.mat')['aaa']
        # slice_img = slice_img / slice_img.max()
        np.save('slice_img160.npy', slice_img)
        plt.imshow(slice_img, cmap='gray')
        plt.show()
        print(slice_img.shape)
        fanBeam = FanBeam()
        # bbb = fanBeam.FP(aaa, ang_num=580)
        # savemat('./result/' + 'input420.mat', {'sinofromimg': bbb})
        PROJS = fanBeam.LACTFP(slice_img, ang_num=90)
        plt.imshow(PROJS, cmap='gray')
        plt.show()
        # PROJS = zoom(PROJS, (4, 1), order=0)
        sparse_slice = fanBeam.LACTFBP(PROJS, 90)
        savemat('./result/' + '90FBP.mat', {'x_rec': sparse_slice})
        plt.imshow(sparse_slice, cmap='gray')
        plt.show()
        PROJS240 = fanBeam.FP(slice_img, ang_num=240)
        PROJS580 = fanBeam.FP(slice_img, ang_num=580)
        savemat('PROJS.mat', {'PROJS580': PROJS580})
        # slice_noise = 0.5*slice_img/slice_img.max() + 0.5*np.random.normal(0,1, size=sparse_slice.shape)
        # plt.imshow(slice_noise, cmap='gray')
        # plt.show()
        # image3d = sparse_AAPM
        states_sino = torch.load(os.path.join(self.args.log, 'sinogram\checkpoint_100000.pth'),
                                 map_location=self.config.device)
        scorenet_sino = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet_sino = torch.nn.DataParallel(scorenet_sino, device_ids=[1])
        scorenet_sino.load_state_dict(states_sino[0])
        scorenet_sino.eval()

        # state_image = torch.load(os.path.join(self.args.log, 'image\checkpoint_100000.pth'),
        #                          map_location=self.config.device)
        state_image = torch.load(os.path.join(self.args.log, 'xiaoqiu\checkpoint_62500.pth'),
                                 map_location=self.config.device)
        scorenet_image = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet_image = torch.nn.DataParallel(scorenet_image, device_ids=[1])
        scorenet_image.load_state_dict(state_image[0])
        scorenet_image.eval()

        x_img = sparse_slice
        x_sino = PROJS

        maxdegrade_img = x_img.max()
        maxdegrade_sino = x_sino.max()

        x0_img = nn.Parameter(torch.Tensor(np.zeros([1, 10, 512, 512])).uniform_(-1, 1))
        x0_sino_60 = nn.Parameter(torch.Tensor(np.zeros([1, 10, 60, 240])).uniform_(-1, 1))
        x0_sino_120 = nn.Parameter(torch.Tensor(np.zeros([1, 10, 120, 240])).uniform_(-1, 1))
        x0_sino_240 = nn.Parameter(torch.Tensor(np.zeros([1, 10, 240, 240])).uniform_(-1, 1))
        x0_sino_480 = nn.Parameter(torch.Tensor(np.zeros([1, 10, 480, 240])).uniform_(-1, 1))
        x0_sino_580 = nn.Parameter(torch.Tensor(np.zeros([1, 10, 580, 580])).uniform_(-1, 1))

        x01_img = x0_img.cuda()
        x01_sino_60 = x0_sino_60.cuda()
        x01_sino_120 = x0_sino_120.cuda()
        x01_sino_240 = x0_sino_240.cuda()
        x01_sino_480 = x0_sino_480.cuda()
        x01_sino_580 = x0_sino_580.cuda()

        step_lr = 0.6 * 0.00003
        sigmas = np.exp(np.linspace(np.log(1), np.log(0.01), 12))
        n_steps_each = 150
        max_psnr = 0
        max_ssim = 0
        min_hfen = 100
        start_start = time.time()
        # idx从0计算
        for idx, sigma in enumerate(sigmas):
            start_out = time.time()
            print(idx)
            lambda_recon = 1. / sigma ** 2
            # labels是一个数
            labels = torch.ones(1, device=x0_img.device) * idx
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            print('sigma = {}'.format(sigma))
            for step in range(n_steps_each):
                start_in = time.time()
                noise1_img = torch.rand_like(x0_img).cpu().detach() * np.sqrt(step_size * 2)
                noise1_sino_60 = torch.rand_like(torch.Tensor(x0_sino_60)).cpu().detach() * np.sqrt(step_size * 2)
                noise1_sino_120 = torch.rand_like(x0_sino_120).cpu().detach() * np.sqrt(step_size * 2)
                noise1_sino_240 = torch.rand_like(x0_sino_240).cpu().detach() * np.sqrt(step_size * 2)
                noise1_sino_480 = torch.rand_like(x0_sino_480).cpu().detach() * np.sqrt(step_size * 2)
                noise1_sino_580 = torch.rand_like(x0_sino_580).cpu().detach() * np.sqrt(step_size * 2)

                grad1_img = np.zeros([1, 10, 512, 512])
                grad1_sino_60 = np.zeros([1, 10, 60, 240])
                grad1_sino_120 = np.zeros([1, 10, 120, 240])
                grad1_sino_240 = np.zeros([1, 10, 240, 240])
                grad1_sino_480 = np.zeros([1, 10, 480, 240])
                grad1_sino_580 = np.zeros([1, 10, 580, 580])

                grad1_img = torch.from_numpy(grad1_img)
                grad1_sino_60 = torch.from_numpy(grad1_sino_60)
                grad1_sino_120 = torch.from_numpy(grad1_sino_120)
                grad1_sino_240 = torch.from_numpy(grad1_sino_240)
                grad1_sino_480 = torch.from_numpy(grad1_sino_480)
                grad1_sino_580 = torch.from_numpy(grad1_sino_580)

                with torch.no_grad():
                    grad1_img = scorenet_image(x01_img, labels).detach()

                # x0非cuda,noise1非cuda
                x0_img = x0_img + step_size * grad1_img.cpu()
                x01_img = x0_img + noise1_img
                x01_img = torch.tensor(x01_img.cuda(), dtype=torch.float32)
                x0_img = np.array(x0_img.cpu().detach(), dtype=np.float32)
                x1_img = np.squeeze(x0_img)
                # x1是512x512的矩阵
                x1_img = np.mean(x1_img, axis=0)
                x1max_img = x1_img * maxdegrade_img

                #                x1max_img = np.transpose(x1max_img, (1, 2, 0))
                print(x1max_img.max())
                sum_diff = x_img - x1max_img

                x_new_img = 0.5 * fanBeam.LACTSIRT(VOL=x_img.copy(),
                                               proj=PROJS, ang_num=90, iter_num=20) + 0.5 * (
                                    x_img - sum_diff)

                x_img = x_new_img
                if step == n_steps_each - 1:
                    x_sino_mid = fanBeam.FP(img=x_img.copy(), ang_num=580)
                    plt.title(step, fontsize=30)
                    plt.imshow(x_img, cmap='gray')
                    plt.show()
                    savemat('./result/' + str(idx) + 'image.mat', {'x_rec': x_new_img})
                    savemat('./result/' + str(idx) + 'sino.mat', {'y_sino': x_sino_mid})
                ###############################sino_model#############################################
                sino_from_img = fanBeam.FP(img=x_img.copy(), ang_num=580)
                with torch.no_grad():
                    grad1_sino_580 = scorenet_sino(x01_sino_580, labels).detach()
                x0_sino_580 = x0_sino_580 + step_size * grad1_sino_580.cpu()
                x01_sino_580 = x0_sino_580 + noise1_sino_580
                x01_sino_580 = torch.tensor(x01_sino_580.cuda(), dtype=torch.float32)
                x0_sino_580 = np.array(x0_sino_580.cpu().detach(), dtype=np.float32)
                x1_sino_580 = np.squeeze(x0_sino_580)
                x1_sino_580 = np.mean(x1_sino_580, axis=0)
                x1max_sino_580 = x1_sino_580 * maxdegrade_sino
                x_new_sino = PROJS580 * matrix1_0(580, 580, 20) + [
                    (1 - ((step + 100 * idx) / 1200)) * sino_from_img * 0.2 + 0.2 * ((step + 100 * idx) / 1200) * (
                        x1max_sino_580)] * (1 - matrix1_0(580, 580, 20)) + 0.8 * sino_from_img * (
                                     1 - matrix1_0(580, 580, 20))
                # x_new_sino = PROJS580 * matrix1_0(580, 580, 20) + [
                #     (1 - ((step + 100 * idx) / 1200)) * sino_from_img * 0 + 0* ((step + 100 * idx) / 1200) * (
                #         x1max_sino_580)] * (1 - matrix1_0(580, 580, 20)) + 1* sino_from_img * (
                #                      1 - matrix1_0(580, 580, 20))
                x_sino = x_new_sino.squeeze()
                x_rec_sino_580 = x_sino.copy()
                x_rec_sino_580 = x_rec_sino_580 / maxdegrade_sino
                x_mid_sino_580 = np.zeros([1, 10, 580, 580], dtype=np.float32)
                x_rec_sino_580 = np.expand_dims(x_rec_sino_580, 0)
                x_mid_1_sino_580 = np.tile(x_rec_sino_580, [10, 1, 1])
                x_mid_sino_580[0, :, :] = x_mid_1_sino_580
                x0_sino_580 = torch.tensor(x_mid_sino_580, dtype=torch.float32)
                ###############################sino_model#############################################
                #        x_img = 0.99 * fanBeam.SIRT(VOL=x_img, proj=zoom(PROJS, (8, 1), order=1), ang_num=240,
                #                                   iter_num=20) + 0.01 * fanBeam.SIRT(VOL=x_img,
                #                                                                     proj=(
                #                                                                              x1max_sino_240) * (
                #                                                                                      1 - matrix1_0(240, 240,
                #                                                                                                    8)),
                #                                                                     ang_num=240,
                #                                                                     iter_num=20)
                if idx > 111 :
                    # x_img = ((step + 100 * idx) / 1200) * fanBeam.SIRT(VOL=x_img, proj=x_new_sino.squeeze(),
                    #                                                    ang_num=580, iter_num=20) + (
                    #                     1 - ((step + 100 * idx) / 1200)) * x_img
                    x_img = fanBeam.SIRT(VOL=x_img, proj=x_new_sino.squeeze(), ang_num=580,
                                          iter_num=20)
                    #x_img = fanBeam.SIRT(VOL=x_img, proj=PROJS, ang_num=29, iter_num=20)
                x_new_sino = x_new_sino.squeeze()


                x_rec_img = x_img.copy()
                y_sino = fanBeam.FP(img=x_rec_img, ang_num=580)
                x_rec_img = x_rec_img / maxdegrade_img

                # if step == n_steps_each - 1:
                #     savemat('./result/' + str(idx) + 'image.mat', {'x_rec': x_rec_img})
                #     savemat('./result/' + str(idx) + 'sino.mat', {'y_sino': y_sino})
                end_in = time.time()
                print("inner loop:%.2fs" % (end_in - start_in))
                print("current {} step".format(step))
                x_mid_img = np.zeros([1, 10, 512, 512], dtype=np.float32)
                # clip，强行截取上下限操作， 小于0的置为0，大于1的置为1
                x_rec_img = np.clip(x_rec_img, 0, 1)
                x_rec_img = np.expand_dims(x_rec_img, 0)
                x_mid_1_img = np.tile(x_rec_img, [10, 1, 1])
                x_mid_img[0, :, :] = x_mid_1_img
                x0_img = torch.tensor(x_mid_img, dtype=torch.float32)

            end_out = time.time()
            print("outer iter:%.2fs" % (end_out - start_out))

        plt.ioff()
        end_end = time.time()
        print("PSNR:%.2f" % (max_psnr), "SSIM:%.2f" % (max_ssim))
        print("total time:%.2fs" % (end_end - start_start))
