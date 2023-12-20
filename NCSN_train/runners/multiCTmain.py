import sys
import os
import numpy as np
from numpy import matlib
from .loadData import loadData
from scipy.interpolate import interpn
from scipy.interpolate import griddata
from scipy.interpolate import interp2d
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d
import cv2
import time
import astra
import scipy.io as sio
from .datamaking import datamaking_test
from torch.utils.data import DataLoader
from .compose import compose
import shutil
from PIL import Image


class FanBeam():
    def __init__(self):
        self.projGeom29 = astra.create_proj_geom('fanflat', 0.35, 580, np.linspace(0, np.pi, 29, endpoint=False), 500,
                                                 500)
        self.projGeom30 = astra.create_proj_geom('fanflat', 0.35, 240, np.linspace(0, np.pi, 30, endpoint=False), 500,
                                                 500)
        self.projGeom60 = astra.create_proj_geom('fanflat', 0.35, 240, np.linspace(0, np.pi, 60, endpoint=False), 500,
                                                 500)
        self.projGeom120 = astra.create_proj_geom('fanflat', 0.35, 240, np.linspace(0, np.pi, 120, endpoint=False), 500,
                                                  500)
        self.projGeom240 = astra.create_proj_geom('fanflat', 0.35, 240, np.linspace(0, np.pi, 240, endpoint=False), 500,
                                                  500)
        self.projGeom480 = astra.create_proj_geom('fanflat', 0.35, 240, np.linspace(0, np.pi, 480, endpoint=False), 500,
                                                  500)
        self.projGeom580 = astra.create_proj_geom('fanflat', 0.35, 580, np.linspace(0, np.pi, 580, endpoint=False), 500,
                                                  500)
        self.volGeom = astra.create_vol_geom(512, 512, (-512 / 2), (512 / 2),
                                             (-512 / 2), (512 / 2))

        self.projGeomLACT90 = astra.create_proj_geom('fanflat', 0.7, 1500,
                                                     np.linspace(0, np.pi / 2+np.pi/12, 480, endpoint=False), 2000, 500)
        self.projGeomLACT60 = astra.create_proj_geom('fanflat', 1.5, 640,
                                                     np.linspace(0, np.pi / 3, 240, endpoint=False), 1500, 500)

    def FP(self, img, ang_num):
        if ang_num == 30:
            projGeom = self.projGeom30
        elif ang_num == 29:
            projGeom = self.projGeom29
        elif ang_num == 60:
            projGeom = self.projGeom60
        elif ang_num == 120:
            projGeom = self.projGeom120
        elif ang_num == 240:
            projGeom = self.projGeom240
        elif ang_num == 480:
            projGeom = self.projGeom480
        elif ang_num == 580:
            projGeom = self.projGeom580
        volGeom = self.volGeom
        rec_id = astra.data2d.create('-vol', volGeom, img)
        proj_id = astra.data2d.create('-sino', projGeom)
        cfg = astra.astra_dict('FP_CUDA')
        cfg['VolumeDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        #   cfg['option'] = {'VoxelSuperSampling': 2}
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        rec = astra.data2d.get(rec_id).T
        pro = astra.data2d.get(proj_id)
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(proj_id)

        return pro

    def FBP(self, proj, ang_num):
        if ang_num == 30:
            projGeom = self.projGeom30
        elif ang_num == 29:
            projGeom = self.projGeom29
        elif ang_num == 60:
            projGeom = self.projGeom60
        elif ang_num == 120:
            projGeom = self.projGeom120
        elif ang_num == 240:
            projGeom = self.projGeom240
        elif ang_num == 480:
            projGeom = self.projGeom480
        elif ang_num == 580:
            projGeom = self.projGeom580
        volGeom = self.volGeom
        rec_id = astra.data2d.create('-vol', volGeom)
        proj_id = astra.data2d.create('-sino', projGeom, proj)
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        #   cfg['option'] = {'VoxelSuperSampling': 2}
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        rec = astra.data2d.get(rec_id)
        pro = astra.data2d.get(proj_id)
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(proj_id)

        return rec

    def SIRT(self, VOL, proj, ang_num, iter_num):
        if ang_num == 30:
            projGeom = self.projGeom30
        elif ang_num == 29:
            projGeom = self.projGeom29
        elif ang_num == 60:
            projGeom = self.projGeom60
        elif ang_num == 120:
            projGeom = self.projGeom120
        elif ang_num == 240:
            projGeom = self.projGeom240
        elif ang_num == 480:
            projGeom = self.projGeom480
        elif ang_num == 580:
            projGeom = self.projGeom580
        volGeom = self.volGeom
        if VOL is None:
            rec_id = astra.data2d.create('-vol', volGeom)
        else:
            rec_id = astra.data2d.create('-vol', volGeom, VOL)
        proj_id = astra.data2d.create('-sino', projGeom, proj)
        cfg = astra.astra_dict('SIRT_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        #   cfg['option'] = {'VoxelSuperSampling': 2}
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, iter_num)
        rec = astra.data2d.get(rec_id)
        pro = astra.data2d.get(proj_id)
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(proj_id)

        return rec

    def EM(self, VOL, proj, ang_num, iter_num):
        if ang_num == 30:
            projGeom = self.projGeom30
        elif ang_num == 29:
            projGeom = self.projGeom29
        elif ang_num == 60:
            projGeom = self.projGeom60
        elif ang_num == 120:
            projGeom = self.projGeom120
        elif ang_num == 240:
            projGeom = self.projGeom240
        elif ang_num == 480:
            projGeom = self.projGeom480
        elif ang_num == 580:
            projGeom = self.projGeom580
        volGeom = self.volGeom
        if VOL is None:
            rec_id = astra.data2d.create('-vol', volGeom)
        else:
            rec_id = astra.data2d.create('-vol', volGeom, VOL)
        proj_id = astra.data2d.create('-sino', projGeom, proj)
        cfg = astra.astra_dict('EM_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        #   cfg['option'] = {'VoxelSuperSampling': 2}
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, iter_num)
        rec = astra.data2d.get(rec_id)
        pro = astra.data2d.get(proj_id)
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(proj_id)

        return rec

    def LACTFP(self, img, ang_num):
        if ang_num == 60:
            projGeom = self.projGeomLACT60
        elif ang_num == 90:
            projGeom = self.projGeomLACT90
        volGeom = self.volGeom
        rec_id = astra.data2d.create('-vol', volGeom, img)
        proj_id = astra.data2d.create('-sino', projGeom)
        cfg = astra.astra_dict('FP_CUDA')
        cfg['VolumeDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        #   cfg['option'] = {'VoxelSuperSampling': 2}
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        rec = astra.data2d.get(rec_id).T
        pro = astra.data2d.get(proj_id)
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(proj_id)

        return pro

    def LACTSIRT(self, VOL, proj, ang_num, iter_num):
        if ang_num == 60:
            projGeom = self.projGeomLACT60
        elif ang_num == 90:
            projGeom = self.projGeomLACT90
        volGeom = self.volGeom
        if VOL is None:
            rec_id = astra.data2d.create('-vol', volGeom)
        else:
            rec_id = astra.data2d.create('-vol', volGeom, VOL)
        proj_id = astra.data2d.create('-sino', projGeom, proj)
        cfg = astra.astra_dict('SIRT_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        #   cfg['option'] = {'VoxelSuperSampling': 2}
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, iter_num)
        rec = astra.data2d.get(rec_id)
        pro = astra.data2d.get(proj_id)
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(proj_id)

        return rec

    def LACTFBP(self, proj, ang_num):
        if ang_num == 60:
            projGeom = self.projGeomLACT60
        elif ang_num == 90:
            projGeom = self.projGeomLACT90
        volGeom = self.volGeom
        rec_id = astra.data2d.create('-vol', volGeom)
        proj_id = astra.data2d.create('-sino', projGeom, proj)
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        #   cfg['option'] = {'VoxelSuperSampling': 2}
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        rec = astra.data2d.get(rec_id)
        pro = astra.data2d.get(proj_id)
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(proj_id)

        return rec