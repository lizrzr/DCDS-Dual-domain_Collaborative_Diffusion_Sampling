import numpy as np
import os
import scipy.io as scio


# volume:输入3d体数据:numpy
# identifier:文件编号(xxxx-xxxxxx)
# frame_numstart:起始帧数
# frame_numend:结束帧数
# count:每帧数据的数量

def datamaking_test(volume, identifier, frame_numstart, frame_numend, count=509):
    rec_path = os.getcwd()
    for num in range(frame_numstart, frame_numend + 1):
        try:
            os.mkdir(rec_path + './' + '/test/' + str(identifier) + '/')
        except OSError as error:
            print(error)
        try:
            os.mkdir(rec_path + './' + '/test/' + str(identifier) + '/' + str(num) + '/')
            rec_data = volume
            rec_label = volume
            # rec_dic = scio.loadmat(
            #    rec_path + './' + '/result/' + str(identifier) + 'FDK/FDK_frames_' + str(num) + '/rec_FDK.mat')
            for i in range(count):
                data = rec_data[:, :, i:i + 4]
                label = rec_label[:, :, i:i + 4]
                scio.savemat(
                    rec_path + './' + '/test/' + str(identifier) + '/' + str(num) + '/' + str(i + 1).zfill(6) + '.mat',
                    {'data': data}, {'label': label})
            print('第' + str(num) + '帧测试数据集生成完成！')
        except OSError as error:
            print(error)
            print('第' + str(num) + '帧测试数据集已存在')

    return 0
