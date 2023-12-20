import os

import numpy as np
import scipy.io as scio


def compose(location):
    label = np.empty(shape=(512, 512, 1))
    output = np.empty(shape=(512, 512, 1))
    for i in range(505):
        file = location + str(i + 1).zfill(6) + '.mat'
        data = scio.loadmat(file)['data']
        reference = scio.loadmat(file)['reference']
        data1 = data.squeeze()
        data1 = data1.transpose(1,2,0)
        reference1 = reference.squeeze()
        reference1 = reference1.transpose(1,2,0)

        file = location + str(i + 2).zfill(6) + '.mat'
        data = scio.loadmat(file)['data']
        reference = scio.loadmat(file)['reference']
        data2 = data.squeeze()
        data2 = data2.transpose(1,2,0)
        reference2 = reference.squeeze()
        reference2 = reference2.transpose(1,2,0)

        file = location + str(i + 3).zfill(6) + '.mat'
        data = scio.loadmat(file)['data']
        reference = scio.loadmat(file)['reference']
        data3 = data.squeeze()
        data3 = data3.transpose(1,2,0)
        reference3 = reference.squeeze()
        reference3 = reference3.transpose(1,2,0)

        file = location + str(i + 4).zfill(6) + '.mat'
        data = scio.loadmat(file)['data']
        reference = scio.loadmat(file)['reference']
        data4 = data.squeeze()
        data4 = data4.transpose(1,2,0)
        reference4 = reference.squeeze()
        reference4 = reference4.transpose(1,2,0)

        output = np.append(output, (data1[:, :, 3].reshape(512,512,1) +
                                    data2[:, :, 2].reshape(512,512,1) + data3[:, :, 1].reshape(512,512,1) +
                                    data4[:, :, 0].reshape(512,512,1)) / 4, axis=2)
        label = np.append(label, (reference1[:, :, 3].reshape(512,512,1) +
                                  reference2[:, :, 2].reshape(512,512,1) + reference3[:, :, 1].reshape(512,512,1) +
                                  reference4[:, :, 0].reshape(512,512,1)) / 4, axis=2)
    return output

if __name__ == '__main__':
    rec_path = os.getcwd()
    location = rec_path + '/result/20221205-170655/10/'

    output = compose(location)