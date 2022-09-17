# import torch
from sklearn import metrics
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


# def MMD(x, y, kernel):
#     """Emprical maximum mean discrepancy. The lower the result
#        the more evidence that distributions are the same.
#
#     Args:
#         x: first sample, distribution P
#         y: second sample, distribution Q
#         kernel: kernel type such as "multiscale" or "rbf"
#     """
#     xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
#     rx = (xx.diag().unsqueeze(0).expand_as(xx))
#     ry = (yy.diag().unsqueeze(0).expand_as(yy))
#
#     dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
#     dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
#     dxy = rx.t() + ry - 2. * zz  # Used for C in (1)
#
#     XX, YY, XY = (torch.zeros(xx.shape).to(device),
#                   torch.zeros(xx.shape).to(device),
#                   torch.zeros(xx.shape).to(device))
#
#     if kernel == "multiscale":
#
#         bandwidth_range = [0.2, 0.5, 0.9, 1.3]
#         for a in bandwidth_range:
#             XX += a ** 2 * (a ** 2 + dxx) ** -1
#             YY += a ** 2 * (a ** 2 + dyy) ** -1
#             XY += a ** 2 * (a ** 2 + dxy) ** -1
#
#     if kernel == "rbf":
#
#         bandwidth_range = [10, 15, 20, 50]
#         for a in bandwidth_range:
#             XX += torch.exp(-0.5 * dxx / a)
#             YY += torch.exp(-0.5 * dyy / a)
#             XY += torch.exp(-0.5 * dxy / a)
#
#     return torch.mean(XX + YY - 2. * XY)


if __name__ == '__main__':
    test_csv_path = os.path.join(r"C:\Users\tyler\Documents\Fall2021\MUSER\GANs-for-Medical-Data", "wesad_test_start_r_peak.csv")
    fake_sample_filepath = os.path.join(r"C:\Users\tyler\Documents\Fall2021\MUSER\Plotting\All-Subjects-WESAD", 'Epoch-145-Loss_D-0.21139076352119446-Loss_G-3.127946615219116-Time-04_05_55.npy')
    base_dir = r"C:\Users\tyler\Documents\Fall2021\MUSER\Plotting\All-Subjects-WESAD"

    df = pd.read_csv(test_csv_path)
    test_data = df.to_numpy()[:, :-1]

    data_files = glob.glob(os.path.join(base_dir, '*.npy'))
    for file in data_files:
        epoch = str(os.path.basename(file).split('-')[1])
        with open(file, 'rb') as f:
            data = np.load(f).T
            print(f"Epoch: {epoch}")
            print(mmd_rbf(test_data, data))

