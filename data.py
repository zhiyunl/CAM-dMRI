import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.reconst import dti
from dipy.segment.mask import median_otsu

from utils import multi_slice_viewer


def vis_all(data):
    for i in range(data.shape[-1]):
        plt.imshow(data[:, :, i].T, cmap='gray', origin='lower')
        plt.show()


def visualize(data):
    # visualize middle slice

    axial_middle = data.shape[2] // 2
    plt.figure('Showing the datasets')
    plt.subplot(1, 2, 1).set_axis_off()
    # without diffusion weighting
    plt.imshow(data[:, :, axial_middle, 0].T, cmap='gray', origin='lower')
    plt.subplot(1, 2, 2).set_axis_off()
    # with dw
    plt.imshow(data[:, :, axial_middle, 10].T, cmap='gray', origin='lower')
    plt.show()
    # plt.savefig('data.png', bbox_inches='tight')


def visualize3D(FA):
    # plt.figure('FA')
    # ax = plt.axes(projection="3d")
    # plt.show()
    multi_slice_viewer(FA)


def traverseFA(data, fname):
    for i in range(data.shape[2]):
        # plt.imshow(data[:,:,mid,0].T,cmap='gray',origin='lower')
        plt.imsave(fname + str(i) + ".png", data[:, :, i].T, cmap='gray', origin='lower')


def get25DSlice(data, fname, dir=3):
    mid0 = data.shape[0] // 2 - 8
    mid1 = data.shape[1] // 2 - 8
    mid2 = data.shape[2] // 2 - 8
    # i = 15
    # plt.imsave(fname + "mid0_"+str(i)+".png", data[mid0, :, :].T, cmap='gray', origin='lower')
    # if dir==3:
    #     plt.imsave(fname + "mid1_"+str(i)+".png", data[:, mid1, :].T, cmap='gray', origin='lower')
    #     plt.imsave(fname + "mid2_"+str(i)+".png", data[:, :, mid2].T, cmap='gray', origin='lower')
    for i in range(16):
        plt.imsave(fname + "mid0_" + str(i) + ".png", data[mid0 + i, :, :].T, cmap='gray', origin='lower')
        if dir == 3:
            plt.imsave(fname + "mid1_" + str(i) + ".png", data[:, mid1 + i, :].T, cmap='gray', origin='lower')
            plt.imsave(fname + "mid2_" + str(i) + ".png", data[:, :, mid2 + i].T, cmap='gray', origin='lower')


def raw2FA(source, file, des):
    if os.path.exists(des + file[:-5] + "_fa.nii.gz"):
        print(file[:-5], "already processed")
        return
    fbval = source + file
    fdwi = source + file[:-5] + ".nii.gz"
    fbvec = source + file[:-5] + ".bvec"
    data, affine = load_nifti(fdwi)
    print(fdwi, data.shape)
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)

    maskdata, mask = median_otsu(data, vol_idx=range(10, 50), median_radius=3, numpass=1, autocrop=True,
                                 dilate=2)
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(maskdata)
    FA = dti.fractional_anisotropy(tenfit.evals)

    FA[np.isnan(FA)] = 0
    save_nifti(des + file[:-5] + "_fa.nii.gz", FA.astype(np.float32), affine)

    # visualize(data)


def raw2FA_new(fdwi, fbval=None, fbvec=None):
    data, affine = load_nifti(fdwi)
    print(fdwi, data.shape)
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)

    maskdata, mask = median_otsu(data, vol_idx=range(10, 20), median_radius=3, numpass=1, autocrop=True,
                                 dilate=2)
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(maskdata)
    FA = dti.fractional_anisotropy(tenfit.evals)

    FA[np.isnan(FA)] = 0
    plt.imshow(FA[:, :, FA.shape[2] // 2].T, cmap='gray', origin='lower')
    plt.show()
    pass
    # save_nifti(des + file[:-5] + "_fa.nii.gz", FA.astype(np.float32), affine)


def FA2Slice(source, file, des, dir=3):
    if not os.path.exists(des):
        os.mkdir(des)
    FA, affine = load_nifti(source + file)
    print(FA.shape)
    # visualize3D(FA)
    get25DSlice(FA, des + file[:-9], dir)


def FA2Npy(source, file, des):
    if not os.path.exists(des):
        os.mkdir(des)
    FA, affine = load_nifti(source + file)
    # visualize3D(FA)
    # get25DSlice(FA,des+file[:-9])
    FA = np.array(FA)
    print(FA.shape)
    mid = FA.shape[-1] // 2
    # vis_all(FA[:,:,mid-10:mid+10])
    np.save(des + file[:-9], FA[:, :, mid])


def preDataset():
    view = False
    source = "~/MIA-Proj/DATA/HCP/unproc/"

    # nii.gz to FA
    desFA = "~/MIA-Proj/DATA/HCP/proc/"
    if not os.path.exists(desFA):
        os.mkdir(desFA)
    for file in os.listdir(source):
        if file.endswith(".bval"):
            raw2FA(source, file, desFA)

    # FA to Slice
    desSlice = "~/MIA-Proj/DATA/HCP/16x3_slice"
    if not os.path.exists(desSlice):
        os.mkdir(desSlice)
    with open(source + '27slice.csv', newline='') as csvfile:
        labelreader = csv.reader(csvfile)
        labelDict = {rows[0]: rows[2] for rows in labelreader}
    # mapping = {"20-24":0,"25-29":1,"30-34":2,"35-39":3,"40-44":4,"45-59":5}
    mapping = {"8-9": "young", "14-15": "young", "25-35": "old", "45-55": "old", "65-75": "old"}
    for file in os.listdir(desFA):
        label = mapping[labelDict[file[:6]]]  # LS2001
        # label = labelDict[file[:8]] # MGH_1001
        if view:
            FA, affine = load_nifti(source + file)
            visualize3D(FA)
            break
        else:
            FA2Slice(desFA, file, desSlice + label + "/", dir=3)

    # desNpy = "~/MIA-Proj/DATA/HCP/npy/"
    # if not os.path.exists(desNpy):
    #     os.mkdir(desNpy)
    # for file in os.listdir(desFA):
    #     label = labelDict[file[:6]]
    #     FA2Npy(desFA,file,desNpy+label+"/")


if __name__ == "__main__":
    preDataset()
    # raw2FA_new("/home/zhiyunl/MIA-Proj/DATA/HCP/unproc/data.nii.gz",
    #            "/home/zhiyunl/MIA-Proj/DATA/HCP/unproc/bvals",
    #            "/home/zhiyunl/MIA-Proj/DATA/HCP/unproc/bvecs")
