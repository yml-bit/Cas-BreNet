import numpy as np
import matplotlib.pyplot as plt
import pydicom
import shutil
import random
import torch.nn as nn
import os
import glob
import SimpleITK as sitk
import cv2
import pandas as pd
import scipy
from skimage import measure
from PIL import Image
import threading

def to_windowdata(image,WC,WW):
    # image = (image + 1) * 0.5 * 4095
    # image[image == 0] = -2000
    # image=image-1024
    center = WC #40 400//60 300
    width = WW# 200
    try:
        win_min = (2 * center - width) / 2.0 + 0.5
        win_max = (2 * center + width) / 2.0 + 0.5
    except:
        # print(WC[0])
        # print(WW[0])
        center = WC[0]  # 40 400//60 300
        width = WW[0]  # 200
        win_min = (2 * center - width) / 2.0 + 0.5
        win_max = (2 * center + width) / 2.0 + 0.5
    dFactor = 255.0 / (win_max - win_min)
    image = image - win_min
    image = np.trunc(image * dFactor)
    image[image > 255] = 255
    image[image < 0] = 0

    # image=image/255#np.uint8(image)
    # image = (image - 0.5)/0.5
    # plt.subplot(2, 2, 1)
    # plt.imshow(image, cmap='gray')  # ,vmin=0,vmax=255
    # plt.show()
    return image.astype(np.uint8)

def read_dicom(file_path):
    # file_path=file_path.replace('../../../', '../../')
    dicom = sitk.ReadImage(file_path)
    data1 = np.squeeze(sitk.GetArrayFromImage(dicom))  #simpleitk提取的就是HU值
    ds = pydicom.dcmread(file_path, force=True)
    # bit=ds.BitsAllocated
    # print("ppp")
    # print(ds)
    pp=ds.pixel_array
    # if  ds.RescaleIntercept == -1024:  #默认，厦门医院的数据无
    #     image2 = data1 + 1024  # sitk读取的数值比为HU值
    #     image2[image2 < 0] = 0
    # elif ds.RescaleIntercept==-8192:
    #     image2 = data1+8192
    #     image2[image2 < 7168] = 7168
    #     image2[image2 > 9216] = 9216  # 12288 10240
    #     image2=image2-image2.min()
    #     # image2 = image2 / 4095#16384
    # elif ds.RescaleIntercept ==0:
    #     image2 = data1 + 1024  # sitk读取的数值比为HU值
    #     image2[image2 < 0] = 0
    WC=ds.WindowCenter
    WW=ds.WindowWidth
    image2=to_windowdata(data1,WC,WW)
    idd=ds.PatientID
    agg=ds.PatientAge
    sex=ds.PatientSex
    return image2,idd,agg,sex

def PSNR(fake, real):
    a = np.where(real != -1)  # Exclude background
    x = a[0].astype(np.uint8)
    y = a[1].astype(np.uint8)
    if x.size == 0 or y.size == 0:
        mse = np.mean(((fake + 1) / 2. - (real + 1) / 2.) ** 2) + 1e-10
    else:
        mse = np.mean(((fake[x, y] + 1) / 2. - (real[x, y] + 1) / 2.) ** 2)
    # mse = np.mean(((fake + 1) / 2. - (real+ 1) / 2.) ** 2)
    if mse < 1.0e-10:
        return 100
    else:
        PIXEL_MAX = 1
        return 20 * np.log10(PIXEL_MAX / (np.sqrt(mse) + 1e-10))

def MAE(fake, real):
    a = np.where(real != -1)  # Exclude background
    x = a[0]
    y = a[1]
    # print(a[2].shape)
    # print(x.shape)
    # print(y.shape)
    if x.size == 0 or y.size == 0:
        mae = np.nanmean(np.abs(fake - real)) + 1e-10
    else:
        mae = np.nanmean(np.abs(fake[x, y] - real[x, y]))
    return mae / 2  # from (-1,1) normaliz  to (0,1)

def quality(fake, real):
    # fake=fake[180:180+150,153:153+150]
    # real = real[180:180 + 150, 153:153 + 150]
    # fake[fake <200] = 0
    # fake[fake>255]=255
    fake=(fake/255- 0.5)/0.5
    real = (real / 255 - 0.5) / 0.5
    mae=MAE(fake, real)
    psnr=PSNR(fake, real)
    ssim=measure.compare_ssim(fake, real)
    return mae,psnr,ssim

def make_test():
    path = "../output/disp/disp2/"  # CT_CTA disease
    f1 = open("quality_test.txt", "w")
    path_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        input_files = os.listdir(root)
        for file in input_files:
            path=os.path.join(root, file)
            if "000001" in path:
                path_list.append(path)
    path_list.sort()

    ii = 0
    f1.writelines('            mae                     psnr                     ssim '+"\n")
    for sub_path in path_list:
        # aa=sub_path.replace("_000001", '_400001')
        i0=cv2.imread(sub_path, cv2.IMREAD_GRAYSCALE)
        i1 = cv2.imread(sub_path.replace("_000001", '_100001'), cv2.IMREAD_GRAYSCALE)
        i2 = cv2.imread(sub_path.replace("_000001", '_200001'), cv2.IMREAD_GRAYSCALE)

        i4 = cv2.imread(sub_path.replace("_000001", '_400001'), cv2.IMREAD_GRAYSCALE)
        i5 = cv2.imread(sub_path.replace("_000001", '_500001'), cv2.IMREAD_GRAYSCALE)

        # i0 = (i0 - np.mean(i0)) / np.std(i0)*np.std(i1)+np.mean(i1)
        # i2 = (i2 - np.mean(i2)) / np.std(i2)*np.std(i1)+np.mean(i1)
        # i3 = (i3 - np.mean(i3)) / np.std(i3)*np.std(i1)+np.mean(i1)
        # i4 = (i4 - np.mean(i4)) / np.std(i4)*np.std(i1)+np.mean(i1)
        # i5 = (i5 - np.mean(i5)) / np.std(i5)*np.std(i1)+np.mean(i1)
        # if "17_1" in sub_path:
        #     d1=(i0-i1)**2
        #     d2=(i2-i1)**2
        #     d3=(i3-i1)**2
        #     d4=(i4-i1)**2
        #     d5=(i5-i1)**2
        #     print(np.mean(i0))
        #     print(np.mean(i1))
        #     print(np.mean(i2))
        #     print(np.mean(i3))
        #     print(np.mean(i4))
        #     print(np.mean(i5))
        #     aa=1

        q1 =quality(i0, i1)
        q2 = quality(i2, i1)
        q4 = quality(i4, i1)
        q5 = quality(i5, i1)
        f1.writelines(sub_path+"\n")
        f1.writelines(str(q1) + "\n")
        f1.writelines(str(q2) + "\n")
        # try:
        #     i3 = cv2.imread(sub_path.replace("_000001", '_300001'), cv2.IMREAD_GRAYSCALE)
        #     q3 = quality(i3, i1)
        #     f1.writelines(str(q3) + "\n")#Degan
        # except:
        #     continue
        f1.writelines(str(q4) + "\n")
        f1.writelines(str(q5) + "\n")
        f1.writelines("\n")

        ii=ii+1
        if ii%10==0:
            print('numbers:', ii)

def fun(sub_path,i0,train_lista):
    f1 = open("disp1_failure_disp1.txt", "a")
    psnrb=35
    best=""
    best_path=""
    for test_sub_path in train_lista:
        i1, id,agg,sex= read_dicom(test_sub_path)
        psnr = quality(i0, i1)[1]
        # i0 = (i0 / 255 - 0.5) / 0.5
        # i1 = (i1 / 255 - 0.5) / 0.5
        # psnr = PSNR(i0, i1)
        # print(psnr)
        if best!=test_sub_path.split("SE0")[0]:#一列遍历结束
            best=test_sub_path.split("SE0")[0]

            # if best!="" and psnr > psnrb:#大于指定PSNR则保存
            #     try:
            #         i1, id, agg, sex = read_dicom(best_path)
            #     except:
            #         print(best_path)
            #     f1.writelines(
            #         id + '            ' + agg + '            ' + sex + '            ' + sub_path + '            ' + test_sub_path + "\n")
            # best_path=""
            # psnrb = 20
            # # best_path = ""

        if psnr>psnrb:
            psnrb=psnr
            best_path=test_sub_path
            f1.writelines(str(psnr)+'            '+id + '            ' + agg + '            ' + sex + '            ' + sub_path + '            ' + test_sub_path + "\n")

    f1.close()

def cir_for(path_list,train_lista):
    # f1 = open("disp1_nor_disp.txt", "w")
    # f1.writelines('id       agg      sex          dis_path                                          test_sub_path' + "\n")
    # f1.close()
    ii = 0
    for sub_path in path_list:
        # aa=sub_path.replace("_000001", '_400001')
        i0 = cv2.imread(sub_path, cv2.IMREAD_GRAYSCALE)
        threading.Thread(target=fun, args=(sub_path,i0,train_lista)).start()
        ii = ii + 1
        if ii % 10 == 0:
            print('numbers:', ii)

def find_test_road():
    #展示图list
    path = "../output/disp/disp1/failure_disp/"  # p1_data_disp    hnnk_disp
    f = open('./p_test1.txt')  # p_test0
    # f1 = open("disp1.txt", "w")
    path_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        input_files = os.listdir(root)
        for file in input_files:
            path=os.path.join(root, file)
            if "1_3_000001" in path:
                path_list.append(path)

    # path_list.sort()
    # a='../output/disp/disp1/sup/nor_disp/18_1_000001.jpg'
    train_lista = []
    train_listb = []
    for line in f.readlines():
        # if "p1_data" in line:
        line = line.strip('\n')  # 直接将文件中按行读到list里，效果与方法2一样
        train_lista.append(line)
        # train_listb.append(line.replace("SE0", "SE1"))
    train_lista.sort()
    f.close()  # 关
    cir_for(path_list, train_lista)

if __name__ == '__main__':
    # make_test()
    # find_test_road1() #
    find_test_road() #多线程