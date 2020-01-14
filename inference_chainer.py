#!/usr/bin/env python
"""Example code of learning a large scale convnet from ILSVRC2012 dataset.
Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images and scale them to 256x256, and make two lists of space-
separated CSV whose first column is full path to image and second column is
zero-origin label (this format is same as that used by Caffe's ImageDataLayer).
"""
from __future__ import print_function
import argparse
import datetime
import json
import multiprocessing as mp
import os
import random
import sys
import threading
import time

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from scipy import ndimage
import csv
import cv2

import six
import six.moves.cPickle as pickle
from six.moves import queue

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.links import caffe
from matplotlib.ticker import *
from chainer import serializers
from chainer import cuda
# from CNN4IHM import CNN4IHM
from CNN4IHM_new import CNN4IHM

#
# Arguments.
#
parser = argparse.ArgumentParser(
    description='IHM semantic segmentation')
parser.add_argument('list', help='Path to inference image list file')
parser.add_argument('--out_root', '-R', default='./Segmentation', help='Root directory path of segmentation files')
parser.add_argument('--csv', '-c', default='Accuracy.csv', help='Path to Accuracy text file')
parser.add_argument('--model', '-m', default='model', help='Path to model file')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--root', '-r', default='.',
                    help='Root directory path of image files')
parser.add_argument('--RotationFlip', '-rf', default=False, type=bool, help='Flag of image rotation and flip')
parser.add_argument('--debug', '-d', default=0, type=int, help='0:Normal, 1:Output conv image')
args = parser.parse_args()

#
# Preprocessing.
#
model = CNN4IHM()  # Init Chainer network.

# width = model.OUTSIZE
# height = model.OUTSIZE
# frame = args.frame

if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

# mean_image = pickle.load(open(args.mean, 'rb')).astype(np.float32)  # Read a mean image. Convert the mean image to float32.
# mean_image /= 255                                                   # Normalize the mean image.
"""
image = np.asarray(Image.open(args.image)).astype(np.float32)
#image -= mean_image
image /= 255
image = image[np.newaxis,np.newaxis,:]
"""

serializers.load_npz(args.model, model)  # Load trained parameters of the chainer network.
csv_path = open(args.csv, 'w')  # Open a output txt file of Accuracy.
csv_writer = csv.writer(csv_path, lineterminator='\n')  # CSV writer.
try:
    os.mkdir(args.out_root)
except FileExistsError:
    pass

## Visualize Filter ============================
# print(model.conv1.W.shape)
# print(model.conv1.W.data[0,0])

n1, n2, h, w = model.conv1.W.shape
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(n1):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(model.conv1.W.data[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
# plt.savefig(args.model+'_conv1.png',dsp=150)
# cv2.waitKey(0)

n1, n2, h, w = model.conv2.W.shape
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(n1):
    ax = fig.add_subplot(8, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(model.conv2.W.data[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

# cv2.waitKey(0)

n1, n2, h, w = model.conv3.W.shape
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(n1):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(model.conv3.W.data[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

cv2.waitKey(0)
# ============================================


# model.to_cpu()                     # Use CPU for chainer.
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()  # Set device num.
    model.to_gpu()  # Use GPU for chainer.

n1, n1_, h1, w1 = model.conv1.W.shape
n2, n2_, h2, w2 = model.conv2.W.shape
n3, n3_, h3, w3 = model.conv3.W.shape
n4, n4_, h4, w4 = model.conv4.W.shape
n5, n5_, h5, w5 = model.conv5.W.shape
"""
if args.debug==1:
    outpool0 = open(args.model+'_pool0_1042x770.vol','wb')
    outconv1 = open(args.model+'_conv1_517x381.vol','wb')
    outconv2 = open(args.model+'_conv2_257x189.vol','wb')
    outconv3 = open(args.model+'_conv3_127x93.vol','wb')
    outconv4 = open(args.model+'_conv4_62x45.vol','wb')
    outconv5 = open(args.model+'_conv5_60x43.vol','wb')
    outpxshf = open(args.model+'_pxshf_960x688.vol','wb')
    outfinal = open(args.model+'_final_3840x2752.vol','wb')
"""
if args.debug == 1:
    multi_pool0 = np.zeros((1, 1042, 770), np.float32)
    multi_conv1 = np.zeros((n1, 517, 381), np.float32)
    multi_conv2 = np.zeros((n2, 257, 189), np.float32)
    multi_conv3 = np.zeros((n3, 127, 93), np.float32)
    multi_conv4 = np.zeros((n4, 62, 45), np.float32)
    multi_conv5 = np.zeros((n5, 60, 43), np.float32)
    multi_pxshf = np.zeros((3, 960, 688), np.float32)
    multi_final = np.zeros((3, 3840, 2752), np.float32)

# morph_kr = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)  # Kernel of opening morphology.

chainer.config.cudnn_deterministic = True  # Set deterministic mode for cuDNN.
chainer.config.train = False  # Set evaluation mode for Chainer.


def load_image_list(path, root):
    tuples = []
    for line in open(path):
        pair = line.strip().split()
        tuples.append((os.path.join(root, pair[0]), (os.path.join(root, pair[1]))))
    return tuples


# Prepare dataset
img_list = load_image_list(args.list, args.root)


def read_image(img_path, lab_path, augmentation=True):
    # Data loading routine
    image = np.asarray(Image.open(img_path)).astype(np.float32)
    # image = xp.asarray(Image.open(img_path)).astype(np.float32)
    # image -= mean_image
    image /= 255
    image -= 0.5

    label = np.asarray(Image.open(lab_path)).astype(np.int32)
    # label = xp.asarray(Image.open(lab_path)).astype(np.int32)
    label //= 127

    if augmentation:
        if random.random() > 0.5:
            tmp1 = cv2.flip(image, 0)  # Vertical flip
            tmp2 = cv2.flip(label, 0)  # Vertical flip
            # tmp1 = xp.flipud(image)
            # tmp2 = xp.flipud(label)
            image = tmp1
            label = tmp2

        if random.random() > 0.5:
            tmp1 = cv2.flip(image, 1)  # Horizontal flip
            tmp2 = cv2.flip(label, 1)  # Horizontal flip
            # tmp1 = xp.fliplr(image)
            # tmp2 = xp.fliplr(label)
            image = tmp1
            label = tmp2

        """
        M = np.float32([[1,0,random.randint(-20,20)],[0,1,random.randint(-20,20)]])
        tmp1 = cv2.warpAffine(image, M, image.shape)   # Shift
        tmp2 = cv2.warpAffine(label, M, label.shape)   # Shift
        image = tmp1
        label = tmp2
        """

        angle = random.uniform(0, 360)
        tmp1 = ndimage.interpolation.rotate(image, angle, reshape=False, order=1,
                                            mode='reflect')  # Rotation by bilinear
        tmp2 = ndimage.interpolation.rotate(label, angle, reshape=False, order=0,
                                            mode='reflect')  # Rotation by nearest neighbor
        image = tmp1
        label = tmp2

    # img = cv2.resize(image, None, fx=0.25, fy=0.25)
    # lab = cv2.resize(label.astype(np.float32), None, fx=0.25, fy=0.25)
    # cv2.imshow('image',img)
    # cv2.imshow('label',lab)
    # cv2.waitKey(0)

    return image, label


def label2RGB(label):
    rgb = np.zeros((label.shape[0], label.shape[1], 3), np.uint8)

    rgb[:, :, 0] = 255 * (label == 0)
    rgb[:, :, 1] = 255 * (label == 1)
    rgb[:, :, 2] = 255 * (label == 2)

    return rgb


def PDF2label(pdf):
    label = np.argmax(pdf, axis=2).astype(np.int32)

    return label


if __name__ == '__main__':
    # csv_writer.writerow(['Number', 'Accuracy', 'Colony area rate', 'TP for NG', 'FP for NG', 'FN for NG', 'TN for NG', 'Name'])   # Write titles to csv file.
    csv_writer.writerow(['Number', 'Accuracy', 'Colony Area Rate', 'NG Sensitivity', 'NG Specificity', 'OK Sensitivity',
                         'NG Sprcificity', 'BG/BG', 'NG/BG', 'OK/BG', 'BG/NG', 'NG/NG', 'OK/NG', 'BG/OK', 'NG/OK',
                         'OK/OK', 'Name'])  # Write titles to csv file.

    for idx in range(len(img_list)):  # Index loop.
        print(idx + 1, '/', len(img_list))
        img_path, lab_path = img_list[idx]
        image, label = read_image(img_path, lab_path, args.RotationFlip)  # Read images.
        # image, label = read_image(img_path, lab_path, True) # Read images.
        image = image[np.newaxis, np.newaxis, :]
        label = label[np.newaxis, :]

        t0 = time.perf_counter() * 1000
        x = chainer.Variable(xp.asarray(image))  # Set the image to GPU as chainer input.
        # t = chainer.Variable(xp.asarray(label))    # Set the label to GPU as chainer input.
        # hmap = chainer.cuda.to_cpu(model.predict(x,t).data[0,:,:,:])   # Calculate foward propagation of the chainer. Transport tumor PDF to CPU.
        hmap = chainer.cuda.to_cpu(model.predict(x).data[0, :, :,
                                   :])  # Calculate foward propagation of the chainer. Transport tumor PDF to CPU.
        t1 = time.perf_counter() * 1000
        print('%f[msec]' % (t1 - t0))

        if args.debug == 1:
            # Hack hidden layers
            multi_pool0 = chainer.cuda.to_cpu(model.pool0_img.data[0, :, :, :].astype(np.float32))
            multi_conv1 = chainer.cuda.to_cpu(model.conv1_img.data[0, :, :, :].astype(np.float32))
            multi_conv2 = chainer.cuda.to_cpu(model.conv2_img.data[0, :, :, :].astype(np.float32))
            multi_conv3 = chainer.cuda.to_cpu(model.conv3_img.data[0, :, :, :].astype(np.float32))
            multi_conv4 = chainer.cuda.to_cpu(model.conv4_img.data[0, :, :, :].astype(np.float32))
            multi_conv5 = chainer.cuda.to_cpu(model.conv5_img.data[0, :, :, :].astype(np.float32))
            multi_pxshf = chainer.cuda.to_cpu(model.pxshf_img.data[0, :, :, :].astype(np.float32))
            multi_final = chainer.cuda.to_cpu(model.final_img.data[0, :, :, :].astype(np.float32))

            # Output hidden layers
            outpool0 = open(args.model + '_pool0_1042x770.vol', 'wb')
            outconv1 = open(args.model + '_conv1_517x381.vol', 'wb')
            outconv2 = open(args.model + '_conv2_257x189.vol', 'wb')
            outconv3 = open(args.model + '_conv3_127x93.vol', 'wb')
            outconv4 = open(args.model + '_conv4_62x45.vol', 'wb')
            outconv5 = open(args.model + '_conv5_60x43.vol', 'wb')
            outpxshf = open(args.model + '_pxshf_960x688.vol', 'wb')
            outfinal = open(args.model + '_final_3840x2752.vol', 'wb')

            outpool0.write(multi_pool0)
            outconv1.write(multi_conv1)
            outconv2.write(multi_conv2)
            outconv3.write(multi_conv3)
            outconv4.write(multi_conv4)
            outconv5.write(multi_conv5)
            outpxshf.write(multi_pxshf)
            outfinal.write(multi_final)

            outpool0.close()
            outconv1.close()
            outconv2.close()
            outconv3.close()
            outconv4.close()
            outconv5.close()
            outpxshf.close()
            outfinal.close()

        # pdf = cv2.morphologyEx(hmap, cv2.MORPH_OPEN, morph_kr)          # Remove tiny structures by opening morphology.
        # cv2.threshold(pdf, args.threshould/100, 1, cv2.THRESH_TOZERO, pdf)  # Threshould tumor PDF where tumor probability >= threshould.

        pdf = (hmap * 255).astype(np.uint8).transpose(1, 2, 0)  # Transpose RGB.
        seg_label = PDF2label(pdf)  # Convert to segmentation label.
        seg = label2RGB(seg_label)  # Convert to segmentation image.

        # acc = model.accuracy.data
        acc = np.count_nonzero(
            seg_label == label[0, (model.ORGY - model.EFFY) // 2:(model.ORGY - model.EFFY) // 2 + model.EFFY,
                         (model.ORGX - model.EFFX) // 2:(model.ORGX - model.EFFX) // 2 + model.EFFX]) / (
                          model.EFFX * model.EFFY)
        print('Accuracy =', acc)

        lab_hist = np.histogram(label[0, (model.ORGY - model.EFFY) // 2:(model.ORGY - model.EFFY) // 2 + model.EFFY,
                                (model.ORGX - model.EFFX) // 2:(model.ORGX - model.EFFX) // 2 + model.EFFX],
                                bins=[0, 1, 2, 3])
        seg_hist = np.histogram(seg_label, bins=3)
        area_rate = (seg_hist[0][1] + seg_hist[0][2]) / (lab_hist[0][1] + lab_hist[0][2])
        print('Colony Area Rate =', area_rate)

        """
        tp = np.count_nonzero((seg_label == 1) * (label[0,(model.ORGY-model.EFFY)//2:(model.ORGY-model.EFFY)//2+model.EFFY, (model.ORGX-model.EFFX)//2:(model.ORGX-model.EFFX)//2+model.EFFX] == 1))
        fp = np.count_nonzero((seg_label == 1) * (label[0,(model.ORGY-model.EFFY)//2:(model.ORGY-model.EFFY)//2+model.EFFY, (model.ORGX-model.EFFX)//2:(model.ORGX-model.EFFX)//2+model.EFFX] == 2))
        fn = np.count_nonzero((seg_label == 2) * (label[0,(model.ORGY-model.EFFY)//2:(model.ORGY-model.EFFY)//2+model.EFFY, (model.ORGX-model.EFFX)//2:(model.ORGX-model.EFFX)//2+model.EFFX] == 1))
        tn = np.count_nonzero((seg_label == 2) * (label[0,(model.ORGY-model.EFFY)//2:(model.ORGY-model.EFFY)//2+model.EFFY, (model.ORGX-model.EFFX)//2:(model.ORGX-model.EFFX)//2+model.EFFX] == 2))
        if tp+fn!=0:
            sens = tp / (tp+fn)
        else:
            sens = None
        if fp+tn!=0:
            spec = tn / (fp+tn)
        else:
            spec = None
        print('Sensitivity =', sens)
        print('Specificity =', spec)
        """
        # Pixelwise evaluation
        # s0t0 = np.count_nonzero((seg_label == 0) * (label[0,(model.ORGY-model.EFFY)//2:(model.ORGY-model.EFFY)//2+model.EFFY, (model.ORGX-model.EFFX)//2:(model.ORGX-model.EFFX)//2+model.EFFX] == 0))
        # s1t0 = np.count_nonzero((seg_label == 1) * (label[0,(model.ORGY-model.EFFY)//2:(model.ORGY-model.EFFY)//2+model.EFFY, (model.ORGX-model.EFFX)//2:(model.ORGX-model.EFFX)//2+model.EFFX] == 0))
        # s2t0 = np.count_nonzero((seg_label == 2) * (label[0,(model.ORGY-model.EFFY)//2:(model.ORGY-model.EFFY)//2+model.EFFY, (model.ORGX-model.EFFX)//2:(model.ORGX-model.EFFX)//2+model.EFFX] == 0))
        # s0t1 = np.count_nonzero((seg_label == 0) * (label[0,(model.ORGY-model.EFFY)//2:(model.ORGY-model.EFFY)//2+model.EFFY, (model.ORGX-model.EFFX)//2:(model.ORGX-model.EFFX)//2+model.EFFX] == 1))
        # s1t1 = np.count_nonzero((seg_label == 1) * (label[0,(model.ORGY-model.EFFY)//2:(model.ORGY-model.EFFY)//2+model.EFFY, (model.ORGX-model.EFFX)//2:(model.ORGX-model.EFFX)//2+model.EFFX] == 1))
        # s2t1 = np.count_nonzero((seg_label == 2) * (label[0,(model.ORGY-model.EFFY)//2:(model.ORGY-model.EFFY)//2+model.EFFY, (model.ORGX-model.EFFX)//2:(model.ORGX-model.EFFX)//2+model.EFFX] == 1))
        # s0t2 = np.count_nonzero((seg_label == 0) * (label[0,(model.ORGY-model.EFFY)//2:(model.ORGY-model.EFFY)//2+model.EFFY, (model.ORGX-model.EFFX)//2:(model.ORGX-model.EFFX)//2+model.EFFX] == 2))
        # s1t2 = np.count_nonzero((seg_label == 1) * (label[0,(model.ORGY-model.EFFY)//2:(model.ORGY-model.EFFY)//2+model.EFFY, (model.ORGX-model.EFFX)//2:(model.ORGX-model.EFFX)//2+model.EFFX] == 2))
        # s2t2 = np.count_nonzero((seg_label == 2) * (label[0,(model.ORGY-model.EFFY)//2:(model.ORGY-model.EFFY)//2+model.EFFY, (model.ORGX-model.EFFX)//2:(model.ORGX-model.EFFX)//2+model.EFFX] == 2))

        # 240x240 Patchwise evaluation
        seg_label_mini = PDF2label(cv2.resize(seg, None, fx=1 / 240, fy=1 / 240, interpolation=cv2.INTER_AREA))
        label_mini = PDF2label(cv2.resize(label2RGB(
            label[0, (model.ORGY - model.EFFY) // 2:(model.ORGY - model.EFFY) // 2 + model.EFFY,
            (model.ORGX - model.EFFX) // 2:(model.ORGX - model.EFFX) // 2 + model.EFFX]), None, fx=1 / 240, fy=1 / 240,
                                          interpolation=cv2.INTER_AREA))
        s0t0 = np.count_nonzero((seg_label_mini == 0) * (label_mini == 0))
        s1t0 = np.count_nonzero((seg_label_mini == 1) * (label_mini == 0))
        s2t0 = np.count_nonzero((seg_label_mini == 2) * (label_mini == 0))
        s0t1 = np.count_nonzero((seg_label_mini == 0) * (label_mini == 1))
        s1t1 = np.count_nonzero((seg_label_mini == 1) * (label_mini == 1))
        s2t1 = np.count_nonzero((seg_label_mini == 2) * (label_mini == 1))
        s0t2 = np.count_nonzero((seg_label_mini == 0) * (label_mini == 2))
        s1t2 = np.count_nonzero((seg_label_mini == 1) * (label_mini == 2))
        s2t2 = np.count_nonzero((seg_label_mini == 2) * (label_mini == 2))

        sum0 = s0t0 + s1t0 + s2t0
        sum1 = s0t1 + s1t1 + s2t1
        sum2 = s0t2 + s1t2 + s2t2
        if sum1 != 0:
            ng_sens = s1t1 / sum1
        else:
            ng_sens = ''
        if sum0 + sum2 != 0:
            ng_spec = (s0t0 + s2t0 + s0t2 + s2t2) / (sum0 + sum2)
        else:
            ng_spec = ''
        if sum2 != 0:
            ok_sens = s2t2 / sum2
        else:
            ok_sens = ''
        if sum0 + sum1 != 0:
            ok_spec = (s0t0 + s1t0 + s0t1 + s1t1) / (sum0 + sum1)
        else:
            ok_spec = ''
        print('NG Sensitivity =', ng_sens)
        print('NG Specificity =', ng_spec)
        print('OK Sensitivity =', ok_sens)
        print('OK Specificity =', ok_spec)

        # csv_writer.writerow([idx+1, acc, area_rate, tp, fp, fn, tn, img_path])   # Write accuracy to csv file.
        csv_writer.writerow(
            [idx + 1, acc, area_rate, ng_sens, ng_spec, ok_sens, ok_spec, s0t0, s1t0, s2t0, s0t1, s1t1, s2t1, s0t2,
             s1t2, s2t2, img_path])  # Write accuracy to csv file.

        dir, file = os.path.split(lab_path)
        head, tail = os.path.split(dir)
        head, tail = os.path.split(head)
        out_path = args.out_root + '\\' + tail
        try:
            os.mkdir(out_path)
        except FileExistsError:
            pass
        name, ext = os.path.splitext(file)
        cv2.imwrite(out_path + '\\' + name + '_seg.png', pdf)  # Output segmentation image.
        # cv2.imwrite(args.out, pdf)               # Output segmentation image.

        layer1 = cv2.resize(pdf, None, fx=0.25, fy=0.25)
        # dsp = cv2.resize(image[0,0,(model.ORGY-model.EFFY)//2:(model.ORGY-model.EFFY)//2+model.EFFY, (model.ORGX-model.EFFX)//2:(model.ORGX-model.EFFX)//2+model.EFFX], None, fx=0.25, fy=0.25)
        dsp = cv2.resize(image[0, 0, (model.ORGY - model.EFFY) // 2:(model.ORGY - model.EFFY) // 2 + model.EFFY,
                         (model.ORGX - model.EFFX) // 2:(model.ORGX - model.EFFX) // 2 + model.EFFX], None, fx=0.25,
                         fy=0.25)
        # layer2 = cv2.cvtColor((dsp*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)   # Convert the input image to color for display.
        layer2 = cv2.cvtColor(((dsp + 0.5) * 255).astype(np.uint8),
                              cv2.COLOR_GRAY2RGB)  # Convert the input image to color for display.

        dsp = cv2.resize(label[0, (model.ORGY - model.EFFY) // 2:(model.ORGY - model.EFFY) // 2 + model.EFFY,
                         (model.ORGX - model.EFFX) // 2:(model.ORGX - model.EFFX) // 2 + model.EFFX], None, fx=0.25,
                         fy=0.25, interpolation=cv2.INTER_NEAREST)
        lab = label2RGB(dsp)  # Convert the label image to color for display.

        cv2.destroyAllWindows()  # Close all display windows.
        cv2.imshow(img_path, layer1 // 4 + layer2)  # Display superimposed image.
        # cv2.imshow(img_path, layer2)                      # Display superimposed image.
        # cv2.imshow(lab_path, lab)                      # Display teacher label image.
        # cv2.waitKey(0)
        cv2.waitKey(1000)
        # cv2.waitKey(1)

        cv2.imwrite(out_path + '\\' + name + '_reg.png', layer1 // 4 + layer2)  # Output superimposed image.

        print('')
"""
if args.debug==1:
    outpool0.close()
    outconv1.close()
    outconv2.close()
    outconv3.close()
    outconv4.close()
    outconv5.close()
    outpxshf.close()
    outfinal.close()
"""
cv2.destroyAllWindows()  # Close all display windows.
csv_path.close()  # Close the output txt file of accuracy.
