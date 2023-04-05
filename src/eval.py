#!/usr/bin/env python
import os
import sys
from sys import platform
import time
import numpy as np
import json
import cv2
import tqdm
from pose.openposeAPI import OpenPoseEstimator, openpose_header
from pose.pose_angle import get_angle_vec
from pose.classifier import KNNClassifier


openpose = OpenPoseEstimator(op_path='D:/Desktop/openpose',
                             model_folder='../../weights',
                             DEBUG=True)
classifier = KNNClassifier(k=5)

dataset_pth = "../dataset"


def train():
    angle_vecs = []
    labels = []
    short_edge = 360
    for root, _, files in os.walk(os.path.join(dataset_pth, 'train', 'normal')):
        for file in files:
            img = cv2.imread(os.path.join(root, file))
            pose, norm_pose, out_img = openpose.estimate_image(img, short_edge=short_edge)
            angle_vecs.append(get_angle_vec(norm_pose, openpose_header))
            labels.append(0)
    for root, _, files in os.walk(os.path.join(dataset_pth, 'train', 'bad')):
        for file in files:
            img = cv2.imread(os.path.join(root, file))
            pose, norm_pose, out_img = openpose.estimate_image(img, short_edge=short_edge)
            angle_vecs.append(get_angle_vec(norm_pose, openpose_header))
            labels.append(1)
    classifier.fit(angle_vecs, labels)


def eval():
    angle_vecs = []
    labels = []
    short_edge = 360
    for root, _, files in os.walk(os.path.join(dataset_pth, 'test', 'normal')):
        for file in files:
            img = cv2.imread(os.path.join(root, file))
            pose, norm_pose, out_img = openpose.estimate_image(img, short_edge=short_edge)
            angle_vecs.append(get_angle_vec(norm_pose, openpose_header))
            labels.append(0)
    for root, _, files in os.walk(os.path.join(dataset_pth, 'test', 'bad')):
        for file in files:
            img = cv2.imread(os.path.join(root, file))
            pose, norm_pose, out_img = openpose.estimate_image(img, short_edge=short_edge)
            angle_vecs.append(get_angle_vec(norm_pose, openpose_header))
            labels.append(1)
    pred_labels = classifier.classify(angle_vecs)
    correct = np.array(pred_labels) == np.array(labels)
    acc = np.sum(correct) / correct.size
    print("The classification Accuracy is {}".format(acc))


train()
eval()


