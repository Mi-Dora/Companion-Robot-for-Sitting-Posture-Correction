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

load_pose = True

if not load_pose:
    openpose = OpenPoseEstimator(op_path='D:\\Desktop\\openpose',
                                    model_folder='..\\weights',
                                     DEBUG=True)
classifier = KNNClassifier(k=3)

dataset_pth = "../dataset"


def train(load=False):
    poses = []
    angle_vecs = []
    labels = []
    short_edge = 360
    if load:
        poses = np.load('../dataset/train_poses.npy', allow_pickle=True)
        labels = np.load('../dataset/train_labels.npy', allow_pickle=True)
        for pose in poses:
            angle_vecs.append(get_angle_vec(pose[0], openpose_header))
    else:
        for root, _, files in os.walk(os.path.join(dataset_pth, 'train', 'normal')):
            for file in files:
                img = cv2.imread(os.path.join(root, file))
                pose, norm_pose, out_img = openpose.estimate_image(img, short_edge=short_edge)
                if pose is None:
                    print(file)
                    continue
                poses.append(pose)
                angle_vecs.append(get_angle_vec(pose[0], openpose_header))
                labels.append(0)
        for root, _, files in os.walk(os.path.join(dataset_pth, 'train', 'bad')):
            for file in files:
                img = cv2.imread(os.path.join(root, file))
                pose, norm_pose, out_img = openpose.estimate_image(img, short_edge=short_edge)
                if pose is None:
                    print(file)
                    continue
                poses.append(pose)
                angle_vecs.append(get_angle_vec(pose[0], openpose_header))
                labels.append(1)
        poses = np.array(poses)
        labels = np.array(labels)
        np.save('../dataset/train_poses', poses)
        np.save('../dataset/train_labels', labels)
    angle_vecs = np.array(angle_vecs)
    classifier.fit(angle_vecs, labels)


def eval(load=False):
    poses = []
    angle_vecs = []
    labels = []
    short_edge = 360
    if load:
        poses = np.load('../dataset/test_poses.npy', allow_pickle=True)
        labels = np.load('../dataset/test_labels.npy', allow_pickle=True)
        for pose in poses:
            angle_vecs.append(get_angle_vec(pose[0], openpose_header))
    else:
        for root, _, files in os.walk(os.path.join(dataset_pth, 'test', 'normal')):
            for file in files:
                img = cv2.imread(os.path.join(root, file))
                pose, norm_pose, out_img = openpose.estimate_image(img, short_edge=short_edge)
                if pose is None:
                    print(file)
                    continue
                poses.append(pose)
                angle_vecs.append(get_angle_vec(pose[0], openpose_header))
                labels.append(0)
        for root, _, files in os.walk(os.path.join(dataset_pth, 'test', 'bad')):
            for file in files:
                img = cv2.imread(os.path.join(root, file))
                pose, norm_pose, out_img = openpose.estimate_image(img, short_edge=short_edge)
                if pose is None:
                    print(file)
                    continue
                poses.append(pose)
                angle_vecs.append(get_angle_vec(pose[0], openpose_header))
                labels.append(1)
        poses = np.array(poses)
        labels = np.array(labels)
        np.save('../dataset/test_poses', poses)
        np.save('../dataset/test_labels', labels)
    angle_vecs = np.array(angle_vecs)
    pred_labels = classifier.classify(angle_vecs)
    correct = np.array(pred_labels) == np.array(labels).reshape(-1, 1)
    acc = np.sum(correct) / correct.size
    print("The classification Accuracy is {}".format(acc))


train(load=load_pose)
eval(load=load_pose)


