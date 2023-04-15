#!/usr/bin/env python3

import cv2
import numpy as np
import os
from sklearn import svm


class KNNClassifier(object):

    def __init__(self, k=10):
        self.k = k
        # self.svm = svm.SVC()
        self.knn = cv2.ml.KNearest_create()

    def fit(self, train_data=None, train_labels=None):
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        self.knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
        # self.svm.fit(train_data, train_labels)

    def classify(self, test_data):
        test_data = np.array(test_data)
        # results = self.svm.predict(test_data)
        # return results
        ret, results, neighbours, dist = self.knn.findNearest(test_data, self.k)
        return results
