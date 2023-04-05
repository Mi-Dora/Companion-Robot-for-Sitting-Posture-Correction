import numpy as np
import os
import cv2
import csv
import time
import math


def video2image(video_file, image_path, interval=50):
    os.makedirs(os.path.join(image_path, os.path.basename(video_file).split('.')[0]), exist_ok=True)
    # Open the input movie file
    input_movie = cv2.VideoCapture(video_file)
    cnt = 0
    while True:
        ret, frame = input_movie.read()
        if not ret:
            break
        if cnt % interval == 0:
            f_name = os.path.join(image_path, os.path.basename(video_file).split('.')[0] + '_%06d.png' % cnt)
            cv2.imwrite(f_name, frame)
        cnt += 1

    # All done!
    print(video_file + ' saved.')
    input_movie.release()


if __name__ == '__main__':
    for root, _, files in os.walk('../video'):
        for file in files:
            video2image(os.path.join(root, file), '../dataset')