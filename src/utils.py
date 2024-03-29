import numpy as np
import os
import cv2
import csv
import time
import math


def video2image(video_file, image_path, interval=120):
    print("Begin: {}".format(video_file))
    os.makedirs(image_path, exist_ok=True)
    # Open the input movie file
    input_movie = cv2.VideoCapture(video_file)
    fps = int(round(input_movie.get(cv2.CAP_PROP_FPS)))
    interval = interval * fps / 30
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
    control = "/Users/midora/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/MyDocs/CS 7633/Project/UserStudy/control"
    app = "/Users/midora/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/MyDocs/CS 7633/Project/UserStudy/app"
    robot = "/Users/midora/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/MyDocs/CS 7633/Project/UserStudy/robot"
    for root, _, files in os.walk(control):
        for file in files:
            video2image(os.path.join(root, file), '../dataset/side/control')
    for root, _, files in os.walk(app):
        for file in files:
            video2image(os.path.join(root, file), '../dataset/side/app')
    for root, _, files in os.walk(robot):
        for file in files:
            video2image(os.path.join(root, file), '../dataset/side/robot')
    print("Done!")
