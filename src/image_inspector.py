import cv2
import os

path = '../dataset/side'

normal = 0
bad = 0
for root, _, files in os.walk(path):
    for file in files:
        img = cv2.imread(os.path.join(root, file))
        cv2.imshow("Inspector", img)
        while True:
            key = cv2.waitKey(0)
            if key == ord('a'):
                normal += 1
                print("{}, normal = {}".format(file, normal))
                break
            elif key == ord('s'):
                bad += 1
                print("{}, bad = {}".format(file, bad))
                break
        cv2.destroyWindow("Inspector")
print("Healthy Posture Ratio = ", normal / (normal+bad))
