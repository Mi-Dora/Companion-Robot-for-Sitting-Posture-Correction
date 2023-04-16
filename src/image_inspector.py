import cv2
import os

path = '../dataset/side/control'
try:
    with open("{}.log".format(path), 'r') as f:
        lines = f.readlines()
        normal = int(lines[1].split(':')[1])
        bad = int(lines[2].split(':')[1])
        total = int(lines[3].split(':')[1])
except FileNotFoundError:
    normal = 0
    bad = 0
    total = 0

for root, _, files in os.walk(path):
    files.sort()
    for i, file in enumerate(files):
        if file.startswith('.'):
            continue
        if i < total:
            continue

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
        with open("{}.log".format(path), 'w') as f:
            f.writelines(file + "\n")
            f.writelines("normal:{}\n".format(normal))
            f.writelines("bad:{}\n".format(bad))
            f.writelines("total:{}\n".format(normal+bad))
        cv2.destroyWindow("Inspector")
print("Healthy Posture Ratio = ", normal / (normal+bad))
