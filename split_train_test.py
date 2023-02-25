# # Creating Train / Val / Test folders (One time use)
import os
import numpy as np
import shutil
import random

root_dir = "/home/saadi09/fInal_face_verification/"  # data root path
data_path = 'images'
classes_dir = os.listdir(data_path)
print(classes_dir)
test_ratio = 0.05

for cls in classes_dir:
    os.makedirs(root_dir + 'dataset/train/' + cls)
    os.makedirs(root_dir + 'dataset/test/' + cls)

    src = data_path + '/' + cls
    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)

    train_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                               [int(len(allFileNames) * (1 - test_ratio))])

    train_FileNames = [src + '/' + name for name in train_FileNames.tolist()]
    test_FileNames = [src + '/' + name for name in test_FileNames.tolist()]

    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, root_dir + 'dataset/train/' + cls)

    for name in test_FileNames:
        shutil.copy(name, root_dir + 'dataset/test/' + cls)