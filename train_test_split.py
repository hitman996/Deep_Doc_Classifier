import os
from os.path import join
from glob import glob
from shutil import copyfile, copy2
from pathlib import Path

dataset_path = Path(r'Data/Tobacco3482-jpg')
classes = os.listdir(dataset_path)
train_folder = os.path.join('Data', 'train')
test_folder = os.path.join('Data', 'test')

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

split_ratio = 0.7

# remove any hidden elements such as folders starting with dot.
if '.' in classes[0]:
    classes.pop(0)

for class_name in classes:
    print('[INFO] Processing class : ', class_name)
    file_names = glob(join(dataset_path, class_name, '*.jpg'))
    os.makedirs(join(train_folder, class_name), exist_ok=True)
    os.makedirs(join(test_folder, class_name), exist_ok=True)
    for file in range(round(len(file_names)*split_ratio)):
        copy2(file_names[file], join(train_folder, class_name))

    for file in range(round(len(file_names)*split_ratio), len(file_names)):
        copy2(file_names[file], join(test_folder, class_name))

        
print('[INFO] Job done . . . ')