from __future__ import print_function
import seaborn as sn
import pandas as pd
from collections import OrderedDict
import numpy as np
import sys, os
from PIL import Image

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models
import config as cf
from glob import glob

array = [[33,2,0,0,0,0,0,0,0,1,3],
        [3,31,0,0,0,0,0,0,0,0,0],
        [0,4,41,0,0,0,0,0,0,0,1],
        [0,1,0,30,0,6,0,0,0,0,1],
        [0,0,0,0,38,10,0,0,0,0,0],
        [0,0,0,3,1,39,0,0,0,0,4],
        [0,2,2,0,4,1,31,0,0,0,2],
        [0,1,0,0,0,0,0,36,0,2,0],
        [0,0,0,0,0,0,1,5,37,5,1],
        [3,0,0,0,0,0,0,0,0,39,0],
        [0,0,0,0,0,0,0,0,0,0,38]]
def to3channels(image_path):
    img = Image.open(image_path)
    imarray = np.array(img)
    stacked_img = np.stack((imarray,) * 3, axis=-1)
    three_channel_image = Image.fromarray(stacked_img, 'RGB')
    return three_channel_image


class_vector = os.listdir(r'Data/test/')
class_vector.sort()
model = models.alexnet(pretrained=True)
num_input_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_input_features, 10)

model.cuda()
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True
state_dict = torch.load('./checkpoint.pth')

new_state_dict = OrderedDict()

# Convert the model keys from gpu trained to cpu type so that we can load the dictionary into AlexNet
for k, v in state_dict.items():
    if 'module' not in k:
        k = 'module.'+k
    else:
        k = k.replace('module.module.module.', 'module.')
    new_state_dict[k]=v

# load the modified dictionary
model.load_state_dict(new_state_dict)
model.eval()

loader = transforms.Compose([transforms.Scale(cf.crop_size), transforms.ToTensor()])

# Convert the image from single to 3 channels
files=[]
for class_name in os.listdir(r'Data/test/'):
    files += glob(r'Data/test/{}/*.jpg'.format(class_name))

scores = np.zeros(shape=(len(class_vector),len(class_vector)))

for file in files:
    img = to3channels(file)  # Load image as PIL.Image
    x = loader(img)  # Preprocess image
    x = x.unsqueeze(0)  # Add batch dimension

    output = model(x)  # Forward pass
    pred = torch.argmax(output, 1)  # Get predicted class if multi-class classification
    # print('Sample {} predicted as {}'.format(file.split('/')[2], class_vector[pred]))
    scores[class_vector.index(file.split('/')[2]), pred] += 1

df_cm = pd.DataFrame(scores, index=[i for i in class_vector],
                  columns=[i for i in class_vector])
plt.figure(figsize=(10, 10))
sn.heatmap(df_cm, annot=True)
plt.yticks(rotation=0)
plt.show()
print('hold')