from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from .data_process import preprocess, random_crop_or_pad
import imgviz
from labelme import utils
import json
import random

class BasicDataset(Dataset):
    def __init__(self,data_dir='../marine_data/'):
        self.trainset  = self.read_traindata_names(data_dir)
        self.num_train = len(self.trainset)
        self.labels=3

    def __len__(self):
        return 500

    def __getitem__(self, i):
        random_line = random.choice(self.trainset)
        image,truth_mask,lbl_viz = self.json2data(random_line)
        image = Image.fromarray(image.astype('uint8')).convert('RGB')
        truth_mask = Image.fromarray(truth_mask.astype('uint8'))
        image,truth_mask = preprocess(image,truth_mask)   
        print(np.max(truth_mask))
        # truth_mask=truth_mask+1
        image = image/255
        # truth_mask = (np.arange(self.labels) == truth_mask[...,None]-1).astype(int) # encode to one-hot-vector

        return {
            'image': torch.from_numpy(image.transpose((2,0,1))).type(torch.FloatTensor),
            'mask': torch.from_numpy(truth_mask).type(torch.FloatTensor)
        }
    def read_traindata_names(self,data_dir):
        trainset=[]
        for i in range(12):
            find_dir = data_dir + str(i+1) + '/images/'
            files = self.find_target_file(find_dir,'.json')
            trainset+=files
        return trainset

    def json2data(self, json_file):
        data = json.load(open(json_file))
        imageData = data.get('imageData')
        img = utils.img_b64_to_arr(imageData)

        label_name_to_value = {'_background_': 0}
        for shape in sorted(data['shapes'], key=lambda x: x['label']):
            label_name = shape['label']
            if label_name in label_name_to_value:
               label_value = label_name_to_value[label_name]
            else:
               label_value = len(label_name_to_value)
               label_name_to_value[label_name] = label_value
        lbl, _ = utils.shapes_to_label(
           img.shape, data['shapes'], label_name_to_value
        )
        label_names = [None] * (max(label_name_to_value.values()) + 1)
        for name, value in label_name_to_value.items():
            label_names[value] = name

        lbl_viz = imgviz.label2rgb(
            label=lbl, img=imgviz.asgray(img), label_names=label_names, loc='rb'
        )
        return img,lbl,lbl_viz
    def find_target_file(self,find_dir,format_name):
        files= [find_dir+file for file in listdir(find_dir) if file.endswith(format_name)]
        return files