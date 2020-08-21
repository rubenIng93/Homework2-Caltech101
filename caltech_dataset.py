from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''
        
        
        split_path = str('Caltech101/'+split+'.txt') # define the split path; in this case
                                                     # it can be only Caltech101/train.txt or Caltech101/test.txt
        
        # N.B: in in both txt files and 101Folder, BACKGROUND images are present -> avoid to select them!
        
        classes, class_to_idx = self._find_classes(self.root)
         
        split_array=[]
        with open(split_path, 'r') as f:
            split_array = f.readlines()
            
        f.close()
        
        images = [] # img list
        labels = [] # labels list
        
        count = 0
        #images = {} #dictionary k=index, v= image
        #img_lab = {} # dictionary k=index v=label
        for image in split_array:
            if image.split('/')[0].find('BACKGROUND') < 0:# filter that removes BACKGROUND 
                rgb = pil_loader(root+'/'+image.strip()) # e.g. Caltech101/101_ObjectCategories/accordion/image_0002.jpg
                #images[count] = rgb
                images.append(rgb)
                labels.append(image.split('/')[0])
                # img_lab[count] = image.split('/')[0]
                #count += 1
                
        #df = pd.DataFrame({'img':list(images.values()), 'label':list(img_lab.values())})
        #print(count)
        self.images = images
        #self.labels = img_lab
        self.labels = labels
        self.class_to_index = class_to_idx
        #self.count = count
        #self.dataframe = df
        #self.images = df['img']
        #self.labels = df['label']
        
        
    def train_val_split(self, train_size):

        train, val = [], []
        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size)
        for train_idx, val_idx in sss.split(self.images, self.labels):
            train.append(train_idx)
            val.append(val_idx)
        
        return train_idx, val_idx
        
    def get_class_int(self, class_name):
        
        return self.class_to_idx[class_name]
    #def get_df(self):
     #   return self.dataframe 
    
    
    def _find_classes(self, dir):
        # It find the class folder in the dataset
        
        classes = [d.name for d in os.scandir(dir) if d.is_dir() and d.name.find('BACKGROUND') < 0] # It removes the background folder
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        # class_to_idx is a dictionary where key= class_name , value= class_index
        
        return classes, class_to_idx
    
    

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image = self.images[index]
        label = self.class_to_index[self.labels[index]]
        #image = self.dataframe.loc[index, 'img']
        #label = self.class_to_index[self.dataframe.loc[index, 'label']]
        # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        
        return len(self.images)
