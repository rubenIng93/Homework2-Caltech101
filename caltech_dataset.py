from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys

import numpy as np


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
         
            
        # Loading the split path exploiting Numpy
        split_array = np.loadtxt(split_path, dtype=str)
        
        images={}
        for image in split_array:
            if image.split('/')[0] in classes:# filter that removes BACKGROUND 
                rgb = pil_loader(root+'/'+image)
                images[rgb] = class_to_idx[image.split('/')[0]]
                
                
        self.dataset = images
        self.class_to_index = class_to_idx
    
        
        
    
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

        image = self.dataset.keys()[index]
        label = self.dataset[image]
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
        length = len(self.dataset.keys())
        return length
