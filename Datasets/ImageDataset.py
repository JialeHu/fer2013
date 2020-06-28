
import os
import numpy as np
from skimage import io
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob


class faceImageDataset(Dataset):
    """Face Images dataset."""

    def __init__(self, image_dir, transform=None):

        dirpath, dirnames, _ = next(os.walk(image_dir))

        labelDic = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 
                 'Sad': 4, 'Surprise': 5, 'Neutral': 6}

        imageNames = []
        imagePaths = []
        labelStr = np.empty([0,1], dtype=str)
        labelNum = np.empty([0,1], dtype=int)

        for dirname in dirnames:

            img_dir = dirpath + '/' + dirname
            
            # _, _, img_names = next(os.walk(img_dir))
            os.chdir(img_dir)
            img_names = glob.glob('*.{}'.format('jpg'))
            
            img_paths = [img_dir + '/' + img_name for img_name in img_names]
            print(img_dir, len(img_paths))

            label_str = np.empty([len(img_names),1], dtype=object)
            label_str[:] = dirname
            
            label_num = np.empty([len(img_names),1], dtype=int)
            label_num[:] = labelDic[dirname]
            
            imageNames = np.append(imageNames, img_names, axis=0)
            imagePaths = np.append(imagePaths, img_paths, axis=0)
            labelStr = np.append(labelStr, label_str, axis=0)
            labelNum = np.append(labelNum, label_num, axis=0)

        self.imageNames = imageNames
        self.imagePaths = imagePaths
        self.labelStr = labelStr
        self.labelNum = labelNum

        self.transform = transform

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imageName = self.imageNames[idx]
        imagePath = self.imagePaths[idx]
        labelStr = self.labelStr[idx, 0]
        labelNum = self.labelNum[idx, 0]

        try:
            image = io.imread(imagePath, as_gray=True)    # 2D ndarray
        except:
            print('imread error file:' + imagePath + '\n')

        # image = image.reshape([1, len(image), len(image)])
        image = image.reshape([len(image), len(image), 1])
        
        sample = {'Image': image, 'LabelNum': labelNum, 'ImageName': imageName}

        if self.transform:
            sample = self.transform(sample)

        return sample



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        Image, LabelNum, ImageName = sample['Image'], sample['LabelNum'], sample['ImageName']
        LabelNum = np.asarray(LabelNum)

        tf = transforms.ToTensor()

        return {'Image': tf(Image),
                'LabelNum': torch.from_numpy(LabelNum),
                'ImageName': ImageName}

class Normalize(object):
    """Normalize Image Tensors."""

    def __call__(self, sample):
        Image, LabelNum, ImageName = sample['Image'], sample['LabelNum'], sample['ImageName']

        tf = transforms.Normalize([0.5], [0.5])

        return {'Image': tf(Image),
                'LabelNum': LabelNum,
                'ImageName': ImageName}



