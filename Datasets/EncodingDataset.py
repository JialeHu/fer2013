
import os
import glob

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


class faceEncodingDataset(Dataset):
	"""Face Encodings dataset."""

	def __init__(self, encoding_dir, transform=None):

		extension = 'csv'
		os.chdir(encoding_dir)
		files = glob.glob('*.{}'.format(extension))		# List of csv files

		imageNames = np.empty([0,1], dtype=str)
		Encodings = np.empty([0,128], dtype=float)
		labelStr = np.empty([0,1], dtype=str)
		labelNum = np.empty([0,1], dtype=int)
		for filename in files:							# Loop through csv files
			imagenames, encodings, labelstr, labelnum = self.getEncoding(filename)

			imageNames = np.append(imageNames, imagenames, axis=0)
			Encodings = np.append(Encodings, encodings, axis=0)
			labelStr = np.append(labelStr, labelstr, axis=0)
			labelNum = np.append(labelNum, labelnum, axis=0)

		self.imageNames = imageNames
		self.Encodings = Encodings
		self.labelStr = labelStr
		self.labelNum = labelNum

		self.transform = transform


	def __len__(self):
		return len(self.Encodings)


	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		imageName = self.imageNames[idx, 0]
		Encoding = self.Encodings[idx, :]
		labelStr = self.labelStr[idx, 0]
		labelNum = self.labelNum[idx, 0]

		sample = {'Encoding': Encoding, 'LabelNum': labelNum}

		if self.transform:
			sample = self.transform(sample)

		return sample


	def getEncoding(self, csv_file):

		file = pd.read_csv(csv_file)
		data = file.to_numpy()
		
		encodingStr = data[:, 1]
		encodings = np.empty((0,128), dtype=float)	# initialize empty array
		nanIdx = np.empty([1,0], dtype=int)
		for idx, string in enumerate(encodingStr):	# loop thru rows in csv
			if type(string) is str:
				tempNum = np.fromstring(string, dtype=float, sep=' ').reshape(1,128)
				encodings = np.append(encodings, tempNum, axis=0)
			else:
				nanIdx = np.append(nanIdx, idx)

		imagename = data[:, 0]
		imageName = np.delete(imagename, nanIdx)	# delete nan image names
		imageName = imageName.reshape([len(imageName),1])

		label = csv_file.replace('.csv','')			# Expression String
		labelDic = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 
					'Sad': 4, 'Surprise': 5, 'Neutral': 6}

		labelStr = np.empty([len(encodings),1], dtype=object)
		labelStr[:] = label

		labelNum = np.empty([len(encodings),1], dtype=int)
		labelNum[:] = labelDic[label]

		return imageName, encodings, labelStr, labelNum


class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):
		Encoding, LabelNum = sample['Encoding'], sample['LabelNum']
		LabelNum = np.asarray(LabelNum)

		return {'Encoding': torch.from_numpy(Encoding),
				'LabelNum': torch.from_numpy(LabelNum)}



