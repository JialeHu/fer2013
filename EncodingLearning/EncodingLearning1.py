
import os
os.chdir('/Users/hkh/Desktop/FR')
import EncodingDataset as dataset 
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from tqdm.notebook import tqdm


## Import Dataset
train_data = dataset.faceEncodingDataset('/Users/hkh/Desktop/data/Encodings_train',
                                         transform=transforms.Compose([dataset.ToTensor()]))
test_data = dataset.faceEncodingDataset('/Users/hkh/Desktop/data/Encodings_privatetest',
                                        transform=transforms.Compose([dataset.ToTensor()]))

## Load Data Batch
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

## Training
# Initialize parameters (weight & bias)
W = torch.randn(128, 7)/np.sqrt(128)
W = W.double()
W.requires_grad_()
b = torch.zeros(7, requires_grad=True)

# Optimizer
optimizer = torch.optim.SGD([W,b], lr=0.1)

# Iterate through train set minibatchs 
for item in tqdm(train_loader):
   
	# Zero out the gradients
    optimizer.zero_grad()
	
	# Forward pass
    x = item['Encoding']
    y = torch.matmul(x, W) + b
    cross_entropy = F.cross_entropy(y, item['LabelNum'])
	# Backward pass
    cross_entropy.backward()
    optimizer.step()

## Testing
correct = 0
total = len(test_data)

with torch.no_grad():
    # Iterate through test set minibatchs 
    for item in tqdm(test_loader):
        # Forward pass
        x = item['Encoding']
        y = torch.matmul(x, W) + b
        
        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == item['LabelNum']).float())
    
print('Test accuracy: {}'.format(correct/total))




