
import os
os.chdir('/Users/hkh/Desktop/FR')
import EncodingDataset as dataset 
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm

class EncodingNet(nn.Module):

    def __init__(self):
        super(EncodingNet, self).__init__()

        # Layers
        self.lin = nn.Linear(128, 7)

        self.lin1 = nn.Linear(128, 64, bias=True)
        self.lin1_bn = nn.BatchNorm1d(64)
        self.lin2 = nn.Linear(64, 32, bias=True)
        self.lin2_bn = nn.BatchNorm1d(32)
        self.lin3 = nn.Linear(32, 16, bias=True)
        self.lin3_bn = nn.BatchNorm1d(64)
        self.lin4 = nn.Linear(16, 7, bias=True)

    def forward(self, x):

        """
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.softmax(self.lin4(x), dim=1)
        """
        x = self.lin(x)

        return x


print('Loading Dataset')
## Import Dataset
train_data = dataset.faceEncodingDataset('/Users/hkh/Desktop/data/Encodings_train',
                                         transform=transforms.Compose([dataset.ToTensor()]))
test_data = dataset.faceEncodingDataset('/Users/hkh/Desktop/data/Encodings_privatetest',
                                        transform=transforms.Compose([dataset.ToTensor()]))

## Load Data Batch
train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, num_workers=6)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False, num_workers=6)

## Training
# Instantiate model
model = EncodingNet()

model_parameters_dir = '/Users/hkh/Desktop/FR/Models/En2.pt'  
epochNum = 50

if os.path.exists(model_parameters_dir):
    model.load_state_dict(torch.load(model_parameters_dir))
    print('Model Parameters Loaded')
else:
    print('No Model Parameters File, Initiate New Model')

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

print('Training ...')

for epoch in range(epochNum):
    running_loss = 0.0
    
    # Iterate through train set minibatchs 
#   for item in tqdm(train_loader):
    for i, item in enumerate(train_loader, 0):
        # Zero out the gradients
        optimizer.zero_grad()
        
        # Forward pass
        x = item['Encoding']
        y = model(x.float())
        loss = criterion(y, item['LabelNum'])
        # Backward pass
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 200 mini-batches
            print('[%d/%d, %5d/%d] loss: %.3f' % 
                (epoch + 1, epochNum, i + 1, len(train_loader), running_loss / 200))
            running_loss = 0.0

    torch.save(model.state_dict(), model_parameters_dir)
    print('Model Parameters Saved')

print('Training Finished')

## Testing

correct = 0
total = len(train_data)

with torch.no_grad():
    # Iterate through test set minibatchs 
    for item in tqdm(train_loader):
        # Forward pass
        x = item['Encoding']
        y = model(x.float())
        
        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == item['LabelNum']).float())
    
print('Train Data accuracy: {}'.format(correct/total))


correct = 0
total = len(test_data)

with torch.no_grad():
    # Iterate through test set minibatchs 
    for item in tqdm(test_loader):
        # Forward pass
        x = item['Encoding']
        y = model(x.float())
        
        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == item['LabelNum']).float())
    
print('Test accuracy: {}'.format(correct/total))





