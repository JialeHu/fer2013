
import sys
sys.path.insert(0, '/Users/hjl/Desktop/FR/Datasets')
sys.path.insert(0, '/Users/hjl/Desktop/FR/Models')
# os.chdir('/Users/hkh/Desktop/FR/Datasets')
import ImageDataset as dataset 
import ImageNet4 as net

import os
import time
import torch
import torch.nn as nn
from torchvision import transforms
# from tqdm.notebook import tqdm


def ValidateModel(model, device, dataset, data_loader):

    model.eval()

    correct = 0
    total = len(dataset)
    with torch.no_grad():
        # Iterate through test set minibatchs 
        for i, item in enumerate(data_loader):
            # Forward pass
            x = item['Image'].to(device)
            y = model(x)
            
            predictions = torch.argmax(y, dim=1)
            correct += torch.sum((predictions == item['LabelNum'].to(device)).float())

            if i % 200 == 199:    # print every 200 mini-batches
                print('[%5d/%d]' % (i + 1, len(data_loader)))
    
    return correct/total


if __name__ == '__main__':
    
    ## Setting
    save_model_parameters = True
    validation = True
    model_parameters_dir = '/Users/hjl/Desktop/FR/Models/ImageNet4.pt'
    epochNum = 1
    batchSize = 64
    learningRate = 0.001

    # Print Settings
    print('- Epoch: %d, Batch Size: %d, Learning Rate: %.5f' % 
        (epochNum, batchSize, learningRate))

    if save_model_parameters:
        print('- Model Parameters Saving is On: \n  ' + model_parameters_dir)
    else:
        print('- Model Parameters Saving is Off')

    if validation:
        print('- Validation is On')
    else:
        print('- Validation is Off')

    ## Check Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('- Computing Device:')
    print(device)

    ## Instantiate Model
    model = net.ImageNet()
    model.to(device)

    # Print Model Stats
    train_params = model.num_train_parameters()
    print('- Number of Trainable Parameters: {}'.format(train_params))
    
    ## Import Dataset
    print('- Loading Dataset:')

    data_transform = transforms.Compose([
        dataset.ToTensor(),
        dataset.Normalize()
        ])
    
    train_data = dataset.faceImageDataset('/Users/hjl/Desktop/FR/Data/Training',
                                          transform=data_transform)
    print('-')
    test_data = dataset.faceImageDataset('/Users/hjl/Desktop/FR/Data/PrivateTest',
                                         transform=data_transform)
    print('-')
    testP_data = dataset.faceImageDataset('/Users/hjl/Desktop/FR/Data/PublicTest',
                                         transform=data_transform)

    ## Load Data Batch
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batchSize,
                                                shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batchSize, 
                                                shuffle=False, num_workers=4)
    testP_loader = torch.utils.data.DataLoader(testP_data, batch_size=batchSize, 
                                                shuffle=False, num_workers=4)

    ## Training
    model.train()

    # Load model parameters
    if os.path.exists(model_parameters_dir):
        model.load_state_dict(torch.load(model_parameters_dir, map_location=torch.device(device)))
        print('- Model Parameters Loaded')
    else:
        print('- No Model Parameters File, Initiate New Model')

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    print('- Training ...')
    start_time = time.time()

    for epoch in range(epochNum):
        running_loss = 0.0
        epoch_loss = 0.0
        
        # Iterate through train set minibatchs 
    #   for item in tqdm(train_loader):
        for i, item in enumerate(train_loader, 0):
            # Zero out the gradients
            optimizer.zero_grad()
            
            # Forward pass
            x = item['Image'].to(device)
            y = model(x)
            loss = criterion(y, item['LabelNum'].to(device))
            # Backward pass
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()
            # print every mini-batche
            if i % 200 == 199:    # print every 200 mini-batches
                print('[%d/%d, %5d/%d] loss: %.4f' % (epoch + 1, epochNum, 
                    i + 1, len(train_loader), running_loss / 200))
                running_loss = 0.0
              
        # Print every epoch                     
        if save_model_parameters:
             # Save Model per Epoch
            torch.save(model.state_dict(), model_parameters_dir)

            print('-----[%d/%d] loss: %.4f----- Model Parameters Saved' %
            (epoch + 1, epochNum, epoch_loss / len(train_loader)))
        else:
            print('-----[%d/%d] loss: %.4f----- Model Parameters Not Saved' %
            (epoch + 1, epochNum, epoch_loss / len(train_loader)))
        
    print('- Training Finished')
    print('Time Elapsed: %.2f s' % (time.time()-start_time))


    ## Testing
    if validation:

        print('- Testing ...')

        accTr = ValidateModel(model, device, train_data, train_loader)
        print('Train Data Accuracy: {}'.format(accTr))

        accPr = ValidateModel(model, device, test_data, test_loader)
        print('Private Test Data Accuracy: {}'.format(accPr))

        accPu = ValidateModel(model, device, testP_data, testP_loader)
        print('Public Test Data Accuracy: {}'.format(accPu))

        print('- Validation Finished')




