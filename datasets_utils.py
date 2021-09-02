from torchvision import datasets, transforms
from PIL import Image
import json
import torch
import numpy as np

"""
Author: Khalid Ali 
Data: 25/8/2021
"""

def load_datasets(data_dir):
    """load train, valid, and test data 
    
    parameters:
    
    data_dir (String): path of the folder of data
    
    return:
    
    (DataLoader): trainloader
    (DataLoader): validloader
    (DataLoader): testloader
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.Resize(225), 
                                     transforms.CenterCrop(224), 
                                     transforms.RandomHorizontalFlip(), 
                                     transforms.RandomRotation(30),  
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    eval_transforms = transforms.Compose([transforms.Resize(225), 
                                          transforms.CenterCrop(224), 
                                          transforms.ToTensor(), 
                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir,transform = train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir,transform = eval_transforms)
    test_datasets = datasets.ImageFolder(test_dir,transform = eval_transforms)

    #  Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size = 32, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size = 32, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size = 32, shuffle = True)
    return trainloader, validloader, testloader

def get_cat_to_name(path):
    """load the category to name map
    parameters:
    
    path (String): path of the map
    
    return:
    
    (Map): the map 
    """
    with open(path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Done: Process a PIL image for use in a PyTorch model
    means =  [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    img = image.resize((256,256))
    img = img.crop((16,16,240,240))
    img  = np.array(img)
    img = img.transpose()/255
    for i,chan in enumerate(img):
        img[i] = (chan - means[i])/stds[i]
    img= torch.from_numpy(img)
    img = img.type(torch.FloatTensor)
    img = img.unsqueeze(0)
    return img

def read_img(path):
    """ read the image from directory
    
    parameters:
    
    path (String): the path of image
    
    return:
    (Tensor): img
    """
    img = Image.open(path)
    img = process_image(img)
    return img
  