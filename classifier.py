from torch import nn
from torchvision import models

import torch
import json
import torch.nn.functional as F

"""
Author: Khalid Ali 
Data: 25/8/2021
"""


class Classifier(nn.Module):
    #arch_size = [input_size, output size]
    def __init__(self, arch_size, hidden_units):
        super().__init__()
       
        size = hidden_units
        self.fc1 = nn.Linear(arch_size[0], size )
        self.fc2 = nn.Linear(size, size * 2 // 3 )
        size = size * 2 // 3 
        self.out = nn.Linear(size, arch_size[1])
        
        self.dropout = nn.Dropout(p = 0.2)
        
    def forward(self, imgs):
        imgs = self.dropout(F.relu(self.fc1(imgs)))
        imgs = self.dropout(F.relu(self.fc2(imgs)))
        imgs = F.log_softmax(self.out(imgs),dim = 1)
        return imgs
    
def create_model(arch, output_size):
    """ create a new model of arch and output layer size
    
    parameters:
    
    arch (String): architecture of pre-trained model
    output_size (int): output layer units count
    
    return:
    (torch.models): model
    (int): input size of model's classifier
    """
    config = 'model = models.'+arch+'(pretrained = True)'
    exec(config,locals(),globals())
    size= [model.classifier[0].in_features , output_size]
    return model, size[0]

def build_model_from_checkpoints(arch, path):
    """ build model from checkpoints
    
    parameters:
    
    arch (String): the architecture of the model
    path (String): the path of checkpoints file
    """
#     print(path)
    checkpoints = torch.load(path)
    classi = Classifier([checkpoints['input_size'],checkpoints['output_size']],checkpoints['hidden_units'])
    config = 'model = models.'+arch+'(pretrained = True)'
    exec(config,locals(),globals())
    model.classifier = classi
    model.load_state_dict(checkpoints['state_dict'])
    return model, checkpoints['class_to_idx']
                                 
def save_model(path, model, in_size, out_size, cat_to_name_path, hidden_units):
    """ save the model
    
    parameters:
    path (String): the path to save the checkpoints to 
    model (torch.models): the model to be saved
    in_size (int): input layer size
    out_size (int): output layer size
    cat_to_name_path (String): path of categories map to store with checkpoints
    hidden_units (int): number of hidden units
    
    return:
    """
    with open(cat_to_name_path, 'r') as f:
        cat_to_name = json.load(f)
    model.to('cpu')
    checkpoints = {'input_size':in_size,
                   'output_size':out_size,
                   'hidden_units':hidden_units,
                   'class_to_idx':cat_to_name,
                   'state_dict':model.state_dict()
                    }
    torch.save(checkpoints, path)