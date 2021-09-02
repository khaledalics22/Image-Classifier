from classifier import build_model_from_checkpoints
from datasets_utils import read_img
import torch, sys, os, argparse, json

"""
Author: Khalid Ali 
Data: 25/8/2021
    
    paramters:
    
        image_path        : input image directory 
        checkpoints        :  model checkpoints path 
        --arch          : architecture of model , vgg19, vgg16, or any other that has a layer "calssifier"
        --gpu           : used to ask for training in gpu mode
        --cat_to_name   : map that translate class to name 
        --topk   : number of classes to be displayed 
    
    """

def create_args():
    """ create the argument used in the command line
    return:
    
    ArgumentParser: arguments
    """
    parser = argparse.ArgumentParser(description = "Train A neural netowrk on datasets")
   
    parser.add_argument('image_path',
                        action = "store",
                        default = "flowers/test/10/image_07090.jpg",
                        help = "Image dirctory to be predicted")
    parser.add_argument('checkpoints',
                        action = "store",
                        default = 'saved_models/checkpoints.pth',
                        help = "path to checkpoints to load from")
    parser.add_argument('--arch',
                        action = "store",
                        default = "vgg19",
                        help = "The architecture of the model used e.g vgg19")
    
    parser.add_argument('--gpu',
                        action = "store_true",
                        default = False, 
                        help = "Whether to use GPU or Not")
    parser.add_argument('--cat_to_name', 
                        action = "store",
                        default = 'cat_to_name.json', 
                        help = "json file that map the categories to the actual names")
    parser.add_argument('--topk', 
                        action = "store",
                        default = 1, 
                        type = int, 
                        help = "json file that map the categories to the actual names")

    return parser.parse_args()     

def predict( checkpoints_path, image_path, arch, topk, gpu):
    """ predict the class of an input image
    
    parameters: 
    
    checkpoints_path (String): the path of checkpoints of  a model
    image_path (String): the path of input image
    arch (String) : the architecture used for the checkpoints
    topk (int): the top classes needed to be displayed
    gpu (booL): if use gpu mode  
    
    return: 
    
    (numpy): top_ps 
    (numpy): top_classes
    (map): class_to_idx -the category to name map
    """
    #build the model from the checkpoints file 
    model,cat_to_indx = build_model_from_checkpoints(arch, checkpoints_path )
    device = 'cuda' if gpu else 'cpu'
    # turn of grad
    torch.no_grad()
    # turn of dropout
    model.eval()
    img = read_img(image_path)
    model.to(device)
    img = img.to(device)
    # feed forward
    out = model(img)
    ps = torch.exp(out)
    # get top classes and probability
    top_p , top_class = ps.topk(topk, dim = 1)
    top_p , top_class = top_p .to('cpu'), top_class.to('cpu')
    
    return top_p.detach().numpy()[0], top_class.detach().numpy()[0], cat_to_indx







if __name__ == '__main__':
    """ the mian function used to predict the input image
    
    """
    
    # create the arguments of the command line
    args = create_args()  
    #predict image
    top_ps, top_classes, cat_to_name = predict(args.checkpoints ,args.image_path ,args.arch, args.topk, args.gpu) 
    # if no cat_to_name found in the checkpoints
    # load the one from the command line
    if cat_to_name == None:
        try:
            with open(args.cat_to_name, 'r') as f:
                cat_to_name = json.load(f)
        except:
            print('Failed to load cat_to_name')
            sys.exit()
    # parse classes to names       
    top_classes = [cat_to_name[f'{item}'] for item in top_classes]
    top_ps = ['{:.3f}'.format(item) for item in top_ps]
    print('Real type',cat_to_name['71'])
    
    # print the top k classes
    out = zip(top_classes, top_ps)
    out =  [(i, j) for (i, j) in out]
    out = sorted(out, key=lambda x:x[1],reverse = True)
    print(f'''Predicted as '{out[0][0]}' with probability of {out[0][1]}''' )