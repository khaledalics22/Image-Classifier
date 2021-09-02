from classifier import create_model, save_model , build_model_from_checkpoints, Classifier
from datasets_utils import load_datasets
from workspace_utils import active_session
from torch import nn, optim
import torch
import argparse
import sys
import os

""" Author: Khalid Ali Abdualwahab 
    Data: 25/8/`
    
    
    paramters:
    
        data_dir        : folder directory of data load, valid, and test
        save_dir        : directory to save model checkpoints to 
        --arch          : architecture of model , vgg19, vgg16, or any other that has a layer "calssifier"
        --learning_rate : learining rate of model
        --hidden_units  : hidden units count used as parameter in model architecture
        --epochs        : epochs count 
        --gpu           : used to ask for training in gpu mode
        --cat_to_name   : map that translate class to name 
        --cat_outputs   : number of outputs of layer
    
    """

def test(model, testloader, gpu = False ):
    """this function is used to test the model and print the accuracy 

    paramters: 
    model (torch.models): the trained model
    testloader (DataLoader): the loader that holds the test images
    gpu (bool): if use gpu to test
    
    return:
    
    """
    device = 'cuda' if gpu else 'cpu'
    # move model to cpu or gpu
    model.to(device)
    model.eval()
    accuracy = 0
    test_loss = 0
    criterion = nn.NLLLoss()
    
    # test the images batch by batch 
    for imgs, labels in testloader:
        imgs,labels =imgs.to(device),labels.to(device)
        # feed forward
        logits = model(imgs)
        # calculate loss
        test_loss += criterion(logits, labels)
        # get probabilities of the output
        ps = torch.exp(logits)
        # get top class
        top_p, top_class = ps.topk(1, dim = 1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
    else:
        
        print( "Test Loss: {:.3f}".format(test_loss/len(testloader)),
             "Test accuracy: {:.3f}".format(accuracy/len(testloader)*100))
    model.train()

def train(arch, output_size, hidden_units, lr, epochs, data_dir, gpu):
    """load datasets , create model, and train the model
    
    parameters: 
    arch (String): the architecture used for the model e.g vgg19 or vgg16
    output_size (int):the output layer units count of the model 
    hidden_units (int): the hidden_units count for a hidden layer
    lr (float): the learning rate of the model
    epochs (int): number of epochs used to train the model
    data_dir (String): the images folder directory
    gpu (bool): if to use gpu to train the model
    
    return:
    troch.models: model
    int: input_size
    """
    
    
    trainloader, validloader, testloader = load_datasets(data_dir)
    print("building model...")
    model, input_size = create_model(arch, output_size)
    
    # freeze the parameters of the model 
    for param in model.parameters():
        param.requires_grad = False
        
    # create our classifier
    classifier = Classifier([input_size, output_size], hidden_units)
    
    #update model classifier
    model.classifier = classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr =lr)
    criterion = nn.NLLLoss()
    device = 'cuda' if gpu else 'cpu'
    # move model to gpu or cpu
    model.to(device)
    print(f"training model in {device} mode for {epochs} epochs...")
    # train the model for number of epochs
    for i in range(epochs):
        run_loss = 0
        # train the model batch by batch 
        for imgs, labels in trainloader:
            imgs,labels =imgs.to(device),labels.to(device)
            optimizer.zero_grad()       
            # feed forward
            logits = model(imgs)
            # calculate the loss 
            loss = criterion(logits,labels)
            # backpropagation to calculate the grads 
            loss.backward()
            # update our weights and bias 
            optimizer.step()
            run_loss += loss
        else:
            valid_loss = 0
            accuracy = 0
            # turn of grad for validation
            with torch.no_grad():
                model.eval()
                # loop over the validation data batch by batch 
                # to calculate the validation accuracy and loss 
                for imgs,labels in validloader:
                    imgs,labels =imgs.to(device),labels.to(device)
                    logits = model(imgs)
                    valid_loss += criterion(logits, labels)
                    ps = torch.exp(logits)
                    top_p, top_class = ps.topk(1, dim = 1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
                  
                else:
                    print(f"epoch : {i+1} \n",
                        "Train Loss: {:.3f}".format(run_loss/len(trainloader)),
                         "Valid Loss: {:.3f}".format(valid_loss/len(validloader)),
                         "Valid accuracy: {:.3f}".format(accuracy/len(validloader)*100))

            model.train()
    print("testing trained model...")
    
    # test the model on the test data 
    test(model, testloader, gpu)
    return model, input_size

def create_args():
    """ create the argument used in the command line
    parameters:
    
    
    return:
    
    ArgumentParser: arguments
    """
    parser = argparse.ArgumentParser(description = "Train A neural netowrk on datasets")
    parser.add_argument('data_dir',
                        action = "store",
                        help = "director of dataset main folder")
    parser.add_argument('--save_dir',
                        action = "store",
                        default = "saved_models/checkpoints.pth",
                        help = "The Directory to save the checkpoints of network, must end with .pth")
    parser.add_argument('--arch',
                        action = "store",
                        default = "vgg19",
                        help = "The architecture of the model used e.g vgg19")
    parser.add_argument('--learning_rate',
                        type = float, 
                        action = "store",
                        default = 0.001,
                        help = "learining rate of the model e.g 0.001")
    parser.add_argument('--hidden_units', 
                        type = int, 
                        action = "store",
                        default = 512,
                        help = "Number of units in a hidden layer inside the architecture")
    parser.add_argument('--epochs', 
                        type = int, 
                        action = "store",
                        default = 5, 
                        help = "Number of epochs for training network")
    parser.add_argument('--gpu',
                        action = "store_true",
                        default = False, 
                        help = "Whether to use GPU or Not")
    parser.add_argument('--cat_to_name', 
                        action = "store",
                        default = 'cat_to_name.json', 
                        help = "json file that map the categories to the actual names")
    parser.add_argument('--cat_outputs', 
                        action = "store",
                        type = int, 
                        default = 102, 
                        help = "number of output categories")
    return parser.parse_args()                    
                        
if __name__ == '__main__':
    """ main function that train and save the model
    
    """
    try:
        # creat the arguments of command line
        args = create_args()
        data_dir = str(args.data_dir)
        # check if data dirctory valid 
        if not os.path.isdir(data_dir):
            print("Invalid Data Directory Format")
            sys.exit()
        # for this session
        with active_session():    
            # train the model
            model ,input_size = train(arch = args.arch, output_size = args.cat_outputs,
                                     hidden_units =  args.hidden_units, lr = args.learning_rate, 
                                      epochs=args.epochs, data_dir = data_dir, gpu=args.gpu)
            # save the model checkpoints
            save_model(args.save_dir, model, input_size, args.cat_outputs , args.cat_to_name , args.hidden_units)
    except:
        
        print("Invalid inputs")
        sys.exit()
    