from torchvision import datasets, transforms, models
import torch
import numpy as np
from torch import optim
import PIL
import json


def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=50)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=50)
    
    return trainloader, validloader, testloader, train_data

def load_names(category_names_file):
    with open(category_names_file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def save_checkpoint(model, optimizer, train_data, epochs, arch, learning_rate, save_directory):
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'epoch': epochs,
                  'arch': arch,
                  'learning_rate': learning_rate,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),}

    torch.save(checkpoint, save_directory + 'checkpoint.pth')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    model =  getattr(models,checkpoint['arch'])(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
   
    model.classifier = checkpoint['classifier']    

    model.class_to_idx = checkpoint['class_to_idx']
    epochs = checkpoint['epoch']
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   
    return model, optimizer, epochs

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Open image 
    image = PIL.Image.open(image_path)

    # resize shortest side to 256
    if image.size[0] < image.size[1]: 
        new_size = (256, 256*image.size[1]/image.size[0])
    else: 
        new_size = (256*image.size[0]/image.size[1], 256)
    image.thumbnail(new_size)

    # crop to center 224 x 224
    left = (image.size[0] - 224)/2
    right = left + 224
    upper = (image.size[1] - 244)/2
    lower = upper + 244

    image = image.crop((left, upper, right, lower))

    # convert the image to a numpy array
    np_image = np.array(image)
    np_image = np_image/256

    normalize_means = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalize_means)/normalize_std

    np_image = np_image.transpose(2, 0, 1)
    return np_image

def predict(image_path, model, topk, cat_to_name, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # process image and convert from numpy to pytorchtensor
    np_image = process_image(image_path)       


    model.to(device)
    
    tensor_image = torch.from_numpy(np_image)
    tensor_image = tensor_image.float()
    tensor_image = tensor_image.unsqueeze(0)
    tensor_image = tensor_image.to(device)
    
    # make prediction
    model.eval()
    predictions = model.forward(tensor_image)
    predictions = torch.exp(predictions)
    
    top_probabilities, top_classes = predictions.topk(topk)
    
    top_probabilities = top_probabilities.cpu()
    top_probabilities = top_probabilities.detach().numpy().tolist()
    top_classes = top_classes.cpu()
    top_classes = top_classes.detach().numpy().tolist()
    top_flowers = [0 for x in range(topk)]
    
    inv_class_to_idx = {v: k for k, v in model.class_to_idx.items()}
    
    if cat_to_name != '':
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
        
        for i in range(topk):
            top_flowers[i] = cat_to_name[str(inv_class_to_idx[top_classes[0][i]])]
    
    return top_probabilities, top_classes, top_flowers