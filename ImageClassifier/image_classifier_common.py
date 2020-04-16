import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from collections import OrderedDict
from datetime import datetime
from PIL import Image
import numpy as np
import json
DATE_FORMAT = "%H:%M:%S"

ARCH_INPUT_LAYER = {
    "vgg16": 25088
}

with open('cat_to_name.json', 'r') as f:
    CAT_TO_NAME = json.load(f)


COMMON_NORMALIZE_TRANSFORM = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

NUM_OUT_CLASSES = len(CAT_TO_NAME)

def get_train_transforms():
    return transforms.Compose(transforms=[transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          COMMON_NORMALIZE_TRANSFORM])


def get_test_valid_transforms():
    return transforms.Compose(transforms=[transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          COMMON_NORMALIZE_TRANSFORM])


def get_data_set(root_data_dir, sub_dir, selected_transform):
    """

    :param root_data_dir: flowers root data dir
    :param sub_dir: train, valid or test
    :param selected_transform: train_transforms for train sub_dir and test_valid_transforms for valid or test
    :return:
    """
    selected_dir = root_data_dir + '/' + sub_dir
    return datasets.ImageFolder(root=selected_dir, transform=selected_transform)


def get_train_loader(root_data_dir):
    """
    :param root_data_dir: flowers root data dir
    """
    tarin_dataset = get_data_set(root_data_dir, "train", get_train_transforms())
    return DataLoader(dataset=tarin_dataset, batch_size=32, shuffle=True)


def get_valid_loader(root_data_dir):
    """
    :param root_data_dir: flowers root data dir
    """
    valid_dataset = get_data_set(root_data_dir, "valid", get_test_valid_transforms())
    return DataLoader(dataset=valid_dataset, batch_size=32, shuffle=False)


def get_test_loader(root_data_dir):
    """
    :param root_data_dir: flowers root data dir
    """
    test_dataset = get_data_set(root_data_dir, "test", get_test_valid_transforms())
    return DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)


def get_device(is_gpu_selected):
    """
    :param is_gpu_selected:
    :return:
    """
    if is_gpu_selected is False:
        return "cpu"
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(arch, device, hidden_layer_in, hidden_layer_out, lr_value):

    model = None
    if arch is None:
        model = models.vgg16(pretrained=True)
    else:
        model_eval_str="models." + arch+"(pretrained=True)"
        print("model eval: {} ".format( model_eval_str))
        model = eval(model_eval_str)

    model = models.vgg16(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # setting classifier of VGG model
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(in_features=ARCH_INPUT_LAYER.get(arch, ARCH_INPUT_LAYER["vgg16"]), out_features=hidden_layer_in)),
        ('relu1', nn.ReLU(inplace=True)),
        ('dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(in_features=hidden_layer_in, out_features=hidden_layer_out)),
        ('relu2', nn.ReLU(inplace=True)),
        ('dropout2', nn.Dropout(p=0.5)),
        ('fc3', nn.Linear(hidden_layer_out, NUM_OUT_CLASSES)),
        ('output', nn.LogSoftmax(dim=1))]), )

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr_value)
    model.to(device)
    print("model: {} ", model)

    return model, criterion, optimizer


def train_model(device,  model, criterion, optimizer , trainloader, validloader, epochs):
    steps = 0
    print_every = 10
    running_loss = 0
    print("Training - Start. Epochs: {} ".format(epochs))
    for epoch in range(epochs):
        print("{} - epoch: {} starts....".format(datetime.now().strftime(DATE_FORMAT), epoch + 1))
        print("--------------------------")
        running_loss = 0
        for inputs_train, labels_train in trainloader:
            steps += 1

            inputs_train, labels_train = inputs_train.to('cuda'), labels_train.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs_train = model.forward(inputs_train)
            train_loss = criterion(outputs_train, labels_train)
            train_loss.backward()
            optimizer.step()

            running_loss += train_loss.item()

            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                valid_accuracy = 0
                with torch.no_grad():
                    for inputs_valid, labels_valid in validloader:
                        optimizer.zero_grad()
                        inputs_valid, labels_valid = inputs_valid.to(device), labels_valid.to(device)
                        model.to(device)

                        outputs_valid = model.forward(inputs_valid)
                        valid_loss = criterion(outputs_valid, labels_valid)
                        ps = torch.exp(outputs_valid).data
                        equality = (labels_valid.data == ps.max(1)[1])
                        valid_accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("{},".format(datetime.now().strftime(DATE_FORMAT)),
                      "Epoch: {}/{}... ".format(epoch + 1, epochs),
                      "Training Running Loss: {:.4f}".format(running_loss / print_every),
                      "Validation Lost {:.4f}".format(valid_loss / len(validloader)),
                      "Accuracy: {:.4f}".format(valid_accuracy / len(validloader)))
                running_loss = 0

    print("Training - All Done")
    return model

def test_network(device, model, testloader):
    true_prediction = 0
    total_num_of_labels = 0
    model.to(device)
    print("{} - start test network ".format(datetime.now().strftime(DATE_FORMAT)))
    with torch.no_grad():
        for inputs_test, labels_test in testloader:
            inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
            outputs_test = model(inputs_test)
            _, predicted = torch.max(outputs_test.data, 1)

            total_num_of_labels += labels_test.size(0)
            true_prediction += (predicted == labels_test).sum().item()

    print(datetime.now().strftime(DATE_FORMAT), ' Accuracy of the network on the 10000 test images: %d %%' % (100 * true_prediction / total_num_of_labels))
    print("{} done test network ".format(datetime.now().strftime(DATE_FORMAT)))


def save_checkpoint(arch, model, train_dataset, checkpoint_file_name):
    """

    :param arch:
    :param model:
    :param train_dataset:
    :param checkpoint_file_name:
    :return:
    """
    model.class_to_idx = train_dataset.class_to_idx

    checkpoint = {'structure': arch ,
                  'input_size': ARCH_INPUT_LAYER.get(arch,ARCH_INPUT_LAYER.get("vgg16") ),
                  'output_size': NUM_OUT_CLASSES,
                  'classifier': model.classifier,
                  # 'hidden_layers': [each.out_features for each in model.hidden_layers],
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    #checkpoint_file_name = arch + '_image_classifier.pth'
    print("save checkpoint to {} ".format(checkpoint_file_name))
    torch.save(checkpoint, checkpoint_file_name)

def load_check_point(checkpoint_file_name):
    """
    :param checkpoint_file_name:
    :return:
    """
    checkpoint = torch.load(checkpoint_file_name)
    model = models.vgg16(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Load stuff from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return model


def process_image(image):
    """
    :param image: image_path
    :return:
    """

    # TODO: Process a PIL image for use in a PyTorch model

    proc_img = Image.open(image)
    prepoceess_img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pymodel_img = prepoceess_img(proc_img)
    return pymodel_img

def predict(device, image_path, model, topk=5):
    """
    :param device:
    :param image_path:
    :param model:
    :param topk:
    :return:
    """

    model.to(device)
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    with torch.no_grad():
        output = model.forward(img_torch.cuda())

    probability = F.softmax(output.data, dim=1)

    return probability.topk(topk)