from torch import nn
from torchvision import models

# Based on: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def initialize_model(model_name:str, num_classes:int, num_hidden:int, feature_extract:bool=True, use_pretrained:bool=True):
    # Initialize these variables which will be set in this if statement.
    model_ft = None

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = get_classifier_layer(num_ftrs, num_classes, num_hidden, 0.4)

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = get_classifier_layer(num_ftrs, num_classes, num_hidden, 0.4)

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = get_classifier_layer(num_ftrs, num_classes, num_hidden, 0.4)

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = get_classifier_layer(512, num_classes, num_hidden, 0.4)
        model_ft.num_classes = num_classes

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = get_classifier_layer(num_ftrs, num_classes, num_hidden, 0.4)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft


def set_parameter_requires_grad(model, feature_extracting:bool=True):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

            
# Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
def get_classifier_layer(num_ftrs, num_classes, num_hidden, dropout_p=0.4):
    return nn.Sequential(
        nn.Dropout(dropout_p),
        nn.Linear(num_ftrs, num_hidden),
        nn.ReLU(),
        nn.Linear(num_hidden, num_classes),
    )