import torch.nn as nn

def copy_parameters_from_prenfl(model, pretrained):
    model_dict = model.state_dict()
    pretrained_dict = pretrained['state_dict']

    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict and 'dense' not in k}
    model_dict.update(pretrained_dict)
    for k in pretrained_dict.keys():
        print(k)
    model.load_state_dict(model_dict)
    return model

def copy_parameters(model, pretrained):
    model_dict = model.state_dict()
    pretrained_dict = pretrained.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and pretrained_dict[k].size()==model_dict[k].size()}

    for k, v in pretrained_dict.items():
        print(k)

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def copy_parameters_from_vgg16(model, vgg16, copy_dense=False):
    for l1, l2 in zip(vgg16.features, model.features):
        if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
            assert l1.weight.size() == l2.weight.size()
            assert l1.bias.size() == l2.bias.size()
            l2.weight.data = l1.weight.data
            l2.bias.data = l1.bias.data

    if copy_dense:
        for i in [0, 3]:
            l1 = vgg16.classifier[i]
            l2 = vgg16.classifier[i]
            if isinstance(l1, nn.Linear) and isinstance(l2, nn.Linear):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
    return model

def copy_parameters_from_checkpoint(model, weights):
    model_dict = model.state_dict()
    pretrained_dict = weights

    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    for k in pretrained_dict.keys():
        print(k)
    model.load_state_dict(model_dict)
    return model