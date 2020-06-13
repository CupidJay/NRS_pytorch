from .uci_loader import *
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torchvision import datasets
import numbers
import numpy as np

def perm_mask(dd, dH, dW, nMul):
    #generating a mask for permutation into dH x dW x (dd*nMul) tensor
    m = np.random.permutation(dd)
    #m = np.arange(dd)
    for i in range(1, dH*dW*nMul):
        m = np.concatenate((m, np.random.permutation(dd)))
    return m


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def center_crop_with_flip(img, size, vertical_flip=False):
    crop_h, crop_w = size
    first_crop = F.center_crop(img, (crop_h, crop_w))
    if vertical_flip:
        img = F.vflip(img)
    else:
         img = F.hflip(img)
    second_crop = F.center_crop(img, (crop_h, crop_w))
    return (first_crop, second_crop)

class CenterCropWithFlip(object):
    def __init__(self, size, vertical_flip=False):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size
        self.vertical_flip = vertical_flip

    def __call__(self, img):
        return center_crop_with_flip(img, self.size, self.vertical_flip)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, vertical_flip={1})'.format(self.size, self.vertical_flip)


def get_dataset(cfg):
    '''
    param cfg: dataset_name, cub200 or uci_dataset
    return: pytorch train_loader, test_loader
    '''
    val_loader = None
    if cfg.DATASETS.NAME.startswith("cub"):
        im_size = cfg.INPUT.SIZE

        train_transforms = transforms.Compose([
            transforms.Resize(size=im_size),
            #transforms.Resize(size=256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=im_size),
            transforms.ToTensor(),
            normalize
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(size=im_size),
            transforms.CenterCrop(size=im_size),
            transforms.ToTensor(),
            normalize
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(im_size),
            CenterCropWithFlip(im_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
        ])

        root = '/opt/caoyh/datasets/cub200/'

        MyTrainData = datasets.ImageFolder(os.path.join(root, 'train'), transform=train_transforms)
        MyValData = datasets.ImageFolder(os.path.join(root, 'val'), transform=val_transforms)
        MyTestData = datasets.ImageFolder(os.path.join(root, 'val'), transform=test_transforms)
        '''

        MyTrainData = CUB200(root='/opt/Dataset/CUB_200_2011/CUB_200_2011/images', train=True,
                             transform=train_transforms)
        MyValData = CUB200(root='/opt/Dataset/CUB_200_2011/CUB_200_2011/images', train=False,
                            transform=val_transforms)
        MyTestData = CUB200(root='/opt/Dataset/CUB_200_2011/CUB_200_2011/images', train=False,
                            transform=test_transforms)
        #'''
        print(len(MyTrainData), len(MyTestData))

        train_loader = torch.utils.data.DataLoader(dataset=MyTrainData,
                                                   batch_size=cfg.SOLVER.TRAIN_PER_BATCH,
                                                   shuffle=True, num_workers=8, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(dataset=MyValData, batch_size=cfg.SOLVER.TEST_PER_BATCH,
                                                  num_workers=8, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=MyTestData, batch_size=cfg.SOLVER.TEST_PER_BATCH,
                                                  num_workers=8, pin_memory=True)
    else:
        load_data = 'Load_' + cfg.DATASETS.NAME
        x_train, y_train, x_val, y_val, x_test, y_test = eval(load_data)()

        MyTrainData = NormLoader(x_train, y_train)
        MyValData = NormLoader(x_val, y_val)
        MyTestData = NormLoader(x_test, y_test)

        train_loader = torch.utils.data.DataLoader(dataset=MyTrainData,
                                                        batch_size=cfg.SOLVER.TRAIN_PER_BATCH,
                                                        shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=MyValData,
                                                  batch_size=cfg.SOLVER.TEST_PER_BATCH)
        test_loader = torch.utils.data.DataLoader(dataset=MyTestData,
                                                       batch_size=cfg.SOLVER.TEST_PER_BATCH)
        val_loader = test_loader
    return train_loader, val_loader, test_loader
