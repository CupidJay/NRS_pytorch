import os
import numpy as np
import logging
from sklearn.datasets import load_svmlight_files, load_svmlight_file
import torch
import torchvision
import torch.utils.data as data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class NormLoader(data.Dataset):
    def __init__(self, matrix, labels):
        self.matrix = matrix
        self.labels = torch.tensor(labels,dtype=torch.long)

    def __getitem__(self, index):
        item_data = self.matrix[index]
        return item_data, self.labels[index]

    def __len__(self):
        return len(self.matrix)


class VectorLoader(data.Dataset):
    def __init__(self, matrix, labels, mask, dH=2, dW=2, nMul=5):
        self.matrix = matrix
        self.labels = torch.tensor(labels,dtype=torch.long)
        self.dd = matrix.shape[1]
        self.dH = dH
        self.dW = dW
        self.nMul = nMul
        self.mask = mask

    def __getitem__(self, index):
        item_data = self.matrix[index]
        #permute feature vector and concatenate
        #return item_data, self.labels[index]

        item_data = item_data[self.mask]

        item_data = np.reshape(item_data, (self.dd*self.nMul, self.dH, self.dW))
        item_data = np.transpose(item_data, (1, 2, 0))
        #item_data = np.reshape(item_data, (self.dH, self.dW, self.dd*self.nMul))
        return torchvision.transforms.ToTensor()(item_data), self.labels[index]

    def __len__(self):
        return len(self.matrix)

def Load_satimage():
    logger = logging.getLogger(__name__)
    logger.info("start loadding satimage file")
    # Download these 2 files from the LIBSVM dataset page
    x_train, y_train, x_test, y_test = load_svmlight_files(
        ("./datasets/satimage/satimage.scale", "./datasets/satimage/satimage.scale.t"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x_train = x_train.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)
    y_train = y_train - 1
    y_test = y_test - 1

    x_val, y_val = x_test, y_test
    logger.info("x_train size is {}, x_test size is {}".format(x_train.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_letter():
    logger = logging.getLogger(__name__)
    logger.info("start loadding letter file")
    #Download these 2 files from the LIBSVM dataset page
    x_train, y_train, x_test, y_test = load_svmlight_files(
        ("./datasets/letter/letter.scale", "./datasets/letter/letter.scale.t"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x_train = x_train.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)
    y_train = y_train - 1
    y_test = y_test - 1
    x_val, y_val = x_test, y_test
    logger.info("x_train size is {}, x_test size is {}".format(x_train.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_usps():
    logger = logging.getLogger(__name__)
    logger.info("start loadding usps file")
    #Download these 2 files from the LIBSVM dataset page
    x_train, y_train, x_test, y_test = load_svmlight_files(
        ("./datasets/usps/usps", "./datasets/usps/usps.t"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x_train = x_train.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)
    y_train = y_train - 1
    y_test = y_test - 1
    x_val, y_val = x_test, y_test
    logger.info("x_train size is {}, x_test size is {}".format(x_train.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_mnist():
    logger = logging.getLogger(__name__)
    logger.info("start loading mnist file")
    x_train, y_train, x_test, y_test = load_svmlight_files(("./datasets/mnist/mnist", "./datasets/mnist/mnist.t"))

    x_train = x_train.toarray().astype(np.float32) / 255.0
    x_test = x_test.toarray().astype(np.float32) / 255.0

    logger.info("x_train size is {}, x_test size is {}".format(x_train.shape, x_test.shape))
    x_val, y_val = x_test, y_test
    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_gisette():
    logger = logging.getLogger(__name__)
    logger.info("start loading mnist file")
    x_train, y_train, x_test, y_test = load_svmlight_files(("./datasets/gisette/gisette_scale", "./datasets/gisette/gisette_scale.t"))
    x_train = x_train.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)
    y_train = (y_train + 1) // 2
    y_test = (y_test + 1) // 2

    x_val, y_val = x_test, y_test

    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_weatherAUS():
    logger = logging.getLogger(__name__)
    logger.info("start loading weatherAUS file")

    dataset = pd.read_csv('datasets/weatherAUS/weatherAUS.csv')

    dataset.drop(labels=['Date', 'Location', 'Evaporation', 'Sunshine', 'Cloud3pm', 'Cloud9am', 'RISK_MM'],
                 axis=1, inplace=True)

    dataset['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
    dataset['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)

    dataset.dropna(inplace=True)

    print(dataset.shape)

    categorical = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
    dataset = pd.get_dummies(dataset, columns=categorical, drop_first=True)



    sc = StandardScaler()

    x = dataset.drop(labels=['RainTomorrow'], axis=1)
    y = dataset['RainTomorrow'].values



    x_train, x_test, y_train, \
    y_test = train_test_split(x, y, test_size=0.2, random_state=40)

    sc.fit(x_train)

    x_train, x_test = sc.transform(x_train), sc.transform(x_test)

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    return x_train, y_train, x_test, y_test

def Load_dna():
    logger = logging.getLogger(__name__)
    logger.info("start loading DNA file")
    # Download these 2 files from the LIBSVM dataset page
    x_train, y_train, x_val, y_val, x_test, y_test = load_svmlight_files(
        ("./datasets/dna/dna.scale.tr",
         "./datasets/dna/dna.scale.val",
         "./datasets/dna/dna.scale.t"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x_train = x_train.toarray().astype(np.float32)
    x_val = x_val.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)
    y_train = y_train - 1
    y_val = y_val-1
    y_test = y_test - 1
    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_ijcnn1():
    logger = logging.getLogger(__name__)
    logger.info("start loading ijcnn1 file")
    # Download these 2 files from the LIBSVM dataset page
    x_train, y_train, x_val, y_val, x_test, y_test = load_svmlight_files(
        ("./datasets/ijcnn1/ijcnn1.tr",
         "./datasets/ijcnn1/ijcnn1.val",
         "./datasets/ijcnn1/ijcnn1.t"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x_train = x_train.toarray().astype(np.float32)
    x_val = x_val.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)
    y_train = (y_train+1)//2
    y_val = (y_val+1)//2
    y_test = (y_test+1)//2
    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_shuttle():
    logger = logging.getLogger(__name__)
    logger.info("start loading shuttle file")
    # Download these 2 files from the LIBSVM dataset page
    x_train, y_train, x_val, y_val, x_test, y_test = load_svmlight_files(
        ("./datasets/shuttle/shuttle.scale.tr",
         "./datasets/shuttle/shuttle.scale.val",
         "./datasets/shuttle/shuttle.scale.t"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x_train = x_train.toarray().astype(np.float32)
    x_val = x_val.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)
    y_train = y_train - 1
    y_val = y_val-1
    y_test = y_test - 1
    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_protein():
    logger = logging.getLogger(__name__)
    logger.info("start loading protein file")
    # Download these 2 files from the LIBSVM dataset page
    x_train, y_train, x_val, y_val, x_test, y_test = load_svmlight_files(
        ("./datasets/protein/svm-protein.tr",
         "./datasets/protein/svm-protein.val",
         "./datasets/protein/svm-protein.t"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x_train = x_train.toarray().astype(np.float32)
    x_val = x_val.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)

    sc = StandardScaler()
    sc.fit(x_train)
    x_train, x_val, x_test = sc.transform(x_train), sc.transform(x_val), sc.transform(x_test)

    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_pendigits():
    logger = logging.getLogger(__name__)
    logger.info("start loading pendigits file")
    # Download these 2 files from the LIBSVM dataset page
    x_train, y_train, x_val, y_val, x_test, y_test = load_svmlight_files(
        ("./datasets/pendigits/pendigits",
         "./datasets/pendigits/pendigits.t",
         "./datasets/pendigits/pendigits.t"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x_train = x_train.toarray().astype(np.float32)
    x_val = x_val.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)

    sc = StandardScaler()
    sc.fit(x_train)
    x_train, x_val, x_test = sc.transform(x_train), sc.transform(x_val), sc.transform(x_test)

    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_madelon():
    logger = logging.getLogger(__name__)
    logger.info("start loading madelon file")
    # Download these 2 files from the LIBSVM dataset page
    x_train, y_train, x_val, y_val, x_test, y_test = load_svmlight_files(
        ("./datasets/madelon/madelon",
         "./datasets/madelon/madelon.t",
         "./datasets/madelon/madelon.t"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x_train = x_train.toarray().astype(np.float32)
    x_val = x_val.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)

    y_train = (y_train + 1) // 2
    y_val = (y_val + 1) // 2
    y_test = (y_test + 1) // 2

    sc = StandardScaler()
    sc.fit(x_train)
    x_train, x_val, x_test = sc.transform(x_train), sc.transform(x_val), sc.transform(x_test)

    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_poker():
    logger = logging.getLogger(__name__)
    logger.info("start loading poker file")
    # Download these 2 files from the LIBSVM dataset page
    x_train, y_train, x_val, y_val, x_test, y_test = load_svmlight_files(
        ("./datasets/poker/poker",
         "./datasets/poker/poker.t",
         "./datasets/poker/poker.t"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x_train = x_train.toarray().astype(np.float32)
    x_val = x_val.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)

    y_train = (y_train + 1) // 2
    y_val = (y_val + 1) // 2
    y_test = (y_test + 1) // 2

    sc = StandardScaler()
    sc.fit(x_train)
    x_train, x_val, x_test = sc.transform(x_train), sc.transform(x_val), sc.transform(x_test)

    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_SVHN():
    logger = logging.getLogger(__name__)
    logger.info("start loading SVHN file")
    # Download these 2 files from the LIBSVM dataset page
    x_train, y_train, x_val, y_val, x_test, y_test = load_svmlight_files(
        ("./datasets/SVHN/SVHN.scale",
         "./datasets/SVHN/SVHN.scale.t",
         "./datasets/SVHN/SVHN.scale.t"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x_train = x_train.toarray().astype(np.float32)
    x_val = x_val.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)

    y_train = y_train - 1
    y_val = y_val - 1
    y_test = y_test - 1

    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_svmguide3():
    logger = logging.getLogger(__name__)
    logger.info("start loading svmguide3 file")
    # Download these 2 files from the LIBSVM dataset page
    x_train, y_train, x_val, y_val, x_test, y_test = load_svmlight_files(
        ("./datasets/svmguide3/svmguide3",
         "./datasets/svmguide3/svmguide3.t",
         "./datasets/svmguide3/svmguide3.t"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x_train = x_train.toarray().astype(np.float32)
    x_val = x_val.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)

    y_train = (y_train + 1) // 2
    y_val = (y_val + 1) // 2
    y_test = (y_test + 1) // 2

    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_splice():
    logger = logging.getLogger(__name__)
    logger.info("start loading splice file")
    # Download these 2 files from the LIBSVM dataset page
    x_train, y_train, x_val, y_val, x_test, y_test = load_svmlight_files(
        ("./datasets/splice/splice",
         "./datasets/splice/splice.t",
         "./datasets/splice/splice.t"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x_train = x_train.toarray().astype(np.float32)
    x_val = x_val.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)

    y_train = (y_train + 1) // 2
    y_val = (y_val + 1) // 2
    y_test = (y_test + 1) // 2

    sc = StandardScaler()
    sc.fit(x_train)
    x_train, x_val, x_test = sc.transform(x_train), sc.transform(x_val), sc.transform(x_test)

    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_sensorless():
    logger = logging.getLogger(__name__)
    logger.info("start loading sensorless file")
    # Download these 2 files from the LIBSVM dataset page
    x_train, y_train, x_val, y_val, x_test, y_test = load_svmlight_files(
        ("./datasets/sensorless/Sensorless.scale.tr",
         "./datasets/sensorless/Sensorless.scale.val",
         "./datasets/sensorless/Sensorless.scale.val"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x_train = x_train.toarray().astype(np.float32)
    x_val = x_val.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)

    y_train = y_train - 1
    y_val = y_val - 1
    y_test = y_test - 1

    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_SensIT_acoustic():
    logger = logging.getLogger(__name__)
    logger.info("start loading SensIT_acoustic file")
    # Download these 2 files from the LIBSVM dataset page
    x_train, y_train, x_val, y_val, x_test, y_test = load_svmlight_files(
        ("./datasets/SensIT-acoustic/acoustic_scale",
         "./datasets/SensIT-acoustic/acoustic_scale.t",
         "./datasets/SensIT-acoustic/acoustic_scale.t"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x_train = x_train.toarray().astype(np.float32)
    x_val = x_val.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)

    y_train = y_train - 1
    y_val = y_val - 1
    y_test = y_test - 1

    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_SensIT_seismic():
    logger = logging.getLogger(__name__)
    logger.info("start loading SensIT_seismic file")
    # Download these 2 files from the LIBSVM dataset page
    x_train, y_train, x_val, y_val, x_test, y_test = load_svmlight_files(
        ("./datasets/SensIT-seismic/seismic_scale",
         "./datasets/SensIT-seismic/seismic_scale.t",
         "./datasets/SensIT-seismic/seismic_scale.t"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x_train = x_train.toarray().astype(np.float32)
    x_val = x_val.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)

    y_train = y_train - 1
    y_val = y_val - 1
    y_test = y_test - 1

    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_SensIT_combined():
    logger = logging.getLogger(__name__)
    logger.info("start loading SensIT_combined file")
    # Download these 2 files from the LIBSVM dataset page
    x_train, y_train, x_val, y_val, x_test, y_test = load_svmlight_files(
        ("./datasets/SensIT-combined/combined_scale",
         "./datasets/SensIT-combined/combined_scale.t",
         "./datasets/SensIT-combined/combined_scale.t"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x_train = x_train.toarray().astype(np.float32)
    x_val = x_val.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)

    y_train = y_train - 1
    y_val = y_val - 1
    y_test = y_test - 1

    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_vowel():
    logger = logging.getLogger(__name__)
    logger.info("start loading vowel file")
    # Download these 2 files from the LIBSVM dataset page
    x_train, y_train, x_val, y_val, x_test, y_test = load_svmlight_files(
        ("./datasets/vowel/vowel.scale",
         "./datasets/vowel/vowel.scale.t",
         "./datasets/vowel/vowel.scale.t"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x_train = x_train.toarray().astype(np.float32)
    x_val = x_val.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)

    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_mushrooms():
    logger = logging.getLogger(__name__)
    logger.info("start loading mushrooms file")
    # Download these 2 files from the LIBSVM dataset page
    x, y = load_svmlight_file(
        ("./datasets/mushrooms/mushrooms"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x = x.toarray().astype(np.float32)

    y = y - 1

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, y_val = x_test, y_test


    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_connect4():
    logger = logging.getLogger(__name__)
    logger.info("start loading connect-4 file")
    # Download these 2 files from the LIBSVM dataset page
    x, y = load_svmlight_file(
        ("./datasets/connect4/connect-4"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x = x.toarray().astype(np.float32)

    y = y + 1

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, y_val = x_test, y_test

    print("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))

    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_a1a():
    logger = logging.getLogger(__name__)
    logger.info("start loading adult a1a file")
    # Download these 2 files from the LIBSVM dataset page
    x_train, y_train, x_val, y_val, x_test, y_test = load_svmlight_files(
        ("./datasets/a1a/a1a",
         "./datasets/a1a/a1a.t",
         "./datasets/a1a/a1a.t"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x_train = x_train.toarray().astype(np.float32)
    x_val = x_val.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)

    y_train = (y_train + 1) // 2
    y_val = (y_val + 1) // 2
    y_test = (y_test + 1) // 2

    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_a9a():
    logger = logging.getLogger(__name__)
    logger.info("start loading adult a9a file")
    # Download these 2 files from the LIBSVM dataset page
    x_train, y_train, x_val, y_val, x_test, y_test = load_svmlight_files(
        ("./datasets/a9a/a9a",
         "./datasets/a9a/a9a.t",
         "./datasets/a9a/a9a.t"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x_train = x_train.toarray().astype(np.float32)
    x_val = x_val.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)

    y_train = (y_train + 1) // 2
    y_val = (y_val + 1) // 2
    y_test = (y_test + 1) // 2

    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_aloi():
    logger = logging.getLogger(__name__)
    logger.info("start loading aloi file")
    # Download these 2 files from the LIBSVM dataset page
    x, y = load_svmlight_file(
        ("./datasets/aloi/aloi.scale"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x = x.toarray().astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, y_val = x_test, y_test
    print("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))

    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_covtype_binary():
    logger = logging.getLogger(__name__)
    logger.info("start loading covtype.binary file")
    # Download these 2 files from the LIBSVM dataset page
    x, y = load_svmlight_file(
        ("./datasets/covtype.binary/covtype.libsvm.binary.scale"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x = x.toarray().astype(np.float32)
    y = y - 1

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, y_val = x_test, y_test

    print("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))

    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_cod_rna():
    logger = logging.getLogger(__name__)
    logger.info("start loading cod_rna file")
    # Download these 2 files from the LIBSVM dataset page
    x_train, y_train, x_val, y_val, x_test, y_test = load_svmlight_files(
        ("./datasets/cod-rna/cod-rna",
         "./datasets/cod-rna/cod-rna.t",
         "./datasets/cod-rna/cod-rna.t"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x_train = x_train.toarray().astype(np.float32)
    x_val = x_val.toarray().astype(np.float32)
    x_test = x_test.toarray().astype(np.float32)

    y_train = (y_train + 1) // 2
    y_val = (y_val + 1) // 2
    y_test = (y_test + 1) // 2

    sc = StandardScaler()
    sc.fit(x_train)
    x_train, x_val, x_test = sc.transform(x_train), sc.transform(x_val), sc.transform(x_test)
    print("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))

    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_SUSY():
    logger = logging.getLogger(__name__)
    logger.info("start loading SUSY file")
    # Download these 2 files from the LIBSVM dataset page
    x, y = load_svmlight_file(
        ("./datasets/SUSY/SUSY"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x = x.toarray().astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, y_val = x_test, y_test

    sc = StandardScaler()
    sc.fit(x_train)
    x_train, x_val, x_test = sc.transform(x_train), sc.transform(x_val), sc.transform(x_test)

    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_cifar10():
    logger = logging.getLogger(__name__)
    logger.info("start loading cifar10 file")
    x_train, y_train, x_test, y_test = load_svmlight_files(("./datasets/cifar10/cifar10", "./datasets/cifar10/cifar10.t"))

    x_train = x_train.toarray().astype(np.float32) / 255.0
    x_test = x_test.toarray().astype(np.float32) / 255.0

    x_val, y_val = x_test, y_test

    logger.info("x_train size is {}, x_test size is {}".format(x_train.shape, x_test.shape))

    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_segment():
    logger = logging.getLogger(__name__)
    logger.info("start loading segment file")
    # Download these 2 files from the LIBSVM dataset page
    x, y = load_svmlight_file(
        ("./datasets/segment/segment.scale"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x = x.toarray().astype(np.float32)
    y = y-1

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, y_val = x_test, y_test
    print("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))

    logger.info("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_australian():
    logger = logging.getLogger(__name__)
    logger.info("start loading australian file")
    # Download these 2 files from the LIBSVM dataset page
    x, y = load_svmlight_file(
        ("./datasets/australian/australian_scale"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x = x.toarray().astype(np.float32)
    y = (y + 1) // 2

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, y_val = x_test, y_test

    print("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_breast_cancer():
    logger = logging.getLogger(__name__)
    logger.info("start loading breast_cancer file")
    # Download these 2 files from the LIBSVM dataset page
    x, y = load_svmlight_file(
        ("./datasets/breast_cancer/breast-cancer_scale"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x = x.toarray().astype(np.float32)
    y = y // 2 - 1

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, y_val = x_test, y_test

    print("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_fourclass():
    logger = logging.getLogger(__name__)
    logger.info("start loading fourclass file")
    # Download these 2 files from the LIBSVM dataset page
    x, y = load_svmlight_file(
        ("./datasets/fourclass/fourclass_scale"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x = x.toarray().astype(np.float32)
    y = (y + 1) // 2

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, y_val = x_test, y_test

    print("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_german_numer():
    logger = logging.getLogger(__name__)
    logger.info("start loading german.numer file")
    # Download these 2 files from the LIBSVM dataset page
    x, y = load_svmlight_file(
        ("./datasets/german.number/german.numer_scale"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x = x.toarray().astype(np.float32)
    y = (y + 1) // 2

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, y_val = x_test, y_test

    print("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_diabetes():
    logger = logging.getLogger(__name__)
    logger.info("start loading diabetes file")
    # Download these 2 files from the LIBSVM dataset page
    x, y = load_svmlight_file(
        ("./datasets/diabetes/diabetes_scale"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x = x.toarray().astype(np.float32)
    y = (y + 1) // 2

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, y_val = x_test, y_test

    print("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_heart():
    logger = logging.getLogger(__name__)
    logger.info("start loading heart file")
    # Download these 2 files from the LIBSVM dataset page
    x, y = load_svmlight_file(
        ("./datasets/heart/heart_scale"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x = x.toarray().astype(np.float32)
    y = (y + 1) // 2

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, y_val = x_test, y_test

    print("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_vehicle():
    logger = logging.getLogger(__name__)
    logger.info("start loading vehicle file")
    # Download these 2 files from the LIBSVM dataset page
    x, y = load_svmlight_file(
        ("./datasets/vehicle/vehicle.scale"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x = x.toarray().astype(np.float32)
    y = y-1

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, y_val = x_test, y_test

    print("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_wine():
    logger = logging.getLogger(__name__)
    logger.info("start loading wine file")
    # Download these 2 files from the LIBSVM dataset page
    x, y = load_svmlight_file(
        ("./datasets/wine/wine.scale"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x = x.toarray().astype(np.float32)
    y = y-1

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, y_val = x_test, y_test

    print("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_sonar():
    logger = logging.getLogger(__name__)
    logger.info("start loading sonar file")
    # Download these 2 files from the LIBSVM dataset page
    x, y = load_svmlight_file(
        ("./datasets/sonar/sonar_scale"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x = x.toarray().astype(np.float32)
    y = (y + 1) // 2

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, y_val = x_test, y_test

    print("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_glass():
    logger = logging.getLogger(__name__)
    logger.info("start loading glass file")
    # Download these 2 files from the LIBSVM dataset page
    x, y = load_svmlight_file(
        ("./datasets/glass/glass.scale"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x = x.toarray().astype(np.float32)
    y = (y + 1) // 2

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, y_val = x_test, y_test

    print("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test


def Load_ionosphere():
    logger = logging.getLogger(__name__)
    logger.info("start loading ionosphere file")
    # Download these 2 files from the LIBSVM dataset page
    x, y = load_svmlight_file(
        ("./datasets/ionosphere/ionosphere_scale"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x = x.toarray().astype(np.float32)
    y = (y + 1) // 2

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, y_val = x_test, y_test

    print("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test

def Load_phishing():
    logger = logging.getLogger(__name__)
    logger.info("start loading phishing file")
    # Download these 2 files from the LIBSVM dataset page
    x, y = load_svmlight_file(
        ("./datasets/phishing/phishing"))
    # for simplicity, convert sparse matrices to dense ones. We need to handle sparse input later.
    x = x.toarray().astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, y_val = x_test, y_test

    sc = StandardScaler()
    sc.fit(x_train)
    x_train, x_val, x_test = sc.transform(x_train), sc.transform(x_val), sc.transform(x_test)

    print("x_train size is {}, x_val size is {}, x_test size is {}".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test