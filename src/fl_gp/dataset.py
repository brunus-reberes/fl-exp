import numpy
from mnist import MNIST
from pathlib import Path

def load_dataset(dataset, datasets_path="../datasets", train_size=None, test_size=None):
    path = Path(datasets_path).absolute().joinpath(dataset)
    if dataset == "mnist":
        return trim(*mnist(path), train_size, test_size)
    elif dataset == "mnist-rot":
        return trim(*mnist_rot(path), train_size, test_size)
    elif dataset == "mnist-back-image":
        return trim(*mnist_back_image(path), train_size, test_size)
    elif dataset == "mnist-back-rand":
        return trim(*mnist_back_rand(path), train_size, test_size)
    elif dataset == "mnist-rot-back-image":
        return trim(*mnist_rot_back_image(path), train_size, test_size)

def trim(train_data, train_labels, test_data, test_labels, train_size=None, test_size=None):
    if train_size:
        train_data = train_data[:train_size]
        train_labels = train_labels[:train_size]
    if test_size:
        test_data = test_data[:test_size]
        test_labels = test_labels[:test_size]
    return train_data, train_labels, test_data, test_labels

def mnist(path):
    dataset = MNIST(path, return_type='numpy')
    train_data, train_labels = dataset.load_training()
    test_data, test_labels = dataset.load_testing()
    train_data = train_data.reshape(len(train_data),28,28).astype('uint8') / 255
    test_data = test_data.reshape(len(test_data),28,28).astype('uint8') / 255
    return train_data, train_labels, test_data, test_labels

def mnist_rot(path):
    test_set = numpy.loadtxt(path.joinpath("mnist_all_rotation_normalized_float_test.amat"))
    train_set = numpy.loadtxt(path.joinpath("mnist_all_rotation_normalized_float_train_valid.amat"))
    test_data = test_set[:, :-1].reshape((len(test_set), 28, 28))
    test_labels = test_set[:, -1].astype('uint8')
    train_data = train_set[:, :-1].reshape((len(train_set), 28, 28))
    train_labels = train_set[:, -1].astype('uint8')
    return train_data, train_labels, test_data, test_labels

def mnist_back_image(path):
    test_set = numpy.loadtxt(path.joinpath("mnist_background_images_test.amat"))
    train_set = numpy.loadtxt(path.joinpath("mnist_background_images_train.amat"))
    test_data = test_set[:, :-1].reshape((len(test_set), 28, 28))
    test_labels = test_set[:, -1].astype('uint8')
    train_data = train_set[:, :-1].reshape((len(train_set), 28, 28))
    train_labels = train_set[:, -1].astype('uint8')
    return train_data, train_labels, test_data, test_labels

def mnist_back_rand(path):
    test_set = numpy.loadtxt(path.joinpath("mnist_background_random_test.amat"))
    train_set = numpy.loadtxt(path.joinpath("mnist_background_random_train.amat"))
    test_data = test_set[:, :-1].reshape((len(test_set), 28, 28))
    test_labels = test_set[:, -1].astype('uint8')
    train_data = train_set[:, :-1].reshape((len(train_set), 28, 28))
    train_labels = train_set[:, -1].astype('uint8')
    return train_data, train_labels, test_data, test_labels

def mnist_rot_back_image(path):
    test_set = numpy.loadtxt(path.joinpath("mnist_all_background_images_rotation_normalized_test.amat"))
    train_set = numpy.loadtxt(path.joinpath("mnist_all_background_images_rotation_normalized_train_valid.amat"))
    test_data = test_set[:, :-1].reshape((len(test_set), 28, 28))
    test_labels = test_set[:, -1].astype('uint8')
    train_data = train_set[:, :-1].reshape((len(train_set), 28, 28))
    train_labels = train_set[:, -1].astype('uint8')
    return train_data, train_labels, test_data, test_labels

