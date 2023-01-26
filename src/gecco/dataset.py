from mnist import MNIST
from pathlib import Path

def load_dataset(dataset, datasets_path="../datasets", train_size=None, test_size=None):
    path = Path(datasets_path).absolute().joinpath(dataset)
    if dataset == "mnist":
        return trim(*mnist(path), train_size, test_size)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def load_dataset_batches(dataset_name, datasets_path="../datasets", train_size=None, test_size=None):
    dataset = load_dataset(dataset_name, datasets_path)
    train_data_batch = list(batch(dataset[0], train_size))
    train_labels_batch = list(batch(dataset[1], train_size))
    test_data_batch = list(batch(dataset[2], test_size))
    test_labels_batch = list(batch(dataset[3], test_size))
    return train_data_batch, train_labels_batch, test_data_batch, test_labels_batch
    

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

