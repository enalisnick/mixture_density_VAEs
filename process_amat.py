import h5py
import numpy as np


def turn_into_array(file_path):

    f = open(file_path, 'rb')

    list_of_lines = []
    for line in f:
        list_of_lines.append(map(int, line[:-1].split(' ')))

    dataset_array = np.array(list_of_lines)

    return dataset_array


if __name__ == "__main__":
    h5file = h5py.File('./MNIST/binarized_mnist.h5')
    test = turn_into_array('./MNIST/binarized_mnist_test.amat')
    h5file.create_dataset('test', shape=test.shape, dtype=test.dtype, data=test)
    train = turn_into_array('./MNIST/binarized_mnist_train.amat')
    h5file.create_dataset('train', shape=train.shape, dtype=train.dtype, data=train)
    valid = turn_into_array('./MNIST/binarized_mnist_valid.amat')
    h5file.create_dataset('valid', shape=valid.shape, dtype=valid.dtype, data=valid)
    h5file.close()