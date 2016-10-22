import h5py
import numpy as np


def turn_into_array(file_path):
    print("Processing ", file_path)
    data = pd.read_table(file_path, delim_whitespace=True, header=None)

    return data.as_matrix()


if __name__ == "__main__":
    h5file = h5py.File('./MNIST/binarized_mnist.h5')
    test = turn_into_array('./MNIST/binarized_mnist_test.amat')
    h5file.create_dataset('test', shape=test.shape, dtype=test.dtype, data=test)
    train = turn_into_array('./MNIST/binarized_mnist_train.amat')
    h5file.create_dataset('train', shape=train.shape, dtype=train.dtype, data=train)
    valid = turn_into_array('./MNIST/binarized_mnist_valid.amat')
    h5file.create_dataset('valid', shape=valid.shape, dtype=valid.dtype, data=valid)
    h5file.close()