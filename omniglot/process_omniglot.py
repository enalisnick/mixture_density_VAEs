import h5py
import numpy as np
from PIL import Image

def get_image_paths(dataset):
    image_paths = []
    filelist = open('{}_filelist.txt'.format(dataset))
    for line in filelist:
        if line[:-1].endswith('.png'):
            image_paths.append(line[:-1])
    return image_paths

def turn_png_to_array(img):
    return np.array(img).astype('float32').reshape(-1)

def process_all_pngs(dataset='evaluation'):
    image_paths = get_image_paths(dataset)
    num_images = len(image_paths)
    image_array = np.zeros((num_images, 11025))

    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path)
        except IOError:
            continue
        img = turn_png_to_array(img)
        image_array[i, :] = img

    return image_array

def get_omniglot(processed=True):
    h5file = h5py.File('./omniglot.h5')
    bg = np.copy(h5file['background'])
    eval = np.copy(h5file['evaluation'])
    h5file.close()

    X = np.vstack([eval, bg])

    if processed:
        X -= X.mean(axis=0)
        X /= (X.std(axis=0) + 0.00001)
        np.random.shuffle(X)

    return X


if __name__ == "__main__":
    eval = process_all_pngs('evaluation')
    bg = process_all_pngs('background')

    h5file = h5py.File('./omniglot.h5')
    h5file.create_dataset('background', shape=bg.shape, dtype='float32', data=bg)
    h5file.create_dataset('evaluation', shape=eval.shape, dtype='float32', data=eval)
    h5file.close()