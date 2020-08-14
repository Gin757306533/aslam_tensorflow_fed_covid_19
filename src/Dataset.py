import numpy as np
from tensorflow.keras.utils import to_categorical
import cv2 as cv
def load_all_data(path = "../COVID19-XRay-Dataset-master/train/"):
    import pathlib
    import tensorflow as tf
    data_root_orig = path
    data_root = pathlib.Path(data_root_orig)

    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    print(all_image_paths[:10])

    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    print(label_names[:4])

    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_image_labels = [[label_to_index[pathlib.Path(path).parent.name]]
                        for path in all_image_paths]
    all_image_labels = np.array(all_image_labels, dtype='int32')

    img_raw_list = [cv.imread(all_image_paths[i]) for i in range(len(all_image_paths))]

    # img_tensor_list = [tf.image.decode_image(img_raw) for img_raw in img_raw_list]
    # [img_tensor.set_shape([1080, 1080, 3]) for img_tensor in img_tensor_list]
    # 1080
    img_final_list = [cv.resize(img_tensor, (32, 32)).astype("float32")/255 for img_tensor in img_raw_list]
    img_final_list = np.array(img_final_list, dtype=np.float32)
    return img_final_list, all_image_labels
def load_data():
    x_train, y_train = load_all_data()
    x_test, y_test = load_all_data("../COVID19-XRay-Dataset-master/test/")
    return (x_train, y_train), (x_test, y_test)

class BatchGenerator:
    def __init__(self, x, yy):
        self.x = x
        self.y = yy
        self.size = len(x)
        self.random_order = list(range(len(x)))
        np.random.shuffle(self.random_order)
        self.start = 0
        return

    def next_batch(self, batch_size):
        if self.start + batch_size >= len(self.random_order):
            overflow = (self.start + batch_size) - len(self.random_order)
            perm0 = self.random_order[self.start:] +\
                 self.random_order[:overflow]
            self.start = overflow
        else:
            perm0 = self.random_order[self.start:self.start + batch_size]
            self.start += batch_size

        assert len(perm0) == batch_size

        return self.x[perm0], self.y[perm0]

    # support slice
    def __getitem__(self, val):
        return self.x[val], self.y[val]


class Dataset(object):
    def __init__(self, load_data_func, one_hot=True, split=0, local_split=1):
        # (x_train, y_train), (x_test, y_test) = load_data_func()
        (x_train, y_train), (x_test, y_test) = load_data()
        print("Dataset: train-%d, test-%d" % (len(x_train), len(x_test)))

        if one_hot:
            y_train = to_categorical(y_train, 3)
            y_test = to_categorical(y_test, 3)

        # x_train = x_train.astype('float32')
        # x_test = x_test.astype('float32')
        # x_train /= 255
        # x_test /= 255

        if split == 0:
            self.train = BatchGenerator(x_train, y_train)
        else:
            self.train = self.splited_batch(x_train, y_train, split, local_split+1)

        self.test = BatchGenerator(x_test, y_test)

    def splited_batch(self, x_data, y_data, count, local_split):
        res = []
        l = len(x_data)
        for i in range(0, l, l//count):
            temp_res = []
            for j in range(i, i + l//count, l//count//local_split):
                temp_res.append(
                    BatchGenerator(x_data[j:j + l // count//local_split],
                                   y_data[j:j + l // count//local_split])
                )
            res.append(temp_res[:])
        return res