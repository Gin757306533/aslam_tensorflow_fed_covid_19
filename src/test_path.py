
def load_all_data(path = "C:\\data\\aslam\\COVID19-XRay-Dataset-master\\train\\"):
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
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]
    print(label_to_index)
    print(all_image_labels[:-10])

    img_raw = tf.io.read_file(all_image_paths[0])
    img_tensor = tf.image.decode_image(img_raw)
    img_tensor.set_shape([None, None, 3])
    print(img_tensor.shape)
    print(img_tensor.dtype)

    img_final = tf.image.resize(img_tensor, [1080, 1080])
    img_final = img_final / 255.0
    return img_final, all_image_labels

import cv2 as cv
import numpy as np
def load_all_data2(path = "E:\\data\\aslam\\COVID19-XRay-Dataset-master\\COVID19-XRay-Dataset-master\\train\\"):
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

    img_final_list = [cv.resize(img_tensor, (1080, 1080)).astype("float32")/255 for img_tensor in img_raw_list]
    img_final_list = np.array(img_final_list, dtype=np.float32)
    return img_final_list, all_image_labels
def load_data():
    x_train, x_test = load_all_data2()
    y_train, y_test = load_all_data2("C:\\data\\aslam\\COVID19-XRay-Dataset-master\\COVID19-XRay-Dataset-master\\test\\")
    return (x_train, y_train), (x_test, y_test)

load_data()