import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

#训练样本在本地磁盘中的地址
file_dir='E:\\data\\aslam\\COVID19-XRay-Dataset-master\\COVID19-XRay-Dataset-master\\train\\covid\\' # 这里是输入数据的地址


def get_files(file_dir):
    lung_img = [];
    label_lung_img = [];
    for file in os.listdir(file_dir):
        lung_img.append( file_dir + file)
        label_lung_img.append(1)

    image_list = np.hstack((lung_img))

    label_list = np.hstack((label_lung_img))

    temp = np.array([lung_img, label_lung_img]).T
    #利用shuffle打乱数据
    np.random.shuffle(temp)
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(i) for i in label_list]
    return image_list, label_list
#
#将上面生成的List传入get_batch() ，转换类型，产生一个输入队列queue，因为img和lab
#是分开的，所以使用tf.train.slice_input_producer()，然后用tf.read_file()从队列中读取图像
def get_batch(image,label,batch_size):

    image_W, image_H = 221, 181

    #将python.list类型转换成tf能够识别的格式
    image=tf.cast(image,tf.string)
    label=tf.cast(label,tf.int32)

    #产生一个输入队列queue
    epoch_num = 50 #防止无限循环
    input_queue=tf.train.slice_input_producer([image,label], num_epochs=epoch_num)

    label=input_queue[1]
    image_contents=tf.read_file(input_queue[0])
    #将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用png等。
    image=tf.image.decode_png(image_contents,channels=1)

    #将数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
    image=tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
    image=tf.image.per_image_standardization(image)

    #生成batch
    min_after_dequeue=1000
    capacity=min_after_dequeue+3*batch_size
    image_batch,label_batch=tf.train.shuffle_batch([image,label],batch_size=batch_size,num_threads=64,capacity=capacity,min_after_dequeue=min_after_dequeue)

    #重新排列标签，行数为[batch_size]
#    label_batch=tf.reshape(label_batch,[batch_size])
    image_batch = tf.reshape(image_batch,[batch_size,image_W,image_H,1])
    image_batch=tf.cast(image_batch,np.float32)

    return image_batch, label_batch

if __name__ == "__main__":
    image_list, label_list = get_files(file_dir)
    image_batch, label_batch = get_batch(image_list, label_list, 64)
    with tf.Session() as sess:
        ## 初始化工作，相当重要
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop() and i<1: # 加上i其实是强制终止线程，但是文件队列线程并没有结束，因为我们产生的文件队列结束为epoch_num个epoch

                img, label = sess.run([image_batch, label_batch])

                # just test one batch
                for j in np.arange(64):
                    print('label: %d' %label[j])
                    plt.imshow(img[j,:,:,0])
                    plt.show()
                i+=1

        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
            print('-----------')
        coord.join(threads)