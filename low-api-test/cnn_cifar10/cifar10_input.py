# 如果是python2的代码，会有不兼容，这里把python3的特性引入，使得python2也可以运行该代码。
from __future__ import absolute_import  # 绝对引用
from __future__ import division  # 精确除法
from __future__ import print_function  # print函数

import os

import tensorflow as tf
# xrange返回一个生成器，range返回一个列表，xrange在生成大范围数据的时候更节省内存。
from six.moves import xrange

IMAGE_SIZE = 24  # 图像尺寸
# 描述 CIFAR-10 数据的全局常量。
NUM_CLASSES = 10  # 类别数目
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000  # 训练样本数目
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000  # 测试样本数目


def read_cifar10(filename_queue):
    """从文件名队列读取二进制数据并提取出单张图像数据
    建议:
      如果想N个线程同步读取, 调用这个函数N次即可。
      这将会产生N个独立的Readers从不同文件不同位置读取数据，进而产生更好的样本混合效果。

    输入参数:
      filename_queue: 一个包含文件名列表的字符串队列
    返回一个类，包含了单张图像的各种数据。
    """

    class CIFAR10Record(object):  # 定义一个类
        pass

    result = CIFAR10Record()  # 建立类的实体对象

    # 输入数据格式
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth

    record_bytes = label_bytes + image_bytes
    # 定义固定长度阅读器读取长度为record_bytes的数据，从文件名队列中获取文件并读出单张图像数据。
    # CIFAR-10格式数据没有头数据和尾数据，所以我们令 header_bytes 和 footer_bytes 保持默认值0。
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes,
                                        header_bytes=0,
                                        footer_bytes=0)
    result.key, value = reader.read(filename_queue)
    # 使用解码器把字符串类型转化为 uint8 类型的数据
    record_bytes = tf.decode_raw(value, tf.uint8)
    # 第一个字节代表着标签数据，所以我们把它的格式从uint8转化为int32。
    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes]),
        tf.int32
    )
    # 剩下的字节表示图像数据，我们首先根据CIFAR-10的数据排列[depth * height * width]
    # 把它转化为 [depth, height, width] 形状的张量。
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes,
                         [label_bytes],
                         [label_bytes + image_bytes]),
        [result.depth, result.height, result.width]
    )
    # 把 [depth, height, width] 转化为 [height, width, depth] 形状的张量，这是TensorFlow处理图像的格式。
    # transpose: 转置；变换
    # perm 参数的意义．假如一个矩阵有 n 维，那么 perm 就是一个大小为 n 的列表，其值就是［０：n-1］的一个排列.
    # perm 中，位置 i 的值 j 代表转置后的矩阵 j 维就是转置前矩阵 i 维的值．
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    """生成图像和标签的 batch 数据
    输入参数:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, 每次出队后队伍中剩下的样本数量的最小值。
      batch_size: 每个batch的图像数量。
      shuffle: boolean 值表明是否对图像队列随机排序。
    返回数据:
      images: batch 图像数据. 4D tensor of [batch_size, height, width, 3] size.
      labels: batch 标签数据. 1D tensor of [batch_size] size.
    """
    # 创造一个样本队列，根据需求决定是否对样本随机排序，每次从队列中出队 batch_size 个图像数据和标签数据
    num_preprocess_threads = 16
    # 16个 Reader 平行读取，每个 Reader 读不同的文件或者位置，可以充分的混合样本数据
    if shuffle:  # 对样本队列随机排序
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:  # 不对样本队列随机排序
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)
    # 添加summary节点以便在TensorBoard中对图像信息进行可视化
    tf.summary.image('images', images)
    return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
    """ 构造训练数据并对其进行失真处理。
    输入参数:
      data_dir: CIFAR-10数据的存放路径.
      batch_size: 每个batch中图像的数目.
    返回数据:
      images: batch图像数据. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: batch标签数据. 1D tensor of [batch_size] size.
    """
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    # 创建一个文件名队列
    filename_queue = tf.train.string_input_producer(filenames)
    # 调用read_cifar10函数从文件名队列中读取文件，并得到单张图像信息
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    height = IMAGE_SIZE  # 24
    width = IMAGE_SIZE  # 24
    # 对训练数据图像预处理，包括多个随机失真操作。
    # 把原始的32*32的图像随机裁剪为24*24
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
    # 随机左右翻转
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # 随机改变图像亮度和对比度
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)
    # 把图像进行归一化处理，变为0均值和1方差。
    float_image = tf.image.per_image_standardization(distorted_image)
    # 有时候graph没法推断出tensors的形状，我们可以手动保存tensors的形状信息
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])
    # 当采用随机生成batch操作时设定min_after_dequeue的值为50000*0.4=20000，保证足够的随机性。
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d CIFAR images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)
    # 通过建立样本队列来生成batch图像数据和batch标签数据
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size, shuffle=True)


def inputs(eval_data, data_dir, batch_size):
    """构造测试数据.
    输入参数:
      eval_data: bool型，表明使用训练数据还是测试数据，True为测试集
      data_dir: CIFAR-10数据集的存放路径
      batch_size: 每个batch的图片数量
    返回数据:
      images: batch图像数据. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: batch标签数据. 1D tensor of [batch_size] size.
    """
    if not eval_data:  # 训练数据
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                     for i in xrange(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:  # 测试数据
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    for f in filenames:  # 检查文件是否存在
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    # 创建一个文件名队列
    filename_queue = tf.train.string_input_producer(filenames)
    # 从文件名队列中的文件中读取单个样本
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    # 测试时的图像预处理
    # 沿中心裁剪28*28大小的图像
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)
    # 图像归一化处理
    float_image = tf.image.per_image_standardization(resized_image)
    # 设置张量的形状
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])
    # 设置min_after_dequeue参数为20000(训练数据)，4000(测试数据)，保证足够随机性。
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)
    # 通过建立一个样本队列生成batch图像数据和batch标签数据
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size, shuffle=False)
