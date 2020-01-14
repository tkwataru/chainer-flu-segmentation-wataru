"""MarkerlessTrainingFCN_para.py 改造版"""
import argparse
import logging.config
import multiprocessing
import os
import random
from scipy import ndimage

import chainer
import cv2
import numpy as np
from chainer import cuda
from chainer.training import extensions
from model_unet_chainer import UNet
from matplotlib.mlab import bivariate_normal
# from tqdm import tqdm
import json

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# logging.config.fileConfig('logging.config')
# LOGGER = logging.getLogger('root')

SNAPSHOT_TRIGGER = (10, 'epoch')
# WEIGHT_DECAY_PER_EPOCH = 0.5
N_CLASS = 2


def parse_args(args=None):
    def gamma(s):
        value = float(s)
        if value < 1:
            raise argparse.ArgumentTypeError('must be greater than or equal to one: {}'.format(value))
        return value

    def sigma(s):
        value = float(s)
        if value < 0:
            raise argparse.ArgumentTypeError('must be greater than or equal to zero: {}'.format(value))
        return value

    parser = argparse.ArgumentParser(
        description='Learning CNN(para) for IHM')
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--val_batchsize', '-b', type=int, default=200,
                        help='Validation minibatch size')
    parser.add_argument('--epoch', '-e', default=10, type=int,
                        help='Number of epochs to learn')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--loaderjob', '-j', default=15, type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--root', '-r', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--resume',
                        help='Initialize training from given snapshot file')
    parser.add_argument('--out', '-o', default='result',
                        help='Root path to save files')
    parser.add_argument('--test', dest='test', action='store_true', default=False)
    return parser.parse_args(args)


xp = None


def setup_gpu(gpu):
    global xp

    if gpu < 0:
        xp = np
    else:
        xp = cuda.cupy
        cuda.check_cuda_available()
        cuda.get_device_from_id(gpu).use()


def load_model(n_class):
    #LOGGER.info("# load MyFcn model")
    #model = MyFcn()
    #LOGGER.info("# load CNN model")
    model = UNet(n_class)

    model = model.to_gpu()
    #model.class_weight = xp.array([1, 45], dtype=xp.float32)

    return model


def load_path_pairs(image_list_path, root):
    with open(image_list_path) as fp:
        for line in fp:
            if not line.strip():
                continue
            path_pair = line.strip().split()
            assert len(path_pair) == 2, path_pair

            yield tuple(os.path.join(root, path) for path in path_pair)


def load_json(json_path):
    with open(json_path, 'r', encoding="utf-8_sig") as fr:
        annotation_json = json.load(fr)
    return annotation_json


def read_image(img_path):
    #image = cv2.imread(img_path)[..., 0]
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    assert image.ndim == 2, image.shape
    assert image.dtype == np.uint8, image.dtype
    return image


def read_label(label_path):
    #label = cv2.imread(label_path)[..., 0]
    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
    assert label.ndim == 2, label.shape
    assert label.dtype == np.uint8, label.dtype
    return label


class Dataset(chainer.dataset.DatasetMixin):
    # def __init__(self, path_pairs, for_validation=False):
    def __init__(self, annotation_json, root, for_validation=False):
        # self._path_pairs = list(path_pairs)
        self._annotation_json = annotation_json
        self._root = root
        self._for_validation = for_validation

        self.class_names = [
            '__background__',
            'back_wall',
            'bubble',
            'follicular',
            'other',
            'saliva',
            'shining_stick',
            'tongue',
            'tonsil',
            'uvula',
            'white_moss',
            'wood_stick'
        ]
        self.valid_classes = [
            0,
            2,
            3,
        ]

    def _get_raw_pair(self, i):
        image_path, label_path = self._path_pairs[i]
        image = read_image(image_path)
        label = read_label(label_path)
        return image, label

    def _get_raw_image(self, i):
        image_path = self._annotation_json[i]['image_file_name']
        image_file = os.path.join(
            self.root,
            image_path)
        image = read_image(image_file)
        return image

    def _create_label(self, index, image_shape):
        height, width, channels = image_shape
        label = np.zeros((height, width), dtype=np.uint8)
        # label = np.zeros((height, width, 1), dtype=np.uint8)

        # annotation_file = os.path.join(
        #     self.root,
        #     "Annotations",
        #     "%s.xml" %
        #     image_id)
        # objects = ET.parse(annotation_file).findall("object")
        objects = self._annotation_json[index]['annotations']

        for obj in objects:
            # class_name = obj.find('name').text.lower().strip()
            class_name = obj['class']

            if self.class_names.index(class_name) in self.valid_classes:
                # print(class_name)
                # bbox = obj.find('bndbox')
                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(obj['xmin']) - 1
                y1 = float(obj['ymin']) - 1
                x2 = float(obj['xmax']) - 1
                y2 = float(obj['ymax']) - 1

                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                axes = (x2 - x1, y2 - y1)
                angle = 0
                box = (center, axes, angle)

                cv2.ellipse(label, box, self.class_names.index(class_name), thickness=-1)

        # cv2.imshow('label', label * 25)
        # cv2.waitKey(0)

        return label

    def get_example(self, i):
        # image, label = self._get_raw_pair(i)
        image = self._get_raw_image(i)
        label = self._create_label(i, image.shape)

        image = image.astype(np.float32)/255 - 0.5
        label = label.astype(np.int32)

        if self._for_validation:
            pass
        else:
            if random.random() > 0.5:
                tmp1 = cv2.flip(image, 0)  # Vertical flip
                tmp2 = cv2.flip(label, 0)  # Vertical flip
                #tmp1 = xp.flipud(image)
                #tmp2 = xp.flipud(label)
                image = tmp1
                label = tmp2

            if random.random() > 0.5:
                tmp1 = cv2.flip(image, 1)  # Horizontal flip
                tmp2 = cv2.flip(label, 1)  # Horizontal flip
                #tmp1 = xp.fliplr(image)
                #tmp2 = xp.fliplr(label)
                image = tmp1
                label = tmp2

            """
            M = np.float32([[1,0,random.randint(-20,20)],[0,1,random.randint(-20,20)]])
            tmp1 = cv2.warpAffine(image, M, image.shape)   # Shift
            tmp2 = cv2.warpAffine(label, M, label.shape)   # Shift
            image = tmp1
            label = tmp2
            """

            angle = random.uniform(0,360)
            tmp1 = ndimage.interpolation.rotate(image, angle, reshape=False, order=1, mode='reflect')    # Rotation by bilinear
            tmp2 = ndimage.interpolation.rotate(label, angle, reshape=False, order=0, mode='reflect')    # Rotation by nearest neighbor
            image = tmp1
            label = tmp2

        return image, label

    def __len__(self):
        # return len(self._path_pairs)
        return len(self._annotation_json)

"""
class ClassWeightUpdater(chainer.training.Extension):
    def __init__(self, model, trigger=(1, 'iteration')):
        self._model = model
        self._trigger = chainer.training.trigger.get_trigger(trigger)
        self._decay_count = 0

    def _update(self, epoch):
        self._model.class_weight[0] = 1
        self._model.class_weight[1] = 1 + 44 * WEIGHT_DECAY_PER_EPOCH ** epoch

    def __call__(self, trainer):
        if self._trigger(trainer):
            with chainer.cuda.get_device_from_array(self._model.class_weight):
                self._update(epoch=trainer.updater.epoch_detail)
"""

def _main():
    args = parse_args()
    setup_gpu(gpu=args.gpu)
    assert xp is not None

    # Model
    model = load_model(N_CLASS)
    # optimizer = chainer.optimizers.Adam(0.001)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Dataset
    train_iter = chainer.iterators.MultiprocessIterator(
        # Dataset(load_path_pairs(args.train, args.root)),
        Dataset(load_json(args.train), args.root),
        batch_size=args.batchsize, n_processes=args.loaderjob)

    test_iter = chainer.iterators.MultiprocessIterator(
        # Dataset(load_path_pairs(args.val, args.root), for_validation=True),
        Dataset(load_json(args.val), args.root, for_validation=True),
        batch_size=args.val_batchsize, n_processes=args.loaderjob, repeat=False, shuffle=False)

    """
    # show test data
    for i in range(len(test_iter.dataset)):
        image, label = test_iter.dataset.get_example(i)
        os.makedirs(args.out + r'\validation_data', exist_ok=True)
        cv2.imwrite(args.out + r'\validation_data\image_{:06d}.png'.format(i), (image[0] * 256).astype('u1'))
        cv2.imwrite(args.out + r'\validation_data\label_{:06d}.png'.format(i), label.astype('u1'))
    """

    # Updater, Trainer, Evaluater
    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    evaluator = extensions.Evaluator(test_iter, model, device=args.gpu)
    evaluator.trigger = SNAPSHOT_TRIGGER
    trainer.extend(evaluator)

    # Weight decay
    # trainer.extend(ClassWeightUpdater(model))

    # Snapshot
    trainer.extend(extensions.snapshot(
        # filename='snapshot_iter-{.updater.iteration:06d}'),
        filename='snapshot_epoch-{.updater.epoch:04d}'),
        trigger=SNAPSHOT_TRIGGER)
    trainer.extend(extensions.snapshot(
        filename='snapshot_latest'),
        trigger=SNAPSHOT_TRIGGER)
    trainer.extend(extensions.snapshot_object(
        target=model,
        # filename='model_iter-{.updater.iteration:06d}'),
        filename='model_epoch-{.updater.epoch:04d}'),
        trigger=SNAPSHOT_TRIGGER)
    trainer.extend(extensions.snapshot_object(
        target=model,
        filename='model_latest'),
        trigger=SNAPSHOT_TRIGGER)

    # Report
    # trainer.extend(extensions.LogReport(log_name='log.txt', trigger=(1, 'iteration')))
    trainer.extend(extensions.LogReport(log_name='log.txt', trigger=(1, 'epoch')))
    trainer.extend(extensions.PrintReport(
        entries=['epoch', 'iteration', 'main/loss', 'validation/main/loss', 'elapsed_time'],
        # log_report=extensions.LogReport(log_name=None, trigger=(10, 'iteration'))))
        log_report=extensions.LogReport(log_name=None, trigger=(1, 'epoch'))))
    trainer.extend(extensions.PlotReport(
        y_keys=['main/loss', 'validation/main/loss'],
        x_key='iteration',
        # x_key='epoch',
        file_name='loss.png',
        postprocess=lambda fig, ax, summary: ax.semilogy(),
        marker='',
        trigger=(1, 'iteration')))
    trainer.extend(extensions.PlotReport(
        y_keys=['main/error_rate', 'validation/main/error_rate'],
        x_key='iteration',
        # x_key='epoch',
        file_name='error_rate.png',
        postprocess=lambda fig, ax, summary: ax.semilogy(),
        marker='',
        trigger=(1, 'iteration')))

    # resume
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # Go !
    trainer.run()

    import time
    time.sleep(60)


if __name__ == '__main__':
    _main()
