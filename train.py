import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam, SGD

from nets import freeze_layers, get_model_from_name
from utils.callbacks import ExponentDecayScheduler, LossHistory
from utils.dataloader import ClsDatasets
from utils.utils import get_classes
from nets.Loss import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    classes_path    = 'model_data/cls_classes.txt' 
    input_shape     = [224, 224]
    backbone        = "vgg16"
    alpha           = 0.25
    model_path      = "model_data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    Freeze_Train    = True
    annotation_path = "cls_train.txt"
    val_split       = 0.1
    num_workers     = 1
    class_names, num_classes = get_classes(classes_path)
    # print(class_names)
    # print(num_classes)

    assert backbone in ["mobilenet", "resnet50", "vgg16"]
    if backbone == "mobilenet":
        model = get_model_from_name[backbone](input_shape=[input_shape[0], input_shape[1], 3], classes=num_classes, alpha=alpha)
    else:
        model = get_model_from_name[backbone](input_shape=[input_shape[0], input_shape[1], 3], classes=num_classes)
    if model_path != "":
        print('Load weights {}.'.format(model_path))
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
    logging         = TensorBoard(log_dir = 'logs/')
    checkpoint      = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr       = ExponentDecayScheduler(decay_rate = 0.94, verbose = 1)
    early_stopping  = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1)
    loss_history    = LossHistory('logs/')
    with open(annotation_path, "r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val     = int(len(lines) * val_split)
    num_train   = len(lines) - num_val

    if Freeze_Train:
        for i in range(freeze_layers[backbone]):
            model.layers[i].trainable = False
    if True:
        batch_size      = 32
        Lr              = 1e-3
        Init_Epoch      = 0
        Freeze_Epoch    = 50

        epoch_step          = num_train // batch_size
        epoch_step_val      = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
            
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

        # 损失函数 ce ce 'categorical_crossentropy' loss
        model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = Lr), metrics = ['categorical_accuracy'])


        train_dataloader    = ClsDatasets(lines[:num_train], input_shape, batch_size, num_classes, train = True)
        val_dataloader      = ClsDatasets(lines[num_train:], input_shape, batch_size, num_classes, train = False)

        model.fit_generator(
            generator           = train_dataloader,
            steps_per_epoch     = epoch_step,
            validation_data     = val_dataloader,
            validation_steps    = epoch_step_val,
            epochs              = Freeze_Epoch,
            initial_epoch       = Init_Epoch,
            use_multiprocessing = True if num_workers > 1 else False,
            workers             = num_workers,
            callbacks           = [logging, checkpoint, reduce_lr, early_stopping, loss_history]
        )


    if Freeze_Train:
        for i in range(freeze_layers[backbone]):
            model.layers[i].trainable = True

    if True:
        # batch_size      = 32
        batch_size      = 16
        Lr              = 1e-4
        Freeze_Epoch    = 50
        Epoch           = 200

        epoch_step          = num_train // batch_size
        epoch_step_val      = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
            
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        # loss : ce
        model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = Lr), metrics = ['categorical_accuracy'])

        train_dataloader    = ClsDatasets(lines[:num_train], input_shape, batch_size, num_classes, train = True)
        val_dataloader      = ClsDatasets(lines[num_train:], input_shape, batch_size, num_classes, train = False)

        model.fit_generator(
            generator           = train_dataloader,
            steps_per_epoch     = epoch_step,
            validation_data     = val_dataloader,
            validation_steps    = epoch_step_val,
            epochs              = Epoch,
            initial_epoch       = Freeze_Epoch,
            use_multiprocessing = True if num_workers > 1 else False,
            workers             = num_workers,
            callbacks           = [logging, checkpoint, reduce_lr, early_stopping, loss_history]
        )
