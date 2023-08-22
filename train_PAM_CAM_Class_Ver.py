import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from AGPAD.attention import PAM, CAM
import glob
import numpy as np


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', use_activation=True):
    x = tf.keras.layers.Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    if use_activation:
        x = tf.keras.layers.Activation('relu')(x)
        return x
    else:
        return x


def focal_loss(gamma=2, alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


class AGPAD(object):
    def __init__(self, tr_path, te_path, Gmodel, fold):
        super(AGPAD, self).__init__()
        self.fold = fold
        self.Gmodel = Gmodel
        self.batchSize = 32
        self.MaxEpochs = 30
        self.Tr_datasetPath = tr_path
        self.Te_datasetPath = te_path

    def run(self):
        if self.fold == '1-fold':
            train_path = f'{self.Tr_datasetPath}/A/iris'
            test_path = f'{self.Te_datasetPath}/B/iris'
            traincnt = 4554
            testcnt = 5018
        else:
            train_path = f'{self.Tr_datasetPath}/B/iris'
            test_path = f'{self.Te_datasetPath}/A/iris'
            traincnt = 5018
            testcnt = 4554

        train_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=self.batchSize, shuffle=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(test_path, target_size=(224, 224), batch_size=self.batchSize, shuffle=False)

        BackBone = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        print('Adding Average Pooling Layer and Softmax Output Layer ...')
        output = BackBone.get_layer(
            index=-1).output  # Shape: (7, 7, 2048) # for Densetnet: 1024; Resnet/Inception: 2048

        pam = PAM()(output)
        pam = tf.keras.layers.Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(pam)
        pam = tf.keras.layers.BatchNormalization(axis=3)(pam)
        pam = tf.keras.layers.Activation('relu')(pam)
        pam = tf.keras.layers.Dropout(0.5)(pam)
        pam = tf.keras.layers.Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(pam)

        cam = CAM()(output)
        cam = tf.keras.layers.Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(cam)
        cam = tf.keras.layers.BatchNormalization(axis=3)(cam)
        cam = tf.keras.layers.Activation('relu')(cam)
        cam = tf.keras.layers.Dropout(0.5)(cam)
        cam = tf.keras.layers.Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(cam)

        feature_sum = tf.keras.layers.Add()([pam, cam])

        feature_sum = Conv2d_BN(feature_sum, 512, 1)
        feature_sum = tf.keras.layers.GlobalAveragePooling2D()(feature_sum)
        attention_output = tf.keras.layers.Dense(2, activation='softmax')(feature_sum)

        DenseNet_model = tf.keras.Model(BackBone.input, attention_output)
        DenseNet_model.summary()

        lr = 0.0001
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        test_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        loss_mean = tf.keras.metrics.Mean()
        DenseNet_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        ###########################################################################################
        #                                   set CallBack
        ###########################################################################################
        log_dir = f'Z:/2nd_paper/backup/Compare/Detectors/AGPAD/{self.Gmodel}/{self.fold}/CrossFold/try_5/logs/'
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        best_model_file = f'Z:/2nd_paper/backup/Compare/Detectors/AGPAD/{self.Gmodel}/{self.fold}/CrossFold/try_5/ckp/weight.h5'
        best_model = tf.keras.callbacks.ModelCheckpoint(best_model_file, monitor='val_accuracy', save_best_only=True, verbose=True)

        DenseNet_model.fit_generator(train_generator, epochs=30, steps_per_epoch=(traincnt // self.batchSize),
                                     validation_data=test_generator, validation_steps=(testcnt // self.batchSize),
                                     callbacks=[best_model, tb_callback])

        # weights_path = glob.glob(f'Z:/2nd_paper/backup/Compare/Detectors/AGPAD/{self.Gmodel}/{self.fold}/CrossFold/try_0/ckp/*.h5')[-1]
        # print('Loading model and weights from training process ...')
        #
        # DenseNet_model = load_model(weights_path, custom_objects={'PAM': PAM, 'CAM': CAM})
        # pred = DenseNet_model.predict(test_generator)
        #
        # label = []
        # for i in range(len(test_generator)):
        #     label.append(np.argmax(test_generator[i][1][0]))
        #
        # tn, fp, fn, tp = tf.math.confusion_matrix(label, pred)
        best_score = {'epoch': 0, 'acc': 0, 'apcer': 0, 'bpcer': 0, 'acer': 0}
        #
        # best_score['epoch'] = weights_path
        # best_score['acc'] = 1-((best_score[2] + best_score[3]) / 2)
        # best_score['apcer'] = (fp / (tn + fp) if (tn + fp) != 0 else 0) * 100
        # best_score['bpcer'] = (fn / (fn + tp) if (fn + tp) != 0 else 0) * 100
        # best_score['acer'] = (best_score[2] + best_score[3]) / 2

        return best_score
