import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from attention import PAM, CAM
from makeDataset import MkDataset
import numpy as np
import datetime
import time


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


def get_confusion_matrix(label, pred):
    """output : tn, fp, fn, tp"""
    return np.ravel(tf.math.confusion_matrix(labels=label, predictions=pred, num_classes=2))

###########################################################################################
#                                       set dataset
###########################################################################################

# original data
# batchsz = 2
# traincnt = 5154
# testcnt = 5182
# valcnt = 1191

# cyclegan data
batchsz = 2
Acnt = 4554
Bcnt = 5018
# valcnt = 1036

path = 'Z:/2nd_paper/dataset/ND/ROI/Compare/iDCGAN/'

A_path = f'{path}/Attack/Gamma/A/gamma_0.8/iris'
B_path = f'{path}/Attack/Gamma/B/gamma_0.8/iris'

classes = ['fake', 'live']
Mkds = MkDataset()

"""A dataset Path list"""
A_iris_Fdata = Mkds.make_path_list(A_path, classes[0])
A_iris_Tdata = Mkds.make_path_list(A_path, classes[1])

A_iris_path = np.concatenate((A_iris_Fdata, A_iris_Tdata), axis=None)

"""B dataset Path list"""
B_iris_Fdata = Mkds.make_path_list(B_path, classes[0])
B_iris_Tdata = Mkds.make_path_list(B_path, classes[1])

B_iris_path = np.concatenate((B_iris_Fdata, B_iris_Tdata), axis=None)

"""make A dataset"""
# make label dataset
A_labels = Mkds.get_label(A_iris_path)
A_labels = tf.one_hot(A_labels, 2)
A_labels_ds = tf.data.Dataset.from_tensor_slices(A_labels)

# make image dataset
A_iris_ds = Mkds.make_ds(A_iris_path)
# zip three roi image dataset, label dataset
A_ds = tf.data.Dataset.zip((A_iris_ds, A_labels_ds))

"""make B dataset"""
# make label dataset
B_labels = Mkds.get_label(B_iris_path)
B_labels = tf.one_hot(B_labels, 2)
B_labels_ds = tf.data.Dataset.from_tensor_slices(B_labels)

# make image dataset
B_iris_ds = Mkds.make_ds(B_iris_path)

# zip three roi image dataset, label dataset
B_ds = tf.data.Dataset.zip((B_iris_ds, B_labels_ds))


###########################################################################################
#                                       model
###########################################################################################

BackBone = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

print('Adding Average Pooling Layer and Softmax Output Layer ...')
output = BackBone.get_layer(index=-1).output  # Shape: (7, 7, 2048) # for Densetnet: 1024; Resnet/Inception: 2048

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
feature_sum = tf.keras.layers.Dropout(0.5)(feature_sum)

feature_sum = Conv2d_BN(feature_sum, 512, 1)
feature_sum = tf.keras.layers.GlobalAveragePooling2D()(feature_sum)
attention_output = tf.keras.layers.Dense(2, activation='softmax')(feature_sum)

DenseNet_model = tf.keras.Model(BackBone.input, attention_output)
DenseNet_model.summary()

# lr = 0.0001
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
test_acc_metric = tf.keras.metrics.CategoricalAccuracy()
loss_mean = tf.keras.metrics.Mean()

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# set CheckPoint
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=DenseNet_model)

###########################################################################################
#                                   set CallBack
###########################################################################################


@tf.function
def test_step(x, y):
    val_logits = DenseNet_model(x, training=False)
    loss_val = loss_fn(y, val_logits)
    test_acc_metric.update_state(y, val_logits)

    return loss_val, val_logits


epochs = 50
best_epoch_acc = [0, 0, 0, 0, 0]
for epoch in range(2, 3):
    tp, tn, fp, fn = 0, 0, 0, 0
    print("\nStart test of epoch %d" % (epoch,))
    start_time = time.time()

    ckp_path = f"Z:/2nd_paper/backup/Compare/Detectors/Linux/AGPAD/iDCGAN/CrossFold/2-fold/try_0/ckp/ckpt-{epoch}"
    ckpt.restore(ckp_path)

    A_ds_shuffle = Mkds.configure_for_performance(A_ds, Acnt, batchsz, shuffle=False)
    A_ds_it = iter(A_ds_shuffle)

    B_ds_shuffle = Mkds.configure_for_performance(B_ds, Bcnt, batchsz, shuffle=False)
    B_ds_it = iter(B_ds_shuffle)

    for i in range(Bcnt//batchsz):
        iris_img, iris_label = next(B_ds_it)
        loss_val, val_logits = test_step(iris_img, iris_label)

        prediction = tf.nn.softmax(val_logits)
        prediction = prediction.numpy()
        prediction = np.argmax(prediction, axis=1).ravel()
        label = np.array(np.argmax(iris_label, axis=1)).ravel()

        tn_batch, fp_batch, fn_batch, tp_batch = get_confusion_matrix(label, prediction)

        tp += tp_batch
        tn += tn_batch
        fp += fp_batch
        fn += fn_batch


    test_acc = test_acc_metric.result()
    test_acc_metric.reset_states()
    loss_mean.reset_states()

    if best_epoch_acc[1] <= float(test_acc):
        best_epoch_acc[0] = epoch
        best_epoch_acc[1] = float(test_acc)
        best_epoch_acc[2] = fp / (tn + fp) if (tn + fp) != 0 else 0
        best_epoch_acc[3] = fn / (fn + tp) if (fn + tp) != 0 else 0
        best_epoch_acc[4] = (best_epoch_acc[2] + best_epoch_acc[3]) / 2

    print("Validation acc: %.4f" % (float(test_acc),))
    print(f'BEST VALIDATION EPOCH AND ACC')
    print(f'||epoch : {best_epoch_acc[0]} | Acc : {best_epoch_acc[1]} ||')
    print(f'||APCER: {best_epoch_acc[2] * 100}  |  BPCER: {best_epoch_acc[3] * 100}  |  ACER: {best_epoch_acc[4] * 100}||')

    print("Validation acc: %.4f" % (float(test_acc),))
    train_acc_metric.reset_states()
    print("Time taken: %.2fs" % (time.time() - start_time))

print(f'BEST VALIDATION EPOCH AND ACC')
print(f'||epoch : {best_epoch_acc[0]} | Acc : {best_epoch_acc[1]} ||')
print(f'||APCER: {best_epoch_acc[2] * 100}  |  BPCER: {best_epoch_acc[3] * 100}  |  ACER: {best_epoch_acc[4] * 100}||')