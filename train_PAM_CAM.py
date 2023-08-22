import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from attention import PAM, CAM


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
traincnt = 5018
testcnt = 4554
# valcnt = 1036

label = ['fake', 'live']
path = 'Z:/2nd_paper/dataset/ND/ROI/Ablation/UVC_GAN'

train_path = f'{path}/2-fold/A/iris'
test_path = f'{path}/1-fold/B/iris'

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=batchsz)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = train_datagen.flow_from_directory(test_path, target_size=(224, 224), batch_size=batchsz)

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
log_dir = 'Z:/2nd_paper/backup/Compare/Detectors/Linux/AGPAD/UVC_GAN/CrossFold/1-fold/try_3/logs/'
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

best_model_file = "Z:/2nd_paper/backup/Compare/Detectors/Linux/AGPAD/UVC_GAN/CrossFold/1-fold/try_3/ckp/weights-{epoch:04d}.h5"
best_model = tf.keras.callbacks.ModelCheckpoint(best_model_file, monitor='val_accuracy', save_best_only=True)


DenseNet_model.fit_generator(train_generator, epochs=30, steps_per_epoch=(traincnt//batchsz),
                                validation_data=test_generator, validation_steps=(testcnt//batchsz), callbacks=[best_model, tb_callback])
