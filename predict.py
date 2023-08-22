# import matplotlib

# matplotlib.use('Agg')
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import pandas as pd
import numpy as np

# import glob
# import matplotlib.pyplot as plt
# import cv2
# from sklearn import metrics
# import pickle
# # from models.attention_module import attach_attention_module
from tensorflow.keras import backend as K
import tensorflow as tf
# from sklearn.metrics import confusion_matrix
from attention import PAM, CAM
# from layers.attention_module import cbam_block
# from layers.gc import global_context_block

from shutil import copyfile
import time

# Compatible with tensorflow backend
# batchsz = 1
# traincnt = 4554
# testcnt = 5018
# valcnt = 1036

# cyclegan data
batchsz = 2
traincnt = 4554
testcnt = 5018
# valcnt = 1036

eps = 10e-8
label = ['fake', 'live']
path = 'Z:/2nd_paper/dataset/ND/ROI/Ablation/NestedUVC_DualAttention_Parallel_Fourier_MSE'

train_path = f'{path}/1-fold/A/iris'
test_path = f'{path}/1-fold/B/iris'
attack_path = f'{path}/Attack/Gamma/B/gamma_0.8/iris'

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=batchsz, shuffle=False)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = train_datagen.flow_from_directory(attack_path, target_size=(224, 224), batch_size=batchsz, shuffle=False)
# print(len(test_generator))
# print(np.argmax(test_generator[0][1][0]))
def focal_loss(gamma=1., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = K.clip(y_pred, eps, 1 - eps)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon() + pt_1)) - K.mean(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))

    return focal_loss_fixed


# img_width = 224
# img_height = 224
# batch_size = 32
# nbr_test_samples = 1000

FishNames = ['fake', 'live']

root_path = "Z:/2nd_paper/backup/Compare/Detectors/Linux/AGPAD/UVC_GAN/1-fold/NotCross/try_3/ckp/"  # path to the saved model

weights_path = os.path.join(root_path, 'weights-0001.h5')

print(weights_path)

print('Loading model and weights from training process ...')
# InceptionV3_model = load_model(weights_path)
DenseNet_model = load_model(weights_path, custom_objects={'PAM': PAM, 'CAM': CAM})
# InceptionV3_model = load_model(weights_path,custom_objects={'cbam_block':cbam_block})

print('Begin to predict for testing data ...')
history = DenseNet_model.evaluate(test_generator)
print(history)

# for step in range(6):
#     # img, label = next(test_generator)
#     # start = int(round(time.time() * 1000))
#     # output = DenseNet_model(img, training=False)
#     # end = int(round(time.time() * 1000))
#     # print("milli second : ", end - start)
# predict = DenseNet_model.predict(train_generator)
# print(predict)
#
# label = []
# for i in range(len(train_generator)):
#     label.append(np.argmax(train_generator[i][1][0]))
#
# label_df = pd.DataFrame(np.array(label))
# df = pd.DataFrame(predict)
#
# # df = pd.merge([df, label_df])
#
# OUTPUT_DIR = 'Z:/Iris_dataset/scores/Warsaw/AGPAD/2-fold'
# if not os.path.exists(OUTPUT_DIR):
#     os.makedirs(OUTPUT_DIR)
#
# df.to_csv(f'{OUTPUT_DIR}/original_2.csv')
# label_df.to_csv(f'{OUTPUT_DIR}/blur_label.csv')
# prediction = 1
# if prediction == 1:
#     gt_labels = []
#     pd_labels = []
#     pd_scores = []
#     for filename in glob.glob(f'{test_path}/*.png'):  #
#         # determine the GT label, remember this filename contains entire directory
#         if 'Live' in filename:  # Hence, the directory name must not contain "Live"
#             class_label = 0
#         elif 'live' in filename:
#             class_label = 0
#         else:
#             class_label = 1
#         gt_labels.append(class_label)
#
#         image = cv2.imread(filename, -1)
#         save_image = image
#         image = np.float32(cv2.resize(image, (img_height, img_width)))
#         image = image.reshape(1, img_height, img_width, 3) * 1. / 255
#         start = time.time()
#         out = InceptionV3_model.predict(image)
#         end = time.time()
#         class_prediction = list(out[0]).index(max(out[0]))
#         path, img_filename = os.path.split(filename)
#         if out[0][0] > out[0][1]:
#             pd_label = 0
#             pd_scores.append(1 - out[0][0])
#         else:
#             pd_label = 1
#             pd_scores.append(out[0][1])
#         pd_labels.append(pd_label)
#
#         # debug cls errors
#         if out[0][0] > out[0][1] and class_label == 1:
#             print(filename, out[0])
#             copyfile(filename, 'error_samples/' + img_filename)
#         if out[0][0] < out[0][1] and class_label == 0:
#             print(filename, out[0])
#             copyfile(filename, 'error_samples/' + img_filename)
#
#     confusion_matrix(gt_labels, pd_labels)
#     tn, fp, fn, tp = confusion_matrix(gt_labels, pd_labels).ravel()
#     # print tn, fp, fn, tp
#     # calculate error rate
#     apcer = 1.0 * fn / (fn + tp)
#     bpcer = 1.0 * fp / (tn + fp)
#
#     apcer * 100, bpcer * 100
#     # print accuracy_score(gt_labels, pd_labels)
#     fpr, tpr, thresholds = metrics.roc_curve(gt_labels, pd_scores, pos_label=1)
#
#     metrics.auc(fpr, tpr)
#     plt.semilogx(fpr * 100, tpr * 100, 'm', linewidth=2.0, linestyle='-.')
#     plt.ylabel('True Detection Rate (%)')
#     plt.xlabel('False Detection Rate (%)')
#     plt.axvline(x=0.2, color='k', linestyle='--')
#     # plt.ylim(80, 100)
#     plt.grid()
#     # plt.show()
#
#     plt.savefig('ROC.png')
#
#     textfile_fpr = open('fpr_output.txt', 'w')
#     textfile_tpr = open('tpr_output.txt', 'w')
#     textfile_threshold = open('threshold_output.txt', 'w')
#     i = 0
#     while i < len(fpr):
#         textfile_fpr.write("%s\n" % fpr[i])
#         textfile_tpr.write("%s\n" % tpr[i])
#         textfile_threshold.write("%s\n" % thresholds[i])
#         i = i + 1
#     textfile_fpr.close()
#     textfile_tpr.close()
#     textfile_threshold.close()