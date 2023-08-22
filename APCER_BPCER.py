import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Detection Rate')
    plt.ylabel('True Detection Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()

path_list = ['Z:/Iris_dataset/scores/Warsaw/AGPAD/1-fold/original_2.csv',
             'Z:/Iris_dataset/scores/Warsaw/AGPAD/2-fold/original_2.csv']

full_apcer = 0
full_bpcer = 0
count = 1
for path in path_list:
    original = pd.read_csv(path)
    proba_value = np.array(original.drop('Unnamed: 0', axis=1).astype('float32'))

    print(proba_value)
    predict_list = []
    a = 0
    b = 0

    hold = 2592
    if count == 2:
        hold = 2576

    print(f'Hold : {hold}')
    for i in range(len(proba_value)):
        if proba_value[i][0] > proba_value[i][1]:
            if i <= hold:
                predict_list.append([i, 0, 0])
            else:
                predict_list.append([i, 0, 1])
        else:
            if i <= hold:
                predict_list.append([i, 1, 0])
            else:
                predict_list.append([i, 1, 1])

    print(len(predict_list))

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(predict_list)):
        if predict_list[i][1] == 1 and predict_list[i][2] == 0:
            fp = fp + 1
        elif predict_list[i][1] == 1 and predict_list[i][2] == 1:
            tp = tp + 1
        elif predict_list[i][1] == 0 and predict_list[i][2] == 1:
            fn = fn + 1
        elif predict_list[i][1] == 0 and predict_list[i][2] == 0:
            tn = tn + 1

    apcer = fn / (tp + fn)
    bpcer = fp / (fp + tn)

    print(tp)
    print(fp)
    print(tn)
    print(fn)

    count = count + 1

    print(f"APCER : {apcer * 100}")
    print(f"BPCER : {bpcer * 100}")
    print(f"ACER : {((apcer + bpcer)/2)*100}")

    full_apcer = full_apcer + apcer
    full_bpcer = full_bpcer + bpcer

a = full_apcer / 2
b = full_bpcer / 2
c = (a + b) / 2

print(f"avg APCER : {a * 100}")
print(f"avg BPCER : {b * 100}")
print(f"avg APCER : {c * 100}")