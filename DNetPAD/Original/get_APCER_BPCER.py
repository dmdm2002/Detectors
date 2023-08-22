import pandas as pd
import numpy as np

path = './OutputPath/warsaw/1-fold/Results/DesNet121_Match_Scores.csv'
results_df = pd.read_csv(path, header=None)
results_arr = np.array(results_df)

label = [sub_arr[1] for sub_arr in results_arr]

# predict_prob = [sub_arr[2] for sub_arr in results_arr]

predict_label = []
for sub_arr in results_arr:
    if sub_arr[2] > 0.5:
        predict_label.append(1)
    else:
        predict_label.append(0)

tp = 0
fp = 0
tn = 0
fn = 0

for i in range(len(predict_label)):
    if label[i] == 1 and predict_label[i] == 1:
        tp = tp + 1

    elif label[i] == 0 and predict_label[i] == 0:
        tn = tn + 1

    elif label[i] == 1 and predict_label[i] == 0:
        fn = fn + 1

    elif label[i] == 0 and predict_label[i] == 1:
        fp = fp + 1

apcer = fn / (tp + fn)
bpcer = fp / (fp + tn)

print(tp)
print(fp)
print(tn)
print(fn)

# print(6 / 2509)
print(f"APCER : {apcer}")
print(f"BPCER : {bpcer}")