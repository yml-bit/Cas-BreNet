import os
import numpy as np
import SimpleITK as sitk
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import json

def read_predictions(file_path):
    predictions = {}
    with open(file_path, 'r') as file:
        for line in file:
            id, pred, true = line.strip().split()
            # id = int(id)
            pred = int(pred)
            true = int(true)
            if id not in predictions:
                predictions[id] = {'preds': [], 'true': true}
            predictions[id]['preds'].append(pred)
    return predictions

def majority_voting(predictions):
    voted_preds = []
    for id, data in predictions.items():
        pred_counts = {}
        for pred in data['preds']:
            if pred not in pred_counts:
                pred_counts[pred] = 0
            pred_counts[pred] += 1
        voted_pred = max(pred_counts, key=pred_counts.get)
        voted_preds.append((id, voted_pred, data['true']))
    return voted_preds

def process_files(file_paths):
    all_predictions = {}
    for file_path in file_paths:
        predictions = read_predictions(file_path)
        for id, data in predictions.items():
            if id not in all_predictions:
                all_predictions[id] = {'preds': [], 'true': data['true']}
            all_predictions[id]['preds'].extend(data['preds'])
    voted_preds = majority_voting(all_predictions)
    return voted_preds

def calculate_metrics(confusion_matrix):
    # 2-TP/TN/FP/FN的计算
    weight=confusion_matrix.sum(axis=0)/confusion_matrix.sum()## 求出每列元素的和
    FN = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FP = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)#所有对的 TP.sum=TP+TN
    TN = confusion_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # print(TP)
    # print(TN)
    # print(FP)
    # print(FN)

    # 3-其他的性能参数的计算
    TPR = TP / (TP + FN)  # Sensitivity/ hit rate/ recall/ true positive rate 对于ground truth
    TNR = TN / (TN + FP)  # Specificity/ true negative rate  对于
    PPV = TP / (TP + FP)  # Precision/ positive predictive value  对于预测而言
    NPV = TN / (TN + FN)  # Negative predictive value
    FPR = FP / (FP + TN)  # Fall out/ false positive rate
    FNR = FN / (TP + FN)  # False negative rate
    FDR = FP / (TP + FP)  # False discovery rate
    sub_ACC = TP / (TP + FN)  # accuracy of each class
    acc=(TP+TN).sum()/(TP+TN+FP+FN).sum()
    average_acc=TP.sum() / (TP.sum() + FN.sum())
    F1_Score=2*TPR*PPV/(PPV+TPR)
    Macro_F1=F1_Score.mean()
    return TPR.mean(), TNR.mean(), average_acc,Macro_F1

if __name__ == '__main__':
    path_list = []
    #set1_step1
    for root, dirs, files in os.walk('./output/CNN3D', topdown=False):#CNN3D
        for file in files:
            path = os.path.join(root, file)
            if "record" in path:
                path_list.append(path)
    voted_results = process_files(path_list)
    file = "./output/CNN3D/total_internal.txt" #CNN3D MSCNN3D
    f_test = open(file, "w")
    y_true=[]
    y_pred=[]
    for result in voted_results:
        # print(f"{result[0]} {result[1]} {result[2]}")
        line = f"{result[0]}    {result[1]} {result[2]}"
        f_test.writelines(str(line) + "\n")
        y_true.append(result[2])
        y_pred.append(result[1])
    confm = confusion_matrix(y_true, y_pred)
    ckap = cohen_kappa_score(y_true, y_pred)
    Sensitivity, specificity, accuracy, f1_macro = calculate_metrics(confm)
    # f1_macro = f1_score(y_true, y_pred, average='macro')
    print("Current  Avg. confm:\n {} ".format(confm))
    print("Current  Avg. ckap: {} ".format(ckap))
    print("Current  Avg. acc: {} ".format(accuracy))
    print("Current  Avg. Sensitivity: {} ".format(Sensitivity))
    print("Current  Avg. specificity: {} ".format(specificity))
    print("Current  Avg. f1_score: {} ".format(f1_macro))
    f_test.close()



