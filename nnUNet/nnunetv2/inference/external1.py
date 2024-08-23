import inspect
import itertools
import multiprocessing
import os
import traceback
from copy import deepcopy
from time import sleep
from typing import Tuple, Union, List, Optional

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy, preprocessing_iterator_fromfiles, \
    preprocessing_iterator_fromnpy
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
from nnunetv2.postprocessing.remove_connected_components import remove_all_but_largest_component_from_segmentation
import copy
from nnunetv2.paths import nnUNet_results, nnUNet_raw
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from collections import Counter
import SimpleITK as sitk
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from nnunetv2.paths import nnUNet_raw,nnUNet_preprocessed
import json

def read_predictions(file_path):
    exclude=['3250156','3227288']
    predictions = {}
    with open(file_path, 'r') as file:
        for line in file:
            id, pred, true = line.strip().split()
            id = str(id)
            pred = int(pred)
            true = int(true)
            if id in exclude:
                continue
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

def kappa_cal(confusion_matrix):
    pe_rows = np.sum(confusion_matrix, axis=0)
    pe_cols = np.sum(confusion_matrix, axis=1)
    sum_total = sum(pe_cols)
    pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
    po = np.trace(confusion_matrix) / float(sum_total)
    return (po - pe) / (1 - pe)

def total_caculate():
    a1=[[82,22,0],
       [19,12,0],
        [6,0,43]]
    a2=[[32,3,0],
       [13,3,1],
       [1,0,9]]

    confm = np.array(a2)
    # ckap = cohen_kappa_score(y_true, y_pred)
    ckap=kappa_cal(confm)
    Sensitivity, specificity, accuracy, f1_macro = calculate_metrics(confm)
    # f1_macro = f1_score(y_true, y_pred, average='macro')
    print("Current  Avg. confm:\n {} ".format(confm))
    print("Current  Avg. ckap: {} ".format(ckap))
    print("Current  Avg. acc: {} ".format(accuracy))
    print("Current  Avg. Sensitivity: {} ".format(Sensitivity))
    print("Current  Avg. specificity: {} ".format(specificity))
    print("Current  Avg. f1_score: {} ".format(f1_macro))

if __name__ == '__main__':
    total_caculate()
    path_list = []
    #set1_step1
    for root, dirs, files in os.walk("./set1_step2/", topdown=False):#set1_step1 set1_step2
        for file in files:
            path = os.path.join(root, file)
            if "external" in path:
                path_list.append(path)
    voted_results = process_files(path_list)
    file = "./set1_step2/total_external.txt" #注意剔除第一步就将病变漏诊的数据
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



