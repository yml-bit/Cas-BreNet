from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from pathlib import Path
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw,nnUNet_preprocessed
import random
import SimpleITK as sitk
import numpy as np
import nibabel as nib
import copy

def make_out_dirs(dataset_id: int, task_name="Breast"):
    dataset_name = f"Dataset{dataset_id:03d}_{task_name}"
    out_dir = Path(nnUNet_raw.replace('"', "")) / dataset_name
    train_img_dir = out_dir / "imagesTr"
    train_labels_dir = out_dir / "labelsTr"
    test_img_dir = out_dir / "imagesTs"
    test_labels_dir = out_dir / "labelsTs"
    os.makedirs(out_dir, exist_ok=True)

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_labels_dir, exist_ok=True)

    return out_dir, train_img_dir, train_labels_dir, test_img_dir,test_labels_dir

def convert_Breast(src_data_folder: str, dataset_id=301):
    out_path=nnUNet_raw
    patients_train = []  # 这里为总的，后续会划分训练集和测试集

    # for root, dirs, files in os.walk(src_data_folder, topdown=False):
    #     for file in files:
    #         path = os.path.join(root, file)
    #         if "0.nii.gz" in path and "normal" not in path:
    #             patients_train.append(path)

    for root, dirs, files in os.walk(src_data_folder.replace("set1","external"), topdown=False):
        for file in files:
            path = os.path.join(root, file)
            if "0.nii.gz" in path and "normal" not in path:
                patients_train.append(path)
    ii=0
    for path_str in patients_train:
        file = Path(path_str)
        if file.name == "0.nii.gz":
            # if "XQ486074" in path_str:
            #     a=1
            # We split the stem and append _0000 to the patient part.
            se0output = os.path.join(nnUNet_raw, path_str.split("breast/")[1])
            se1output = se0output.replace("0.nii.gz", "1.nii.gz")
            os.makedirs(se0output.split("/0.nii.gz")[0])
            read0 = sitk.ReadImage(file, sitk.sitkInt16)
            img = sitk.GetArrayFromImage(read0)
            image_size = img.shape
            if "external" in path_str:
                img = img // 4

            label_path = path_str.replace("0.nii.gz", "1.nii.gz")
            read1 = sitk.ReadImage(label_path, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
            label = sitk.GetArrayFromImage(read1)

            left_mask = copy.deepcopy(label)
            right_mask = copy.deepcopy(label)
            left_mask[:, :, image_size[1] // 2:] = 0
            right_mask[:, :, :image_size[1] // 2] = 0
            if np.sum(left_mask) >= np.sum(right_mask):
                label = left_mask
            elif np.sum(left_mask) < np.sum(right_mask):
                label = right_mask
            binary_mask = (copy.deepcopy(label) != 0)
            img = img * binary_mask
            index = np.nonzero(binary_mask)
            index = np.transpose(index)
            z_min, y_min, x_min = index.min(axis=0)
            z_max, y_max, x_max = index.max(axis=0)
            img_array = img[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]
            label_array = label[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]

            # if label_array.max()>4.5:
            #     continue

            label_array[(label_array >= 0.5) & (label_array <= 2.5)] = 1
            label_array[(label_array >= 2.5) & (label_array <= 4.5)] = 2
            label_array[label_array > 4.5] = 3

            out0 = sitk.GetImageFromArray(img_array)
            sitk.WriteImage(out0, se0output)
            out1 = sitk.GetImageFromArray(label_array)
            sitk.WriteImage(out1, se1output)

            ii = ii + 1
            if ii % 10 == 0:
                print('numbers:', ii)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        help="The downloaded Breast dataset dir. Should contain extracted 'training' and 'testing' folders.",
    )
    parser.add_argument(
        "-d", "--dataset_id", required=False, type=int, default=301, help="nnU-Net Dataset ID, default: 301"
    )
    args = parser.parse_args()
    print("Converting...")
    # convert_Breast(args.input_folder, args.dataset_id)
    input_folder="/media/bit301/data/yml/data/xy/breast/set1"  #set1 external
    dataset_id=301
    convert_Breast(input_folder, dataset_id)
    print("Done!")
