from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from pathlib import Path
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
import random
import SimpleITK as sitk
import numpy as np

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

def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

def copy_files(src_data_folder: Path, train_dir: Path, labelTr_dir: Path, test_dir: Path,labelTs_dir: Path):
    """Copy files from the ACDC dataset to the nnUNet dataset folder. Returns the number of training cases."""

    patients_train = []#这里为总的，后续会划分训练集和测试集
    for root, dirs, files in os.walk(src_data_folder, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            if "0.nii.gz" in path:
                patients_train.append(path)

    f = open("train.txt")  # train
    path_list = []
    for line in f.readlines():
            path_list.append(line.split('\n')[0])
    path_list.sort()

    num_cases = 0
    num_training_cases=0
    train =os.path.join(nnUNet_raw,"Dataset301_Breast","train.json")
    trainObject = open(train, 'a', encoding='utf-8')
    trainObject.seek(0)#
    trainObject.truncate()#clear content

    num_test_cases = 0
    test = os.path.join(nnUNet_raw, "Dataset301_Breast", "test.json")
    testObject = open(test, 'a', encoding='utf-8')
    testObject.seek(0)  #
    testObject.truncate()  # clear content

    splits_file = os.path.join(nnUNet_raw,"Dataset301_Breast", "splits_final.json")
    splits = []
    train_keys=[]
    test_keys=[]
    # Copy training files and corresponding labels.
    for path_str in patients_train:
        file=Path(path_str)
        if file.name == "0.nii.gz":
            # We split the stem and append _0000 to the patient part.
            shutil.copy(file, train_dir / f"Breast_{str(num_cases).zfill(4)}_0000.nii.gz")
            label_path=path_str.replace("0.nii.gz","1.nii.gz")
            se0output=labelTr_dir / f"Breast_{str(num_cases).zfill(4)}.nii.gz"
            read0 = sitk.ReadImage(label_path, sitk.sitkInt16)  # 使用sitk重新保存，这样占用内存小很多
            img_array0 = sitk.GetArrayFromImage(read0)
            # img_array0=np.where(img_array0>2,3,img_array0)
            out0 = sitk.GetImageFromArray(img_array0)
            sitk.WriteImage(out0, se0output)
            # shutil.copy(path_str.replace("0.nii.gz","2.nii.gz"), labelTr_dir / f"Breast_{str(num_cases).zfill(4)}.nii.gz")
            new_data = {f"Breast_{str(num_cases).zfill(4)}": path_str}
            js = json.dumps(new_data, ensure_ascii=False)
            if path_str in path_list:#path_list
                num_training_cases += 1
                trainObject.write(js +'\n')
                train_keys.append(f"Breast_{str(num_cases).zfill(4)}")
            else:
                num_test_cases += 1
                testObject.write(js + '\n')
                test_keys.append(f"Breast_{str(num_cases).zfill(4)}")
            num_cases += 1
        else:
            print(path_str)
    splits.append({})
    splits[-1]['train'] = train_keys#list(train_keys)
    splits[-1]['val'] = test_keys#list(test_keys)
    save_json(splits, splits_file)
    trainObject.close()
    testObject.close()
    return num_cases,num_test_cases

def convert_Breast(src_data_folder: str, dataset_id=301):
    out_dir, train_img_dir, train_labels_dir, test_img_dir,test_labels_dir = make_out_dirs(dataset_id=dataset_id)
    num_cases,num_test_cases = copy_files(src_data_folder, train_img_dir, train_labels_dir, test_img_dir,test_labels_dir)

    generate_dataset_json(
        str(out_dir),
        channel_names={
            0: "MRI",
        },
        labels={
            "background": 0,
            "lesion": 1,
        },
        file_ending=".nii.gz",
        num_training_cases=num_cases,
    )


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
    input_folder="/root/autodl-tmp/yml/data/xy/breast"  #define training and test datasets by self
    dataset_id=301
    convert_Breast(input_folder, dataset_id)
    print("Done!")
