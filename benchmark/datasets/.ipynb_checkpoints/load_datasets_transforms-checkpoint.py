import SimpleITK as sitk
import os
import numpy as np
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from .augmentation import intensity_shift, intensity_scale, random_rotate
import torch.nn.functional as F
import copy


from monai.transforms import (
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    SpatialPadd,
    ResizeWithPadOrCropd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    KeepLargestConnectedComponentd,
    Spacingd,
    ToTensord,
    RandAffined,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandRotate90d,
    EnsureTyped,
    Invertd,
    KeepLargestConnectedComponentd,
    SaveImaged,
    Activationsd
)

def data_loader(args):
    dataset = args.dataset
    out_classes = args.num_classes

    if args.mode == 'train':
        f = open(args.train_list)  # test train
        train_lista = []
        train_listb = []
        train_listc = []
        for line in f.readlines():
            line = line.strip('\n')  # 直接将文件中按行读到list里，效果与方法2一样
            train_lista.append(line)
            train_listb.append(line.replace("0.nii.gz", "1.nii.gz"))
            train_listc.append(line.split("/")[-2])
        f.close()  # 关

        train_samples = {}
        valid_samples = {}
        train_samples['images'] = train_lista
        train_samples['labels'] = train_listb
        train_samples['ID'] = train_listc

        #val
        f = open(args.val_list)  # test train
        val_lista = []
        val_listb = []
        val_listc=[]
        for line in f.readlines():
            line = line.strip('\n')  # 直接将文件中按行读到list里，效果与方法2一样
            val_lista.append(line)
            val_listb.append(line.replace("0.nii.gz", "1.nii.gz"))
            val_listc.append(line.split("/")[-2])
        f.close()  # 关

        valid_samples['images'] = val_lista
        valid_samples['labels'] = val_listb
        valid_samples['ID'] = val_listc

        print('Finished loading all training samples from dataset: {}!'.format(dataset))
        print('Number of classes for segmentation: {}'.format(out_classes))
        return train_samples, valid_samples, out_classes

    elif args.mode == 'val':
        valid_samples = {}
        f = open(args.val_list)  # test train
        val_lista = []
        val_listb = []
        val_listc=[]
        for line in f.readlines():
            line = line.strip('\n')  # 直接将文件中按行读到list里，效果与方法2一样
            val_lista.append(line)
            val_listb.append(line.replace("0.nii.gz", "1.nii.gz"))
            val_listc.append(line.split("/")[-2])
        f.close()  # 关

        valid_samples['images'] = val_lista
        valid_samples['labels'] = val_listb
        valid_samples['ID'] = val_listc

        print('Finished loading all inference samples from dataset: {}!'.format(dataset))
        print('Number of classes for segmentation: {}'.format(out_classes))
        return valid_samples, out_classes

    elif args.mode == 'test':
        test_samples = {}
        f = open(args.test_list)  # test train
        test_lista = []
        test_listb = []
        test_listc=[]
        for line in f.readlines():
            line = line.strip('\n')  # 直接将文件中按行读到list里，效果与方法2一样
            test_lista.append(line)
            test_listb.append(line.replace("0.nii.gz", "1.nii.gz"))
            test_listc.append(line.split("/")[-2])
        f.close()  # 关
        test_samples['images'] = test_lista
        test_samples['targets'] = test_listb
        test_samples['ID'] = test_listc

        # ## Input inference data
        # test_img = sorted(glob.glob(os.path.join(root_dir, 'imagesTs', '*.nii.gz')))
        # test_samples['images'] = test_img
        print('Finished loading all inference samples from dataset: {}!'.format(dataset))
        return test_samples, out_classes

def data_transforms(args):
    dataset = args.dataset
    if args.mode == 'train':
        crop_samples = args.crop_sample
    else:
        crop_samples = None
    if dataset == '301':#(96, 96, 96)  (96, 96, 48)  (1, 512, 462, 332)
        # patch_size=(128, 128, 48)#(96, 96, 64)#(96, 96, 48)
        patch_size = np.array(args.patch, dtype=int)  # (96,96,48)
        train_transforms = Compose(
            [
                # LoadImaged(keys=["image","label"]),
                EnsureChannelFirstd(keys=["image","label"],channel_dim='no_channel'),
                # Spacingd(keys=["image", "label"], pixdim=(
                #     1.0, 1.0, 1.0), mode=("bilinear", "nearest")), #1.5, 1.5, 2.0  重采样
                Orientationd(keys=["image","label"], axcodes="SAR"),#RAS
                ScaleIntensityRanged(
                    keys=["image"],a_min=0, a_max=500, #[-175 250]
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image","label"], source_key="image",allow_smaller=True), #去掉空白留白,保证输入的是数值图，
                # RandSpatialCrop(roi_size, max_roi_size=None, random_center=True, random_size=False, lazy=False)[source]
                RandCropByPosNegLabeld(
                    keys=["image","label"],
                    label_key="label",
                    spatial_size=patch_size,
                    pos=1,
                    neg=2, #阴性阳性比
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                    allow_smaller=True,  # 允许裁剪出的patch小于设定尺寸，如果病灶较小
                ),
                RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 2)),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=['image','label'],
                    mode=('bilinear','nearest'),
                    prob=1.0, spatial_size=patch_size,
                    rotate_range=(0, 0, np.pi / 30),
                    scale_range=(0.1, 0.1, 0.1)),
                # ToTensord(keys=["image","label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),
                # Spacingd(keys=["image", "label"], pixdim=(
                #     1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image","label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"],a_min=0, a_max=500, #[-175 250]
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                # CropForegroundd(keys=["image","target", "label"], source_key="image"),
                # ToTensord(keys=["image","target", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"],channel_dim='no_channel'),
                # Spacingd(keys=["image"], pixdim=(
                #     1.5, 1.5, 2.0), mode=("bilinear")),#1.5, 1.5, 2.0
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=500, #[-175 250]
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                # CropForegroundd(keys=["image"], source_key="image"),
                # ToTensord(keys=["image"]),#original
            ]
        )
        overal_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),
                # Spacingd(keys=["image", "label"], pixdim=(
                #     1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image","label"], axcodes="RAS"),
            ]
        )

    if args.mode == 'train':
        print('Cropping {} sub-volumes for training!'.format(str(crop_samples)))
        print('Performed Data Augmentations for all samples!')
        return train_transforms, val_transforms

    elif args.mode == 'val':
        print('Performed transformations for all samples!')
        return val_transforms

    elif args.mode == 'test':
        print('Performed transformations for all samples!')
        return test_transforms
    elif args.mode=="overal_test":
        return overal_transforms

def infer_post_transforms(output, test_transforms, out_classes):
    print("notice:comment the line 558 of monai/transforms/utility/dictionary.py")
    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Activationsd(keys="pred", softmax=True),

        Invertd(
            keys="pred",  # invert the `pred` data field, also support multiple fields
            transform=test_transforms,
            orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
            # then invert `pred` based on this information. we can use same info
            # for multiple fields, also support different orig_keys for different fields
            meta_keys="pred_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
            orig_meta_keys="image_meta_dict",  # get the meta data from `img_meta_dict` field when inverting,
            # for example, may need the `affine` to invert `Spacingd` transform,
            # multiple fields can use the same meta data to invert
            meta_key_postfix="meta_dict",  # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
            # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
            # otherwise, no need this arg during inverting
            nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
            # to ensure a smooth output, then execute `AsDiscreted` transform
            to_tensor=True,  # convert to PyTorch Tensor after inverting
        ),
        ## If monai version <= 0.6.0:
        # AsDiscreted(keys="pred", threshold=0.5),
        AsDiscreted(keys="pred", argmax=True, n_classes=out_classes),
        ## If moani version > 0.6.0:
        # AsDiscreted(keys="pred", argmax=True)
        # KeepLargestConnectedComponentd(keys='pred', applied_labels=[1, 3]),
        # SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=output,
        #            output_postfix="seg", output_ext=".nii.gz", separate_folder=False,resample=True),
    ])

    return post_transforms


#已测试
def remove_regions(mask):
    mask1=mask
    mask1=np.where(mask1>0,1,0)
    # mask2 = mask
    # mask2=np.where(mask2>1,1,0)
    segmentation_sitk = sitk.GetImageFromArray(mask1)
    # 计算标签图像的连通分量
    connected_components_filter = sitk.ConnectedComponentImageFilter()
    labeled_image = connected_components_filter.Execute(segmentation_sitk)

    # 获取每个连通分量的大小
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(labeled_image)

    # 找到最大的连通分量ID
    max_size = 0
    largest_label = 0
    for i in range(1, label_shape_filter.GetNumberOfLabels() + 1):  # Label index starts from 1
        if label_shape_filter.GetNumberOfPixels(i) > max_size:
            max_size = label_shape_filter.GetNumberOfPixels(i)
            largest_label = i

    # 仅保留最大连通分量
    binary_mask = sitk.Equal(labeled_image, largest_label)
    cleaned_segmentation = sitk.Cast(binary_mask, segmentation_sitk.GetPixelID())
    cleaned_segmentation = sitk.GetArrayFromImage(cleaned_segmentation)
    cleaned_segmentation=cleaned_segmentation*mask
    # print(cleaned_segmentation.max())
    return cleaned_segmentation

class RandomGenerator(object):
    def __init__(self, output_size, mode):
        self.output_size = output_size
        self.mode = mode

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        min_value = np.min(image)
        # centercop
        # crop alongside with the ground truth
        z, y, x = self.output_size
        if np.sum(label)==0:
            index = np.nonzero(image)
            index = np.transpose(index)
        else:
            index = np.nonzero(label)
            index = np.transpose(index)

        z_min = np.min(index[:, 0])
        z_max = np.max(index[:, 0])
        y_min = np.min(index[:, 1])
        y_max = np.max(index[:, 1])
        x_min = np.min(index[:, 2])
        x_max = np.max(index[:, 2])

        # middle point
        z_middle = int((z_min + z_max) / 2)
        y_middle = int((y_min + y_max) / 2)
        x_middle = int((x_min + x_max) / 2)

        if random.random() > 0.3:
            Delta_z = int((z_max - z_min) / 3)
            Delta_y = int((y_max - y_min) / 8)
            Delta_x = int((x_max - x_min) / 8)

        else:
            Delta_z = int((z_max - z_min) / 2) + self.output_size[0]
            Delta_y = int((y_max - y_min) / 8)  # 8
            Delta_x = int((x_max - x_min) / 8)

        z_random = random.randint(z_middle - Delta_z, z_middle + Delta_z)
        y_random = random.randint(y_middle - Delta_y, y_middle + Delta_y)
        x_random = random.randint(x_middle - Delta_x, x_middle + Delta_x)

        # crop patch
        crop_z_down = z_random - int(self.output_size[0] / 2)
        crop_z_up = z_random + int(self.output_size[0] / 2)
        crop_y_down = y_random - int(self.output_size[1] / 2)
        crop_y_up = y_random + int(self.output_size[1] / 2)
        crop_x_down = x_random - int(self.output_size[2] / 2)
        crop_x_up = x_random + int(self.output_size[2] / 2)

        if crop_z_down < 0 or crop_z_up > image.shape[0]:
            delta_z = np.maximum(np.abs(crop_z_down), np.abs(crop_z_up - image.shape[0]))
            image = np.pad(image, ((delta_z, delta_z), (0, 0), (0, 0)), 'constant', constant_values=min_value)
            label = np.pad(label, ((delta_z, delta_z), (0, 0), (0, 0)), 'constant', constant_values=0.0)

            crop_z_down = crop_z_down + delta_z
            crop_z_up = crop_z_up + delta_z

        if crop_y_down < 0 or crop_y_up > image.shape[1]:
            delta_y = np.maximum(np.abs(crop_y_down), np.abs(crop_y_up - image.shape[1]))
            image = np.pad(image, ((0, 0), (delta_y, delta_y), (0, 0)), 'constant', constant_values=min_value)
            label = np.pad(label, ((0, 0), (delta_y, delta_y), (0, 0)), 'constant', constant_values=0.0)

            crop_y_down = crop_y_down + delta_y
            crop_y_up = crop_y_up + delta_y

        if crop_x_down < 0 or crop_x_up > image.shape[2]:
            delta_x = np.maximum(np.abs(crop_x_down), np.abs(crop_x_up - image.shape[2]))
            image = np.pad(image, ((0, 0), (0, 0), (delta_x, delta_x)), 'constant', constant_values=min_value)
            label = np.pad(label, ((0, 0), (0, 0), (delta_x, delta_x)), 'constant', constant_values=0.0)

            crop_x_down = crop_x_down + delta_x
            crop_x_up = crop_x_up + delta_x
        # a,b,c=label.shape
        # if b!=c:
        #     print("wrong")
        label = label[crop_z_down: crop_z_up, crop_y_down: crop_y_up, crop_x_down: crop_x_up]
        image = image[crop_z_down: crop_z_up, crop_y_down: crop_y_up, crop_x_down: crop_x_up]

        label = np.round(label)

        # data augmentation
        if self.mode == 'train':
            if random.random() > 0.5:
                image = intensity_shift(image)
            if random.random() > 0.5:
                image = intensity_scale(image)
            if random.random() > 0.5:
                image, label = random_rotate(image, label, min_value)
                label = np.round(label)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).float()
        label = torch.from_numpy(label.astype(np.float32)).float()
        if label.shape[-1] == 0:
            label=torch.zeros_like(image.squeeze(dim=0))
        label=label[np.newaxis,:,:,:,]
        sample = {'image': torch.Tensor(image), 'label': torch.Tensor(label.long())}
        # print(" # crop alongside with the ground truth")
        return sample

class Generator(object):
    def __init__(self, output_size, mode):
        self.output_size = output_size
        self.mode = mode

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        patch_size=self.output_size
        image_size=image.shape
        index = np.nonzero(label)
        index = np.transpose(index)
        z_min, y_min, x_min = index.min(axis=0)
        z_max, y_max, x_max = index.max(axis=0)

        # 确保即使病灶位于边缘也能获取到完整的patch
        z_middle = max(z_min, z_max - patch_size[0] + 1) if z_max - patch_size[0] > z_min else z_min
        y_middle = max(y_min, y_max - patch_size[1] + 1) if y_max - patch_size[1] > y_min else y_min
        x_middle = max(x_min, x_max - patch_size[2] + 1) if x_max - patch_size[2] > x_min else x_min

        # 计算patch的起始坐标
        z_index = int(z_middle - patch_size[0] // 2)
        y_index = int(y_middle - patch_size[1] // 2)
        x_index = int(x_middle - patch_size[2] // 2)

        # 限制patch在原始图像范围之内
        patch_start = [max(0, z_index),
                       max(0, y_index),
                       max(0, x_index)]

        # 根据维度顺序调整patch_end的计算
        patch_end = [min(image_size[0], z_index + patch_size[0]),
                     min(image_size[1], y_index + patch_size[1]),
                     min(image_size[2], x_index + patch_size[2])]

        img_patche = image[patch_start[0]:patch_end[0],
                    patch_start[1]:patch_end[1],
                    patch_start[2]:patch_end[2]]
        img_patche = img_patche[np.newaxis,np.newaxis,:,:,:]
        label_patche = label[patch_start[0]:patch_end[0],
                    patch_start[1]:patch_end[1],
                    patch_start[2]:patch_end[2]]
        label_patche = label_patche[np.newaxis,np.newaxis,:,:,:]
        sample = {'image': torch.Tensor(img_patche), 'label': torch.Tensor(label_patche)}
        return sample

class dataset(Dataset):
    def __init__(self, list_dir, split, num_classes, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.num_classes = num_classes

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            img_path = self.sample_list[idx].strip('\n')
            ID=img_path.split('/')[-2]
            image = sitk.ReadImage(img_path)
            label_path = img_path.replace("0.nii.gz", "1.nii.gz")
            label = sitk.ReadImage(label_path)
            image = sitk.GetArrayFromImage(image)
            label = sitk.GetArrayFromImage(label)
            image = np.clip(image, 0, 500) / 500 ##monai需要注释掉            
            image = image.astype(np.float32)
            label = label.astype(np.float32)
            image_size = image.shape
            left_mask=copy.deepcopy(label)
            right_mask=copy.deepcopy(label)
            left_mask[:, :, image_size[1] // 2:] = 0
            right_mask[:, :, :image_size[1] // 2] = 0
            if np.sum(left_mask) > 0 and np.sum(right_mask) > 0:#左右两侧均有病灶时，随机选择一侧
                if random.random()>0.5:
                    label=left_mask
                else:
                    label =right_mask
            sample = {'image': image, 'label': label}
            if self.transform:
                sample = self.transform(sample)
                # print(" # crop alongside with the ground truth")
                if torch.sum(sample['label']) > 5:
                    flag = 1
                    # print("flag=1")
                else:
                    flag = 0
                    # print("flag=0")
                
                # #monai需要注释掉
                # a = self.transform(sample)
                # if torch.sum(a[0]['label'])>5:
                #     flag=1
                # else:
                #     flag=0
                # sample = {'image': a[0]['image'], 'label': a[0]['label']}
                
            sample['flag']=flag
            sample['case_name'] = img_path
            sample['ID'] = ID

        elif self.split == "val":
            img_path = self.sample_list[idx].strip('\n')
            ID = img_path.split('/')[-2]
            image = sitk.ReadImage(img_path)
            label_path = img_path.replace("0.nii.gz", "1.nii.gz")
            label = sitk.ReadImage(label_path)
            image = sitk.GetArrayFromImage(image)
            label = sitk.GetArrayFromImage(label)

            image = np.clip(image, 0, 500) / 500
            image = image.astype(np.float32)
            label = label.astype(np.float32)
            left_mask=copy.deepcopy(label)
            right_mask=copy.deepcopy(label)
            image_size = image.shape
            left_mask[:, :, image_size[1] // 2:] = 0
            right_mask[:, :, :image_size[1] // 2] = 0
            if np.sum(left_mask) > np.sum(right_mask):#左右两侧均有病灶时，随机选择一侧
                label=left_mask
            else:
                label =right_mask
            sample = {'image': image, 'label': label}
            if self.transform:
                sample = self.transform(sample)
            #     if torch.sum(sample['label']) > 5:
            #         flag = 1
            #     else:
            #         flag = 0
            # sample['flag']=flag
            sample['case_name'] = img_path
            sample['ID'] = ID
        else:
            img_path = self.sample_list[idx].strip('\n')
            ID = img_path.split('/')[-2]
            image = sitk.ReadImage(img_path)
            label_path = img_path.replace("0.nii.gz", "1.nii.gz")
            label = sitk.ReadImage(label_path)

            image = sitk.GetArrayFromImage(image)
            label = sitk.GetArrayFromImage(label)

            image = np.clip(image, 0, 500) / 500
            image = image.astype(np.float32)
            label = label.astype(np.float32)
            image_size = image.shape
            left_mask[:, :, image_size[1] // 2:] = 0
            right_mask[:, :, :image_size[1] // 2] = 0
            if np.sum(left_mask) > np.sum(right_mask):#左右两侧均有病灶时，随机选择一侧
                label=left_mask
            else:
                label =right_mask
            sample = {'image': image, 'label': label}
            if self.transform:
                sample = self.transform(sample)
            sample['case_name'] = img_path
            sample['ID'] = ID
        # label=label.transpose((2, 1, 0))
        # label[label < 0.5] = 0.0
        # label[label > 5.5] = 0.0
        return sample

def patch_extraction(image_path, patch_size, overlap_rate):
    ct_image = sitk.ReadImage(image_path)
    # 提取CT数据的数组表示形式
    ct_array = sitk.GetArrayFromImage(ct_image)

    # 获取CT数据的尺寸
    z_size, y_size, x_size = ct_array.shape

    # 计算patch的偏移量
    z_stride = int(patch_size[0] * (1 - overlap_rate))
    x_stride = int(patch_size[2] * (1 - overlap_rate))
    y_stride = int(patch_size[1] * (1 - overlap_rate))

    # 划分CT数据成patch
    patches = []
    for z in range(0, z_size - patch_size[0], z_stride):
        for y in range(0, y_size - patch_size[1], y_stride):
            for x in range(0, x_size - patch_size[2], x_stride):
                patch = ct_array[z:z + patch_size[0], y:y + patch_size[1], x:x + patch_size[2]]
                patches.append(patch)
    return patches

def patch_reconstruction11(segmented_patches, img_size, patch_size, overlap_rate):
    #计算patch的偏移量
    # step = tuple(int(s * overlap_rate) for s in patch_size)
    reconstructed_img = np.zeros(img_size)
    count = 0
    count_map = np.zeros(img_size)
    for i in range(0, img_size[0] - patch_size[0] + 1, int(patch_size[0] * (1 - overlap_rate))):
        for j in range(0, img_size[1] - patch_size[1] + 1, int(patch_size[1] * (1 - overlap_rate))):
            for k in range(0, img_size[2] - patch_size[2] + 1, int(patch_size[2] * (1 - overlap_rate))):
                # 如果segmented_patches是一个四维张量，索引应该是 [count, :, :, :]
                reconstructed_img[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]] += segmented_patches[count]
                count_map[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]] += 1
                count += 1

    # Avoid dividing by zero and take the average
    masked_count = np.ma.masked_equal(count_map, 0)
    reconstructed_img /= masked_count.filled(1)
    reconstructed_img = np.where(reconstructed_img >= 0.5, 1, 0)
    return reconstructed_img

def patch_reconstruction(label_patches, original_shape, patch_size, overlap_rate):
    z_size, y_size, x_size = original_shape
    z_stride = int(patch_size[0] * (1 - overlap_rate))
    y_stride = int(patch_size[1] * (1 - overlap_rate))
    x_stride = int(patch_size[2] * (1 - overlap_rate))

    # 初始化用于存储重建后图像和记录每个像素被覆盖次数的计数器
    reconstructed_label = np.zeros(original_shape)
    count_map = np.zeros(original_shape)

    for idx, label_patch in enumerate(label_patches):
        # 计算patch在原始图像中的起始坐标
        z_start = idx // ((y_size - patch_size[1]) // y_stride + 1) // ((x_size - patch_size[2]) // x_stride + 1) * z_stride
        y_start = (idx // ((x_size - patch_size[2]) // x_stride + 1)) % ((y_size - patch_size[1]) // y_stride + 1) * y_stride
        x_start = idx % ((x_size - patch_size[2]) // x_stride + 1) * x_stride

        # 考虑到最后一块patch可能不完整，使用min函数确保不会越界
        end_z = min(z_start + patch_size[0], z_size)
        end_y = min(y_start + patch_size[1], y_size)
        end_x = min(x_start + patch_size[2], x_size)

        # 将当前patch添加到相应的区域，并增加计数器
        reconstructed_label[z_start:end_z, y_start:end_y, x_start:end_x] += label_patch
        count_map[z_start:end_z, y_start:end_y, x_start:end_x] += 1

    # Avoid dividing by zero and take the average
    masked_count = np.ma.masked_equal(count_map, 0)
    reconstructed_label /= masked_count.filled(1)
    reconstructed_label = np.where(reconstructed_label >= 0.5, 1, 0)
    return reconstructed_label


def patch(image,mask,image_size,patch_size):
    index = np.nonzero(mask)
    index = np.transpose(index)
    z_min, y_min, x_min = index.min(axis=0)
    z_max, y_max, x_max = index.max(axis=0)

    # 确保即使病灶位于边缘也能获取到完整的patch
    z_middle = max(z_min, z_max - patch_size[0] + 1) if z_max - patch_size[0] > z_min else z_min
    y_middle = max(y_min, y_max - patch_size[1] + 1) if y_max - patch_size[1] > y_min else y_min
    x_middle = max(x_min, x_max - patch_size[2] + 1) if x_max - patch_size[2] > x_min else x_min

    # 计算patch的起始坐标
    z_index = int(z_middle - patch_size[0] // 2)
    y_index = int(y_middle - patch_size[1] // 2)
    x_index = int(x_middle - patch_size[2] // 2)

    # 限制patch在原始图像范围之内
    patch_start = [max(0, z_index),
                   max(0, y_index),
                   max(0, x_index)]

    # 根据维度顺序调整patch_end的计算
    patch_end = [min(image_size[0], z_index + patch_size[0]),
                 min(image_size[1], y_index + patch_size[1]),
                 min(image_size[2], x_index + patch_size[2])]

    Rio_patch = image[patch_start[0]:patch_end[0],
               patch_start[1]:patch_end[1],
               patch_start[2]:patch_end[2]]
    Rio_patch=Rio_patch.unsqueeze(dim=0)
    Rio_patch=Rio_patch.unsqueeze(dim=0)
    return Rio_patch

def roi_extraction(image_path, lesion_mask, patch_size):
    image = sitk.ReadImage(image_path[0])
    image = sitk.GetArrayFromImage(image)
    image = image / image.max()
    image_size = image.shape
    image = torch.from_numpy(image.astype(np.float32)).float()
    # image_size=image.GetSize()
    left_mask=lesion_mask.copy()
    right_mask=lesion_mask.copy()
    left_mask[:,:,image_size[1]//2:]=0
    right_mask[:,:,:image_size[1]//2]=0
    patches = []
    if np.sum(left_mask) > 0:
        roi=patch(image, left_mask, image_size, patch_size)
        patches.append(roi)
    if np.sum(right_mask) > 0:
        roi = patch(image, right_mask, image_size, patch_size)
        patches.append(roi)
    return patches
