from pathlib import Path
import json
import random
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.backends import cudnn
import torchvision
import SimpleITK as sitk
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader

from opts import parse_opts
from model import (generate_model, make_data_parallel)

class CTSInferenceDataset(Dataset):
    def __init__(self, image_path, input_D=30, input_H=224, input_W=224):
        self.image_path = image_path
        self.input_D = input_D
        self.input_H = input_H
        self.input_W = input_W

    def __len__(self):
        return 1  # 只处理单张图片

    def __getitem__(self, idx):
        """处理单个图像文件或DICOM文件夹"""
        if os.path.isfile(self.image_path) and self.image_path.endswith('.nii.gz'):
            # 处理nii.gz文件
            nii_data = sitk.ReadImage(self.image_path)
            nii_array = sitk.GetArrayFromImage(nii_data)
        
        elif os.path.isdir(self.image_path):
            # 处理DICOM文件夹
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(self.image_path)
            if not dicom_names:
                raise ValueError(f"No DICOM files found in {self.image_path}")
            reader.SetFileNames(dicom_names)
            dicom_image = reader.Execute()
            nii_array = sitk.GetArrayFromImage(dicom_image)
        
        else:
            raise ValueError("Input path must be either a .nii.gz file or a directory containing DICOM files")

        # 确保数组是3D的
        if len(nii_array.shape) > 3:
            nii_array = nii_array[:, :, :, 0]  # 如果是4D数据，取第一个通道
        
        # 调整大小
        [depth, height, width] = nii_array.shape
        scale = [self.input_D * 1.0 / depth, self.input_H * 1.0 / height, self.input_W * 1.0 / width]
        nii_array = ndimage.interpolation.zoom(nii_array, scale, order=0)
        
        # 添加通道维度并归一化
        nii_array = np.expand_dims(nii_array, axis=0)
        mean = nii_array.mean()
        std = nii_array.std()
        nii_array = (nii_array - mean) / std
        
        # 转换为tensor
        nii_array = torch.from_numpy(nii_array).float()
        
        return nii_array, self.image_path

def get_inference_utils(image_path):
    """创建用于推理的数据加载器"""
    inference_data = CTSInferenceDataset(image_path)
    inference_loader = DataLoader(
        inference_data,
        batch_size=1,  # 单张图片推理，batch size设为1
        shuffle=False,
        num_workers=1,
        pin_memory=True)
    
    return inference_loader

def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)

def get_opt(args=None):
    opt = parse_opts(args)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    
    if opt.dilation_flag:
        opt.dilation = [1,1,2,4]
        opt.stride = [2,1,1]
    else:
        opt.dilation = [1,1,1,1]
        opt.stride = [2,2,2]

    # print(opt)
    return opt

def resume_model(resume_path, arch, model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    assert arch == checkpoint['arch']

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    return model

def inference(data_loader, model, device):
    """执行推理"""
    model.eval()
    
    with torch.no_grad():
        # 获取第一个batch
        for inputs, image_path in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            
            return outputs.cpu().numpy()[0], image_path[0]

def main_worker(opt, model):
    # 获取数据加载器
    inference_loader = get_inference_utils(opt.image_path)
    # 进行推理
    prediction, image_path = inference(inference_loader, model, opt.device)
    print(f"Image path: {image_path}")
    print(f"Prediction probabilities: {prediction}")
    predicted_class = np.argmax(prediction)
    # 修改输出逻辑
    class_name = "normal" if predicted_class == 1 else "abnormal"
    print(f"Predicted class: {class_name} (class {predicted_class})")
    return prediction, class_name

def get_model(args=None):
    opt = get_opt(args)
    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    model = generate_model(opt)
    if opt.resume_path is not None:
        model = resume_model(opt.resume_path, opt.arch, model)
    model = make_data_parallel(model, opt.distributed, opt.device)
    return model

def helper(model, args=None):
    opt = get_opt(args)
    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    if not opt.no_cuda:
        cudnn.benchmark = True
    if opt.accimage:
        torchvision.set_image_backend('accimage')
    # 运行推理
    return main_worker(opt, model)