import os

# 手动设置 nnU-Net 环境变量
os.environ['nnUNet_raw'] = r'D:\AI-data\nnUNet_raw'
os.environ['nnUNet_preprocessed'] = r'D:\AI-data\nnUNet_preprocessed'
os.environ['nnUNet_results'] = r'D:\AI-data\nnUNet_results'

nnUNet_raw = os.environ.get('nnUNet_raw')
nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')
nnUNet_results = os.environ.get('nnUNet_results')

print(nnUNet_raw, nnUNet_preprocessed, nnUNet_results)
