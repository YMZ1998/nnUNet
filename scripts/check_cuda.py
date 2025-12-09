import torch

# 查看 PyTorch 版本
print("PyTorch version:", torch.__version__)

# 查看 CUDA 是否可用
print("CUDA available:", torch.cuda.is_available())

# 查看当前 CUDA 版本（PyTorch 编译使用的版本）
print("CUDA version (compiled with):", torch.version.cuda)

# 查看当前 cuDNN 版本
print("cuDNN version:", torch.backends.cudnn.version())

# 查看 GPU 数量
print("Number of GPUs:", torch.cuda.device_count())

# 查看每张 GPU 名称
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# 查看当前默认 GPU
print("Current CUDA device index:", torch.cuda.current_device())
