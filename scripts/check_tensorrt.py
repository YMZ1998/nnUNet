import glob

import tensorrt as trt

if __name__ == "__main__":
    print(glob.glob(r"D:\code\TensorRT-8.6.0.12\python\*"))
    print(r"pip install D:\code\TensorRT-8.6.0.12\python\tensorrt-8.6.0-cp39-none-win_amd64.whl")
    print(f"TensorRT version: {trt.__version__}")
