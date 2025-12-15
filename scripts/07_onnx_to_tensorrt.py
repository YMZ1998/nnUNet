import os
import shutil
import time

import pycuda.driver as cuda
import tensorrt as trt

from scripts.nnunet_model_paths import NNUnetModelPaths


def get_max_workspace_size(fraction: float = 0.5):
    """
    根据当前 GPU 显存动态计算 TensorRT 最大工作空间
    fraction: 使用显存的比例，默认 0.5 (50%)
    """
    handle = cuda.Context.get_device()
    free_mem, total_mem = cuda.mem_get_info()
    workspace_size = int(total_mem * fraction)
    print(f"GPU total memory: {total_mem / (1024 ** 3):.2f} GB, "
          f"Free memory: {free_mem / (1024 ** 3):.2f} GB, "
          f"Using {workspace_size / (1024 ** 3):.2f} GB for TensorRT workspace")
    return workspace_size


def onnx_to_trt(onnx_file: str, trt_file: str, fp16: bool = True, workspace_fraction: float = 0.5):
    start_time = time.time()

    if not os.path.exists(onnx_file):
        raise FileNotFoundError(f"ONNX file not found: {onnx_file}")

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    print("Building TensorRT engine from ONNX...")
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX file")

    config = builder.create_builder_config()
    max_workspace_size = get_max_workspace_size(workspace_fraction)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    config.flags = 0
    if fp16 and builder.platform_has_fast_fp16:
        config.flags |= 1 << int(trt.BuilderFlag.FP16)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # 保存 engine
    with open(trt_file, 'wb') as f:
        f.write(engine)
    print(f"TensorRT engine saved to {trt_file}")

    end_time = time.time()
    print(f"Total conversion/loading time: {end_time - start_time:.2f} seconds")
    return engine


if __name__ == "__main__":
    TASK_ID = 999
    model_config = (
        'nnUNetTrainer__nnUNetPlans__3d_fullres'
        if TASK_ID > 900
        else 'nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres'
    )
    paths = NNUnetModelPaths(task_id=TASK_ID, model_config=model_config)

    onnx_file = paths.onnx_file
    trt_file = paths.trt_file
    print("trt_file:", trt_file)
    print(f"TensorRT version: {trt.__version__}")
    print(f"trtexec --onnx={onnx_file}  --saveEngine={trt_file} --fp16")
    engine = onnx_to_trt(onnx_file, trt_file, fp16=True, workspace_fraction=0.8)

    shutil.copy(trt_file, os.path.join(r"D:\code\dipper.ai\output\release\engine", os.path.basename(trt_file)))

