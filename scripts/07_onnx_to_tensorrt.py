import os
import time
import tensorrt as trt
from scripts.nnunet_model_paths import NNUnetModelPaths


def onnx_to_trt(onnx_file: str, trt_file: str, fp16: bool = True, max_workspace_size: int = 4 << 30):
    start_time = time.time()

    if not os.path.exists(onnx_file):
        raise FileNotFoundError(f"ONNX file not found: {onnx_file}")

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
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
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    config.flags = 0
    if fp16 and builder.platform_has_fast_fp16:
        config.flags |= 1 << int(trt.BuilderFlag.FP16)

    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    with open(trt_file, 'wb') as f:
        f.write(engine.serialize())

    end_time = time.time()
    print(f"TensorRT engine saved to {trt_file}")
    print(f"Total conversion time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    TASK_ID = 1
    model_config = 'nnUNetTrainer__nnUNetPlans__3d_fullres'
    paths = NNUnetModelPaths(task_id=TASK_ID, model_config=model_config)
    onnx_file = paths.onnx_file
    trt_file = paths.trt_file

    print(f"TensorRT version: {trt.__version__}")
    onnx_to_trt(onnx_file, trt_file, fp16=True)
