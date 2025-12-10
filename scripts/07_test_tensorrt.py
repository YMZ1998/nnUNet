import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # 初始化上下文
import tensorrt as trt

from scripts.nnunet_model_paths import NNUnetModelPaths


def test_trt_engine(trt_file, input_shape=(1, 1, 192, 160, 64)):
    """
    测试 TensorRT engine 是否可用
    Args:
        trt_file: TensorRT engine 文件路径
        input_shape: 输入张量 shape (N, C, D, H, W)
    """
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(TRT_LOGGER)

    # 反序列化 engine
    with open(trt_file, 'rb') as f:
        engine_data = f.read()
    engine = runtime.deserialize_cuda_engine(engine_data)
    if engine is None:
        raise RuntimeError("Failed to load TensorRT engine")

    # 创建执行上下文
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create execution context")

    # 准备输入输出
    n_bindings = engine.num_bindings
    bindings = [None] * n_bindings
    stream = cuda.Stream()

    host_inputs = []
    host_outputs = []
    d_inputs = []
    d_outputs = []

    for i in range(n_bindings):
        size = trt.volume(engine.get_binding_shape(i)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(i))
        # 分配 host 和 device 内存
        if engine.binding_is_input(i):
            host_input = np.random.rand(*input_shape).astype(dtype)
            d_input = cuda.mem_alloc(host_input.nbytes)
            host_inputs.append(host_input)
            d_inputs.append(d_input)
            bindings[i] = int(d_input)
        else:
            host_output = np.empty(size, dtype=dtype)
            d_output = cuda.mem_alloc(host_output.nbytes)
            host_outputs.append(host_output)
            d_outputs.append(d_output)
            bindings[i] = int(d_output)

    # 将输入数据复制到 GPU
    for h, d in zip(host_inputs, d_inputs):
        cuda.memcpy_htod_async(d, h, stream)

    # 推理
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # 将输出从 GPU 拷贝回 host
    for h, d in zip(host_outputs, d_outputs):
        cuda.memcpy_dtoh_async(h, d, stream)

    stream.synchronize()

    print("TensorRT engine test completed successfully.")
    for i, out in enumerate(host_outputs):
        print(f"Output {i} shape: {out.shape}, dtype: {out.dtype}")
    return host_outputs

# -----------------------------
# 示例
# -----------------------------
if __name__ == "__main__":
    TASK_ID = 1
    model_config = 'nnUNetTrainer__nnUNetPlans__3d_fullres'
    paths = NNUnetModelPaths(task_id=TASK_ID, model_config=model_config)

    trt_file = paths.trt_file
    print("trt_file:", trt_file)
    outputs = test_trt_engine(trt_file, input_shape=(1, 1, 128, 128, 128))
