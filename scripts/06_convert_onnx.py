import os
from os.path import join

import numpy as np
import onnx
import onnxruntime
import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def convert_and_validate_onnx(model_dir, fold, export_file):
    # -----------------------------
    # 1. 初始化 nnUNetPredictor
    # -----------------------------
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    predictor.initialize_from_trained_model_folder(
        model_dir,
        use_folds=fold,
        checkpoint_name='checkpoint_best.pth',
    )
    predictor.network.load_state_dict(predictor.list_of_parameters[0])

    # 打印信息
    mean_intensity = predictor.plans_manager.foreground_intensity_properties_per_channel['0']['mean']
    std_intensity = predictor.plans_manager.foreground_intensity_properties_per_channel['0']['std']
    lower_bound = predictor.plans_manager.foreground_intensity_properties_per_channel['0']['percentile_00_5']
    upper_bound = predictor.plans_manager.foreground_intensity_properties_per_channel['0']['percentile_99_5']
    current_spacing = predictor.configuration_manager.spacing
    patch_size = predictor.configuration_manager.patch_size
    labels = predictor.dataset_json['labels']
    nb_of_classes = len(labels) - 1

    print("mean_intensity", mean_intensity)
    print("std_intensity", std_intensity)
    print("lower_bound", lower_bound)
    print("upper_bound", upper_bound)
    print("current_spacing", current_spacing[::-1])
    print("patch_size", patch_size[::-1])
    print("nb_of_classes", nb_of_classes)
    print("labels", labels)

    # -----------------------------
    # 2. 网络准备
    # -----------------------------
    device = torch.device('cuda', 0)
    network = predictor.network.cuda(device)
    network.eval()

    # 修复 InstanceNorm
    for m in network.modules():
        if isinstance(m, (torch.nn.InstanceNorm1d,
                          torch.nn.InstanceNorm2d,
                          torch.nn.InstanceNorm3d)):
            m.track_running_stats = True
            if m.running_mean is None:
                m.running_mean = torch.zeros(m.num_features, device=m.weight.device)
            if m.running_var is None:
                m.running_var = torch.ones(m.num_features, device=m.weight.device)
            m.eval()

    # 创建输入张量 (1, 1, D, H, W)
    input_tensor = torch.randn((1, 1, *patch_size)).cuda(device, non_blocking=True)

    # -----------------------------
    # 3. 导出 ONNX
    # -----------------------------
    torch.onnx.export(
        network,
        input_tensor,
        export_file,
        input_names=['input'],
        output_names=['output'],
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
        training=torch.onnx.TrainingMode.EVAL
    )
    print(f"ONNX model exported to: {export_file}")

    # -----------------------------
    # 4. 验证 ONNX 输出
    # -----------------------------
    torch_output = network(input_tensor)
    try:
        onnx_model = onnx.load(export_file)
        onnx.checker.check_model(onnx_model)
        print("ONNX model check passed.")
    except Exception as e:
        print(f"ONNX model check failed: {e}")
        return

    ort_session = onnxruntime.InferenceSession(export_file, providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_tensor)}
    ort_outputs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(to_numpy(torch_output), ort_outputs[0], rtol=1e-2, atol=1e-4)
    print("ONNXRuntime output matches PyTorch output.")


if __name__ == "__main__":
    TASK_ID = 1
    fold = '1'
    dataset_name = maybe_convert_to_dataset_name(TASK_ID)

    model_config = 'nnUNetTrainer__nnUNetPlans__3d_fullres'
    model_dir = join(nnUNet_results, dataset_name, model_config)
    print("Model directory:", model_dir)

    export_file = os.path.join(model_dir, 'model.onnx')
    convert_and_validate_onnx(model_dir, fold, export_file)
