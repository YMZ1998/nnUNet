import os
from os.path import join

import numpy as np
import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


def convert_onnx(model_dir, fold, export_file):
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,  # 切片步长，用于控制切片的大小。
        use_gaussian=True,  # 是否使用高斯模糊
        use_mirroring=False,  # 是否使用镜像
        perform_everything_on_device=True,  # 是否使用GPU
        device=torch.device('cuda', 0),  # 使用哪个GPU
        verbose=False,  # 是否输出详细信息
        verbose_preprocessing=False,  # 是否输出预处理信息。
        allow_tqdm=True  # 是否使用tqdm
    )

    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        model_dir,
        use_folds=fold,
        # checkpoint_name='checkpoint_final.pth',
        checkpoint_name='checkpoint_best.pth',
    )
    predictor.network.load_state_dict(predictor.list_of_parameters[0])

    current_spacing = predictor.configuration_manager.spacing
    patch_size = predictor.configuration_manager.patch_size
    labels = predictor.dataset_json['labels']
    nb_of_classes = len(labels) - 1

    print("current_spacing", current_spacing[::-1])
    print("patch_size", patch_size[::-1])
    print("nb_of_classes", nb_of_classes)
    print("labels", labels)

    device = torch.device('cuda', 0)
    network = predictor.network.cuda(device)
    network.eval()
    input = torch.randn(tuple(patch_size)).cuda(device, non_blocking=True)
    input = input[np.newaxis, np.newaxis, :]
    torch.onnx.export(network, input, export_file, input_names=['input'], output_names=['output'])


if __name__ == "__main__":
    TASK_ID = 3
    fold = 'all'
    dataset_name = maybe_convert_to_dataset_name(TASK_ID)

    model_config = 'nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres'
    model_dir = join(nnUNet_results, dataset_name, model_config)
    print("Model directory:", model_dir)

    export_file = os.path.join(model_dir, 'model.onnx')
    convert_onnx(model_dir, fold, export_file)
