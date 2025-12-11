import os
from os.path import join

import torch
from monai.apps.nnunet.nnunet_bundle import load_json

import nnunetv2
from nnunetv2.paths import nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def convert_to_ONNX(
    model_dir: str,
    onnx_model_path: str,
    fold: str = "all",
):
    model_path = f"{model_dir}/fold_{fold}/checkpoint_best.pth"
    dataset_json = load_json(join(model_dir, 'dataset.json'))
    plans = load_json(join(model_dir, 'plans.json'))
    plans_manager = PlansManager(plans)

    parameters = []
    use_folds = [fold]
    for i, f in enumerate(use_folds):
        f = int(f) if f != 'all' else f
        checkpoint = torch.load(join(model_dir, f'fold_{f}', model_path),
                                map_location=torch.device('cpu'))
        if i == 0:
            trainer_name = checkpoint['trainer_name']
            configuration_name = checkpoint['init_args']['configuration']
            inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                'inference_allowed_mirroring_axes' in checkpoint.keys() else None

        parameters.append(checkpoint['network_weights'])

    configuration_manager = plans_manager.get_configuration(configuration_name)
    # restore network
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                trainer_name, 'nnunetv2.training.nnUNetTrainer')

    device = torch.device('cuda', 0)
    model = trainer_class.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
        enable_deep_supervision=False
    ).to(device)
    for params in parameters:
        model.load_state_dict(params)
    model.eval()

    # convert to onnx
    patch_size = configuration_manager.patch_size
    input_tensor = torch.randn((1, 1, *patch_size)).to(device)
    print("input_tensor", input_tensor.shape)
    torch.onnx.export(
        model,
        input_tensor,
        onnx_model_path,
        verbose=False,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )


if __name__ == '__main__':
    TASK_ID = 3
    fold = 'all'
    dataset_name = maybe_convert_to_dataset_name(TASK_ID)

    model_config = 'nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres'
    model_dir = join(nnUNet_results, dataset_name, model_config)
    print("Model directory:", model_dir)

    export_file = os.path.join(model_dir, 'model.onnx')
    convert_to_ONNX(model_dir, export_file, fold)
