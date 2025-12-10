import os
import shutil
from os.path import join

import torch

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_results, nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


def predict_nnunet(model_dir, fold, checkpoint, input_paths, output_paths, gpu_id=0):
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device('cuda', gpu_id),
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=True
    )

    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=model_dir,
        use_folds=fold,
        checkpoint_name=checkpoint
    )

    io = SimpleITKIO()
    for input_path, output_path in zip(input_paths, output_paths):
        img, props = io.read_images([input_path])
        pred = predictor.predict_single_npy_array(img, props, None, None, False)
        io.write_seg(pred, output_path, props)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    TASK_ID = 1
    checkpoint = 'checkpoint_best.pth'
    fold = '1'
    dataset_name = maybe_convert_to_dataset_name(TASK_ID)

    model_dir = join(nnUNet_results, dataset_name, 'nnUNetTrainer__nnUNetPlans__3d_fullres')
    # images_dir = join(nnUNet_raw, dataset_name, 'imagesTr')
    images_dir = join(nnUNet_raw, maybe_convert_to_dataset_name(2), 'imagesTr')
    input_files = [join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.nii', '.nii.gz'))]

    output_dir = join(model_dir, 'validation')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_files = [join(output_dir, os.path.basename(f).replace('_0000.nii.gz', '.nii.gz')) for f in input_files]

    predict_nnunet(model_dir, fold, checkpoint, input_files, output_files)
