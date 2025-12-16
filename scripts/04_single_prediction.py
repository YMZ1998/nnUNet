import os
import time
from os.path import join

import torch

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_results, nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


def nnUNet_predict(model_dir, checkpoint_name, input_file, output_file):
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device('cuda', 1),
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=True
    )

    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=model_dir,
        use_folds="all",
        checkpoint_name=checkpoint_name,
    )

    img, props = SimpleITKIO().read_images([input_file])
    print(f"Image properties: {props}")

    pred = predictor.predict_single_npy_array(img, props, None, None, False)
    print(pred.shape)

    SimpleITKIO().write_seg(pred, output_file, props)
    print(f"Prediction saved to: {output_file}")


if __name__ == "__main__":
    start_time = time.time()
    TASK_ID = 999
    checkpoint_name = 'checkpoint_best.pth'
    model_config = 'nnUNetTrainer__nnUNetPlans__3d_fullres'
    # model_config = 'nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres'
    dataset_name = maybe_convert_to_dataset_name(TASK_ID)

    model_dir = join(nnUNet_results, dataset_name, model_config)

    # images_tr_dir = join(nnUNet_raw, dataset_name, 'imagesTr')
    # input_file = join(images_tr_dir, os.listdir(images_tr_dir)[0])
    # print(f"Input file: {input_file}")
    # output_file = join(model_dir, os.path.basename(input_file).replace('.nii.gz', '_seg.nii.gz'))

    input_file = r'D:\Data\Test\case10\Thorax.nii.gz'
    # input_file=r'D:\Data\Test\case9\unet3d_pre_vol.nii.gz'
    output_file = input_file.replace('.nii.gz', '_seg.nii.gz')
    print(f"Output file: {output_file}")

    nnUNet_predict(model_dir, checkpoint_name, input_file, output_file)

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
