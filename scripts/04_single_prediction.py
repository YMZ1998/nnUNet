import os
from os.path import join

import torch
from totalsegmentator.nnunet import nnUNetv2_predict

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_results, nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


def nnUNet_predict(model_dir, checkpoint_name, input_file, output_file):

    predictor = nnUNetPredictor(
        tile_step_size=0.5,  # 切片步长，用于控制切片的大小
        use_gaussian=True,  # 是否使用高斯模糊
        use_mirroring=False,  # 是否使用镜像增强
        perform_everything_on_device=True,  # 是否全程在 GPU 上处理
        device=torch.device('cuda', 0),  # 使用哪块 GPU
        verbose=True,  # 输出详细信息
        verbose_preprocessing=True,  # 输出预处理信息
        allow_tqdm=True  # 是否显示 tqdm
    )

    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=model_dir,
        use_folds=[1],  # 使用哪个 fold
        checkpoint_name=checkpoint_name,
    )

    img, props = SimpleITKIO().read_images([input_file])
    print(f"Image properties: {props}")

    pred = predictor.predict_single_npy_array(img, props, None, None, False)

    SimpleITKIO().write_seg(pred, output_file, props)
    print(f"Prediction saved to: {output_file}")


if __name__ == "__main__":
    TASK_ID = 1
    checkpoint_name = 'checkpoint_best.pth'
    dataset_name = maybe_convert_to_dataset_name(TASK_ID)

    model_dir = join(nnUNet_results, dataset_name, 'nnUNetTrainer__nnUNetPlans__3d_fullres')


    images_tr_dir = join(nnUNet_raw, dataset_name, 'imagesTr')
    input_file = join(images_tr_dir, os.listdir(images_tr_dir)[0])
    print(f"Input file: {input_file}")

    out_dir = join(model_dir, 'predictions')
    os.makedirs(out_dir, exist_ok=True)
    output_file = join(out_dir, os.path.basename(input_file))
    input_file=r'D:\Data\Test\case8\Thorax.nii.gz'
    output_file = input_file.replace('.nii.gz', '_seg.nii.gz')
    print(f"Output file: {output_file}")

    nnUNet_predict(model_dir, checkpoint_name, input_file, output_file)
