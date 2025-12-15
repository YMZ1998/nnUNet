"""
TotalSegmentator has a full resolution approach and a faster, lower resolution approach.
The full resolution approach is trained on isotropic 1.5mm CT scans, while the lower resolution
approach is trained on 3mm CT scans.

The lower resolution approach uses a single nnUNet model to predict all 104 classes.
The full resolution approach uses 5 nnUNet models, each predicting 21 classes (total of 104).
"""

import json
import multiprocessing
import os
from pathlib import Path
from typing import List, Tuple, Union

import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm

from nnunetv2.paths import nnUNet_raw
from scripts.convert_nii_direction import convert_LPI_to_RAI
from scripts.dir_process import remove_and_create_dir
from scripts.total_seg_map import class_map_5_parts

SITK_INTERPOLATOR_DICT = {
    'nearest': sitk.sitkNearestNeighbor,
    'linear': sitk.sitkLinear,
    'gaussian': sitk.sitkGaussian,
    'label_gaussian': sitk.sitkLabelGaussian,
    'bspline': sitk.sitkBSpline,
    'hamming_sinc': sitk.sitkHammingWindowedSinc,
    'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
    'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
    'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
}


def resample_image_to_spacing(image, new_spacing, default_value, interpolator='linear'):
    assert interpolator in SITK_INTERPOLATOR_DICT, \
        (f"Interpolator '{interpolator}' not part of SimpleITK. "
         f"Please choose one of the following {list(SITK_INTERPOLATOR_DICT.keys())}.")
    assert image.GetDimension() == len(new_spacing), \
        (f"Input is {image.GetDimension()}-dimensional while "
         f"the new spacing is {len(new_spacing)}-dimensional.")

    interpolator = SITK_INTERPOLATOR_DICT[interpolator]
    spacing = image.GetSpacing()
    size = image.GetSize()
    new_size = [int(round(siz * spac / n_spac)) for siz, spac, n_spac in zip(size, spacing, new_spacing)]

    img_resampled = sitk.Resample(
        image,
        new_size,  # size
        sitk.Transform(),  # transform
        interpolator,  # interpolator
        image.GetOrigin(),  # outputOrigin
        new_spacing,  # outputSpacing
        image.GetDirection(),  # outputDirection
        default_value,  # defaultPixelValue
        image.GetPixelID()  # outputPixelType
    )
    # print(img_resampled.GetDirection())
    # 再转 RAI
    img_RAI = convert_LPI_to_RAI(img_resampled)
    # print(img_RAI.GetDirection())
    return img_RAI


def merge_masks(segmentations_path: str, class_map: dict) -> sitk.Image:
    """Merge the masks of a patient into a single mask. The masks are merged according to the class map.
    When masks overlap, the latter mask will overwrite the former mask in those areas.

    Args:
        segmentations_path (str): Path to the segmentations directory of a patient.
        class_map (dict): A dictionary mapping the class names to their label values.

    Returns:
        sitk.Image: The merged mask.
    """

    # Read all masks and assign them their label value based on `class_map`
    for i, (label_value, label) in enumerate(class_map.items()):
        mask_path = Path(segmentations_path) / f"{label}.nii.gz"
        mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
        mask = mask * label_value

        # The first mask is the base mask, all other masks are added to it
        if i == 0:
            combined_mask = mask
            continue

        # Add the mask to the base mask.
        # When masks overlap, the latter mask will overwrite the former mask in those areas.
        # https://github.com/wasserth/TotalSegmentator/issues/8#issuecomment-1222364214
        try:
            combined_mask = sitk.Maximum(combined_mask, mask)
        except RuntimeError:
            print(f"Failed to add mask {label} for {segmentations_path.parent.name},"
                  " likely due to different physical space. Enforcing the same space and retrying.")
            mask.SetSpacing(combined_mask.GetSpacing())
            mask.SetDirection(combined_mask.GetDirection())
            mask.SetOrigin(combined_mask.GetOrigin())
            combined_mask = sitk.Maximum(combined_mask, mask)
            print("Success!")

    return combined_mask


def process_patient(patient: str,
                    output_dir: str,
                    target_spacing: Union[List, Tuple],
                    class_map: dict,
                    df: pd.DataFrame) -> None:
    """Resample the images and masks to the target spacing, merge the masks, and save them
    to the target directory in the nnUNet format.

    Args:
        patient (str): path to the patient directory.
        output_dir (str): path to the output directory.
        target_spacing (Union[List, Tuple]): spacing to resample the images and masks to.
        class_map (dict): A dictionary mapping the class names to their label values.
        df (pd.DataFrame): The metadata provided in the TotalSegmentator dataset loaded as a DataFrame.
    """
    try:
        # Resample the images and masks to the target spacing, merge the masks, and save them to the target directory
        scan = sitk.ReadImage(str(patient / "ct.nii.gz"))
        scan = resample_image_to_spacing(scan, new_spacing=target_spacing, default_value=-1024, interpolator="linear")

        # Merge the masks according to the class map
        mask = merge_masks(patient / "segmentations", class_map)
        mask = resample_image_to_spacing(mask, new_spacing=target_spacing, default_value=0, interpolator="nearest")
        mask = sitk.Cast(mask, sitk.sitkUInt8)

        # Get the split (train, val, test) of the patient
        split = df.loc[df["image_id"] == patient.name, "split"].values[0]
        # nnUNet's naming is "imagesTr" for train and "imagesTs" for test, there is no val split directory.
        # Instead, it used cross-validation. However, TotalSegmentator has a predefined train, val, and test split,
        # and we achieve that by copying the `splits_final.json` into the nnUNet's preprocessed directory of the dataset.
        train_or_test = "Ts" if split == "test" else "Tr"

        # TotalSegmentator's naming is "sXXXX" (e.g., s0191). Get the last 4 characters to use as the nnUNet case identifier.
        case_identifier = patient.name[-4:]
        scan_output_path = output_dir / f"images{train_or_test}/TotalSegmentator_{case_identifier}_0000.nii.gz"
        mask_output_path = output_dir / f"labels{train_or_test}/TotalSegmentator_{case_identifier}.nii.gz"

        sitk.WriteImage(scan, str(scan_output_path), useCompression=True)
        sitk.WriteImage(mask, str(mask_output_path), useCompression=True)
    except Exception as e:
        # 捕获异常，打印错误并继续
        print(f"[ERROR] Failed to process patient {patient.name}: {e}")


def create_dataset(input_dir: str,
                   output_dir: str,
                   target_spacing: list,
                   class_map: dict,
                   num_cores: int = -1) -> None:
    """Create the dataset for nnUNet v2 by resampling the images and masks to the target spacing,
    merging the masks, and saving them to the target directory in the nnUNet format.
    Support multiprocessing by using the `num_cores` argument.

    Args:
        input_dir (str): TotalSegmentator dataset directory.
        output_dir (str): nnUNet v2 raw dataset directory.
        target_spacing (list): spacing to resample the images and masks to.
        class_map (dict): A dictionary mapping the class names to their label values.
        num_cores (int, optional): Number of cores to use. Defaults to -1, which means all cores.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    remove_and_create_dir(output_dir)

    # Get all patient directories
    patients = [x for x in input_dir.iterdir() if x.is_dir()][:200]

    # Read the metadata provided in the TotalSegmentator dataset
    df = pd.read_csv(input_dir / "meta.csv", delimiter=";")

    # Create the dataset.json file required by nnUNet
    dataset_json = {
        "channel_names": {"0": "CT"},
        # nnUNet v2 requries the the label names to be keys, and the label values to be values, flip them.
        "labels": {v: k for k, v in class_map.items()} | {"background": 0},
        # Equal to the train and val splits combined as nnUNet does cross-validation by default.
        "numTraining": df.loc[df["split"] != "test"].shape[0],
        "file_ending": ".nii.gz"
    }

    # Save the dataset.json file
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4, sort_keys=True)

    # Create the imagesTr, imagesTs, labelsTr, labelsTs directories
    for name in ["imagesTr", "imagesTs", "labelsTr", "labelsTs"]:
        (output_dir / name).mkdir(exist_ok=True, parents=True)

    # Multiprocessing
    if num_cores == -1:
        print("All cores selected.")
        num_cores = os.cpu_count()
    if num_cores > 1:
        print(f"Running in multiprocessing mode with cores: {num_cores}.")
        with multiprocessing.Pool(processes=num_cores) as pool:
            pool.starmap(process_patient,
                         [(patient, output_dir, target_spacing, class_map, df) for patient in patients])
    else:
        print("Running in main process only.")
        for patient in tqdm(patients):
            process_patient(patient, output_dir, target_spacing, class_map, df)


if __name__ == "__main__":
    create_dataset(
        input_dir=r"D:\Data\seg\Totalsegmentator_dataset_v201",
        output_dir=os.path.join(nnUNet_raw, "Dataset006_TotalSegmentator_cardiac"),
        target_spacing=[3, 3, 3],
        class_map=class_map_5_parts["class_map_part_cardiac"],
        num_cores=128
    )
