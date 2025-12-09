import os
import shutil

import SimpleITK as sitk
import numpy as np

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

# ==== 参数配置 =====
INPUT_ROOT = r"D:\Data\seg\open_atlas\test_atlas\data"
OUTPUT_ROOT = r"D:\AI-data\nnUNet_raw"
TASK_NAME = "Dataset001_Heart"
OUTPUT_ROOT = os.path.join(OUTPUT_ROOT, TASK_NAME)

STRUCT_ORDER = ["Heart",
                "A_Aorta", "A_Cflx", "A_Coronary_L", "A_Coronary_R", "A_LAD",
                "A_Pulmonary", "Atrium_L", "Atrium_R", "V_Venacava_S",
                "Ventricle_L", "Ventricle_R"
                ]  # 顺序决定 label ID
# ====================
if os.path.exists(OUTPUT_ROOT):
    print(f"{OUTPUT_ROOT} exists, please delete it first.")
    shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)
os.makedirs(f"{OUTPUT_ROOT}/imagesTr", exist_ok=True)
os.makedirs(f"{OUTPUT_ROOT}/labelsTr", exist_ok=True)

patients = sorted(os.listdir(INPUT_ROOT))[:]

for pid in patients:
    case_dir = os.path.join(INPUT_ROOT, pid)
    img_path = os.path.join(case_dir, "IMAGES", "CT.nii.gz")
    struct_dir = os.path.join(case_dir, "STRUCTURES")

    if not os.path.exists(img_path):
        continue

    print(f"Processing {pid} ...")

    # 读取 CT 图像
    img = sitk.ReadImage(img_path)

    arr_merged = np.zeros(sitk.GetArrayFromImage(img).shape, dtype=np.uint8)
    for idx, sname in enumerate(STRUCT_ORDER, start=1):
        mask_path = os.path.join(struct_dir, f"{sname}.nii.gz")
        if os.path.exists(mask_path):
            m = sitk.ReadImage(mask_path)
            arr_mask = sitk.GetArrayFromImage(m)
            arr_mask = arr_mask > 0
            arr_merged[arr_mask] = idx

    merged = sitk.GetImageFromArray(arr_merged)
    merged.CopyInformation(img)

    # 保存 nnUNet 格式
    out_img = f"{OUTPUT_ROOT}/imagesTr/{pid}_0000.nii.gz"
    out_lab = f"{OUTPUT_ROOT}/labelsTr/{pid}.nii.gz"

    sitk.WriteImage(img, out_img)
    sitk.WriteImage(merged, out_lab)

print("== Conversion done! ==")

labels =  {str(i + 1): name for i, name in enumerate(STRUCT_ORDER)}
labels["background"] = 0  # background

generate_dataset_json(
    output_folder=OUTPUT_ROOT,
    channel_names={"0": "CT"},
    labels=labels,
    num_training_cases=len(patients),
    file_ending=".nii.gz",
    dataset_name=TASK_NAME,
    converted_by="Jin"
)

print("dataset.json created!")
