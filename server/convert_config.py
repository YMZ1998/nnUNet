import datetime
import json
import os
import re
import shutil

from scripts.compare_roi_colors import color_hsv_uniform
from scripts.nnunet_model_paths import NNUnetModelPaths


def generate_color(i):
    return [(i * 37) % 256, (i * 59) % 256, (i * 83) % 256]


def convert_config(task_name, plans_file, dataset_json_file):
    match = re.search(r'Dataset\d{3}_(\w+)', task_name)
    if match:
        task_name = match.group(1)
    print(f"Task name: {task_name}")

    # 加载 plans.json
    if not os.path.exists(plans_file):
        raise FileNotFoundError(f"Plans file not found: {plans_file}")
    with open(plans_file, 'r') as file:
        json_plan = json.load(file)

    json_config = json_plan['configurations']['3d_fullres']
    json_fg = json_plan['foreground_intensity_properties_per_channel']['0']

    spacing = json_config['spacing']
    patch_size = json_config['patch_size']

    now = datetime.datetime.now()
    formatted_time = now.strftime('%Y%m%d%H%M%S')
    config_name = f"dipper.ai.contour.target.{task_name.lower()}.unet3d.json"

    cfg = {
        "task": task_name,
        "group": "Target",
        "active": "True",
        "dev_nb": "0",
        "roi_geometry_header": "True",
        "roi_restore_size": "True",
        "contours_smooth_parameter": 4,
        "contours_save_now": "False",
        "model_path": "./engine/" + config_name.replace(".json", ".engine"),
        "mean_intensity": json_fg['mean'],
        "std_intensity": json_fg['std'],
        "lower_bound": json_fg.get('min', json_fg['mean'] - 3 * json_fg['std']),
        "upper_bound": json_fg.get('max', json_fg['mean'] + 3 * json_fg['std']),
        "current_spacing": [spacing[2], spacing[1], spacing[0]],
        "patch_size": [patch_size[2], patch_size[1], patch_size[0]],
        "batch_size": 1,
        "step": 2,
        "translation": "",
        "creator": "Datu Medical AI Service",
        "date": formatted_time,
        "description": "",
        "class_dic": {},
        "expect_roi_list": [],
        "roi_list": []
    }

    # 加载 dataset.json
    if not os.path.exists(dataset_json_file):
        raise FileNotFoundError(f"Dataset JSON not found: {dataset_json_file}")
    with open(dataset_json_file, 'r') as f:
        dataset_json = json.load(f)

    # labels: name → id, 反转为 id → name, 跳过背景(0)
    labels_dict = dataset_json['labels']
    roi_name_list = [(v, k) for k, v in labels_dict.items() if v != 0]
    roi_name_list.sort(key=lambda x: x[0])  # 按 ID 排序

    # 自动生成 class_dic 和 roi_list
    for label_id, roi_name in roi_name_list:
        # color = generate_color(label_id)  # 根据 label_id 生成颜色
        color = color_hsv_uniform(label_id, len(roi_name_list))  # 根据 label_id 生成颜色
        cfg['class_dic'][str(label_id)] = roi_name
        cfg['expect_roi_list'].append(roi_name)
        cfg['roi_list'].append({
            "roi_name": roi_name,
            "color": color,
            "type": "TARGET",
            "translation": roi_name.replace("_", " ").title()
        })

    out_filename = os.path.join(os.path.dirname(dataset_json_file), config_name)
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    with open(out_filename, 'w') as json_file:
        json.dump(cfg, json_file, indent=4)
    print(f"Config saved to {out_filename}")
    return out_filename


if __name__ == '__main__':
    TASK_ID = 999
    model_config = (
        'nnUNetTrainer__nnUNetPlans__3d_fullres'
        if TASK_ID > 900
        else 'nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres'
    )
    paths = NNUnetModelPaths(task_id=TASK_ID, model_config=model_config)

    convert_config(paths.dataset_name, paths.plans_file, paths.dataset_json_file)
