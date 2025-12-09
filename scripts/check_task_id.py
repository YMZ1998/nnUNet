import os
import re

from nnunetv2.paths import nnUNet_raw


def get_all_task_ids(nnUNet_raw_path):
    task_ids = []
    for folder_name in os.listdir(nnUNet_raw_path):
        # 匹配 TaskXXX_... 格式
        match = re.match(r'Task(\d+)_.*', folder_name)
        if match:
            task_ids.append(int(match.group(1)))
    task_ids.sort()
    return task_ids


if __name__ == "__main__":
    all_task_ids = get_all_task_ids(nnUNet_raw)
    print("Found task IDs:", all_task_ids)
