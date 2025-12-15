import re
from os.path import join

from nnunetv2.paths import nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


class NNUnetModelPaths:
    def __init__(self, task_id, model_config='nnUNetTrainer__nnUNetPlans__3d_fullres'):
        self.task_id = task_id
        self.dataset_name = maybe_convert_to_dataset_name(task_id)
        self.model_config = model_config

        match = re.search(r'Dataset\d{3}_(\w+)', self.dataset_name)
        if match:
            self.task_name = match.group(1)

        # 模型目录
        self.model_dir = join(nnUNet_results, self.dataset_name, self.model_config)

        # 常用文件
        self.dataset_json_file = join(self.model_dir, 'dataset.json')
        self.plans_file = join(self.model_dir, 'plans.json')
        self.checkpoint_best = join(self.model_dir, 'checkpoint_best.pth')
        self.checkpoint_final = join(self.model_dir, 'checkpoint_final.pth')
        self.onnx_file = join(self.model_dir, 'model.onnx')
        self.trt_file = join(self.model_dir, f"dipper.ai.contour.target.{self.task_name.lower()}.unet3d.engine")

    def __repr__(self):
        return f"<NNUnetModelPaths(task_id={self.task_id}, model_dir={self.model_dir})>"


if __name__ == "__main__":
    paths = NNUnetModelPaths(task_id=1)

    print("Model directory:", paths.model_dir)
    print("Dataset JSON:", paths.dataset_json_file)
    print("Plans JSON:", paths.plans_file)
    print("Best checkpoint:", paths.checkpoint_best)
    print("ONNX file path:", paths.onnx_file)
