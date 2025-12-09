import json

import torch
from batchgenerators.utilities.file_and_folder_operations import load_json, join

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import nnUNetTrainerNoMirroring
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

# python ./nnunetv2/run/run_training.py 1 3d_fullres all -num_gpus 1 -tr nnUNetTrainerNoMirroring
task_id = 1
dataset_json = load_json(join(nnUNet_preprocessed, maybe_convert_to_dataset_name(task_id), 'dataset.json'))
plans = load_json(join(nnUNet_preprocessed, maybe_convert_to_dataset_name(task_id), 'nnUNetPlans.json'))

# now get plans and configuration managers
plans_manager = PlansManager(plans)
configuration_manager = plans_manager.get_configuration('3d_fullres')
# configuration_manager['batch_size'] = 8
print(configuration_manager)

trainer = nnUNetTrainer(
    plans=plans,
    configuration='3d_fullres',
    fold=0,
    dataset_json=dataset_json,
    device=torch.device('cuda')
)

trainer.progress_bar = True  # 强制开启 tqdm（如果版本支持）
# trainer.initialize()

trainer.run_training()
