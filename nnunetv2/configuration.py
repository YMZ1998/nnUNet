import os

from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

os.environ['nnUNet_def_n_proc'] = '8'
os.environ['nnUNet_n_proc_DA'] = '0'
default_num_processes = 8 if 'nnUNet_def_n_proc' not in os.environ else int(os.environ['nnUNet_def_n_proc'])

ANISO_THRESHOLD = 3  # determines when a sample is considered anisotropic (3 means that the spacing in the low
# resolution axis must be 3x as large as the next largest spacing)

default_n_proc_DA = get_allowed_n_proc_DA()

# print(f"Using {default_num_processes} processes for preprocessing")
# print(f"Using {default_n_proc_DA} processes for data augmentation")
