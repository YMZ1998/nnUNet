import os.path
import subprocess

import torch

import nnunetv2
from server.convert_config import convert_config
from server.convert_onnx import convert_and_validate_onnx
from server.run_api import nnUNet_results
from server.utils import get_config

# 网络参数
network_trainer = 'nnUNetTrainerNoMirroring'
plans_identifier = 'nnUNetPlans'
network = '3d_fullres'
fold = 'all'
proc = None


# 停止训练
def stop_train():
    global proc
    if proc is not None:
        poll = proc.poll()
        if poll is None:
            # p.subprocess is alive
            print("停止已有训练任务！")
            proc.kill()
            proc = None
    # if proc is not None and proc.is_alive():
    # 强制释放显存
    os.system('killall python')


# 开始训练,
def run_train(task, continue_training):
    stop_train()
    gpus = torch.cuda.device_count()
    print('gpu count: ' + str(gpus))
    print(task)

    num_epochs = get_config('DEFAULT', 'num_epochs')
    # 执行训练
    cmd = ['python', nnunetv2.__path__[0] + '/run/run_training.py', str(task), str(network), str(fold)]
    cmd += ['-num_epochs', str(num_epochs)]
    cmd += ['-num_gpus', str(1)]
    cmd += ['-tr', 'nnUNetTrainerNoMirroring']
    cmd += ['--c'] if continue_training else []
    print('cmd: ', cmd)
    print("开始训练!")
    global proc
    proc = subprocess.Popen(cmd)
    proc.wait()


# 获取训练结果路径
def get_nnunet_output_folder(task):
    output_folder_name = os.path.join(nnUNet_results, task,
                              network_trainer + "__" + plans_identifier + "__" + network).replace("\\", "/")
    return output_folder_name


# 获取下载路径
def get_nnunet_download_folder(task):
    download_path = nnUNet_results.replace('results', 'download')
    output_folder_name = os.path.join(download_path, task).replace("\\", "/")
    return output_folder_name


# 获取训练进度
def find_progress(task_id):
    output_folder_name = get_nnunet_output_folder(task_id)
    fname = os.path.join(output_folder_name, 'fold_all', "progress.png")
    return fname


# 返回onnx
def get_onnx(task_name):
    model_dir = get_nnunet_output_folder(task_name)
    export_file = os.path.join(model_dir, 'fold_all', 'checkpoint.onnx')
    return export_file


def get_plans_json_file(task_name):
    model_dir = get_nnunet_output_folder(task_name)
    filename = os.path.join(model_dir, 'plans.json')
    return filename


# onnx转换
def convert_checkpoint_onnx(task_name):
    model_dir = get_nnunet_output_folder(task_name)
    export_file = get_onnx(task_name)
    convert_and_validate_onnx(model_dir, export_file)
    return export_file


# 生成tensor-rt config
def convert_tensorrt_config(task_name):
    output_folder_name = get_nnunet_output_folder(task_name)
    plans_file = os.path.join(output_folder_name, 'plans.json')
    dataset_json_file = os.path.join(output_folder_name, 'dataset.json')
    out_filename=convert_config(task_name, plans_file, dataset_json_file)
    return out_filename

