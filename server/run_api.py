import base64
import json
import os.path
import re
import shutil
import threading
import time

import pymongo
import requests
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from flask import Flask
from flask import request

from server.utils import get_config

# 初始化nnUnet内部数据环境变量
print('enter api!')
base = get_config('DEFAULT', 'data_root')
print("base:", base)
if base is None:
    print('ERROR: invalid root path')
nnUNet_raw = os.path.join(base, "nnUNet_raw").replace("\\", "/")
nnUNet_preprocessed = os.path.join(base, "nnUNet_preprocessed").replace("\\", "/")
nnUNet_results = os.path.join(base, "nnUNet_results").replace("\\", "/")
maybe_mkdir_p(nnUNet_raw)
maybe_mkdir_p(nnUNet_preprocessed)
maybe_mkdir_p(nnUNet_results)
os.environ["nnUNet_raw"] = nnUNet_raw
os.environ["nnUNet_preprocessed"] = nnUNet_preprocessed
os.environ["nnUNet_results"] = nnUNet_results

from DipperConnection import DipperConnection
from download_data import download_volume, prepare_data
from plan_and_preprocess import nnUNet_plan_and_preprocess
from nnunetv2.paths import *
import server.adapter as adapter

app = Flask(__name__)
prefix = "/service/python"


# 清除下载本地数据
@app.route(prefix + "/free_data", methods=["POST"])
def free_data():
    response = {
        "error_code": "success",
        "msg": ""
    }
    data = json.loads(request.data)
    model_name = data['model_name']
    print('free_data: {}'.format(model_name))
    if model_name is None or model_name == '':
        response['msg'] = 'invalid model name'
        return response
    data_path = adapter.get_nnunet_download_folder(model_name)
    print("remove data path:", data_path)
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    return response


# 下载数据到本地
@app.route(prefix + "/download_data", methods=["POST"])
def download_data():
    response = {
        "error_code": "success",
        "msg": ""
    }

    data = json.loads(request.data)
    image_guid = data['image_guid']  # 图像GUID
    type = data['type']  # 数据类型, train, validation, test
    model_guid = data['model_guid']  # 训练模型GUID
    print('begin download_data {}'.format(image_guid))

    # 读取dipper服务器配置及本地文件缓存路径
    dipper_ip = get_config('DEFAULT', 'dipper_database_ip')
    dipper_username = get_config('DEFAULT', 'dipper_username')
    dipper_pwd = get_config('DEFAULT', 'dipper_pwd')
    print("dipper_ip: ", dipper_ip)

    # 查找RTSS
    url = 'mongodb://datu_super_root:c74c112dc3130e35e9ac88c90d214555__strong@' + dipper_ip + ":27227/default_db?authSource=datu_data&directConnection=true"
    mongo_connect = pymongo.MongoClient(url, tz_aware=True)
    mongo_db = mongo_connect['datu_data']
    rtss = mongo_db["rtss"].find_one({"ref_image_guid": image_guid}, {"_id": 1, "roi_list": 1, "ref_patient_guid": 1})
    if rtss is None:
        response["msg"] = "Invalid RTSS"
        return response
    model = mongo_db["ai_model"].find_one({"_id": model_guid}, {"name": 1, "roi_name_list": 1})
    if model is None:
        response["msg"] = "Invalid Model"
        return response
    if "roi_name_list" not in model or model["roi_name_list"] is None:
        response["msg"] = "Invalid Model, empty roi"
        return response
    mongo_connect.close()

    # 匹配ROI number
    roi_list = []
    for roi_name in model["roi_name_list"]:
        roi = None
        for item in rtss["roi_list"]:
            if item["name"] == roi_name:
                roi = item
                break
        if roi is None:
            response["msg"] = "Invalid ROI" + roi_name
            return response
        roi_list.append([roi_name, roi["number"]])

    # 创建文件夹
    model_name = model["name"]
    download_path = adapter.get_nnunet_download_folder(model_name)
    sub_dir = os.path.join(download_path, type, image_guid)
    print("create dir:", sub_dir)
    if os.path.exists(sub_dir):
        shutil.rmtree(sub_dir)
    maybe_mkdir_p(sub_dir)

    # 下载CT图像
    print("download data:", image_guid)
    dipp_conn = DipperConnection(host=dipper_ip, username=dipper_username, password=dipper_pwd)
    fn = os.path.join(sub_dir, 'img.nii.gz')
    print("download ct")
    download_volume(dipp_conn, image_guid, 'ct', -1, fn)
    # 下载ROI
    for roi in roi_list:
        roi_name = roi[0]
        roi_number = roi[1]
        fn = os.path.join(sub_dir, roi_name + '.nii.gz')
        print("download roi", roi_name)
        download_volume(dipp_conn, rtss["_id"], 'rtss', roi_number, fn)
    print("done!")
    return response


# 数据预处理
@app.route(prefix + "/preprocess_data", methods=["POST"])
def preprocess_data():
    response = {
        "error_code": "success",
        "msg": ""
    }

    print("begin process data")
    data = json.loads(request.data)
    model_name = data['model_name']
    roi_list = data['model_roi_list']

    # 使用正则表达式提取 "Dataset" 和 "_" 之间的数字
    match = re.search(r'Dataset(\d+)_', model_name)
    if match:
        task_id = int(match.group(1))  # 提取匹配的数字部分
        print("task_id: ", task_id)  # 输出: 010
    else:
        print("没有找到匹配的数字")
        response["msg"] = "invalid model name"
        return

    # 清除文件
    if os.path.exists(os.path.join(nnUNet_raw, model_name)):
        shutil.rmtree(os.path.join(nnUNet_raw, model_name))
    if os.path.exists(os.path.join(nnUNet_preprocessed, model_name)):
        shutil.rmtree(os.path.join(nnUNet_preprocessed, model_name))
    if os.path.exists(os.path.join(nnUNet_results, model_name)):
        shutil.rmtree(os.path.join(nnUNet_results, model_name))

    download_path = adapter.get_nnunet_download_folder(model_name)
    prepare_data(model_name, roi_list, download_path)

    # 数据预处理及数据分析，验证数据一致性
    overwrite_target_spacing = None
    configurations = ['3d_fullres']
    num_preprocess = [4]
    task_ids = [task_id]
    nnUNet_plan_and_preprocess(task_ids, configurations, num_preprocess, overwrite_target_spacing)

    print("done!")
    return response


# 开始训练
@app.route(prefix + "/begin_train", methods=["POST"])
def begin_train():
    print("begin train")
    data = json.loads(request.data)
    task = data['model_name']
    # run train
    thread_proc = threading.Thread(target=adapter.run_train, args=[task, False], daemon=True)
    thread_proc.start()
    thread_proc.join()

    response = {
        "error_code": "success",
        "msg": ""
    }
    print("call done!")
    return response


# 结束训练
@app.route(prefix + "/stop_train", methods=["POST"])
def stop_train():
    print('stop train')
    # 结束训练
    adapter.stop_train()
    response = {
        "error_code": "success",
        "msg": ""
    }
    return response


# 继续训练
@app.route(prefix + "/resume_train", methods=["POST"])
def resume_train():
    print('resume train')
    # 恢复训练
    data = json.loads(request.data)
    task = data['model_name']

    # run train
    thread_proc = threading.Thread(target=adapter.run_train, args=[task, True], daemon=True)
    thread_proc.start()
    thread_proc.join()

    response = {
        "error_code": "success",
        "msg": ""
    }
    return response


# 获取训练进度
@app.route(prefix + "/update_train_progress", methods=["POST"])
def update_train_progress():
    print('update_train_progress')
    # 更新训练进度
    data = json.loads(request.data)
    model_name = data['model_name']
    filename = adapter.find_progress(model_name)
    with open(filename, 'rb') as file:
        file_content = file.read()
        onnx_base64 = base64.b64encode(file_content).decode('utf-8')
    # return send_file(filename, mimetype=None, as_attachment=True, download_name='progress.png')
    response = {
        "error_code": "success",
        "data": onnx_base64
    }
    return response


# 下载训练结果到dipper.ai服务器
@app.route(prefix + "/download_train_result", methods=["POST"])
def download_train_result():
    print("begin download train result")
    data = json.loads(request.data)
    model_name = data['model_name']

    # 1.转换onnx
    onnx_filename = adapter.convert_checkpoint_onnx(model_name)

    # load roi_list from database
    dipper_ip = get_config('DEFAULT', 'dipper_database_ip')
    dipper_ai_ip = get_config('DEFAULT', 'dipper_ai_ip')
    url = 'mongodb://datu_super_root:c74c112dc3130e35e9ac88c90d214555__strong@' + dipper_ip + ":27227/default_db?authSource=datu_data&directConnection=true"
    mongo_connect = pymongo.MongoClient(url, tz_aware=True)
    mongo_db = mongo_connect['datu_data']
    model = mongo_db["ai_model"].find_one({"name": model_name})
    if model is None:
        print('invalid model')
        return
    if "roi_name_list" not in model or model["roi_name_list"] is None:
        print('Invalid Model, empty roi')
        return
    mongo_connect.close()
    roi_name_list = model["roi_name_list"]
    # 2. 生成tensor-rt config文件
    config_filename = adapter.convert_tensorrt_config(model_name)
    print('config_filename', config_filename)
    print('onnx_filename', onnx_filename)

    # 3. 调用dipper.ai接口将onnx转换为tensorrt
    # 因为在转换tensorrt时，需要在dipper服务电脑上运行(当前服务器上的显卡)，所以将该服务写在dipper.ai中
    with open(onnx_filename, 'rb') as file:
        file_content = file.read()
        onnx_base64 = base64.b64encode(file_content).decode('utf-8')
    with open(config_filename, 'rb') as file:
        file_content = file.read()
        config_base64 = base64.b64encode(file_content).decode('utf-8')
    with requests.Session() as session:
        req = {
            'checkpoint_onnx': onnx_base64,
            'tensorrt_config': config_base64,
        }
        url = 'http://{}:8899/service/ai/convert_tensorrt'.format(dipper_ai_ip)
        print(url)
        res = session.post(url=url, json=req,
                           headers={'content_type': 'application/json'})
        print(url, 'res.status_code', res.status_code)
        if res.status_code != 200:  # 请求成功
            print(f"convert_tensorrt请求失败，状态码: {res.status_code}")
            return
    response = {
        "error_code": "success",
        "msg": ""
    }
    print('response', response)
    print("done!")
    return response


def register_butler(port):
    dipper_ip = get_config('DEFAULT', 'dipper_database_ip')
    butler_addr = 'http://' + dipper_ip + ':9999/register'
    check = {
        "http": '/service/python/health',
        "method": "GET",
        "interval": '10s',
        "timeout": '30s'
    }
    request = {
        "name": 'dt_ai',
        "port": port,
        "tags": ['/service/python'],
        "check": check
    }
    print('butler_addr', butler_addr)

    max_retries = 1000
    retries = 0

    while retries < max_retries:
        try:
            response = requests.post(butler_addr, data=json.dumps(request))
            if response.ok:
                print(response)
                break
            else:
                print('POST请求失败:', response.status_code)
                break
        except requests.exceptions.RequestException as e:
            time.sleep(3)
            print(f'请求异常: {e}')

        retries += 1


@app.route(prefix + "/health", methods=["GET"])
def health():
    return "OK"


if __name__ == '__main__':
    print('enter main')
    register_butler(5000)
    print("Start AI server!")
    from waitress import serve

    serve(app, host='0.0.0.0', port=5000)
