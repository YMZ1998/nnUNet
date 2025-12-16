import json
import os
import shutil
from collections import OrderedDict
from collections import defaultdict

import SimpleITK as sitk
import numpy as np
import pymongo
from batchgenerators.utilities.file_and_folder_operations import save_json, save_pickle, maybe_mkdir_p, join

import pyproto.file_list_pb2 as file_list_pb2
import pyproto.ws_pb2 as ws_pb2
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from server.DipperConnection import DipperConnection


# 下载图像或勾画数据
def download_volume(dipp_conn, guid, vtype, number, fn):
    params = {'uid': guid, 'number': number, 'type': vtype}
    req = ws_pb2.WS()
    req.data = str.encode(json.dumps(params))
    req.method = '/common/get_volume'
    req.request_id = 1
    url = '/common/'
    ret = dipp_conn.send_ws(url, req.SerializeToString())
    ans = ws_pb2.WS()
    ans.ParseFromString(ret)
    ff = file_list_pb2.File()
    ff.ParseFromString(ans.data)
    js = json.loads(ff.json_header)
    size = js['size']
    spacing = js['spacing']
    origin = js['origin']
    dt = np.uint8
    if vtype == 'ct':
        dt = np.int16
    vol = np.frombuffer(ff.data, dtype=dt).reshape(size[::-1])
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing(np.array(spacing))
    img.SetOrigin(np.array(origin))
    sitk.WriteImage(img, fn)


# 合并ROI mask
def merge_masks(masks):
    # get first roi
    tmp = masks[0][2]
    sz = tmp.GetSize()
    ps = tmp.GetSpacing()
    og = tmp.GetOrigin()
    dc = tmp.GetDirection()
    # allocate numpy memory
    label = np.zeros([sz[2], sz[1], sz[0]], np.uint8)
    for (idx, name, roi) in masks:
        # print(name)
        data = sitk.GetArrayFromImage(roi)
        label[data > 0] = idx
    label = sitk.GetImageFromArray(label)
    label.SetSpacing(ps)
    label.SetOrigin(og)
    label.SetDirection(dc)
    return label


# 提取皮肤
def extract_body(input_img):
    img = input_img.copy()
    mask_vol = np.zeros(img.shape, dtype=np.uint8)
    boarder_value = -2000
    img[:, :, 0] = boarder_value
    img[:, :, -1] = boarder_value
    img[:, 0, :] = boarder_value
    img[:, -1, :] = boarder_value
    for idx, slc in enumerate(img):
        s = sitk.GetImageFromArray(slc)
        mask = sitk.ConnectedThreshold(s, seedList=[[0, 0]], lower=-20000, upper=-200, connectivity=1)
        mask = sitk.BinaryDilate(mask, [10] * mask.GetDimension())
        mask = 1 - mask
        mask_vol[idx, :] = sitk.GetArrayFromImage(mask)
    # find the largest region as body region
    tmp_vol = sitk.GetImageFromArray(mask_vol)
    conn = sitk.ConnectedComponent(tmp_vol)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(conn)
    labels = [(lbl, stats.GetNumberOfPixels(lbl)) for lbl in stats.GetLabels()]
    labels = sorted(labels, reverse=True, key=lambda l: l[1])
    label = labels[0][0]
    mask_vol = np.zeros(img.shape, dtype=np.uint8)
    conn = sitk.GetArrayFromImage(conn)
    mask_vol[conn == label] = 1
    for idx, mask in enumerate(mask_vol):
        mask = sitk.GetImageFromArray(mask)
        mask = sitk.BinaryDilate(mask, [12] * mask.GetDimension())
        mask = sitk.GetArrayFromImage(mask)
        img[idx][mask == 0] = -2000
        s = sitk.GetImageFromArray(img[idx])
        mask = sitk.ConnectedThreshold(s, seedList=[[0, 0]], lower=-20000, upper=-200)
        mask = 1 - mask
        mask = sitk.BinaryErode(mask, [5] * mask.GetDimension())
        # clear fragments here
        mask = sitk.BinaryDilate(mask, [10] * mask.GetDimension())
        mask = sitk.BinaryFillhole(mask)
        mask = sitk.BinaryErode(mask, [5] * mask.GetDimension())
        mask_vol[idx] = sitk.GetArrayFromImage(mask)
    return mask_vol


def make_data_description(task, roi_list, train_cases, validation_cases, test_cases, from_root):
    out_base = os.path.join(nnUNet_raw, task).replace("\\", "/")
    os.makedirs(out_base, exist_ok=True)
    # 训练数据
    os.makedirs(os.path.join(out_base, "imagesTr").replace("\\", "/"), exist_ok=True)
    os.makedirs(os.path.join(out_base, "labelsTr").replace("\\", "/"), exist_ok=True)
    os.makedirs(os.path.join(out_base, "bodyTr").replace("\\", "/"), exist_ok=True)
    # 测试数据
    os.makedirs(os.path.join(out_base, "imagesTs").replace("\\", "/"), exist_ok=True)
    os.makedirs(os.path.join(out_base, "labelsTs").replace("\\", "/"), exist_ok=True)
    os.makedirs(os.path.join(out_base, "bodyTs").replace("\\", "/"), exist_ok=True)
    # # 验证数据
    os.makedirs(os.path.join(out_base, "imagesVal").replace("\\", "/"), exist_ok=True)
    os.makedirs(os.path.join(out_base, "labelsVal").replace("\\", "/"), exist_ok=True)
    os.makedirs(os.path.join(out_base, "bodyVal").replace("\\", "/"), exist_ok=True)

    # 拷贝数据到nnUNet_raw_data
    for c in train_cases:
        shutil.copy(os.path.join(from_root, 'train', c, "img.nii.gz").replace("\\", "/"),
                    os.path.join(out_base, "imagesTr", c + "_0000.nii.gz").replace("\\", "/"))
        shutil.copy(os.path.join(from_root, 'train', c, "label.nii.gz").replace("\\", "/"),
                    os.path.join(out_base, "labelsTr", c + ".nii.gz").replace("\\", "/"))
        shutil.copy(os.path.join(from_root, 'train', c, "Body.nii.gz").replace("\\", "/"),
                    os.path.join(out_base, "bodyTr", c + ".nii.gz").replace("\\", "/"))
    for c in validation_cases:
        # 验证数据也拷贝到imagesTr目录下，与训练数据放到一起，通过splits_final.pkl文件会进行分割
        shutil.copy(os.path.join(from_root, 'validation', c, "img.nii.gz").replace("\\", "/"),
                    os.path.join(out_base, "imagesTr", c + "_0000.nii.gz").replace("\\", "/"))
        shutil.copy(os.path.join(from_root, 'validation', c, "label.nii.gz").replace("\\", "/"),
                    os.path.join(out_base, "labelsTr", c + ".nii.gz").replace("\\", "/"))
        shutil.copy(os.path.join(from_root, 'validation', c, "Body.nii.gz").replace("\\", "/"),
                    os.path.join(out_base, "bodyTr", c + ".nii.gz").replace("\\", "/"))
    for c in test_cases:
        shutil.copy(os.path.join(from_root, 'test', c, "img.nii.gz").replace("\\", "/"),
                    os.path.join(out_base, "imagesTs", c + "_0000.nii.gz").replace("\\", "/"))
        shutil.copy(os.path.join(from_root, 'test', c, "label.nii.gz").replace("\\", "/"),
                    os.path.join(out_base, "labelsTs", c + ".nii.gz").replace("\\", "/"))
        shutil.copy(os.path.join(from_root, 'test', c, "Body.nii.gz").replace("\\", "/"),
                    os.path.join(out_base, "bodyTs", c + ".nii.gz").replace("\\", "/"))

    # json_dict = {}
    # json_dict['name'] = task
    # json_dict['description'] = ""
    # json_dict['tensorImageSize'] = "4D"
    # json_dict['reference'] = ""
    # json_dict['licence'] = ""
    # json_dict['release'] = "0.0"
    # json_dict['modality'] = {
    #     "00": "CT",
    # }
    # json_dict['labels'] = {
    #     "00": "Background"
    # }
    # for idx, roi in enumerate(roi_list):
    #     json_dict['labels'].update({str("%02d" % (idx + 1)): roi})
    # json_dict['numTraining'] = len(train_cases)
    # json_dict['numTest'] = len(test_cases)
    # json_dict['numValidation'] = len(validation_cases)
    # json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in train_cases]
    # json_dict['test'] = [{'image': "./imagesTs/%s.nii.gz" % i, "label": "./labelsTs/%s.nii.gz" % i} for i in test_cases]
    # json_dict['validation'] = [{'image': "./imagesVal/%s.nii.gz" % i, "label": "./labelsVal/%s.nii.gz" % i} for i in validation_cases]
    # save_json(json_dict, os.path.join(out_base, "dataset.json").replace("\\", "/"))

    labels = {"background": 0}
    for idx, roi in enumerate(roi_list):
        labels[roi] = idx + 1

    dataset_json = {
        'channel_names': {0: "CT"},  # previously this was called 'modality'. I didn't like this so this is
        # channel_names now. Live with it.
        'labels': labels,
        'numTraining': len(train_cases),
        'file_ending': '.nii.gz',
        'name': task,
        'reference': '',
        'release': '',
        'licence': '',
        'description': '',
    }
    save_json(dataset_json, os.path.join(out_base, "dataset.json").replace("\\", "/"))

    # 使用我们划分好的数据集
    splits = []
    splits.append(OrderedDict())
    splits[-1]['train'] = train_cases
    splits[-1]['val'] = validation_cases
    splits[-1]['test'] = test_cases
    maybe_mkdir_p(join(nnUNet_preprocessed, task))
    save_pickle(splits, join(nnUNet_preprocessed, task, "splits_final.pkl").replace("\\", "/"))


def prepare_data(task, roi_list, from_root):
    type_list = ['train', 'validation', 'test']
    cases = defaultdict(list)
    for type in type_list:
        sub_dir = os.path.join(from_root, type).replace("\\", "/")
        cases[type] = []
        if not os.path.exists(sub_dir):
            continue
        filenames = os.listdir(sub_dir)
        for k, case in enumerate(filenames):
            print('{}/{} prepare data: {}'.format(k + 1, len(filenames), case))
            # 合并ROI
            masks = []
            invalid_data = False
            for name in roi_list:
                id = roi_list.index(name) + 1  # get roi id, 0 reserved for background
                # load roi
                fn = os.path.join(sub_dir, case, name + '.nii.gz').replace("\\", "/")
                if not os.path.exists(fn):
                    print("无效数据，缺少ROI", name)
                    invalid_data = True
                    break
                roi = sitk.ReadImage(fn)
                masks.append([id, name, roi])
            if invalid_data:
                continue
            label = merge_masks(masks)
            sitk.WriteImage(label, os.path.join(sub_dir, case, 'label.nii.gz').replace("\\", "/"))

            # 皮肤分割
            filename = os.path.join(sub_dir, case, 'Body.nii.gz').replace("\\", "/")
            if not os.path.exists(filename):
                img = sitk.ReadImage(os.path.join(sub_dir, case, 'img.nii.gz').replace("\\", "/"))
                body = extract_body(sitk.GetArrayFromImage(img))
                body = sitk.GetImageFromArray(body)
                body.CopyInformation(img)
                sitk.WriteImage(body, filename)
            cases[type].append(case)
    print("cases: ", cases)
    make_data_description(task, roi_list, cases['train'], cases['validation'], cases['test'], from_root)


def download_data(image_guid, type, model_guid, dipper_ip, dipper_username, dipper_pwd, data_root):
    # 查找RTSS
    url = 'mongodb://datu_super_root:c74c112dc3130e35e9ac88c90d214555__strong@' + dipper_ip + ":27227/default_db?authSource=datu_data&directConnection=true"
    mongo_connect = pymongo.MongoClient(url, tz_aware=True)
    mongo_db = mongo_connect['datu_data']
    rtss = mongo_db["rtss"].find_one({"ref_image_guid": image_guid}, {"_id": 1, "roi_list": 1, "ref_patient_guid": 1})
    if rtss is None:
        print("ERROR: Invalid RTSS")
        return
    model = mongo_db["ai_model"].find_one({"_id": model_guid}, {"name": 1, "roi_name_list": 1})
    if model is None:
        print("ERROR: Invalid Model")
        return
    if "roi_name_list" not in model or model["roi_name_list"] is None:
        print("ERROR: Invalid Model, empty roi")
        return
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
            print("ERROR: Invalid ROI" + roi_name)
            return
        roi_list.append([roi_name, roi["number"]])

    # 创建文件夹
    task = model["name"]
    sub_dir = os.path.join(data_root, task, type, image_guid)
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


if __name__ == '__main__':
    data_root = r'D:\test\ai\original_data'
    dipper_ip = '192.168.0.71'
    dipper_username = 'admin'
    dipper_pwd = '202cb962ac59075b964b07152d234b70'
    model_guid = '5aafe6b5-ba72-4a49-a49a-7b2ab00f5708'

    url = 'mongodb://datu_super_root:c74c112dc3130e35e9ac88c90d214555__strong@' + dipper_ip + ":27227/default_db?authSource=datu_data&directConnection=true"
    mongo_connect = pymongo.MongoClient(url, tz_aware=True)
    mongo_db = mongo_connect['datu_data']
    ai_model = mongo_db["ai_model"].find_one({"_id": model_guid})
    if ai_model is None:
        print("ERROR: invalid ai_model")
        exit(1)

    # 训练数据集
    ai_dataset = mongo_db["ai_dataset"].find_one({"_id": ai_model['ref_train_guid']})
    if ai_dataset is None:
        print("ERROR: invalid ai_dataset")
        exit(1)
    for k, image_guid in enumerate(ai_dataset['ref_image_guid_list']):
        print('{}/{} download train data: {}'.format(k + 1, len(ai_dataset['ref_image_guid_list']), image_guid))
        type = 'train'
        download_data(image_guid, type, model_guid, dipper_ip, dipper_username, dipper_pwd, data_root)

    # 验证数据集
    ai_dataset = mongo_db["ai_dataset"].find_one({"_id": ai_model['ref_valid_guid']})
    if ai_dataset is None:
        print("ERROR: invalid ai_dataset")
        exit(1)
    for k, image_guid in enumerate(ai_dataset['ref_image_guid_list']):
        print('{}/{} download valid data: {}'.format(k + 1, len(ai_dataset['ref_image_guid_list']), image_guid))
        type = 'validation'
        download_data(image_guid, type, model_guid, dipper_ip, dipper_username, dipper_pwd, data_root)

    mongo_connect.close()

    prepare_data(ai_model["name"], ai_model["roi_name_list"], data_root)
