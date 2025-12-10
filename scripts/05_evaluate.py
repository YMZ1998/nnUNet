import json
from os.path import join

from monai.apps.nnunet.nnunet_bundle import load_json

from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder, compute_metrics_on_folder2
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.paths import nnUNet_results, nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


def auto_regions_from_dataset_json(dataset_json_path):
    with open(dataset_json_path, "r") as f:
        data = json.load(f)

    if "regions" in data:
        regions = data["regions"]
        # 组合区域必须转成 tuple
        return [tuple(region) for region in regions.values()]

    else:
        # 退回到 labels 模式
        labels = data["labels"]
        label_ids = [
            int(v) for v in labels.values()
            if int(v) != 0
        ]
        return sorted(label_ids)


if __name__ == '__main__':
    TASK_ID = 1
    dataset_name = maybe_convert_to_dataset_name(TASK_ID)

    model_config = 'nnUNetTrainer__nnUNetPlans__3d_fullres'

    model_dir = join(nnUNet_results, dataset_name, model_config)
    print(model_dir)

    # folder_ref = join(nnUNet_raw, dataset_name, 'labelsTr')
    folder_ref = join(nnUNet_raw, maybe_convert_to_dataset_name(2), 'labelsTr')
    folder_pred = join(model_dir, 'validation')
    dataset_json_file = join(nnUNet_results, dataset_name, model_config, 'dataset.json')
    plans_file = join(nnUNet_results, dataset_name, model_config, 'plans.json')
    output_file = join(folder_pred, 'summary.json')

    image_reader_writer = SimpleITKIO()
    file_ending = '.nii.gz'
    # regions = labels_to_list_of_regions([0, 1])
    regions = auto_regions_from_dataset_json(dataset_json_file)
    print("regions: ", regions)
    ignore_label = None
    num_processes = 1
    # compute_metrics_on_folder(folder_ref, folder_pred, output_file, image_reader_writer, file_ending, regions,
    #                           ignore_label,
    #                           num_processes,True)

    compute_metrics_on_folder2(folder_ref, folder_pred, dataset_json_file, plans_file,
                               output_file,
                               num_processes, False)
    # result_json = load_json(output_file)
    # print(json.dumps(result_json, indent=4, ensure_ascii=False))

    from analyze_metrics import analyze_metrics
    analyze_metrics(output_file)


