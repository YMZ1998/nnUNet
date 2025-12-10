from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess


def nnUNet_plan_and_preprocess(task_ids, configurations=['2d', '3d_fullres', '3d_lowres'], num_preprocess=[8, 4, 8],
                               overwrite_target_spacing=None):
    # fingerprint extraction
    print("Fingerprint extraction...")
    extract_fingerprints(task_ids)

    # experiment planning
    print('Experiment planning...')
    plan_experiments(task_ids, overwrite_target_spacing=overwrite_target_spacing)

    # preprocessing
    print('Preprocessing...')
    preprocess(task_ids, configurations=configurations, num_processes=num_preprocess)


if __name__ == "__main__":
    # configurations = ['2d', '3d_fullres', '3d_lowres']
    # num_preprocess = [8, 4, 8]
    overwrite_target_spacing = None
    configurations = ['3d_fullres']
    num_preprocess = [4]

    task_ids = [2]

    nnUNet_plan_and_preprocess(task_ids, configurations, num_preprocess, overwrite_target_spacing)
