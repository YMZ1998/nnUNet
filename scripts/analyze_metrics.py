import json
import numpy as np

def summarize_section_dice(section):
    summary = {}

    for label, metrics in section.items():
        if "Dice" in metrics:
            summary[label] = {"Dice": float(metrics["Dice"])}
        else:
            summary[label] = {"Dice": None}
    return summary


def summarize_metric_per_case_dice(metric_per_case):
    labels = list(metric_per_case[0]["metrics"].keys())
    summary = {}

    for label in labels:
        values = [
            case["metrics"][label]["Dice"]
            for case in metric_per_case
        ]
        arr = np.array(values, dtype=float)

        summary[label] = {
            "Dice_mean": float(np.nanmean(arr)),
            "Dice_std": float(np.nanstd(arr)),
            "Dice_min": float(np.nanmin(arr)),
            "Dice_max": float(np.nanmax(arr))
        }

    return summary


def analyze_metrics(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    print("\n===== 🔍 FOREGROUND MEAN → Dice =====")
    print(json.dumps(
        summarize_section_dice({"foreground_mean": data["foreground_mean"]}),
        indent=4
    ))

    print("\n===== 🔍 MEAN（每类） → Dice =====")
    print(json.dumps(
        summarize_section_dice(data["mean"]),
        indent=4
    ))

    if "metric_per_case" in data:
        print("\n===== 📊 METRIC PER CASE（跨 case） → Dice =====")
        print(json.dumps(
            summarize_metric_per_case_dice(data["metric_per_case"]),
            indent=4
        ))
    else:
        print("\n⚠ metric_per_case 不存在，无法跨 case 统计 Dice")


if __name__ == "__main__":
    analyze_metrics(
        r"D:\AI-data\nnUNet_results\Dataset001_Heart\nnUNetTrainer__nnUNetPlans__3d_fullres\validation\summary.json"
    )
