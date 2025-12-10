import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def summarize_section_dice(section):
    summary = {}
    dice_values = []

    for label, metrics in section.items():
        if "Dice" in metrics and metrics["Dice"] is not None:
            dice = float(metrics["Dice"])
            summary[label] = {"Dice": dice}
            dice_values.append(dice)
        else:
            summary[label] = {"Dice": None}

    # 计算整体平均 Dice
    if dice_values:
        summary["overall_mean"] = {"Dice": float(np.mean(dice_values))}
    else:
        summary["overall_mean"] = {"Dice": None}

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
            "Dice_max": float(np.nanmax(arr)),
            "Dice_values": arr.tolist()  # 保留原始值用于可视化
        }

    return summary


def visualize_dice(summary_metric_per_case):
    labels = list(summary_metric_per_case.keys())

    dice_data = [summary_metric_per_case[label]["Dice_values"] for label in labels]

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=dice_data)
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45)
    plt.ylabel("Dice")
    plt.title("Dice")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def analyze_metrics(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    print("\n===== 🔍 FOREGROUND MEAN → Dice =====")
    print(json.dumps(
        summarize_section_dice({"foreground_mean": data["foreground_mean"]}),
    ))

    print("\n===== 🔍 MEAN（每类） → Dice =====")
    mean_summary = summarize_section_dice(data["mean"])
    print(json.dumps( mean_summary))
    print(mean_summary["overall_mean"])

    if "metric_per_case" in data:
        print("\n===== 📊 METRIC PER CASE（跨 case） → Dice =====")
        summary_metric_per_case = summarize_metric_per_case_dice(data["metric_per_case"])

        visualize_dice(summary_metric_per_case)
    else:
        print("\n metric_per_case 不存在")


if __name__ == "__main__":
    analyze_metrics(
        r"D:\AI-data\nnUNet_results\Dataset001_Heart\nnUNetTrainer__nnUNetPlans__3d_fullres\validation\summary.json"
    )
