import pandas as pd
import wandb

from multistyleseg.data.fundus.factory import ALL_DATASETS


def get_formatted_runs():
    api = wandb.Api()
    runs = api.runs("liv4d-polytechnique/MultiStyle Fundus Segmentation")
    data = []
    for run in runs:
        row = {}
        for dataset in ALL_DATASETS:
            key1 = f"DiceScore {dataset.value}_split_1"
            key2 = f"DiceScore {dataset.value}_test"
            if key1 in run.summary:
                row[f"Dice {dataset.value}"] = run.summary[key1]
            elif key2 in run.summary:
                row[f"Dice {dataset.value}"] = run.summary[key2]

        row["Model"] = run.name
        data.append(row)
    df = pd.DataFrame(data)
    return df
