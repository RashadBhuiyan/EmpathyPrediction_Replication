import gc
import os
import pandas as pd
import torch
from torchmetrics import SpearmanCorrCoef, F1Score, PearsonCorrCoef, Precision, Recall, MeanSquaredError, Accuracy
import config as cfg

if __name__ == '__main__':
    # config and paths
    config = cfg.load_config()
    DATA_DIR = config["DATA_DIR"]
    EFP_TRAIN = config["EFP_TRAIN"]
    EFP_TEST = config["EFP_TEST"]
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    input_path = os.path.join(parent_dir, "src", "logs", "LLMPrompt_results.csv")

    # read input csv file
    df = pd.read_csv(input_path)

    # split into unique sets based on model name
    model_names = df['model_name'].unique()
    for model_name in model_names:
        print(f"Processing model: {model_name}")
        df_model = df[df['model_name'] == model_name]

        # establish metric baseline
        zero_spearman = SpearmanCorrCoef()
        five_spearman = SpearmanCorrCoef()
        zero_pearson = PearsonCorrCoef()
        five_pearson = PearsonCorrCoef()
        zero_mse = MeanSquaredError()
        five_mse = MeanSquaredError()
        zero_accuracy = Accuracy(task="binary")
        five_accuracy = Accuracy(task="binary")
        zero_F1 = F1Score(task="binary")
        five_F1 = F1Score(task="binary")
        zero_precision = Precision(task="binary")
        five_precision = Precision(task="binary")
        zero_recall = Recall(task="binary")
        five_recall = Recall(task="binary")

        actual = torch.tensor(df['empathy_score'].values / 100.0, dtype=torch.float32)
        zero = torch.tensor(df['zero_shot_result'].values, dtype=torch.float32)
        five = torch.tensor(df['five_shot_result'].values, dtype=torch.float32)

        threshold = 0.5
        actual_bin = (actual >= threshold).int()
        zero_bin = (zero >= threshold).int()
        five_bin = (five >= threshold).int()

        # single update calls
        zero_spearman.update(zero, actual)
        five_spearman.update(five, actual)

        zero_pearson.update(zero, actual)
        five_pearson.update(five, actual)

        zero_mse.update(zero, actual)
        five_mse.update(five, actual)

        zero_accuracy.update(zero_bin, actual_bin)
        five_accuracy.update(five_bin, actual_bin)

        zero_F1.update(zero_bin, actual_bin)
        five_F1.update(five_bin, actual_bin)

        zero_precision.update(zero_bin, actual_bin)
        five_precision.update(five_bin, actual_bin)

        zero_recall.update(zero_bin, actual_bin)
        five_recall.update(five_bin, actual_bin)

        # print results
        print(f"LLM Metrics Results for {model_name}:")
        print("Zero Shot Results:")
        print(f"Spearman Correlation: {zero_spearman.compute()}")
        print(f"Pearson Correlation: {zero_pearson.compute()}")
        print(f"Mean Squared Error: {zero_mse.compute()}")
        print(f"Accuracy: {zero_accuracy.compute()}")
        print(f"F1 Score: {zero_F1.compute()}")
        print(f"Precision: {zero_precision.compute()}")
        print(f"Recall: {zero_recall.compute()}")
        print("\nFive Shot Results:")
        print(f"Spearman Correlation: {five_spearman.compute()}")
        print(f"Pearson Correlation: {five_pearson.compute()}")
        print(f"Mean Squared Error: {five_mse.compute()}")
        print(f"Accuracy: {five_accuracy.compute()}")
        print(f"F1 Score: {five_F1.compute()}")
        print(f"Precision: {five_precision.compute()}")
        print(f"Recall: {five_recall.compute()}")

        # delete variables to free up memory
        del zero_spearman, five_spearman, zero_pearson, five_pearson, zero_mse, five_mse, zero_accuracy, five_accuracy, zero_F1, five_F1, zero_precision, five_precision, zero_recall, five_recall, actual, zero, five, actual_bin, zero_bin, five_bin
        gc.collect()