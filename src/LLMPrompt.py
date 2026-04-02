# Models Evaluated: Claude 3 Opus, GPT-4o, and Claude 4.5 Sonnet
# 0-shot and 5-shot prompting was done
# story only and story+place+why prompting was done w/ [SEP]
# Prompt template:
# "You are a psychologist with expertise in analyzing empathy. You can predict how much people might empathize with each other, based on their past experiences. You will receive two stories, one from person A and the other from person B. Please predict, on a scale from 0 to 1, where 0 is not empathetic and 1 is extremely empathetic, how much person A would empathize with story B. Return just the number, no other text."

import os
import pandas as pd
import csv
from dotenv import load_dotenv, find_dotenv
from torchmetrics import SpearmanCorrCoef, F1Score, PearsonCorrCoef, Precision, Recall, MeanSquaredError, Accuracy
import config as cfg
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import pipeline

# Set environment variables for Hugging Face cache and token
os.environ["HF_HOME"] = "/scratch/rashadb/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/rashadb/huggingface"

# config and paths
config = cfg.load_config()
DATA_DIR = config["DATA_DIR"]
EFP_TRAIN = config["EFP_TRAIN"]
EFP_TEST = config["EFP_TEST"]
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
output_path = os.path.join(parent_dir, "src", "logs", "LLMPrompt_results.csv")
efp_train_path = os.path.join(parent_dir, DATA_DIR, EFP_TRAIN)
efp_test_path = os.path.join(parent_dir, DATA_DIR, EFP_TEST)
model_list = ["Qwen/Qwen2.5-72B-Instruct"] # "deepseek-ai/DeepSeek-V2-Chat", "meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen2.5-72B-Instruct"
initial_prompt = "You are a psychologist with expertise in analyzing empathy. You can predict how much people might empathize with each other, based on their past experiences. You will receive two stories, one from person A and the other from person B."

# load environment variables from local .env file
_ = load_dotenv(find_dotenv())
hf_token = os.getenv("HF_TOKEN_KEY")

if __name__ == '__main__':
    # Get data from CSV files
    efp_train_d = pd.read_csv(efp_train_path)
    efp_test_d = pd.read_csv(efp_test_path)

    # Get top 5 rows of training data and format to five-shot prompt
    efp_train_d_top5 = efp_train_d.head(5)
    few_shot_prompt = f"""
    {initial_prompt} 
    Here are some examples of story pairs and their empathy scores:
    Story A: {efp_train_d_top5.iloc[0]["story_A"]}
    Story B: {efp_train_d_top5.iloc[0]["story_B"]}
    Empathy Score: {efp_train_d_top5.iloc[0]["empathy"]}
    Story A: {efp_train_d_top5.iloc[1]["story_A"]}
    Story B: {efp_train_d_top5.iloc[1]["story_B"]}
    Empathy Score: {efp_train_d_top5.iloc[1]["empathy"]}
    Story A: {efp_train_d_top5.iloc[2]["story_A"]}
    Story B: {efp_train_d_top5.iloc[2]["story_B"]}
    Empathy Score: {efp_train_d_top5.iloc[2]["empathy"]}
    Story A: {efp_train_d_top5.iloc[3]["story_A"]}
    Story B: {efp_train_d_top5.iloc[3]["story_B"]}
    Empathy Score: {efp_train_d_top5.iloc[3]["empathy"]}
    Story A: {efp_train_d_top5.iloc[4]["story_A"]}
    Story B: {efp_train_d_top5.iloc[4]["story_B"]}
    Empathy Score: {efp_train_d_top5.iloc[4]["empathy"]}
    """

    # loop through the set of models
    for model_name in model_list:
        print(f"Testing model: {model_name}")

        # Establish connection to model for initial prompting
        pipe = pipeline(
            "text-generation", 
            model=model_name,
            token=hf_token,
            device_map="auto",
            dtype=torch.float16
        )

        # loop through the test data and print the story pairs and empathy scores
        for index, row in efp_test_d.iterrows():
            story_A = row["story_A"]
            story_B = row["story_B"]
            empathy_score = row["empathy"]

            zero_prompt = f"""
            {initial_prompt}

            Here is your set of stories:
            Story A: {story_A}
            Story B: {story_B}

            Please predict, on a scale from 0 to 1, where 0 is not empathetic and 1 is extremely empathetic, how much person A would empathize with story B. Return just the number, no other text.
            """
            
            five_prompt = f"""
            {few_shot_prompt}

            Here is your set of stories:
            Story A: {story_A}
            Story B: {story_B}

            Please predict, on a scale from 0 to 1, where 0 is not empathetic and 1 is extremely empathetic, how much person A would empathize with story B. Return just the number, no other text.
            """
            
            # Establish connection to model for zero-shot prompting
            zero_message = [{"role": "user", "content": zero_prompt},]
            five_shot_prompt_message = [{"role": "user", "content": five_prompt},]
            zero_output = pipe(zero_message)
            five_shot_output = pipe(five_shot_prompt_message)
            zero_shot_result = zero_output[0]["generated_text"][-1]["content"].strip()
            five_shot_result = five_shot_output[0]["generated_text"][-1]["content"].strip()

            del zero_message, five_shot_prompt_message, zero_output, five_shot_output
            torch.cuda.empty_cache()

            # Calculate torch metrics for zero-shot prompting

            # save results to CSV in logs/LLMPrompt (under file LLMPrompt_results.csv) with columns: model_name, story_A, story_B, empathy_score, predicted_empathy_score
            output_headers = ["model_name", "story_A", "story_B", "empathy_score", "zero_shot_result", "five_shot_result"]
            write_header = not os.path.exists(output_path)
            with open(output_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(output_headers)
                writer.writerow([model_name, story_A, story_B, empathy_score, zero_shot_result, five_shot_result])