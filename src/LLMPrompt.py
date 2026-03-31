# Models Evaluated: Claude 3 Opus, GPT-4o, and Claude 4.5 Sonnet
# 0-shot and 5-shot prompting was done
# story only and story+place+why prompting was done w/ [SEP]
# Prompt template:
# "You are a psychologist with expertise in analyzing empathy. You can predict how much people might empathize with each other, based on their past experiences. You will receive two stories, one from person A and the other from person B. Please predict, on a scale from 0 to 1, where 0 is not empathetic and 1 is extremely empathetic, how much person A would empathize with story B. Return just the number, no other text."

import os
import pandas as pd
import csv
from dotenv import load_dotenv, find_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from torchmetrics import SpearmanCorrCoef, F1Score, PearsonCorrCoef, Precision, Recall, MeanSquaredError, Accuracy
import config as cfg

# config and paths
config = cfg.load_config()
DATA_DIR = config["DATA_DIR"]
EFP_TRAIN = config["EFP_TRAIN"]
EFP_TEST = config["EFP_TEST"]
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
efp_train_path = os.path.join(parent_dir, DATA_DIR, EFP_TRAIN)
efp_test_path = os.path.join(parent_dir, DATA_DIR, EFP_TEST)
model_list = ["deepseek-ai/DeepSeek-V2-Chat", "meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen2.5-72B-Instruct"]
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
    print(efp_train_d_top5.iloc[0]["story_A"])
    five_shot_prompt = f"""
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
            
            prompt = f"""
            {five_shot_prompt}

            Here is your set of stories:
            Story A: {story_A}
            Story B: {story_B}

            Please predict, on a scale from 0 to 1, where 0 is not empathetic and 1 is extremely empathetic, how much person A would empathize with story B. Return just the number, no other text.
            """

            # Establish connection to model for zero-shot prompting
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
            model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
            zero_message = [{"role": "user", "content": zero_prompt},]
            five_shot_prompt_message = [{"role": "user", "content": prompt},]
            zero_shot_inputs = tokenizer.apply_chat_template(
                zero_message,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            five_shot_inputs = tokenizer.apply_chat_template(
                five_shot_prompt_message,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            zer_shot_outputs = model.generate(**zero_shot_inputs, max_new_tokens=40)
            zero_shot_result = tokenizer.decode(zer_shot_outputs[0][zero_shot_inputs["input_ids"].shape[-1]:])
            five_shot_outputs = model.generate(**five_shot_inputs, max_new_tokens=40)
            five_shot_result = tokenizer.decode(five_shot_outputs[0][five_shot_inputs["input_ids"].shape[-1]:])

            # Calculate torch metrics for zero-shot prompting

            # save results to CSV in logs/LLMPrompt (under file LLMPrompt_results.csv) with columns: model_name, story_A, story_B, empathy_score, predicted_empathy_score
            with open("logs/LLMPrompt/LLMPrompt_results.csv", "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([model_name, story_A, story_B, empathy_score, zero_shot_result, five_shot_result])


            

    
