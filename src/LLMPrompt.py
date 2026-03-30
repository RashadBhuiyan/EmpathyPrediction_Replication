# Models Evaluated: Claude 3 Opus, GPT-4o, and Claude 4.5 Sonnet
# 0-shot and 5-shot prompting was done
# story only and story+place+why prompting was done w/ [SEP]
# Prompt template:
# "You are a psychologist with expertise in analyzing empathy. You can predict how muchpeople might empathize with each other, based on their past experiences. You will receive two stories, one from person A and the other from person B. Please predict, on a scale from 0 to 1, where 0 is not empathetic and 1 is extremely empathetic, how much person A would empathize with story B. Return just the number, no other text."
import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import anthropic
from openai import OpenAI
import config as cfg

# config and paths
config = cfg.load_config()
DATA_DIR = config["DATA_DIR"]
EFP_TRAIN = config["EFP_TRAIN"]
EFP_TEST = config["EFP_TEST"]
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
efp_train_path = os.path.join(parent_dir, DATA_DIR, EFP_TRAIN)
efp_test_path = os.path.join(parent_dir, DATA_DIR, EFP_TEST)

# load environment variables from local .env file
_ = load_dotenv(find_dotenv())

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if __name__ == '__main__':
    # Get data from CSV files
    efp_train_d = pd.read_csv(efp_train_path)
    efp_test_d = pd.read_csv(efp_test_path)

    # Get top 5 rows of training data for 5-shot prompting
    efp_train_d_top5 = efp_train_d.head(5)

    # Initialize clients for both APIs
    anthropic_client = anthropic.Client(anthropic_api_key)
    openai_client = OpenAI(openai_api_key)
