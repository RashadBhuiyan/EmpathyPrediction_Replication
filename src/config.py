# # # Created by: Rashad Ahmed Bhuiyan
# Creation Date: 2026-03-03
# Last Edit: 2026-03-03

import os
import json

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CONFIG_PATH = os.path.join(parent_dir, "src", "config.json")

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)
    
def save_config(config_dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config_dict, f, indent=4)