import os

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

model_folder = "./rl_models"

os.makedirs(model_folder, exist_ok=True)
NEPTUNE_PROJECT = os.environ.get("NEPTUNE_PROJECT")
NEPTUNE_API_KEY = os.environ.get("NEPTUNE_API_KEY")

assert NEPTUNE_PROJECT is not None, "NEPTUNE_PROJECT is not defined"

input_data_path = "./input_data/blue_prints.txt"
if os.path.exists(input_data_path):
    logger.info("data found in local")
    with open(input_data_path, "r") as f:
        input_data = f.readlines()
else:
    logger.info("data not found in local")
    raise FileNotFoundError(
        "data not found in local. Please download it from 'https://adventofcode.com/2022/day/19/input'"
    )
