# main.py
"""
Main script to run the data preprocessing, classification training,
and hurdle regression training.
"""

import os
from data.preprocessing import preprocess_data
from models.train_classification import train_classification
from models.train_hurdle import train_hurdle

if __name__ == "__main__":

    # ===========================
    # Paths
    # ===========================
    RAW_DATA_PATH = "./../data/default_of_credit_card_clients.csv"
    PROCESSED_DATA_PATH = "./../data/df_encoded.csv"
    MODELS_DIR = "./../models/"

    # ===========================
    # 1. Preprocess Data
    # ===========================
    print("Preprocessing data...")
    preprocess_data(input_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH)

    # ===========================
    # 2. Train Classification Models
    # ===========================
    print("\nTraining classification models...")
    train_classification(
        data_path=PROCESSED_DATA_PATH,
        save_dir=MODELS_DIR
    )

    # ===========================
    # 3. Train Hurdle Regression Models
    # ===========================
    print("\nTraining hurdle regression models...")
    train_hurdle(
        data_path=PROCESSED_DATA_PATH,
        save_dir=MODELS_DIR
    )

    print("\nAll models trained and saved successfully!")
