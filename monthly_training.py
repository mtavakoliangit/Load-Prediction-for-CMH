# -*- coding: utf-8 -*-
"""main.ipynb"""

from src.training import Training

def monthly_training():
    trained_model = Training()
    trained_model.dev_and_evaluate_models_for_categories()

if __name__ == "__main__":
    monthly_training()
