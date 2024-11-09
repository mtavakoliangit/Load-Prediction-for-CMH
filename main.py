# -*- coding: utf-8 -*-
"""main.py"""

from src.training import Training

def main():
    trained_model = Training()
    trained_model.dev_and_evaluate_models_for_categories()

if __name__ == "__main__":
    main()
