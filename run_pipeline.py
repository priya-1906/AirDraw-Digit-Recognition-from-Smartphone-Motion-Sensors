"""Run the IMU pipeline using the existing modules (data, model, training, utils)."""

from __future__ import annotations

import copy
import sys

import training


def main():
    # Use the same CLI options defined in training.py
    parser = training.build_arg_parser()
    args = parser.parse_args()

    # If the user explicitly selects a model, run only that model.
    if "--model-type" in sys.argv:
        training.train_and_evaluate(args)
        return

    # Default: run both models sequentially and save outputs for each.
    for model_type in ["1D_CNN_Feature_Extractor", "GRU_temporal_modeling"]:
        run_args = copy.copy(args)
        run_args.model_type = model_type
        training.train_and_evaluate(run_args)


if __name__ == "__main__":
    main()
