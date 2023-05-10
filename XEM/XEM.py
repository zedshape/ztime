"""
This script has been modified from the original XEM code to support randomized search
to replicate the experiment settings in the paper
"""

import os
import yaml
import argparse
import numpy as np
from lce import LCEClassifier
from sklearn.preprocessing import LabelEncoder
import sys

# XEM imports
from utils.helpers import create_logger, save_experiment
from utils.data_loading import import_data
from utils.results_export import get_res_mts

from sklearn.metrics import accuracy_score


if __name__ == "__main__":

    # Load configuration
    parser = argparse.ArgumentParser(description="XEM")
    parser.add_argument(
        "-c", "--config", default="configuration/config.yml", help="Configuration File"
    )
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        configuration = yaml.safe_load(config_file)

    max_accuracy = -1
    best_params = []
    log = None

    for window in configuration["window"]:
        # Create experiment folder
        xp_dir = (
            "results/"
            + str(configuration["dataset"])
            + "/window_"
            + str(int(window * 100))
            + "/xp_"
            + str(configuration["experiment_run"])
            + "/"
        )
        save_experiment(xp_dir, args.config)
        log, logclose = create_logger(log_filename=os.path.join(xp_dir, "experiment.log"))

        # Load dataset
        X_train, y_train, X_validation, y_validation, X_test, y_test = import_data(
            configuration["dataset"],
            window,
            xp_dir,
            configuration["validation_split"],
            log,
        )

        for max_depth in configuration["max_depth"]:
            for trees in configuration["trees"]:
                print(window, max_depth, trees)
                # Fit label encoder
                encoder = LabelEncoder()
                encoder.fit(np.concatenate((y_train, y_validation, y_test), axis=0))

                # Fit LCE model - documentation: https://lce.readthedocs.io/en/latest/generated/lce.LCEClassifier.html
                clf = LCEClassifier(
                    n_estimators=trees,
                    max_depth=max_depth,
                    max_samples=configuration["max_samples"],
                    n_jobs=configuration["n_jobs"],
                    random_state=configuration["random_state"],
                )
                clf.fit(X_train[:, 2:], y_train)

                y_pred_train = clf.predict_proba(X_train[:, 2:])
                res_train_mts = get_res_mts(X_train, y_train, y_pred_train, encoder)
                accuracy_train = accuracy_score(
                    res_train_mts.target_num, res_train_mts.pred_num.astype(int)
                )

                accuracy_validation = "-"
                res_validation_mts = []
                if configuration["validation_split"][1] != 0:
                    y_pred_validation = clf.predict_proba(X_validation[:, 2:])
                    res_validation_mts = get_res_mts(
                        X_validation, y_validation, y_pred_validation, encoder
                    )
                    accuracy_validation = accuracy_score(
                        res_validation_mts.target_num, res_validation_mts.pred_num.astype(int)
                        )

                    if accuracy_validation > max_accuracy:
                        max_accuracy = accuracy_validation
                        best_params = [window, trees, max_depth]
                        best_clf = clf
                        
                y_pred_test = clf.predict_proba(X_test[:, 2:])
                res_test_mts = get_res_mts(X_test, y_test, y_pred_test, encoder)
                accuracy_test = accuracy_score(
                    res_test_mts.target_num, res_test_mts.pred_num.astype(int)
                )

                print(accuracy_validation, accuracy_test)
    xp_dir = (
            "results/"
            + str(configuration["dataset"])
            + "/window_"
            + str(int(best_params[0] * 100))
            + "/xp_"
            + str(configuration["experiment_run"])
            + "/"
        )
    
    X_train, y_train, X_validation, y_validation, X_test, y_test = import_data(
            configuration["dataset"],
            best_params[0],
            xp_dir,
            [1, 0],
            log,
        )
    encoder = LabelEncoder()
    encoder.fit(np.concatenate((y_train, y_validation, y_test), axis=0))

    best_clf = LCEClassifier(
                    n_estimators=best_params[1],
                    max_depth=best_params[2],
                    max_samples=configuration["max_samples"],
                    n_jobs=configuration["n_jobs"],
                    random_state=configuration["random_state"],
                )
    best_clf.fit(X_train[:, 2:], y_train)
    
    y_pred_test = best_clf.predict_proba(X_test[:, 2:])
    res_test_mts = get_res_mts(X_test, y_test, y_pred_test, encoder)
    accuracy_test = accuracy_score(
        res_test_mts.target_num, res_test_mts.pred_num.astype(int)
    )

    print(best_params, accuracy_test)