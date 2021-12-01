import pandas as pd
import itertools
from metrics import METRICS

def read_results_file(csv_filepath, metrics):
    try:
        results = pd.read_csv(csv_filepath, sep=";", index_col=0)
    except IOError:
        metrics_train = [m + "_train" for m in metrics]
        metrics_test = [m + "_test" for m in metrics]
        
        results = pd.DataFrame(
            columns=[
                "MODEL",
                "MODEL_INDEX",
                "MODEL_DESCRIPTION",
                "FORECAST_HORIZON",
                "PAST_HISTORY_FACTOR",
                "PAST_HISTORY",
                "BATCH_SIZE",
                "EPOCHS",
                "STEPS",
                "OPTIMIZER",
                "LEARNING_RATE",
                "NORMALIZATION",
                "TEST_TIME",
                "TRAINING_TIME",
                *metrics_train,
                *metrics_test,
                "LOSS",
                "VAL_LOSS",
            ]
        )
    return results


def check_params(models, results_path, parameters, metrics, csv_filename):
    assert all(
        param in parameters.keys()
        for param in [
            "normalization_method",
            "past_history_factor",
            "batch_size",
            "epochs",
            "max_steps_per_epoch",
            "learning_rate",
            "model_params",
        ]
    ), "Some parameters are missing in the parameters file."
    assert all(
        model in parameters["model_params"] for model in models
    ), "models parameter is not well defined."
    assert metrics is None or all(m in METRICS.keys() for m in metrics)
    

def product(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))