import os
import argparse
import json
import itertools
import time
import requests
from multiprocessing import Process, Manager
import numpy as np
import pandas as pd
from metrics import METRICS, evaluate
from preprocessing import read_data, denormalize, windows_preprocessing
from main_ml import main_ml
from utils import product,check_params, read_results_file


def _run_experiment(
        gpu_device,
        results_path,
        csv_filepath,
        metrics,
        epochs,
        normalization_method,
        past_history_factor,
        max_steps_per_epoch,
        batch_size,
        learning_rate,
        model_name,
        model_index,
        model_args,
):
    import gc
    import tensorflow as tf
    from models import create_model

    tf.keras.backend.clear_session()

    def select_gpu_device(gpu_number):
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if len(gpus) >= 2 and gpu_number is not None:
            device = gpus[gpu_number]
            tf.config.experimental.set_memory_growth(device, True)
            tf.config.experimental.set_visible_devices(device, "GPU")

    select_gpu_device(gpu_device)

    results = read_results_file(csv_filepath, metrics)

    x_train, y_train, x_test, y_test, y_train_denorm, y_test_denorm, norm_params, index_target_series = read_data(
        normalization_method, past_history_factor
    )    
    
    x_train = tf.convert_to_tensor(x_train)
    y_train = tf.convert_to_tensor(y_train)
    x_test = tf.convert_to_tensor(x_test)
    y_test = tf.convert_to_tensor(y_test)
    y_train_denorm = tf.convert_to_tensor(y_train_denorm)
    y_test_denorm = tf.convert_to_tensor(y_test_denorm)

    past_history = x_test.shape[2]
    forecast_horizon = y_test.shape[1]
    steps_per_epoch = min(
        int(np.ceil(x_train.shape[0] / batch_size)), max_steps_per_epoch,
    )

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    model = create_model(
        model_name,
        x_train.shape,
        output_size=forecast_horizon,
        optimizer=optimizer,
        loss="mae",
        **model_args
    )
    print(model.summary())

    training_time_0 = time.time()
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=(x_test, y_test),
        shuffle=True,
    )
    training_time = time.time() - training_time_0
    
    train_forecast = model(x_train).numpy()
    
    for i in range(train_forecast.shape[0]):
        nparams = norm_params[index_target_series]    # The index_target_series has to match the index of cam variable
        train_forecast[i] = denormalize(
            train_forecast[i], nparams, method=normalization_method,
    )
        
    # Get validation metrics
    test_time_0 = time.time()
    test_forecast = model(x_test).numpy()
    test_time = time.time() - test_time_0

    for i in range(test_forecast.shape[0]):
        nparams = norm_params[index_target_series]
        test_forecast[i] = denormalize(
            test_forecast[i], nparams, method=normalization_method,
    )
        
    if metrics:
        train_metrics = evaluate(np.expand_dims(y_train_denorm.squeeze(1), 0), np.expand_dims(train_forecast, 0), metrics)
        train_metrics = {k + "_train": v for k,v in train_metrics.items()}
        test_metrics = evaluate(np.expand_dims(y_test_denorm.squeeze(1), 0), np.expand_dims(test_forecast, 0), metrics)
        test_metrics = {k + "_test": v for k,v in test_metrics.items()}

    else:
        train_metrics = {}
        test_metrics = {}

    # Save results
    predictions_path = "{}/Norm_{}/{}/DL/{}/{}/{}/{}/".format(
        results_path,
        normalization_method,
        past_history_factor,
        epochs,
        batch_size,
        learning_rate,
        model_name,
    )
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)
        
    np.save(
        predictions_path + str(model_index) + "_train.npy", train_forecast,
    )
    np.save(
        predictions_path + str(model_index) + "_test.npy", test_forecast,
    )
    
    results = results.append(
        {
            "MODEL": model_name,
            "MODEL_INDEX": model_index,
            "MODEL_DESCRIPTION": str(model_args),
            "FORECAST_HORIZON": forecast_horizon,
            "PAST_HISTORY_FACTOR": past_history_factor,
            "PAST_HISTORY": past_history,
            "BATCH_SIZE": batch_size,
            "EPOCHS": epochs,
            "STEPS": steps_per_epoch,
            "OPTIMIZER": "Adam",
            "LEARNING_RATE": learning_rate,
            "NORMALIZATION": normalization_method,
            "TEST_TIME": test_time,
            "TRAINING_TIME": training_time,
            **train_metrics,
            **test_metrics,
            "LOSS": str(history.history["loss"]),
            "VAL_LOSS": str(history.history["val_loss"]),
        },
        ignore_index=True,
    )

    results.to_csv(
        csv_filepath, sep=";",
    )

    print('END OF EXPERIMENT -> ./results/{}/{}/{}/{}/{}/{}/{}'.format(normalization_method,
                                                                           past_history_factor, epochs, learning_rate,
                                                                           batch_size, model_name, model_index))
    gc.collect()
    del model, x_train, x_test, y_train, y_test, test_forecast, y_train_denorm, y_test_denorm


def run_experiment(
        error_dict,
        gpu_device,
        results_path,
        csv_filepath,
        metrics,
        epochs,
        normalization_method,
        past_history_factor,
        max_steps_per_epoch,
        batch_size,
        learning_rate,
        model_name,
        model_index,
        model_args,
):
    try:
        _run_experiment(
            gpu_device,
            results_path,
            csv_filepath,
            metrics,
            epochs,
            normalization_method,
            past_history_factor,
            max_steps_per_epoch,
            batch_size,
            learning_rate,
            model_name,
            model_index,
            model_args,
        )
    except Exception as e:
        error_dict["status"] = -1
        error_dict["message"] = str(e)
    else:
        error_dict["status"] = 1


def main(args):
    models = args.models
    results_path = args.output
    gpu_device = args.gpu
    metrics = args.metrics
    csv_filename = args.csv_filename

    parameters = None
    with open(args.parameters, "r") as params_file:
        parameters = json.load(params_file)

    check_params(models, results_path, parameters, metrics, csv_filename)

    if len(models) == 0:
        models = list(parameters["model_params"].keys())

    if metrics is None:
        metrics = list(METRICS.keys())

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    csv_filepath = results_path + "/{}".format(csv_filename)
    results = read_results_file(csv_filepath, metrics)
    current_index = results.shape[0]
    print("CURRENT INDEX", current_index)

    experiments_index = 0
    num_total_experiments = np.prod(
        [len(parameters[k]) for k in parameters.keys() if k != "model_params"]
        + [
            np.sum(
                [
                    np.prod(
                        [
                            len(parameters["model_params"][m][k])
                            for k in parameters["model_params"][m].keys()
                        ]
                    )
                    for m in models
                ]
            )
        ]
    )
    


    for epochs, normalization_method, past_history_factor in itertools.product(
            parameters["epochs"],
            parameters["normalization_method"],
            parameters["past_history_factor"],
    ):
        for batch_size, learning_rate in itertools.product(
                parameters["batch_size"], parameters["learning_rate"],
        ):
            for model_name in models:
                for model_index, model_args in enumerate(
                        product(**parameters["model_params"][model_name])
                ):
                    experiments_index += 1
                    if experiments_index <= current_index:
                        continue

                    # Run each experiment in a new Process to avoid GPU memory leaks
                    manager = Manager()
                    error_dict = manager.dict()

                    p = Process(
                        target=run_experiment,
                        args=(
                            error_dict,
                            gpu_device,
                            results_path,
                            csv_filepath,
                            metrics,
                            epochs,
                            normalization_method,
                            past_history_factor,
                            parameters["max_steps_per_epoch"][0],
                            batch_size,
                            learning_rate,
                            model_name,
                            model_index,
                            model_args,
                        ),
                    )
                    p.start()
                    p.join()

                    assert error_dict["status"] == 1, error_dict["message"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--models",
        nargs="*",
        default=[],
        help="Models to experiment over (separated by comma)",
    )
    parser.add_argument(
        "-ml",
        "--models_ml",
        nargs="*",
        default=[],
        help="ML Models to experiment over (separated by comma)",
    )
    parser.add_argument(
        "-p", "--parameters", help="Parameters file path",
    )
    parser.add_argument(
        "-o", "--output", default="./results", help="Output path",
    )
    parser.add_argument(
        "-c", "--csv_filename", default="results.csv", help="Output csv filename",
    )
    parser.add_argument("-g", "--gpu", type=int, default=None, help="GPU device")
    parser.add_argument(
        "-s",
        "--metrics",
        nargs="*",
        default=None,
        help="Metrics to use for evaluation. If not define it will use all possible metrics.",
    )
    args = parser.parse_args()

    main(args)
    main_ml(args.models_ml, args.parameters, args.metrics, args.output)
    # obtain_best_results()
    # get_metrics()
