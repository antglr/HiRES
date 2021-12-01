import json
import itertools
import time
import os
import numpy as np
import pandas as pd

from metrics import evaluate
from preprocessing import read_data, denormalize, windows_preprocessing
from models import create_model_ml
from utils import read_results_file


def train_ml(model_name, iter_params, x_train, y_train, x_test, norm_params, index_target_series, normalization_method):
    model = create_model_ml(model_name, iter_params)
    
    # Train
    x_train2 = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    print('x_train: {} -> {}'.format(x_train.shape, x_train2.shape))
    training_time_0 = time.time()
    model.fit(x_train2, y_train)
    training_time = time.time() - training_time_0
    
    train_forecast = model.predict(x_train2)
    for i in range(train_forecast.shape[0]):
        nparams = norm_params[index_target_series]    # The index_target_series has to match the index of cam variable
        train_forecast[i] = denormalize(
            train_forecast[i], nparams, method=normalization_method,
        )

    # Test
    x_test2 = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
    print('x_test: {} -> {}'.format(x_test.shape, x_test2.shape))
    test_time_0 = time.time()
    test_forecast = model.predict(x_test2)
    test_time = time.time() - test_time_0

    for i in range(test_forecast.shape[0]):
        nparams = norm_params[index_target_series]
        test_forecast[i] = denormalize(
            test_forecast[i], nparams, method=normalization_method,
        )

    return train_forecast, test_forecast, training_time, test_time


def main_ml(models_ml, parameters_file, metrics, results_path):
    for model_name in models_ml:

        with open(parameters_file, "r") as params_file:
            parameters = json.load(params_file)

        for normalization_method, past_history_factor in itertools.product(
                parameters['normalization_method'],
                parameters['past_history_factor']
        ):
            csv_filepath = '{}/results.csv'.format(results_path)
            results = read_results_file(csv_filepath, metrics)

            x_train, y_train, x_test, y_test, y_train_denorm, y_test_denorm, norm_params, index_target_series = read_data(
                normalization_method, past_history_factor
            )    
            
            past_history = x_test.shape[2]
            forecast_horizon = y_test.shape[1]

            parameters_models = parameters['model_params'][model_name]

            list_parameters_models = []
            for parameter_field in parameters_models.keys():
                list_parameters_models.append(parameters_models[parameter_field])

            model_id = 0
            for iter_params in itertools.product(*list_parameters_models):
                train_forecast, test_forecast, training_time, test_time = train_ml(
                    model_name,
                    iter_params,
                    x_train,
                    y_train,
                    x_test,
                    norm_params,
                    index_target_series,
                    normalization_method
                )

                if metrics:
                    train_metrics = evaluate(y_train_denorm, train_forecast, metrics)
                    train_metrics = {k + "_train": v for k,v in train_metrics.items()}
                    test_metrics = evaluate(y_test_denorm, test_forecast, metrics)
                    test_metrics = {k + "_test": v for k,v in test_metrics.items()}

                else:
                    train_metrics = {}
                    test_metrics = {}

                prediction_path = '{}/Norm_{}/{}/{}/{}/'.format(
                    results_path,
                    normalization_method,
                    str(past_history_factor),
                    'ML',
                    model_name,
                )

                if not os.path.exists(prediction_path):
                    os.makedirs(prediction_path)

                np.save(prediction_path + str(model_id) + '_train.npy', train_forecast)
                np.save(prediction_path + str(model_id) + '_test.npy', test_forecast)

                results = results.append(
                    {
                        "MODEL": model_name,
                        "MODEL_INDEX": model_id,
                        "MODEL_DESCRIPTION": str(iter_params),
                        "FORECAST_HORIZON": forecast_horizon,
                        "PAST_HISTORY_FACTOR": past_history_factor,
                        "PAST_HISTORY": past_history,
                        "BATCH_SIZE": '',
                        "EPOCHS": '',
                        "STEPS": '',
                        "OPTIMIZER": '',
                        "LEARNING_RATE": '',
                        "NORMALIZATION": normalization_method,
                        "TEST_TIME": test_time,
                        "TRAINING_TIME": training_time,
                        **train_metrics,
                        **test_metrics,
                        "LOSS": '',
                        "VAL_LOSS": '',
                    },
                    ignore_index=True
                )

                print('END OF EXPERIMENT -> ./results/Norm_{}/{}/{}/{}.npy'.format(
                    normalization_method,
                    past_history_factor,
                    model_name,
                    model_id
                ))
                model_id += 1

                results.to_csv(csv_filepath, sep=";")


if __name__ == '__main__':
    models_ml = ['rf']
    metrics = [ 'mae']
    parameters_file = './parameters.json'
    main_ml(models_ml, parameters_file, metrics, 'results')
