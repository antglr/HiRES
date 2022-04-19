import matplotlib
matplotlib.use("Agg")
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


def forecast(model, X, past_history, n_variables):
    forecast = []
    X2 = X.reshape(X.shape[0], past_history, n_variables)
    for k in range(len(X)):        
        if (len(forecast)<past_history):
            if (len(forecast)==0):
                n_value_neede = past_history
                x_cam_first =  X2[k, :n_value_neede, 0]
                x_cam = x_cam_first #Check concatenation axis
            else:
                x_cam_last = np.squeeze(np.array(forecast[-len(forecast):]),1)
                n_value_neede = past_history - len(forecast)
                x_cam_first =  X2[k, :n_value_neede, 0]
                x_cam_last = np.array(x_cam_last).reshape(-1,)
                x_cam = np.concatenate((x_cam_first,x_cam_last),axis=0) #Check concatenation axis
            x_external = X2[k,:,1:]
            new_X = np.concatenate((np.expand_dims(x_cam,1),x_external),axis=1)
            new_X = new_X.reshape(1, new_X.shape[0] * new_X.shape[1])
            local_forecast = model.predict(new_X)
            if len(local_forecast.shape) > 1:
                local_forecast = np.squeeze(local_forecast, 1)
            forecast.append(local_forecast)
        else:
            x_cam = np.squeeze(np.array(forecast[-past_history:]),1)
            x_external = X2[k,:,1:]
            new_X = np.concatenate((np.expand_dims(x_cam,1),x_external),axis=1)
            new_X = new_X.reshape(1, new_X.shape[0] * new_X.shape[1])
            local_forecast = model.predict(new_X)
            if len(local_forecast.shape) > 1:
                local_forecast = np.squeeze(local_forecast, 1)
            forecast.append(local_forecast)
    return np.array(forecast).squeeze(1)

def train_ml(model_name, iter_params, x_train, y_train, x_test, norm_params, index_target_series, normalization_method):
    model = create_model_ml(model_name, iter_params)
    past_history =  x_train.shape[1]
    # Train
    x_train2 = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    print('x_train: {} -> {}'.format(x_train.shape, x_train2.shape))
    training_time_0 = time.time()
    model.fit(x_train2, y_train)
    training_time = time.time() - training_time_0
    
    train_forecast = model.predict(x_train2)
    # train_forecast_stream = forecast(model, x_train, past_history, 3)

    for i in range(train_forecast.shape[0]):
        nparams = norm_params[index_target_series]    # The index_target_series has to match the index of cam variable
        train_forecast[i] = denormalize(
            train_forecast[i], nparams, method=normalization_method,
        )
        # train_forecast_stream[i] = denormalize(
        #     train_forecast_stream[i], nparams, method=normalization_method,
        # )
        

    # Test
    x_test2 = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
    print('x_test: {} -> {}'.format(x_test.shape, x_test2.shape))
    test_time_0 = time.time()
    test_forecast = model.predict(x_test2)
    test_forecast_stream =  forecast(model, x_test, past_history, 3)
    test_time = time.time() - test_time_0

    for i in range(test_forecast.shape[0]):
        nparams = norm_params[index_target_series]
        test_forecast[i] = denormalize(
            test_forecast[i], nparams, method=normalization_method,
        )
        test_forecast_stream[i] = denormalize(
            test_forecast_stream[i], nparams, method=normalization_method,
        )

    return train_forecast, test_forecast, test_forecast_stream, training_time, test_time



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
            
            past_history = x_test.shape[1]
            forecast_horizon = y_test.shape[1]

            parameters_models = parameters['model_params'][model_name]

            list_parameters_models = []
            for parameter_field in parameters_models.keys():
                list_parameters_models.append(parameters_models[parameter_field])

            model_id = 0
            for iter_params in itertools.product(*list_parameters_models):
                train_forecast, test_forecast, test_forecast_stream, training_time, test_time = train_ml(
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
                    train_metrics = evaluate(np.expand_dims(y_train_denorm.squeeze(1), 0), np.expand_dims(train_forecast, 0), metrics)
                    train_metrics = {k + "_train": v for k,v in train_metrics.items()}
                    test_metrics = evaluate(np.expand_dims(y_test_denorm.squeeze(1), 0), np.expand_dims(test_forecast, 0), metrics)
                    test_metrics = {k + "_test": v for k,v in test_metrics.items()}
                    test_stream_metrics = evaluate(np.expand_dims(y_test_denorm.squeeze(1), 0), np.expand_dims(test_forecast_stream, 0), metrics)
                    test_stream_metrics = {k + "_test_stream": v for k,v in test_stream_metrics.items()}
                    

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
                        **test_stream_metrics,
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
    metrics = [ 'mae', 'rmse', 'std', 'std_diff']
    parameters_file = './parameters.json'
    main_ml(models_ml, parameters_file, metrics, 'results')
