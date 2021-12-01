import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import itertools
import time


def windows_preprocessing(time_series, camera_series, past_history_factor, forecast_horizon):
    x, y = [], []
    for j in range(past_history_factor, time_series.shape[1] - forecast_horizon + 1, forecast_horizon):
        indices = list(range(j - past_history_factor, j))

        window_ts = []
        for i in range(time_series.shape[0]):
            window_ts.append(time_series[i, indices])
        window = np.array(window_ts)

        x.append(window)
        y.append(camera_series[j: j + forecast_horizon])
    return np.array(x), np.array(y)


def rf(params):

    n_stimators_value, max_depth_value, min_samples_split_value, min_samples_leaf_value = params

    model = RandomForestRegressor(criterion='mse', n_jobs=-1, n_estimators=n_stimators_value,
                                  max_depth=max_depth_value, min_samples_split=min_samples_split_value,
                                  min_samples_leaf=min_samples_leaf_value)
    return model

def create_model_ml(model_name, params):
    assert model_name in model_factory_ML.keys(), "Model '{}' not supported".format(
        model_name
    )
    return model_factory_ML[model_name](params)

def train_trees(model_name, iter_params, x_train, y_train, x_test):#, norm_params, normalization_method):
    model = create_model_ml(model_name, iter_params)

    x_train2 = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    print('x_train: {} -> {}'.format(x_train.shape, x_train2.shape))
    training_time_0 = time.time()
    model.fit(x_train2, y_train)
    train_forecast = model.predict(x_train2)
    training_time = time.time() - training_time_0

    x_test2 = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
    print('x_test: {} -> {}'.format(x_test.shape, x_test2.shape))
    test_time_0 = time.time()
    test_forecast = model.predict(x_test2)
    test_time = time.time() - test_time_0

    # for i in range(test_forecast.shape[0]):
    #     nparams = norm_params[0]
    #     test_forecast[i] = denormalize(
    #         test_forecast[i], nparams, method=normalization_method,
    #     )

    return train_forecast, test_forecast, training_time, test_time

parameters= { 
    "normalization_method": ["minmax", "zscore"],
    "past_history_factor": [15, 30, 60, 100, 120],
    "model_params": {
        "rf" : {
            "n_stimators" : [100, 200, 300],
            "max_depth" : [3,6,8],
            "min_samples_split" : [2,4,8],
            "min_samples_leaf": [1,3, 7]
        }
    }
}

# "max_depth" : [2, 4, 6, 8, 10],
# "min_samples_split" : [2, 4, 6, 8],
# "min_samples_leaf": [1, 3, 5, 7]
model_factory_ML = {
    "rf": rf
}
TRAIN_ML = {
    'rf':  train_trees
}



## Loading Files 
cam = np.load("data/Camera.npy")
phs = np.load("data/OL_Phase.npy")
amp = np.load("data/OL_Magnitude.npy")

##SplittingRatio ML 
percentage = 80 #-- Train
split = int(np.shape(cam)[0]*percentage/100)
forecast_horizon = 1

print("Shape Camera  -->",np.shape(cam))
print("Shape Phase  -->",np.shape(phs))
print("Shape Amplitude  -->",np.shape(amp))
print("------------------------------------------")
print("")
assert(np.shape(cam)==np.shape(cam)==np.shape(amp))

cam_train, cam_test = cam[:split], cam[split:]
phs_train, phs_test = phs[:split], phs[split:]
amp_train, amp_test = amp[:split], amp[split:]
print("Shape Camera -- train -->",np.shape(cam_train))
print("Shape Camera -- test -->",np.shape(cam_test))
print("------------------------------------------")
print("")
train = np.array([phs_train, amp_train, cam_train])
test = np.array([phs_test, amp_test, cam_test ])

model_name="rf"
csv_filepath = 'results.csv'

results = pd.DataFrame(
columns=[
    "MODEL",
    "MODEL_INDEX",
    "MODEL_DESCRIPTION",
    "FORECAST_HORIZON",
    "PAST_HISTORY",
    "TRAIN_MAE",
    "TEST_MAE",
    "TRAIN_TIME",
    "TEST_TIME",
]
)

for normalization_method, past_history_factor in itertools.product(
                parameters['normalization_method'],
                parameters['past_history_factor']
):      
        # TODO: Normalization
        X_train, Y_train = windows_preprocessing(train, cam_train, past_history_factor, forecast_horizon)
        X_test, Y_test = windows_preprocessing(test, cam_test, past_history_factor, forecast_horizon)
        
        past_history_factor = X_train.shape[2]
        forecast_horizon = Y_train.shape[1]
        parameters_models = parameters['model_params'][model_name]

        list_parameters_models = []
        for parameter_field in parameters_models.keys():
            list_parameters_models.append(parameters_models[parameter_field])

        model_id = 0
        for iter_params in itertools.product(*list_parameters_models):

            train_forecast, test_forecast, training_time, test_time = TRAIN_ML[model_name](
                model_name,
                iter_params,
                X_train,
                Y_train,
                X_test,
                # norm_params,
                # normalization_method
            )
        
            train_metrics = np.mean(np.abs(train_forecast, np.squeeze(Y_train)))
            test_metrics = np.mean(np.abs(test_forecast, np.squeeze(Y_test)))
            
            results = results.append(
                {
                    "MODEL": model_name,
                    "MODEL_INDEX": model_id,
                    "MODEL_DESCRIPTION": str(iter_params),
                    "FORECAST_HORIZON": forecast_horizon,
                    "PAST_HISTORY": past_history_factor,
                    "TRAIN_MAE": train_metrics,
                    "TEST_MAE": test_metrics,
                    "TRAIN_TIME": training_time,
                    "TEST_TIME": test_time
                },
                ignore_index=True
            )
            
            model_id += 1
            print("Index -->",model_id)

results.to_csv(csv_filepath, sep=",", index=False)
