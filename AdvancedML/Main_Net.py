import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scipy.io as sio
from keras import backend as K

def normalize(data, norm_params, normalization_method="zscore"):
    """
    Normalize time series
    :param data: time series
    :param norm_params: tuple with params mean, std, max, min
    :param method: zscore or minmax
    :return: normalized time series
    """
    assert normalization_method in ["zscore", "minmax", "None"]

    if normalization_method == "zscore":
        std = norm_params["std"]
        if std == 0.0:
            std = 1e-10
        return (data - norm_params["mean"]) / norm_params["std"]

    elif normalization_method == "minmax":
        denominator = norm_params["max"] - norm_params["min"]

        if denominator == 0.0:
            denominator = 1e-10
        return (data - norm_params["min"]) / denominator

    elif normalization_method == "None":
        return data

def denormalize(data, norm_params, normalization_method="zscore"):
    """
    Reverse normalization time series
    :param data: normalized time series
    :param norm_params: tuple with params mean, std, max, min
    :param normalization_method: zscore or minmax
    :return: time series in original scale
    """
    assert normalization_method in ["zscore", "minmax", "None"]

    if normalization_method == "zscore":
        return (data * norm_params["std"]) + norm_params["mean"]

    elif normalization_method == "minmax":
        return data * (norm_params["max"] - norm_params["min"]) + norm_params["min"]

    elif normalization_method == "None":
        return data

def get_normalization_params(data):
    """
    Obtain parameters for normalization
    :param data: time series
    :return: dict with string keys
    """
    d = data.flatten()
    norm_params = {}
    norm_params["mean"] = d.mean()
    norm_params["std"] = d.std()
    norm_params["max"] = d.max()
    norm_params["min"] = d.min()
    return norm_params

def normalize_dataset(train, test, normalization_method, dtype="float32"):
    # Normalize train data
    norm_params = []
    for i in range(train.shape[0]):
        nparams = get_normalization_params(train[i])
        train[i] = normalize(np.array(train[i], dtype=dtype), nparams, normalization_method)
        norm_params.append(nparams)

    # Normalize test data
    test = np.array(test, dtype=dtype)
    for i in range(test.shape[0]):
        nparams = norm_params[i]
        test[i] = normalize(test[i], nparams, normalization_method)

    return train, test, norm_params


def windows_preprocessing(time_series, past_history, forecast_horizon):
    x, y = [], []
    camera_series = time_series[0]
    for j in range(past_history, time_series.shape[1] - forecast_horizon + 1, forecast_horizon):
        indices = list(range(j - past_history, j))

        window_ts = []
        for i in range(time_series.shape[0]):
            window_ts.append(time_series[i, indices])
        window = np.array(window_ts).transpose((1,0))

        x.append(window)
        y.append(camera_series[j: j + forecast_horizon])
    return np.array(x), np.array(y)

def rmse(y_pred, y_true):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

## Loading Files 
InOrOut = "OutLoop"
#InOrOut = "InLoop" 
#camFit = np.load("data/"+InOrOut+"/CameraFit.npy")    #Y
#camProj = np.load("data/"+InOrOut+"/CameraProj.npy")  #Y
OL_phs = np.load("data/"+InOrOut+"/OL_Phase.npy")     #X
OL_amp = np.load("data/"+InOrOut+"/OL_Magnitude.npy") #X
ILmOL_phs = np.load("data/"+InOrOut+"/OL_Phase.npy") - np.load("data/"+InOrOut+"/IL_Phase.npy") #X 
ILmOL_amp = np.load("data/"+InOrOut+"/OL_Magnitude.npy") - np.load("data/"+InOrOut+"/IL_Magnitude.npy") #X
laser_Phs = np.load("data/"+InOrOut+"/Laser_Phs.npy")  #X 
laser_amp = np.load("data/"+InOrOut+"/Laser_Amp.npy")  #X

cam = sio.loadmat("data/"+InOrOut+"/Data.mat") 
cam = cam['saveData']
cam = cam[:,1]
##SplittingRatio ML 
percentage = 80 #-- Train
fit_or_proj = "fit" 
past_history = 60
forecast_horizon = 1
normalization_method = 'minmax'

targetShift = -2
fetureSelection = 0

data_ln = len(cam)
# SHIFT
if (targetShift!=0):
    OL_phs = np.roll(OL_phs,targetShift)
    OL_amp = np.roll(OL_amp,targetShift)
    ILmOL_phs = np.roll(ILmOL_phs,targetShift)
    ILmOL_amp = np.roll(ILmOL_amp,targetShift)
    laser_Phs = np.roll(laser_Phs,targetShift)
    laser_amp = np.roll(laser_amp,targetShift)
    # camFit  = camFit[:targetShift]
    # camProj = camProj[:targetShift]
    cam  = cam[:targetShift]
    OL_phs = OL_phs[:targetShift]
    OL_amp = OL_amp[:targetShift]
    ILmOL_phs = ILmOL_phs[:targetShift]
    ILmOL_amp = ILmOL_amp[:targetShift]
    laser_Phs = laser_Phs[:targetShift]
    laser_amp = laser_amp[:targetShift]

if fetureSelection:
    # Selected variables
    FullDataset = np.array([cam, OL_phs, OL_amp, ILmOL_amp, laser_amp]) 
else:
    # All variables
    FullDataset = np.array([cam, OL_phs, OL_amp, ILmOL_phs, ILmOL_amp, laser_Phs, laser_amp])

splitting_traintest = 3
traintest_size = data_ln//splitting_traintest
split = int(traintest_size*percentage/100)

metric_train = []
metric_test = []
metric_train_pre = []
print("")
print("Dataset length:",data_ln)
for i in range(splitting_traintest):
    start_train = traintest_size*i
    stop_train = split*(i+1)
    start_test = stop_train + 1
    stop_test = traintest_size*(i+1)-1

    print("Train start--stop:",start_train,"--",stop_train)
    print("Test start--stop:",start_test,"--",stop_test)
    print("")
    train, test = FullDataset[:,start_train:stop_train], FullDataset[:,start_test:stop_test]

    train, test, norm_params = normalize_dataset(train, test, normalization_method, dtype="float64")
    X_train, Y_train = windows_preprocessing(train, past_history, forecast_horizon)
    X_test, Y_test = windows_preprocessing(test, past_history, forecast_horizon)

    print(np.shape(X_test))
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
    
    model = RandomForestRegressor(criterion='mse', n_jobs=-1, n_estimators=100, max_depth=10,min_samples_split=2,min_samples_leaf=3)
    #model = LinearRegression()
    #model = MLPRegressor(hidden_layer_sizes=[32,16,8], random_state=1, max_iter=500)
    ###train_forecast_pre = model.predict(X_train) <-- Dose not work....
    model.fit(X_train,Y_train)
    train_forecast = model.predict(X_train)
    
    #inputs = tf.keras.layers.Input(shape=np.shape(X_train)[-2:])
    #x = tf.keras.layers.LSTM(units=128, return_sequences=False)(inputs)
    #x = tf.keras.layers.Flatten()(x)
    #x = tf.keras.layers.Dense(forecast_horizon)(x)
    #model = tf.keras.Model(inputs=inputs, outputs=x)
    #model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=rmse)   
    ##callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5) 
    #train_forecast_pre = model(X_train).numpy()
    ##history = model.fit(X_train,Y_train,batch_size=8, epochs=20, validation_data=(X_test,Y_test), shuffle=False, callbacks=[callback])
    #history = model.fit(X_train,Y_train,batch_size=8, epochs=20, validation_data=(X_test,Y_test), shuffle=False)
    #train_forecast = model(X_train).numpy()
    
    Y_train_denorm = np.zeros(Y_train.shape)
    for j in range(Y_train.shape[0]):
        nparams = norm_params[0]
        train_forecast[j] = denormalize(train_forecast[j], nparams, normalization_method)
        Y_train_denorm[j] = denormalize(Y_train[j], nparams, normalization_method)
    train_forecast = np.squeeze(train_forecast)
    metrics_train = np.abs(train_forecast-np.squeeze(Y_train_denorm))

    #test_forecast = model(X_test).numpy()
    test_forecast = model.predict(X_test)

    Y_test_denorm = np.zeros(Y_test.shape)
    for j in range(Y_test.shape[0]):
        nparams = norm_params[0]
        test_forecast[j] = denormalize(test_forecast[j], nparams, normalization_method)
        Y_test_denorm[j] = denormalize(Y_test[j], nparams, normalization_method)
    test_forecast = np.squeeze(test_forecast)
    metrics_test = np.abs(test_forecast-np.squeeze(Y_test_denorm))

    ###metric_train_pre.append(mean_squared_error(Y_train, train_forecast_pre))

    metric_train.append(mean_squared_error(Y_train_denorm, train_forecast, squared=False))

    metric_test.append(mean_squared_error(Y_test_denorm, test_forecast, squared=False))

    # train_loss = history.history['loss']
    # test_loss = history.history['val_loss']
    # fig, ax = plt.subplots(figsize=(8,6))
    # ax.plot(train_loss,"k",label="Training")
    # ax.plot(test_loss,"r",label="Test")
    # ax.set_xlabel("Epochs")
    # ax.set_ylabel("Loss")
    # plt.legend()

###print("Initial Guess ",metric_train_pre)
metric_train = [np.format_float_scientific(m, precision=2) for m in metric_train]
print("Training RMSE",metric_train)
metric_test = [np.format_float_scientific(m, precision=2) for m in metric_test]
print("Test RMSE",metric_test)

# plt.show()