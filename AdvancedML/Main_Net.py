import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scipy.io as sio
import pandas as pd

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
    # time_series = time_series[1:]
    for j in range(past_history, time_series.shape[1] - forecast_horizon + 1, forecast_horizon):
        indices = list(range(j - past_history, j))

        window_ts = []
        for i in range(time_series.shape[0]):
            window_ts.append(time_series[i, indices])
        window = np.array(window_ts).transpose((1,0))

        x.append(window)
        y.append(camera_series[j: j + forecast_horizon])
    return np.array(x), np.array(y)


def plotting_2(train_forecast,Y_train,test_forecast,Y_test,i):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,6),sharey=True)
    ax1.plot(train_forecast,np.squeeze(Y_train),"+k")
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax1.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    ax1.set_ylabel("Label")
    ax1.set_xlabel("Prediction")
    ax1.set_title("Train")
    ax1.grid(axis="x")
    ax1.grid(axis="y")
    ax2.plot(test_forecast,np.squeeze(Y_test),"+k")
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax2.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    ax2.set_xlabel("Prediction")
    ax2.set_title("Test")
    ax2.grid(axis="x")
    ax2.grid(axis="y")
    plt.suptitle('SubSet -->{}'.format(i+1), fontsize=20)
    plt.savefig("plot2.png")
    return 

def plotting_3(train_forecast,Y_train,test_forecast,Y_test,i):
    massimo = max(np.max(Y_train),np.max(train_forecast),np.max(Y_test),np.max(test_forecast))
    minimo = min(np.min(Y_train),np.min(train_forecast),np.min(Y_test),np.min(test_forecast))

    fig, (ax1,ax2) = plt.subplots(2,figsize=(16,6))
    ax1.plot(np.squeeze(Y_train),"r",label= "Label")
    ax1.plot(train_forecast,"k",label= "Prediction")
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax1.set_ylabel("Centroid Error")
    ax1.set_title("Train")
    ax1.grid(axis="x")
    ax1.grid(axis="y")
    ax1.set_ylim((minimo, massimo))
    ax1.legend()
    ax2.plot(np.squeeze(Y_test),"r",label= "Label")
    ax2.plot(test_forecast,"k",label= "Prediction")
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax2.set_ylabel("Centroid Error")
    ax2.set_xlabel("Time")
    ax2.set_title("Test")
    ax2.grid(axis="x")
    ax2.grid(axis="y")
    ax2.set_ylim((minimo, massimo))
    ax2.legend()
    plt.suptitle('SubSet -->{}'.format(i+1), fontsize=20)
    plt.savefig("plot3.png")
    return 

## Loading Files 
InOrOut = "OutLoop"
# InOrOut = "InLoop" 
#camFit = np.load("data/"+InOrOut+"/CameraFit.npy")    #Y
#camProj = np.load("data/"+InOrOut+"/CameraProj.npy")  #Y
OL_phs = np.load("data/"+InOrOut+"/OL_Phase.npy")     #X
OL_amp = np.load("data/"+InOrOut+"/OL_Magnitude.npy") #X
ILmOL_phs = np.load("data/"+InOrOut+"/OL_Phase.npy") - np.load("data/"+InOrOut+"/IL_Phase.npy") #X 
ILmOL_amp = np.load("data/"+InOrOut+"/OL_Magnitude.npy") - np.load("data/"+InOrOut+"/IL_Magnitude.npy") #X
laser_Phs = np.load("data/"+InOrOut+"/Laser_Phs.npy")  #X 
laser_amp = np.load("data/"+InOrOut+"/Laser_Amp.npy")  #X
Egain =  np.load("data/"+InOrOut+"/OL_Energy.npy")


cam = sio.loadmat("data/"+InOrOut+"/Data.mat") 
cam = cam['saveData']
cam = cam[:,1]
##SplittingRatio ML 
percentage = 80 #-- Train
fit_or_proj = "fit" 
past_history = 30
forecast_horizon = 1
normalization_method = 'zscore'

targetShift = -2
fetureSelection = 1
shouldIplot = 1

data_ln = len(cam)
# SHIFT
if (targetShift!=0):
    cam = np.roll(cam, targetShift)
    cam  = cam[:targetShift]
    OL_phs = OL_phs[:targetShift]
    OL_amp = OL_amp[:targetShift]
    ILmOL_phs = ILmOL_phs[:targetShift]
    ILmOL_amp = ILmOL_amp[:targetShift]
    laser_Phs = laser_Phs[:targetShift]
    laser_amp = laser_amp[:targetShift]    
    Egain = Egain[:targetShift]

if fetureSelection:
    # Selected variables
    FullDataset = np.array([cam, OL_amp, OL_phs]) 
else:
    # All variables
    FullDataset = np.array([cam, OL_phs, OL_amp, ILmOL_phs, ILmOL_amp, laser_Phs, laser_amp])


splitting_traintest = 1
traintest_size = data_ln//splitting_traintest
split = int(traintest_size*percentage/100)

metric_train = []
metric_test = []
metric_train_pre = []
print("")
print("Dataset length:",data_ln)
for i in range(splitting_traintest):
    start_train = traintest_size*i
    stop_train = traintest_size*i + split
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
    print(np.shape(X_train))    
    
    inputs = tf.keras.layers.Input(shape=np.shape(X_train)[-2:])
    # x = tf.keras.layers.LSTM(units=300, return_sequences=True, dropout=0.1)(inputs)
    # x = tf.keras.layers.LSTM(units=300, return_sequences=True, dropout=0.1)(inputs)
    x = tf.keras.layers.LSTM(units=300, return_sequences=False, dropout=0.1)(inputs)
    x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    # x = tf.keras.layers.Dense(256, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    # x = tf.keras.layers.Dense(128, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    # x = tf.keras.layers.Dense(32, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(forecast_horizon)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    
    # x = tf.keras.layers.Flatten()(inputs)
    # x = tf.keras.layers.Dense(512, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    # # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dense(64, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    # # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dense(32, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    # # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dense(16, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    # # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dense(8, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    # # x = tf.keras.layers.BatchNormalization()(x)f
    # x = tf.keras.layers.Dense(forecast_horizon)(x)
    # model = tf.keras.Model(inputs=inputs, outputs=x)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')       
    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=1e-2,
    #     decay_steps=100,
    #     decay_rate=0.9)
    # optimizer = keras.optimizers.Adam(learning_rate=lr_schedule) 
    # model.compile(optimizer=optimizer, loss='mse')  

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5) 
    train_forecast_pre = model(X_train).numpy()
    #history = model.fit(X_train,Y_train,batch_size=8, epochs=20, validation_data=(X_test,Y_test), shuffle=True, callbacks=[callback])
    history = model.fit(X_train,Y_train,batch_size=64, epochs=100, validation_data=(X_test,Y_test), shuffle=True)
    train_forecast = model(X_train).numpy()

    

    Y_train_denorm = np.zeros(Y_train.shape)
    for j in range(Y_train.shape[0]):
        nparams = norm_params[0]
        train_forecast[j] = denormalize(train_forecast[j], nparams, normalization_method)
        Y_train_denorm[j] = denormalize(Y_train[j], nparams, normalization_method)
    train_forecast = np.squeeze(train_forecast)
    metrics_train = np.abs(train_forecast-np.squeeze(Y_train_denorm))

    test_forecast = model(X_test).numpy()

    Y_test_denorm = np.zeros(Y_test.shape)
    for j in range(Y_test.shape[0]):
        nparams = norm_params[0]
        test_forecast[j] = denormalize(test_forecast[j], nparams, normalization_method)
        Y_test_denorm[j] = denormalize(Y_test[j], nparams, normalization_method)
    test_forecast = np.squeeze(test_forecast)
    metrics_test = np.abs(test_forecast-np.squeeze(Y_test_denorm))

    ###metric_train_pre.append(mean_squared_error(Y_train, train_forecast_pre))


    metric_train.append([mean_squared_error(Y_train_denorm, train_forecast, squared=False), mean_absolute_percentage_error(Y_train_denorm, train_forecast)])

    metric_test.append([mean_squared_error(Y_test_denorm, test_forecast, squared=False), mean_absolute_percentage_error(Y_test_denorm, test_forecast)])

    train_loss = history.history['loss']
    test_loss = history.history['val_loss']
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(train_loss,"k",label="Training")
    ax.plot(test_loss,"r",label="Test")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    plt.legend()
    plt.savefig("loss_curves.png")
    plotting_2(train_forecast,Y_train_denorm,test_forecast,Y_test_denorm,i)
    plotting_3(train_forecast,Y_train_denorm,test_forecast,Y_test_denorm,i)
    if shouldIplot:
        plt.show()

###print("Initial Guess ",metric_train_pre)
m_train = [np.format_float_scientific(m[0], precision=2) for m in metric_train]
print("Training RMSE",m_train)
m_test = [np.format_float_scientific(m[0], precision=2) for m in metric_test]
print("Test RMSE",m_test)

m_train = [m[1]*100 for m in metric_train]
print("Training MAPE",m_train)
m_test = [m[1]*100 for m in metric_test]
print("Test MAPE",m_test)

