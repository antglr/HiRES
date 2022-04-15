from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
import scipy.io as sio
import pandas as pd
import time
from statsmodels.tsa.seasonal import seasonal_decompose
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K


def normalize(data, norm_params, normalization_method="zscore"):
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
    assert normalization_method in ["zscore", "minmax", "None"]
    if normalization_method == "zscore":
        return (data * norm_params["std"]) + norm_params["mean"]
    elif normalization_method == "minmax":
        return data * (norm_params["max"] - norm_params["min"]) + norm_params["min"]
    elif normalization_method == "None":
        return data
def denormalization(forecast, X, Y, norm_params, normalization_method):
    Y_denorm = np.zeros(Y.shape)
    for j in range(Y.shape[0]):
        nparams = norm_params[0]
        forecast[j] = denormalize(forecast[j], nparams, normalization_method)
        Y_denorm[j] = denormalize(Y[j], nparams, normalization_method)
    forecast = np.squeeze(forecast)
    metrics_train = np.abs(forecast-np.squeeze(Y_denorm))
    return forecast, metrics_train, Y_denorm
def get_normalization_params(data):
    d = data.flatten()
    norm_params = {}
    norm_params["mean"] = d.mean()
    norm_params["std"] = d.std()
    norm_params["max"] = d.max()
    norm_params["min"] = d.min()
    return norm_params

def normalize_dataset(x, y, normalization_method):
    norm_params = []
    for i in range(x.shape[0]):
        nparams = get_normalization_params(x[i])
        x[i] = normalize(np.array(x[i]), nparams, normalization_method)
        norm_params.append(nparams)
        
    nparams = get_normalization_params(y)
    y = normalize(y, nparams, normalization_method)
    norm_params.append(nparams)
    return x, y, norm_params


def shifting(targetShift, cam, OL_phs, OL_amp, ILmOL_phs, ILmOL_amp, laser_Phs, laser_amp, Egain):
    cam = np.roll(cam,targetShift)
    cam  = cam[:targetShift]
    OL_phs = OL_phs[:targetShift]
    OL_amp = OL_amp[:targetShift]
    ILmOL_phs = ILmOL_phs[:targetShift]
    ILmOL_amp = ILmOL_amp[:targetShift]
    laser_Phs = laser_Phs[:targetShift]
    laser_amp = laser_amp[:targetShift]
    Egain = Egain[:targetShift]
    return cam, OL_phs, OL_amp, ILmOL_phs, ILmOL_amp, laser_Phs, laser_amp, Egain
def loading(InOrOut, both, targetShift):
    OL_phs = np.load("data/"+InOrOut+"/OL_Phase.npy")     
    OL_amp = np.load("data/"+InOrOut+"/OL_Magnitude.npy") 
    ILmOL_phs = np.load("data/"+InOrOut+"/OL_Phase.npy") - np.load("data/"+InOrOut+"/IL_Phase.npy") 
    ILmOL_amp = np.load("data/"+InOrOut+"/OL_Magnitude.npy") - np.load("data/"+InOrOut+"/IL_Magnitude.npy") 
    laser_Phs = np.load("data/"+InOrOut+"/Laser_Phs.npy")  
    laser_amp = np.load("data/"+InOrOut+"/Laser_Amp.npy") 
    Egain =  np.load("data/"+InOrOut+"/OL_Energy.npy")
    cam = sio.loadmat("data/"+InOrOut+"/Data.mat") 
    cam = cam['saveData']
    cam = cam[:,1]

    cam, OL_phs, OL_amp, ILmOL_phs, ILmOL_amp, laser_Phs, laser_amp, Egain = shifting(targetShift, cam, OL_phs, OL_amp, ILmOL_phs, ILmOL_amp, laser_Phs, laser_amp, Egain)
    if both:
        if (InOrOut == "InLoop"):
            InOrOut1 = "OutLoop"
        else:
            InOrOut1 = "InLoop"
        OL_phs1,OL_amp1,ILmOL_phs1,ILmOL_amp1,laser_Phs1,laser_amp1,Egain1,cam1 = loading(InOrOut1)
        cam1, OL_phs1, OL_amp1, ILmOL_phs1, ILmOL_amp1, laser_Phs1, laser_amp1, Egain1 = shifting(targetShift, cam1, OL_phs1, OL_amp1, ILmOL_phs1, ILmOL_amp1, laser_Phs1, laser_amp1, Egain1)

        OL_phs = np.concatenate((OL_phs, OL_phs1), axis = 0)
        OL_amp = np.concatenate((OL_amp, OL_amp1), axis = 0)
        ILmOL_phs = np.concatenate((ILmOL_phs, ILmOL_phs1), axis = 0)
        ILmOL_amp = np.concatenate((ILmOL_amp, ILmOL_amp1), axis = 0)
        laser_Phs = np.concatenate((laser_Phs, laser_Phs1), axis = 0)
        laser_amp = np.concatenate((laser_amp, laser_amp1), axis = 0)
        Egain = np.concatenate((Egain, Egain1), axis = 0)
        cam = np.concatenate((cam, cam1), axis = 0)

    return OL_phs, OL_amp, ILmOL_phs, ILmOL_amp, laser_Phs, laser_amp, Egain, cam

def plotting_component(data, decomposition):
    trend    = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, figsize=(16,6))
    ax1.plot(data, 'tab:blue')
    ax1.set_title('Original Data set')
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))   
    ax2.plot(trend, 'tab:orange')
    ax2.set_title('Trend')
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))   
    ax3.plot(seasonal, 'tab:green')
    ax3.set_title('Seasonal')
    ax3.ticklabel_format(axis="y", style="sci", scilimits=(0,0))   
    ax4.plot(residual, 'tab:red')
    ax4.set_title('Residual')
    ax4.ticklabel_format(axis="y", style="sci", scilimits=(0,0))   
    plt.tight_layout()
    plt.savefig("0_plotting_component.png")
    plt.show()
    return 

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
def mlp():
    inputs = tf.keras.layers.Input(shape=np.shape(X_train)[-1:])
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(2048, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(254, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(forecast_horizon)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

if __name__ == "__main__":
    t = time.time()
    percentage = 80 
    fit_or_proj = "fit" 
    past_history = 30
    forecast_horizon = 1
    normalization_method = 'zscore'
    fetureSelection = 0

    #InLoop
    OL_phs,OL_amp,ILmOL_phs,ILmOL_amp,laser_Phs,laser_amp,Egain,cam = loading(InOrOut = "OutLoop", both = False, targetShift = -2)
    ind = np.arange(0, len(cam))
    cam_panda = pd.DataFrame({'data': cam}, index=ind)
    decomposition = seasonal_decompose(cam_panda, model='additive', freq= 60)
    plotting_component(cam, decomposition)
    
    cma_trend    = (decomposition.trend).to_numpy()
    y = cma_trend[30:-30]

    if fetureSelection:
        x = np.array([OL_amp[30:-30], OL_phs[30:-30]]) 
    else:
        x = np.array([OL_phs[30:-30], OL_amp[30:-30], ILmOL_phs[30:-30], ILmOL_amp[30:-30], laser_Phs[30:-30], laser_amp[30:-30]])
    x, y, norm_params = normalize_dataset(x, y, normalization_method)
    
    stop_train = int(len(cma_trend)*percentage/100)
    Y_train, Y_test = y[:stop_train], y[stop_train:]
    X_train, X_test = x[:,:stop_train], x[:,stop_train:]
      
    X_train = X_train.reshape(X_train.shape[1],  X_train.shape[0] )
    Y_train = Y_train.reshape(Y_train.shape[0],1)

    X_test = X_test.reshape(X_test.shape[1],  X_test.shape[0] )
    Y_test = Y_test.reshape(Y_test.shape[0],1)

    #model = LinearRegression()
    #model.fit(X_train,Y_train)
    #train_forecast = model.predict(X_train)    
    model = mlp()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss = root_mean_squared_error)  
    #callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50) 
    history = model.fit(X_train,Y_train,batch_size=64, epochs=100, validation_split=0.1, shuffle=True)
    train_forecast = model(X_train).numpy()

    plt.figure()
    plt.plot(Y_train,'b',label="Label")
    plt.plot(train_forecast,'r', label="Prediction")
    plt.legend()
    plt.savefig("0_Train.png")

    test_forecast = model(X_test).numpy()
    plt.figure()
    plt.plot(Y_test,'b',label="Label")
    plt.plot(test_forecast,'r', label="Prediction")
    plt.legend()
    plt.savefig("0_Test.png")
