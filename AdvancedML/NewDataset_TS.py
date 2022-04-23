import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import scipy.io as sio
import pandas as pd
import time
from sklearn.linear_model import SGDRegressor
import click

def clrscr():
    click.clear()
def _check_keys(dict_value):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict_value:
        if isinstance(dict_value[key], sio.matlab.mio5_params.mat_struct):
            dict_value[key] = _todict(dict_value[key])
    return dict_value
def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict_value = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict_value[strg] = _todict(elem)
        else:
            dict_value[strg] = elem
    return dict_value
def loadmat(filename):
    """
    this function should be called instead of direct scipy.io .loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def to_dataframe(dictionary):
    full_data = dictionary["syncData"]
    
    x_Laser = full_data["LCam1_Gauss"][:,0]
    
    xRMS_Laser = full_data["LCam1_Gauss"][:,1]
    y_Laser = full_data["LCam1_Gauss"][:,2]
    yRMS_Laser = full_data["LCam1_Gauss"][:,3]
    u_Laser = full_data["LCam1_Gauss"][:,4]
    uRMS_Laser = full_data["LCam1_Gauss"][:,5]
    v_Laser = full_data["LCam1_Gauss"][:,6]
    vRMS_Laser = full_data["LCam1_Gauss"][:,7]
    sum_Laser = full_data["LCam1_Gauss"][:,8]
    rf_amp = full_data["Cav_Amp"]
    rf_phs = full_data["Cav_Phs"]
    fw2_amp =full_data["Fwd2_Amp"] 
    fw2_phs =full_data["Fwd2_Phs"] 
    rv_amp = full_data["Rev_Amp"]
    rv_phs = full_data["Rev_Phs"]
    fw1_amp = full_data["Fwd1_Amp"]
    fw1_phs = full_data["Fwd1_Phs"]
    laser_amp = full_data["LP_Amp"]
    laser_phs = full_data["LP_Phase"]
    
    cam = full_data["AdjUCam1Pos"]
    return  x_Laser, xRMS_Laser, y_Laser, yRMS_Laser, u_Laser, uRMS_Laser, v_Laser, vRMS_Laser, sum_Laser, rf_amp, rf_phs, fw2_amp, fw2_phs, rv_amp, rv_phs, fw1_amp, fw1_phs, laser_amp, laser_phs, cam

def get_normalization_params(data):
    d = data.flatten()
    norm_params = {}
    norm_params["mean"] = d.mean()
    norm_params["std"] = d.std()
    norm_params["max"] = d.max()
    norm_params["min"] = d.min()
    return norm_params
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

def denormalize(data, norm_params, normalization_method="zscore"):
    assert normalization_method in ["zscore", "minmax", "None"]
    if normalization_method == "zscore":
        return (data * norm_params["std"]) + norm_params["mean"]
    elif normalization_method == "minmax":
        return data * (norm_params["max"] - norm_params["min"]) + norm_params["min"]
    elif normalization_method == "None":
        return data
def denormalization(forecast, Y, norm_params, normalization_method):
    Y_denorm = np.zeros(Y.shape)
    for j in range(Y.shape[0]):
        nparams = norm_params[0]
        forecast[j] = denormalize(forecast[j], nparams, normalization_method)
        Y_denorm[j] = denormalize(Y[j], nparams, normalization_method)
    forecast = np.squeeze(forecast)
    Y_denorm = np.squeeze(Y_denorm)
    return forecast, Y_denorm

def windows_preprocessing_Antonio(time_series, past_history):
    x, y = [], []
    camera_series = time_series[0].copy()
    #time_series = time_series[1:] # This line removes cam from the X
    for j in range(past_history, time_series.shape[1]):
        indices = list(range(j - past_history, j+1))
        
        window_ts = time_series[:, indices].copy()
        window_ts[0,-1] = 0 #np.mean(window_ts[0,:-1]) # Remove (Zero) the value of the cam to be guessed -- To improve!!!!!!!
        window = np.array(window_ts).transpose((1,0))
        x.append(window)
        
        y.append(camera_series[j: j + 1])
    return np.array(x), np.array(y)

def testing(X_test, past_history, model):
    test_forecast = []
    #lastguess = 0
    X_test_copy = X_test.copy()
    lastguess = 0 #np.mean(X_test_copy[:, :past_history-1])
    for k in range(past_history, np.shape(X_test)[1]):        
        indices = list(range(k - past_history, k+1))
        
        X_test_copy[0,indices[-1]] = lastguess
        X_test_new = X_test_copy[:, indices].copy()
        #X_test_new[0,-1] = lastguess 
        X_test_new = np.array(X_test_new).transpose((1,0))
         
        X_test_new = X_test_new.reshape(1,-1) #MLP - LR 
        #X_test_new = X_test_new.reshape(1,*np.shape(X_test_new)) # LST
        lastguess = model.predict(X_test_new).flatten()
        test_forecast.append(lastguess)
    return np.array(test_forecast)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
def lstm():
    inputs = tf.keras.layers.Input(shape=np.shape(X_train)[-2:])
    x = tf.keras.layers.LSTM(units=128, return_sequences=False)(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model
def cnn():
    inputs = tf.keras.layers.Input(shape=np.shape(X_train)[-2:])
    x = tf.keras.layers.Conv1D(148, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)    
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

def plotting_1(y_train,train_forecast,y_test, test_forecast, l): 
    if (l==0):
        len1 = len(y_train)
        len2 = len(train_forecast)
    else:
        len1 = l
        len2 = l   
    fig, (ax1, ax2) = plt.subplots(2, figsize=(16,6))
    ax1.plot(y_train[:len1],'b',label="Label")
    ax1.plot(train_forecast[:len1],'r', label="Prediction")
    ax1.set_title('Train')
    ax1.set_xlabel('N sample')
    ax1.set_ylabel('Centroid [pixel]')
    
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))   
    ax1.legend()
    ax2.plot(y_test[:len2],'b',label="Label")
    ax2.plot(test_forecast[:len2],'r', label="Prediction")
    ax2.set_title('Test')
    ax2.set_xlabel('N sample')
    ax2.set_ylabel('Centroid [pixel]')
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))   
    ax2.legend()
    ax2.set_xlabel('N sample')
    ax2.set_ylabel('Centroid [pixel]')
    plt.tight_layout()
    plt.savefig("NewDataset_TS_Train_Test.png")
    plt.show()
    return


if __name__ == "__main__":
    clrscr()
    t = time.time()
    
    past_history = 35
    normalization_method = 'minmax'
    #"zscore", "minmax",
    fetureSelection = 0
    plt_len = 2000

    #filname = "new_dataset/Fourth_Dataset/OpenLoop1postp.mat" #-->OpenLoop
    filname = "new_dataset/Fourth_Dataset/ClosedLoop1postp.mat" #-->CloseLoop
    
    dict_all = loadmat(filname)
    x_Laser, xRMS_Laser, y_Laser, yRMS_Laser, u_Laser, uRMS_Laser, v_Laser, vRMS_Laser, sum_Laser, rf_amp, rf_phs, fw2_amp, fw2_phs, rv_amp, rv_phs, fw1_amp, fw1_phs, laser_amp, laser_phs, cam = to_dataframe(dict_all) 
    x_Laser, xRMS_Laser, y_Laser, yRMS_Laser, u_Laser, uRMS_Laser, v_Laser, vRMS_Laser, sum_Laser, rf_amp, rf_phs, fw2_amp, fw2_phs, rv_amp, rv_phs, fw1_amp, fw1_phs, laser_amp, laser_phs, cam  = x_Laser[:-500], xRMS_Laser[:-500], y_Laser[:-500], yRMS_Laser[:-500], u_Laser[:-500], uRMS_Laser[:-500], v_Laser[:-500], vRMS_Laser[:-500], sum_Laser[:-500], rf_amp[:-500], rf_phs[:-500], fw2_amp[:-500], fw2_phs[:-500], rv_amp[:-500], rv_phs[:-500], fw1_amp[:-500], fw1_phs[:-500], laser_amp[:-500], laser_phs[:-500], cam[:-500] 
    
    if fetureSelection:
        FullDataset = np.array([cam, y_Laser, u_Laser, v_Laser, rf_amp, fw2_amp, rv_amp, rv_phs, fw1_phs, laser_amp]) 
    else:
        FullDataset = np.array([cam, x_Laser, xRMS_Laser, y_Laser, yRMS_Laser, u_Laser, uRMS_Laser, v_Laser, vRMS_Laser, sum_Laser, rf_amp, rf_phs, fw2_amp, fw2_phs, rv_amp, rv_phs, fw1_amp, fw1_phs, laser_amp, laser_phs])    

    start_train = 0
    stop_train = int(len(cam)*0.8)
    print("Train start--stop:",start_train,"--",stop_train-1)
    print("Test start--stop:",stop_train,"--",len(cam))
    print("")
    
    train, test = FullDataset[:,start_train:stop_train], FullDataset[:,stop_train:]
    train, test, norm_params = normalize_dataset(train, test, normalization_method, dtype="float64")
    
    #Only TRAIN
    X_train, Y_train = windows_preprocessing_Antonio(train, past_history)
    # FOR Sklearn
    X_train= X_train.reshape(X_train.shape[0], -1)
    #model = MLPRegressor(hidden_layer_sizes=(148), solver='adam',  learning_rate='constant', learning_rate_init=0.01,  
    #                     activation="relu" ,random_state=42, max_iter=2000, shuffle=True, early_stopping=True, 
    #                     validation_fraction=0.1, n_iter_no_change=30, verbose=0).fit(X_train, Y_train)    
    model = LinearRegression().fit(X_train, Y_train)
    
    # model = cnn()
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=root_mean_squared_error)  
    # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50) 
    # history = model.fit(X_train, Y_train, batch_size=64, epochs=200, validation_split=(0.1), shuffle=True)
    
    train_forecast = model.predict(X_train)    
    train_forecast, Y_train_denorm= denormalization(train_forecast, Y_train, norm_params, normalization_method)
    std_data_train = np.std(Y_train_denorm)
    rms_train = np.sqrt(np.mean(Y_train_denorm**2))
    std_train = np.std(Y_train_denorm-train_forecast)    
    rmse_train = mean_squared_error(Y_train_denorm, train_forecast, squared=False)
    
    print("")
    print("-------------------------------------- TRAIN --------------------------------------")
    print("STD (Lable)  ---------------------------------------------------->",round(std_data_train, 3))
    print("RMS  ----------------------------------------------------------->",round(rms_train, 3))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("STD (Lable - Prediction)  --------------------------------------->",round(std_train, 3))
    print("RMSE  ----------------------------------------------------------->",round(rmse_train, 3))
    print("-----------------------------------------------------------------------------------")
    print("")
    print("")
    
    Y_test = test[0,past_history:]
    test_forecast = testing(test, past_history, model)     
    test_forecast, Y_test_denorm= denormalization(test_forecast, Y_test, norm_params, normalization_method)    
    
    std_data_test = np.std(Y_test_denorm)
    rms_test = np.sqrt(np.mean(Y_test_denorm**2))
    std_test = np.std(Y_test_denorm-test_forecast)       
    rmse_test = mean_squared_error(Y_test_denorm, test_forecast, squared=False)
    
    print("--------------------------------------- TEST ---------------------------------------")
    print("STD (Lable)  ---------------------------------------------------->",round(std_data_test, 3))
    print("RMS  ----------------------------------------------------------->",round(rms_test, 3))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("STD (Lable - Prediction)  --------------------------------------->",round(std_test, 3))
    print("RMSE ------------------------------------------------------------>",round(rmse_test, 3))
    print("-----------------------------------------------------------------------------------") 

    plotting_1(Y_train_denorm,train_forecast, Y_test_denorm, test_forecast, l=0)
    elapsed = time.time() - t
    print("TIME:",elapsed)
################################################################################################################
################################################################################################################
################################################################################################################