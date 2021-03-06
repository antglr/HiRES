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
def denormalization(forecast, X, Y, norm_params, normalization_method):
    Y_denorm = np.zeros(Y.shape)
    for j in range(Y.shape[0]):
        nparams = norm_params[0]
        forecast[j] = denormalize(forecast[j], nparams, normalization_method)
        Y_denorm[j] = denormalize(Y[j], nparams, normalization_method)
    forecast = np.squeeze(forecast)
    metrics_train = np.abs(forecast-np.squeeze(Y_denorm))
    return forecast, metrics_train, Y_denorm
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


def plotting_1(history,i):
    train_loss = history.history['loss']
    test_loss = history.history['val_loss']
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(train_loss,"k",label="Training")
    ax.plot(test_loss,"r",label="Test")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    plt.legend()
    plt.savefig("loss_curves.png")
    save_name = "plot1_" +str(i) + ".png"
    plt.savefig(save_name)
    return 
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
    save_name = "plot2_" +str(i) + ".png"
    plt.savefig(save_name)
    return 
def plotting_3(train_forecast,Y_train,test_forecast,Y_test,i, r):
    massimo = max(np.max(Y_train),np.max(train_forecast),np.max(Y_test),np.max(test_forecast))
    minimo = min(np.min(Y_train),np.min(train_forecast),np.min(Y_test),np.min(test_forecast))

    diff_train = np.abs(Y_train - train_forecast)
    diff_test = np.abs(Y_test - test_forecast)
    RMS_train = np.sqrt(np.mean(Y_train**2))
    RMSE_train = np.sqrt(np.mean((diff_train)**2))
    RMS_test = np.sqrt(np.mean(Y_test**2))
    RMSE_test = np.sqrt(np.mean((diff_test)**2))

    fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,figsize=(16,6))
    ax1.plot(np.squeeze(Y_train[:r]),"r",label= "Label")
    ax1.plot(train_forecast[:r],"k",label= "Prediction")
    ax1.plot(RMS_train, color='g', linestyle='-',label= "RMS:{}".format(np.format_float_scientific(RMS_train, precision=2)))
    ax1.plot(RMSE_train, color='m', linestyle='-',label= "RMSE:{}".format(np.format_float_scientific(RMSE_train, precision=2)))
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax1.set_ylabel("Centroid")
    ax1.set_title("Train")
    ax1.grid(axis="x")
    ax1.grid(axis="y")
    ax1.set_ylim((minimo, massimo))
    ax1.legend()
    ax2.plot(np.squeeze(Y_train[:r]) - train_forecast[:r], "b", label= "Label - Prediction")
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))   
    ax2.set_ylabel("Label - Prediction")
    ax2.set_xlabel("Time")
    ax2.grid(axis="x")
    ax2.grid(axis="y")
    ax2.set_ylim((-(massimo-minimo)/2, (massimo-minimo)/2))
    ax3.plot(np.squeeze(Y_test),"r",label= "Label")
    ax3.plot(test_forecast,"k",label= "Prediction")
    ax3.plot(RMS_test, color='g', linestyle='-',label= "RMS:{}".format(np.format_float_scientific(RMS_test, precision=2)))
    ax3.plot(RMSE_test, color='m', linestyle='-',label= "RMSE:{}".format(np.format_float_scientific(RMSE_test, precision=2)))
    ax3.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax3.set_ylabel("Centroid")
    ax3.set_xlabel("Time")
    ax3.set_title("Test")
    ax3.grid(axis="x")
    ax3.grid(axis="y")
    ax3.set_ylim((minimo, massimo))
    ax3.legend()
    ax4.plot(np.squeeze(Y_test) - test_forecast, "b", label= "Label - Prediction")
    ax4.ticklabel_format(axis="y", style="sci", scilimits=(0,0))   
    ax4.set_ylabel("|Label - Prediction|")
    ax4.set_xlabel("Time")
    ax4.grid(axis="x")
    ax4.grid(axis="y")
    ax4.set_ylim((-(massimo-minimo)/2, (massimo-minimo)/2)) 
    save_name = "plot3_" +str(i) + ".png"
    plt.savefig(save_name)
    return 
def plotting_4(train_forecast,Y_train,test_forecast,Y_test,i):
    # massimo = max(np.max(Y_train),np.max(train_forecast),np.max(Y_test),np.max(test_forecast))
    # minimo = min(np.min(Y_train),np.min(train_forecast),np.min(Y_test),np.min(test_forecast))
    
    Y_train = np.squeeze(Y_train)
    Y_test = np.squeeze(Y_test)

    start = 0
    stop = 60
    Rolling_Y_train = []
    Rolling_train_forecast =[]
    for j in range(0,len(Y_train)-60):
        Rolling_Y_train.append(np.std(Y_train[start:stop]))
        Rolling_train_forecast.append(np.std(Y_train[start:stop] - train_forecast[start:stop]))
        start += 1
        stop += 1
    Rmean_Y_train = np.mean(Rolling_Y_train)
    Rmean_train_forecast = np.mean(Rolling_train_forecast)


    start = 0
    stop = 60
    Rolling_Y_test = []
    Rolling_test_forecast = []
    for j in range(0,len(Y_test)-60):
        Rolling_Y_test.append(np.std(Y_test[start:stop]))
        Rolling_test_forecast.append(np.std(Y_test[start:stop] - test_forecast[start:stop]))
        start += 1
        stop += 1
    Rmean_Y_test = np.mean(Rolling_Y_test)
    Rmean_test_forecast = np.mean(Rolling_test_forecast)
    
    fig, (ax1,ax2) = plt.subplots(2,figsize=(16,6))
    ax1.plot(Rolling_Y_train,"r",label= "Rolling STD -- Mean:{}".format(np.format_float_scientific(Rmean_Y_train, precision=2)))
    ax1.plot(Rolling_train_forecast,"k",label= "Rolling STD (Label - Prediction) -- Mean:{}".format(np.format_float_scientific(Rmean_train_forecast, precision=2)))
    ax1.hlines(y=np.std(Y_train), xmin=0, xmax=len(Rolling_Y_train), color='orange', label= "STD :{}".format(np.format_float_scientific(np.std(Y_train), precision=2)))
    ax1.hlines(y=np.std(Y_train - train_forecast), xmin=0, xmax=len(Rolling_train_forecast), color='gray', label= "STD  (Label - Prediction):{}".format(np.format_float_scientific(np.std(Y_train - train_forecast), precision=2)))
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))   

    ax1.set_ylabel("STD")
    ax1.set_title("TRAIN")
    ax1.grid(axis="x")
    ax1.grid(axis="y")
    ax1.legend()


    ax2.plot(Rolling_Y_test,"r",label= "Rolling STD  -- Mean:{}".format(np.format_float_scientific(Rmean_Y_test, precision=2)))
    ax2.plot(Rolling_test_forecast,"k",label= "Rolling STD  (Label - Prediction) -- Mean:{}".format(np.format_float_scientific(Rmean_test_forecast, precision=2)))
    ax2.hlines(y=np.std(Y_test), xmin=0, xmax=len(Rolling_Y_test), color='orange', label= "STD :{}".format(np.format_float_scientific(np.std(Y_test), precision=2)))
    ax2.hlines(y=np.std(Y_test - test_forecast), xmin=0, xmax=len(Rolling_test_forecast), color='grey', label= "STD  (Label - Prediction):{}".format(np.format_float_scientific(np.std(Y_test - test_forecast), precision=2)))
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))   

    ax2.set_ylabel("STD")
    ax2.set_title("TEST")
    ax2.grid(axis="x")
    ax2.grid(axis="y")
    ax2.set_xlabel("Acq point)")
    ax2.legend()
    
    save_name = "plot4_" +str(i) + ".png"
    plt.savefig(save_name)
    return 

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

def root_mean_squared_error(y_true, y_pred):
    print(type(y_true))
    print(type(y_pred))
    print("y_pred:",y_pred)
    print("y_true:",y_true)
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
def lstm():
    inputs = tf.keras.layers.Input(shape=np.shape(X_train)[-2:])
    x = tf.keras.layers.LSTM(units=300, return_sequences=False)(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(forecast_horizon)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model
def mlp():
    inputs = tf.keras.layers.Input(shape=np.shape(X_train)[-2:])
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(forecast_horizon)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

def cnn():
    inputs = tf.keras.layers.Input(shape=np.shape(X_train)[-2:])
    x = tf.keras.layers.Conv1D(32, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)    
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)        
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(forecast_horizon)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


def testing(X_test, past_history, FullDataset):
    test_forecast = []
    X_test2 = X_test.reshape(X_test.shape[0], past_history, len(FullDataset))        
    for k in range(len(X_test)):        
        if (len(test_forecast)<past_history):
            if (len(test_forecast)==0):
                n_value_neede = past_history
                x_cam_first =  X_test2[k, :n_value_neede, 0]
                x_cam = x_cam_first 
            else:
                x_cam_last = np.squeeze(np.array(test_forecast[-len(test_forecast):]),1)
                n_value_neede = past_history - len(test_forecast)
                x_cam_first =  X_test2[k, :n_value_neede, 0]
                x_cam_last = np.array(x_cam_last).reshape(-1,)
                x_cam = np.concatenate((x_cam_first,x_cam_last),axis=0) 
            x_external = X_test2[k,:,1:]
            new_X_test = np.concatenate((np.expand_dims(x_cam,1),x_external),axis=1)
            new_X_test = new_X_test.reshape(1, new_X_test.shape[0] , new_X_test.shape[1])
            local_forecast = model(new_X_test)
            test_forecast.append(local_forecast)
        else:
            x_cam = np.squeeze(np.array(test_forecast[-past_history:]),1)
            x_external = X_test2[k,:,1:]
            new_X_test = np.concatenate((x_cam,x_external),axis=1)
            new_X_test = new_X_test.reshape(1, new_X_test.shape[0] , new_X_test.shape[1])
            local_forecast = model(new_X_test)
            test_forecast.append(local_forecast)
    return test_forecast

if __name__ == "__main__":
    t = time.time()
    percentage = 80 
    fit_or_proj = "fit" 
    past_history = 60
    forecast_horizon = 1
    normalization_method = 'zscore'
    fetureSelection = 0

    #InLoop
    OL_phs,OL_amp,ILmOL_phs,ILmOL_amp,laser_Phs,laser_amp,Egain,cam = loading(InOrOut = "OutLoop", both = False, targetShift = -2)
    if fetureSelection:
        FullDataset = np.array([cam, OL_amp, OL_phs]) 
    else:
        FullDataset = np.array([cam, OL_phs, OL_amp, ILmOL_phs, ILmOL_amp, laser_Phs, laser_amp])

    splitting_traintest = 1
    traintest_size = len(cam)//splitting_traintest
    split = int(traintest_size*percentage/100)

    rms_train = []
    rms_test = []
    rmse_train = []
    rmse_test = []
    for i in range(splitting_traintest):
        # Splitting dataset
        start_train = traintest_size*i
        stop_train = traintest_size*i + split
        start_test = stop_train + 1
        stop_test = traintest_size*(i+1)-1
        print("Train start--stop:",start_train,"--",stop_train)
        print("Test start--stop:",start_test,"--",stop_test)
        print("")
        train, test = FullDataset[:,start_train:stop_train], FullDataset[:,start_test:stop_test]
        # Normalization
        train, test, norm_params = normalize_dataset(train, test, normalization_method, dtype="float64")
        # Windowing
        X_train, Y_train = windows_preprocessing(train, past_history, forecast_horizon)
        X_test, Y_test = windows_preprocessing(test, past_history, forecast_horizon)
  
        # Model
        
        print("X_train",type(X_train))
        print("X_train",np.shape(X_train))
        print("Y_train",type(Y_train))
        print("Y_train",np.shape(Y_train))
        model = lstm()
        # model = mlp()
        # model = cnn()

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss = root_mean_squared_error)  
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50) 
        train_forecast_pre = model(X_train).numpy()
        history = model.fit(X_train,Y_train,batch_size=64, epochs=100, validation_split=0.1, shuffle=True)
        # Test
        train_forecast = model(X_train).numpy()
        test_forecast = testing(X_test = X_test, past_history = past_history, FullDataset = FullDataset)
        # Denormalize
        train_forecast, metrics_train, Y_train_denorm= denormalization(forecast=train_forecast, X=X_train, Y = Y_train, norm_params= norm_params, normalization_method= normalization_method)
        test_forecast, metrics_test, Y_test_denorm= denormalization(forecast=test_forecast, X=X_test, Y = Y_test, norm_params= norm_params, normalization_method= normalization_method)
        # Metrics
        Y_train_denorm = np.squeeze(Y_train_denorm)
        Y_test_denorm = np.squeeze(Y_test_denorm)
        diff_train = np.abs(Y_train_denorm - train_forecast)
        diff_test = np.abs(Y_test_denorm - test_forecast)

        rms_train.append(np.sqrt(np.mean(Y_train_denorm**2)))
        rms_test.append(np.sqrt(np.mean(Y_test_denorm**2)))
        rmse_train.append(np.sqrt(np.mean((diff_train)**2)))
        rmse_test.append(np.sqrt(np.mean((diff_test)**2)))

        # Plotting
        plotting_1(history,i)
        plotting_2(train_forecast,Y_train_denorm,test_forecast,Y_test_denorm,i)
        plotting_3(train_forecast,Y_train_denorm,test_forecast,Y_test_denorm,i, r = 200)
        plotting_4(train_forecast,Y_train_denorm,test_forecast,Y_test_denorm,i)


    #Printing
    sci_rms_train = [np.format_float_scientific(m, precision=2) for m in rms_train]
    sci_rms_test = [np.format_float_scientific(m, precision=2) for m in rms_test]
    sci_rmse_train = [np.format_float_scientific(m, precision=2) for m in rmse_train]
    sci_rmse_test = [np.format_float_scientific(m, precision=2) for m in rmse_test]
    
    print("")
    print("-------------------------------------- TRAIN --------------------------------------")
    print("RMS Label ----------------------------------->",sci_rms_train)
    print("RMSE  --------------------------------------->",sci_rmse_train)
    print("-----------------------------------------------------------------------------------")
    print("")
    print("")
    print("--------------------------------------- TEST ---------------------------------------")
    print("RMS Label ----------------------------------->",sci_rms_test)
    print("RMSE ---------------------------------------->",sci_rmse_test)
    print("-----------------------------------------------------------------------------------")
    elapsed = time.time() - t
    print("TIME:",elapsed)
