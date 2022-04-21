from calendar import c
from operator import length_hint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import time
import scipy.io
import click
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import scipy.io as sio
import pandas as pd
import time

def clrscr():
    click.clear()
def _check_keys( dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict
def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict
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
    dict = dictionary["syncData"]
    
    x_Laser = dict["LCam1_Gauss"][:,0]
    
    xRMS_Laser = dict["LCam1_Gauss"][:,1]
    y_Laser = dict["LCam1_Gauss"][:,2]
    yRMS_Laser = dict["LCam1_Gauss"][:,3]
    u_Laser = dict["LCam1_Gauss"][:,4]
    uRMS_Laser = dict["LCam1_Gauss"][:,5]
    v_Laser = dict["LCam1_Gauss"][:,6]
    vRMS_Laser = dict["LCam1_Gauss"][:,7]
    sum_Laser = dict["LCam1_Gauss"][:,8]
    rf_amp = dict["Cav_Amp"]
    rf_phs = dict["Cav_Phs"]
    fw2_amp =dict["Fwd2_Amp"] 
    fw2_phs =dict["Fwd2_Phs"] 
    rv_amp = dict["Rev_Amp"]
    rv_phs = dict["Rev_Phs"]
    fw1_amp = dict["Fwd1_Amp"]
    fw1_phs = dict["Fwd1_Phs"]
    laser_amp = dict["LP_Amp"]
    laser_phs = dict["LP_Phase"]
    
    cam = dict["AdjUCam1Pos"]
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
        window_ts[0,-1] = np.mean(window_ts[0,:-1]) # Remove (Zero) the value of the cam to be guessed -- To improve!!!!!!!
        window = np.array(window_ts).transpose((1,0))
        x.append(window)
        
        y.append(camera_series[j: j + 1])
    return np.array(x), np.array(y)

def testing(X_test, past_history, model):
    test_forecast = []
    lastguess = 0
    X_test_copy = X_test.copy()
    for k in range(past_history, np.shape(X_test)[1]):        
        indices = list(range(k - past_history, k+1))
        
        X_test_copy[0,indices[-1]] = lastguess
        X_test_new = X_test_copy[:, indices].copy()
        #X_test_new[0,-1] = lastguess 
        X_test_new = np.array(X_test_new).transpose((1,0))
         
        X_test_new = X_test_new.reshape(1,-1)
        lastguess = model.predict(X_test_new)
        test_forecast.append(lastguess)
    return test_forecast

if __name__ == "__main__":
    clrscr()
    t = time.time()
    
    past_history = 20
    normalization_method = 'minmax'
    #"zscore", "minmax",
    fetureSelection = 0
    plt_len = 2000

    filname = "new_dataset/Fourth_Dataset/OpenLoop1postp.mat" #-->OpenLoop
    #filname = "new_dataset/Fourth_Dataset/ClosedLoop1postp.mat" #-->CloseLoop
    
    dict = loadmat(filname)
    x_Laser, xRMS_Laser, y_Laser, yRMS_Laser, u_Laser, uRMS_Laser, v_Laser, vRMS_Laser, sum_Laser, rf_amp, rf_phs, fw2_amp, fw2_phs, rv_amp, rv_phs, fw1_amp, fw1_phs, laser_amp, laser_phs, cam = to_dataframe(dict) 
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
    X_train= X_train.reshape(X_train.shape[0], -1)
    model = MLPRegressor(hidden_layer_sizes=(64,64,64,64,148), solver='adam',  learning_rate='constant', learning_rate_init=0.01,  
                        activation="relu" ,random_state=42, max_iter=2000, shuffle=True, early_stopping=True, 
                        validation_fraction=0.1, n_iter_no_change=30, verbose=0).fit(X_train, Y_train)    
    train_forecast = model.predict(X_train)
    plt.figure()
    plt.plot(train_forecast[:plt_len], label="pred")
    plt.plot(Y_train[:plt_len],label="gt")
    plt.legend()
    plt.savefig("p1.png")
    
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
    
    plt.figure()
    plt.plot(test_forecast[:plt_len], label="pred")
    plt.plot(Y_test[:plt_len],label="gt")
    plt.legend()
    plt.savefig("p2.png")
    
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

    elapsed = time.time() - t
    print("TIME:",elapsed)
################################################################################################################
################################################################################################################
################################################################################################################
   
   
# def testing(X_test, past_history, FullDataset, model):
#     test_forecast = []
#     X_test2 = X_test.reshape(X_test.shape[0], past_history, len(FullDataset))        
#     for k in range(len(X_test)):        
#         if (len(test_forecast)<past_history):
#             if (len(test_forecast)==0):
#                 n_value_neede = past_history
#                 x_cam_first =  X_test2[k, :n_value_neede, 0]
#                 x_cam = x_cam_first 
#             else:
#                 x_cam_last = np.squeeze(np.array(test_forecast[-len(test_forecast):]),1)
#                 n_value_neede = past_history - len(test_forecast)
#                 x_cam_first =  X_test2[k, :n_value_neede, 0]
#                 x_cam_last = np.array(x_cam_last).reshape(-1,)
#                 x_cam = np.concatenate((x_cam_first,x_cam_last),axis=0) 
#             x_external = X_test2[k,:,1:]
#             new_X_test = np.concatenate((np.expand_dims(x_cam,1),x_external),axis=1)
#             if "sklearn" in str(type(model)):
#                 new_X_test = new_X_test.reshape(1, new_X_test.shape[0] * new_X_test.shape[1])
#             else:
#                 new_X_test = new_X_test.reshape(1, new_X_test.shape[0] , new_X_test.shape[1])
            
#             local_forecast = model.predict(new_X_test)
#             test_forecast.append(local_forecast)
#         else:
#             x_cam = np.squeeze(np.array(test_forecast[-past_history:]),1)
#             x_external = X_test2[k,:,1:]
#             new_X_test = np.concatenate((x_cam,x_external),axis=1)
#             if "sklearn" in str(type(model)):
#                 new_X_test = new_X_test.reshape(1, new_X_test.shape[0] * new_X_test.shape[1])
#             else:
#                 new_X_test = new_X_test.reshape(1, new_X_test.shape[0] , new_X_test.shape[1])
#             local_forecast = model.predict(new_X_test)
#             test_forecast.append(local_forecast)
#     return test_forecast
 
# # Deep Learning
    # # model = lstm()
    # # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')  
    # # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50) 
    # # # history = model.fit(X_train,Y_train,batch_size=64, epochs=100, validation_split=(0.1), shuffle=True)
    # # history = model.fit(X_train,Y_train,batch_size=64, epochs=20, validation_data=(X_test,Y_test), shuffle=True)
    

# def windows_preprocessing(time_series, past_history, forecast_horizon):
#     x, y = [], []
#     camera_series = time_series[0]
#     for j in range(past_history, time_series.shape[1] - forecast_horizon + 1, forecast_horizon):
#         indices = list(range(j - past_history, j))

#         window_ts = []
#         for i in range(time_series.shape[0]):
#             window_ts.append(time_series[i, indices])
#         window = np.array(window_ts).transpose((1,0))

#         x.append(window)
#         y.append(camera_series[j: j + forecast_horizon])
# #     return np.array(x), np.array(y)
# def windows_preprocessing2(time_series, past_history, forecast_horizon):
#     x, y = [], []
#     camera_series = time_series[0]
#     time_series = time_series[1:] # This line removes cam from the X
#     for j in range(past_history, time_series.shape[1] - forecast_horizon + 1, forecast_horizon):
#         indices = list(range(j - past_history, j+1))

#         window_ts = []
#         for i in range(time_series.shape[0]):
#             window_ts.append(time_series[i, indices])
#         window = np.array(window_ts).transpose((1,0))

#         x.append(window)
#         y.append(camera_series[j: j + forecast_horizon])
#     return np.array(x), np.array(y)

# def plotting_1(history,i):
#     train_loss = history.history['loss']
#     test_loss = history.history['val_loss']
#     fig, ax = plt.subplots(figsize=(8,6))
#     ax.plot(train_loss,"k",label="Training")
#     ax.plot(test_loss,"r",label="Test")
#     ax.set_xlabel("Epochs")
#     ax.set_ylabel("Loss")
#     plt.legend()
#     plt.savefig("loss_curves.png")
#     save_name = "plot1_" +str(i) + ".png"
#     plt.savefig(save_name)
#     return 
# def plotting_2(train_forecast,Y_train,test_forecast,Y_test,i):
#     fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,6),sharey=True)
#     ax1.plot(train_forecast,np.squeeze(Y_train),"+k")
#     ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#     ax1.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
#     ax1.set_ylabel("Label")
#     ax1.set_xlabel("Prediction")
#     ax1.set_title("Train")
#     ax1.grid(axis="x")
#     ax1.grid(axis="y")
#     ax2.plot(test_forecast,np.squeeze(Y_test),"+k")
#     ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#     ax2.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
#     ax2.set_xlabel("Prediction")
#     ax2.set_title("Test")
#     ax2.grid(axis="x")
#     ax2.grid(axis="y")
#     plt.suptitle('SubSet -->{}'.format(i+1), fontsize=20)
#     save_name = "plot2_" +str(i) + ".png"
#     plt.savefig(save_name)
#     return 
# def plotting_3(train_forecast,Y_train,test_forecast,Y_test,i, r):
#     massimo = max(np.max(Y_train),np.max(train_forecast),np.max(Y_test),np.max(test_forecast))
#     minimo = min(np.min(Y_train),np.min(train_forecast),np.min(Y_test),np.min(test_forecast))

#     diff_train = np.abs(Y_train - train_forecast)
#     diff_test = np.abs(Y_test - test_forecast)
#     RMS_train = np.sqrt(np.mean(Y_train**2))
#     RMSE_train = np.sqrt(np.mean((diff_train)**2))
#     RMS_test = np.sqrt(np.mean(Y_test**2))
#     RMSE_test = np.sqrt(np.mean((diff_test)**2))

#     fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,figsize=(16,6))
#     ax1.plot(np.squeeze(Y_train[:r]),"r",label= "Label")
#     ax1.plot(train_forecast[:r],"k",label= "Prediction")
#     ax1.plot(RMS_train, color='g', linestyle='-',label= "RMS:{}".format(np.format_float_scientific(RMS_train, precision=2)))
#     ax1.plot(RMSE_train, color='m', linestyle='-',label= "RMSE:{}".format(np.format_float_scientific(RMSE_train, precision=2)))
#     ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#     ax1.set_ylabel("Centroid")
#     ax1.set_title("Train")
#     ax1.grid(axis="x")
#     ax1.grid(axis="y")
#     ax1.set_ylim((minimo, massimo))
#     ax1.legend()
#     ax2.plot(np.squeeze(Y_train[:r]) - train_forecast[:r], "b", label= "Label - Prediction")
#     ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))   
#     ax2.set_ylabel("Label - Prediction")
#     ax2.set_xlabel("Time")
#     ax2.grid(axis="x")
#     ax2.grid(axis="y")
#     ax2.set_ylim((-(massimo-minimo)/2, (massimo-minimo)/2))
#     ax3.plot(np.squeeze(Y_test[:r]),"r",label= "Label")
#     ax3.plot(test_forecast[:r],"k",label= "Prediction")
#     ax3.plot(RMS_test, color='g', linestyle='-',label= "RMS:{}".format(np.format_float_scientific(RMS_test, precision=2)))
#     ax3.plot(RMSE_test, color='m', linestyle='-',label= "RMSE:{}".format(np.format_float_scientific(RMSE_test, precision=2)))
#     ax3.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#     ax3.set_ylabel("Centroid")
#     ax3.set_xlabel("Time")
#     ax3.set_title("Test")
#     ax3.grid(axis="x")
#     ax3.grid(axis="y")
#     ax3.set_ylim((minimo, massimo))
#     ax3.legend()
#     ax4.plot(np.squeeze(Y_test[:r]) - test_forecast[:r], "b", label= "Label - Prediction")
#     ax4.ticklabel_format(axis="y", style="sci", scilimits=(0,0))   
#     ax4.set_ylabel("|Label - Prediction|")
#     ax4.set_xlabel("Time")
#     ax4.grid(axis="x")
#     ax4.grid(axis="y")
#     ax4.set_ylim((-(massimo-minimo)/2, (massimo-minimo)/2)) 
#     save_name = "plot3_" +str(i) + ".png"
#     plt.savefig(save_name)
#     return 
# def plotting_4(train_forecast,Y_train,test_forecast,Y_test,i):
#     Y_train = np.squeeze(Y_train)
#     Y_test = np.squeeze(Y_test)
#     start = 0
#     stop = 60
#     Rolling_Y_train = []
#     Rolling_train_forecast =[]
#     for j in range(0,len(Y_train)-60):
#         Rolling_Y_train.append(np.std(Y_train[start:stop]))
#         Rolling_train_forecast.append(np.std(Y_train[start:stop] - train_forecast[start:stop]))
#         start += 1
#         stop += 1
#     Rmean_Y_train = np.mean(Rolling_Y_train)
#     Rmean_train_forecast = np.mean(Rolling_train_forecast)
#     start = 0
#     stop = 60
#     Rolling_Y_test = []
#     Rolling_test_forecast = []
#     for j in range(0,len(Y_test)-60):
#         Rolling_Y_test.append(np.std(Y_test[start:stop]))
#         Rolling_test_forecast.append(np.std(Y_test[start:stop] - test_forecast[start:stop]))
#         start += 1
#         stop += 1
#     Rmean_Y_test = np.mean(Rolling_Y_test)
#     Rmean_test_forecast = np.mean(Rolling_test_forecast)
#     fig, (ax1,ax2) = plt.subplots(2,figsize=(16,6))
#     ax1.plot(Rolling_Y_train,"r",label= "Rolling STD -- Mean:{}".format(np.format_float_scientific(Rmean_Y_train, precision=2)))
#     ax1.plot(Rolling_train_forecast,"k",label= "Rolling STD (Label - Prediction) -- Mean:{}".format(np.format_float_scientific(Rmean_train_forecast, precision=2)))
#     ax1.hlines(y=np.std(Y_train), xmin=0, xmax=len(Rolling_Y_train), color='orange', label= "STD :{}".format(np.format_float_scientific(np.std(Y_train), precision=2)))
#     ax1.hlines(y=np.std(Y_train - train_forecast), xmin=0, xmax=len(Rolling_train_forecast), color='gray', label= "STD  (Label - Prediction):{}".format(np.format_float_scientific(np.std(Y_train - train_forecast), precision=2)))
#     ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))   
#     ax1.set_ylabel("STD")
#     ax1.set_title("TRAIN")
#     ax1.grid(axis="x")
#     ax1.grid(axis="y")
#     ax1.legend()
#     ax2.plot(Rolling_Y_test,"r",label= "Rolling STD  -- Mean:{}".format(np.format_float_scientific(Rmean_Y_test, precision=2)))
#     ax2.plot(Rolling_test_forecast,"k",label= "Rolling STD  (Label - Prediction) -- Mean:{}".format(np.format_float_scientific(Rmean_test_forecast, precision=2)))
#     ax2.hlines(y=np.std(Y_test), xmin=0, xmax=len(Rolling_Y_test), color='orange', label= "STD :{}".format(np.format_float_scientific(np.std(Y_test), precision=2)))
#     ax2.hlines(y=np.std(Y_test - test_forecast), xmin=0, xmax=len(Rolling_test_forecast), color='grey', label= "STD  (Label - Prediction):{}".format(np.format_float_scientific(np.std(Y_test - test_forecast), precision=2)))
#     ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))   
#     ax2.set_ylabel("STD")
#     ax2.set_title("TEST")
#     ax2.grid(axis="x")
#     ax2.grid(axis="y")
#     ax2.set_xlabel("Acq point)")
#     ax2.legend()
#     save_name = "plot4_" +str(i) + ".png"
#     plt.savefig(save_name)
#     return 