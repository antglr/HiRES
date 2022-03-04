import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
import scipy.io as sio
from xgboost import XGBRegressor
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
    save_name = "plot2_" +str(i) + ".png"
    plt.savefig(save_name)
    return 

def plotting_3(train_forecast,Y_train,test_forecast,Y_test,i):
    massimo = max(np.max(Y_train),np.max(train_forecast),np.max(Y_test),np.max(test_forecast))
    minimo = min(np.min(Y_train),np.min(train_forecast),np.min(Y_test),np.min(test_forecast))

    fig, (ax1,ax2,ax3) = plt.subplots(3,figsize=(16,6))
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
    ax3.plot(np.abs(np.squeeze(Y_test)-test_forecast),"k")
    ax3.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    M = np.mean(np.abs(np.squeeze(Y_test)-test_forecast))
    ax3.axhline(y=M, color='k', linestyle='-',label= str(M))
    ax3.set_ylabel("|Label-Prediction|")
    ax3.set_xlabel("Time")
    ax3.grid(axis="x")
    ax3.grid(axis="y")
    ax3.legend()
    plt.suptitle('SubSet -->{}'.format(i+1), fontsize=20)
    save_name = "plot3_" +str(i) + ".png"
    plt.savefig(save_name)
    return 

def plot_feature_importances(model, feature_names, X_train, past_history):
    feat_names = [ feature_names[int(i%len(feature_names))] + '_' + 
                  str(int(i/len(feature_names))) for i in range(X_train.shape[1]) ]
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feat_names)
    fig, ax = plt.subplots(figsize=(60,10))
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    # fig.tight_layout()
    plt.savefig("feats_ph{}_bis.png".format(past_history))

def loading(InOrOut):
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
    return OL_phs,OL_amp,ILmOL_phs,ILmOL_amp,laser_Phs,laser_amp,Egain,cam


#InLoop
OL_phs,OL_amp,ILmOL_phs,ILmOL_amp,laser_Phs,laser_amp,Egain,cam = loading(InOrOut = "OutLoop")
percentage = 80 
fit_or_proj = "fit" 
past_history = 30
forecast_horizon = 1
normalization_method = 'zscore'
targetShift = -2
fetureSelection = 1
shouldIplot = 1

loadboth = 0

data_ln = len(cam)
# SHIFT
if (targetShift!=0):
    cam = np.roll(cam,targetShift)
    cam  = cam[:targetShift]
    OL_phs = OL_phs[:targetShift]
    OL_amp = OL_amp[:targetShift]
    ILmOL_phs = ILmOL_phs[:targetShift]
    ILmOL_amp = ILmOL_amp[:targetShift]
    laser_Phs = laser_Phs[:targetShift]
    laser_amp = laser_amp[:targetShift]
    Egain = Egain[:targetShift]

if loadboth:
    OL_phs1,OL_amp1,ILmOL_phs1,ILmOL_amp1,laser_Phs1,laser_amp1,Egain1,cam1 = loading(InOrOut = "InLoop")
    OL_phs = np.concatenate((OL_phs, OL_phs1), axis = 0)
    OL_amp = np.concatenate((OL_amp, OL_amp1), axis = 0)
    ILmOL_phs = np.concatenate((ILmOL_phs, ILmOL_phs1), axis = 0)
    ILmOL_amp = np.concatenate((ILmOL_amp, ILmOL_amp1), axis = 0)
    laser_Phs = np.concatenate((laser_Phs, laser_Phs1), axis = 0)
    laser_amp = np.concatenate((laser_amp, laser_amp1), axis = 0)
    Egain = np.concatenate((Egain, Egain1), axis = 0)
    cam = np.concatenate((cam, cam1), axis = 0)
    data_ln = len(cam)

if fetureSelection:
    # Selected variables
    FullDataset = np.array([cam, OL_amp, OL_phs]) 
else:
    # All variables
    FullDataset = np.array([cam, OL_phs, OL_amp, ILmOL_phs, ILmOL_amp, laser_Phs, laser_amp])

# feature_names = ['Cam', 'RF P', 'RF A', 'RF PD', 'RF AD', 'Laser P', 'Laser A']
feature_names = ['Cam',  'RF A', 'RF P']



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
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
    
    #model = RandomForestRegressor(criterion='mse', n_jobs=-1, n_estimators=100, max_depth=10,min_samples_split=2,min_samples_leaf=3)
    # model = XGBRegressor(n_estimators=100, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8, n_jobs=-1)
    # model = LinearRegression()
    model = MLPRegressor(hidden_layer_sizes=[256, 128, 64, 32, 16, 8], random_state=1, max_iter=1000)
    model.fit(X_train,Y_train)
    train_forecast = model.predict(X_train)
    # plot_feature_importances(model, feature_names, X_train, past_history)

    

    Y_train_denorm = np.zeros(Y_train.shape)
    for j in range(Y_train.shape[0]):
        nparams = norm_params[0]
        train_forecast[j] = denormalize(train_forecast[j], nparams, normalization_method)
        Y_train_denorm[j] = denormalize(Y_train[j], nparams, normalization_method)
    train_forecast = np.squeeze(train_forecast)
    metrics_train = np.abs(train_forecast-np.squeeze(Y_train_denorm))

    #test_forecast = model.predict(X_test)
    test_forecast = []
    X_test2 = X_test.reshape(X_test.shape[0], past_history, len(FullDataset))
    for k in range(len(X_test)):        
        if (len(test_forecast)<past_history):
            if (len(test_forecast)==0):
                n_value_neede = past_history
                x_cam_first =  X_test2[k, :n_value_neede, 0]
                x_cam = x_cam_first #Check concatenation axis
            else:
                x_cam_last = np.squeeze(np.array(test_forecast[-len(test_forecast):]),1)
                n_value_neede = past_history - len(test_forecast)
                x_cam_first =  X_test2[k, :n_value_neede, 0]
                x_cam = np.concatenate((x_cam_first,x_cam_last),axis=0) #Check concatenation axis
            x_external = X_test2[k,:,1:]
            new_X_test = np.concatenate((np.expand_dims(x_cam,1),x_external),axis=1)
            new_X_test = new_X_test.reshape(1, new_X_test.shape[0] * new_X_test.shape[1])
            local_forecast = model.predict(new_X_test)
            test_forecast.append(local_forecast)
        else:
            x_cam = np.squeeze(np.array(test_forecast[-past_history:]),1)
            x_external = X_test2[k,:,1:]
            new_X_test = np.concatenate((np.expand_dims(x_cam,1),x_external),axis=1)
            new_X_test = new_X_test.reshape(1, new_X_test.shape[0] * new_X_test.shape[1])
            local_forecast = model.predict(new_X_test)
            test_forecast.append(local_forecast)

    Y_test_denorm = np.zeros(Y_test.shape)
    for j in range(Y_test.shape[0]):
        nparams = norm_params[0]
        test_forecast[j] = denormalize(test_forecast[j], nparams, normalization_method)
        Y_test_denorm[j] = denormalize(Y_test[j], nparams, normalization_method)
    test_forecast = np.squeeze(test_forecast)
    metrics_test = np.abs(test_forecast-np.squeeze(Y_test_denorm))

    metric_train.append([mean_squared_error(Y_train_denorm, train_forecast, squared=False), mean_absolute_percentage_error(Y_train_denorm, train_forecast)])

    metric_test.append([mean_squared_error(Y_test_denorm, test_forecast, squared=False), mean_absolute_percentage_error(Y_test_denorm, test_forecast)])

    plotting_2(train_forecast,Y_train_denorm,test_forecast,Y_test_denorm,i)
    plotting_3(train_forecast,Y_train_denorm,test_forecast,Y_test_denorm,i)
    if shouldIplot:
        plt.show()

m_train = [np.format_float_scientific(m[0], precision=2) for m in metric_train]
print("Training RMSE",m_train)
m_test = [np.format_float_scientific(m[0], precision=2) for m in metric_test]
print("Test RMSE",m_test)

m_train = [m[1]*100 for m in metric_train]
print("Training MAPE",m_train)
m_test = [m[1]*100 for m in metric_test]
print("Test MAPE",m_test)

