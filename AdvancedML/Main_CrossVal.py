import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit


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

def denormalize(data, norm_params, method="zscore"):
    """
    Reverse normalization time series
    :param data: normalized time series
    :param norm_params: tuple with params mean, std, max, min
    :param method: zscore or minmax
    :return: time series in original scale
    """
    assert method in ["zscore", "minmax", "None"]

    if method == "zscore":
        return (data * norm_params["std"]) + norm_params["mean"]

    elif method == "minmax":
        return data * (norm_params["max"] - norm_params["min"]) + norm_params["min"]

    elif method == "None":
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


## Loading Files 
InOrOut = "OutLoop"
#InOrOut = "InLoop" 
camFit = np.load("data/"+InOrOut+"/CameraFit.npy")    #Y
camProj = np.load("data/"+InOrOut+"/CameraProj.npy")  #Y
OL_phs = np.load("data/"+InOrOut+"/OL_Phase.npy")     #X
OL_amp = np.load("data/"+InOrOut+"/OL_Magnitude.npy") #X
ILmOL_phs = np.load("data/"+InOrOut+"/OL_Phase.npy") - np.load("data/"+InOrOut+"/IL_Phase.npy") #X -- OT USED  
ILmOL_amp = np.load("data/"+InOrOut+"/OL_Magnitude.npy") - np.load("data/"+InOrOut+"/IL_Magnitude.npy") #X
laser_Phs = np.load("data/"+InOrOut+"/Laser_Phs.npy")  #X -- NOT USED  
laser_amp = np.load("data/"+InOrOut+"/Laser_Amp.npy")  #X

##SplittingRatio ML 
percentage = 80 #-- Train
fit_or_proj = "fit"
if (fit_or_proj == 'fit'):
    split = int(np.shape(camFit)[0]*percentage/100)
else:
    split = int(np.shape(camProj)[0]*percentage/100)  
past_history = 60
forecast_horizon = 1
targetShift = -2
fetureSelection = 0
normalization_method = 'minmax'

tscv = TimeSeriesSplit(gap=0, n_splits=5)

# SHIFT
if (targetShift!=0):
    data_ln = len(camFit)
    OL_phs_shift = np.zeros([data_ln])
    OL_amp_shift = np.zeros([data_ln])
    ILmOL_phs_shift = np.zeros([data_ln])
    ILmOL_amp_shift = np.zeros([data_ln])
    laser_Phs_shift = np.zeros([data_ln])
    laser_amp_shift = np.zeros([data_ln])
    OL_phs_shift = np.roll(OL_phs,targetShift)
    OL_amp_shift = np.roll(OL_amp,targetShift)
    ILmOL_phs_shift = np.roll(ILmOL_phs,targetShift)
    ILmOL_amp_shift = np.roll(ILmOL_amp,targetShift)
    laser_Phs_shift = np.roll(laser_Phs,targetShift)
    laser_amp_shift = np.roll(laser_amp,targetShift)
    camFit_2  = camFit[:targetShift]
    camProj_2 = camProj[:targetShift]
    OL_phs_2 = OL_phs_shift[:targetShift]
    OL_amp_2 = OL_amp_shift[:targetShift]
    ILmOL_phs_2 = ILmOL_phs_shift[:targetShift]
    ILmOL_amp_2 = ILmOL_amp_shift[:targetShift]
    laser_Phs_2 = laser_Phs_shift[:targetShift]
    laser_amp_2 = laser_amp_shift[:targetShift]
    if (fit_or_proj == 'fit'):
        cam_train, cam_test = camFit_2[:split], camFit_2[split:]
    else:
        cam_train, cam_test = camProj_2[:split], camProj_2[split:]
    phs_train, phs_test = OL_phs_2[:split], OL_phs_2[split:]
    amp_train, amp_test = OL_amp_2[:split], OL_amp_2[split:]
    diffPhs_train, diffPhs_test = ILmOL_phs_2[:split], ILmOL_phs_2[split:]
    diffAmp_train, diffAmp_test = ILmOL_amp_2[:split], ILmOL_amp_2[split:]
    Lphs_train, Lphs_test = laser_Phs_2[:split], laser_Phs_2[split:]
    Lamp_train, Lamp_test = laser_amp_2[:split], laser_amp_2[split:]
else:
    #WITHOUT SHIFT
    if (fit_or_proj == 'fit'):
        cam_train, cam_test = camFit[:split], camFit[split:]
    else:
        cam_train, cam_test = camProj[:split], camProj[split:] 
    phs_train, phs_test = OL_phs[:split], OL_phs[split:]
    amp_train, amp_test = OL_amp[:split], OL_amp[split:]
    diffPhs_train, diffPhs_test = ILmOL_phs[:split], ILmOL_phs[split:]
    diffAmp_train, diffAmp_test = ILmOL_amp[:split], ILmOL_amp[split:]
    Lphs_train, Lphs_test = laser_Phs[:split], laser_Phs[split:]
    Lamp_train, Lamp_test = laser_amp[:split], laser_amp[split:]

if fetureSelection:
    # Selected variables
    train = np.array([cam_train, phs_train, amp_train, diffAmp_train, Lamp_train])
    test = np.array([cam_test, phs_test, amp_test, diffAmp_test, Lamp_test])
else:
    # All variables
    train = np.array([cam_train, phs_train, amp_train, diffPhs_train, diffAmp_train, Lphs_train, Lamp_train]) 
    test = np.array([cam_test, phs_test, amp_test, diffPhs_test, diffAmp_test, Lphs_test, Lamp_test])


FullDataset = np.concatenate(((train,test)),axis=1)

metric_train = []
metric_test = []
print(list(tscv.split(FullDataset.transpose((1,0)))))
for train_index, test_index in tscv.split(FullDataset.transpose((1,0))):
    #print("TRAIN:", train_index, "TEST:", test_index)
    train, test = FullDataset[:,train_index], FullDataset[:,test_index]
    train, test, norm_params = normalize_dataset(train, test, normalization_method, dtype="float64")
    X_train, Y_train = windows_preprocessing(train, past_history, forecast_horizon)
    X_test, Y_test = windows_preprocessing(test, past_history, forecast_horizon)

    #Reshape for RF model -- No temporal dimenison 
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

    #Performing the ML
    # #model = RandomForestRegressor(criterion='mse', n_jobs=-1, n_estimators=100,max_depth=6,min_samples_split=6,min_samples_leaf=7)
    # #model = LinearRegression()
    model = MLPRegressor(hidden_layer_sizes=[32,16,8], random_state=1, max_iter=500)
    model.fit(X_train,Y_train)
    train_forecast = model.predict(X_train)
    Y_train_denorm = np.zeros(Y_train.shape)
    for i in range(Y_train.shape[0]):
        nparams = norm_params[0]
        train_forecast[i] = denormalize(train_forecast[i], nparams, normalization_method)
        Y_train_denorm[i] = denormalize(Y_train[i], nparams, normalization_method)
    train_forecast = np.squeeze(train_forecast)
    metrics_train = np.abs(train_forecast-np.squeeze(Y_train_denorm))
    test_forecast = model.predict(X_test)
    Y_test_denorm = np.zeros(Y_test.shape)
    for i in range(Y_test.shape[0]):
        nparams = norm_params[0]
        test_forecast[i] = denormalize(test_forecast[i], nparams, normalization_method)
        Y_test_denorm[i] = denormalize(Y_test[i], nparams, normalization_method)
    test_forecast = np.squeeze(test_forecast)
    metrics_test = np.abs(test_forecast-np.squeeze(Y_test_denorm))

    metric_train.append(mean_squared_error(Y_train_denorm,train_forecast, squared=False))

    metric_test.append(mean_squared_error(Y_test_denorm, test_forecast, squared=False))

metric_train = [np.format_float_scientific(m, precision=2) for m in metric_train]
print("Training RMSE",metric_train)
metric_test = [np.format_float_scientific(m, precision=2) for m in metric_test]
print("Test RMSE",metric_test)

# #Plotting
# fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,6),sharey=True)
# ax1.plot(metrics_train,"+k")
# ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# ax1.hlines(np.mean(metrics_train), 0, len(metrics_train), colors='r', linestyles='--')
# ax1.set_ylabel("Metrics")
# ax1.set_xlabel("Samples")
# ax1.set_title("Train")
# ax2.plot(metrics_test,"+k")
# ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# ax2.hlines(np.mean(metrics_test), 0, len(metrics_test), colors='r', linestyles='--')
# ax2.set_xlabel("Samples")
# ax2.set_title("Test")
# plt.show()

# fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,6),sharey=True)
# ax1.plot(train_forecast,np.squeeze(Y_train),"+k")
# ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# ax1.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
# ax1.set_ylabel("Label")
# ax1.set_xlabel("Prediction")
# ax1.set_title("Train")
# ax1.grid(axis="x")
# ax1.grid(axis="y")
# ax2.plot(test_forecast,np.squeeze(Y_test),"+k")
# ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# ax2.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
# ax2.set_xlabel("Prediction")
# ax2.set_title("Test")
# ax2.grid(axis="x")
# ax2.grid(axis="y")
# plt.show()

# massimo = max(np.max(Y_train_denorm),np.max(train_forecast),np.max(Y_test_denorm),np.max(test_forecast))
# minimo = min(np.min(Y_train_denorm),np.min(train_forecast),np.min(Y_test_denorm),np.min(test_forecast))

# fig, (ax1,ax2) = plt.subplots(2,figsize=(16,6))
# ax1.plot(np.squeeze(Y_train_denorm),"r",label= "Label")
# ax1.plot(train_forecast,"k",label= "Prediction")
# ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# ax1.set_ylabel("Centroid Error")
# ax1.set_title("Train")
# ax1.grid(axis="x")
# ax1.grid(axis="y")
# ax1.set_ylim((minimo, massimo))
# ax1.legend()
# ax2.plot(np.squeeze(Y_test_denorm),"r",label= "Label")
# ax2.plot(test_forecast,"k",label= "Prediction")
# ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# ax2.set_ylabel("Centroid Error")
# ax2.set_xlabel("Time")
# ax2.set_title("Test")
# ax2.grid(axis="x")
# ax2.grid(axis="y")
# ax2.set_ylim((minimo, massimo))
# ax2.legend()
# plt.show()