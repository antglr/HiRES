import numpy as np
import copy


def read_data(normalization_method, past_history_factor):
    ## Loading Files 
    #InOrOut = "OutLoop"
    InOrOut = "InLoop" 
    # cam = np.load("../data/Camera.npy")
    # phs = np.load("../data/OL_Phase.npy")
    # amp = np.load("../data/OL_Magnitude.npy")
    camFit = np.load("../data/"+InOrOut+"/CameraFit.npy")    
    #camProj = np.load("../data/"+InOrOut+"/CameraProj.npy")  
    OL_phs = np.load("../data/"+InOrOut+"/OL_Phase.npy")     
    OL_amp = np.load("../data/"+InOrOut+"/OL_Magnitude.npy") 
    ILmOL_phs = np.load("../data/"+InOrOut+"/OL_Phase.npy") - np.load("../data/"+InOrOut+"/IL_Phase.npy") 
    ILmOL_amp = np.load("../data/"+InOrOut+"/OL_Magnitude.npy") - np.load("../data/"+InOrOut+"/IL_Magnitude.npy") 
    laser_Phs = np.load("../data/"+InOrOut+"/Laser_Phs.npy")  
    laser_amp = np.load("../data/"+InOrOut+"/Laser_Amp.npy")  

    ##SplittingRatio ML 
    percentage = 80 #-- Train
    split = int(np.shape(camFit)[0]*percentage/100)
    #split = int(np.shape(camProj)[0]*percentage/100)
    forecast_horizon = 1
    targetShift = -2
    fetureSelection = 0

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
        #camProj_2 = camProj[:targetShift]
        OL_phs_2 = OL_phs_shift[:targetShift]
        OL_amp_2 = OL_amp_shift[:targetShift]
        ILmOL_phs_2 = ILmOL_phs_shift[:targetShift]
        ILmOL_amp_2 = ILmOL_amp_shift[:targetShift]
        laser_Phs_2 = laser_Phs_shift[:targetShift]
        laser_amp_2 = laser_amp_shift[:targetShift]
        
        cam_train, cam_test = camFit_2[:split], camFit_2[split:] 
        #cam_train, cam_test = camProj_2[:split], camProj_2[split:]       
        phs_train, phs_test = OL_phs_2[:split], OL_phs_2[split:]
        amp_train, amp_test = OL_amp_2[:split], OL_amp_2[split:]
        IOphs_train, IOphs_test = ILmOL_phs_2[:split], ILmOL_phs_2[split:]
        IOamp_train, IOamp_test = ILmOL_amp_2[:split], ILmOL_amp_2[split:]
        Lphs_train, Lphs_test = laser_Phs_2[:split], laser_Phs_2[split:]
        Lamp_train, Lamp_test = laser_amp_2[:split], laser_amp_2[split:]
    else:
        #WITHOUT SHIFT
        cam_train, cam_test = camFit[:split], camFit[split:]
        #cam_train, cam_test = camProj[:split], camProj[split:] 
        phs_train, phs_test = OL_phs[:split], OL_phs[split:]
        amp_train, amp_test = OL_amp[:split], OL_amp[split:]
        IOphs_train, IOphs_test = ILmOL_phs[:split], ILmOL_phs[split:]
        IOamp_train, IOamp_test = ILmOL_amp[:split], ILmOL_amp[split:]
        Lphs_train, Lphs_test = laser_Phs[:split], laser_Phs[split:]
        Lamp_train, Lamp_test = laser_amp[:split], laser_amp[split:]    

    if fetureSelection:
        # Selected variables
        train = np.array([cam_train, phs_train, amp_train, IOamp_train, Lamp_train])
        test = np.array([cam_test, phs_test, amp_test, IOamp_test, Lamp_test])
    else:
        # All variables
        train = np.array([cam_train, phs_train, amp_train,IOphs_train, IOamp_train, Lphs_train, Lamp_train]) 
        test = np.array([cam_test, phs_test, amp_test,IOphs_test, IOamp_test, Lphs_test, Lamp_test])
    index_target_series = 0     # The index of cam_train, the target variable

    train, test, norm_params = normalize_dataset(
        train, test, normalization_method, dtype="float64"
    )
    
    X_train, Y_train = windows_preprocessing(train, train[index_target_series], past_history_factor, forecast_horizon)
    X_test, Y_test = windows_preprocessing(test, test[index_target_series], past_history_factor, forecast_horizon)
    
    print("TRAINING DATA")
    print("Input shape", X_train.shape)
    print("Output_shape", Y_train.shape)
    print("TEST DATA")
    print("Input shape", X_test.shape)
    print("Output_shape", Y_test.shape)
    
    Y_train_denorm = copy.deepcopy(Y_train)
    for i in range(Y_train.shape[0]):
        Y_train_denorm[i] = denormalize(Y_train[i], norm_params[index_target_series], method=normalization_method)
        
    Y_test_denorm = copy.deepcopy(Y_test)
    for i in range(Y_test.shape[0]):
        Y_test_denorm[i] = denormalize(Y_test[i], norm_params[index_target_series], method=normalization_method)
    
    return X_train, Y_train, X_test, Y_test, Y_train_denorm, Y_test_denorm, norm_params, index_target_series



def windows_preprocessing(time_series, target_time_series, past_history_factor, forecast_horizon):
    x, y = [], []
    for j in range(past_history_factor, time_series.shape[1] - forecast_horizon + 1, forecast_horizon):
        indices = list(range(j - past_history_factor, j))

        window_ts = []
        for i in range(time_series.shape[0]):
            window_ts.append(time_series[i, indices])
        window = np.array(window_ts)

        x.append(window)
        y.append(target_time_series[j: j + forecast_horizon])
    return np.array(x), np.array(y)


def normalize(data, norm_params, method="zscore"):
    """
    Normalize time series
    :param data: time series
    :param norm_params: tuple with params mean, std, max, min
    :param method: zscore or minmax
    :return: normalized time series
    """
    assert method in ["zscore", "minmax", "None"]

    if method == "zscore":
        std = norm_params["std"]
        if std == 0.0:
            std = 1e-10
        return (data - norm_params["mean"]) / norm_params["std"]

    elif method == "minmax":
        denominator = norm_params["max"] - norm_params["min"]

        if denominator == 0.0:
            denominator = 1e-10
        return (data - norm_params["min"]) / denominator

    elif method == "None":
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
        train[i] = normalize(
            np.array(train[i], dtype=dtype), nparams, method=normalization_method
        )
        norm_params.append(nparams)

    # Normalize test data
    test = np.array(test, dtype=dtype)
    for i in range(test.shape[0]):
        nparams = norm_params[i]
        test[i] = normalize(test[i], nparams, method=normalization_method)

    return train, test, norm_params