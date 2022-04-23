from cProfile import label
from calendar import c
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
pd.options.mode.chained_assignment = None
import time
import scipy.io
import click
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor  
from sklearn.decomposition import PCA



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
    dataset = pd.DataFrame({'x_Laser': dict["LCam1_Gauss"][:,0], 'xRMS_Laser': dict["LCam1_Gauss"][:,1], 'y_Laser': dict["LCam1_Gauss"][:,2], 'yRMS_Laser': dict["LCam1_Gauss"][:,3], 
                            'u_Laser': dict["LCam1_Gauss"][:,4], 'uRMS_Laser': dict["LCam1_Gauss"][:,5], 'v_Laser': dict["LCam1_Gauss"][:,6], 'vRMS_Laser': dict["LCam1_Gauss"][:,7], 
                            'sum_Laser': dict["LCam1_Gauss"][:,8], 'rf_amp': dict["Cav_Amp"],  'rf_phs': dict["Cav_Phs"],  'fw2_amp': dict["Fwd2_Amp"],  'fw2_phs': dict["Fwd2_Phs"], 
                            'rv_amp': dict["Rev_Amp"],  'rv_phs': dict["Rev_Phs"],  'fw1_amp': dict["Fwd1_Amp"],  'fw1_phs': dict["Fwd1_Phs"], 'laser_phs': dict["LP_Amp"], 'laser_phs': dict["LP_Phase"]})
    cam = dict["AdjUCam1Pos"]
    return dataset, cam

def normalization_train(dataset):
    C_mean = []
    C_std = []
    for column in dataset.columns:
        column_mean = dataset[column].mean()
        column_std = dataset[column].std()
        dataset.loc[:,column] = (dataset[column] - column_mean) / column_std
        C_mean.append(column_mean)
        C_std.append(column_std)
    return dataset,C_mean,C_std
def normalization_test(dataset, C_mean, C_std):
    i = 0
    for column in dataset.columns:
        column_mean = C_mean[i]
        column_std = C_std[i]
        dataset.loc[:,column]  = (dataset[column] - column_mean) / column_std
        i += 1 
    return dataset
# def denormalization(C_mean, C_std, dataset):
#     i = 0
#     for column in dataset.columns:
#         column_mean = C_mean[i]
#         column_std = C_std[i]
#         dataset[column] = (dataset[column] * column_std) + column_mean
#         i += 1 
#     return dataset

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
    plt.savefig("NewDataset_Train_Test.png")
    plt.show()
    return



#################################################################################
#################################################################################
#################################################################################

if __name__ == "__main__":
    clrscr()
    
    filname = "new_dataset/Fourth_Dataset/OpenLoop1postp.mat" #-->OpenLoop
    #filname = "new_dataset/Fourth_Dataset/ClosedLoop1postp.mat" #-->CloseLoop
    dict = loadmat(filname)
    # print(dict["syncData"])
    dataset, cam = to_dataframe(dict) # X and Y
    
    
    dataset.drop(dataset.tail(500).index, inplace = True)
    cam = cam[:-500]
    
    #X_train, X_test, y_train, y_test = train_test_split(dataset, cam, random_state=42, test_size=0.2, shuffle=False)
    stop = int(0.8*len(dataset))
    X_train = dataset.iloc[:stop]
    X_test = dataset.iloc[stop:]
    y_train = cam[:stop]
    y_test = cam[stop:]

    
    X_train,C_mean,C_std = normalization_train(X_train)
    X_test = normalization_test(X_test, C_mean, C_std)
    # pca = PCA(n_components=0.90)
    # X_Train_new = pca.fit_transform(X_train) 
    # X_Test_new = pca.transform(X_test) 
    # #X_train, X_test = X_train.reshape(X_train.shape[0], -1), X_test.reshape(X_test.shape[0], -1)
    
    
    #model = RandomForestRegressor().fit(X_train, y_train)
    model = MLPRegressor(hidden_layer_sizes=(128,128), solver='adam',  learning_rate='constant', learning_rate_init=0.01,  
                         activation="relu" ,random_state=42, max_iter=2000, shuffle=True, early_stopping=True, 
                         validation_fraction=0.1, n_iter_no_change=30, verbose=0).fit(X_train, y_train)
    # plt.plot(model.loss_curve_,'r',label="loss_curve")
    # plt.legend()
    # plt.show()
    # plt.savefig("01_loss_curve.png")

    #model = LinearRegression().fit(X_train, y_train)

    train_forecast=model.predict(X_train)
    test_forecast=model.predict(X_test)

    plotting_1(y_train,train_forecast,y_test, test_forecast, 0)
    # plotting_4(y_train, train_forecast, y_test, test_forecast)
    std_data_train = np.std(y_train)
    std_train = np.std(y_train-train_forecast)    
    std_data_test = np.std(y_test)
    std_test = np.std(y_test-test_forecast)   
    
    rms_train = np.sqrt(np.mean(y_train**2))
    rmse_train = mean_squared_error(y_train, train_forecast, squared=False)
    rms_test = np.sqrt(np.mean(y_test**2))
    rmse_test = mean_squared_error(y_test, test_forecast, squared=False)
        
    
    print("")
    print("-------------------------------------- TRAIN --------------------------------------")
    print("STD (Label)  ---------------------------------------------------->",round(std_data_train, 3))
    print("RMS  ----------------------------------------------------------->",round(rms_train, 3))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("STD (Label - Prediction)  --------------------------------------->",round(std_train, 3))
    print("RMSE  ----------------------------------------------------------->",round(rmse_train, 3))
    print("-----------------------------------------------------------------------------------")
    print("")
    print("")
    print("--------------------------------------- TEST ---------------------------------------")
    print("STD (Label)  ---------------------------------------------------->",round(std_data_test, 3))
    print("RMS  ----------------------------------------------------------->",round(rms_test, 3))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("STD (Label - Prediction)  --------------------------------------->",round(std_test, 3))
    print("RMSE ------------------------------------------------------------>",round(rmse_test, 3))
    print("-----------------------------------------------------------------------------------") 

    
    #Addestra con i non stabilizzati e testa sugli stabilizzati
    
    
    # def plotting_4(y_train, train_forecast, y_test, test_forecast):
    # start = 0
    # stop = 60
    # Rolling_Y_train = []
    # Rolling_train_forecast =[]
    # for j in range(0,len(y_train)-60):
    #     Rolling_Y_train.append(np.std(y_train[start:stop]))
    #     Rolling_train_forecast.append(np.std(y_train[start:stop] - train_forecast[start:stop]))
    #     start += 1
    #     stop += 1
    # start = 0
    # stop = 60
    # Rolling_Y_test = []
    # Rolling_test_forecast = []
    # for j in range(0,len(y_test)-60):
    #     Rolling_Y_test.append(np.std(y_test[start:stop]))
    #     Rolling_test_forecast.append(np.std(y_test[start:stop] - test_forecast[start:stop]))
    #     start += 1
    #     stop += 1
    # fig, (ax1,ax2) = plt.subplots(2,figsize=(16,6))
    # ax1.plot(Rolling_Y_train,"r",label= "[Rolling STD]")
    # ax1.hlines(y=np.std(y_train), xmin=0, xmax=len(Rolling_Y_train), color='orange', label= ["STD :",round(np.std(y_train), 3)])
    # ax1.plot(Rolling_train_forecast,"k",label= "[Rolling STD (Label - Prediction)]")
    # ax1.hlines(y=np.std(y_train - train_forecast), xmin=0, xmax=len(Rolling_train_forecast), color='gray', label= ["STD  (Label - Prediction):",round(np.std(y_train - train_forecast),3)])
    # ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))   
    # ax1.set_ylabel("STD")
    # ax1.set_title("TRAIN")
    # ax1.grid(axis="x")
    # ax1.grid(axis="y")
    # ax1.legend()
    # ax2.plot(Rolling_Y_test,"r",label= "[Rolling STD]")
    # ax2.hlines(y=np.std(y_test), xmin=0, xmax=len(Rolling_Y_test), color='orange', label= ["STD :",round(np.std(y_test), 3)])
    # ax2.plot(Rolling_test_forecast,"k",label= "[Rolling STD  (Label - Prediction)]")
    # ax2.hlines(y=np.std(y_test - test_forecast), xmin=0, xmax=len(Rolling_test_forecast), color='grey', label= ["STD  (Label - Prediction):",round(np.std(y_test - test_forecast),3)])
    # ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))   
    # ax2.set_ylabel("STD")
    # ax2.set_title("TEST")
    # ax2.grid(axis="x")
    # ax2.grid(axis="y")
    # ax2.set_xlabel("Acq point)")
    # ax2.legend()
    # plt.tight_layout()
    # plt.savefig("NewDataset_STDs.png")
    # plt.show()
    # return  