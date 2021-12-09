import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

def windows_preprocessing(time_series, camera_series, past_history, forecast_horizon):
    x, y = [], []
    for j in range(past_history, time_series.shape[1] - forecast_horizon + 1, forecast_horizon):
        indices = list(range(j - past_history, j))

        window_ts = []
        for i in range(time_series.shape[0]):
            window_ts.append(time_series[i, indices])
        window = np.array(window_ts)

        x.append(window)
        y.append(camera_series[j: j + forecast_horizon])
    return np.array(x), np.array(y)

## Loading Files 
#InOrOut = "OutLoop"
InOrOut = "InLoop" 
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

print("Shape Camera  -->",np.shape(camFit))
print("Shape RF Phase  -->",np.shape(OL_phs))
print("Shape RF Amplitude  -->",np.shape(OL_amp))
print("Shape Laser Phase  -->",np.shape(laser_Phs))
print("Shape Laser Amplitude  -->",np.shape(laser_amp))
print("------------------------------------------")
print("")

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


print("Shape Camera -- train -->",np.shape(cam_train))
print("Shape Camera -- test -->",np.shape(cam_test))
print("------------------------------------------")
print("")
if fetureSelection:
    # Selected variables
    train = np.array([phs_train, amp_train, diffAmp_train, Lamp_train, cam_train])
    test = np.array([phs_test, amp_test, diffAmp_test, Lamp_test, cam_test])
else:
    # All variables
    train = np.array([phs_train, amp_train,diffPhs_train, diffAmp_train, Lphs_train, Lamp_train, cam_train]) 
    test = np.array([phs_test, amp_test,diffPhs_test, diffAmp_test, Lphs_test, Lamp_test, cam_test])


#Performing windowing with: Phs, Ampl, and Cam (X_train/test) to prefict Cam+1 (Y_train/test) 
X_train, Y_train = windows_preprocessing(train, cam_train, past_history, forecast_horizon)
X_test, Y_test = windows_preprocessing(test, cam_test, past_history, forecast_horizon)
print("X -- train -->",np.shape(X_train))
print("Y -- train -->",np.shape(Y_train))
print("------------------------------------------")
print("")


#Saving dataset for hyperparameter tuning 
np.save("X_train.npy",X_train)
np.save("Y_train.npy",Y_train)
np.save("X_test.npy",X_test)
np.save("Y_test.npy",Y_test)


# Reshape for RF model -- No temporal dimenison 
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
print("X -- train -->",np.shape(X_train))
print("Y -- train -->",np.shape(Y_train))
print("------------------------------------------")
print("")

# Performing the ML
model = RandomForestRegressor(criterion='mse', n_jobs=-1, n_estimators=100)
# model = LinearRegression()

model.fit(X_train,Y_train)
forecast_train = model.predict(X_train)
metrics_train = np.abs(forecast_train-np.squeeze(Y_train))
forecast = model.predict(X_test)
metrics_test = np.abs(forecast-np.squeeze(Y_test))

print("")
print("Training mean error",np.mean(metrics_train))
print("Test mean error",np.mean(metrics_test))
print("")

#Plotting
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,6),sharey=True)
ax1.plot(metrics_train,"+k")
ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax1.hlines(np.mean(metrics_train), 0, len(metrics_train), colors='r', linestyles='--')
ax1.set_ylabel("Metrics")
ax1.set_xlabel("Samples")
ax1.set_title("Train")
ax2.plot(metrics_test,"+k")
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax2.hlines(np.mean(metrics_test), 0, len(metrics_test), colors='r', linestyles='--')
ax2.set_xlabel("Samples")
ax2.set_title("Test")
plt.show()

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,6),sharey=True)
ax1.plot(forecast_train,np.squeeze(Y_train),"+k")
ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax1.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
ax1.set_ylabel("Label")
ax1.set_xlabel("Prediction")
ax1.set_title("Train")
ax1.grid(axis="x")
ax1.grid(axis="y")
ax2.plot(forecast,np.squeeze(Y_test),"+k")
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax2.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
ax2.set_xlabel("Prediction")
ax2.set_title("Test")
ax2.grid(axis="x")
ax2.grid(axis="y")
plt.show()

massimo = max(np.max(Y_train),np.max(forecast_train),np.max(Y_test),np.max(forecast))
minimo = min(np.min(Y_train),np.min(forecast_train),np.min(Y_test),np.min(forecast))

fig, (ax1,ax2) = plt.subplots(2,figsize=(16,6))
ax1.plot(np.squeeze(Y_train),"r",label= "Label")
ax1.plot(forecast_train,"k",label= "Prediction")
ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax1.set_ylabel("Centroid Error")
ax1.set_title("Train")
ax1.grid(axis="x")
ax1.grid(axis="y")
ax1.set_ylim((minimo, massimo))
ax1.legend()
ax2.plot(np.squeeze(Y_test),"r",label= "Label")
ax2.plot(forecast,"k",label= "Prediction")
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax2.set_ylabel("Centroid Error")
ax2.set_xlabel("Time")
ax2.set_title("Test")
ax2.grid(axis="x")
ax2.grid(axis="y")
ax2.set_ylim((minimo, massimo))
ax2.legend()
plt.show()