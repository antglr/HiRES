import numpy as np

#InOrOut = "OutLoop/"
InOrOut = "InLoop/"


f1 = InOrOut+"RF.post"
# RF
with open(f1, 'r') as FR:
    h1 = FR.readline().replace('#','')
    h1 = h1.split(',')
d1 = np.loadtxt(f1).transpose()
IL_Magnitude= d1[0,:]
np.save(InOrOut+"IL_Magnitude.npy",IL_Magnitude)
IL_Phase = d1[1,:]
np.save(InOrOut+"IL_Phase.npy",IL_Phase)
OL_Magnitude = d1[2,:]
np.save(InOrOut+"OL_Magnitude.npy",OL_Magnitude)
OL_Phase = d1[3,:]
np.save(InOrOut+"OL_Phase.npy",OL_Phase)
IL_Energy = d1[4,:]
np.save(InOrOut+"IL_Energy.npy",IL_Energy)
OL_Energy = d1[5,:]
np.save(InOrOut+"OL_Energy.npy",OL_Energy)
Laser_Amp = d1[6,:]
np.save(InOrOut+"Laser_Amp.npy",Laser_Amp)
Laser_Phs = d1[7,:]
np.save(InOrOut+"Laser_Phs.npy",Laser_Phs)

f2 = InOrOut+"CenterError_Fit.post"
# Cam
with open(f2, 'r') as FR:
    h2 = FR.readline().replace('#','')
    h2 = h2.split(',')
cam = np.loadtxt(f2)
np.save(InOrOut+"CameraFit.npy",cam)

f3 = InOrOut+"CenterError_Proj.post"
# Cam
with open(f2, 'r') as FR:
    h2 = FR.readline().replace('#','')
    h2 = h2.split(',')
cam = np.loadtxt(f2)
np.save(InOrOut+"CameraProj.npy",cam)