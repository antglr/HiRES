import numpy as np

f1 = "RF.post"
f2 = "Cam.post"

# RF
with open(f1, 'r') as FR:
    h1 = FR.readline().replace('#','')
    h1 = h1.split(',')
d1 = np.loadtxt(f1).transpose()
IL_Magnitude= d1[0,:]
np.save("IL_Magnitude.npy",IL_Magnitude)
IL_Phase = d1[1,:]
np.save("IL_Phase.npy",IL_Phase)
OL_Magnitude = d1[2,:]
np.save("OL_Magnitude.npy",OL_Magnitude)
OL_Phase = d1[3,:]
np.save("OL_Phase.npy",OL_Phase)
IL_Energy = d1[4,:]
np.save("IL_Energy.npy",IL_Energy)
OL_Energy = d1[5,:]
np.save("OL_Energy.npy",OL_Energy)

# Cam
with open(f2, 'r') as FR:
    h2 = FR.readline().replace('#','')
    h2 = h2.split(',')
cam = np.loadtxt(f2)
np.save("Camera.npy",cam)