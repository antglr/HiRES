import sys
import numpy as np
from matplotlib import pyplot as plt

f1 = sys.argv[1]
f2 = sys.argv[2]

# RF
with open(f1, 'r') as FR:
    h1 = FR.readline().replace('#','')
    h1 = h1.split(',')
d1 = np.loadtxt(f1).transpose()

# Cam
with open(f2, 'r') as FR:
    h2 = FR.readline().replace('#','')
    h2 = h2.split(',')
cam = np.loadtxt(f2)

# Normalize all signals so they are within 1, -1
for i in range(d1.shape[0]):
    m = np.max(abs(d1[i]))
    d1[i] = d1[i]/m

cam = cam/np.max(abs(cam))

# Optionally slide cam and rf w.r.t each other
cam_start = 0
cam_end = -2
rf_start = 2
rf_end = d1.shape[1]

cam = cam[cam_start:cam_end]

# Correlate all RF signals with beam centroid
for i, (d, h) in enumerate(zip(d1, h1)):
    # Signal energies
    d_energy = np.sum(d**2)
    cam_energy = np.sum(cam**2)
    norm = np.sqrt(d_energy * cam_energy)

    # Optionally discard images
    d = d[rf_start:rf_end]
    c = np.correlate(d, cam)
    c_norm = 100.0 * c / norm
    print("[%s] vs [centroid] correlation: %f (%f %%)" % (h, c, c_norm))
    h1[i] += " corr=%f (%f %%)" % (c, c_norm)

# Plotting
use_plotly = True
if use_plotly:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=1)

for d, h in zip(d1, h1):
    if use_plotly:
        x = np.arange(d.shape[0])
        fig.add_trace(go.Scatter(x=x, y=d[rf_start:rf_end], mode='lines', name=h), row=1, col=1)
    else:
        plt.plot(d, label=h)

if use_plotly:
    x = np.arange(cam.shape[0])
    fig.add_trace(go.Scatter(x=x, y=cam, mode='lines', name=h2[0]), row=1, col=1)
else:
    plt.plot(d, label=h)

if use_plotly:
    fig.update_layout(title="RF vs CAM correlation [%s] [%s]" % (f1, f2))
    fig.show()
else:
    plt.legend()
    plt.show()
