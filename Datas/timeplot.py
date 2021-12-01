import sys
from matplotlib import pyplot as plt
import numpy as np
import time

ts = []
with open(sys.argv[1], 'r') as FH:
    ts = np.array(["T".join(x.split()) for x in FH.readlines()], dtype="datetime64")
td = np.diff(ts)
# print(ts)
# print(td)
print("Time elapsed:", ts[-1]-ts[0])

# Isolate actual discontinuities in timestep by ignoring entries with 0 time diff
zero_td = np.timedelta64(0, 'us')
td_jump = td[td > zero_td].astype('float')

# Moving average of time discontinuities
N = 10 # len(td_jump)
td_jump_avg = np.convolve(td_jump, np.ones(N)/N)#, mode='valid')

# Calculate and trace max/min; beware of boundaries
ymin = min(td_jump_avg[N:-N])
ymax = max(td_jump_avg[N:-N])

plt.subplot2grid((2,2), (0,0))
plt.plot(ts, marker='x', ls='--', color='green')
plt.ylabel("UTC Time")

plt.subplot2grid((2,2), (0,1))
plt.plot(td, marker='+')
plt.ylabel("Time [us]")
plt.suptitle(sys.argv[1])

plt.subplot2grid((2,2), (1,0), colspan=2)
plt.plot(td_jump, marker='+')
plt.plot(td_jump_avg, label="Moving average N=%d"%N)
plt.axhline(ymin, color='tab:red', label="Min = %f"%np.min(ymin))
plt.axhline(ymax, color='tab:red', label="Max = %f"%np.max(ymax))
plt.ylabel("Time [us]")
plt.legend()

plt.show()






