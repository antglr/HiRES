README file for the data from 2022-04-11

Open loop acquisition with 7994 data points (OpenLoop1).
Closed loop acquisition with 12001 data points (ClosedLoop1).

Closed loop dataset: last five minutes are to be omitted; used for synchronization tests
Three RF shots are missing from closed loop dataset.  

Asynchrnous analysis removes several other shots from both OpenLoop1 and ClosedLoop1 during which no dipole values were recorded.  
After removal, ClosedLoop1 has 11959 entries and OpenLoop1 has XXXXX entries.

Format of the synchronous data: everything is contained in a structure named syncData.  For OpenLoop1, the file is called OpenLoop1postp.mat. 
For ClosedLoop1, the file is called ClosedLoop1postp.mat.

-syncData.params: acquisition parameters
-syncData.UCam1_Gauss: image analysis on UCam1 images using 1-D gaussian fitting.  
Columns:  [xCentroid, xrms, yCentroid, yrms, uCentroid, urms, vCentroid, vrms, Gaussian_Sum] where u and v are the rotated coordinates.
All values are in pixels.  Higher xCentroid pixel values correspond to lower energies.
-syncData.UCam1_GaussBG: same as above, but a planar background is subtracted before analysis.
-syncData.UCam1_COM: Center of mass calculation for the centroid in x
-syncData.UCam1_tt: camonitor time tags for UCam1 acquisitions.
-syncData.LCam1_Gauss: image analysis on LCam1 images using 1-D gaussian fitting.  
Columns:  [xCentroid, xrms, yCentroid, yrms, uCentroid, urms, vCentroid, vrms, Gaussian_Sum] where u and v are the rotated coordinates.
All values are in pixels.  
-syncData.LCam1_GaussBG: same as above, but a planar background is subtracted before analysis.
-syncData.LCam1_COM: Center of mass calculation for the centroid in x
-syncData.LCam1_tt: camonitor time tags for UCam1 acquisitions.
-syncData.MagData: Magnet values acquired by camonitor.  Currently not used in analysis
-syncData.Cav_Amp: the mean cavity amplitude for the 28us window.  Measured by the in loop probe.
-syncData.Cav_Amp_tt: camonitor time tags for Cav_Amp.
-syncData.Cav_Phs: the mean cavity phase for the 28us window.  Measured by the in loop probe.
-syncData.Cav_Amp_tt: camonitor time tags for Cav_Phs.
-syncData.Fwd2_Amp: the mean cavity amplitude for the 28us window.  Measured by the out of loop probe.
-syncData.Fwd2_Amp_tt: camonitor time tags for Fwd2_Amp.
-syncData.Fwd2_Phs: the mean cavity phase for the 28us window.  Measured by the out of loop probe.
-syncData.Fwd2_Amp_tt: camonitor time tags for Fwd2_Phs.
-syncData.Rev_Amp: the mean forward power amplitude for the 28us window.  
-syncData.Rev_Amp_tt: camonitor time tags for Rev_Amp.
-syncData.Rev_Phs: the mean forward power phase for the 28us window.  
-syncData.Rev_Amp_tt: camonitor time tags for Rev_Phs.
-syncData.Fwd1_Amp: the mean reverse power amplitude for the 28us window. 
-syncData.Fwd1_Amp_tt: camonitor time tags for Fwd1_Amp.
-syncData.Fwd1_Phs: the mean reverse power phase for the 28us window.  
-syncData.Fwd1_Amp_tt: camonitor time tags for Fwd1_Phs.
-syncData.LP_Amp: the mean "laser amplitude" for the 28us window. 
-syncData.LP_Amp_tt: camonitor time tags for LP_Amp.
-syncData.LP_Phase: the mean lase phase for the 28us window.  
-syncData.LP_Phase_tt: camonitor time tags for LP_Phase.
-syncData.AdjUCam1Pos: UCam1 positions with dipole and LCam correlations subtracted.
-syncData.syncDataOld: Each time syncData was edited, the previous version is archived here.  
Going through nested versions, one can find the original dataset.


The following parameters were taken asynchronously using the data browser.  These parameters are recorded in params1.mat.  The channels correspond to:
channel0:Gun:HPA:Grid1Current
channel1:Gun:HPA:Grid2Current
channel2:Gun:RF:Temp9
channel3:Gun:RF:Temp13
channel4:Gun:RF:Temp14
channel5:Gun:RF:Temp15
channel6:Gun:RF:Loop_Coupler_Outlet_Temp
channel7:Gun:RF:Temp10
channel8:CavityTuner:LoadAvg
channel9:UDIP1:CurrentRBV

The asynchronous data is on a different clock than the other data, so here are the start and end times for the data sets using this clock:
OpenLoop1: 2022-04-11 13:00.59.729 to 2022-04-11 15:13:48.729
ClosedLoop1: 2022-04-11 17:50:37.130 to 2022-04-11 21:10:38.130
For the dipole asynchronous data, these start times correspond to the following indices:
OpenLoop1: 179777
ClosedLoop1: 246150
