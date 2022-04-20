from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import click
import scipy.io as sio
from scipy.stats import pearsonr


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
    
    x_Laser = dict["LCam1_Gauss"][:,0]
    
    xRMS_Laser = dict["LCam1_Gauss"][:,1]
    y_Laser = dict["LCam1_Gauss"][:,2]
    yRMS_Laser = dict["LCam1_Gauss"][:,3]
    u_Laser = dict["LCam1_Gauss"][:,4]
    uRMS_Laser = dict["LCam1_Gauss"][:,5]
    v_Laser = dict["LCam1_Gauss"][:,6]
    vRMS_Laser = dict["LCam1_Gauss"][:,7]
    sum_Laser = dict["LCam1_Gauss"][:,8]
    rf_amp = dict["Cav_Amp"]
    rf_phs = dict["Cav_Phs"]
    fw2_amp =dict["Fwd2_Amp"] 
    fw2_phs =dict["Fwd2_Phs"] 
    rv_amp = dict["Rev_Amp"]
    rv_phs = dict["Rev_Phs"]
    fw1_amp = dict["Fwd1_Amp"]
    fw1_phs = dict["Fwd1_Phs"]
    laser_amp = dict["LP_Amp"]
    laser_phs = dict["LP_Phase"]
    
    cam = dict["AdjUCam1Pos"]
    return  x_Laser, xRMS_Laser, y_Laser, yRMS_Laser, u_Laser, uRMS_Laser, v_Laser, vRMS_Laser, sum_Laser, rf_amp, rf_phs, fw2_amp, fw2_phs, rv_amp, rv_phs, fw1_amp, fw1_phs, laser_amp, laser_phs, cam

def plotting_variable(x_Laser, xRMS_Laser, y_Laser, yRMS_Laser, u_Laser, uRMS_Laser, v_Laser, vRMS_Laser, sum_Laser, rf_amp, rf_phs, fw2_amp, fw2_phs, rv_amp, rv_phs, fw1_amp, fw1_phs, laser_amp, laser_phs, cam):
    fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,figsize=(12,6),sharex=True)
    ax1.plot(x_Laser, color ='#00FFFF')
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax1.set_ylabel("x_Laser")
    ax1.grid(axis="x")
    ax1.grid(axis="y")
    ax2.plot(xRMS_Laser, color ='#7FFFD4')
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax2.set_ylabel("xRMS_Laser")
    ax2.grid(axis="x")
    ax2.grid(axis="y")
    ax3.plot(y_Laser, color ='#76EEC6')
    ax3.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax3.set_ylabel("y_Laser")
    ax3.grid(axis="x")
    ax3.grid(axis="y")
    ax4.plot(yRMS_Laser, color ='#66CDAA')
    ax4.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax4.set_ylabel("yRMS_Laser")
    ax4.grid(axis="x")
    ax4.grid(axis="y")
    ax5.plot(u_Laser, color ='#458B74')
    ax5.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax5.set_ylabel("u_Laser")
    ax5.grid(axis="x")
    ax5.grid(axis="y")
    save_name = "First5Variable.png"
    plt.savefig(save_name)
    
    
    fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,figsize=(12,6),sharex=True)
    ax1.plot(uRMS_Laser, color ='#A52A2A')
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax1.set_ylabel("uRMS_Laser")
    ax1.grid(axis="x")
    ax1.grid(axis="y")
    ax2.plot(v_Laser, color ='#FF4040')
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax2.set_ylabel("v_Laser")
    ax2.grid(axis="x")
    ax2.grid(axis="y")
    ax3.plot(vRMS_Laser, color ='#EE3B3B')
    ax3.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax3.set_ylabel("vRMS_Laser")
    ax3.grid(axis="x")
    ax3.grid(axis="y")
    ax4.plot(sum_Laser, color ='#CD3333')
    ax4.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax4.set_ylabel("sum_Laser")
    ax4.grid(axis="x")
    ax4.grid(axis="y")
    ax5.plot(rf_amp, color ='#8B2323')
    ax5.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax5.set_ylabel("rf_amp")
    ax5.grid(axis="x")
    ax5.grid(axis="y")
    save_name = "First10Variable.png"
    plt.savefig(save_name)
    
    fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,figsize=(12,6),sharex=True)
    ax1.plot(rf_phs, color ='#7FFF00')
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax1.set_ylabel("rf_phs")
    ax1.grid(axis="x")
    ax1.grid(axis="y")
    ax2.plot(fw2_amp, color ='#76EE00')
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax2.set_ylabel("fw2_amp")
    ax2.grid(axis="x")
    ax2.grid(axis="y")
    ax3.plot(fw2_phs, color ='#66CD00')
    ax3.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax3.set_ylabel("fw2_phs")
    ax3.grid(axis="x")
    ax3.grid(axis="y")
    ax4.plot(rv_amp, color ='#458B00')
    ax4.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax4.set_ylabel("rv_amp")
    ax4.grid(axis="x")
    ax4.grid(axis="y")
    ax5.plot(rv_phs, color ='#3D9140')
    ax5.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax5.set_ylabel("rv_phs")
    ax5.grid(axis="x")
    ax5.grid(axis="y")
    save_name = "First15Variable.png"
    plt.savefig(save_name)
    
    fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,figsize=(12,6),sharex=True)
    ax1.plot(fw1_amp, color ='#B8860B')
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax1.set_ylabel("fw1_amp")
    ax1.grid(axis="x")
    ax1.grid(axis="y")
    ax2.plot(fw1_phs, color ='#FFB90F')
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax2.set_ylabel("fw1_phs")
    ax2.grid(axis="x")
    ax2.grid(axis="y")
    ax3.plot(laser_amp, color ='#EEAD0E')
    ax3.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax3.set_ylabel("laser_amp")
    ax3.grid(axis="x")
    ax3.grid(axis="y")
    ax4.plot(laser_phs, color ='#CD950C')
    ax4.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax4.set_ylabel("laser_phs")
    ax4.grid(axis="x")
    ax4.grid(axis="y")
    ax5.plot(cam, color ='#8B6508')
    ax5.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax5.set_ylabel("cam")
    ax5.grid(axis="x")
    ax5.grid(axis="y")
    save_name = "First20Variable.png"
    plt.savefig(save_name)
    return 

def plotting_correlation(x_Laser, xRMS_Laser, y_Laser, yRMS_Laser, u_Laser, uRMS_Laser, v_Laser, vRMS_Laser, sum_Laser, rf_amp, rf_phs, fw2_amp, fw2_phs, rv_amp, rv_phs, fw1_amp, fw1_phs, laser_amp, laser_phs, cam):
    fig, axs = plt.subplots(5, 4, figsize=(12,6),sharex=True)
    corr, _ = pearsonr(cam,x_Laser)
    axs[0,0].plot(cam,x_Laser,"o", color = "#FF8C00", label= 'Pearsons: %.2f' % corr)
    axs[0,0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[0,0].set_ylabel("x_Laser")
    axs[0,0].grid(axis="x")
    axs[0,0].grid(axis="y")
    axs[0,0].legend()
    corr, _ = pearsonr(cam,xRMS_Laser)
    axs[1,0].plot(cam,xRMS_Laser,"o", color = "#FF7F00", label= 'Pearsons: %.2f' % corr)
    axs[1,0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[1,0].set_ylabel("xRMS_Laser")
    axs[1,0].grid(axis="x")
    axs[1,0].grid(axis="y")
    axs[1,0].legend()
    corr, _ = pearsonr(cam,y_Laser)
    axs[2,0].plot(cam,y_Laser,"o", color = "#EE7600", label= 'Pearsons: %.2f' % corr)
    axs[2,0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[2,0].set_ylabel("y_Laser")
    axs[2,0].grid(axis="x")
    axs[2,0].grid(axis="y")
    axs[2,0].legend()
    corr, _ = pearsonr(cam,yRMS_Laser)
    axs[3,0].plot(cam, yRMS_Laser,"o", color = "#CD6600", label= 'Pearsons: %.2f' % corr)
    axs[3,0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[3,0].set_ylabel("yRMS_Laser")
    axs[3,0].grid(axis="x")
    axs[3,0].grid(axis="y")
    axs[3,0].legend()
    corr, _ = pearsonr(cam,u_Laser)
    axs[4,0].plot(cam, u_Laser,"o", color = "#8B4500", label= 'Pearsons: %.2f' % corr)
    axs[4,0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[4,0].set_ylabel("u_Laser")
    axs[4,0].grid(axis="x")
    axs[4,0].grid(axis="y")    
    axs[4,0].legend()
    corr, _ = pearsonr(cam,uRMS_Laser)
    axs[0,1].plot(cam, uRMS_Laser,"o", color = "#9932CC", label= 'Pearsons: %.2f' % corr)
    axs[0,1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[0,1].set_ylabel("uRMS_Laser")
    axs[0,1].grid(axis="x")
    axs[0,1].grid(axis="y")
    axs[0,1].legend()
    corr, _ = pearsonr(cam,v_Laser)
    axs[1,1].plot(cam, v_Laser,"o", color = "#BF3EFF", label= 'Pearsons: %.2f' % corr)
    axs[1,1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[1,1].set_ylabel("v_Laser")
    axs[1,1].grid(axis="x")
    axs[1,1].grid(axis="y")
    axs[1,1].legend()
    corr, _ = pearsonr(cam,vRMS_Laser)
    axs[2,1].plot(cam, vRMS_Laser,"o", color = "#B23AEE", label= 'Pearsons: %.2f' % corr)
    axs[2,1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[2,1].set_ylabel("vRMS_Laser")
    axs[2,1].grid(axis="x")
    axs[2,1].grid(axis="y")
    axs[2,1].legend()
    corr, _ = pearsonr(cam,sum_Laser)
    axs[3,1].plot(cam, sum_Laser,"o", color = "#9A32CD", label= 'Pearsons: %.2f' % corr)
    axs[3,1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[3,1].set_ylabel("sum_Laser")
    axs[3,1].grid(axis="x")
    axs[3,1].grid(axis="y")
    axs[3,1].legend()
    corr, _ = pearsonr(cam,rf_amp)
    axs[4,1].plot(cam, rf_amp,"o", color = "#68228B", label= 'Pearsons: %.2f' % corr)
    axs[4,1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[4,1].set_ylabel("rf_amp")
    axs[4,1].grid(axis="x")
    axs[4,1].grid(axis="y")    
    axs[4,1].legend()
    corr, _ = pearsonr(cam,rf_phs)
    axs[0,2].plot(cam, rf_phs,"o", color = "#00BFFF", label= 'Pearsons: %.2f' % corr)
    axs[0,2].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[0,2].set_ylabel("rf_phs")
    axs[0,2].grid(axis="x")
    axs[0,2].grid(axis="y")
    axs[0,2].legend()
    corr, _ = pearsonr(cam,fw2_amp)
    axs[1,2].plot(cam, fw2_amp,"o", color = "#00B2EE", label= 'Pearsons: %.2f' % corr)
    axs[1,2].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[1,2].set_ylabel("fw2_amp")
    axs[1,2].grid(axis="x")
    axs[1,2].grid(axis="y")
    axs[1,2].legend()
    corr, _ = pearsonr(cam,fw2_phs)
    axs[2,2].plot(cam, fw2_phs,"o", color = "#009ACD", label= 'Pearsons: %.2f' % corr)
    axs[2,2].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[2,2].set_ylabel("fw2_phs")
    axs[2,2].grid(axis="x")
    axs[2,2].grid(axis="y")
    axs[2,2].legend()
    corr, _ = pearsonr(cam,rv_amp)
    axs[3,2].plot(cam, rv_amp,"o", color = "#00688B", label= 'Pearsons: %.2f' % corr)
    axs[3,2].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[3,2].set_ylabel("rv_amp")
    axs[3,2].grid(axis="x")
    axs[3,2].grid(axis="y")
    axs[3,2].legend()
    corr, _ = pearsonr(cam,rv_phs)
    axs[4,2].plot(cam, rv_phs,"o", color = "#104E8B", label= 'Pearsons: %.2f' % corr)
    axs[4,2].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[4,2].set_ylabel("rv_phs")
    axs[4,2].grid(axis="x")
    axs[4,2].grid(axis="y")
    axs[4,2].legend()
    corr, _ = pearsonr(cam,fw1_amp)
    axs[0,3].plot(cam, fw1_amp,"o", color = "#8FBC8F", label= 'Pearsons: %.2f' % corr)
    axs[0,3].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[0,3].set_ylabel("fw1_amp")
    axs[0,3].grid(axis="x")
    axs[0,3].grid(axis="y")
    axs[0,3].legend()
    corr, _ = pearsonr(cam,fw1_phs)
    axs[1,3].plot(cam, fw1_phs,"o", color = "#C1FFC1", label= 'Pearsons: %.2f' % corr)
    axs[1,3].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[1,3].set_ylabel("fw1_phs")
    axs[1,3].grid(axis="x")
    axs[1,3].grid(axis="y")
    axs[1,3].legend()
    corr, _ = pearsonr(cam,laser_amp)
    axs[2,3].plot(cam, laser_amp,"o", color = "#B4EEB4", label= 'Pearsons: %.2f' % corr)
    axs[2,3].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[2,3].set_ylabel("laser_amp")
    axs[2,3].grid(axis="x")
    axs[2,3].grid(axis="y")
    axs[2,3].legend()
    corr, _ = pearsonr(cam,laser_phs)
    axs[3,3].plot(cam, laser_phs,"o", color = "#9BCD9B", label= 'Pearsons: %.2f' % corr)
    axs[3,3].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[3,3].set_ylabel("laser_phs")
    axs[3,3].grid(axis="x")
    axs[3,3].grid(axis="y")
    axs[3,3].legend()
    corr, _ = pearsonr(cam,cam)
    axs[4,3].plot(cam, cam,"o", color = "#698B69", label= 'Pearsons: %.2f' % corr)
    axs[4,3].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[4,3].set_ylabel("cam")
    axs[4,3].grid(axis="x")
    axs[4,3].grid(axis="y")
    axs[4,3].legend()
    fig.tight_layout()
    save_name = "First200Variable.png"
    plt.savefig(save_name)
    return 



if __name__ == "__main__":
    clrscr()
    t = time.time()
    
    #filname = "new_dataset/Fourth_Dataset/ClosedLoop1postp.mat" #-->CloseLoop
    filname = "new_dataset/Fourth_Dataset/OpenLoop1postp.mat" #-->OpenLoop
    dict = loadmat(filname)
    x_Laser, xRMS_Laser, y_Laser, yRMS_Laser, u_Laser, uRMS_Laser, v_Laser, vRMS_Laser, sum_Laser, rf_amp, rf_phs, fw2_amp, fw2_phs, rv_amp, rv_phs, fw1_amp, fw1_phs, laser_amp, laser_phs, cam = to_dataframe(dict) 
    
    plotting_variable(x_Laser, xRMS_Laser, y_Laser, yRMS_Laser, u_Laser, uRMS_Laser, v_Laser, vRMS_Laser, sum_Laser, rf_amp, rf_phs, fw2_amp, fw2_phs, rv_amp, rv_phs, fw1_amp, fw1_phs, laser_amp, laser_phs, cam)
    plotting_correlation(x_Laser, xRMS_Laser, y_Laser, yRMS_Laser, u_Laser, uRMS_Laser, v_Laser, vRMS_Laser, sum_Laser, rf_amp, rf_phs, fw2_amp, fw2_phs, rv_amp, rv_phs, fw1_amp, fw1_phs, laser_amp, laser_phs, cam)    