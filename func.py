import tdt
import numpy as np
import scipy
from scipy.signal import detrend, resample, medfilt, butter, lfilter, filtfilt

#########################
## filtering functions ##
#########################
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

#########################
# deeplabcut functions ##
#########################

def load_dlc(csv_file_path):
    import numpy
    M = np.loadtxt(csv_file_path, delimiter=',', skiprows=3)
    # create nose & TTI vectors
    Nx = M[:, 1]
    Ny = M[:, 2]
    Tx = M[:, 10]
    Ty = M[:, 11]
    # calculate centroids using euclidean distance b/w them
    Cx = (Nx + Tx)/2
    Cy = (Ny + Ty)/2
    return Cx, Cy

def load_fp(fp_file_path):
    import tdt
    data = tdt.read_block(fp_file_path, evtype=['streams', 'epocs'])
    GCAMP = data.streams['GCP1'].data[2,:]
    ISOS = data.streams['GCP2'].data[2,:]
    fs = data.streams['GCP1'].fs
    fram = data.epocs['Fram'].onset
    return GCAMP, ISOS, fram, fs

def load_dlc_nose(csv_file_path):
    M = np.loadtxt(csv_file_path, delimiter=',',skiprows=3)
    Nx = M[:, 1]
    Ny = M[:, 2]
    return Nx, Ny

def calculate_animal_centroids(csv_file_path):
    import numpy
    M = np.loadtxt(csv_file_path, delimiter=',', skiprows=3)
    # create nose & TTI vectors
    Nx = M[:, 1]
    Ny = M[:, 2]
    Tx = M[:, 13]
    Ty = M[:, 14]
    Rx = (Nx + Tx)/2
    Ry = (Ny + Ty)/2

    # intruder nose: 19 and 20, intruder TTI: 28 and 29
    WNx = M[:, 19]
    WNy = M[:, 20]
    WTx = M[:, 28]
    WTy = M[:, 29]
    Ix = (WNx + WTx)/2
    Iy = (WNy + WTy)/2
    
    return Rx, Ry, Ix, Iy

def calculate_intruder_distance(Rx, Ry, Ix, Iy):
    # calculate euclidean distance between centroids
    import math
    Distance = []
    for i in range(len(Rx)):
        dist = math.sqrt(((Rx[i].astype(int) - Ix[i])**2)+((Ry[i].astype(int)-Ry[i])**2))
        Distance.append(dist)
        
    return Distance

def downsample_FP(signal, fram, fs, Cx, offset=1, cam_fr=40):
    """
    Inputs:
    -------
    signal: GCAMP (or isosbestic) array of signal values from the TDT system
    fram: array of camera onset times accoording to TDT
    fs: sampling frequency of data
    cam_fr: camera frame rate. in our case values are the same for all cage day trials, but they may not always be!
    Cx: required for downsampling based on positional tracking using dlc
    
    Returns:
    --------
    signal_ds: downsampled signal based on dx
    """
    import math
    sr_diff = fs/cam_fr
    rec_onset = fram[0]
    rec_offset = fram[-1]
    
    # Turn the GCAMP frames into seconds 
    signal_times = np.arange(0, (len(signal)/fs), (1/fs))
   
    # Calculate the index within the GCAMP stream that the camera turns on 
    cam_on_idx = (np.where(signal_times==(find_nearest(signal_times,rec_onset))))
    
    # Calculate the time in sec within the GCAMP stream that the camera turns on
    cam_on_sec = signal_times[cam_on_idx]
    
    # Calculate the index within the GCAMP stream that the camera turns off 
    cam_off_idx = (np.where(signal_times==(find_nearest(signal_times, rec_offset))))
    
    # Calculate the time in sec within the GCAMP stream that the camera turns off
    cam_off_sec = signal_times[cam_off_idx]
    
    # crop GCAMP signal from camera on to camera off
    signal_cropped = signal[cam_on_idx[0][0]:cam_off_idx[0][0]]
    
    # Create time array based on number of samples and sample frequency
    # crop time signal from camera on to camera off
    npts = len(signal)
    time_x = np.linspace(1, npts, npts) / fs
    
    # do the same for the time vector
    signal_cropped_time = len(signal_cropped)/fs
    time_cropped = time_x[cam_on_idx[0][0]:cam_off_idx[0][0]]
    
    # Downsample the cropped GCAMP recording to align with the camera frames so can match up with 
    # the Cx and Cy vectors
    signal_ds = []
    for idx, frame in enumerate(Cx):
        signal_ds.append(signal_cropped[int(np.round((idx*np.floor(sr_diff-offset))))])
    
    time_ds = []
    for idx, frame in enumerate(Cx):
        time_ds.append(time_cropped[int(np.round((idx*np.floor(sr_diff-offset))))])    
    
    return signal_ds, time_ds

def find_nearest(array,value):
    '''
    Finds the closest value in a given sorted array to your input value. 
    Only works on an increasing sorted array!
    '''
    import math
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]
    
def calculate_cage_distance(csv_file_path, Cx, Cy):
    # import necessary packages
    import math
    # calculate distance from top corner
    TCx = np.mean(Cx)
    TCy = np.max(Cy)
    TCDistance = []
    for i in range(len(Cx)):
        dist = math.sqrt(((Cx[i].astype(int) - TCx)**2)+((Cy[i].astype(int)-TCy)**2))
        TCDistance.append(dist)
    # calculate distance from left cornerr
    LCx = np.min(Cx)
    LCy = 200
    LCDistance = []
    for i in range(len(Cx)):
        dist = math.sqrt(((Cx[i].astype(int) - LCx)**2)+((Cy[i].astype(int)-LCy)**2))
        LCDistance.append(dist)   
    # calculate distance from right corner    
    RCx = np.max(Cx)
    RCy = 200
    RCDistance = []
    for i in range(len(Cx)):
        dist = math.sqrt(((Cx[i].astype(int) - RCx)**2)+((Cy[i].astype(int)-RCy)**2))
        RCDistance.append(dist) 
    return TCDistance, LCDistance, RCDistance

def filter_signal(signal_ds, fs):
    import scipy
#     signal = scipy.signal.detrend(signal_ds)
#     sig1 = butter_lowpass_filter(signal,80,fs,order=3)
#     sig2 = butter_highpass_filter(sig1,0.01,fs,order=3)
    sig = scipy.signal.detrend(scipy.stats.zscore(signal_ds))
    return sig

def updated_filter(signal_ds):
    # input: downsampled signal! don't try this on ur raw data!
    signal = scipy.signal.detrend(signal_ds)
    sig1 = scipy.stats.zscore(signal)
    sig2 = scipy.signal.detrend(scipy.signal.savgol_filter(sig1, 15, 3))
    return sig2

def downsampled_peaks(signal_ds, height):
    import scipy
    peaks, _ = scipy.signal.find_peaks(signal_ds, height=height, threshold=0.001)
    return peaks

def new_fp_peaks(raw_signal,fs):
    sig = detrend(raw_signal)
    sig2 = butter_highpass_filter(sig,0.1,fs,order=3)
    sig3 = butter_lowpass_filter(sig2,30,fs,order=3)
    return sig3

import math  
def calculateDistance(x1,y1,x2,y2):  
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
    return dist 