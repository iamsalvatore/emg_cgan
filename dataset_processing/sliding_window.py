
import shutil,os
import pandas as pd
from numpy import linspace, max, min, average, std, sum, sqrt, where, argmax
import scipy as sp
from scipy import signal
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
numpy.random.seed(seed=42)

def import_emg():
    file = Path('/Users/salvatoreesposito/Downloads/emg.csv')
    emg_data=pd.read_csv(file, sep=",")
    channel1_data = emg_data[emg_data.columns[0]]
    label = Path('/Users/salvatoreesposito/Downloads/restimulus.csv')
    y = pd.read_csv(label, sep=",")

    return channel1_data, y

def stft_spec(X,fs,image_no, path="/Users/salvatoreesposito/Documents/emg_data"):
    f, t, Zxx = signal.stft(X, fs=2000)
    fig = plt.figure()
    _spec = plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, shading='gouraud').get_array()
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.close(fig)
    # plt.savefig(path+'/'+str(image_no) + "_" + '.jpg', dpi=150)
    return _spec

def create_idx_window(data, inc=100, win_size=600):
    idx1=np.arange(0, len(data), inc)
    idx2=np.arange(win_size,len(data), inc)
    idx1 = idx1[:len(idx2)]
    # diff = len(idx1) - len(idx2)
    # idx1 = idx1[:-diff]

    return idx1, idx2

def label_subfolders(y):
    label_vals = np.unique(y, return_counts=False)
    for i in label_vals:
        os.makedirs(os.path.join("/Users/salvatoreesposito/Documents/emg_data/", str(i)))


def arrange_specs(current_label,image_no, folder="/Users/salvatoreesposito/Documents/emg_data/"):
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    # path_tail = os.path.split(str(subfolders))
    for folder in subfolders:
        if folder.split("/")[-1] == str(current_label):
            return folder

def slide_channels(x, idx1, idx2, labels, fs):
    _, channels = x.shape
    for i in range(channels):
        run_sliding_window(idx1, idx2, x, labels, fs)

def run_sliding_window(idx1,idx2,data,labels, fs, path=None):
    assert len(data) == len(labels), f"Length of channel is {len(data)} while labels is {len(labels)}"
    image_no = 1
    for i, (id1, id2) in enumerate(zip(idx1, idx2)):
        print(id1, id2)
        windows = data[id1:id2]
        window_labels = labels[id1:id2]
        label_vals, label_counts = np.unique(window_labels, return_counts=True)
        idx_max = np.argmax(label_counts)
        current_label = label_vals[idx_max]
        spec_temp= stft_spec(windows, fs, image_no, path)
        # print('spec', spec_temp.shape, type(spec_temp))
        path = arrange_specs(current_label, image_no)
        fname = (path + '/' + str(image_no) +'.npy')
        save_spec(np.array(spec_temp), fname)
        image_no += 1
    return spec_temp

def save_spec(spec_temp, fname):
    '''
    @param: fname: filename with *.npy extension
    '''
    with open(fname, 'wb') as f:
        np.save(f, spec_temp)

def main():
    channel1_data, y = import_emg()
    idx1, idx2 = create_idx_window(channel1_data)
    run_sliding_window(idx1, idx2, channel1_data,y, fs=2000, path="/Users/salvatoreesposito/Documents/emg_data")

if __name__ == '__main__':
    main()