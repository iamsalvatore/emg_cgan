import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.fftpack
import scipy.signal as sigg
import scipy.io.wavfile
import copy
import math


def stft_spec_2(X, fs=2000):
    # ///////////////////////////
    freq_threshold = 500
    f, t, Zxx = sigg.stft(X, fs=2000)
    # Zxx = Zxx.astype(float)
    nearest = f[np.abs(f - freq_threshold).argmin()]
    cut_idx = np.argwhere(f == nearest)[0][0]
    _spec = plt.pcolormesh(t, f[:cut_idx], np.abs(Zxx[: cut_idx, :]), vmin=0, shading='gouraud').get_array()
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    print('shape 1: ', _spec.shape, f.shape, t.shape, cut_idx)
    _spec = np.reshape(_spec, (cut_idx, t.shape[0]))
    print('shape 2: ', _spec.shape, f.shape, t.shape)
    return _spec


def save_spec(spec_temp, fname):
    '''
    @param: fname: filename with *.npy extension
    '''
    with open(fname, 'wb') as f:
        np.save(f, spec_temp)

# def random_frex(lower_frex=[50,100,150], higher_frex=[200,250,300]):
#
#     l_frex = np.random.choice(lower_frex)
#     h_frex = np.random.choice(higher_frex)
#     while h_frex <= l_frex:
#         h_frex = np.random.choice(higher_frex)
#
#     chosen = [l_frex, h_frex]
#     return chosen

def random_frex(frex1, frex2):
    a = np.arange(frex1, frex2)
    chann = 60
    ll = []
    for i in range(chann):
        ll.append(np.random.choice(a))

    ll = np.array(ll).reshape(12, 5)
    ll = np.sort(ll, axis=1)

    return ll

# def bla():
#     for i in range(10000):
#         get_noise_specs  4 asmnples

def get_noise_specs(lower_frex, higher_frex, srate=2000, iter=25, no_chann=12, root_path="/Users/salvatoreesposito/Documents/50_100Hz_pure/0/"):

    npnts = srate * 10  # 2 seconds
    time = np.arange(0, npnts) / srate

    spec_sample_no = 0
    for k in range(iter):
        for i, (frex1, frex2) in enumerate(zip(lower_frex, higher_frex)):
            print("i:", i)

            frex = random_frex(frex1, frex2)
            # frex = [12,2]
            temp_list = []
            counter = 0

            for i in range(frex.shape[0]):
                sig = np.zeros(len(time))
                for j in range(0, frex.shape[1]):
                    sig = sig + np.sin(2 * np.pi * frex[i,j] * time)
                # sig = sig + np.random.randn(len(sig))
                sig = sig[:600]  # would be good to define this magic no. forgot the resoning behind it though
                # call spec on each new signal
                spec_temp = stft_spec_2(sig, fs=srate)
                temp_list.append(spec_temp)

                counter += 1

                if counter == no_chann:
                    # save spec 3d array
                    fname = (root_path + '/' + str(spec_sample_no) + '.npy')
                    save_spec(np.array(temp_list), fname)
                    print('Counter no.', counter, '3d spec shape', np.array(temp_list).shape)

                    counter = 0
                    temp_list = []
                    spec_sample_no += 1

def main():
    # lower_frex = [50, 100, 150, 200]
    # higher_frex = [250, 300, 350, 400]
    lower_frex = [50,150]
    higher_frex = [100,400]
    get_noise_specs(lower_frex, higher_frex, srate=2000, iter=50000, no_chann=12, root_path="/Users/salvatoreesposito/Documents/4sig_Hz/0/")

if __name__ == '__main__':
    main()


