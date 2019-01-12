import numpy as np
from scipy.fftpack import dct
from scipy.signal.windows import get_window
from python_speech_features import mfcc

class dctlayer():#(nn.Module):
    def __init__(self,):
        #super(dctlayer, self).__init__()
        pass

    def forward(self, signal, ndct=512, numoverlap=256, window='boxcar', MFCC=False):
        signal = np.array(signal)
        if MFCC == False:
            win = get_window(window, ndct)
            listdct = []
            for i in range(0, signal.shape[0]-ndct, numoverlap):
                listdct.append(dct(signal[i:i+ndct]*win))
        else:
            listdct = mfcc(signal, samplerate=512, winlen=1, winstep=0.5, nfft=ndct)

        return np.array(listdct)

A = dctlayer()
sig = np.random.rand(16000)

w = A.forward(sig, window='hann')
wmfcc = A.forward(sig, MFCC=True)

print('Размеры:', sig.shape, w.shape, wmfcc.shape)      
