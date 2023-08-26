
class ConfigSED(object):
    def __init__(self, window='hann', center=True, pad_mode='reflect', windows_size=2048, hop_size=512,
                sample_rate=32000, mel_bins=128, fmin=50, fmax=16000, ref=1.0, amin=1e-10, top_db=None):
        # Spectrogram
        self.window = window
        self.center = center 
        self.pad_mode = pad_mode
        self.windows_size = windows_size 
        self.hop_size = hop_size

        # Logmel Spectrogram
        self.sample_rate=sample_rate
        self.mel_bins=mel_bins 
        self.fmin=fmin 
        self.fmax=fmax
        self.ref = ref 
        self.amin = amin 
        self.top_db = top_db