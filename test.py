from audio_sed import * 
import torch 
from torch import nn 
from audio_sed.pytorch.sed_models import AudioClassifierSequenceSED, AudioSED, shape_from_backbone
from audio_sed.sed_config import ConfigSED
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import timm

if __name__ == '__main__':
    import numpy as np 
    class Config():
        def __init__(self):
            self.num_classes = 206
            self.sample_rate=32000
            self.max_length = 5
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.use_apex = True
            self.use_apex_val = False
            self.verbose=True
            self.epochs = 200 #45#205
            self.accumulation = 1
            self.batch_size = 16
        # self.l_mixup_spec=0.0
        # self.l_mixup_signal = 0.5
            self.lr =2e-4 # 6e-5# 
            self.kfold=5
            self.path = "checkpoints"
            self.model_name = "efficientnet_b1"
            self.description="finetuning eff b1 from jax 15 sec. mixup, gpu spectrogram  train, aug spec, permutation, label smoothing "
            self.name = f"{self.model_name}-fold0-sequencemean-baseline-v1-wav"
            self.save_name = f"{self.path}/{self.name}/{self.name}" # 
            # Options for Logmel
            self.mel_bins = 256
            self.fmin = 10
            self.fmax = 16000
            self.window_size = 2048
            self.hop_size = 320 # 
            self.early_stopping=20
            self.smoothing =  0.0#5 # 0.00 #
            self.mixup_spec = 0.
            self.mixup_signal = 0.5 # 0.5
            self.normalization=False
            self.permutation_chunks = True # True
            self.noise_cpu =False #  True
            self.noise_gpu= False # noise signal level
            self.augmentations= False # dropstride etc on spectrogram
            self.spec_augment = False # True #  True # augmentation on specc horizontal + coarsedropout
            self.alpha = 1.
            self.cutmix_p = 0.
            self.injection_p = 0.
            self.pos_weights=1.
            self.weights=1.
            self.act_fn = "sigmoid"
            self.background_nocall = False
            self.nocall_cls = False
            self.use_secondary = False
            self.unk_cls = False # True
            self.model_from_jax = True
            self.use_bn = False #
            #self.remove_nocall = False
        
    cfg = Config()
    cfg_sed =  ConfigSED(window='hann', center=False, pad_mode='reflect', windows_size=cfg.window_size, hop_size=cfg.hop_size,
                sample_rate=cfg.sample_rate, mel_bins=cfg.mel_bins, fmin=cfg.fmin, fmax=cfg.fmax, ref=1.0, amin=1e-10, top_db=None)

    inputs = np.random.uniform(0, 1, (4, 1, 500, 256))

    inputs = torch.from_numpy(inputs).float()
    spectrogram_extractor = Spectrogram(n_fft=cfg_sed.windows_size, hop_length=cfg_sed.hop_size, 
            win_length=cfg_sed.windows_size, window=cfg_sed.window, center=cfg_sed.center, pad_mode=cfg_sed.pad_mode, 
            freeze_parameters=True)

    # Logmel feature extractor
    logmel_extractor = LogmelFilterBank(sr=cfg.sample_rate, n_fft=cfg_sed.windows_size, 
        n_mels=cfg_sed.mel_bins, fmin=cfg_sed.fmin, fmax=cfg_sed.fmax, ref=cfg_sed.ref, amin=cfg_sed.amin, 
        top_db=cfg_sed.top_db, freeze_parameters=True)
    extractor = nn.Sequential(spectrogram_extractor, logmel_extractor)

    backbone = timm.create_model('tf_efficientnet_b1_ns', pretrained=True, num_classes=0, global_pool=None, in_chans=1)
    backbone.global_pool = nn.Identity()
    in_features = shape_from_backbone(inputs=torch.as_tensor(np.random.uniform(0, 1, (1, int(cfg.max_length * cfg.sample_rate)))).float(), 
                                      model=backbone, num_channel=1, use_logmel=True, config_sed = cfg_sed.__dict__)[1]

    print(in_features)
    model = AudioSED(backbone, num_classes=[206], in_features=in_features, in_channel=1, hidden_size=1024, activation= 'sigmoid', use_logmel=True, 
            spectrogram_augmentation = None, apply_attention="step", drop_rate = [0.5, 0.5], config_sed= cfg_sed.__dict__)
   
    model.spectrogram_extractor = nn.Identity()
    model.logmel_extractor = nn.Identity()
    with torch.no_grad():
        output = model(inputs)
        print(output[0]['clipwise'].shape, output[0]['segmentwise'].shape)