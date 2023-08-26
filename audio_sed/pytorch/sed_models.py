from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from audio_sed.sed_config import ConfigSED
from audio_sed.pytorch.sed_block import SED_Block, GlobalAttention
from torch import nn 
import torch
from torch.cuda.amp import autocast

def shape_from_backbone(inputs, model, use_logmel=True, config_sed = ConfigSED().__dict__ ):
    #print(config_sed)
    sample_rate = config_sed["sample_rate"]
    window_size = config_sed["windows_size"]
    hop_size = config_sed["hop_size"]
    window = config_sed["window"]
    center = config_sed["center"]
    pad_mode = config_sed["pad_mode"]
    mel_bins = config_sed["mel_bins"]
    fmin = config_sed["fmin"]
    fmax = config_sed["fmax"]
    ref = config_sed["ref"]
    amin = config_sed["amin"]
    top_db = config_sed["top_db"]

    spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

    # Logmel feature extractor
    logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
        n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, 
        top_db=top_db, freeze_parameters=True)

    with torch.no_grad():
        x =  spectrogram_extractor(inputs)   # (batch_size, 1, time_steps, freq_bins)
        print("sepctrogram :", x.shape)
        if use_logmel:
            with autocast(False):
                x =  logmel_extractor(x) # (batch_size, 1, time_steps, mel_bins)
                print("logmel:", x.shape)
        x = model(torch.cat([x, x, x], dim=1))
    return x.shape


class AudioClassifierSequenceSED(nn.Module):
    def __init__(self, backbone):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(AudioClassifierSequenceSED, self).__init__()        
        self.audio_backbone = backbone
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.global_attention = nn.ModuleList([GlobalAttention(n) for n in backbone.num_classes])
        self.num_output = len(backbone.num_classes)

    def init_output(self ):
        outs = [] 
         
        for _ in range(self.num_output):
            outs.append({})
        
        return outs

    def merge_output(self, features ):
        # features: list of dict {"clipwise":clipwise_output, "segmentwise":segmentwise_output, "norm_att":norm_att}
        outs = self.init_output()

        # iterate throught sequences
        
        for feat in features: # seq
            for i, feat_class in enumerate(feat): # class
                    for k, v in feat_class.items():
                        if k not in outs[i]:
                            outs[i][k] = []
                        outs[i][k].append(v)
        
        for i, o in enumerate(outs): # class
            for k in o.keys():
                outs[i][k] = torch.stack(outs[i][k], axis=1)
        return outs


    def forward(self, input, mixup_fn=None ):
        """Input: (batch_size, length)
        """
        device = self.dummy_param.device

        features = []
        for i in range(input.shape[1]):
            o = self.audio_backbone(input[:, i].to(device), mixup_fn)
            features.append(o)

        features = self.merge_output(features )
        
        for i, global_attn in enumerate(self.global_attention):
            features[i]["output"], features[i]["attention_global"] = global_attn(features[i]["clipwise"])
            

        return features

class AudioSED(nn.Module):
    def __init__(self, backbone, num_classes:list, in_features:int, hidden_size=1024, activation= 'sigmoid', use_logmel=True, 
                spectrogram_augmentation = None, apply_attention="step", drop_rate = [0.5, 0.5], config_sed:dict = ConfigSED().__dict__):
        """
            Classifier for a new task using pretrained CNN as a sub module.
        """
        super(AudioSED, self).__init__()    
        self.num_classes = num_classes
        self.use_logmel = use_logmel
        self.sample_rate = config_sed["sample_rate"]
        self.window_size = config_sed["windows_size"]
        self.hop_size = config_sed["hop_size"]
        self.window = config_sed["window"]
        self.center = config_sed["center"]
        self.pad_mode = config_sed["pad_mode"]
        self.mel_bins = config_sed["mel_bins"]
        self.fmin = config_sed["fmin"]
        self.fmax = config_sed["fmax"]
        self.ref = config_sed["ref"]
        self.amin = config_sed["amin"]
        self.top_db = config_sed["top_db"]
        self.spectrogram_augmentation = spectrogram_augmentation
        self.bn = nn.BatchNorm2d(self.mel_bins)
        # Spectrogram extractor 
        self.spectrogram_extractor = Spectrogram(n_fft=self.window_size, hop_length=self.hop_size, 
            win_length=self.window_size, window=self.window, center=self.center, pad_mode=self.pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=self.sample_rate, n_fft=self.window_size, 
            n_mels=self.mel_bins, fmin=self.fmin, fmax=self.fmax, ref=self.ref, amin=self.amin, 
            top_db=self.top_db, freeze_parameters=True)

        self.backbone =  backbone 
   
        self.sed_block = SED_Block(num_classes=num_classes, in_features=in_features, hidden_size=hidden_size, 
                activation = activation, drop_rate = drop_rate,   apply_attention=apply_attention)
                
    def forward(self, input, mixup_fn=None ):
        """Input: (batch_size, length, data_length)
        """
        with torch.no_grad():
            x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
            #print("sepctrogram :", x.shape)
            if self.use_logmel:
                with autocast(False):
                    x = self.logmel_extractor(x) # (batch_size, 1, time_steps, mel_bins)


        # (BS, C=1, H, W)
        frames_num = x.shape[2]   
        x = x.transpose(1, 3)
        x = self.bn(x) # BN applied on melbins
        x = x.transpose(1, 3)
        
        if self.training and self.spectrogram_augmentation:
            x = self.spectrogram_augmentation(x)
        
        # Mixup on spectrogram
        if self.training and mixup_fn is not None:
            x = mixup_fn(x)  

        if x.shape[1] == 1:        
            x = torch.cat([x,x,x], dim=1)
        
        x = self.backbone(x) # (batch size, channels, steps, freq)
        
        outputs = self.sed_block(x)

        return outputs