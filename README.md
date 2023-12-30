# Sound Event Detection

This repository implement a class which allow to build classifier for audio signal following the SED (Sound Event Detection) architecture.
The model create as well mel spectrogram and use CNN backbone.
This structure has been used during the birdcall competition which allow me to reach the 3th position in 2021.

This is implemented on pytorch and tensorflow, however the tensorflow version has not been tested for training purpose (only inference). I would advice to use only the pytorch version.

# Installation 

> pip install audio-sed

if you use tensorflow:

> pip install tflibrosa

# Examples

## Pytorch
```{python}
import torch
import timm
from torch import nn
import numpy as np
from audio_sed.sed_config import ConfigSED
from audio_sed.pytorch.sed_models import AudioClassifierSequenceSED, AudioSED, shape_from_backbone

def load_model(model_name, num_classe, cfg_sed):
    backbone = timm.create_model( model_name, pretrained=False)
    if "efficientnet" in   model_name:
        backbone.global_pool =  nn.Identity()
        in_feat = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
    elif "convnext" in cfg.model_name:
        in_feat = backbone.head.fc.in_features
        backbone.head = nn.Identity()

    in_features = shape_from_backbone(inputs=torch.as_tensor(np.random.uniform(0, 1, (1, int(5 * cfg_sed.sample_rate)))).float(), model=backbone, 
                                      use_logmel=True, config_sed = cfg_sed.__dict__)[2] # (batch size, channels, num_steps, y_axis) 
    print("Num timestamps features:",in_features)
    model = AudioSED(backbone, num_classes=[num_classe], in_features=in_feat, hidden_size=1024, activation= 'sigmoid', use_logmel=True, 
                spectrogram_augmentation = None, apply_attention="step", drop_rate = [0.5, 0.5], config_sed= cfg_sed.__dict__)

    model2 = AudioClassifierSequenceSED(model)
    
    return model, model2

cfg_sed =  ConfigSED(window='hann', center=True, pad_mode='reflect', windows_size=1024, hop_size=320,
                sample_rate=32000, mel_bins=128, fmin=50, fmax=16000, ref=1.0, amin=1e-10, top_db=None)

model_5, model = load_model(model_name="tf_efficientnet_b2_ns", num_classe=575, cfg_sed=cfg_sed)
inputs = torch.as_tensor(np.random.uniform(0,1, (20*32000)).reshape(1,4,-1)).float()
with torch.no_grad():
    o = model(inputs)
print(o[0]['clipwise'])
```

## Tensorflow 

```{python}
import tensorflow as tf
from audio_sed.sed_config import ConfigSED
from audio_sed.tensorflow.sed_models import AudioClassifierSequenceSED as AudioClassifierSequenceSEDTF, AudioSED as AudioSEDTF, shape_from_backbone as shape_from_backboneTF



def load_modeltf(model_name, num_classe, cfg_sed):
    backbone =tf.keras.applications.efficientnet.EfficientNetB2(
    include_top=False)   
    if "efficientnet" in   model_name:
        in_feat = backbone.layers[-1].output.shape[-1] 
    elif "convnext" in cfg.model_name:
        in_feat = backbone.layers[-1].output.shape[-1]
    # batch size, num_steps, y_axis, channels
    in_features = shape_from_backboneTF(inputs=np.random.uniform(0, 1, (1, int(5 * cfg_sed.sample_rate))), model=backbone, use_logmel=True, config_sed = cfg_sed.__dict__)[1]
    print("Num timestamps features:",in_features)
    model = AudioSEDTF(backbone, num_classes=[num_classe], in_features=in_feat, hidden_size=1024, activation= 'sigmoid', use_logmel=True, 
                spectrogram_augmentation = None, apply_attention="step", drop_rate = [0.5, 0.5], config_sed= cfg_sed.__dict__)

    model = AudioClassifierSequenceSEDTF(model)
    
    return model

cfg_sed =  ConfigSED(window='hann', center=True, pad_mode='reflect', windows_size=1024, hop_size=320,
                sample_rate=32000, mel_bins=128, fmin=50, fmax=16000, ref=1.0, amin=1e-10, top_db=None)

inputs = np.random.uniform(0,1, (1, 4, 5*32000))
o_tf = model_tf.predict(inputs)
print(o_tf[0])

```

# Examples 2:

You can find [here](https://github.com/Shiro-LK/Portfolio-project/tree/main/BirdsCall_Detection) an example of how this model is used with a GUI for inference with some checkpoint available.



# Citation 

- [PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://arxiv.org/abs/1912.10211)
- PANNs model: https://github.com/qiuqiangkong/audioset_tagging_cnn
- https://github.com/Shiro-LK/tflibrosa