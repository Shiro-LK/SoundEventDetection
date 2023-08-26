from audio_sed.pytorch.sed_block import SED_Block 
from audio_sed.pytorch.sed_models import AudioSED,  shape_from_backbone, AudioClassifierSequenceSED
import torch 
from torch import nn 
import numpy as np 
import timm
from audio_sed.sed_config import ConfigSED
import pytest


@pytest.mark.unittest
def test_sed_block_pt():
    # (Batch size, channels, steps, freq)

    batch_size = 4
    channels = 16
    steps = 12 
    freq = 64
    num_classes = [10, 2]
    inputs = torch.as_tensor(np.random.uniform(0, 1, (batch_size, channels, steps, freq))).float()
    block = SED_Block(num_classes=num_classes, in_features=channels, hidden_size=1024, activation="sigmoid",
                    drop_rate=[0.5, 0.5], apply_attention="step")
    with torch.no_grad():
        output = block(inputs)
    assert len(output) == len(num_classes)
    assert type(output[0]) == dict

@pytest.mark.unittest
def test_model_pt():
    batch_size = 4
    channels = 16
    steps = 12 
    freq = 64
    num_classes = [10, 2]
    num_step = 3
    inputs = torch.as_tensor(np.random.uniform(0, 1, (batch_size, channels, steps, freq))).float()

    backbone = timm.create_model('tf_efficientnet_b0.ns_jft_in1k', num_classes=0)
    backbone.global_pool = nn.Identity()

    inputs = torch.as_tensor(np.random.uniform(0, 1, (batch_size, 32000*5))).float()
    inputs2 = torch.as_tensor(np.random.uniform(0, 1, (batch_size, num_step, 32000*5))).float()

    shape =  shape_from_backbone(inputs, backbone, use_logmel=True)
    model = AudioSED( backbone, num_classes=num_classes, in_features=shape[1], hidden_size=1024, activation= 'sigmoid', use_logmel=True, 
                    spectrogram_augmentation = None, apply_attention="step")

    model_seq = AudioClassifierSequenceSED(model)
    with torch.no_grad():
        output = model(inputs)
        output2 = model_seq(inputs2)

    assert type(output) == list
    assert len(output) == len(num_classes)
    assert output[0]['clipwise'].shape == (batch_size, num_classes[0]) and output[1]['clipwise'].shape == (batch_size, num_classes[1])

    assert type(output2) == list
    assert len(output2) == len(num_classes)
    assert output2[0]['output'].shape == (batch_size, num_classes[0]) and output2[1]['output'].shape == (batch_size, num_classes[1])
    assert output2[0]['clipwise'].shape == (batch_size, num_step, num_classes[0]) and output2[1]['clipwise'].shape == (batch_size, num_step, num_classes[1])

if __name__ == "__main__":
    pass