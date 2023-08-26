from audio_sed.tensorflow.sed_block import SED_Block 
from audio_sed.tensorflow.sed_models import AudioSED,  shape_from_backbone, AudioClassifierSequenceSED
import tensorflow as tf
import numpy as np 
import tfimm
from audio_sed.sed_config import ConfigSED
import pytest

@pytest.mark.unittest
def test_sed_block():
    batch_size = 4
    channels = 16
    steps = 12 
    freq = 64
    num_classes = [10, 2]
    
    inputs = np.random.uniform(0, 1, (batch_size, steps, freq, channels))

    block = SED_Block(num_classes=num_classes, in_features=channels, hidden_size=1024, activation="sigmoid",
                    drop_rate=[0.5, 0.5], apply_attention="step")
    output = block(inputs, return_dict=True)
    assert len(output) == len(num_classes)
    assert type(output[0]) == dict

    output = block(inputs, return_dict=False)
    assert len(output) == len(num_classes)

@pytest.mark.unittest 
def test_model():
    batch_size = 4
    channels = 16
    steps = 12 
    freq = 64
    num_step = 3
    num_classes = [10, 2]
    
    inputs = np.random.uniform(0, 1, (batch_size, steps, freq, channels))
    backbone = tfimm.create_model('efficientnet_b0_ns', nb_classes=0)
    backbone.pool = tf.keras.layers.Activation("linear")
    backbone.flatten = tf.keras.layers.Activation("linear")
    backbone.dropout = tf.keras.layers.Activation("linear")

    inputs =  np.random.uniform(0, 1, (batch_size, 32000*5))  
    inputs2 =  np.random.uniform(0, 1, (batch_size, num_step, 32000*5)) 

    shape =  shape_from_backbone(inputs, backbone, use_logmel=True)
    model = AudioSED( backbone, num_classes=num_classes, in_features=shape[1], hidden_size=1024, activation= 'sigmoid', use_logmel=True, 
                    spectrogram_augmentation = None, apply_attention="step")

    model_seq = AudioClassifierSequenceSED(model)
    
    output = model(inputs)
    assert type(output) == list
    assert len(output) == len(num_classes)
    assert output[0].shape == (batch_size, num_classes[0]) and output[1].shape == (batch_size, num_classes[1])

    output2 = model_seq(inputs2, return_dict=True)
    print(output2[0]['output'])
    assert type(output2) == list
    assert len(output2) == len(num_classes)
    assert output2[0]['output'].shape == (batch_size, num_classes[0]) and output2[1]['output'].shape == (batch_size, num_classes[1])
    assert output2[0]['clipwise'].shape == (batch_size, num_step, num_classes[0]) and output2[1]['clipwise'].shape == (batch_size, num_step, num_classes[1])



    output2 = model_seq(inputs2, return_dict=False)
    assert type(output2) == list
    assert len(output2) == len(num_classes)
    assert output2[0].shape == (batch_size, num_classes[0]) and output2[1].shape == (batch_size, num_classes[1])


if __name__ == "__main__":
    pass