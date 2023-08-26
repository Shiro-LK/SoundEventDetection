from tensorflow.keras import initializers, Sequential
from tensorflow.keras.layers import Dropout, Permute, Dense,   AveragePooling1D, MaxPooling1D
import tensorflow as tf 
from typing import List, Optional, Tuple, Union

class AttBlock(tf.keras.layers.Layer):
    def __init__(self, n_in, n_out, activation='linear', temperature=1.,**kwargs):
        super(AttBlock, self).__init__(**kwargs)
        
        self.activation = activation
        self.temperature = temperature
        self.att = tf.keras.layers.Conv1D(filters=n_out, kernel_size=1, strides=1, padding="valid", use_bias=True, name="att", #activation="tanh",
                                          bias_initializer=initializers.Zeros(), kernel_initializer=tf.initializers.GlorotUniform())
        self.cla = tf.keras.layers.Conv1D(filters=n_out, kernel_size=1, strides=1, padding="valid", use_bias=True, activation=activation, name="cla", 
                                          bias_initializer=initializers.Zeros(), kernel_initializer=tf.initializers.GlorotUniform())
        
        #self.bn_att = tf.keras.layers.BatchNormalization(name="bn_att", beta_initializer=initializers.Zeros(), gamma_initializer=initializers.Ones())#(n_out)

         
    def call(self, x, training=False):
        # x: (n_samples, step, channel)
        att =   tf.clip_by_value(self.att(x), -10, 10)  
        norm_att = tf.nn.softmax(att, axis=1) # apply softmax on step dim
        cla = self.cla(x)
        x = tf.math.reduce_sum(norm_att * cla, axis=1)
        return x, norm_att, cla

        
class GlobalAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_size, **kwargs):
        super(GlobalAttention , self).__init__(**kwargs)
        self.W = tf.keras.layers.Dense(hidden_size, use_bias=False, name="W")

    def call(self, h_i, training=False): # (bs, step, proba)
        u_i = self.W(h_i) # (bs, step, proba)
        a_i =  tf.nn.softmax(u_i, axis=1)#µ/2.0
        v = tf.math.reduce_sum( a_i * h_i, axis=1) 
        return v, a_i

class Identity(tf.keras.layers.Layer):
    def __init__(self ):
        super(Identity , self).__init__()
        self.act = tf.keras.layers.Activation("linear")
    def call(self, inputs):
        return self.act(inputs)

class SED_Block(tf.keras.layers.Layer):
    def __init__(self, num_classes:List[int], in_features:int, hidden_size:int = 1024, activation:int = "sigmoid",
                drop_rate:List[int] = [0.5, 0.5], apply_attention="step"):
        """Apply SED block on output of a backbone.
        """
        super(SED_Block, self).__init__()
        assert apply_attention in ["step", "channel"]
        self.apply_attention = apply_attention
        self.num_classes = num_classes
        #self.dropout = nn.Dropout(drop_rate[0])
        self.max_pool_1d = MaxPooling1D(pool_size=3, strides=1, padding="same")
        self.avg_pool_1d = AveragePooling1D(pool_size=3, strides=1, padding="same")
        self.blocks = []
        self.attn_blocks = []
        for num_class in self.num_classes:
            # 1 ) att on step => (bs, channel, step) = > transpose => (bs, step, channel) = > linear => (bs, step, hidden size)
            # => transpose => (bs, hidden size, step) => attblock on hstep => (bs , num_classes, step )
            # 2 ) att on channels => (bs, channel, step) = > linear => (bs, channel, hidden size)
            # => transpose => (bs, hidden size, channel) => attblock on channel => (bs , num_classes, channel  ) => (bs , num_classes,   )
            self.blocks.append(Sequential([Dropout(drop_rate[0]),
                                             Permute((1,2)) if self.apply_attention == "step" else Identity(), 
                                            Dense(hidden_size, activation="relu"),
                                            Permute((1,2)) ,
                                            Dropout(drop_rate[1])]))
            self.attn_blocks.append(AttBlock(hidden_size,  num_class, activation=activation, temperature=1.))

    def call(self, inputs, return_dict=False):
        """
            inputs: (Batch size, steps, freq, channels)
            @return: List[num_classes] each element of the list following this format : 
                    {'clipwise':(BATCH SIZE, n), 'segmentwise':(BATCH SIZE, step, n),
                    'norm_att':(BATCH SIZE, step)}
        """
        outputs = []
        x = tf.reduce_mean(inputs, axis=2) # Average freq features
    
        # smooth channels
        x1 = self.max_pool_1d(x) # pool on steps
        x2 = self.avg_pool_1d(x)
        x = x1 + x2

        for attn_block, block in zip(self.attn_blocks, self.blocks):
            o = block(x)
            (clipwise_output, norm_att, segmentwise_output) = attn_block(o)
            segmentwise_output = tf.transpose(segmentwise_output, perm=[0,2,1]) # batch size, steps, channels
            if return_dict:
                outputs.append({"clipwise":clipwise_output, "segmentwise":segmentwise_output, "norm_att":norm_att})
            else:
                outputs.append(clipwise_output)
        return outputs