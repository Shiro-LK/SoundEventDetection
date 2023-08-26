import numpy as np 
import random
import math

class PermutationChunks(object):
    def __init__(self, min_, max_, sr=32000):
        self.min_ = min_
        self.max_ = max_
        self.sr = sr 
    
    def __call__(self, samples):
        resize = False 
        shape = samples.shape
        if len(shape) == 2:
            samples = samples.reshape(-1,)
            resize = True
        N = int(math.floor(len(samples)/self.sr) )

        vals = np.random.uniform(self.min_, self.max_, (N,))
        chunks = []
        i = 0
        for v in vals:
            t = int(v * self.sr)
            chunks.append(samples[i:i+t])
            i = i+t
            if i > len(samples):
                break
        random.shuffle(chunks)
        chunks = np.concatenate(chunks)
        if resize:
            chunks = chunks.reshape(*shape)
        return chunks 