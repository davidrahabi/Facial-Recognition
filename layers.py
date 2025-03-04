# Simamese L1 distance class
import tensorflow as tf
from tensorflow.keras.layers import Layer
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
    
    def call(self, input_embedding, validation_embedding):
        """
        Takes anchor embedding, and pos/neg embedding
        Returns similarity calculation between two images
        """
        return(tf.math.abs(input_embedding - validation_embedding))