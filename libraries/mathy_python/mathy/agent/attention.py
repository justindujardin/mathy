"""TensorFlow compatible import of SeqSelfAttention"""
import os

# required to tell library to use TF backend
os.environ["TF_KERAS"] = "1"
from keras_self_attention import SeqSelfAttention
