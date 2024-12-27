import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import os
from pathlib import Path
path = ""
class CustomDepthwiseConv2D(layers.DepthwiseConv2D):
    def __init__(self, *args, groups=None, **kwargs):
        # Loại bỏ tham số groups nếu có
        super().__init__(*args, **kwargs)
        
    def get_config(self):
        config = super().get_config()
        if 'groups' in config:
            del config['groups']
        return config

def load_fish_model():
    """
    Load pretrained fish classification model with custom DepthwiseConv2D layer
    """
    try:
        # Clear Keras session
        K.clear_session()
        
        # Define custom objects
        custom_objects = {
            'DepthwiseConv2D': CustomDepthwiseConv2D
        }
        
        # Load model with custom objects
        basePath = Path(__file__).parent
        basePath = Path(__file__).parent
        filePath = os.path.join(basePath, 'saved_model', 'fish_mobilenet.h5')
        model = load_model(
            filePath,
            custom_objects=custom_objects,
            compile=False
        )
        
        print("Model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print(f"Failed to load model. Check path: {filePath}")

        # Raise lỗi để biết chi tiết vấn đề
        raise e