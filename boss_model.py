import cv2
import keras
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.utils.generic_utils import CustomObjectScope
# Load pretrained model

with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    model = load_model('./model/model.h5')

IMAGE_SIZE=224

def prepare_image(img):
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

def predict(img):
    """
    Predict face crop from frame
    :param img:
    :return: If boss is appear when open the code IDE
    """
    try:
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        probs = model.predict(prepare_image(img))
        is_boss = np.argmax(probs[0])
        return is_boss
    except:
        return False
