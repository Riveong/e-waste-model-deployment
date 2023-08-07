import numpy as np
import tensorflow as tf
import numpy as np
import typing
from PIL import Image
from datetime import datetime


model = tf.keras.models.load_model("model/xception_latest.h5")
class_names = [
    "Battery",
    "Cable",
    "CRT TV",
    "E-kettle",
    "Refrigerator",
    "Keyboard",
    "Laptop",
    "Light Bulb",
    "Monitor",
    "Mouse",
    "PCB",
    "Phone",
    "Printer",
    "Rice Cooker",
    "Washing Machine",
]


def preprocess_image(image: Image) -> np.ndarray:
    """Preprocesses an image for prediction.

    Args:
        image (PIL.Image): The image to be preprocessed.

    Returns:
        np.ndarray: The preprocessed image as a numpy array.
    """
    image = image.resize((300, 300))
    image_array = np.array(image)
    image_array = tf.keras.applications.xception.preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array




def predict_image(image: np.ndarray) -> dict:
    """Predicts the top 3 classes for an image using a pre-trained model.

    Args:
        image (np.ndarray): An image represented as a NumPy array.

    Returns:
        dict: A dictionary containing the top 3 predicted classes and their probabilities.
    """
    predictions = model.predict(image)
    top_3_indices = np.argsort(predictions)[0, -3:][::-1]

    data_list = []
    for i in top_3_indices:
        data_predictions = {}
        data_predictions["Components Name"] = (class_names[i])
        data_predictions["Components Value"] = (predictions[0, i] * 100)
        data_list.append(data_predictions)

    return data_list


def timer(start_time: datetime = None) -> "typing.Union[datetime.datetime, str]":
    """
    Measures the time elapsed from a given start time.
    If no start time is provided, returns the current time. If a start time is provided, returns a formatted string
    representing the time elapsed from the start time to the current time.
    Args:
        start_time (datetime.datetime, optional): The start time to measure elapsed time from, or None to get the current time. Defaults to None.
    Returns:
        Union[datetime.datetime, str]: The current time if no start time is provided, or a formatted string representing the elapsed time.
    """
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        return "%i hours %i minutes and %s seconds." % (
            thour,
            tmin,
            round(tsec, 2),
        )
