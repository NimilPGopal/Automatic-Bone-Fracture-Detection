import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings(
    "ignore",
    category=UserWarning
)

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow_addons as tfa
from core.preprocessing import  clahe_preprocessing
from core.config import (
    WEIGHTS_DIR,
    IMAGE_SIZE,
    THRESHOLDS,
    CATEGORIES_PARTS,
    MODEL_FILES
)

# Model Cache
MODELS: dict[str, tf.keras.Model] = {}

# Helper Function: Load a Model
def load_model(model_name: str):
    if model_name not in MODELS:
        model_path = WEIGHTS_DIR / model_name

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please train the models first or place them in the 'weights' folder."
            )

        print(f"[INFO] Loading model: {model_path.name}")

        MODELS[model_name] = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "F1Score": tfa.metrics.F1Score
            },
            compile=False
        )

    return MODELS[model_name]


# Helper Function: Select Appropriate Model
def get_model(model_type: str = "Parts"):

    if model_type not in MODEL_FILES:
        raise ValueError(
            f"Invalid model type: {model_type}. "
            f"Valid options are: {list(MODEL_FILES.keys())}"
        )

    return load_model(MODEL_FILES[model_type])


# Helper Function: Preprocess Image
def preprocess_image(img_path: str):

    """
    Load and preprocess image for DenseNet121 prediction.
    """

    temp_img = load_img(img_path)

    x = img_to_array(temp_img)

    x = clahe_preprocessing(x)

    x = np.expand_dims(x, axis=0)

    return x


# Main Prediction Function
def predict(
    img_path: str,
    model: str = "Parts",
    return_confidence: bool = False,
    return_probs: bool = False
):

   
    img_array = preprocess_image(img_path)
    model_instance = get_model(model)

    prediction_probs = model_instance.predict(
        img_array,
        verbose=0
    )[0]

    # -----------------------------------------
    # BODY PART CLASSIFIER
    # -----------------------------------------
    if model == "Parts":

        predicted_index = np.argmax(prediction_probs)

        prediction_label = CATEGORIES_PARTS[predicted_index]

        confidence = float(
            prediction_probs[predicted_index]
        )

    # -----------------------------------------
    # FRACTURE CLASSIFIER
    # -----------------------------------------
    else:

        fractured_prob = float(prediction_probs[0])
        normal_prob = float(prediction_probs[1])

        threshold = THRESHOLDS.get(model, 0.5)

        predicted_index = np.argmax(prediction_probs)

        if predicted_index == 0:
            prediction_label = "fractured"
            confidence = fractured_prob
        else:
            prediction_label = "normal"
            confidence = normal_prob

    confidence_percent = round(confidence * 100, 2)

    print("Prediction:", prediction_label)
    print("Confidence:", confidence_percent, "%")

    if not return_confidence and not return_probs:
        return prediction_label

    if return_confidence and not return_probs:
        return prediction_label, confidence_percent

    return prediction_label, confidence_percent, prediction_probs