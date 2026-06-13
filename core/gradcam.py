from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
from core.preprocessing import preprocess_visual_image
from core.predictions import get_model, preprocess_image

# Base directories
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "plots" / "GradCAM"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output file used by the GUI
OUTPUT_FILE = OUTPUT_DIR / "latest_gradcam.png"



# -----------------------------------------------------------------------------
# Create heatmap
# -----------------------------------------------------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if pred_index is None:

            # Always visualize fracture class
            pred_index = 0

        class_channel = predictions[:, pred_index]

    # Gradients of the target class with respect to conv outputs
    grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling over height and width
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by importance
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        tf.multiply(conv_outputs, pooled_grads),
        axis=-1
    )

    # Normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)

    if max_val == 0:
        return np.zeros(heatmap.shape, dtype=np.float32)

    heatmap = heatmap / max_val
    return heatmap.numpy()


# -----------------------------------------------------------------------------
# Apply heatmap to original image
# -----------------------------------------------------------------------------
def overlay_heatmap(original_image_path, heatmap, alpha=0.25):
    from matplotlib import cm

    # Load original image
    original_np = preprocess_visual_image(
        original_image_path
    ).astype(np.float32)

    # Resize heatmap to match image size
    heatmap_img = Image.fromarray(
        np.uint8(heatmap * 255)
    ).resize(
        (original_np.shape[1], original_np.shape[0])
    )

    heatmap_np = np.array(heatmap_img).astype(np.float32) / 255.0

    # Apply INFERNO colormap (RGB values in range [0, 1])
    colored_heatmap = cm.get_cmap("inferno")(heatmap_np)[..., :3]

    # Convert to [0, 255]
    colored_heatmap = (colored_heatmap * 255).astype(np.float32)

    # Blend original image and heatmap
    blended = original_np * (1 - alpha) + colored_heatmap * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # Save result
    result = Image.fromarray(blended)
    result.save(OUTPUT_FILE)

    return OUTPUT_FILE

def save_gradcam(model_type, img_path, output_path):
    """
    Generate and save GradCAM visualization.
    """

    # Load model
    model = get_model(model_type)

    # Preprocess image
    img_array = preprocess_image(img_path)

    # Find last conv layer automatically
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            last_conv_layer_name = layer.name
            break

    if last_conv_layer_name is None:
        raise ValueError("No convolutional layer found.")

    # Generate heatmap
    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        last_conv_layer_name
    )

    # Load original image
    original_np = preprocess_visual_image(
        img_path
    ).astype(np.float32)

    # Resize heatmap
    heatmap_img = Image.fromarray(
        np.uint8(heatmap * 255)
    ).resize(
        (original_np.shape[1], original_np.shape[0])
    )

    heatmap_np = np.array(heatmap_img).astype(np.float32) / 255.0

    # Apply colormap
    from matplotlib import cm

    colored_heatmap = cm.get_cmap("inferno")(heatmap_np)[..., :3]
    colored_heatmap = (colored_heatmap * 255).astype(np.float32)

    # Blend
    blended = original_np * 0.6 + colored_heatmap * 0.4
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # Save
    result = Image.fromarray(blended)
    result.save(output_path)

    return str(output_path)

# -----------------------------------------------------------------------------
# Main function used by GUI
# -----------------------------------------------------------------------------
def generate_gradcam(image_path, model_type):
    """
    Generate Grad-CAM visualization for an image.

    Parameters
    ----------
    image_path : str
        Path to the X-ray image.

    model_type : str
        "Parts", "Elbow", "Hand", or "Shoulder"

    Returns
    -------
    str
        Path to the generated Grad-CAM image.
    """
    # Load model using existing project utilities
    model = get_model(model_type)

    # Preprocess image using the same preprocessing as prediction
    img_array = preprocess_image(image_path)

    # Automatically locate last convolutional layer
    last_conv_layer_name = None
    last_conv_layer_name = "conv5_block16_concat"

    # Generate heatmap
    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        last_conv_layer_name,
    )

    # Overlay on original image and save
    output_path = overlay_heatmap(image_path, heatmap)

    return str(output_path)


# -----------------------------------------------------------------------------
# Standalone test
# -----------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    test_image = input("Enter image path: ").strip()
    model_name = input("Enter model type (Parts/Elbow/Hand/Shoulder): ").strip()

    output = generate_gradcam(test_image, model_name)
    print(f"Grad-CAM saved to: {output}")