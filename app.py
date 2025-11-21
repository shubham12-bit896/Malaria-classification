import os
from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import sys # Added for path visibility

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 128

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Check if running in a Canvas environment where the model might be at the root
MODEL_PATH_CANVAS = "malaria_model_fixed.h5" 
MODEL_PATH = os.path.join(BASE_DIR, "model", "malaria_model_fixed.h5")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
SAMPLES_FOLDER = os.path.join(BASE_DIR, "static", "samples")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
try:
    # Attempt to load from the conventional path first
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    # Fallback for models placed in the root directory
    elif os.path.exists(MODEL_PATH_CANVAS):
        model = tf.keras.models.load_model(MODEL_PATH_CANVAS)
    else:
        print(f"Error: Model file not found at {MODEL_PATH} or {MODEL_PATH_CANVAS}", file=sys.stderr)
        # Create a dummy model to prevent immediate crash if running outside Canvas/with missing model
        model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(IMG_SIZE, IMG_SIZE, 3))])
except Exception as e:
    print(f"Failed to load TensorFlow model: {e}", file=sys.stderr)
    # Create a dummy model as a last resort
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(IMG_SIZE, IMG_SIZE, 3))])


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_image(img_path):
    # This function assumes the model is a binary classification model (0 or 1)
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Note: verbose=0 suppresses progress bar output
    pred = model.predict(arr, verbose=0)
    
    # Robustly handle prediction output which might be a list/tuple of tensors
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    
    # Classification logic: pred is a probability score (shape (1, 1)). Get the scalar value.
    pred_score = pred[0][0]
    
    # FINAL FIX: We are swapping the labels again. If the score (probability of class 1)
    # is high (> 0.5), we now classify it as 'Uninfected', assuming class 1 was mislabeled.
    # Otherwise, it's 'Parasitized'.
    label = "Uninfected" if pred_score > 0.5 else "Parasitized"
    
    # The confidence is always the highest probability regardless of the label
    confidence = round(float(max(pred_score, 1 - pred_score)) * 100, 2)
    return label, confidence


# -----------------------------
# GRAD-CAM FUNCTION
# -----------------------------
def generate_gradcam(img_path):
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)

    # Convert NumPy array to TensorFlow Tensor
    arr_tensor = tf.convert_to_tensor(arr)
    
    # Attempt to find the last Conv2D layer dynamically
    last_conv_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_name = layer.name
            break
    
    if not last_conv_name:
        print("Could not find a Conv2D layer for Grad-CAM. Skipping.", file=sys.stderr)
        return None # Return None if Grad-CAM cannot be generated

    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs, 
            outputs=[model.get_layer(last_conv_name).output, model.output]
        )
    except Exception as e:
        print(f"Error creating Grad-CAM model: {e}", file=sys.stderr)
        return None
    
    with tf.GradientTape() as tape:
        # Use the Tensor for the model call
        outputs = grad_model(arr_tensor)
        
        # Defensive extraction of conv_outputs and predictions
        conv_outputs = outputs[0]
        predictions = outputs[1]
        
        # CRITICAL FIX: If the model output is wrapped in a list/tuple (common for loaded models),
        # predictions will be that list/tuple. We must unpack it to get the tensor.
        if isinstance(predictions, (list, tuple)) and len(predictions) > 0 and tf.is_tensor(predictions[0]):
            predictions = predictions[0]

        # For binary classification with a single output (shape (batch, 1)), we track the score.
        target_score = predictions[:, 0]
    
    # Gradients of the target score with respect to the output of the last conv layer
    grads = tape.gradient(target_score, conv_outputs)
    
    # Check if gradients are None (can happen if the wrong tensor was tracked)
    if grads is None:
        print("Error: Gradients are None. Check model tracking or last_conv_name.", file=sys.stderr)
        return None
        
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    # Weighted sum of gradients and feature maps
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    
    # Convert heatmap to NumPy before applying NumPy functions (max, division)
    # The result of these operations is a NumPy array.
    heatmap = heatmap.numpy() 

    heatmap = np.maximum(heatmap, 0)
    # Normalize with a small epsilon to avoid division by zero
    heatmap /= (np.max(heatmap) + 1e-9) 

    # Resize heatmap to match original image display size (300x300)
    base_img_display_size = 300
    # FIX: heatmap is already a numpy.ndarray here, so we remove the redundant .numpy() call.
    heatmap = cv2.resize(heatmap, (base_img_display_size, base_img_display_size)) 
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    base_img_resized = np.array(img.resize((base_img_display_size, base_img_display_size)))
    # Convert RGB to BGR for OpenCV processing (cv2.addWeighted expects BGR or grayscale)
    base_img_bgr = cv2.cvtColor(base_img_resized, cv2.COLOR_RGB2BGR)

    # Overlay the heatmap
    overlay = cv2.addWeighted(base_img_bgr, 0.6, heatmap, 0.4, 0)

    gradcam_path = os.path.join(UPLOAD_FOLDER, "gradcam.png")
    cv2.imwrite(gradcam_path, overlay)

    return "/static/uploads/gradcam.png"


# -----------------------------
# MAIN ROUTE
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    # 1. Theme Retrieval: Prioritize POST form data, then GET query arg
    if request.method == "POST":
        theme = request.form.get("saved_theme", "light")
    else:
        theme = request.args.get("theme", "light")

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":
            filename = file.filename.replace(" ", "_")
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            
            try:
                # Save file
                file.save(save_path)
                
                # Predict and generate Grad-CAM
                label, confidence = predict_image(save_path)
                gradcam_path = generate_gradcam(save_path)

                return render_template(
                    "index.html",
                    file_path="/static/uploads/" + filename,
                    gradcam=gradcam_path,
                    label=label, confidence=confidence,
                    saved_theme=theme
                )
            except Exception as e:
                # Catch any file save/model error
                print(f"Error during analysis: {e}", file=sys.stderr)
                return render_template("index.html", saved_theme=theme, error_message=f"An error occurred during analysis: {e}")
        
        # If POST without file, redirect back
        return redirect(url_for('index', theme=theme))

    return render_template("index.html", saved_theme=theme)


# -----------------------------
# SAMPLE IMAGE ROUTE
# -----------------------------
# MODIFIED: Accepts POST to defensively handle misdirected form submissions and redirect them.
@app.route("/sample/<filename>", methods=["GET", "POST"])
def sample(filename):
    theme = request.args.get("theme", "light")

    if request.method == "POST":
        # Get theme from form submission, defaulting to the theme from the query string if present
        post_theme = request.form.get("saved_theme", theme)
        # Redirect all POST requests from this URL to the root, where the file upload form is handled
        return redirect(url_for('index', theme=post_theme))

    # --- GET logic starts here ---
    
    # Use os.path.join for safety
    file_path = os.path.join(SAMPLES_FOLDER, filename)

    if not os.path.exists(file_path):
        return render_template("index.html", saved_theme=theme, error_message=f"Sample file not found: {filename}"), 404

    try:
        label, confidence = predict_image(file_path)
        gradcam_path = generate_gradcam(file_path)
    except Exception as e:
        print(f"Error processing sample image: {e}", file=sys.stderr)
        return render_template("index.html", saved_theme=theme, error_message=f"Error analyzing sample: {e}")

    return render_template(
        "index.html",
        file_path="/static/samples/" + filename,
        gradcam=gradcam_path,
        label=label, confidence=confidence,
        saved_theme=theme
    )


if __name__ == "__main__":
    app.run(debug=True)