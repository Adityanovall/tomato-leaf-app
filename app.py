import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import cv2
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# ===============================
# CONFIG
# ===============================
IMG_SIZE = 224
MODEL_PATH = "student_kd_fp16.tflite"
LABEL_PATH = "labels.txt"
CONFIDENCE_THRESHOLD = 0.70  # Confidence threshold untuk menampilkan prediksi

# ===============================
# LOAD LABELS
# ===============================
with open(LABEL_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ===============================
# LOAD TFLITE MODEL
# ===============================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ===============================
# PREPROCESS IMAGE
# ===============================
def preprocess_image(image):
    """
    Preprocess gambar untuk inference - SESUAI TRAINING
    Menggunakan tf.keras.applications.mobilenet_v3.preprocess_input
    """
    # Resize ke 224x224 dengan BILINEAR interpolation
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    
    # Convert ke numpy array
    image_array = np.array(image, dtype=np.float32)
    
    # Apply preprocess_input dari MobileNetV3 (sama seperti saat training)
    image_array = preprocess_input(image_array)
    
    # Expand dimensi untuk batch (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

# ===============================
# PREPROCESS IMAGE - ALTERNATIF (DEPRECATED - jangan gunakan)
# ===============================
def preprocess_image_alt(image):
    """
    DEPRECATED - Gunakan preprocess_image saja
    """
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    image_array = np.array(image, dtype=np.float32)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# ===============================
# SOFTMAX FUNCTION
# ===============================
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# ===============================
# ENTROPY CALCULATION
# ===============================
def calculate_entropy(probabilities):
    """
    Hitung entropy dari probability distribution
    """
    probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
    entropy = -np.sum(probabilities * np.log(probabilities))
    return entropy

# ===============================
# GREEN LEAF DETECTION
# ===============================
def detect_green_leaf(image):
    """
    Deteksi apakah gambar mengandung daun tomat (warna hijau)
    Return: (is_valid, green_percentage)
    - Menggunakan HSV color space untuk deteksi warna hijau
    - Daun tomat umumnya memiliki warna hijau dengan saturation tinggi
    """
    # Convert PIL image ke numpy array
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert ke HSV untuk deteksi warna yang lebih baik
    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
    
    # Range untuk warna hijau dalam HSV
    # Green hue range: 35-90 (lebih luas untuk berbagai shade hijau)
    lower_green = np.array([25, 40, 40])      # Lower bound: hue, saturation, value
    upper_green = np.array([90, 255, 255])    # Upper bound
    
    # Create mask untuk warna hijau
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Hitung persentase pixel hijau
    green_pixels = cv2.countNonZero(mask)
    total_pixels = image_cv.shape[0] * image_cv.shape[1]
    green_percentage = green_pixels / total_pixels
    
    # Valid jika ada cukup pixel hijau
    is_valid = green_percentage >= GREEN_DETECTION_THRESHOLD
    
    return is_valid, green_percentage

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(
    page_title="Tomato Leaf Disease Detection",
    layout="centered"
)

st.title("🍅 Tomato Leaf Disease Detection")
st.write("MobileNetV3 + Knowledge Distillation + Quantization")

uploaded_file = st.file_uploader(
    "Upload gambar daun tomat",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Input", use_column_width=True)

    # Preprocessing
    input_data = preprocess_image(image)

    # ===============================
    # INFERENCE
    # ===============================
    interpreter.set_tensor(input_details[0]["index"], input_data)

    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time

    # Ambil logits
    logits = interpreter.get_tensor(output_details[0]["index"])[0]

    # Softmax → probabilitas
    probabilities = softmax(logits)

    prediction = np.argmax(probabilities)
    confidence = np.max(probabilities)

    # ===============================
    # VALIDATION - CEK CONFIDENCE
    # ===============================
    if confidence >= CONFIDENCE_THRESHOLD:
        # ===============================
        # OUTPUT - TAMPILKAN HASIL
        # ===============================
        st.subheader("🔍 Hasil Prediksi")
        st.write(f"**Kelas:** {class_names[prediction]}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")
        st.write(f"**Waktu Inferensi:** {inference_time:.4f} detik")

        # ===============================
        # PROBABILITY CHART
        # ===============================
        st.subheader("📊 Probabilitas Kelas")
        prob_dict = {
            class_names[i]: float(probabilities[i])
            for i in range(len(class_names))
        }
        st.bar_chart(prob_dict)
    else:
        st.warning(f"⚠️ Confidence terlalu rendah untuk prediksi yang reliable.")
        st.info(f"Confidence: {confidence * 100:.2f}% (minimum: {CONFIDENCE_THRESHOLD * 100:.0f}%)")
        st.info("💡 Silakan upload gambar lain.")
