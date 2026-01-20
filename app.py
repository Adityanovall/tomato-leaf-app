import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time

# ===============================
# CONFIG
# ===============================
IMG_SIZE = 224
MODEL_PATH = "student_kd_fp16.tflite"
LABEL_PATH = "labels.txt"

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
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image).astype(np.float32)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ===============================
# SOFTMAX FUNCTION (WAJIB)
# ===============================
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

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
    # OUTPUT
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
