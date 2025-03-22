import cv2
import streamlit as st
from ultralytics import YOLO
import os
import numpy as np
from PIL import Image
from plyer import notification  # Bildirim gönderme kütüphanesi
import threading # Bildirimi Ayrı Bir İş Parçacığında Çalıştırma
# from win10toast import ToastNotifier # is designed for Windows notifications, it does not work properly in a Linux-based or remote environment


# Modeli yükleme
model_path = "best.pt"
model = YOLO(model_path)

# Yardımcı fonksiyonlar
def get_output_filename(input_path, suffix="_output"):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    extension = os.path.splitext(input_path)[1]
    return f"{base_name}{suffix}{extension}"

# Stil tanımlamaları

# CSS dosyasını yükleme
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        """ 
        <style>
        [data-testid="stSidebar"] {
            background-color: #001f3f; /* Lacivert Arka Plan */
            color: white;
        }
        .sidebar-logo {
            border: 2px solid white;  /* Beyaz anahat */
            padding: 0px;
            margin: 0px;  /* Kenar boşluğu */
            display: flex;
            justify-content: center;
            align-items: center;
            height:2px;
        }
        .stRadio>div>label {
            color: white !important;  /* Radio buton etiketlerinin rengi beyaz */
            font-size: 18px !important;  /* Font boyutunu büyütme */
        }

        
        </style>
        """, 
        unsafe_allow_html=True
    )



def display_logo():
    if os.path.exists('welder.png'):
        # Logo'yu div içine yerleştiriyoruz
        st.sidebar.markdown('<div class="sidebar-logo">', unsafe_allow_html=True)
        st.sidebar.image('welder.png', use_container_width=True)
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
    else:
        st.sidebar.warning("Logo file couldn't find.")

def upload_file():
    uploaded_file = st.file_uploader("Please upload a file.", type=["mp4", "avi", "mov", "jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            # Temp klasörünü oluştur
            os.makedirs("temp", exist_ok=True)
            
            # Orijinal dosya adını koru
            original_filename = uploaded_file.name
            temp_file_path = os.path.join("temp", original_filename)
            
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            return temp_file_path, uploaded_file.type
        except Exception as e:
            st.error(f"An error occurred during file processing: {e}")
            return None, None
    return None, None

# Yardımcı Fonksiyon: Kötü kaynak tespiti ve bildirim gönderme
def check_bad_resource(frame):
    # Model tahminini yap
    results = model(frame, conf=0.25)

    # Tahmin sonuçları
    boxes = results[0].boxes
    confidences = boxes.conf
    labels = boxes.cls  # Sınıf etiketlerini alalım (iyi veya kötü kaynak)

    # Kötü kaynakları tespit et
    for label, confidence in zip(labels, confidences):
        if label == 1:  # 0, kötü kaynak sınıfı olduğunu varsayıyoruz
            if confidence >= 0.8:  # Eğer güven skoru 0.70 veya daha yüksekse
                return True  # Kötü kaynak tespit edildi
    return False  # Kötü kaynak tespit edilmedi

# Yardımcı Fonksiyon: Bildirim gönderme
# Bildirim gönderme fonksiyonu
def send_windows_notification(message):
    notification.notify(
        title="Welding Quality Warning",
        msg=message,
        timeout=3,  # Bildirim 10 saniye görünecek        
    )
    # Yeni bir thread başlat ve bildirimi çalıştır
    notification_thread = threading.Thread(target=notify)
    notification_thread.start()

# Anlık tahmin için video akışı işleme
def predict_with_video_streaming(video_path, send_notification=False):
    try:
        st.success("Predictions are being made through the video...")

        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("Video failed to turn on!")
            return

        # Ekrana video ve tahmin çıktısını göndermek için Streamlit'e bir konteyner oluşturuyoruz
        frame_placeholder = st.empty()

        while cap.isOpened() and not st.session_state.get('stop_prediction', False):
            ret, frame = cap.read()
            if not ret:
                break

            # Model tahminini yap
            results = model(frame, conf=0.25)

            # Sonuçları üzerine çizme
            annotated_frame = results[0].plot()

            # Frame'i RGB'ye çevirip Streamlit'e gönder
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Streamlit konteyneri ile her frame'i anlık olarak güncelle
            frame_placeholder.image(annotated_frame_rgb, channels="BGR", use_container_width=True)

            # Kötü kaynak tespiti ve bildirim gönderme
            bad_resource_detected = check_bad_resource(frame)
            if bad_resource_detected and send_notification:
                send_windows_notification("Bad welding detected!")  # Bildirim gönder

        cap.release()

    except Exception as e:
        st.error(f"An error occurred during video processing: {str(e)}")

# Görsel üzerinden tahmin yapmak
def predict_with_image(image_path, send_notification=False):
    try:
        st.success("Predictions are being made on the image...")

        image = Image.open(image_path)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Model tahminini yap
        results = model(image, conf=0.25)

        # Sonuçları üzerine çizme
        annotated_image = results[0].plot()

        # Görseli Streamlit'e gönder
        st.image(annotated_image, channels="BGR", use_container_width=True)

        # Kötü kaynak tespiti ve bildirim gönderme
        bad_resource_detected = check_bad_resource(image)
        if bad_resource_detected and send_notification:
            send_windows_notification("Bad welding detected!")  # Bildirim gönder

    except Exception as e:
        st.error(f"An error occurred during image processing: {str(e)}")

# Webcam ile canlı tahmin yapma
def predict_with_webcam(send_notification=False):
    try:
        st.success("Predictions are being made through the webcam...")

        cap = cv2.VideoCapture(0)  # Webcam açılır
        
        if not cap.isOpened():
            st.error("Webcam failed to turn on!")
            return

        # Ekrana video ve tahmin çıktısını göndermek için Streamlit'e bir konteyner oluşturuyoruz
        frame_placeholder = st.empty()

        while True and not st.session_state.get('stop_prediction', False):
            ret, frame = cap.read()
            if not ret:
                break

            # Model tahminini yap
            results = model(frame, conf=0.25)

            # Sonuçları üzerine çizme
            annotated_frame = results[0].plot()

            # Frame'i RGB'ye çevirip Streamlit'e gönder
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Streamlit konteyneri ile her frame'i anlık olarak güncelle
            frame_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)

            # Kötü kaynak tespiti ve bildirim gönderme
            bad_resource_detected = check_bad_resource(frame)
            if bad_resource_detected and send_notification:
                send_windows_notification("Bad welding detected!")  # Bildirim gönder

        cap.release()

    except Exception as e:
        st.error(f"An error occurred during webcam operation: {str(e)}")

def main():
    st.title("Welding Quality Control Application")

    # Temp klasörünü oluştur
    os.makedirs("temp", exist_ok=True)

    display_logo()

    uploaded_file, file_type = upload_file()

    # Sidebar'da bildirimleri açma/kapatma seçeneği     
    send_notification = st.sidebar.radio("Sending Notifications", ("Yes", "No")) == "Yes"

    if uploaded_file is not None:
        if file_type.startswith("image"):
            # Görsel yüklenmişse
            st.image(uploaded_file, use_container_width=True)
            if st.button("Predict Image"):
                predict_with_image(uploaded_file, send_notification)
        elif file_type.startswith("video"):
            # Video yüklenmişse
            st.video(uploaded_file)  # Video burada ekranda oynar
            if st.button("Predict Video"):
                predict_with_video_streaming(uploaded_file, send_notification)

    # Webcam üzerinden tahmin yapmak için buton
    webcam_button = st.button("Live Prediction")
    if webcam_button:
        predict_with_webcam(send_notification)

if __name__ == "__main__":
    main()