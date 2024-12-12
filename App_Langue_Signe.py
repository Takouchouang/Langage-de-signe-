import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time
import csv
from playsound import playsound

# Charger le modèle
model = load_model(r'C:\Users\frede\Desktop\TPE_2024\action.h5')  # Chemin vers votre modèle .h5

# Dictionnaire d'actions avec des icônes pour chaque geste
actions = {
    'hello': '🖐️',
    'thanks': '🙏',
    'iloveyou': '🤟',
    'call': '🤲',
    'yes': '👍',
    
}

# Dimensions pour redimensionner l'image
width, height = 1662, 1662

# Configurer l'interface Streamlit
st.set_page_config(page_title="Reconnaissance Avancée de Langue des Signes", page_icon="🖐️")

# En-tête avec logos et titre centralisé
st.markdown(
    """
    <style>
    .title { font-size: 1.8rem; font-weight: bold; color: #4a4a4a; text-align: center; }
    .flash { animation: flash 0.5s ease-in-out; color: #FF6347; font-weight: bold; }
    .pred-box { background-color: #f8f9fa; padding: 10px; border-radius: 5px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); }
    .captured-image { width: 100%; max-width: 500px; margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True
)

# Colonnes pour le logo et le titre
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.image("C:/Users/frede/Desktop/TPE_2024/Projet Tpe/logo_vnu.jpg", width=100)
with col2:
    st.markdown("<div class='title'>Reconnaissance Avancée de Langue des Signes</div>", unsafe_allow_html=True)
with col3:
    st.image("C:/Users/frede/Desktop/TPE_2024/Projet Tpe/logo_ifi.jpg", width=100)

# Introduction de l'application
st.markdown(
    "<p style='text-align: center; color: #6c757d; font-size: 1.1rem;'>"
    "Cette application utilise l'apprentissage profond pour reconnaître et interpréter des signes en temps réel."
    "</p>", unsafe_allow_html=True
)

# Sidebar avec Guide d'Utilisation déroulant
st.sidebar.title("Guide d'Utilisation")
with st.sidebar.expander("Cliquez ici pour le guide d'utilisation"):
    st.write("""
    1. **Démarrer la Webcam** : Activez la webcam pour lancer la détection en temps réel des gestes.
    2. **Voir la Prédiction** : Observez la prédiction de geste affichée sur l'image en direct.
    3. **Arrêter la Webcam** : Arrêtez la détection en temps réel à tout moment en cliquant sur ce bouton.
    4. **Options Avancées** :
        - **Activer le son** : Joue un son lorsque le geste est détecté.
        - **Capture automatique d'image** : Capture automatiquement une image pour chaque geste détecté.
        - **Sauvegarder l'historique des prédictions** : Sauvegarde chaque prédiction dans un fichier CSV pour un historique.
    """)

st.sidebar.header("Résultat de la Prédiction")
prediction_placeholder = st.sidebar.empty()

# Options avancées
st.sidebar.markdown("### Options avancées")
play_sound = st.sidebar.checkbox("Activer le son pour chaque nouveau geste")
auto_capture = st.sidebar.checkbox("Activer la capture automatique d'image")
save_to_csv = st.sidebar.checkbox("Sauvegarder l'historique des prédictions")

# Historique des prédictions
st.sidebar.header("Historique des gestes détectés")
history_placeholder = st.sidebar.empty()
predictions_history = []

# Démarrer la webcam
placeholder = st.empty()

# Boutons de démarrage, d'arrêt et de capture d'images
col1, col2, col3 = st.columns(3)
with col1:
    start_webcam = st.button("Démarrer la Webcam")
with col2:
    stop_webcam = st.button("Arrêter la Webcam")
with col3:
    capture_image = st.button("Capturer Image")

if start_webcam:
    cap = cv2.VideoCapture(0)
    stframe = placeholder.empty()
    previous_prediction = None
    captured_image = None
    save_path = "gesture_predictions.csv"

    # Initialiser le fichier CSV pour sauvegarder les prédictions
    if save_to_csv:
        with open(save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Geste détecté"])

    while start_webcam and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Erreur: Impossible de lire la vidéo.")
            break

        frame_resized = cv2.resize(frame, (width, height))
        gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY) / 255.0
        gray_frame = np.expand_dims(img_to_array(gray_frame), axis=0)

        # Prédiction
        prediction = model.predict(gray_frame)
        predicted_action = actions[np.argmax(prediction)]

        # Notification sonore et flash si la prédiction change
        if predicted_action != previous_prediction:
            if play_sound:
                playsound('sound_notification.mp3')
            prediction_placeholder.markdown(
                f"<div class='flash'><b>Geste détecté :</b> {predicted_action}</div>",
                unsafe_allow_html=True
            )
            previous_prediction = predicted_action

            # Sauvegarder dans l'historique
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            predictions_history.append(f"{timestamp} - {predicted_action}")
            history_placeholder.write("\n".join(predictions_history))

            # Enregistrer l'historique dans un fichier CSV
            if save_to_csv:
                with open(save_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, predicted_action])

        else:
            prediction_placeholder.markdown(
                f"<div class='pred-box'><b>Geste détecté :</b> {predicted_action}</div>",
                unsafe_allow_html=True
            )

        # Afficher le flux vidéo avec la prédiction
        cv2.putText(frame, f'{predicted_action}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Capture automatique d'image
        if auto_capture:
            img_name = f"gesture_{predicted_action}_{timestamp}.png"
            cv2.imwrite(img_name, frame)
            captured_image = img_name
            st.sidebar.write(f"Image capturée automatiquement sous {img_name}")

        # Afficher le flux vidéo
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        # Arrêter la webcam
        if stop_webcam:
            break

    cap.release()
    cv2.destroyAllWindows()
    placeholder.empty()
    st.sidebar.write("La webcam est arrêtée.")
else:
    placeholder.write("Cliquez sur **Démarrer la Webcam** pour lancer la reconnaissance des gestes.")

# Pied de page avec bibliographie et photo
st.markdown("---")
col1, col2 = st.columns([1, 4])

with col1:
    st.image("C:/Users/frede/Desktop/TPE_2024/Projet Tpe/photo.jfif", width=100)

with col2:
    st.markdown(
        """
        <div style='text-align: left; font-size: 0.9rem; color: #6c757d;'>
        Application développée par Ing.Takouchouang Fraisse Sacre, Étudiant à l'IFI - Promotion 27.<br><br>
        <b>Bibliographie :</b><br>
        - [1] Takouchouang Fraisse Sacre.<br>
        - [2] Camerounaise.<br>
        - [3] Etudiant en Master I a l'IFI(Institut Francophone International) , en Informatique option Systeme Intelligent et Multimedia(SIM), en double Diplomation , de la Promotion 27.
        </div>
        """, unsafe_allow_html=True
    )
