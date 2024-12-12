import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Modèles Holistic et utilitaires de dessin de MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Fonction pour utiliser MediaPipe pour la détection
def mediapipe_detection(image, model):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results
    except Exception as e:
        print(f"Erreur lors du traitement de l'image avec MediaPipe: {e}")
        return image, None

# Fonction pour dessiner les landmarks avec style
def draw_styled_landmarks(image, results):
    try:
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_face.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
            )
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
            )
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
            )
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
    except Exception as e:
        print(f"Erreur lors du dessin des landmarks: {e}")

# Fonction pour extraire les points clés en 3D (x, y, z uniquement)
def extract_keypoints(results):
    try:
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(99)  # 33 landmarks
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
        return np.concatenate([pose, face, lh, rh])
    except Exception as e:
        print(f"Erreur lors de l'extraction des points clés: {e}")
        return np.zeros(1662)  # Valeur par défaut en cas d'erreur

# Chemin des données exportées
DATA_PATH = os.path.join('MP_DATA')

# Actions à détecter
actions = np.array(['hello', 'thanks', 'iloveyou', 'call', 'yes'])

# Nombre de séquences et longueur
no_sequences = 30
sequence_length = 30

# Création des dossiers si nécessaires
for action in actions:
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

# Capture vidéo
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        print(f"Collecte pour l'action: {action}")
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    print("Échec de la capture de la frame")
                    break

                # Détection MediaPipe
                image, results = mediapipe_detection(frame, holistic)

                # Dessin des landmarks
                draw_styled_landmarks(image, results)

                # Affichage
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.imshow('Flux OpenCV', image)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, f'Collecting frames for {action} video {sequence}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                cv2.imshow('Flux OpenCV', image)

                # Extraction des points clés
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                np.save(npy_path, keypoints)

                # Sortie si 'q'
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()

# Création des labels et séquences
label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            res = np.load(npy_path)
            if res.shape != (1662,):  # Vérification pour x, y, z uniquement
                print(f"Problème à l'action '{action}', séquence {sequence}, frame {frame_num}")
                res = np.zeros(1662)
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# Conversion en arrays
sequences = np.array(sequences)
labels = np.array(labels)
print("Forme des séquences:", sequences.shape)
print("Forme des labels:", labels.shape)

# One-hot encoding des labels
y = to_categorical(labels).astype(int)

# Split des données
x_train, x_test, y_train, y_test = train_test_split(sequences, y, test_size=0.5, random_state=42)
print("Formes après séparation:")
print("x_train:", x_train.shape, "y_train:", y_train.shape)
print("x_test:", x_test.shape, "y_test:", y_test.shape)
