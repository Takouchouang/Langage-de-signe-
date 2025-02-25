# Langue de signe 
Conception  d'un Système de Reconnaissance des Langues de Signes en utilisant **CNN-LSTM** en integrant a un interface Streamlit.

# Description : 
Ce projet vise à concevoir un système de reconnaissance des Langues des Signes (**LS**) en utilisant une combinaison des réseaux de neurones convolutifs (**CNN**) et des réseaux de neurones à mémoire à long terme (**LSTM**). Le modèle **CNN** est utilisé pour extraire les caractéristiques visuelles des signes, tandis que LSTM est utilisé pour capturer la dynamique temporelle des mouvements des mains et des gestes.

L'objectif est de fournir une méthode efficace pour interpréter les gestes de la langue des signes et permettre une communication fluide entre les personnes sourdes et entendantes.

# Objectifs du projet :
Créer un système de reconnaissance des gestes de la langue des signes en utilisant les architectures CNN pour l'extraction des caractéristiques d'images et LSTM pour la prédiction séquentielle des gestes.
Utiliser Mediapipe pour la détection en temps réel des repères corporels, notamment les mains et le visage, afin d'extraire les gestes pour l'entraînement et la reconnaissance.
Prévoir une haute précision de reconnaissance pour un large éventail de signes dans différentes conditions.


# Technologies utilisées : 
Python : Langage de programmation principal.
Keras/TensorFlow : Frameworks pour la construction du modèle CNN-LSTM.
Mediapipe : Utilisé pour la détection des repères du corps humain en temps réel, y compris les mains et le visage, afin de capturer les gestes de la langue des signes.
OpenCV : Pour le traitement des images et la capture en temps réel via la caméra.
Streamlit : Pour l'interface utilisateur.


 # Installation
1. Cloner le projet :
- git clone https://github.com/Takouchouang/Langage-de-signe.git
- cd langage-de-signe

2. Créer un environnement virtuel
- python -m venv env
- **Pour Linux/Mac** :
source env/bin/activate :
- **Pour Windows** :
env\Scripts\activate :
  

3. Installer les dépendances :
**pip install -r requirements.txt**

 
# Structure du projet : 

- MP_DATA : Contient les jeux de données d'entraînement et de test pour les gestes de la langue des signes.
- models : Contient les fichiers du modèle pré-entraîné CNN-LSTM.
- scripts : Scripts pour l'entraînement du modèle, l'inférence en temps réel, etc.
- interface : Contient les fichiers pour l'interface utilisateur en temps réel (Streamlit).

# Auteurs : 
- **Mr.Takouchouang Fraisse Sacre**


