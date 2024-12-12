import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# --- Préparation des données ---
class SignLanguageDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

DATA_PATH = r'C:\Users\frede\Desktop\TPE_2024\MP_DATA'
actions = ['hello', 'thanks', 'iloveyou', 'call', 'yes']
sequence_length = 30
no_sequences = 30

label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            try:
                res = np.load(npy_path)
                if res.shape == (1662,):
                    window.append(res)
            except Exception as e:
                print(f"Erreur chargement {npy_path}: {e}")
        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])

sequences = np.array(sequences)
labels = np.array(labels)

# Normalisation des données
mean = np.mean(sequences, axis=(0, 1))
std = np.std(sequences, axis=(0, 1))
sequences = (sequences - mean) / std

# Division des données
x_train, x_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# Conversion en Dataset PyTorch
train_dataset = SignLanguageDataset(x_train, y_train)
test_dataset = SignLanguageDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --- Définition du modèle ---
class SignLanguageModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.4):
        super(SignLanguageModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Dernier état caché
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

input_dim = sequences.shape[2]
hidden_dim = 256
output_dim = len(actions)

model = SignLanguageModel(input_dim, hidden_dim, output_dim)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# --- Entraînement ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

def train_model(model, train_loader, criterion, optimizer, num_epochs=50):
    model.train()
    train_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    return train_losses

train_losses = train_model(model, train_loader, criterion, optimizer, num_epochs=50)

all_data = []  # Contiendra toutes les séquences de vos données
for inputs, labels in train_loader:  # Utilisez votre DataLoader ici
    all_data.append(inputs.numpy())  # Convertissez en NumPy si ce n'est pas déjà le cas

# Concaténez toutes les données
train_data = np.concatenate(all_data, axis=0)  # Shape: (num_samples, num_features)

# Calcul de la moyenne et de l'écart type
mean = np.mean(train_data, axis=0)  # Moyenne pour chaque colonne (feature)
std = np.std(train_data, axis=0)    # Écart type pour chaque colonne (feature)

# Sauvegarder dans des fichiers `.npy`
np.save('mean.npy', mean)
np.save('std.npy', std)

print("Moyenne et écart type sauvegardés dans 'mean.npy' et 'std.npy'.")



# Sauvegarde du modèle
torch.save(model, 'sign_language_model.pth')

# --- Évaluation ---
def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")
    return all_labels, all_preds

all_labels, all_preds = evaluate_model(model, test_loader)

# Matrice de confusion et rapport de classification
def plot_confusion_matrix(true_labels, pred_labels, class_names):
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prédictions')
    plt.ylabel('Réel')
    plt.title('Matrice de Confusion')
    plt.show()

plot_confusion_matrix(all_labels, all_preds, actions)
print(classification_report(all_labels, all_preds, target_names=actions))

# Courbe de perte
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Courbe de Perte")
plt.legend()
plt.show()
