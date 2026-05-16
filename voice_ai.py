import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# 1. SET PATH DATASET LOKAL
# ==========================================
# Karena folder 'dataset' satu lokasi dengan file notebook ini, cukup tulis nama foldernya
DATASET_PATH = "processed_audio" 

# ==========================================
# 1. BUSINESS UNDERSTANDING
# ==========================================
# Tujuan: Mendeteksi apakah sebuah sampel audio adalah suara asli (Real) atau hasil kloning AI (Fake).
# Masalah: Meningkatnya penyalahgunaan AI untuk penipuan berbasis suara (Deepfake Voice).
# Metrik: Akurasi dan Recall (terutama meminimalkan False Negative pada suara Fake).

# ==========================================
# 2. DATA UNDERSTANDING & FEATURE EXTRACTION
# ==========================================
def load_and_extract_features(dataset_path):
    data = []
    labels = []
    
    for label_name, label_val in [('real', 0), ('fake', 1)]:
        folder_path = os.path.join(dataset_path, label_name)
        if not os.path.exists(folder_path):
            print(f"Peringatan: Folder {folder_path} tidak ditemukan!")
            continue
            
        print(f"Sedang mengekstrak fitur dari folder: {label_name}...")
        files = os.listdir(folder_path)
        for file_name in files:
            if file_name.endswith(('.wav', '.mp3', '.m4a')):
                file_path = os.path.join(folder_path, file_name)
                try:
                    # Muat audio 
                    audio, sr = librosa.load(file_path, duration=3, res_type='kaiser_fast')
                    # Ekstrak MFCC (Mel-frequency cepstral coefficients)
                    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                    mfccs_processed = np.mean(mfccs.T, axis=0)
                    
                    data.append(mfccs_processed)
                    labels.append(label_val)
                except Exception as e:
                    print(f"Gagal memproses {file_name}: {e}")
                    
    return np.array(data), np.array(labels)

X, y = load_and_extract_features(DATASET_PATH)

if X.size == 0:
    print("Error: Dataset kosong.")
    exit()

print(f"\nSukses! Total data: {X.shape[0]} sampel.")

# ==========================================
# 3. VISUALISASI ANALISIS (SEBELUM MODELING)
# ==========================================
print("\nMenghasilkan visualisasi analisis...")

# A. Bar Chart: Distribusi Kelas (Melihat Ketidakseimbangan Data)
plt.figure(figsize=(6, 4))
counts = [np.sum(y == 0), np.sum(y == 1)]
plt.bar(['Real', 'Fake'], counts, color=['green', 'red'])
plt.title('Distribusi Kelas Audio')
plt.ylabel('Jumlah Sampel')
plt.show() # Menampilkan grafik langsung
# plt.savefig('distribusi_kelas.png')

# B. Histogram: Distribusi MFCC (Melihat perbedaan karakteristik fitur)
plt.figure(figsize=(8, 5))
plt.hist(X[y==0][:, 0], bins=30, alpha=0.5, label='Real', color='green')
plt.hist(X[y==1][:, 0], bins=30, alpha=0.5, label='Fake', color='red')
plt.title('Histogram Distribusi Fitur MFCC Pertama')
plt.legend()
plt.show() # Menampilkan grafik langsung
# plt.savefig('distribusi_mfcc.png')

# C. Boxplot (Representasi Grup & Statistik): Membandingkan rata-rata fitur (Mirip ANOVA)
plt.figure(figsize=(8, 5))
df_box = pd.DataFrame({'MFCC_1': X[:, 0], 'Label': ['Real' if i==0 else 'Fake' for i in y]})
sns.boxplot(x='Label', y='MFCC_1', data=df_box)
plt.title('Boxplot Fitur MFCC: Real vs Fake')
plt.show() # Menampilkan grafik langsung
# plt.savefig('boxplot_comparison.png')

# ==========================================
# 4. MODELING & EVALUATION
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n--- HASIL EVALUASI MODEL ---")
print(f"Akurasi: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake'], zero_division=0))

# D. Heatmap: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.title('Confusion Matrix - Deteksi Voice')
plt.show() # Menampilkan grafik langsung
# plt.savefig('confusion_matrix.png')

print("\nAnalisis selesai.")