# Spam Detection Flask Backend

API deteksi spam berbahasa Indonesia menggunakan Machine Learning (SVM) dan NLP (TF-IDF).

## 📁 Struktur Folder

```
spam-flask-backend/
├── app.py                      # Flask API utama
├── preprocessing.py            # Modul preprocessing teks (cleaning, tokenizing, stopword, stemming)
├── requirements.txt            # Dependensi Python
├── Spam.ipynb                  # Notebook training model (opsional)
├── data/
│   ├── datafix.csv             # Dataset mentah dengan label spam/non-spam
│   └── hasil_preprocessing.csv # Dataset hasil preprocessing untuk history
├── model/
│   ├── svm_modellink.pkl       # Model SVM yang sudah dilatih
│   ├── tfidf_vectorizerlink.pkl # Vectorizer TF-IDF
│   ├── svm_modelasli.pkl       # Model SVM asli (backup)
│   ├── tfidf_vectorizerasli.pkl # Vectorizer TF-IDF asli (backup)
│   └── df_cleaned.pkl          # Data cleaned (backup)
└── static/                     # Folder untuk file statis (jika diperlukan)
```

## 🚀 Fitur

- **Prediksi Spam**: Deteksi spam pada teks berbahasa Indonesia
- **Preprocessing Lengkap**: Cleaning, tokenizing, stopword removal, dan stemming
- **Statistik Dataset**: Melihat jumlah dan persentase spam vs non-spam
- **History**: Melihat riwayat preprocessing dari dataset

## 🛠️ Teknologi

- **Flask**: Web framework Python
- **Scikit-learn**: Machine Learning (SVM Classifier)
- **NLTK**: Tokenizing dan stopword removal bahasa Indonesia
- **Sastrawi**: Stemming bahasa Indonesia
- **Pandas**: Manipulasi data
- **Joblib**: Serialisasi model

## 📦 Instalasi

1. Clone repository:
```bash
git clone <repository-url>
cd spam-flask-backend
```

2. Install dependensi:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (jika belum):
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## 🔧 Menjalankan Server

```bash
python app.py
```

Server akan berjalan di `http://127.0.0.1:5000`

## 📡 API Endpoints

### 1. Prediksi Spam

**Endpoint**: `POST /predict`

**Request Body**:
```json
{
  "text": "isi teks yang akan dideteksi"
}
```

**Response**:
```json
{
  "text": "teks asli",
  "prediction": "spam" atau "non-spam",
  "preprocessing": {
    "cleaned": "teks setelah cleaning",
    "tokens": ["token1", "token2"],
    "removed_stopwords": ["stopword1"],
    "stemmed": ["kata dasar1", "kata dasar2"]
  }
}
```

### 2. Statistik Dataset

**Endpoint**: `GET /stats`

**Response**:
```json
{
  "spam": 1500,
  "non_spam": 2000,
  "percent_spam": 42.86,
  "percent_non_spam": 57.14
}
```

### 3. History Preprocessing

**Endpoint**: `GET /history`

**Response**:
```json
[
  {
    "text": "teks asli",
    "cleaning": "teks cleaned",
    "tokenizing": ["token1", "token2"],
    "stopword": ["stop1"],
    "stemming": ["stem1"],
    "prediction": "1"
  }
]
```

## 📝 Preprocessing Pipeline

1. **Cleaning**:
   - Hapus URL dan ganti dengan "link"
   - Hapus tag HTML
   - Hapus karakter khusus dan angka
   - Lowercase

2. **Tokenizing**: Pecah teks menjadi kata-kata

3. **Stopword Removal**: Hapus kata-kata umum bahasa Indonesia

4. **Stemming**: Ubah kata ke bentuk dasar menggunakan Sastrawi

## ⚙️ Konfigurasi

- **Model**: `model/svm_modellink.pkl`
- **Vectorizer**: `model/tfidf_vectorizerlink.pkl`
- **Dataset**: `data/datafix.csv`
- **History**: `data/hasil_preprocessing.csv`

## 🔒 CORS

API ini sudah dikonfigurasi dengan CORS enabled untuk mengizinkan request dari frontend react/lainnya

## 📄 License

Project ini merupakan bagian dari Skripsi - Deteksi Spam Berbahasa Indonesia
