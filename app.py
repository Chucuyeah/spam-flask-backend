from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from preprocessing import preprocess
import os

app = Flask(__name__)
CORS(app)

# Load model dan vectorizer
model = joblib.load('model/svm_modellink.pkl')
vectorizer = joblib.load('model/tfidf_vectorizerlink.pkl')

# Global state
stats = {'spam': 0, 'non_spam': 0}
history = []

# Fungsi load statistik awal dari datafix.csv
def load_initial_stats_from_csv():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(BASE_DIR, 'data', 'datafix.csv')

        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='latin1')

        if 'label' not in df.columns or 'text' not in df.columns:
            print("ERROR: Dataset harus memiliki kolom 'text' dan 'label'")
            return

        stats['spam'] = 0
        stats['non_spam'] = 0

        for _, row in df.iterrows():
            label = str(row['label']).strip().lower()
            if label == 'spam':
                stats['spam'] += 1
            elif label == 'non-spam' or label == 'non_spam':
                stats['non_spam'] += 1

        print("INFO: Statistik awal berhasil dimuat dari datafix.csv ✅")

    except Exception as e:
        print(f"ERROR: Gagal memuat dataset: {e}")

# Load statistik saat startup
load_initial_stats_from_csv()

# Route prediksi 1 teks
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '').strip()  # tambahkan strip() agar deteksi spasi kosong juga

    if not text:
        return jsonify({'error': 'Kalimat belum dimasukkan'}), 400

    prep = preprocess(text)
    text_vector = vectorizer.transform([' '.join(prep['stemmed'])])
    prediction = str(model.predict(text_vector.toarray())[0])

    return jsonify({
        'text': text,
        'prediction': prediction,
        'preprocessing': prep
    })

# Route statistik
@app.route('/stats', methods=['GET'])
def get_stats():
    total = stats['spam'] + stats['non_spam']
    percent_spam = percent_non_spam = 0

    if total == 0:
        return jsonify({'error': 'Dataset tidak terdeteksi atau kosong'}), 400

    percent_spam = round((stats['spam'] / total) * 100, 2)
    percent_non_spam = round((stats['non_spam'] / total) * 100, 2)

    return jsonify({
        'spam': stats['spam'],
        'non_spam': stats['non_spam'],
        'percent_spam': percent_spam,
        'percent_non_spam': percent_non_spam
    })


@app.route('/history', methods=['GET'])
def get_history():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(BASE_DIR, 'data', 'hasil_preprocessing.csv')

        df = pd.read_csv(csv_path, encoding='utf-8')

        # Konversi isi list string ke array Python jika masih dalam bentuk string
        for col in ['token', 'stop', 'stemmed']:
            df[col] = df[col].apply(eval)  # Hati-hati: eval aman di lingkungan terkontrol seperti ini

        # Konversi dataframe ke list of dict
        history_data = []
        for _, row in df.iterrows():
            history_data.append({
                'text': row['text'],
                'cleaning': row.get('text_clean', ''),
                'tokenizing': row.get('token', []),
                'stopword': row.get('stop', []),
                'stemming': row.get('stemmed', []),
                'prediction': '1' if row['label'].lower() == 'spam' else '0'
            })

        return jsonify(history_data), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run app
if __name__ == '__main__':
    app.run(debug=True)
