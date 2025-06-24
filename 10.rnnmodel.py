import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, render_template, flash
import os
import pickle

# --- Konfigurasi dan Inisialisasi Aplikasi Flask ---
app = Flask(__name__)
app.secret_key = 'supersecretkey' # Diperlukan untuk flash messages

# --- Bagian Pelatihan Model RNN (dijalankan sekali) ---
# Fungsi ini akan melatih dan menyimpan model jika belum ada.
def train_and_save_model():
    # 1. Mempersiapkan Dataset (sesuai dari dokumen) 
    texts = ["this is a good movie", "this is a bad movie", "i love this film", "i hate this film"]
    labels = np.array([1, 0, 1, 0]) # Label target (1: positif, 0: negatif) 

    # 2. Encoding teks menjadi angka (Tokenization) 
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Simpan tokenizer untuk digunakan nanti saat prediksi
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # 3. Padding sequence agar memiliki panjang yang sama 
    padded_sequences = pad_sequences(sequences, maxlen=5, padding='post')

    # 4. Membuat Model RNN (sesuai dari dokumen) 
    vocab_size = len(tokenizer.word_index) + 1  # Ukuran kosakata 
    embedding_dim = 8                          # Dimensi embedding 
    rnn_units = 16                             # Jumlah unit dalam layer RNN 
    output_units = 1                           # Jumlah neuron output untuk klasifikasi biner 
    input_length = 5                           # Panjang sekuens input 

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length), # 
        SimpleRNN(rnn_units, activation='tanh'), # 
        Dense(output_units, activation='sigmoid') # 
    ])

    # 5. Compile Model 
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', # 
                  metrics=['accuracy']) # 

    # 6. Train Model
    # Dalam aplikasi nyata, data latih akan lebih banyak.
    # Di sini kita menggunakan semua data untuk pelatihan.
    model.fit(padded_sequences, labels, epochs=20, batch_size=2, verbose=0)
    
    # 7. Simpan model yang sudah dilatih
    model.save('rnn_sentiment_model.h5')
    print("Model dan tokenizer berhasil dilatih dan disimpan.")


# --- Fungsi untuk Memuat Model dan Tokenizer ---
def load_prediction_assets():
    try:
        model = load_model('rnn_sentiment_model.h5')
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except (IOError, EOFError) as e:
        print(f"Error loading assets: {e}")
        return None, None

# --- Cek apakah model sudah ada, jika tidak, latih dulu ---
if not os.path.exists('rnn_sentiment_model.h5') or not os.path.exists('tokenizer.pickle'):
    print("Model atau tokenizer tidak ditemukan. Memulai proses pelatihan...")
    train_and_save_model()

# --- Memuat model dan tokenizer ke memori ---
model, tokenizer = load_prediction_assets()
if model is None or tokenizer is None:
    print("Gagal memuat model atau tokenizer. Aplikasi tidak dapat berjalan dengan benar.")
    # Hentikan aplikasi jika aset tidak dapat dimuat
    exit()

# --- Rute Aplikasi Flask ---
@app.route('/', methods=['GET'])
def index():
    # Menampilkan halaman utama dengan form input
    return render_template('rnn.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Mengambil teks dari form input
        input_text = request.form['text']
        
        if not input_text.strip():
            flash("Input tidak boleh kosong!")
            return render_template('rnn.html')

        # 1. Preprocessing input teks sesuai dengan data latih
        sequence = tokenizer.texts_to_sequences([input_text])
        padded_sequence = pad_sequences(sequence, maxlen=5, padding='post')

        # 2. Melakukan prediksi
        prediction = model.predict(padded_sequence)
        prediction_proba = prediction[0][0]

        # 3. Menentukan hasil sentimen
        # Threshold 0.5: jika probabilitas > 0.5 -> Positif, jika tidak -> Negatif
        if prediction_proba > 0.5:
            sentiment = "Positif"
        else:
            sentiment = "Negatif"

        # Menampilkan halaman hasil
        return render_template('rnnresult.html', text=input_text, sentiment=sentiment, score=prediction_proba)

# --- Menjalankan Aplikasi Flask ---
if __name__ == '__main__':
    app.run(debug=True)