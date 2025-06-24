import os
import re
import string
import numpy as np
from flask import Flask, render_template, request, jsonify
import nltk
from gensim.models import Word2Vec, FastText, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.decomposition import PCA

# Mengatur backend Matplotlib agar tidak memerlukan GUI
# Ini penting untuk menjalankan Flask di lingkungan server
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Mengunduh tokenizer 'punkt' dari NLTK jika belum ada
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Mengunduh paket 'punkt' NLTK...")
    nltk.download('punkt')
    print("Unduhan selesai.")

# --- Persiapan Data dan Model ---

# Deskripsi 15 kalimat untuk data training
# Karakter kutip non-standar telah diganti dengan yang standar
deskripsi_model = [
    "Word2Vec adalah model pembelajaran representasi kata yang menggunakan neural network.",
    "Model ini menciptakan vektor kata berdasarkan konteks kata dalam kalimat.",
    "Word2Vec memiliki dua arsitektur utama: CBOW dan Skip-gram.",
    "GloVe (Global Vectors) mengandalkan matriks ko-occurrence kata di korpus besar.",
    "GloVe menggabungkan informasi global dan lokal untuk menghasilkan embedding.",
    "FastText mengembangkan Word2Vec dengan memperhitungkan sub-kata atau n-gram.",
    "Hal ini membuat FastText mampu merepresentasikan kata baru atau jarang muncul.",
    "Word embedding memungkinkan pemrosesan bahasa alami lebih efektif dalam ML.",
    "Model-model ini membantu memahami semantik dan hubungan antar kata.",
    "GloVe cocok untuk korpus besar dan menghasilkan embedding yang stabil.",
    "FastText unggul dalam menangani kata yang tidak ada dalam data pelatihan.",
    "Word2Vec sering digunakan untuk berbagai aplikasi NLP seperti analisis sentimen.",
    "FastText juga digunakan dalam klasifikasi teks dan tugas bahasa lainnya.",
    "Ketiga model ini adalah fondasi embedding kata modern dalam NLP.",
    "Pemilihan model bergantung pada kebutuhan dan karakteristik data."
]

# Tokenisasi dan normalisasi data teks
tokens_list = []
for kalimat in deskripsi_model:
    # Mengubah ke huruf kecil dan melakukan tokenisasi
    tokens = nltk.tokenize.word_tokenize(kalimat.lower())
    # Membersihkan token dari tanda baca dan karakter non-alfanumerik
    bersih = [t for t in tokens if t not in string.punctuation and re.match(r'^\w+$', t)]
    tokens_list.append(bersih)

# --- Training Model Word2Vec ---
print("Training model Word2Vec...")
model_w2v = Word2Vec(
    sentences=tokens_list,
    vector_size=100,  # Ukuran vektor
    window=5,         # Jarak antara kata target dan konteks
    min_count=1,      # Abaikan kata dengan frekuensi di bawah 1
    sg=0,             # 0 untuk CBOW, 1 untuk Skip-gram
    workers=4,        # Jumlah thread worker
    epochs=20         # Jumlah iterasi training
)
print("Training Word2Vec selesai.")

# --- Training Model FastText ---
print("Training model FastText...")
model_ft = FastText(
    sentences=tokens_list,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,             # Menggunakan Skip-gram untuk FastText
    epochs=20,
    min_n=3,          # Panjang n-gram minimal
    max_n=6,          # Panjang n-gram maksimal
)
print(f"Training FastText selesai. Ukuran kosakata: {len(model_ft.wv.index_to_key)}")

# --- Pemuatan Model GloVe ---
# GloVe memiliki format berbeda, jadi kita perlu mengonversinya ke format Word2Vec terlebih dahulu
glove_file = "glove.6B.300d.txt"
word2vec_output_file = glove_file + '.word2vec'
model_glove = None

# Cek apakah file GloVe yang sudah dikonversi ada
if os.path.isfile(word2vec_output_file):
    print(f"Memuat model GloVe dari file yang sudah dikonversi: {word2vec_output_file}")
    model_glove = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    print("Model GloVe berhasil dimuat.")
# Jika tidak ada, cek apakah file GloVe asli ada untuk dikonversi
elif os.path.isfile(glove_file):
    print(f"File GloVe asli ditemukan. Mengonversi {glove_file} ke format Word2Vec...")
    try:
        glove2word2vec(glove_file, word2vec_output_file)
        print("Konversi berhasil. Memuat model GloVe...")
        model_glove = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
        print("Model GloVe berhasil dimuat.")
    except Exception as e:
        print(f"Gagal mengonversi atau memuat model GloVe: {e}")
# Jika file GloVe asli tidak ditemukan
else:
    print(f"PERINGATAN: File GloVe '{glove_file}' tidak ditemukan.")
    print("Silakan unduh dari: https://nlp.stanford.edu/data/glove.6B.zip")
    print(f"Ekstrak dan letakkan file '{glove_file}' di direktori yang sama dengan 'app.py'.")

# --- Menggabungkan Kosakata untuk Tampilan di UI ---
vocab_w2v = set(model_w2v.wv.index_to_key)
vocab_ft = set(model_ft.wv.index_to_key)
combined_vocab = vocab_w2v.union(vocab_ft)

# Hanya tambahkan kosakata GloVe jika model berhasil dimuat
if model_glove:
    # Kita batasi jumlah kosakata GloVe agar halaman tidak terlalu berat
    glove_vocab_sample = list(model_glove.index_to_key)[:2000] 
    combined_vocab = combined_vocab.union(set(glove_vocab_sample))

combined_vocab = sorted(list(combined_vocab))

# --- Fungsi untuk Visualisasi PCA ---
def generate_pca_plot(words, model, model_name):
    vectors = []
    valid_words = []
    for w in words:
        try:
            # Dapatkan vektor kata. Strukturnya berbeda untuk model Gensim vs KeyedVectors
            vec = model.wv[w] if hasattr(model, 'wv') else model[w]
            vectors.append(vec)
            valid_words.append(w)
        except KeyError:
            print(f"Kata '{w}' tidak ditemukan di model {model_name} untuk PCA.")
            continue
    
    # PCA memerlukan setidaknya 2 titik data
    if len(vectors) < 2:
        return None
        
    try:
        # Kurangi dimensi vektor ke 2D menggunakan PCA
        pca = PCA(n_components=2)
        reduced_vectors = pca.fit_transform(vectors)
        
        plt.figure(figsize=(10, 8))
        for i, word in enumerate(valid_words):
            x, y = reduced_vectors[i][0], reduced_vectors[i][1]
            plt.scatter(x, y)
            plt.annotate(word, (x, y), fontsize=12, ha='right')
            
        plt.title(f"Visualisasi PCA 2D untuk Kata Serupa ({model_name.title()})")
        
        # Pastikan direktori 'static' ada
        static_dir = 'static'
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)
            
        # Simpan plot sebagai gambar PNG
        path = os.path.join(static_dir, f"pca_{model_name}.png")
        plt.savefig(path)
        plt.close() # Tutup plot untuk melepaskan memori
        return path
    except Exception as e:
        print(f"Terjadi kesalahan saat membuat plot PCA: {e}")
        return None

# --- Rute-rute Flask (API Endpoints) ---

@app.route('/')
def index():
    """Menampilkan halaman utama dengan daftar kosakata."""
    return render_template('WordVectorGloveFastText.html', vocabulary=combined_vocab)

@app.route('/get_vector', methods=['POST'])
def get_vector():
    """Mengambil vektor untuk sebuah kata."""
    word = request.form['word'].lower().strip()
    model_name = request.form['model']
    model = {'word2vec': model_w2v, 'fasttext': model_ft, 'glove': model_glove}.get(model_name)
    
    if not model:
        return jsonify({'success': False, 'error': 'Model yang dipilih tidak tersedia atau gagal dimuat.'})
    
    try:
        vec = model.wv[word] if hasattr(model, 'wv') else model[word]
        # Mengembalikan 10 elemen pertama untuk tampilan dan vektor penuh untuk data
        return jsonify({
            'success': True, 
            'vector_preview': vec[:10].tolist(), 
            'vector_full': vec.tolist()
        })
    except KeyError:
        return jsonify({'success': False, 'error': f'Kata "{word}" tidak ditemukan dalam model {model_name}.'})
    except Exception as e:
        print(f"Error di /get_vector: {e}")
        return jsonify({'success': False, 'error': 'Terjadi kesalahan internal.'})

@app.route('/get_similar', methods=['POST'])
def get_similar():
    """Mencari kata-kata yang paling mirip."""
    word = request.form['word'].lower().strip()
    model_name = request.form['model']
    model = {'word2vec': model_w2v, 'fasttext': model_ft, 'glove': model_glove}.get(model_name)

    if not model:
        return jsonify({'success': False, 'error': 'Model yang dipilih tidak tersedia atau gagal dimuat.'})

    try:
        # Dapatkan 5 kata paling mirip
        sim_words_tuples = model.wv.most_similar(word, topn=5) if hasattr(model, 'wv') else model.most_similar(word, topn=5)
        
        # Buat daftar kata untuk plot PCA (kata asli + kata-kata mirip)
        words_for_plot = [word] + [w for w, score in sim_words_tuples]
        
        # Hasilkan plot PCA
        plot_path = generate_pca_plot(words_for_plot, model, model_name)
        
        return jsonify({
            'success': True, 
            'similar_words': sim_words_tuples, 
            'plot_path': plot_path
        })
    except KeyError:
        return jsonify({'success': False, 'error': f'Kata "{word}" tidak ditemukan dalam model {model_name}.'})
    except Exception as e:
        print(f"Error di /get_similar: {e}")
        return jsonify({'success': False, 'error': 'Terjadi kesalahan internal.'})

@app.route('/get_similarity', methods=['POST'])
def get_similarity():
    """Menghitung skor kesamaan antara dua kata."""
    word1 = request.form['word1'].lower().strip()
    word2 = request.form['word2'].lower().strip()
    model_name = request.form['model']
    model = {'word2vec': model_w2v, 'fasttext': model_ft, 'glove': model_glove}.get(model_name)

    if not model:
        return jsonify({'success': False, 'error': 'Model yang dipilih tidak tersedia atau gagal dimuat.'})

    try:
        # Cek apakah kedua kata ada di dalam kosakata model
        vocab = model.wv.key_to_index if hasattr(model, 'wv') else model.key_to_index
        if word1 not in vocab or word2 not in vocab:
            missing = word1 if word1 not in vocab else word2
            return jsonify({'success': False, 'error': f'Kata "{missing}" tidak ditemukan.'})
            
        # Hitung kesamaan
        sim_score = model.wv.similarity(word1, word2) if hasattr(model, 'wv') else model.similarity(word1, word2)
        return jsonify({'success': True, 'similarity': float(sim_score)})
    except Exception as e:
        print(f"Error di /get_similarity: {e}")
        return jsonify({'success': False, 'error': 'Terjadi kesalahan saat menghitung kesamaan.'})

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    # Pastikan direktori 'static' ada
    if not os.path.exists('static'):
        os.makedirs('static')
    # Pastikan direktori 'templates' ada
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True, port=5100)