from flask import Flask, render_template, request, jsonify
import nltk
from nltk.tokenize import word_tokenize
import gensim
from gensim.models import Word2Vec, FastText
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import numpy as np
import string
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

app = Flask(__name__)

# Download NLTK tokenizer data jika belum ada
nltk.download('punkt')

# Data deskripsi Word2Vec, GloVe, FastText (15 baris)
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

# Tokenisasi dan pembersihan data
tokens_list = []
for kalimat in deskripsi_model:
    toks = word_tokenize(kalimat.lower())
    bersih = [w for w in toks if w not in string.punctuation and re.match(r'^\w+$', w)]
    tokens_list.append(bersih)

# Training Word2Vec
model_w2v = Word2Vec(
    sentences=tokens_list,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=0,
    epochs=10
)
model_w2v.save("word2vec.model")

# Training FastText
print("Memulai pelatihan model FastText...")
model_ft = FastText(
    sentences=tokens_list,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=10,
    epochs=5,
    min_n=3,
    max_n=6,
    bucket=2000000,
    alpha=0.025
)
model_ft.save("fasttext.model")
print(f"Model FastText selesai dan disimpan dengan ukuran kosakata: {len(model_ft.wv.index_to_key)}")

# Load GloVe model jika tersedia
model_glove = None
file_glove = "glove.6B.300d.txt"
file_glove_w2v = "glove.6B.300d.w2v.txt"
if os.path.isfile(file_glove):
    try:
        if not os.path.isfile(file_glove_w2v):
            print(f"Mengonversi {file_glove} ke format Word2Vec...")
            glove2word2vec(file_glove, file_glove_w2v)
            print("Konversi selesai.")
        model_glove = KeyedVectors.load_word2vec_format(file_glove_w2v, binary=False)
        print("Model GloVe berhasil dimuat.")
    except Exception as err:
        print(f"Gagal memuat model GloVe: {err}")
else:
    print(f"File {file_glove} tidak ditemukan. Silakan unduh dan tempatkan di direktori proyek.")

# Gabungkan kosakata
vocab_w2v = set(model_w2v.wv.index_to_key)
vocab_ft = set(model_ft.wv.index_to_key)
combined_vocab = sorted(vocab_w2v.union(vocab_ft))
if model_glove:
    vocab_glove = set(model_glove.index_to_key)
    combined_vocab = sorted(combined_vocab.union(vocab_glove))
print("Kosakata gabungan:", combined_vocab)

# Fungsi buat PCA plot untuk kata-kata
def generate_pca_plot(words, model, model_name):
    vectors = []
    valid_words = []
    for w in words:
        if w in model.wv:
            vectors.append(model.wv[w])
            valid_words.append(w)

    if len(vectors) < 2:
        return None

    try:
        vectors = np.array(vectors)
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(vectors)

        plt.figure(figsize=(8, 6))
        for i, w in enumerate(valid_words):
            plt.scatter(reduced[i, 0], reduced[i, 1])
            plt.annotate(w, (reduced[i, 0], reduced[i, 1]))
        plt.title(f"Plot PCA 2D Vektor Kata ({model_name})")
        plt.xlabel("Komponen PCA 1")
        plt.ylabel("Komponen PCA 2")
        plt.legend(valid_words)
        if not os.path.exists('static'):
            os.makedirs('static')
        path = f"static/pca_{model_name.lower()}.png"
        plt.savefig(path)
        plt.close()
        return path
    except Exception as e:
        print(f"Error generate PCA plot: {e}")
        return None

@app.route('/')
def index():
    return render_template('WordVectorGloveFastText.html', vocabulary=combined_vocab)

@app.route('/get_vector', methods=['POST'])
def get_vector():
    word = request.form['word'].lower().strip()
    model_name = request.form['model']

    models = {
        'word2vec': model_w2v,
        'fasttext': model_ft,
        'glove': model_glove if model_glove else model_w2v
    }
    model = models.get(model_name)
    if not model:
        return jsonify({'success': False, 'error': 'Model tidak tersedia'})

    try:
        vector = model.wv[word].tolist()
        return jsonify({'success': True, 'vector': vector[:10], 'full_vector': vector})
    except KeyError:
        return jsonify({'success': False, 'error': f'Kata "{word}" tidak ditemukan dalam kosakata'})
    except Exception as e:
        return jsonify({'success': False, 'error': 'Kesalahan internal saat mengambil vektor'})

@app.route('/get_similar', methods=['POST'])
def get_similar():
    word = request.form['word'].lower().strip()
    model_name = request.form['model']

    models = {
        'word2vec': model_w2v,
        'fasttext': model_ft,
        'glove': model_glove if model_glove else model_w2v
    }
    model = models.get(model_name)
    if not model:
        return jsonify({'success': False, 'error': 'Model tidak tersedia'})

    try:
        similar = model.wv.most_similar(word, topn=5)
        filtered = [(w, score) for w, score in similar if w not in string.punctuation and re.match(r'^\w+$', w)]
        filtered = filtered[:5]
        plot_words = [word] + [w for w, _ in filtered]
        plot_path = generate_pca_plot(plot_words, model, model_name)
        return jsonify({'success': True, 'similar_words': filtered, 'plot_path': plot_path})
    except KeyError:
        return jsonify({'success': False, 'error': f'Kata "{word}" tidak ditemukan dalam kosakata'})
    except Exception:
        return jsonify({'success': False, 'error': 'Kesalahan internal saat mencari kata serupa'})

@app.route('/get_similarity', methods=['POST'])
def get_similarity():
    word1 = request.form['word1'].lower().strip()
    word2 = request.form['word2'].lower().strip()
    model_name = request.form['model']

    models = {
        'word2vec': model_w2v,
        'fasttext': model_ft,
        'glove': model_glove if model_glove else model_w2v
    }
    model = models.get(model_name)
    if not model:
        return jsonify({'success': False, 'error': 'Model tidak tersedia'})

    try:
        if word1 not in model.wv or word2 not in model.wv:
            missing = word1 if word1 not in model.wv else word2
            return jsonify({'success': False, 'error': f'Kata "{missing}" tidak ditemukan dalam kosakata'})
        sim = model.wv.similarity(word1, word2)
        return jsonify({'success': True, 'similarity': float(sim)})
    except Exception:
        return jsonify({'success': False, 'error': 'Kesalahan internal saat menghitung kesamaan'})

if __name__ == "__main__":
    app.run(debug=True, port=5100)
