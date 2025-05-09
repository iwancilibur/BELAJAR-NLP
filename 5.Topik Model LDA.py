import gensim
from gensim import corpora
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pandas as pd

# 1. Persiapan Data
documents = [
    "Pemerintah mengumumkan kebijakan ekonomi baru untuk UMKM",
    "Harga minyak dunia naik signifikan akibat konflik geopolitik",
    "Tim sepak bola nasional lolos ke Piala Dunia 2026",
    "Pasar saham menunjukkan pertumbuhan positif di kuartal ini",
    "Bank Indonesia prediksi inflasi akan stabil di bawah 4%",
    "Teknologi AI mulai banyak digunakan di industri perbankan",
    "Pembangunan infrastruktur jalan tol trans Jawa hampir selesai",
    "Ekspor komoditas pertanian meningkat 15% tahun ini"
]

# 2. Preprocessing Teks
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stopword_factory = StopWordRemoverFactory()
stopwords = stopword_factory.get_stop_words() + ['untuk', 'di', 'pada', 'akan']

processed_docs = []
for doc in documents:
    tokens = doc.lower().split()
    tokens = [stemmer.stem(token) for token in tokens if token not in stopwords and len(token) > 3]
    processed_docs.append(tokens)

# 3. Pembuatan Dictionary & Corpus
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# 4. Pelatihan Model LDA
lda_model = gensim.models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=3,
    random_state=42,
    passes=15,
    alpha='auto'
)

# 5. Visualisasi Hasil
print("=== TOPIK YANG DIIDENTIFIKASI ===")
for idx, topic in lda_model.print_topics():
    print(f"Topik {idx}: {topic}")

# 6. Prediksi Topik untuk Dokumen Baru
new_doc = "Pertumbuhan ekonomi kuartal III mencapai 5.1% menurut BI"
processed_new_doc = [stemmer.stem(word) for word in new_doc.lower().split() 
                    if word not in stopwords and len(word) > 3]
bow = dictionary.doc2bow(processed_new_doc)
print("\n=== PREDIKSI TOPIK UNTUK DOKUMEN BARU ===")
print(lda_model[bow])