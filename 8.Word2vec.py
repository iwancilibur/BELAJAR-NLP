from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

bahasa = [
    "Pemrograman komputer adalah keterampilan yang penting.",
    "Data digunakan untuk analisis dalam berbagai bidang.",
    "Kecerdasan buatan membantu mengembangkan teknologi baru.",
    "Pembelajaran mesin adalah bagian dari kecerdasan buatan.",
    "Jaringan internet menghubungkan perangkat di seluruh dunia.",
    "Perangkat keras dan perangkat lunak adalah komponen komputer.",
    "Sistem informasi mengelola data dan pengetahuan.",
    
]

# Tokenisasi data
tokenized_data = [word_tokenize(text) for text in bahasa]

# Membuat model Word2Vec
wordvec = Word2Vec(
    sentences=tokenized_data,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=0,
    epochs=10
)

# Daftar stopwords sederhana yang ingin dihilangkan dari hasil
stopwords = {'dan', 'di', 'yang', 'adalah'}

print("Kata-kata paling mirip dengan 'komputer' (stopwords diabaikan):")
similar_words = wordvec.wv.most_similar('komputer', topn=10)

# Filter hasil supaya stopwords tidak muncul
filtered_words = [(word, score) for word, score in similar_words if word not in stopwords]

# Tampilkan hasil yang sudah difilter (misal 5 teratas)
for word, score in filtered_words[:5]:
    print(f"{word}: {score:.4f}")
