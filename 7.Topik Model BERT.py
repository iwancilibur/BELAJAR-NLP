# TOPIC MODELING WITH BERT - VARIATIVE DATASET IMPLEMENTATION
# ============================================================

from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
from umap import UMAP
import pandas as pd
import matplotlib.pyplot as plt
import os

# (Opsional) Tambahan: Dataset eksternal dari HuggingFace
try:
    from datasets import load_dataset
    USE_HUGGINGFACE = True
except ImportError:
    USE_HUGGINGFACE = False

# 1. Load Dataset dari beberapa kategori berbeda
print("🔍 Memuat dataset dari berbagai kategori...")

categories = [
    'sci.space', 'talk.politics.mideast', 'comp.graphics',
    'rec.sport.hockey', 'sci.med', 'soc.religion.christian'
]

data = fetch_20newsgroups(subset='all', categories=categories,
                          remove=('headers', 'footers', 'quotes'))
docs = data.data[:2500]

# 2. Tambahkan dokumen buatan untuk variasi tambahan
extra_docs = [
    "NASA plans to build a base on the moon by 2030.",
    "The stock market fluctuated heavily due to political tensions.",
    "Python is a great language for machine learning and web development.",
    "Lionel Messi scored a hat-trick in last night's match.",
    "Ayurveda and traditional medicine are gaining popularity.",
    "New regulations impact international trade agreements.",
    "The chef created a fusion dish combining Thai and French cuisine.",
    "A new strain of virus has been discovered in Southeast Asia.",
    "Virtual reality is revolutionizing the education sector.",
    "Buddhism emphasizes mindfulness and inner peace."
]

docs += extra_docs

# 3. (Opsional) Tambahkan data dari HuggingFace
if USE_HUGGINGFACE:
    print("📦 Menambahkan data dari HuggingFace (AG News)...")
    ag_news = load_dataset("ag_news", split="train[:500]")
    hf_docs = [item['text'] for item in ag_news]
    docs += hf_docs

print(f"📊 Total dokumen yang digunakan: {len(docs)}")

# 4. Inisialisasi UMAP dan BERTopic
print("🛠️ Membuat model BERTopic...")

umap_model = UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=42
)

topic_model = BERTopic(
    language="english",
    calculate_probabilities=True,
    verbose=True,
    n_gram_range=(1, 2),
    min_topic_size=20,
    umap_model=umap_model
)

# 5. Training Model
print("🏋️ Melatih model...")
topics, probs = topic_model.fit_transform(docs)

# 6. Analisis dan Informasi Topik
print("\n📋 Informasi Topik:")
topic_info = topic_model.get_topic_info()
print(topic_info.head())

# 7. Visualisasi dengan Error Handling
print("\n🎨 Membuat visualisasi...")
os.makedirs("output", exist_ok=True)

try:
    fig1 = topic_model.visualize_topics()
    fig1.write_html("output/intertopic_map.html")

    fig2 = topic_model.visualize_barchart(top_n_topics=min(10, len(topic_info)))
    fig2.write_html("output/top_topics.html")

    fig3 = topic_model.visualize_documents(docs[:1000])
    fig3.write_html("output/document_map.html")

    print("✅ Visualisasi berhasil disimpan di folder 'output'.")

except Exception as e:
    print(f"⚠️ Error dalam visualisasi: {str(e)}")
    print("Menyimpan alternatif visualisasi...")

    topic_freq = topic_info.set_index('Topic')['Count']
    topic_freq.plot(kind='bar', figsize=(10, 5))
    plt.title("Distribusi Topik")
    plt.tight_layout()
    plt.savefig("output/topic_distribution.png")
    print("✅ Visualisasi alternatif disimpan sebagai topic_distribution.png")

# 8. Simpan Model
topic_model.save("output/my_bertopic_model")
print("💾 Model disimpan sebagai 'output/my_bertopic_model'")

# 9. Contoh Prediksi
new_docs = [
    "AI will shape the future of healthcare.",
    "Cristiano Ronaldo scored twice in the final match."
]
pred_topics, pred_probs = topic_model.transform(new_docs)

print("\n🔮 Prediksi Topik untuk Dokumen Baru:")
for doc, topic in zip(new_docs, pred_topics):
    print(f"'{doc[:40]}...' → Topik {topic}")

print("\n✅ Selesai!")
