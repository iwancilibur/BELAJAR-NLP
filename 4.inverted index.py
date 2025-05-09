from flask import Flask, render_template, request
import pandas as pd
from collections import defaultdict
import re
from flask_paginate import Pagination, get_page_args

app = Flask(__name__)

# Load dataset
df = pd.read_csv("dataset.csv")


# Fungsi untuk memproses teks (sama seperti di tugas)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    return words


# Membangun inverted index dari dataset
def build_inverted_index(documents):
    inverted_index = defaultdict(list)
    for doc_id, text in documents.items():
        words = preprocess_text(text)
        for word in set(words):  # Gunakan set() untuk menghindari duplikasi
            inverted_index[word].append(doc_id)
    return inverted_index


# Membuat dictionary dari dataframe
documents = {idx: row['judul'] for idx, row in df.iterrows()}
inverted_index = build_inverted_index(documents)


# Fungsi query dengan operasi AND (sama seperti di tugas)
def query_inverted_index(query, inverted_index):
    query_words = preprocess_text(query)
    if not query_words:
        return []

    result = set(inverted_index.get(query_words[0], []))
    for word in query_words[1:]:
        result.intersection_update(inverted_index.get(word, set()))
    return sorted(result)


@app.route("/", methods=["GET", "POST"])
def index():
    query = request.form.get("query", "")
    kategori = request.form.get("kategori", "Semua")

    # Filter berdasarkan kategori
    filtered_df = df[df["kategori"] == kategori] if kategori != "Semua" else df.copy()

    if request.method == "POST" and query:
        # Gunakan inverted index untuk pencarian
        matched_indices = query_inverted_index(query, inverted_index)
        # Filter hasil berdasarkan kategori dan query
        filtered_df = filtered_df[filtered_df.index.isin(matched_indices)]
        # Tambahkan kolom similarity dummy untuk kompatibilitas
        filtered_df = filtered_df.assign(similarity=1.0)
    else:
        filtered_df = filtered_df.assign(similarity=0.0)

    # Pagination
    page, per_page, offset = get_page_args(page_parameter='page',
                                           per_page_parameter='per_page')
    total = len(filtered_df)
    paginated_results = filtered_df.iloc[offset: offset + per_page]

    pagination = Pagination(
        page=page,
        per_page=per_page,
        total=total,
        css_framework='bootstrap4',
        search=query,
        record_name='results',
        show_single_page=True,
        bs_version=4,
        additional_args={'kategori': kategori}
    )

    kategori_options = ["Semua"] + sorted(df["kategori"].unique())

    return render_template(
        "index.html",
        query=query,
        results=paginated_results,
        kategori_options=kategori_options,
        selected_kategori=kategori,
        pagination=pagination
    )


if __name__ == "__main__":
    app.run(debug=True)