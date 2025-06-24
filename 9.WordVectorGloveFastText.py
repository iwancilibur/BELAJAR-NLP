import os
import requests
import zipfile
import numpy as np

def download_file(url, destination):
    if os.path.isfile(destination):
        print(f"File sudah ada: {destination}")
        return
    print(f"Mengunduh dari {url} ...")
    response = requests.get(url, stream=True)
    total_size = response.headers.get('content-length')

    with open(destination, 'wb') as file_out:
        if total_size is None:
            file_out.write(response.content)
        else:
            downloaded = 0
            total_size = int(total_size)
            for chunk in response.iter_content(chunk_size=4096):
                downloaded += len(chunk)
                file_out.write(chunk)
                done = int(40 * downloaded / total_size)
                print(f"\r[{'=' * done}{' ' * (40 - done)}]", end='')
    print("\nProses download selesai!")

def unzip_file(zip_filepath, output_dir, target_filename):
    target_filepath = os.path.join(output_dir, target_filename)
    if os.path.isfile(target_filepath):
        print(f"File {target_filename} sudah ada di {output_dir}")
        return
    print(f"Mengekstrak file {zip_filepath} ...")
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print("Ekstraksi selesai!")

def read_glove_file(glove_path):
    embedding_dict = {}
    print(f"Memuat embedding dari file {glove_path} ...")
    with open(glove_path, 'r', encoding='utf-8') as file_in:
        for line in file_in:
            parts = line.strip().split()
            token = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embedding_dict[token] = vector
    print(f"Jumlah kata yang dimuat: {len(embedding_dict)}")
    return embedding_dict

def compute_cosine(vec_a, vec_b):
    numerator = np.dot(vec_a, vec_b)
    denominator = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denominator == 0:
        return 0
    return numerator / denominator

def get_top_similar_words(target_word, embeddings, top_k=5):
    if target_word not in embeddings:
        return f"Kata '{target_word}' tidak ditemukan di embeddings."
    base_vector = embeddings[target_word]
    similarity_scores = {}
    for word, vector in embeddings.items():
        if word == target_word:
            continue
        similarity_scores[word] = compute_cosine(base_vector, vector)
    sorted_words = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:top_k]

if __name__ == "__main__":
    # URL dan nama-nama file GloVe
    url_glove = "http://nlp.stanford.edu/data/glove.6B.zip"
    zip_name = "glove.6B.zip"
    output_dir = "glove_embeddings"
    glove_txt = "glove.6B.50d.txt"

    # Pastikan folder output ada
    os.makedirs(output_dir, exist_ok=True)

    # Download GloVe jika belum ada
    download_file(url_glove, zip_name)

    # Ekstrak file jika belum ada
    unzip_file(zip_name, output_dir, glove_txt)

    # Load embeddings dari file
    file_path = os.path.join(output_dir, glove_txt)
    embeddings_map = read_glove_file(file_path)

    # Minta input kata dari user
    input_word = input("Masukkan kata yang ingin dicari kata miripnya: ").strip().lower()
    result = get_top_similar_words(input_word, embeddings_map)

    if isinstance(result, str):
        print(result)
    else:
        print(f"Kata-kata paling mirip dengan '{input_word}':")
        for word, score in result:
            print(f"{word} (similarity: {score:.4f})")
