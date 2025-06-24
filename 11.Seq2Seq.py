import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, render_template, request
import random
import io

# -- Inisialisasi Aplikasi Flask --
app = Flask(__name__)

# -- Data Dummy (DIUBAH ke Indonesia -> Inggris) --
indonesian_sentences = ["Apa kabar hari ini?", "Siapa nama kamu?", "Saya suka belajar NLP", "Dia adalah seorang dokter"]
english_sentences = ["How are you today?", "What is your name?", "I love to learn NLP", "She is a doctor"]

# -- Preprocessing Data --
# Tambahkan token khusus untuk Start-of-Sequence (SOS) dan End-of-Sequence (EOS)
SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Hitung SOS dan EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# DIUBAH: Menggunakan nama variabel yang lebih sesuai
def prepare_data(lang1_sentences, lang2_sentences):
    input_lang = Lang('ind') # Input: Bahasa Indonesia
    output_lang = Lang('eng') # Output: Bahasa Inggris
    pairs = []
    for i in range(len(lang1_sentences)):
        pairs.append([lang1_sentences[i], lang2_sentences[i]])
        input_lang.addSentence(lang1_sentences[i])
        output_lang.addSentence(lang2_sentences[i])
    return input_lang, output_lang, pairs

# DIUBAH: Memanggil fungsi dengan data baru
input_lang, output_lang, pairs = prepare_data(indonesian_sentences, english_sentences)

# Konversi kalimat ke tensor
def tensorFromSentence(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

# -- Definisi Model Seq2Seq (Encoder-Decoder) --
# Arsitektur model tidak perlu diubah
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

# -- Training Model --
hidden_size = 256
encoder = EncoderRNN(input_lang.n_words, hidden_size)
decoder = DecoderRNN(hidden_size, output_lang.n_words)

# Optimizer & Loss Function
encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)
criterion = nn.NLLLoss()

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = torch.zeros(1, 1, encoder.hidden_size)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0

    # Encoder
    for ei in range(input_tensor.size(0)):
        _, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

    # Decoder
    decoder_input = torch.tensor([[SOS_token]])
    decoder_hidden = encoder_hidden
    
    for di in range(target_tensor.size(0)):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()
        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == EOS_token:
            break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_tensor.size(0)

# Jalankan proses training sederhana
print("Training model untuk Indonesia -> Inggris...")
n_iters = 7000 # Iterasi bisa ditambah agar lebih baik
training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]

for iter in range(1, n_iters + 1):
    training_pair = training_pairs[iter - 1]
    input_tensor = training_pair[0]
    target_tensor = training_pair[1]
    loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
    if iter % 500 == 0:
        print(f'Iter: {iter}/{n_iters}, Loss: {loss:.4f}')

print("Model training selesai.")

# -- Fungsi Evaluasi / Penerjemahan --
def evaluate_and_translate(sentence):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        encoder_hidden = torch.zeros(1, 1, encoder.hidden_size)

        for ei in range(input_tensor.size(0)):
            _, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]])
        decoder_hidden = encoder_hidden
        decoded_words = []

        for di in range(10): # Max length of output
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()

        return ' '.join(decoded_words)

# -- Flask Routes (DIUBAH) --
@app.route('/', methods=['GET', 'POST'])
def home():
    translated_sentence = ''
    original_sentence = ''
    if request.method == 'POST':
        # Mengambil input dari form dengan nama "indonesian_text"
        original_sentence = request.form.get('indonesian_text')
        # Cek jika kalimat ada di dataset (untuk demo)
        if original_sentence in indonesian_sentences:
             translated_sentence = evaluate_and_translate(original_sentence)
        else:
            translated_sentence = "Maaf, kalimat ini tidak ada dalam dataset training sederhana saya."

    # Mengirim data kalimat indonesia ke template
    return render_template('Seq2Seq.html', original=original_sentence, translation=translated_sentence, sentences=indonesian_sentences)

if __name__ == '__main__':
    app.run(debug=True)