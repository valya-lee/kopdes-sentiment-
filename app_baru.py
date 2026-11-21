import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
import emoji

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# -----------------------------
# Pengaturan halaman
# -----------------------------
st.set_page_config(
    page_title="Analisis Sentimen Kopdes",
    page_icon="💬",
    layout="centered"
)

st.title("Analisis Sentimen Kopdes")
st.caption("Chatbot sederhana untuk menganalisis komentar masyarakat.")

STOPWORDS_INDO = [
    "yang", "di", "ke", "dari", "dan", "atau",
    "ini", "itu", "untuk", "dengan", "nya",
    "sih", "aja", "deh", "banget"
]


# -----------------------------
# Text preprocessing
# -----------------------------
def preprocessing(teks):
    teks = str(teks).lower()
    teks = emoji.replace_emoji(teks, replace=" ")
    teks = teks.translate(
        str.maketrans(
            string.punctuation,
            " " * len(string.punctuation)
        )
    )
    teks = re.sub(r"\d+", " ", teks)

    kata_bersih = [
        kata for kata in teks.split()
        if kata not in STOPWORDS_INDO
    ]

    return " ".join(kata_bersih)


# -----------------------------
# Membaca dataset
# -----------------------------
def load_data(file):
    data = pd.read_csv(file, encoding="utf-8")

    kolom_wajib = {"komentar", "sentimen"}
    if not kolom_wajib.issubset(data.columns):
        raise ValueError(
            "CSV harus memiliki kolom bernama 'komentar' dan 'sentimen'."
        )

    data = data.dropna(subset=["komentar", "sentimen"]).copy()
    data["sentimen"] = data["sentimen"].astype(str).str.strip().str.lower()
    data["komentar_bersih"] = data["komentar"].apply(preprocessing)

    data = data[data["komentar_bersih"].str.strip() != ""].copy()

    if data.empty:
        raise ValueError("Dataset tidak memiliki komentar yang dapat diproses.")

    return data


# Pilih dataset: file lokal atau unggah CSV
uploaded_file = st.sidebar.file_uploader(
    "Unggah dataset CSV",
    type=["csv"],
    help="CSV harus memiliki kolom 'komentar' dan 'sentimen'."
)

if uploaded_file is not None:
    source_file = uploaded_file
else:
    try:
        source_file = open("komentar_kopdes.csv", "rb")
    except FileNotFoundError:
        source_file = None

if source_file is None:
    st.info(
        "Silakan unggah file **komentar_kopdes.csv** melalui panel di sebelah kiri. "
        "Pastikan terdapat kolom `komentar` dan `sentimen`."
    )
    st.stop()

try:
    data = load_data(source_file)
except Exception as error:
    st.error(f"Dataset tidak dapat dibaca: {error}")
    st.stop()
finally:
    if uploaded_file is None and source_file is not None:
        source_file.close()


# -----------------------------
# Melatih model Naive Bayes
# -----------------------------
if data["sentimen"].nunique() < 2:
    st.error("Dataset harus memiliki minimal dua kategori sentimen.")
    st.stop()

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data["komentar_bersih"])
y = data["sentimen"]

model = MultinomialNB()
model.fit(X, y)


# -----------------------------
# Ringkasan dan grafik
# -----------------------------
st.subheader("Ringkasan Dataset")
col1, col2 = st.columns(2)
col1.metric("Jumlah komentar", len(data))
col2.metric("Kategori sentimen", data["sentimen"].nunique())

st.subheader("Grafik Jumlah Sentimen")
jumlah_sentimen = data["sentimen"].value_counts()

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(jumlah_sentimen.index, jumlah_sentimen.values)
ax.set_xlabel("Kategori Sentimen")
ax.set_ylabel("Jumlah Komentar")
ax.set_title("Jumlah Komentar Berdasarkan Sentimen")
ax.bar_label(bars, padding=3)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


# -----------------------------
# Chatbot
# -----------------------------
st.subheader("Chatbot Analisis Sentimen")
st.write("Masukkan komentar untuk mengetahui kategori sentimennya.")

if "riwayat" not in st.session_state:
    st.session_state.riwayat = []

with st.form("form_komentar", clear_on_submit=True):
    komentar = st.text_area(
        "Tulis komentar",
        placeholder="Contoh: Pelayanannya bagus dan produknya memuaskan."
    )
    submitted = st.form_submit_button("Analisis komentar")

if submitted:
    if not komentar.strip():
        st.warning(
            "Komentar belum diisi. Silakan masukkan komentar terlebih dahulu."
        )

    elif komentar.strip().lower() == "keluar":
        st.success("Terima kasih! Chatbot selesai.")

    else:
        komentar_bersih = preprocessing(komentar)

        if not komentar_bersih.strip():
            st.warning(
                "Komentar tidak memiliki teks yang dapat dianalisis."
            )

        else:
            komentar_vector = vectorizer.transform([komentar_bersih])
            hasil = model.predict(komentar_vector)[0]

            respons = {
                "bahagia": (
                    "Terima kasih! Sepertinya Anda puas dengan produk kami."
                ),
                "sedih": (
                    "Mohon maaf atas pengalaman Anda. "
                    "Masukan Anda akan menjadi evaluasi kami."
                ),
                "marah": (
                    "Mohon maaf atas ketidaknyamanan yang Anda alami. "
                    "Kami akan mengevaluasi dan memperbaiki layanan kami."
                )
            }.get(
                hasil,
                "Terima kasih atas masukannya. "
                "Kami akan terus meningkatkan layanan."
            )

            st.session_state.riwayat.insert(
                0,
                {
                    "komentar": komentar,
                    "sentimen": hasil,
                    "respons": respons
                }
            )

            st.success(f"Hasil analisis: {hasil.upper()}")
            st.write(f"**Chatbot:** {respons}")


# -----------------------------
# Riwayat komentar
# -----------------------------
st.subheader("Riwayat Komentar")

if st.session_state.riwayat:
    if st.button("Hapus riwayat"):
        st.session_state.riwayat = []
        st.rerun()

    riwayat_df = pd.DataFrame(st.session_state.riwayat)
    riwayat_df.columns = ["Komentar", "Sentimen", "Respons Chatbot"]
    st.dataframe(riwayat_df, use_container_width=True, hide_index=True)
else:
    st.caption("Belum ada komentar yang dianalisis.")

st.caption(
    "Catatan: model dilatih menggunakan seluruh dataset yang diunggah. "
    "Hasil prediksi mengikuti pola dan label pada dataset tersebut."
)
