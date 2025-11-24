def preprocess(text):
    import re, string
    from nltk.tokenize import word_tokenize
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words('indonesian'))
    stemmer = StemmerFactory().create_stemmer()

    def cleaning(kalimat):
        if not kalimat or kalimat.strip() == "":
            return None

        # Jika seluruh kalimat hanya string acak panjang
        if re.fullmatch(r'[A-Za-z0-9]{12,}', kalimat.strip()):
            return None

        # Ganti semua jenis URL dan HTML link jadi 'link'
        kalimat = re.sub(r'https?://\S+|www\.\S+|\S+\.(com|org|net)\S*', 'link', kalimat, flags=re.IGNORECASE)
        kalimat = re.sub(r'\[url=.*?\].*?\[/url\]', 'link', kalimat, flags=re.IGNORECASE)
        kalimat = re.sub(r'<a\s+(?:[^>]*?\s+)?href="[^"]*".*?</a>', 'link', kalimat, flags=re.IGNORECASE | re.DOTALL)
        kalimat = re.sub(r'<.*?>', '', kalimat)

        # Hapus kata-kata acak panjang di dalam kalimat
        kalimat = re.sub(r'\b[a-zA-Z0-9]{10,}\b', '', kalimat)

        # Lanjutkan dengan normalisasi
        kalimat = re.sub(r'\t|\n|\r', ' ', kalimat)
        kalimat = kalimat.encode('ascii', 'ignore').decode('ascii')
        kalimat = re.sub(r'\d+', '', kalimat)
        kalimat = kalimat.translate(str.maketrans("", "", string.punctuation))
        kalimat = re.sub(r'&\w+;', '', kalimat)
        kalimat = kalimat.lower()
        kalimat = ' '.join([word for word in kalimat.split() if len(word) > 1])
        kalimat = re.sub(r'\s+', ' ', kalimat).strip()

        return kalimat if kalimat else None

    cleaned = cleaning(text)
    if not cleaned:
        return {
            'cleaned': None,
            'tokens': [],
            'removed_stopwords': [],
            'stemmed': []
        }

    tokens = word_tokenize(cleaned)
    removed_stopwords = [word for word in tokens if word in stop_words]
    tokens_no_stopwords = [word for word in tokens if word not in stop_words]
    stemmed = [stemmer.stem(word) for word in tokens_no_stopwords]

    return {
        'cleaned': cleaned,
        'tokens': tokens,
        'removed_stopwords': removed_stopwords,
        'stemmed': stemmed,
    }
