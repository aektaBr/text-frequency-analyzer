import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    stop_words_set = set(stopwords.words('english'))

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Convert text to lowercase
    text = text.lower()

    # Remove non-alphabetic characters and extra spaces, ensuring words are properly separated
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    # Tokenize the cleaned text
    words = word_tokenize(cleaned_text)

    # Lemmatize and filtering of stopwords, hyphens
    processed_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words_set and word != '-']

    return processed_words


def process_and_calculate_frequencies(file_path):
    doc_frequencies = Counter()
    global_frequencies = Counter()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            words = preprocess_text(line)
            doc_frequencies.update(set(words))
            global_frequencies.update(words)
    return doc_frequencies, global_frequencies


def generate_files(doc_frequencies, global_frequencies, frequency_threshold=2):
    filtered_words = {word: freq for word, freq in global_frequencies.items() if freq > frequency_threshold}
    sorted_words_alpha = sorted(filtered_words.keys())

    with open('dictionary.txt', 'w', encoding='utf-8') as dict_file:
        for index, word in enumerate(sorted_words_alpha):
            dict_file.write(f"{index} {word}\n")

    word_to_code = {word: index for index, word in enumerate(sorted_words_alpha)}

    sorted_words_freq = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)

    with open('unigram.txt', 'w', encoding='utf-8') as uni_file:
        for word, _ in sorted_words_freq:
            code = word_to_code[word]
            doc_freq = doc_frequencies[word]
            global_freq = global_frequencies[word]
            uni_file.write(f"{code} {word} {doc_freq} {global_freq}\n")


file_path = 'tiny_wikipedia.txt'
doc_freqs, global_freqs = process_and_calculate_frequencies(file_path)
generate_files(doc_freqs, global_freqs)
