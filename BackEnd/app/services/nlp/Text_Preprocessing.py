import re
import string

_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
    "will", "just", "don", "should", "now", "d", "ll", "m", "o", "re", "ve",
    "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven",
    "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn",
    "weren", "won", "wouldn",
}

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

_CONTRACTIONS = {
    "what's": "what is",
    "who's": "who is",
    "where's": "where is",
    "when's": "when is",
    "why's": "why is",
    "how's": "how is",
    "it's": "it is",
    "i'm": "i am",
    "you're": "you are",
    "they're": "they are",
    "we're": "we are",
    "isn't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "won't": "will not",
    "wouldn't": "would not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
}

# Common Speech-to-Text misinterpretations for computer science terms
_SPEECH_CORRECTIONS = {
    r"\btcb\b": "tcp",
    r"\bhand check\b": "handshake",
    r"\bconclusion\b": "convolution",
    r"\bsemaphor\b": "semaphore",
    r"\bthree\b": "tree",
    r"\bmuxes\b": "mux",
}

try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    _lemmatizer = WordNetLemmatizer()
    def _lemmatize(word):
        return _lemmatizer.lemmatize(word)
except ImportError:
    # Fallback lightweight lemmatizer (suffix stripping)
    def _lemmatize(word):
        if word.endswith("ies") and len(word) > 4: return word[:-3] + "y"
        if word.endswith("es") and word[-3] in ["s", "x", "z", "h"]: return word[:-2]
        if word.endswith("s") and len(word) > 3 and word[-2] not in ["s", "u"]: return word[:-1]
        return word

def clean_text(text: str) -> str:
    # Lowercase normalization
    text = text.lower()
    
    # Contraction handling
    for contraction, expansion in _CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
        
    # Punctuation cleanup
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    
    # Fix speech recognition phonetic mistakes (typo tolerance)
    for wrong, right in _SPEECH_CORRECTIONS.items():
        text = re.sub(wrong, right, text)

    tokens = []
    for token in text.split():
        # Stopword removal
        if token not in _STOPWORDS:
            # Lemmatization
            token = _lemmatize(token)
            tokens.append(token)
            
    return " ".join(tokens)