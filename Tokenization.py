import nltk

nltk.download()
nltk.download('punkt')

paragraph = """Hello, world! This is a test paragraph. Let's see how tokenization works."""

sentences = nltk.sent_tokenize(paragraph)
words = nltk.word_tokenize(paragraph)
