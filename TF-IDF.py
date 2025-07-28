# Import the magic library
from sklearn.feature_extraction.text import TfidfVectorizer

# Our example "documents" (like short tweets)
documents = [
    "Cats are fun. Cats rule.",  # Doc 1
    "Dogs are fun.",             # Doc 2
    "Birds fly high."            # Doc 3
]

# Create a TF-IDF wizard (vectorizer)
vectorizer = TfidfVectorizer()

# Fit it to our docs and transform them into TF-IDF scores
tfidf_matrix = vectorizer.fit_transform(documents)

# Get the words (features) for reference
words = vectorizer.get_feature_names_out()

# Print the results in a fun way
print("TF-IDF Scores (Rows: Docs, Columns: Words)\n")
for i, doc in enumerate(documents):
    print(f"Doc {i+1}: {doc}")
    scores = tfidf_matrix[i].toarray().flatten()  # Get scores for this doc
    for word, score in zip(words, scores):
        if score > 0:  # Only show words that appear
            print(f"  - {word}: {score:.3f}")  # Round to 3 decimals
    print()  # Blank line for readability
