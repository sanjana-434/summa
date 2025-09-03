import os
import math
import csv

# -----------------------------
# Load documents
# -----------------------------
def load_documents(folder_path):
    docs = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                docs[filename] = f.read().lower()
    return docs

def load_from_csv(csv_path):
    docs = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row:  # skip empty rows
                continue
            # Join all columns of the row as text
            text = " ".join(row).strip().lower()
            docs[f"doc{i+1}"] = text
    return docs

# -----------------------------
# 1. Inverted Index + Boolean Model
# -----------------------------
def build_inverted_index(docs):
    inverted_index = {}
    for doc_id, text in docs.items():
        words = text.split()
        for word in words:
            if word not in inverted_index:
                inverted_index[word] = []
            if doc_id not in inverted_index[word]:
                inverted_index[word].append(doc_id)
    return inverted_index

def boolean_search(query, inverted_index, all_docs):
    tokens = query.lower().split()
    result = set(all_docs)

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token == "and":
            i += 1
            continue  # handled implicitly by intersection
        elif token == "or":
            i += 1
            term = tokens[i]
            docs_with_term = set(inverted_index.get(term, []))
            result = result.union(docs_with_term)
        elif token == "not":
            i += 1
            term = tokens[i]
            docs_with_term = set(inverted_index.get(term, []))
            result = result.difference(docs_with_term)
        else:  # normal term
            docs_with_term = set(inverted_index.get(token, []))
            if i > 0 and tokens[i-1] == "or":
                result = result.union(docs_with_term)
            elif i > 0 and tokens[i-1] == "not":
                result = result.difference(docs_with_term)
            else:  # default is AND
                result = result.intersection(docs_with_term)
        i += 1

    return list(result)

# -----------------------------
# 2. Vector Space Model (TF-IDF + Cosine Similarity)
# -----------------------------
def compute_tf(doc):
    tf = {}
    words = doc.split()
    for w in words:
        tf[w] = tf.get(w, 0) + 1
    for w in tf:
        tf[w] /= len(words)
    return tf

def compute_idf(docs):
    idf = {}
    N = len(docs)
    for text in docs.values():
        words = set(text.split())
        for w in words:
            idf[w] = idf.get(w, 0) + 1
    for w in idf:
        idf[w] = math.log((N + 1) / (idf[w] + 1)) + 1
    return idf

def vector_space_model(query, docs):
    idf = compute_idf(docs)
    doc_vectors = {}
    for doc_id, text in docs.items():
        tf = compute_tf(text)
        doc_vectors[doc_id] = {w: tf[w] * idf.get(w, 0) for w in tf}

    # Query vector
    q_tf = compute_tf(query)
    q_vector = {w: q_tf[w] * idf.get(w, 0) for w in q_tf}

    # Cosine similarity
    def cosine_similarity(vec1, vec2):
        dot = sum(vec1.get(w, 0) * vec2.get(w, 0) for w in set(vec1) | set(vec2))
        norm1 = math.sqrt(sum(v*v for v in vec1.values()))
        norm2 = math.sqrt(sum(v*v for v in vec2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0
        return dot / (norm1 * norm2)

    scores = []
    for doc_id, vec in doc_vectors.items():
        score = cosine_similarity(q_vector, vec)
        scores.append((doc_id, score))
    return sorted(scores, key=lambda x: x[1], reverse=True)

# -----------------------------
# 3. Binary Independence Model 
# -----------------------------
def binary_independence_model(query, docs):
    doc_names = list(docs.keys())
    doc_words = [set(text.split()) for text in docs.values()]
    query_terms = query.lower().split()
    N = len(docs)

    # Precompute term statistics
    term_stats = {}
    for term in query_terms:
        ni = sum(1 for d in doc_words if term in d)  # docs containing term
        if ni == 0:
            continue
        pi = (ni + 0.5) / (N + 1)
        qi = (N - ni + 0.5) / (N + 1)
        term_stats[term] = (pi, qi)

    scores = []
    for i, doc in enumerate(doc_words):
        score = 0
        for term in query_terms:
            if term not in term_stats:
                continue
            pi, qi = term_stats[term]
            if term in doc:  # term present
                score += math.log((pi * (1 - qi)) / (qi * (1 - pi)))
            else:  # term absent
                score += math.log(((1 - pi) * (1 - qi)) / ((1 - qi) * (1 - pi)))
        scores.append((doc_names[i], score))

    return sorted(scores, key=lambda x: x[1], reverse=True)


# -----------------------------
# Example Run
# -----------------------------
if __name__ == "__main__":
    folder = "/Users/sanjanarajasekaran/Documents/projects/untitled folder/ir_lab/docs"   # put your .txt files in a folder called "docs"
    # docs = load_documents(folder)
    docs = load_from_csv("/Users/sanjanarajasekaran/Documents/projects/untitled folder/ir_lab/docs/doc.csv")

    print(docs)

    query1 = "information AND boolean"
    query2 = "model OR system"
    query3 = "retrieval NOT model"

    inverted_index = build_inverted_index(docs)
    all_docs = list(docs.keys())

    print("\nBoolean Model Results:")
    print("Query:", query1, "->", boolean_search(query1, inverted_index, all_docs))
    print("Query:", query2, "->", boolean_search(query2, inverted_index, all_docs))
    print("Query:", query3, "->", boolean_search(query3, inverted_index, all_docs))

    print("\nVector Space Model Ranking:")
    for doc, score in vector_space_model("information retrieval model", docs):
        print(doc, ":", round(score, 4))

    print("\nBinary Independence Model Ranking:")
    for doc, score in binary_independence_model("information retrieval model", docs):
        print(doc, ":", round(score, 4))
