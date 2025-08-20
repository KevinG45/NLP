import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.preprocessing import normalize
from collections import Counter
import time

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Lab 5: Sparse Vectors")

# --- Default Data ---
TF_DEFAULT_DATA = {
    'term': ['car', 'auto', 'insurance', 'best'],
    'Doc1': [27, 3, 0, 14],
    'Doc2': [4, 33, 33, 0],
    'Doc3': [24, 0, 29, 17]
}
TF_DEFAULT_DF = pd.DataFrame(TF_DEFAULT_DATA).set_index('term')

IDF_DEFAULT_DATA = {'term': ['car', 'auto', 'insurance', 'best'], 'idf': [1.65, 2.08, 1.62, 1.5]}
IDF_DEFAULT_DF = pd.DataFrame(IDF_DEFAULT_DATA).set_index('term')

CORPUS_DEFAULT = [
    "The best car insurance provides great coverage for your new auto.",
    "Find cheap auto insurance by comparing quotes from many providers.",
    "A new car is a major investment; protect it with the best coverage.",
    "The best recipe for apple pie is a family secret.",
    "Good food makes for a good mood, especially a sweet pie."
]

# =======================
# CACHING HELPERS (safe to cache: only hashable params/returns)
# =======================
@st.cache_data(show_spinner=False)
def cache_vectorize(corpus, rep, min_df, max_features):
    """
    Build docs×terms matrix (sparse) and term vectors (terms×docs), row-normalized for cosine.
    Returns: matrix (docs×terms, sparse), word_vectors (terms×docs, sparse), vocab (np.array)
    """
    if rep == "TF counts":
        vectorizer = CountVectorizer(stop_words='english', min_df=min_df,
                                     max_features=(None if max_features == 0 else max_features))
    else:
        vectorizer = TfidfVectorizer(stop_words='english', norm=None, sublinear_tf=True, smooth_idf=True,
                                     min_df=min_df, max_features=(None if max_features == 0 else max_features))
    matrix = vectorizer.fit_transform(corpus)   # docs x terms (sparse)
    vocab = vectorizer.get_feature_names_out()
    # Build word vectors (terms x docs) and L2-normalize each word vector
    word_vectors = normalize(matrix.T, norm='l2', axis=1)
    return matrix, word_vectors, vocab

@st.cache_data(show_spinner=False)
def cache_cosine_sim(selected_vector, word_vectors):
    sims = cosine_similarity(selected_vector, word_vectors).ravel()
    return sims

# ---------- Word–Word PMI/PPMI (corpus-based) ----------
def compute_word_word_ppmi(corpus, window_size=4, min_df=2, max_features=0, positive=True):
    """
    Build word–word PMI/PPMI from a symmetric sliding window.
    Returns:
      vectors (np.ndarray, terms×terms) row-normalized for cosine,
      vocab (list[str]),
      pmi_df (pd.DataFrame) raw PMI/PPMI values for display.
    Notes:
      - Not cached to avoid sparse/dense hashing pitfalls with large objects.
      - Safe for typical lab-sized corpora.
    """
    vec = CountVectorizer(
        stop_words='english',
        min_df=min_df,
        max_features=(None if max_features == 0 else max_features)
    )
    analyzer = vec.build_analyzer()
    X = vec.fit_transform(corpus)
    vocab = vec.get_feature_names_out().tolist()
    V = len(vocab)
    if V == 0:
        return np.zeros((0, 0)), [], pd.DataFrame()

    idx = {w: i for i, w in enumerate(vocab)}

    # Count unigrams and symmetric windowed co-occurrences
    uni = Counter()
    co = Counter()
    for doc in corpus:
        toks = [t for t in analyzer(doc) if t in idx]
        n = len(toks)
        for i, wi_str in enumerate(toks):
            wi = idx[wi_str]
            uni[wi] += 1
            L = max(0, i - window_size)
            R = min(n, i + window_size + 1)
            for j in range(L, R):
                if j == i:
                    continue
                wj = idx[toks[j]]
                co[(wi, wj)] += 1
                # also count the reverse to make it symmetric overall
                # (window loop already produces both directions in practice,
                # but this line ensures symmetry even if logic changes)
                # co[(wj, wi)] += 1  # optional; often redundant

    total_uni = float(sum(uni.values()))
    total_co = float(sum(co.values()))
    if total_uni == 0 or total_co == 0:
        df = pd.DataFrame(0.0, index=vocab, columns=vocab)
        return df.values, vocab, df

    # p(w)
    P_w = np.zeros(V, dtype=float)
    for i, c in uni.items():
        P_w[i] = c / total_uni

    # PMI/PPMI
    M = np.zeros((V, V), dtype=float)
    for (i, j), c in co.items():
        p_ij = c / total_co
        denom = P_w[i] * P_w[j]
        if denom > 0 and p_ij > 0:
            val = np.log2(p_ij / denom)
            if positive:
                val = max(0.0, val)
            M[i, j] = val

    pmi_df = pd.DataFrame(M, index=vocab, columns=vocab)
    vectors = normalize(pmi_df.values, norm='l2', axis=1)  # row-normalize for cosine
    return vectors, vocab, pmi_df

# =================================================================================
# --- Question 1: TF-IDF and Normalization Page ---
# =================================================================================
def page_question_1():
    st.header("Question 1: TF-IDF and Euclidean Normalization")
    st.markdown("---")

    st.subheader("1a. TF-IDF Calculation and Query Scoring")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Term Frequency (TF) Table")
        st.dataframe(TF_DEFAULT_DF)
    with col2:
        st.write("Inverse Document Frequency (IDF) Table")
        st.dataframe(IDF_DEFAULT_DF)

    tfidf_df = TF_DEFAULT_DF.multiply(IDF_DEFAULT_DF['idf'], axis=0)
    st.write("Calculated TF-IDF Matrix (tf × idf)")
    st.dataframe(tfidf_df.style.format("{:.2f}"))

    query = st.text_input("Enter a query (e.g., 'car insurance' or 'best car')", "car insurance").lower()
    query_terms = [t for t in query.split() if t in tfidf_df.index]
    if query.strip() and not query_terms:
        st.warning("None of the query terms are in the vocabulary.")
    elif query_terms:
        scores = tfidf_df.loc[query_terms].sum(axis=0)
        score_df = pd.DataFrame(scores, columns=['Score']).sort_values(by='Score', ascending=False)
        st.dataframe(score_df.style.format("{:.2f}"))
        st.success(f"Conclusion: Document {score_df.index[0]} is the most relevant.")
    else:
        st.info("Enter a query to compute scores.")

    st.markdown("---")
    st.subheader("1b. Euclidean Normalization (on TF) [Optional]")
    tf_norms = np.sqrt(np.sum(TF_DEFAULT_DF**2, axis=0))
    norm_df = pd.DataFrame(tf_norms, columns=['L2 Norm (TF)'])
    st.dataframe(norm_df.style.format("{:.2f}"))
    normalized_tf = TF_DEFAULT_DF.div(tf_norms, axis=1)
    st.write("L2-Normalized TF Matrix")
    st.dataframe(normalized_tf.style.format("{:.2f}"))

# =================================================================================
# --- Question 2: Cosine Similarity Page (Counts / TF-IDF) ---
# =================================================================================
def page_question_2():
    st.header("Question 2: Cosine Similarity")
    st.markdown("---")

    st.subheader("Data Source")
    data_source = st.radio("Choose a corpus:", ("Use default corpus", "Input your own corpus"))
    if data_source == "Input your own corpus":
        user_corpus_text = st.text_area(
            "Enter your documents, one per line.",
            height=150,
            placeholder="The quick brown fox...\nAnother document about cats and dogs...\nAnd a third one here."
        )
        corpus = [doc.strip() for doc in user_corpus_text.split('\n') if doc.strip()]
        if not corpus:
            st.info("Please enter at least one document to proceed.")
            return
    else:
        corpus = CORPUS_DEFAULT

    st.subheader("Vector representation")
    rep = st.radio("Choose representation:", ("TF counts", "TF-IDF"))
    min_df = st.number_input("Min document frequency (filter rare words)", min_value=1, max_value=10, value=2, step=1)
    max_feat = st.number_input("Max features (0 = unlimited)", min_value=0, max_value=10000, value=0, step=100)

    with st.spinner("Vectorizing corpus..."):
        t0 = time.perf_counter()
        try:
            matrix, word_vectors, vocab = cache_vectorize(corpus, rep, min_df, max_feat)
        except ValueError:
            st.error("Could not process the corpus. Please ensure it's not empty or contains only stop words.")
            return
        st.caption(f"Vectorization time: {(time.perf_counter()-t0)*1000:.1f} ms")

    with st.expander("View matrix (may be slow for large vocabularies)"):
        show_matrix = st.checkbox("Render full dense matrix (docs × terms)", value=False)
        if show_matrix:
            max_cells = 5000
            cells = matrix.shape[0] * matrix.shape[1]
            if cells > max_cells:
                st.warning(f"Matrix too big to render densely ({cells} cells > {max_cells}). "
                           "Increase filters or uncheck this option.")
            else:
                st.dataframe(pd.DataFrame(matrix.toarray(), columns=vocab,
                                          index=[f"Doc {i+1}" for i in range(len(corpus))]))

    if len(vocab) < 2:
        st.warning("Vocabulary too small (<2 words). Provide a richer corpus.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        selected_word = st.selectbox("Select a word:", sorted(vocab))
        word_idx = list(vocab).index(selected_word)

        t0 = time.perf_counter()
        sims = cosine_similarity(word_vectors[word_idx], word_vectors).ravel()
        st.caption(f"Similarity time: {(time.perf_counter()-t0)*1000:.1f} ms")

        sim_df = pd.DataFrame({'word': vocab, 'similarity': sims})
        sim_df = sim_df[sim_df['word'] != selected_word].sort_values(by='similarity', ascending=False)

        max_k = max(2, min(20, len(vocab) - 1))
        k = st.slider("Number of neighbors to show", 2, max_k, min(5, max_k))
        st.write(f"Top {k} nearest neighbors to '{selected_word}':")
        st.dataframe(sim_df.head(k).reset_index(drop=True))

    with col2:
        st.write("2D Visualization (TruncatedSVD on row-normalized word vectors)")
        with st.spinner("Projecting to 2D..."):
            t0 = time.perf_counter()
            svd = TruncatedSVD(n_components=2, random_state=42)
            vectors_2d = svd.fit_transform(word_vectors)  # accepts sparse
            vectors_2d = vectors_2d - vectors_2d.mean(axis=0, keepdims=True)
            st.caption(f"SVD time: {(time.perf_counter()-t0)*1000:.1f} ms")

        top_neighbors = set(sim_df.head(k)['word'].tolist())
        idx_map = {w: i for i, w in enumerate(vocab)}
        colors = ['red' if w == selected_word else ('orange' if w in top_neighbors else 'lightgray') for w in vocab]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=colors, alpha=0.85)
        for w in [selected_word] + list(top_neighbors):
            i = idx_map[w]
            ax.annotate(w, (vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=10,
                        color=('red' if w == selected_word else 'black'))
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title("Word Embeddings in 2D Space (Counts/TF-IDF)")
        st.pyplot(fig)

# =================================================================================
# --- Question 3: Pointwise Mutual Information (PMI) — Word–Word ---
# =================================================================================
def page_question_3():
    st.header("Question 3: Pointwise Mutual Information (PMI)")
    st.markdown("---")
    st.write("Build word–word PMI/PPMI vectors from a corpus and use cosine similarity for nearest neighbours.")

    st.subheader("Data Source")
    data_source = st.radio("Choose a corpus:", ("Use default corpus", "Input your own corpus"))
    if data_source == "Input your own corpus":
        user_corpus_text = st.text_area(
            "Enter your documents, one per line.",
            height=150,
            placeholder="The quick brown fox...\nAnother document about cats and dogs...\nAnd a third one here."
        )
        corpus = [doc.strip() for doc in user_corpus_text.split('\n') if doc.strip()]
        if not corpus:
            st.info("Please enter at least one document to proceed.")
            return
    else:
        corpus = CORPUS_DEFAULT

    colA, colB, colC = st.columns(3)
    with colA:
        min_df = st.number_input("Min document frequency", min_value=1, max_value=20, value=2, step=1)
    with colB:
        max_feat = st.number_input("Max features (0 = unlimited)", min_value=0, max_value=10000, value=0, step=100)
    with colC:
        window_size = st.slider("Sliding window size (tokens to each side)", min_value=1, max_value=10, value=4, step=1)

    use_ppmi = st.checkbox("Use Positive PMI (PPMI)", value=True,
                           help="PPMI = max(0, PMI). Checked is recommended for similarity.")

    with st.spinner("Computing word–word (P)PMI..."):
        t0 = time.perf_counter()
        vectors, vocab, pmi_df = compute_word_word_ppmi(
            corpus,
            window_size=window_size,
            min_df=min_df,
            max_features=max_feat,
            positive=use_ppmi
        )
        st.caption(f"(P)PMI compute time: {(time.perf_counter()-t0)*1000:.1f} ms")

    if len(vocab) < 2:
        st.warning("Vocabulary too small (<2 words). Provide a richer corpus or relax filters.")
        return

    st.subheader("(P)PMI Matrix (word × word)")
    with st.expander("Show matrix (may be large)"):
        st.dataframe(pmi_df.style.format("{:.2f}"))

    col1, col2 = st.columns([1, 2])
    with col1:
        selected_word = st.selectbox("Select a word:", sorted(vocab), key="ppmi_select_word")
        max_k = max(2, min(20, len(vocab) - 1))
        k = st.slider("Number of neighbors to show", 2, max_k, min(5, max_k), key="ppmi_k")

        # vectors is dense np.ndarray (row-normalized)
        i = vocab.index(selected_word)
        sims = cosine_similarity(vectors[i:i+1], vectors).ravel()
        sim_df = pd.DataFrame({'word': vocab, 'similarity': sims})
        sim_df = sim_df[sim_df['word'] != selected_word].sort_values(by='similarity', ascending=False)
        st.write(f"Top {k} nearest neighbors to '{selected_word}' using {'PPMI' if use_ppmi else 'PMI'}:")
        st.dataframe(sim_df.head(k).reset_index(drop=True))

    with col2:
        st.write("2D Visualization (TruncatedSVD on row-normalized (P)PMI word vectors)")
        with st.spinner("Projecting to 2D..."):
            t0 = time.perf_counter()
            svd = TruncatedSVD(n_components=2, random_state=42)
            vectors_2d = svd.fit_transform(vectors)  # dense array
            vectors_2d = vectors_2d - vectors_2d.mean(axis=0, keepdims=True)
            st.caption(f"SVD time: {(time.perf_counter()-t0)*1000:.1f} ms")

        top_neighbors = set(sim_df.head(k)['word'].tolist())
        idx_map = {w: i for i, w in enumerate(vocab)}
        colors = ['red' if w == selected_word else ('orange' if w in top_neighbors else 'lightgray') for w in vocab]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=colors, alpha=0.85)
        for w in [selected_word] + list(top_neighbors):
            i = idx_map[w]
            ax.annotate(w, (vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=10,
                        color=('red' if w == selected_word else 'black'))
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title(f"Word Embeddings in 2D Space ({'PPMI' if use_ppmi else 'PMI'})")
        st.pyplot(fig)

# =================================================================================
# --- Main App Navigation ---
# =================================================================================
st.title("Lab 5: Sparse Vector (embedding) - Interactive Solution")
st.markdown("---")

st.sidebar.title("Lab Sections")
option = st.sidebar.radio(
    "Choose a question to view:",
    (
        "Question 1: TF-IDF and Normalization",
        "Question 2: Cosine Similarity",
        "Question 3: Pointwise Mutual Information (PMI)"
    )
)

if option == "Question 1: TF-IDF and Normalization":
    page_question_1()
elif option == "Question 2: Cosine Similarity":
    page_question_2()
elif option == "Question 3: Pointwise Mutual Information (PMI)":
    page_question_3()