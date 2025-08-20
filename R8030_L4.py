import streamlit as st
import nltk
import re
import pandas as pd
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.metrics.distance import edit_distance
from nltk.wsd import lesk
import numpy as np

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# App config
st.set_page_config(page_title="NLP Tasks Explorer", layout="wide", page_icon="🧠")

# Dark mode & light mode CSS
st.markdown("""
    <style>
        html, body, [class^="st-"] {
            background-color: transparent !important;
        }
        .main {
            background-color: transparent !important;
        }
        .stTabs [role="tab"] {
            font-size: 1.1rem;
            padding: 8px 15px;
            margin-right: 2px;
        }
        .stTabs [aria-selected="true"] {
            background: var(--primary-color, #4f8bf9) !important;
            color: var(--text-color, #fff) !important;
        }
        .stTabs [aria-selected="false"] {
            background: var(--secondary-background-color, #f5f7fa) !important;
            color: var(--text-color, #222) !important;
        }
        .highlight {
            background: rgba(79,139,249,0.13);
            border-radius: 6px;
            padding: 8px;
            color: inherit;
        }
        @media (prefers-color-scheme: dark) {
            .main, html, body, [class^="st-"] {
                background-color: #0e1117 !important;
            }
            .highlight {
                background: rgba(79,139,249,0.26);
                color: #f3f6fb !important;
            }
            .stTabs [aria-selected="true"] {
                background: #2561c4 !important;
                color: #fff !important;
            }
            .stTabs [aria-selected="false"] {
                background: #181c28 !important;
                color: #eaeef2 !important;
            }
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 NLP Tasks Explorer")
st.markdown("Explore various Natural Language Processing tasks interactively. Each tab presents a different NLP challenge with visualizations and explanations.")

# Tabs for each question
q1, q2, q3, q4, q5, q6 = st.tabs([
    "1️⃣ Positional Index",
    "2️⃣ Word Matrix",
    "3️⃣ Preprocessing",
    "4️⃣ Edit Distance",
    "5️⃣ POS Tagging",
    "6️⃣ WSD"])

## Question 1: Positional Index
with q1:
    st.header("📌 Positional Index")
    st.markdown("""
        **Build a positional index for your documents.**  
        Enter your documents below. The positional index shows where each word appears in the text.
    """)
    docs = {
        "Doc1": st.text_area("Enter Document 1", "I am a student, and I currently take MDS472C. I was a student in MDS331 last trimester."),
        "Doc2": st.text_area("Enter Document 2", "I was a student. I have taken MDS472C.")
    }

    def create_positional_index(docs):
        index = defaultdict(lambda: defaultdict(list))
        for doc_id, text in docs.items():
            tokens = word_tokenize(text)
            for pos, word in enumerate(tokens):
                word = word.lower()
                index[word][doc_id].append(pos)
        return index

    pos_index = create_positional_index(docs)

    with st.expander("🔍 Full Positional Index Table"):
        df_index = []
        for term in sorted(pos_index):
            row = {'Term': term}
            for d in docs:
                row[d] = ', '.join(str(i) for i in pos_index[term].get(d, []))
            df_index.append(row)
        st.dataframe(pd.DataFrame(df_index).set_index('Term'))

    search_word = st.text_input("Enter word(s) to find positional indexes (comma separated)", "student, MDS472C")
    search_terms = [w.strip().lower() for w in search_word.split(",") if w.strip()]
    st.markdown("**Results:**")
    for word in search_terms:
        result = pos_index.get(word, {})
        st.markdown(f"*<span class='highlight'><b>{word}</b></span>*: {result}", unsafe_allow_html=True)

## Question 2: Word Matrix
with q2:
    st.header("🧮 Word Matrix (Term-Document Incidence)")
    st.markdown("""
        **See which words appear in which document.**  
        The matrix below shows term presence (1) or absence (0) for each document.
    """)
    all_words = sorted(set(word_tokenize(docs["Doc1"]) + word_tokenize(docs["Doc2"])))
    matrix = []
    for word in all_words:
        row = [word,
               int(word in word_tokenize(docs["Doc1"])),
               int(word in word_tokenize(docs["Doc2"]))]
        matrix.append(row)

    df = pd.DataFrame(matrix, columns=["Term", "Doc1", "Doc2"])
    st.dataframe(df.style.highlight_max(axis=0, color='rgba(79,139,249,0.13)'))

## Question 3: Preprocessing
with q3:
    st.header("🔄 Linguistic Preprocessing")
    st.markdown("""
        **Tokenization, Stemming, Lemmatization & Frequency Analysis**  
        Enter multiple documents (each on a new line) to see how they're processed.
    """)
    text_input = st.text_area("Enter your documents (each separated by a new line)", "This is the first doc.\nSecond document is here.")
    user_docs = [d for d in text_input.split("\n") if d.strip()]

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    tokens_list, stemmed_list, lemmatized_list = [], [], []

    for doc in user_docs:
        tokens = word_tokenize(doc.lower())
        stemmed = [stemmer.stem(t) for t in tokens]
        lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
        tokens_list.extend(tokens)
        stemmed_list.extend(stemmed)
        lemmatized_list.extend(lemmatized)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Tokenized**")
        st.code(tokens_list)
    with col2:
        st.markdown("**Stemmed**")
        st.code(stemmed_list)
    with col3:
        st.markdown("**Lemmatized**")
        st.code(lemmatized_list)

    freq = Counter(tokens_list)
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    st.markdown("**Word Frequency:**")
    freq_df = pd.DataFrame(sorted_freq, columns=["Word", "Count"])
    st.bar_chart(freq_df.set_index("Word"))

    st.markdown("---")
    st.subheader("Quick Edit Distance Checker")
    word1 = st.text_input("Enter first word for edit distance", "cat")
    word2 = st.text_input("Enter second word for edit distance", "cut")
    if word1 and word2:
        dist = edit_distance(word1, word2)
        st.success(f"Edit distance between '{word1}' and '{word2}' is **{dist}**")

## Question 4: Levenshtein Distance
with q4:
    st.header("📝 Levenshtein (Edit) Distance Visualizer")
    st.markdown("""
        **See how two words align using edit distance.**  
        Enter two words below to see the step-by-step alignment and operations.
    """)
    wordA = st.text_input("Enter Word A", "characterization")
    wordB = st.text_input("Enter Word B", "categorization")

    def levenshtein_dp_with_trace(a, b):
        m, n = len(a), len(b)
        dp = np.zeros((m+1, n+1), dtype=int)
        ops = [[None]*(n+1) for _ in range(m+1)]

        for i in range(m+1):
            dp[i][0] = i
            ops[i][0] = 'D' if i != 0 else 'M'
        for j in range(n+1):
            dp[0][j] = j
            ops[0][j] = 'I' if j != 0 else 'M'

        for i in range(1, m+1):
            for j in range(1, n+1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                    ops[i][j] = 'M'
                else:
                    insert = dp[i][j-1] + 1
                    delete = dp[i-1][j] + 1
                    substitute = dp[i-1][j-1] + 1
                    min_val = min(insert, delete, substitute)
                    dp[i][j] = min_val
                    if min_val == substitute:
                        ops[i][j] = 'S'
                    elif min_val == insert:
                        ops[i][j] = 'I'
                    else:
                        ops[i][j] = 'D'
        return dp, ops

    def traceback_ops(a, b, ops):
        i, j = len(a), len(b)
        aligned_a = []
        aligned_b = []
        operations = []
        ins = dels = subs = matches = 0

        while i > 0 or j > 0:
            op = ops[i][j]
            if op == 'M':
                aligned_a.append(a[i-1])
                aligned_b.append(b[j-1])
                operations.append('*')
                matches += 1
                i -= 1
                j -= 1
            elif op == 'S':
                aligned_a.append(a[i-1])
                aligned_b.append(b[j-1])
                operations.append('s')
                subs += 1
                i -= 1
                j -= 1
            elif op == 'I':
                aligned_a.append('-')
                aligned_b.append(b[j-1])
                operations.append('-')
                ins += 1
                j -= 1
            elif op == 'D':
                aligned_a.append(a[i-1])
                aligned_b.append('-')
                operations.append('-')
                dels += 1
                i -= 1

        return (''.join(reversed(aligned_a)),
                ''.join(reversed(aligned_b)),
                ''.join(reversed(operations)),
                ins, dels, subs, matches)

    if wordA and wordB:
        dp_matrix, op_matrix = levenshtein_dp_with_trace(wordA, wordB)
        aligned_a, aligned_b, ops, ins, dels, subs, matches = traceback_ops(wordA, wordB, op_matrix)

        index_labels = ["0 "] + [f"{i} {ch}" for i, ch in enumerate(wordA, start=1)]
        column_labels = ["0 "] + [f"{j} {ch}" for j, ch in enumerate(wordB, start=1)]

        labeled_matrix = pd.DataFrame(dp_matrix, index=index_labels, columns=column_labels)

        with st.expander("🟦 Edit Distance Matrix (with Index & Letters)", expanded=False):
            st.dataframe(labeled_matrix)

        st.markdown("**Alignment:**")
        st.code(f"Word A : {aligned_a}\nWord B : {aligned_b}\nOpertn : {ops}")

        st.markdown("""
            <div class='highlight'>
            <b>Edit Summary:</b><br>
            <ul>
                <li>Total Minimum Edit Distance: <b>{}</b></li>
                <li>Matches: <b>{}</b></li>
                <li>Insertions: <b>{}</b></li>
                <li>Deletions: <b>{}</b></li>
                <li>Substitutions: <b>{}</b></li>
            </ul>
            </div>
        """.format(dp_matrix[-1][-1], matches, ins, dels, subs), unsafe_allow_html=True)

## Question 5: POS Tagging with HMM
with q5:
    st.header("🔤 POS Tagging (Hidden Markov Model)")
    st.markdown("""
        **Train a simple HMM for POS tagging!**  
        Enter sentences for training, then see the probabilities for transitions and emissions.
    """)
    default_corpus = """The cat chased the rat
A rat can run
The dog can chase the cat"""
    corpus_input = st.text_area("Enter your training corpus (one sentence per line)", default_corpus)
    test_sentence = st.text_input("Enter test sentence", "The rat can chase the cat")

    corpus = [line.lower().strip() for line in corpus_input.strip().split("\n") if line.strip()]

    tagged_corpus = []
    for sent in corpus:
        tokens = word_tokenize(sent)
        tagged = nltk.pos_tag(tokens)
        tagged_corpus.append(tagged)

    emissions = defaultdict(Counter)
    transitions = defaultdict(lambda: defaultdict(Counter))

    for sentence in tagged_corpus:
        prev_tag = None
        prev_word = None
        for word, tag in sentence:
            emissions[tag][word] += 1
            if prev_tag is not None:
                transitions[prev_tag][prev_word][tag] += 1
            prev_tag = tag
            prev_word = word

    def normalize(counter):
        total = sum(counter.values())
        return {k: v / total for k, v in counter.items()}

    trans_probs = {}
    for prev_tag in transitions:
        for prev_word in transitions[prev_tag]:
            key = f"{prev_tag} ({prev_word})"
            trans_probs[key] = normalize(transitions[prev_tag][prev_word])

    emiss_probs = {k: normalize(v) for k, v in emissions.items()}

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Transition Probabilities (Tag with Word):**")
        trans_df = pd.DataFrame(trans_probs).fillna(0).T
        st.dataframe(trans_df.style.format("{:.3f}"))
    with col2:
        st.markdown("**Emission Probabilities:**")
        emiss_df = pd.DataFrame(emiss_probs).fillna(0).T
        st.dataframe(emiss_df.style.format("{:.3f}"))

## Question 6: Word Sense Disambiguation
with q6:
    st.header("🗣️ Word Sense Disambiguation (WSD)")
    st.markdown("""
        **Disambiguate word senses using Lesk algorithm.**  
        Enter a sentence, and see possible senses and the chosen meaning for each open-class word.
    """)
    sample = st.text_area("Enter a sentence for WSD", "The bank will not lend money to the poor risk.")
    words = word_tokenize(sample)
    tagged = nltk.pos_tag(words)

    def is_open_class(pos):
        return pos.startswith('N') or pos.startswith('V') or pos.startswith('J') or pos.startswith('R')

    senses = {}
    wsd_results = {}
    for word, pos in tagged:
        if is_open_class(pos):
            synsets = wn.synsets(word)
            senses[word] = [s.definition() for s in synsets]
            disambiguated = lesk(words, word)
            if disambiguated:
                wsd_results[word] = disambiguated.definition()

    st.subheader("🔎 All Senses for Open-Class Words:")
    for word, defs in senses.items():
        st.markdown(f"<b>{word}</b>:", unsafe_allow_html=True)
        for i, d in enumerate(defs):
            st.markdown(f"&nbsp;&nbsp;{i+1}. {d}")

    st.subheader("✅ Disambiguated Word Senses (Lesk):")
    for word, meaning in wsd_results.items():
        st.markdown(f"*<span class='highlight'><b>{word}</b></span>*: {meaning}", unsafe_allow_html=True)