import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import random
import time
from io import StringIO

# Page Configuration
st.set_page_config(
    page_title="Lab 6: Dense Vector (Word2Vec) Embeddings",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'vocab' not in st.session_state:
    st.session_state.vocab = []
if 'word2idx' not in st.session_state:
    st.session_state.word2idx = {}
if 'idx2word' not in st.session_state:
    st.session_state.idx2word = {}

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def sigmoid(x):
    """Sigmoid activation function with numerical stability"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -250, 250)))

def preprocess_corpus(text_input):
    """Preprocess the input text and create vocabulary"""
    if isinstance(text_input, str):
        sentences = [line.strip() for line in text_input.split('\n') if line.strip()]
    else:
        sentences = text_input
    
    # Tokenize and clean
    tokens = []
    for sentence in sentences:
        words = sentence.lower().split()
        # Simple cleaning - remove punctuation
        words = [''.join(c for c in word if c.isalnum()) for word in words]
        words = [word for word in words if word and len(word) > 1]
        tokens.extend(words)
    
    # Build vocabulary
    vocab_counter = Counter(tokens)
    # Filter words that appear at least 2 times
    vocab = [word for word, count in vocab_counter.items() if count >= 2]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    return sentences, vocab, word2idx, idx2word, vocab_counter

def generate_training_pairs_skipgram(sentences, word2idx, window_size=2):
    """Generate training pairs for Skip-gram model"""
    pairs = []
    for sentence in sentences:
        words = sentence.lower().split()
        words = [''.join(c for c in word if c.isalnum()) for word in words]
        words = [word for word in words if word in word2idx]
        
        for i, center_word in enumerate(words):
            center_idx = word2idx[center_word]
            for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
                if j != i:
                    context_word = words[j]
                    context_idx = word2idx[context_word]
                    pairs.append((center_idx, context_idx))
    
    return pairs

def generate_training_pairs_cbow(sentences, word2idx, window_size=2):
    """Generate training pairs for CBOW model"""
    pairs = []
    for sentence in sentences:
        words = sentence.lower().split()
        words = [''.join(c for c in word if c.isalnum()) for word in words]
        words = [word for word in words if word in word2idx]
        
        for i, target_word in enumerate(words):
            target_idx = word2idx[target_word]
            context_indices = []
            for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
                if j != i:
                    context_word = words[j]
                    context_indices.append(word2idx[context_word])
            
            if context_indices:
                pairs.append((context_indices, target_idx))
    
    return pairs

def train_skipgram(pairs, vocab_size, vocab_counter, idx2word, embedding_dim=50, epochs=10, learning_rate=0.01, neg_samples=5):
    """Train Skip-gram model with negative sampling"""
    
    # Initialize embeddings
    W_in = (np.random.rand(vocab_size, embedding_dim) - 0.5) / embedding_dim
    W_out = (np.random.rand(vocab_size, embedding_dim) - 0.5) / embedding_dim
    
    # Create negative sampling distribution
    counts = np.array([vocab_counter[idx2word[i]] for i in range(vocab_size)], dtype=np.float64)
    probs = counts ** 0.75
    probs = probs / probs.sum()
    
    losses = []
    
    for epoch in range(epochs):
        random.shuffle(pairs)
        epoch_loss = 0.0
        
        for center_idx, context_idx in pairs:
            v_c = W_in[center_idx]  # center word embedding
            u_o = W_out[context_idx]  # context word embedding
            
            # Positive sample
            score_pos = sigmoid(np.dot(u_o, v_c))
            epoch_loss += -np.log(score_pos + 1e-10)
            
            # Gradients for positive sample
            grad_v = (score_pos - 1.0) * u_o
            grad_uo = (score_pos - 1.0) * v_c
            
            # Update context embedding
            W_out[context_idx] -= learning_rate * grad_uo
            
            # Negative samples
            neg_samples_idx = np.random.choice(vocab_size, size=neg_samples, p=probs)
            for neg_idx in neg_samples_idx:
                if neg_idx != context_idx:  # Avoid sampling the positive context
                    u_n = W_out[neg_idx]
                    score_neg = sigmoid(np.dot(u_n, v_c))
                    epoch_loss += -np.log(1.0 - score_neg + 1e-10)
                    
                    grad_un = score_neg * v_c
                    grad_v += score_neg * u_n
                    
                    W_out[neg_idx] -= learning_rate * grad_un
            
            # Update center embedding
            W_in[center_idx] -= learning_rate * grad_v
        
        losses.append(epoch_loss)
        
        if epoch % 2 == 0:
            st.write(f"Skip-gram Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.3f}")
    
    return W_in, W_out, losses

def train_cbow(pairs, vocab_size, vocab_counter, idx2word, embedding_dim=50, epochs=10, learning_rate=0.01, neg_samples=5):
    """Train CBOW model with negative sampling"""
    
    # Initialize embeddings
    W_in = (np.random.rand(vocab_size, embedding_dim) - 0.5) / embedding_dim
    W_out = (np.random.rand(vocab_size, embedding_dim) - 0.5) / embedding_dim
    
    # Create negative sampling distribution
    counts = np.array([vocab_counter[idx2word[i]] for i in range(vocab_size)], dtype=np.float64)
    probs = counts ** 0.75
    probs = probs / probs.sum()
    
    losses = []
    
    for epoch in range(epochs):
        random.shuffle(pairs)
        epoch_loss = 0.0
        
        for context_indices, target_idx in pairs:
            # Average context word embeddings
            v_context = np.mean(W_in[context_indices], axis=0)
            u_t = W_out[target_idx]
            
            # Positive sample
            score_pos = sigmoid(np.dot(u_t, v_context))
            epoch_loss += -np.log(score_pos + 1e-10)
            
            # Gradients for positive sample
            grad_u_t = (score_pos - 1.0) * v_context
            grad_v_context = (score_pos - 1.0) * u_t
            
            # Update target embedding
            W_out[target_idx] -= learning_rate * grad_u_t
            
            # Negative samples
            neg_samples_idx = np.random.choice(vocab_size, size=neg_samples, p=probs)
            for neg_idx in neg_samples_idx:
                if neg_idx != target_idx:  # Avoid sampling the positive target
                    u_n = W_out[neg_idx]
                    score_neg = sigmoid(np.dot(u_n, v_context))
                    epoch_loss += -np.log(1.0 - score_neg + 1e-10)
                    
                    grad_u_n = score_neg * v_context
                    grad_v_context += score_neg * u_n
                    
                    W_out[neg_idx] -= learning_rate * grad_u_n
            
            # Update context embeddings
            for c_idx in context_indices:
                W_in[c_idx] -= learning_rate * (grad_v_context / len(context_indices))
        
        losses.append(epoch_loss)
        
        if epoch % 2 == 0:
            st.write(f"CBOW Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.3f}")
    
    return W_in, W_out, losses

def find_nearest_words(word, embeddings, word2idx, idx2word, k=5):
    """Find k nearest words to the given word using cosine similarity"""
    if word not in word2idx:
        return []
    
    word_idx = word2idx[word]
    word_vec = embeddings[word_idx].reshape(1, -1)
    
    # Compute cosine similarities
    similarities = cosine_similarity(word_vec, embeddings).flatten()
    
    # Get top k similar words (excluding the word itself)
    top_indices = np.argsort(similarities)[::-1][1:k+1]
    
    results = []
    for idx in top_indices:
        results.append((idx2word[idx], similarities[idx]))
    
    return results

def plot_embeddings_pca(embeddings, vocab, title, selected_words=None):
    """Plot embeddings in 2D using PCA"""
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if selected_words:
        # Highlight selected words
        word_indices = [i for i, word in enumerate(vocab) if word in selected_words]
        other_indices = [i for i, word in enumerate(vocab) if word not in selected_words]
        
        # Plot other words in light gray
        if other_indices:
            ax.scatter(reduced_embeddings[other_indices, 0], 
                      reduced_embeddings[other_indices, 1], 
                      c='lightgray', alpha=0.6, s=20)
        
        # Plot selected words in different colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(selected_words)))
        for i, (word_idx, color) in enumerate(zip(word_indices, colors)):
            ax.scatter(reduced_embeddings[word_idx, 0], 
                      reduced_embeddings[word_idx, 1], 
                      c=[color], s=100, alpha=0.8)
            ax.annotate(vocab[word_idx], 
                       (reduced_embeddings[word_idx, 0], reduced_embeddings[word_idx, 1]),
                       xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    else:
        # Plot all words
        ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)
        for i, word in enumerate(vocab):
            ax.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                       xytext=(2, 2), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_similarity_heatmap(words, embeddings, word2idx, title):
    """Plot similarity heatmap for selected words"""
    if not words:
        return None
    
    # Filter words that exist in vocabulary
    valid_words = [word for word in words if word in word2idx]
    if not valid_words:
        return None
    
    # Get embeddings for selected words
    word_embeddings = np.array([embeddings[word2idx[word]] for word in valid_words])
    
    # Compute similarity matrix
    sim_matrix = cosine_similarity(word_embeddings)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(sim_matrix, 
                xticklabels=valid_words, 
                yticklabels=valid_words,
                annot=True, 
                cmap='viridis', 
                center=0,
                square=True,
                ax=ax)
    ax.set_title(title)
    
    return fig

# Main Streamlit App
def main():
    st.title("🎯 Lab 6: Dense Vector (Word2Vec) Embeddings")
    st.markdown("---")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type:",
        ["Model Training", "Word Similarity", "Visualization"]
    )
    
    if page == "Model Training":
        st.header("📚 Model Training")
        
        # Input method selection
        input_method = st.radio("Choose input method:", ["Use default corpus", "Enter custom text", "Upload text file"])
        
        corpus_text = ""
        if input_method == "Use default corpus":
            corpus_text = """NLP is fun and exciting
We are learning natural language processing
Machine learning powers modern NLP applications
Natural language processing is a fascinating field
Deep learning improves NLP performance
We enjoy exploring text mining techniques
AI is transforming language understanding
Word embeddings capture semantic relationships
Vector representations enable similarity computation
Neural networks learn meaningful word representations"""
        
        elif input_method == "Enter custom text":
            corpus_text = st.text_area("Enter your text corpus (one sentence per line):", 
                                     height=200,
                                     value="NLP is fun and exciting\nWe are learning natural language processing")
        
        elif input_method == "Upload text file":
            uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
            if uploaded_file is not None:
                corpus_text = str(uploaded_file.read(), "utf-8")
        
        if corpus_text:
            st.subheader("Training Parameters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                embedding_dim = st.slider("Embedding Dimension", 10, 100, 50)
                window_size = st.slider("Window Size", 1, 5, 2)
            
            with col2:
                epochs = st.slider("Epochs", 5, 50, 10)
                learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, format="%.3f")
            
            with col3:
                neg_samples = st.slider("Negative Samples", 1, 10, 5)
                min_count = st.slider("Min Word Count", 1, 5, 2)
            
            if st.button("Train Models", type="primary"):
                with st.spinner("Processing corpus and training models..."):
                    # Preprocess corpus
                    sentences, vocab, word2idx, idx2word, vocab_counter = preprocess_corpus(corpus_text)
                    
                    # Store in session state
                    st.session_state.vocab = vocab
                    st.session_state.word2idx = word2idx
                    st.session_state.idx2word = idx2word
                    st.session_state.sentences = sentences
                    st.session_state.vocab_counter = vocab_counter
                    
                    st.success(f"✅ Corpus processed! Vocabulary size: {len(vocab)}")
                    
                    # Generate training pairs
                    skipgram_pairs = generate_training_pairs_skipgram(sentences, word2idx, window_size)
                    cbow_pairs = generate_training_pairs_cbow(sentences, word2idx, window_size)
                    
                    st.info(f"Skip-gram training pairs: {len(skipgram_pairs)}")
                    st.info(f"CBOW training pairs: {len(cbow_pairs)}")
                    
                    # Train models
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🎯 Training Skip-gram Model")
                        start_time = time.time()
                        W_in_sg, W_out_sg, losses_sg = train_skipgram(
                            skipgram_pairs, len(vocab), vocab_counter, idx2word,
                            embedding_dim, epochs, learning_rate, neg_samples
                        )
                        sg_time = time.time() - start_time
                        st.success(f"Skip-gram training completed in {sg_time:.2f}s")
                        
                        # Store in session state
                        st.session_state.W_in_sg = W_in_sg
                        st.session_state.losses_sg = losses_sg
                    
                    with col2:
                        st.subheader("🎯 Training CBOW Model")
                        start_time = time.time()
                        W_in_cbow, W_out_cbow, losses_cbow = train_cbow(
                            cbow_pairs, len(vocab), vocab_counter, idx2word,
                            embedding_dim, epochs, learning_rate, neg_samples
                        )
                        cbow_time = time.time() - start_time
                        st.success(f"CBOW training completed in {cbow_time:.2f}s")
                        
                        # Store in session state
                        st.session_state.W_in_cbow = W_in_cbow
                        st.session_state.losses_cbow = losses_cbow
                    
                    st.session_state.models_trained = True
                    
                    # Plot training losses
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    ax1.plot(losses_sg, 'b-', linewidth=2, label='Skip-gram')
                    ax1.set_title('Skip-gram Training Loss')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss')
                    ax1.grid(True, alpha=0.3)
                    
                    ax2.plot(losses_cbow, 'r-', linewidth=2, label='CBOW')
                    ax2.set_title('CBOW Training Loss')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Loss')
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
        
        else:
            st.warning("Please provide text corpus to train the models.")
    
    elif page == "Word Similarity":
        st.header("🔍 Word Similarity Analysis")
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Please train the models first in the 'Model Training' tab.")
            return
        
        vocab = st.session_state.vocab
        word2idx = st.session_state.word2idx
        idx2word = st.session_state.idx2word
        
        # Word similarity search
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Skip-gram Model")
            query_word_sg = st.selectbox("Select a word for similarity search:", 
                                       vocab, key="sg_word")
            k_sg = st.slider("Number of similar words to find:", 1, min(20, len(vocab)-1), 5, key="sg_k")
            
            if query_word_sg:
                nearest_sg = find_nearest_words(query_word_sg, st.session_state.W_in_sg, 
                                              word2idx, idx2word, k_sg)
                
                st.write(f"**Words most similar to '{query_word_sg}':**")
                for i, (word, similarity) in enumerate(nearest_sg, 1):
                    st.write(f"{i}. **{word}** (similarity: {similarity:.4f})")
        
        with col2:
            st.subheader("🎯 CBOW Model")
            query_word_cbow = st.selectbox("Select a word for similarity search:", 
                                         vocab, key="cbow_word")
            k_cbow = st.slider("Number of similar words to find:", 1, min(20, len(vocab)-1), 5, key="cbow_k")
            
            if query_word_cbow:
                nearest_cbow = find_nearest_words(query_word_cbow, st.session_state.W_in_cbow, 
                                                word2idx, idx2word, k_cbow)
                
                st.write(f"**Words most similar to '{query_word_cbow}':**")
                for i, (word, similarity) in enumerate(nearest_cbow, 1):
                    st.write(f"{i}. **{word}** (similarity: {similarity:.4f})")
        
        # Word input for custom queries
        st.subheader("🔤 Custom Word Query")
        custom_word = st.text_input("Enter a word to find similar words:")
        
        if custom_word and custom_word.lower() in word2idx:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Skip-gram Results:**")
                nearest_sg = find_nearest_words(custom_word.lower(), st.session_state.W_in_sg, 
                                              word2idx, idx2word, 5)
                for word, sim in nearest_sg:
                    st.write(f"• {word}: {sim:.4f}")
            
            with col2:
                st.write("**CBOW Results:**")
                nearest_cbow = find_nearest_words(custom_word.lower(), st.session_state.W_in_cbow, 
                                                word2idx, idx2word, 5)
                for word, sim in nearest_cbow:
                    st.write(f"• {word}: {sim:.4f}")
        
        elif custom_word:
            st.error(f"Word '{custom_word}' not found in vocabulary.")
    
    elif page == "Visualization":
        st.header("📊 Embeddings Visualization")
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Please train the models first in the 'Model Training' tab.")
            return
        
        vocab = st.session_state.vocab
        word2idx = st.session_state.word2idx
        
        # Visualization options
        viz_type = st.selectbox("Choose visualization type:", 
                               ["PCA 2D Plot", "Similarity Heatmap", "Word Clusters"])
        
        if viz_type == "PCA 2D Plot":
            st.subheader("📈 2D PCA Visualization")
            
            # Word selection for highlighting
            selected_words = st.multiselect("Select words to highlight (optional):", vocab, default=[])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Skip-gram Embeddings**")
                fig_sg = plot_embeddings_pca(st.session_state.W_in_sg, vocab, 
                                           "Skip-gram Word Embeddings (PCA)", selected_words)
                st.pyplot(fig_sg)
            
            with col2:
                st.write("**CBOW Embeddings**")
                fig_cbow = plot_embeddings_pca(st.session_state.W_in_cbow, vocab, 
                                             "CBOW Word Embeddings (PCA)", selected_words)
                st.pyplot(fig_cbow)
        
        elif viz_type == "Similarity Heatmap":
            st.subheader("🔥 Similarity Heatmap")
            
            # Select words for heatmap
            heatmap_words = st.multiselect("Select words for similarity heatmap:", 
                                         vocab, default=vocab[:min(10, len(vocab))])
            
            if len(heatmap_words) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Skip-gram Similarity Matrix**")
                    fig_hm_sg = plot_similarity_heatmap(heatmap_words, st.session_state.W_in_sg, 
                                                      word2idx, "Skip-gram Word Similarities")
                    if fig_hm_sg:
                        st.pyplot(fig_hm_sg)
                
                with col2:
                    st.write("**CBOW Similarity Matrix**")
                    fig_hm_cbow = plot_similarity_heatmap(heatmap_words, st.session_state.W_in_cbow, 
                                                        word2idx, "CBOW Word Similarities")
                    if fig_hm_cbow:
                        st.pyplot(fig_hm_cbow)
            else:
                st.warning("Please select at least 2 words for the heatmap.")
        
        elif viz_type == "Word Clusters":
            st.subheader("🎯 Word Clustering Analysis")
            
            # Clustering around a central word
            central_word = st.selectbox("Select central word:", vocab)
            cluster_size = st.slider("Cluster size:", 3, min(15, len(vocab)), 7)
            
            if central_word:
                # Find nearest words for both models
                nearest_sg = find_nearest_words(central_word, st.session_state.W_in_sg, 
                                              word2idx, idx2word, cluster_size-1)
                nearest_cbow = find_nearest_words(central_word, st.session_state.W_in_cbow, 
                                                word2idx, idx2word, cluster_size-1)
                
                # Create cluster words list
                cluster_words_sg = [central_word] + [word for word, _ in nearest_sg]
                cluster_words_cbow = [central_word] + [word for word, _ in nearest_cbow]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Skip-gram Cluster**")
                    fig_cluster_sg = plot_embeddings_pca(st.session_state.W_in_sg, vocab, 
                                                       f"Skip-gram Cluster around '{central_word}'", 
                                                       cluster_words_sg)
                    st.pyplot(fig_cluster_sg)
                    
                    st.write("Cluster members:")
                    for word, sim in nearest_sg:
                        st.write(f"• {word} ({sim:.4f})")
                
                with col2:
                    st.write("**CBOW Cluster**")
                    fig_cluster_cbow = plot_embeddings_pca(st.session_state.W_in_cbow, vocab, 
                                                         f"CBOW Cluster around '{central_word}'", 
                                                         cluster_words_cbow)
                    st.pyplot(fig_cluster_cbow)
                    
                    st.write("Cluster members:")
                    for word, sim in nearest_cbow:
                        st.write(f"• {word} ({sim:.4f})")

    # Sidebar information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    This app implements **Word2Vec** models for dense vector embeddings:
    
    **Models:**
    - **Skip-gram**: Predicts context words from center word
    - **CBOW**: Predicts center word from context words
    
    **Features:**
    - Text preprocessing and vocabulary building
    - Negative sampling for efficient training
    - Cosine similarity for word relationships
    - PCA visualization for 2D embedding plots
    - Interactive similarity search
    
    **Technologies:**
    - NumPy for numerical computations
    - Scikit-learn for similarity and PCA
    - Matplotlib/Seaborn for visualization
    - Streamlit for web interface
    """)

if __name__ == "__main__":
    main()