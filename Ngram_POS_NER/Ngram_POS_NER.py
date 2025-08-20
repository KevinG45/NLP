import streamlit as st
import nltk
import math
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from nltk.tag import hmm
from nltk.corpus import treebank
import pandas as pd

# Page config
st.set_page_config(
    page_title="NLP Lab - Interactive Analysis",
    page_icon="🔤",
    layout="wide"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

@st.cache_resource
def setup_nltk():
    """Download required NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/treebank')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        nltk.data.find('chunkers/maxent_ne_chunker')
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('treebank', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)

@st.cache_resource
def train_hmm_tagger():
    """Train HMM tagger with caching"""
    train_sents = treebank.tagged_sents()[:10000]
    
    hmm_trainer = hmm.HiddenMarkovModelTrainer()
    hmm_tagger = hmm_trainer.train_supervised(train_sents)
    return hmm_tagger

def plot_distribution(counter, title):
    """Create matplotlib plot for distribution"""
    if not counter:
        st.warning("No data to plot")
        return
    
    # Clear any existing plots to prevent memory issues
    plt.clf()
    
    words, counts = zip(*counter.most_common(10))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(words, counts, color='skyblue')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def ngram_analysis():
    """N-gram Analysis Section"""
    st.header("🔤 N-gram Probabilities & Perplexity")
    
    # Input section
    corpus = st.text_area(
        "Enter your corpus:",
        value="Sam likes green apples. Mary likes red apples. Sam went home.",
        height=100,
        help="Enter text to analyze. The model will learn from this corpus."
    )
    
    if corpus.strip():
        st.info(f"**Corpus entered:** {corpus}")
        
        # Process corpus - add sentence boundaries properly
        sentences = nltk.sent_tokenize(corpus.lower())
        all_tokens = []
        
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            # Add sentence boundaries for each sentence
            all_tokens.extend(['<s>'] + tokens + ['</s>'])
        
        # Count n-grams
        unigram_counts = Counter(all_tokens)
        bigram_counts = defaultdict(int)
        
        for i in range(len(all_tokens)-1):
            bigram = (all_tokens[i], all_tokens[i+1])
            bigram_counts[bigram] += 1
        
        # Calculate probabilities
        total_unigrams = sum(unigram_counts.values())
        unigram_probs = {w: c/total_unigrams for w, c in unigram_counts.items()}
        bigram_probs = {}
        
        # Fix: Calculate bigram probabilities correctly
        for (w1, w2), count in bigram_counts.items():
            if unigram_counts[w1] > 0:  # Avoid division by zero
                bigram_probs[(w1, w2)] = count / unigram_counts[w1]
        
        # Display results in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Top Unigram Probabilities")
            unigram_data = []
            for word, prob in sorted(unigram_probs.items(), key=lambda x: x[1], reverse=True)[:10]:
                unigram_data.append({"Word": word, "Probability": f"{prob:.4f}"})
            st.dataframe(pd.DataFrame(unigram_data), use_container_width=True)
        
        with col2:
            st.subheader("📊 Top Bigram Probabilities")
            bigram_data = []
            sorted_bigrams = sorted(bigram_probs.items(), key=lambda x: x[1], reverse=True)[:10]
            for (w1, w2), prob in sorted_bigrams:
                bigram_data.append({
                    "Bigram": f"{w1} → {w2}", 
                    "Probability": f"P({w2}|{w1}) = {prob:.4f}"
                })
            st.dataframe(pd.DataFrame(bigram_data), use_container_width=True)
        
        # Custom queries - Fix: Use actual tokens from corpus
        st.subheader("🔍 Custom Bigram Queries")
        col3, col4 = st.columns(2)
        with col3:
            sam_likes_prob = bigram_probs.get(('sam', 'likes'), 0.0)
            st.metric("P(likes|sam)", f"{sam_likes_prob:.4f}")
        with col4:
            likes_green_prob = bigram_probs.get(('likes', 'green'), 0.0)
            st.metric("P(green|likes)", f"{likes_green_prob:.4f}")
        
        # Perplexity calculation
        st.subheader("🎯 Perplexity Calculation")
        test_sentence = st.text_input(
            "Enter a sentence for perplexity calculation:",
            value="Sam likes apples",
            help="The model will calculate how surprised it is by this sentence"
        )
        
        if test_sentence.strip():
            st.info(f"**Test sentence:** {test_sentence}")
            
            # Process test sentence properly
            test_sentence_lower = test_sentence.lower()
            test_tokens = ['<s>'] + nltk.word_tokenize(test_sentence_lower) + ['</s>']
            log_prob = 0
            N = len(test_tokens) - 1  # Number of bigrams
            
            # Show detailed calculation
            calculation_data = []
            unseen_bigrams = 0
            
            for i in range(N):
                w1, w2 = test_tokens[i], test_tokens[i+1]
                
                # Check if bigram exists in training data
                if (w1, w2) in bigram_probs:
                    prob = bigram_probs[(w1, w2)]
                    status = "✓ Seen"
                else:
                    # Apply simple add-1 smoothing for unseen bigrams
                    vocab_size = len(unigram_counts)
                    prob = 1.0 / (unigram_counts.get(w1, 0) + vocab_size)
                    unseen_bigrams += 1
                    status = "✗ Unseen"
                
                log_prob += math.log(prob) if prob > 0 else math.log(1e-10)
                calculation_data.append({
                    "Bigram": f"{w1} → {w2}",
                    "Probability": f"{prob:.8f}",
                    "Log Probability": f"{math.log(prob if prob > 0 else 1e-10):.4f}",
                    "Status": status
                })
            
            # Calculate perplexity
            perplexity = math.exp(-log_prob / N) if N > 0 else float('inf')
            
            col5, col6 = st.columns([2, 1])
            with col5:
                st.dataframe(pd.DataFrame(calculation_data), use_container_width=True)
                
            with col6:
                st.metric("Perplexity", f"{perplexity:.2f}")
                st.metric("Unseen Bigrams", f"{unseen_bigrams}/{N}")
                
                # More nuanced interpretation based on corpus size and unseen bigrams
                corpus_tokens = len(all_tokens)
                if unseen_bigrams == 0:
                    st.success("Perfect! All bigrams seen in training")
                elif unseen_bigrams == N:
                    st.error("All bigrams unseen - very poor fit")
                elif perplexity < 10:
                    st.success("Excellent model fit")
                elif perplexity < 50:
                    st.info("Good model fit")
                elif perplexity < 200:
                    st.warning("Moderate fit - some unseen patterns")
                else:
                    st.error("Poor fit - many unseen patterns")
                
                # Show interpretation
                st.markdown(f"""
                **Interpretation:**
                - Training corpus: {corpus_tokens} tokens
                - Vocabulary size: {len(unigram_counts)} unique words
                - Lower perplexity = better fit
                """)
        
        # Add example demonstrating the effect
        st.subheader("💡 Try These Examples")
        example_col1, example_col2 = st.columns(2)
        with example_col1:
            st.markdown("**Should have LOW perplexity:**")
            st.code("sam likes green apples")
            st.code("mary likes red apples")
        with example_col2:
            st.markdown("**Should have HIGH perplexity:**")
            st.code("the elephant flies quickly")
            st.code("quantum physics is fascinating")
        
        # Visualizations
        st.subheader("📈 Visualizations")
        col7, col8 = st.columns(2)
        
        with col7:
            fig1 = plot_distribution(unigram_counts, "Top Unigrams Distribution")
            if fig1:
                st.pyplot(fig1)
                plt.close()  # Clean up memory
        
        with col8:
            bigram_words = Counter({f"{k[0]} {k[1]}": v for k, v in bigram_counts.items()})
            fig2 = plot_distribution(bigram_words, "Top Bigrams Distribution")
            if fig2:
                st.pyplot(fig2)
                plt.close()  # Clean up memory

def pos_tagging():
    """POS Tagging Section"""
    st.header("🏷️ POS Tagging using HMM")
    
    # Load model
    with st.spinner("Loading HMM tagger..."):
        hmm_tagger = train_hmm_tagger()
    
    sentence = st.text_input(
        "Enter a sentence to tag:",
        value="The quick brown fox jumps over the lazy dog",
        help="Enter a sentence to get POS tags for each word"
    )
    
    if sentence.strip():
        st.info(f"**Sentence entered:** {sentence}")
        
        tokens = nltk.word_tokenize(sentence)
        with st.spinner("Tagging words..."):
            tagged = hmm_tagger.tag(tokens)
        
        # Display results
        st.subheader("🎯 Tagged Output")
        
        # Create a nice table
        tag_data = []
        for word, tag in tagged:
            tag_data.append({
                "Word": word,
                "POS Tag": tag,
                "Description": get_tag_description(tag)
            })
        
        st.dataframe(pd.DataFrame(tag_data), use_container_width=True)
        
        # Tag distribution
        tag_counts = Counter(tag for _, tag in tagged)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("📊 Tag Statistics")
            for tag, count in tag_counts.most_common():
                st.metric(tag, count)
        
        with col2:
            st.subheader("📈 POS Tag Distribution")
            plt.clf()  # Clear previous plots
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(tag_counts.keys(), tag_counts.values(), color='orange')
            ax.set_title("POS Tag Distribution", fontsize=14, fontweight='bold')
            ax.set_xlabel("POS Tags")
            ax.set_ylabel("Frequency")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()  # Clean up memory

def ner_analysis():
    """Named Entity Recognition using NLTK"""
    st.header("🎭 Named Entity Recognition (NLTK)")
    
    st.info("💡 **Using NLTK NER** - spaCy temporarily unavailable due to compatibility issues")
    
    sentence = st.text_input(
        "Enter a sentence for NER:",
        value="Steve Jobs founded Apple Inc. in Cupertino, California",
        help="Enter text to identify named entities using NLTK"
    )
    
    if sentence.strip():
        st.info(f"**Sentence entered:** {sentence}")
        
        with st.spinner("Processing named entities..."):
            # Tokenize and POS tag
            tokens = nltk.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(tokens)
            
            # Named entity chunking
            tree = nltk.ne_chunk(pos_tags)
        
        # Fix: Better entity extraction logic
        entities = []
        i = 0
        
        for subtree in tree:
            if hasattr(subtree, 'label'):  # It's a named entity
                entity_text = ' '.join([token for token, pos in subtree.leaves()])
                entity_label = subtree.label()
                entities.append((entity_text, entity_label))
        
        # Create IOB format output
        st.subheader("🏷️ NER Output (IOB format)")
        
        ner_data = []
        current_entity = None
        entity_position = "O"
        
        # Fix: Better IOB tagging
        for i, subtree in enumerate(tree):
            if hasattr(subtree, 'label'):  # Named entity
                for j, (token, pos) in enumerate(subtree.leaves()):
                    iob_tag = f"B-{subtree.label()}" if j == 0 else f"I-{subtree.label()}"
                    ner_data.append({
                        "Token": token,
                        "POS": pos,
                        "IOB Tag": iob_tag,
                        "Entity Type": get_nltk_entity_description(subtree.label())
                    })
            else:  # Regular token
                token, pos = subtree
                ner_data.append({
                    "Token": token,
                    "POS": pos,
                    "IOB Tag": "O",
                    "Entity Type": "Not an entity"
                })
        
        st.dataframe(pd.DataFrame(ner_data), use_container_width=True)
        
        # Entity summary
        if entities:
            st.subheader("✨ Identified Entities")
            
            col1, col2 = st.columns(2)
            
            with col1:
                entity_data = []
                for text, label in entities:
                    entity_data.append({
                        "Entity": text,
                        "Type": label,
                        "Description": get_nltk_entity_description(label)
                    })
                st.dataframe(pd.DataFrame(entity_data), use_container_width=True)
            
            with col2:
                # Entity type distribution
                ent_labels = [label for _, label in entities]
                label_counts = Counter(ent_labels)
                
                plt.clf()  # Clear previous plots
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.bar(label_counts.keys(), label_counts.values(), color='blue')
                ax.set_title("Named Entity Types (NLTK)", fontsize=14, fontweight='bold')
                ax.set_xlabel("Entity Label")
                ax.set_ylabel("Count")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()  # Clean up memory
        else:
            st.warning("No named entities found in the text.")

def get_tag_description(tag):
    """Get description for POS tags"""
    tag_descriptions = {
        'DT': 'Determiner',
        'JJ': 'Adjective',
        'NN': 'Noun (singular)',
        'NNS': 'Noun (plural)',
        'NNP': 'Proper noun (singular)',
        'NNPS': 'Proper noun (plural)',
        'VB': 'Verb (base form)',
        'VBD': 'Verb (past tense)',
        'VBG': 'Verb (gerund)',
        'VBN': 'Verb (past participle)',
        'VBP': 'Verb (present)',
        'VBZ': 'Verb (3rd person singular)',
        'RB': 'Adverb',
        'IN': 'Preposition',
        'CC': 'Coordinating conjunction',
        'PRP': 'Personal pronoun',
        'PRP$': 'Possessive pronoun',
        'CD': 'Cardinal number',
        'LS': 'List marker',
        'TO': 'Infinitive marker',
        'WP': 'Wh-pronoun',
        'WDT': 'Wh-determiner'
    }
    return tag_descriptions.get(tag, f'Unknown tag ({tag})')

def get_nltk_entity_description(label):
    """Get description for NLTK NER labels"""
    entity_descriptions = {
        'PERSON': 'People, including fictional characters',
        'ORGANIZATION': 'Companies, agencies, institutions',
        'GPE': 'Geopolitical entities (countries, cities, states)',
        'LOCATION': 'Non-GPE locations',
        'FACILITY': 'Buildings, airports, highways, bridges',
        'GSP': 'Geopolitical entities'
    }
    return entity_descriptions.get(label, f'Entity type: {label}')

def main():
    """Main Streamlit app"""
    # Setup
    setup_nltk()
    
    # Title and description
    st.title("🔤 NLP Lab - Interactive Analysis")
    st.markdown("""
    **Natural Language Processing Laboratory**
    
    Explore three fundamental NLP techniques:
    - **N-gram Language Modeling** with perplexity calculation
    - **Part-of-Speech Tagging** using Hidden Markov Models
    - **Named Entity Recognition** using NLTK
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type:",
        ["N-gram Analysis", "POS Tagging", "Named Entity Recognition"]
    )
    
    # Add information sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    This interactive app demonstrates classical and modern NLP techniques.
    
    **Technologies used:**
    - NLTK for tokenization, HMM, and NER
    - Streamlit for web interface
    - Matplotlib for visualizations
    """)
    
    # Route to appropriate page
    if page == "N-gram Analysis":
        ngram_analysis()
    elif page == "POS Tagging":
        pos_tagging()
    elif page == "Named Entity Recognition":
        ner_analysis()

if __name__ == "__main__":
    main()