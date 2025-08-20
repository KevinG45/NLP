import streamlit as st
import pandas as pd

# Set page config
st.set_page_config(
    page_title="NLP Lab 2: Edit Distance & Sequence Alignment",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Core Functions
def levenshtein_distance(str1, str2):
    """
    Calculate the minimum edit distance between two strings.
    Uses insertion cost 1, deletion cost 1, substitution cost 1.
    """
    m, n = len(str1), len(str2)
    
    # Create matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]  # No cost for match
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # Deletion
                    dp[i][j-1],    # Insertion
                    dp[i-1][j-1]   # Substitution
                )
    
    return dp[m][n], dp

def sequence_alignment(seq1, seq2):
    """
    Align two sequences using Needleman-Wunsch algorithm.
    Match = 2, Mismatch = -1, Gap = -1
    """
    m, n = len(seq1), len(seq2)
    
    # Scoring parameters
    match_score = 2
    mismatch_score = -1
    gap_score = -1
    
    # Create scoring matrix
    score = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize first row and column with gap penalties
    for i in range(m + 1):
        score[i][0] = i * gap_score
    for j in range(n + 1):
        score[0][j] = j * gap_score
    
    # Fill scoring matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = score[i-1][j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_score)
            delete = score[i-1][j] + gap_score
            insert = score[i][j-1] + gap_score
            score[i][j] = max(match, delete, insert)
    
    # Traceback to find alignment
    align1, align2 = "", ""
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and score[i][j] == score[i-1][j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_score):
            align1 = seq1[i-1] + align1
            align2 = seq2[j-1] + align2
            i -= 1
            j -= 1
        elif i > 0 and score[i][j] == score[i-1][j] + gap_score:
            align1 = seq1[i-1] + align1
            align2 = "-" + align2
            i -= 1
        else:
            align1 = "-" + align1
            align2 = seq2[j-1] + align2
            j -= 1
    
    return align1, align2, score

def display_matrix(matrix, str1, str2, title):
    """Display the DP matrix in a nice format"""
    if not matrix:
        return
    
    # Create DataFrame for better display
    # Add empty string for the top-left corner, then characters of str2
    cols = [''] + [''] + list(str2)
    rows = [''] + list(str1)
    
    df_data = []
    for i, row in enumerate(matrix):
        if i < len(rows):
            row_data = [rows[i]] + row
            df_data.append(row_data)
    
    # Make sure we have the right number of columns
    if df_data and len(df_data[0]) != len(cols):
        # Adjust columns to match data
        cols = cols[:len(df_data[0])]
    
    df = pd.DataFrame(df_data, columns=cols)
    st.subheader(title)
    st.dataframe(df, use_container_width=True)

# Streamlit App
def main():
    st.title("🧬 NLP Lab 2: Edit Distance & Sequence Alignment")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a function:",
        ["Edit Distance", "Sequence Alignment", "Test Cases", "About"]
    )
    
    if page == "Edit Distance":
        st.header("📝 Edit Distance Calculator")
        st.markdown("Calculate the minimum edit distance between two strings using dynamic programming.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            str1 = st.text_input("Enter first string:", value="leda", key="edit_str1")
        
        with col2:
            str2 = st.text_input("Enter second string:", value="deal", key="edit_str2")
        
        if st.button("Calculate Edit Distance", type="primary"):
            if str1 and str2:
                distance, dp_matrix = levenshtein_distance(str1, str2)
                
                st.success(f"✨ Edit distance between '{str1}' and '{str2}' is **{distance}**")
                
                # Show step-by-step explanation
                st.markdown("### Step-by-step explanation:")
                st.markdown("""
                - **Insertion**: Add a character (cost = 1)
                - **Deletion**: Remove a character (cost = 1)
                - **Substitution**: Replace a character (cost = 1)
                - **Match**: Characters are the same (cost = 0)
                """)
                
                # Display matrix
                with st.expander("Show DP Matrix"):
                    display_matrix(dp_matrix, str1, str2, "Dynamic Programming Matrix")
            else:
                st.error("Please enter both strings!")
    
    elif page == "Sequence Alignment":
        st.header("🧬 Sequence Alignment")
        st.markdown("Align two sequences using the Needleman-Wunsch algorithm.")
        
        # Default sequences
        default_seq1 = "AGGCTATCACCTGACCTCCAGGCCGATGCCC"
        default_seq2 = "TAGCTATCACGACCGCGGTCGATTTGCCCGAC"
        
        col1, col2 = st.columns(2)
        
        with col1:
            seq1 = st.text_area("Enter sequence 1:", value=default_seq1, height=100, key="align_seq1")
        
        with col2:
            seq2 = st.text_area("Enter sequence 2:", value=default_seq2, height=100, key="align_seq2")
        
        # Scoring parameters info
        st.markdown("### Scoring Parameters:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("**Match**: +2")
        with col2:
            st.info("**Mismatch**: -1")
        with col3:
            st.info("**Gap**: -1")
        
        if st.button("Align Sequences", type="primary"):
            if seq1 and seq2:
                aligned1, aligned2, score_matrix = sequence_alignment(seq1, seq2)
                
                st.success("✨ Sequence alignment completed!")
                
                # Display results
                st.markdown("### Alignment Results:")
                
                # Create two columns for better display
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Sequence 1 (aligned):**")
                    st.code(aligned1, language=None)
                
                with col2:
                    st.markdown("**Sequence 2 (aligned):**")
                    st.code(aligned2, language=None)
                
                # Calculate alignment statistics
                matches = sum(1 for a, b in zip(aligned1, aligned2) if a == b and a != '-')
                mismatches = sum(1 for a, b in zip(aligned1, aligned2) if a != b and a != '-' and b != '-')
                gaps = sum(1 for a, b in zip(aligned1, aligned2) if a == '-' or b == '-')
                
                st.markdown("### Alignment Statistics:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Matches", matches)
                with col2:
                    st.metric("Mismatches", mismatches)
                with col3:
                    st.metric("Gaps", gaps)
                
                # Show scoring matrix
                with st.expander("Show Scoring Matrix"):
                    display_matrix(score_matrix, seq1, seq2, "Needleman-Wunsch Scoring Matrix")
                    
            else:
                st.error("Please enter both sequences!")
    
    elif page == "Test Cases":
        st.header("🧪 Test Cases")
        
        test_type = st.selectbox("Select test type:", ["Edit Distance Tests", "Sequence Alignment Tests"])
        
        if test_type == "Edit Distance Tests":
            st.subheader("Edit Distance Test Cases")
            
            test_cases = [
                ("leda", "deal", 3, "Given example"),
                ("drive", "brief", 3, "From Q1 manual calculation"),
                ("drive", "divers", 3, "From Q1 manual calculation"),
                ("kitten", "sitting", 3, "Classic example"),
                ("abc", "def", 3, "All substitutions"),
                ("", "hello", 5, "Empty to string"),
                ("world", "", 5, "String to empty"),
                ("same", "same", 0, "Identical strings"),
                ("a", "b", 1, "Single substitution"),
                ("insert", "in", 4, "Multiple deletions"),
            ]
            
            if st.button("Run Edit Distance Tests"):
                results = []
                for i, (str1, str2, expected, description) in enumerate(test_cases, 1):
                    result, _ = levenshtein_distance(str1, str2)
                    status = "✅ PASS" if result == expected else "❌ FAIL"
                    results.append({
                        "Test": i,
                        "String 1": str1,
                        "String 2": str2,
                        "Expected": expected,
                        "Got": result,
                        "Status": status,
                        "Description": description
                    })
                
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                passed = sum(1 for r in results if "PASS" in r["Status"])
                total = len(results)
                st.success(f"Tests passed: {passed}/{total}")
        
        else:
            st.subheader("Sequence Alignment Test Cases")
            
            test_cases = [
                ("AGGCTATCACCTGACCTCCAGGCCGATGCCC", "TAGCTATCACGACCGCGGTCGATTTGCCCGAC", "Given DNA sequences"),
                ("ACGT", "AGT", "Simple DNA sequence"),
                ("ATCG", "ATCG", "Identical sequences"),
                ("AAAA", "TTTT", "Completely different sequences"),
                ("ATCGATCG", "ATC", "Subsequence alignment")
            ]
            
            if st.button("Run Sequence Alignment Tests"):
                for i, (seq1, seq2, description) in enumerate(test_cases, 1):
                    with st.expander(f"Test {i}: {description}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Input Sequence 1:**")
                            st.code(seq1, language=None)
                        with col2:
                            st.markdown("**Input Sequence 2:**")
                            st.code(seq2, language=None)
                        
                        aligned1, aligned2, _ = sequence_alignment(seq1, seq2)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Aligned Sequence 1:**")
                            st.code(aligned1, language=None)
                        with col2:
                            st.markdown("**Aligned Sequence 2:**")
                            st.code(aligned2, language=None)
    
    else:  # About page
        st.header("📖 About")
        st.markdown("""
        This Streamlit app implements two fundamental algorithms in bioinformatics and natural language processing:
        
        ## 🔧 Edit Distance (Levenshtein Distance)
        - Calculates the minimum number of operations needed to transform one string into another
        - Operations: insertion, deletion, substitution (each with cost 1)
        - Uses dynamic programming for efficient computation
        - Time complexity: O(m×n) where m and n are string lengths
        
        ## 🧬 Sequence Alignment (Needleman-Wunsch)
        - Global sequence alignment algorithm
        - Scoring system: Match (+2), Mismatch (-1), Gap (-1)
        - Finds optimal alignment between two sequences
        - Commonly used in DNA, RNA, and protein sequence analysis
        
        ## 🎯 Features
        - Interactive web interface
        - Real-time calculations
        - Matrix visualization
        - Comprehensive test cases
        - Detailed explanations
        
        ## 🚀 Usage
        1. Select a function from the sidebar
        2. Enter your input strings/sequences
        3. Click the calculate button
        4. View results and detailed analysis
        
        ---
        **NLP Lab 2 Assignment** | Created with Streamlit 🎈
        """)

if __name__ == "__main__":
    main()