import re

with open('proposal.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Introduction Subsections
text = text.replace(r'\subsection{Research Objectives and Questions}', r'\subsection{Research Objectives and Research Questions}')
text = text.replace(r'\subsection{Key Contributions}', r'\subsection{Research Motivation and Contribution}')

# 2. Rename Literature Review Section
text = text.replace(r'\section{Literature Review}', r'\section{Literature Review and Hypothesis Development}')
text = text.replace(r'\subsection{Tree-Based Ensemble Methods}', r'\subsection{Literature Review}' + '\n\n' + r'\subsubsection{Tree-Based Ensemble Methods}')

# Change subsequent subsections in Lit Review to subsubsections to keep hierarchy correct under Literature Review
text = text.replace(r'\subsection{Temporal and Graph-Based Approaches}', r'\subsubsection{Temporal and Graph-Based Approaches}')
text = text.replace(r'\subsection{Explainable AI in Finance}', r'\subsubsection{Explainable AI in Finance}')
text = text.replace(r'\subsection{Class Imbalance and Validation}', r'\subsubsection{Class Imbalance and Validation}')
text = text.replace(r'\subsection{Research Gap Summary}', r'\subsubsection{Research Gap Summary}')

# 3. Move Research Hypotheses to II
# We are moving it from its current position (which was \subsection{Research Hypotheses} under III. Conceptual Framework)
# to appear at the end of section II.
# Specifically, we want it right before % ============================================================
# % III. CONCEPTUAL FRAMEWORK or whatever it is called now.
hyp_match = re.search(r'(\\subsection\{Research Hypotheses\}.*?)(?=\n% =======+.*?\n% (?:III|IV)\.)', text, re.DOTALL)
if hyp_match:
    hyp_text = hyp_match.group(1)
    # Remove from original
    text = text.replace(hyp_text, "")
    # Change subsection title
    hyp_text_new = hyp_text.replace(r'\subsection{Research Hypotheses}', r'\subsection{Hypothesis Development}')
    
    # Identify where Lit Review ends: just before III. CONCEPTUAL FRAMEWORK
    lit_review_end_pattern = r'(?=\n% =======+.*?\n% III\.)'
    text = re.sub(lit_review_end_pattern, '\n' + hyp_text_new.replace('\\', '\\\\') + '\n', text, count=1)

# 4. Rename III. Conceptual Framework to III. Data and Methodology
text = text.replace(r'% III. CONCEPTUAL FRAMEWORK', r'% III. DATA AND METHODOLOGY')
text = text.replace(r'\section{Conceptual Framework}', r'\section{Data and Methodology}')

# Its first two subsections become 3.1
text = text.replace(r'\subsection{Multi-View Design Principle}', r'\subsection{Conceptual Framework}' + '\n\n' + r'\subsubsection{Multi-View Design Principle}')
text = text.replace(r'\subsection{Research Pipeline}', r'\subsubsection{Research Pipeline}')

# 5. Move Benchmark Datasets to III. Data and Methodology -> 3.2 Data Description
# Currently it's "\subsection{Benchmark Datasets}" somewhere after "% IV. DATA AND METHODOLOGY"
data_match = re.search(r'(\\subsection\{Benchmark Datasets\}.*?)(?=\\subsection\{Feature Engineering)', text, re.DOTALL)
if data_match:
    data_text = data_match.group(1)
    text = text.replace(data_text, "")
    
    data_text_new = data_text.replace(r'\subsection{Benchmark Datasets}', r'\subsection{Data Description}')
    
    # Insert right after \subsubsection{Research Pipeline} and its figure.
    # We find where Research Pipeline ends. In the original, it ends right before \subsection{Research Hypotheses}
    # which we moved. So now it ends right before where we originally had Hypotheses, which is where IV started.
    
    # Let's just place it before "% IV. DATA AND METHODOLOGY"
    iv_pattern = r'(?=\n% =======+.*?\n% IV\.)'
    text = re.sub(iv_pattern, '\n' + data_text_new.replace('\\', '\\\\') + '\n', text, count=1)

# 6. Change IV. DATA AND METHODOLOGY to IV. DATA PREPROCESSING
text = text.replace(r'% IV. DATA AND METHODOLOGY', r'% IV. DATA PREPROCESSING')
text = text.replace(r'\section{Data and Methodology}', r'\section{Data Preprocessing}')

# Combine "Feature Engineering" and "Data Splitting" into "4.1 Data Cleaning & Variable Construction"
text = text.replace(r'\subsection{Feature Engineering: Three-View Representation}', r'\subsection{Data Cleaning \& Variable Construction}' + '\n\n' + r'\subsubsection{Feature Engineering: Three-View Representation}')
text = text.replace(r'\subsection{Data Splitting and Resampling}', r'\subsubsection{Data Splitting and Resampling}')

# Now we need to move the modeling parts ("Multi-View Stacking Architecture", "Four-Level XAI Framework", "Evaluation Strategy") 
# FROM IV. DATA PREPROCESSING back to III. DATA AND METHODOLOGY -> 3.3 Research Methods.
methods_start = text.find(r'\subsection{Multi-View Stacking Architecture}')
if methods_start != -1:
    # Everything from '\subsection{Multi-View Stacking Architecture}' until '% V. EXPECTED RESULTS'
    methods_end = text.find(r'% V. EXPECTED RESULTS', methods_start)
    if methods_end != -1:
        # Actually want to go right before % ======== above V. EXPECTED
        methods_end = text.rfind(r'% ========', methods_start, methods_end)
        
        methods_text = text[methods_start:methods_end]
        text = text[:methods_start] + text[methods_end:]
        
        methods_text_new = r'\subsection{Research Methods}' + '\n\n' + methods_text.replace(r'\subsection{', r'\subsubsection{')
        
        # Insert back into III. Data and Methodology, which is right before IV. DATA PREPROCESSING
        insert_pos = text.find(r'% ==========================================================' + '\n' + r'% IV. DATA PREPROCESSING')
        if insert_pos != -1:
            text = text[:insert_pos] + methods_text_new + '\n' + text[insert_pos:]

# Attempt to write it out
with open('proposal_restructured.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print("Restructured LaTeX written to proposal_restructured.tex")
