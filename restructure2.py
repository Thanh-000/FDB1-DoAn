import re

with open('proposal.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Introduction Subsections
text = text.replace(r'\subsection{Research Objectives and Questions}', r'\subsection{Research Objectives and Research Questions}')
text = text.replace(r'\subsection{Key Contributions}', r'\subsection{Research Motivation and Contribution}')

# 2. Rename Literature Review Section
text = text.replace(r'\section{Literature Review}', r'\section{Literature Review and Hypothesis Development}')
text = text.replace(r'\subsection{Tree-Based Ensemble Methods}', r'\subsection{Literature Review}' + '\n\n' + r'\subsubsection{Tree-Based Ensemble Methods}')

text = text.replace(r'\subsection{Temporal and Graph-Based Approaches}', r'\subsubsection{Temporal and Graph-Based Approaches}')
text = text.replace(r'\subsection{Explainable AI in Finance}', r'\subsubsection{Explainable AI in Finance}')
text = text.replace(r'\subsection{Class Imbalance and Validation}', r'\subsubsection{Class Imbalance and Validation}')
text = text.replace(r'\subsection{Research Gap Summary}', r'\subsubsection{Research Gap Summary}')

# 3. Move Hypotheses from III. Conceptual Framework to II. Literature Review
# Extract the Research Hypotheses block
hyp_match = re.search(r'\\subsection\{Research Hypotheses\}.*?(?=\n% ===+)', text, re.DOTALL)
if hyp_match:
    hyp_block = hyp_match.group(0)
    text = text.replace(hyp_block, "")
    hyp_block_new = hyp_block.replace(r'\subsection{Research Hypotheses}', r'\subsection{Hypothesis Development}')
    
    # Insert at the end of Literature review (just before the III section banner)
    text = re.sub(r'(?=\n% ============================================================\n% III\. CONCEPTUAL FRAMEWORK)', 
                  '\n' + hyp_block_new.replace('\\', '\\\\'), text, count=1)

# 4. Rename III. Conceptual Framework to III. Data and Methodology
text = text.replace(r'% III. CONCEPTUAL FRAMEWORK', r'% III. DATA AND METHODOLOGY')
text = text.replace(r'\section{Conceptual Framework}', r'\section{Data and Methodology}')

# Its first two subsections become 3.1
text = text.replace(r'\subsection{Multi-View Design Principle}', r'\subsection{Conceptual Framework}' + '\n\n' + r'\subsubsection{Multi-View Design Principle}')
text = text.replace(r'\subsection{Research Pipeline}', r'\subsubsection{Research Pipeline}')

# Extract Data Description (Benchmark Datasets) and move it to 3.2
data_match = re.search(r'\\subsection\{Benchmark Datasets\}.*?(?=\n\\subsection\{Feature Engineering:)', text, re.DOTALL)
if data_match:
    data_block = data_match.group(0)
    text = text.replace(data_block, "")
    data_block_new = data_block.replace(r'\subsection{Benchmark Datasets}', r'\subsection{Data Description}')
    
    # Insert right before % IV. DATA AND METHODOLOGY
    text = re.sub(r'(?=\n% ============================================================\n% IV\. DATA AND METHODOLOGY)', 
                  '\n' + data_block_new.replace('\\', '\\\\'), text, count=1)

# Modify section block IV title to DATA PREPROCESSING
text = text.replace(r'% IV. DATA AND METHODOLOGY', r'% IV. DATA PREPROCESSING')
# The section command was \section{Data and Methodology} inside section IV.
# Be careful: we now have two `\section{Data and Methodology}` strings in `text`!
# Let's replace ONLY the one after `% IV. DATA PREPROCESSING`
text = re.sub(r'(% IV\. DATA PREPROCESSING\n% ============================================================\n)\\section\{Data and Methodology\}', 
              r'\1\\section{Data Preprocessing}', text, count=1)

# 6. Make Feature Engineering and Splitting subsubsections of 4.1 Data Cleaning & Variable Construction
text = text.replace(r'\subsection{Feature Engineering: Three-View Representation}', r'\subsection{Data Cleaning \& Variable Construction}' + '\n\n' + r'\subsubsection{Feature Engineering: Three-View Representation}')
text = text.replace(r'\subsection{Data Splitting and Resampling}', r'\subsubsection{Data Splitting and Resampling}')

# 7. Move Research Methods (Stacking, XAI, Evaluation) from IV to III as subsection 3.3
# These start at \subsection{Multi-View Stacking Architecture} and end at % V. EXPECTED RESULTS
methods_match = re.search(r'\\subsection\{Multi-View Stacking Architecture\}.*?(?=\n% ============================================================\n% V\. EXPECTED RESULTS)', text, re.DOTALL)
if methods_match:
    methods_block = methods_match.group(0)
    text = text.replace(methods_block, "")
    
    # Change their level
    methods_block_new = r'\subsection{Research Methods}' + '\n\n' + methods_block
    methods_block_new = methods_block_new.replace(r'\subsection{Multi-View Stacking Architecture}', r'\subsubsection{Multi-View Stacking Architecture}')
    methods_block_new = methods_block_new.replace(r'\subsection{Four-Level XAI Framework}', r'\subsubsection{Four-Level XAI Framework}')
    methods_block_new = methods_block_new.replace(r'\subsection{Evaluation Strategy}', r'\subsubsection{Evaluation Strategy}')
    
    # Insert them at the very end of III (right before IV)
    text = re.sub(r'(?=\n% ============================================================\n% IV\. DATA PREPROCESSING)', 
                  '\n' + methods_block_new.replace('\\', '\\\\') + '\n', text, count=1)

with open('proposal_restructured.tex', 'w', encoding='utf-8') as f:
    f.write(text)

import re
outline = re.findall(r'\\(?:section|subsection|subsubsection)\{.*?\}', text)
print('\n'.join(outline))
