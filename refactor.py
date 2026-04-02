import re

with open('proposal.tex', 'r', encoding='utf-8') as f:
    tex = f.read()

# I. INTRODUCTION
tex = tex.replace(r'\subsection{Research Objectives and Questions}', r'\subsection{Research Objectives and Research Questions}')
tex = tex.replace(r'\subsection{Key Contributions}', r'\subsection{Research Motivation and Contribution}')

# Move Hypotheses from III.3 to II.2
hyp_match = re.search(r'\\subsection\{Research Hypotheses\}.*?(?=\n%.*?IV\.)', tex, re.DOTALL)
if hyp_match:
    hyp_text = hyp_match.group(0)
    tex = tex.replace(hyp_text, "")
    
    # Insert at the end of II. Literature Review
    tex = tex.replace(r'\section{Literature Review}', r'\section{Literature Review and Hypothesis Development}' + '\n\n\\subsection{Literature Review}\nThis subsection reviews tree-based, temporal, and graph-based approaches in fraud detection.\n')
    
    # We will just append the Hypothesis Development at the end of Literature Review
    lit_end_match = re.search(r'\\subsection\{Research Gap Summary\}.*?(?=\n%.*?III\.)', tex, re.DOTALL)
    if lit_end_match:
        lit_end_text = lit_end_match.group(0)
        new_hyp_text = hyp_text.replace(r'\subsection{Research Hypotheses}', r'\subsection{Hypothesis Development}')
        tex = tex.replace(lit_end_text, lit_end_text + "\n" + new_hyp_text)

# Rename III. Conceptual Framework to III. Data and Methodology
tex = tex.replace(r'\section{Conceptual Framework}', r'\section{Data and Methodology}')
tex = tex.replace(r'\subsection{Multi-View Design Principle}', r'\subsection{Conceptual Framework}' + '\n' + r'\subsubsection{Multi-View Design Principle}')
tex = tex.replace(r'\subsection{Research Pipeline}', r'\subsubsection{Research Pipeline}')

# Move Data Description (4.1 Benchmark Datasets) to 3.2 Data Description
data_desc_match = re.search(r'\\subsection\{Benchmark Datasets\}.*?(?=\\subsection\{Feature Engineering)', tex, re.DOTALL)
if data_desc_match:
    data_desc_text = data_desc_match.group(0)
    tex = tex.replace(data_desc_text, "")
    
    # Find end of Conceptual framework (which is before new Hypothesis Development used to be, now it's just before Research Pipeline ends)
    # Actually, we can insert it right before the old IV. DATA AND METHODOLOGY
    new_data_desc = data_desc_text.replace(r'\subsection{Benchmark Datasets}', r'\subsection{Data Description}')
    
    # The old IV section title
    tex = tex.replace(r'\section{Data and Methodology}', r'%%%%%%%% ' + '\n') # Temporarily hide the 2nd one
    # Note: earlier we renamed section Conceptual Framework to Data and Methodology.
    # The original document had \section{Data and Methodology} at Line 289.
    
with open('proposal.tex', 'w', encoding='utf-8') as f:
    f.write(tex)
