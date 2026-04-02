import re

with open('proposal.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Remove Implementation Plan
# From '\subsection{Implementation Plan}' until '\section{Conclusion}'
text = re.sub(r'\\subsection\{Implementation Plan\}.*?(?=\\section\{Conclusion\})', '\n', text, flags=re.DOTALL)

# 2. Remove Multi-View Advantage for Hard Cases
# From '\subsection{Multi-View Advantage for Hard Cases}' until '\subsection{XAI'
text = re.sub(r'\\subsection\{Multi-View Advantage for Hard Cases\}.*?(?=\\subsection\{XAI Effectiveness\})', '\n', text, flags=re.DOTALL)

# 3. Simplify Expected Results (remove the huge table, just summarize)
# Actually, the Model Performance table is good for "Expected Results". 
# But let's shorten the "Ablation Study Design" list.
ablation_long = r"""\begin{enumerate}
    \item \textbf{Baseline:} RF and XGBoost (Tabular only) vs.\ Logistic Regression.
    \item \textbf{View Addition:} Tabular + Sequential vs.\ Tabular + Graph.
    \item \textbf{Loss Impact:} Focal Loss vs.\ Cross-Entropy in LSTM/GAT.
    \item \textbf{Resampling Impact:} K-Means SMOTE vs.\ Standard SMOTE vs.\ No Resampling.
    \item \textbf{Stacking Method:} Soft Voting vs.\ Logistic Regression vs.\ XGBoost Meta.
\end{enumerate}"""
ablation_short = r"The study will execute a 19-experiment matrix isolating the marginal utility of each component (views, Focal Loss, K-Means SMOTE, and stacking methods) to validate the architectural choices scientifically."
text = text.replace(ablation_long, ablation_short)

with open('proposal.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print("Simplified proposal.tex")
