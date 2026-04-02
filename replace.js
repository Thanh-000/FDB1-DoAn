const fs = require('fs');

let content = fs.readFileSync('proposal.tex', 'utf8');

const figRegex = /(% --- Architecture Diagram ---\r?\n\\begin\{figure\}[\s\S]*?\\end\{figure\}\r?\n)/;
const figMatch = content.match(figRegex);

if (!figMatch) {
  console.log('Figure not found');
  process.exit(1);
}
const figBlock = figMatch[0];

const algRegex = /(\\begin\{algorithm\}\[htbp\]\r?\n\\caption\{Temporal MVS-XAI Execution Loop\}[\s\S]*?\\end\{algorithm\}\r?\n)/;
const algMatch = content.match(algRegex);
if (!algMatch) {
  console.log('Algorithm not found');
  process.exit(1);
}
const algBlock = algMatch[0];

const introText = 'The complete system architecture is illustrated in Figure~\\ref{fig:arch}, while the computational flow of the temporal multi-view training and XAI inference loop is formally detailed in Algorithm~\\ref{alg:mvs}.';

const newAlgs = `The complete system architecture is illustrated in Figure~\\ref{fig:arch}.

\\subsection{Algorithmic Formalization}
The computational flow of the temporal multi-view training and the subsequent XAI-driven inference are formally detailed in Algorithm~\\ref{alg:train} and Algorithm~\\ref{alg:infer}, respectively.

\\begin{algorithm}[htbp]
\\caption{Temporal Walk-Forward Multi-View Training}
\\label{alg:train}
\\algrenewcommand\\algorithmicrequire{\\textbf{Input:}}
\\algrenewcommand\\algorithmicensure{\\textbf{Output:}}
\\footnotesize
\\begin{algorithmic}[1]
\\Require Data $\\mathcal{D}$ sorted by time $t$, $K=5$ Temporal Folds
\\Ensure Meta-Learner $\\mathcal{M}_{meta}$
\\State Init $OOF$ matrix of size $N \\times 5$
\\For{$k = 1$ \\textbf{to} $K$}
    \\State \\textit{// Walk-Forward temporal split}
    \\State Extract folds: $\\mathcal{D}_{train}^{(k)}$ (hist), $\\mathcal{D}_{val}^{(k)}$ (future)
    \\State Extract $X_{tab}, X_{seq}, \\mathcal{G}_{ego}$ for train/val splits
    
    \\State \\textit{// Pipeline 1: Tabular Branch (Strict Isolation)}
    \\State $X_{tab\\_new}^{train} \\gets \\text{K-Means-SMOTE}(X_{tab}^{train})$
    \\State Train $\\mathcal{M}_{RF}^{(k)}, \\mathcal{M}_{XGB}^{(k)}, \\mathcal{M}_{LGBM}^{(k)}$ on $X_{tab\\_new}^{train}$
    
    \\State \\textit{// Pipeline 2: Sequential \\& Graph Branches}
    \\State Train $\\mathcal{M}_{LSTM}^{(k)}$ on $X_{seq}^{train}$ with $\\mathcal{L}_{FL}$ ($\\gamma=2$)
    \\State Train $\\mathcal{M}_{GAT}^{(k)}$ on $\\mathcal{G}_{ego}^{train}$ with $\\mathcal{L}_{FL}$ ($\\gamma=2$)
    
    \\State \\textit{// OOF Generation}
    \\State $\\mathbf{p}_{val} \\gets \\text{Predict 5 models on } \\mathcal{D}_{val}^{(k)}$
    \\State $OOF [\\text{idx}(\\mathcal{D}_{val}^{(k)})] \\gets \\mathbf{p}_{val}$
\\EndFor
\\State \\textbf{Train} $\\mathcal{M}_{meta}$ via Logistic Reg. on $OOF$
\\end{algorithmic}
\\end{algorithm}

\\begin{algorithm}[htbp]
\\caption{Dual-Stream XAI Inference and HITL Escalation}
\\label{alg:infer}
\\algrenewcommand\\algorithmicrequire{\\textbf{Input:}}
\\algrenewcommand\\algorithmicensure{\\textbf{Output:}}
\\footnotesize
\\begin{algorithmic}[1]
\\Require New transaction $Txn_i$, $\\mathcal{M}_{meta}$, Base Models
\\Ensure Fraud Decision $\\mathcal{D}_{final}$, Real-time XAI $\\mathcal{E}_{rt}$, Audit XAI $\\mathcal{E}_{adt}$
\\State $\\hat{P}_{fraud} \\gets \\mathcal{M}_{meta}(\\mathbf{p}_{base}(Txn_i))$
\\If{$\\hat{P}_{fraud} \\in [0.5, 0.7]$ \\textbf{or} $Txn\\_amt > P_{95}$}
    \\State \\Return \\textsc{Queue\\_HITL}($Txn_i$, \\textsc{Shap}($Txn_i$)) \\Comment{Human oversight}
\\EndIf
\\State \\textit{// Real-Time Stream ($<$50ms)}
\\State $\\mathcal{E}_{rt} \\gets \\{ \\textsc{Shap}(Txn_i), \\textsc{Anchors}(Txn_i) \\}$
\\State \\textit{// Audit Stream (Async Batch)}
\\State $\\mathcal{E}_{adt} \\gets \\textsc{QueueAsync}(\\textsc{DiCE}(Txn_i), \\text{PSI Shift})$
\\State \\textbf{return} \\textsc{Threshold}(\\hat{P}_{fraud}), $\\mathcal{E}_{rt}, \\mathcal{E}_{adt}$
\\end{algorithmic}
\\end{algorithm}
`;

content = content.replace(figBlock, '');
content = content.replace(algBlock, '');
content = content.replace(introText, newAlgs);

const targetInsert = '\\end{enumerate}\r?\n\r?\n\\\\subsection\\{Data Description\\}';
const regexInsert = new RegExp(targetInsert);
if (!content.match(regexInsert)) {
  console.log('Insert point not found');
}
content = content.replace(regexInsert, '\\end{enumerate}\n\n' + figBlock + '\n\\subsection{Data Description}');

fs.writeFileSync('proposal.tex', content, 'utf8');
console.log('Success!');
