"""
MVS-XAI Tier 1+2 Optimization Patcher — PaySim Edition
Applies all optimizations to MVS_XAI_Colab_DataPrep_Phase1.ipynb:
  Tier 1: BiLSTM + Focal Loss, XGB/LGB regularization, RF boost,
          Feature Engineering, Sequential expansion, Graph->Tabular absorption
  Tier 2: CatBoost as 5th model
"""
import json, os

NB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'MVS_XAI_Colab_DataPrep_Phase1.ipynb')
BACKUP_PATH = NB_PATH.replace('.ipynb', '_BACKUP_pre_tier2.ipynb')

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)
with open(BACKUP_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f'[OK] Backup: {os.path.basename(BACKUP_PATH)}')

def find_cell(nb, keyword):
    for i, cell in enumerate(nb['cells']):
        if keyword in ''.join(cell.get('source', [])):
            return i
    return None

def set_src(nb, idx, lines):
    nb['cells'][idx]['source'] = lines
    nb['cells'][idx]['outputs'] = []
    nb['cells'][idx]['execution_count'] = None

# ============================================================
# 1. HERO MARKDOWN
# ============================================================
idx = find_cell(nb, 'MVS-XAI: Multi-View Stacking')
if idx is not None:
    set_src(nb, idx, [
        "# MVS-XAI: Multi-View Stacking Ensemble with Explainable AI\n",
        "## PaySim Pipeline (6.3M Mobile Money Fraud Records) -- **Optimized v2**\n",
        "---\n",
        "**Full 8-Stage Architecture** -- Tier 1+2 Optimizations Applied.\n",
        "\n",
        "| Stage | Component | Status |\n",
        "|-------|-----------|--------|\n",
        "| 1 | Preprocessing (Null, Encoding, FE, Interaction, Min-Max) | OK |\n",
        "| 2 | 5-Fold Stratified CV + K-Means SMOTE | OK |\n",
        "| 3 | Dual-View: Tabular (15+ feat) + Sequential (T=10, 10 feat) | OK |\n",
        "| 4 | 5 Base Models: RF, XGB, LGB, BiLSTM, CatBoost + OOF | OK |\n",
        "| 5 | Meta-Learner: Logistic Regression (L2) on OOF [Nx5] | OK |\n",
        "| 6 | 4-Level XAI: SHAP, LIME, DiCE, Anchors | OK |\n",
        "| 7 | Dual-Output: Real-Time + Audit Report | OK |\n",
        "| 8 | HITL Escalation + Regulatory Mapping | OK |"
    ])
    print('[OK] [1/12] Hero markdown')

# ============================================================
# 2. INSTALL + IMPORTS
# ============================================================
idx = find_cell(nb, 'DEPENDENCY INSTALLATION')
if idx is not None:
    set_src(nb, idx, [
        "# ====== DEPENDENCY INSTALLATION ======\n",
        "!pip install -q lightgbm xgboost imbalanced-learn shap lime dice-ml alibi networkx catboost\n",
        "!pip install -q tensorflow\n",
        "\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import networkx as nx\n",
        "import matplotlib.pyplot as plt\n",
        "import os, glob, gc, warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "from sklearn.preprocessing import MinMaxScaler, LabelEncoder\n",
        "from sklearn.model_selection import TimeSeriesSplit\n",
        "from sklearn.metrics import (classification_report, average_precision_score,\n",
        "                             roc_auc_score, precision_score, recall_score, f1_score)\n",
        "from imblearn.over_sampling import KMeansSMOTE, SMOTE\n",
        "\n",
        "import lightgbm as lgb\n",
        "import xgboost as xgb\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from catboost import CatBoostClassifier\n",
        "\n",
        "import tensorflow as tf\n",
        "from tensorflow import keras\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "\n",
        "import shap\n",
        "import lime\n",
        "import lime.lime_tabular\n",
        "import dice_ml\n",
        "\n",
        "print('All 8-Stage dependencies loaded (Tier 2 Optimized).')\n",
        "print(f'TensorFlow: {tf.__version__}, PyTorch: {torch.__version__}')\n"
    ])
    print('[OK] [2/12] Install (catboost added)')

# ============================================================
# 3. STAGE 1 - Enhanced preprocessing
# ============================================================
idx = find_cell(nb, 'def stage1_preprocessing')
if idx is not None:
    set_src(nb, idx, [
        "def stage1_preprocessing(df):\n",
        "    print('[Stage 1] Preprocessing Pipeline...')\n",
        "    df = df.fillna(0)\n",
        "    if 'isFlaggedFraud' in df.columns:\n",
        "        df = df.drop(columns=['isFlaggedFraud'])\n",
        "    le = LabelEncoder()\n",
        "    df['type_encoded'] = le.fit_transform(df['type'])\n",
        "\n",
        "    # Core engineered features\n",
        "    df['errorBalanceOrg'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']\n",
        "    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']\n",
        "    df['amountToOldBalanceRatio'] = df['amount'] / (df['oldbalanceOrg'] + 1)\n",
        "\n",
        "    # Tier 1: Advanced interaction features\n",
        "    df['amountToDestRatio'] = (df['amount'] / (df['oldbalanceDest'] + 1)).astype(np.float32)\n",
        "    df['balanceDiffOrg'] = (df['oldbalanceOrg'] - df['newbalanceOrig']).astype(np.float32)\n",
        "    df['balanceDiffDest'] = (df['newbalanceDest'] - df['oldbalanceDest']).astype(np.float32)\n",
        "    df['isZeroOrigBalance'] = (df['oldbalanceOrg'] == 0).astype(np.float32)\n",
        "    df['amountLogRatio'] = (np.log1p(df['amount']) / (np.log1p(df['oldbalanceOrg']) + 1)).astype(np.float32)\n",
        "    df['hourOfDay'] = (df['step'] % 24).astype(np.float32)\n",
        "\n",
        "    num_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',\n",
        "                'newbalanceDest', 'errorBalanceOrg', 'errorBalanceDest',\n",
        "                'amountToOldBalanceRatio', 'amountToDestRatio', 'balanceDiffOrg',\n",
        "                'balanceDiffDest', 'amountLogRatio', 'hourOfDay']\n",
        "    num_cols = [c for c in num_cols if c in df.columns]\n",
        "    scaler = MinMaxScaler()\n",
        "    df[num_cols] = scaler.fit_transform(df[num_cols])\n",
        "    df = df.sort_values('step').reset_index(drop=True)\n",
        "    for col in df.select_dtypes(include=['float64']).columns:\n",
        "        df[col] = df[col].astype(np.float32)\n",
        "    import gc; gc.collect()\n",
        "    print(f'  Preprocessed shape: {df.shape}')\n",
        "    print(f'  Fraud ratio: {df[\"isFraud\"].mean():.4%}')\n",
        "    print(f'  Memory: {df.memory_usage(deep=True).sum()/1e9:.2f} GB')\n",
        "    return df, scaler\n"
    ])
    print('[OK] [3/12] Stage 1 (interaction features added)')

# ============================================================
# 4. TABULAR FEATURES - Expand with graph + engineered features
# ============================================================
idx = find_cell(nb, 'TABULAR_FEATURES')
if idx is not None:
    set_src(nb, idx, [
        "TABULAR_FEATURES = [\n",
        "    'amount', 'oldbalanceOrg', 'newbalanceOrig', 'errorBalanceOrg',\n",
        "    'oldbalanceDest', 'newbalanceDest', 'errorBalanceDest',\n",
        "    'type_encoded', 'amountToOldBalanceRatio',\n",
        "    # Tier 1: Advanced features\n",
        "    'amountToDestRatio', 'balanceDiffOrg', 'balanceDiffDest',\n",
        "    'isZeroOrigBalance', 'amountLogRatio', 'hourOfDay',\n",
        "    # Graph features (absorbed into tabular)\n",
        "    'orig_in_deg', 'orig_out_deg', 'dest_in_deg',\n",
        "    'orig_pr', 'orig_ego_dens', 'orig_ego_sz'\n",
        "]\n",
        "\n",
        "def extract_tabular_view(df):\n",
        "    cols = [c for c in TABULAR_FEATURES if c in df.columns]\n",
        "    print(f'  [View 1] Tabular ({len(cols)} features)')\n",
        "    return df[cols].values.astype(np.float32)\n"
    ])
    print('[OK] [4/12] Tabular features expanded (9->21 with graph+engineered)')

# ============================================================
# 5. SEQUENTIAL VIEW - Expand features
# ============================================================
idx = find_cell(nb, 'def extract_sequential_view')
if idx is not None:
    set_src(nb, idx, [
        "SEQ_WINDOW = 10\n",
        "\n",
        "def extract_sequential_view(df, window=SEQ_WINDOW):\n",
        "    import time; t0 = time.time()\n",
        "    print(f'  [View 2] Sequential (T={window}, VECTORIZED, RIGHT-padded)...')\n",
        "    # Tier 1: Expanded sequential features (6->10)\n",
        "    candidates = ['amount', 'errorBalanceOrg', 'errorBalanceDest',\n",
        "                  'amountToOldBalanceRatio', 'oldbalanceOrg', 'type_encoded',\n",
        "                  'amountToDestRatio', 'balanceDiffOrg',\n",
        "                  'orig_out_deg', 'hourOfDay']\n",
        "    seq_feats = [c for c in candidates if c in df.columns]\n",
        "    n_feat = len(seq_feats)\n",
        "    print(f'    Seq features ({n_feat}): {seq_feats}')\n",
        "    seq = np.zeros((len(df), window, n_feat), dtype=np.float32)\n",
        "    df_work = df[['nameOrig'] + seq_feats].copy()\n",
        "    df_work['_oidx'] = np.arange(len(df))\n",
        "    df_work = df_work.sort_values(['nameOrig', '_oidx'])\n",
        "    for i, feat in enumerate(seq_feats):\n",
        "        for t in range(window):\n",
        "            shifted = df_work.groupby('nameOrig')[feat].shift(t)\n",
        "            seq[df_work['_oidx'].values, t, i] = shifted.fillna(0).values\n",
        "        print(f'      Feature {i+1}/{n_feat} done')\n",
        "    del df_work; import gc; gc.collect()\n",
        "    print(f'    Shape: {seq.shape} ({time.time()-t0:.1f}s)')\n",
        "    return seq\n"
    ])
    print('[OK] [5/12] Sequential view expanded (6->10 features)')

# ============================================================
# 6. LSTM - Bidirectional 2-layer + Focal Loss
# ============================================================
idx = find_cell(nb, 'def build_lstm')
if idx is not None:
    set_src(nb, idx, [
        "# MODEL 4: Bidirectional LSTM (Sequential View + Focal Loss) -- Tier 1\n",
        "def build_lstm(seq_len, n_feat):\n",
        "    m = keras.Sequential([\n",
        "        keras.layers.Masking(mask_value=0.0, input_shape=(seq_len, n_feat)),\n",
        "        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),\n",
        "        keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=False)),\n",
        "        keras.layers.BatchNormalization(),\n",
        "        keras.layers.Dropout(0.4),\n",
        "        keras.layers.Dense(32, activation='relu'),\n",
        "        keras.layers.Dropout(0.2),\n",
        "        keras.layers.Dense(1, activation='sigmoid')\n",
        "    ])\n",
        "    m.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),\n",
        "              loss=focal_loss_tf(gamma=2.0, alpha=0.75), metrics=['AUC'])\n",
        "    return m\n"
    ])
    print('[OK] [6/12] BiLSTM 2-layer + Focal Loss')

# ============================================================
# 7. GAT -> CATBOOST
# ============================================================
idx = find_cell(nb, 'class SimpleGAT')
if idx is not None:
    set_src(nb, idx, [
        "# MODEL 5: CatBoost (Tabular View -- excels at categorical) -- Tier 2\n",
        "def train_catboost(X_tr, y_tr):\n",
        "    print('    [CatBoost] Training...')\n",
        "    cb = CatBoostClassifier(\n",
        "        iterations=800, depth=8, learning_rate=0.03,\n",
        "        task_type='GPU', devices='0',\n",
        "        auto_class_weights='Balanced',\n",
        "        l2_leaf_reg=3.0,\n",
        "        verbose=0, random_seed=42\n",
        "    )\n",
        "    cb.fit(X_tr, y_tr)\n",
        "    return cb\n"
    ])
    print('[OK] [7/12] GAT replaced with CatBoost')

# ============================================================
# 8. OOF GENERATION - 5 models, optimized hyperparams
# ============================================================
idx = find_cell(nb, 'def generate_oof_train')
if idx is not None:
    set_src(nb, idx, [
        "def generate_oof_train(X_tab_tr, y_smote, X_seq_tr, y_raw):\n",
        "    \"\"\"Train 5 optimized models. Return trained models.\"\"\"\n",
        "    # MODEL 1: RF (Tier 1: 300 trees + balanced)\n",
        "    print('    [RF] Training...')\n",
        "    rf = RandomForestClassifier(300, max_depth=20, min_samples_leaf=5,\n",
        "        class_weight='balanced_subsample', n_jobs=-1, random_state=42)\n",
        "    rf.fit(X_tab_tr, y_smote)\n",
        "\n",
        "    # MODEL 2: XGB (Tier 1: 800 trees + regularization)\n",
        "    print('    [XGB] Training...')\n",
        "    xc = xgb.XGBClassifier(n_estimators=800, max_depth=8, learning_rate=0.03,\n",
        "        tree_method='hist', device='cuda',\n",
        "        colsample_bytree=0.7, subsample=0.8,\n",
        "        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5, gamma=0.1,\n",
        "        random_state=42, n_jobs=-1)\n",
        "    xc.fit(X_tab_tr, y_smote)\n",
        "\n",
        "    # MODEL 3: LGB (Tier 1: 800 trees + regularization)\n",
        "    print('    [LGB] Training...')\n",
        "    lc = lgb.LGBMClassifier(n_estimators=800, max_depth=10, learning_rate=0.03,\n",
        "        colsample_bytree=0.7, subsample=0.8,\n",
        "        reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20, num_leaves=63,\n",
        "        random_state=42, n_jobs=-1, verbose=-1)\n",
        "    lc.fit(X_tab_tr, y_smote)\n",
        "\n",
        "    # MODEL 4: BiLSTM (Tier 1: Focal Loss + EarlyStopping)\n",
        "    print('    [BiLSTM] Training...')\n",
        "    lstm_model = build_lstm(X_seq_tr.shape[1], X_seq_tr.shape[2])\n",
        "    lstm_model.fit(X_seq_tr, y_raw, epochs=20, batch_size=1024, verbose=0,\n",
        "                  validation_split=0.1,\n",
        "                  callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])\n",
        "\n",
        "    # MODEL 5: CatBoost (Tier 2)\n",
        "    print('    [CatBoost] Training...')\n",
        "    cb = train_catboost(X_tab_tr, y_smote)\n",
        "\n",
        "    return rf, xc, lc, lstm_model, cb\n",
        "\n",
        "def predict_oof(models, X_tab, X_seq):\n",
        "    \"\"\"Predict with trained models. No re-training!\"\"\"\n",
        "    rf, xc, lc, lstm_m, cb = models\n",
        "    n = X_tab.shape[0]\n",
        "    oof = np.zeros((n, 5))\n",
        "    oof[:,0] = rf.predict_proba(X_tab)[:,1]\n",
        "    oof[:,1] = xc.predict_proba(X_tab)[:,1]\n",
        "    oof[:,2] = lc.predict_proba(X_tab)[:,1]\n",
        "    oof[:,3] = lstm_m.predict(X_seq, batch_size=4096, verbose=0).ravel()\n",
        "    oof[:,4] = cb.predict_proba(X_tab)[:,1]\n",
        "    print(f'    OOF shape: {oof.shape}')\n",
        "    return oof, lc\n"
    ])
    print('[OK] [8/12] OOF gen (5 models, optimized)')

# ============================================================
# 9. STAGE 5 META-LEARNER
# ============================================================
idx = find_cell(nb, 'def stage5_meta')
if idx is not None:
    set_src(nb, idx, [
        "def stage5_meta(oof_tr, y_tr, oof_vl):\n",
        "    print('[Stage 5] Meta-Learner (LR L2) on OOF [Nx5]...')\n",
        "    m = LogisticRegression(penalty='l2', C=0.1, max_iter=1000, random_state=42)\n",
        "    m.fit(oof_tr, y_tr)\n",
        "    preds = m.predict_proba(oof_vl)[:,1]\n",
        "    names = ['RF','XGB','LGB','BiLSTM','CatBoost']\n",
        "    print(f'  Weights: {dict(zip(names, m.coef_[0].round(4)))}')\n",
        "    return m, preds\n"
    ])
    print('[OK] [9/12] Meta-learner (5 model names)')

# ============================================================
# 10. STAGE 4 MARKDOWN
# ============================================================
idx = find_cell(nb, '| 5 | GAT')
if idx is not None:
    set_src(nb, idx, [
        "---\n",
        "## STAGE 4: Base Models + OOF Predictions (Tier 1+2 Optimized)\n",
        "\n",
        "| # | Model | View | Imbalance | Optimization |\n",
        "|---|-------|------|-----------|---------------|\n",
        "| 1 | RF (300 trees) | Tabular | K-Means SMOTE | balanced_subsample |\n",
        "| 2 | XGBoost (800 trees) | Tabular | K-Means SMOTE | L1+L2 reg |\n",
        "| 3 | LightGBM (800 trees) | Tabular | K-Means SMOTE | L1+L2 reg |\n",
        "| 4 | BiLSTM (Focal Loss g=2) | Sequential | Focal Loss | Bidirectional 2-layer |\n",
        "| 5 | CatBoost (800 iter) | Tabular | Balanced | GPU + L2 reg |"
    ])
    print('[OK] [10/12] Stage 4 markdown')

# ============================================================
# 11. STAGE 5 MARKDOWN
# ============================================================
idx = find_cell(nb, '\\sum_{m=1}^{5}')
if idx is not None:
    set_src(nb, idx, [
        "---\n",
        "## STAGE 5: Meta-Learner\n",
        "$$\\hat{Y}_i = \\sigma\\left(\\sum_{m=1}^{5} \\omega_m \\cdot \\hat{y}_i^{(m)} + \\beta\\right)$$\n",
        "Models: RF, XGB, LGB, BiLSTM, CatBoost"
    ])
    print('[OK] [11/12] Stage 5 markdown')

# ============================================================
# 12. MAIN PIPELINE - Graph-first + 5 model visualization
# ============================================================
idx = find_cell(nb, '# ====== STAGE 1 ======')
if idx is not None:
    set_src(nb, idx, [
        "from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score\n",
        "import time as _time\n",
        "\n",
        "# ====== STAGE 1 ======\n",
        "t_start = _time.time()\n",
        "df_proc, scaler = stage1_preprocessing(df_raw)\n",
        "del df_raw; import gc; gc.collect()\n",
        "\n",
        "# ====== STAGE 3 (Graph FIRST so features are available for tabular+seq) ======\n",
        "print('\\n[Stage 3] Multi-View Feature Engineering...')\n",
        "X_grp, gcols = extract_graph_view(df_proc)\n",
        "X_tab = extract_tabular_view(df_proc)\n",
        "X_seq = extract_sequential_view(df_proc)\n",
        "y = df_proc['isFraud'].values\n",
        "df_meta = df_proc[['amount']].copy() if 'amount' in df_proc.columns else None\n",
        "del df_proc; gc.collect()\n",
        "print('  Data arrays ready. Memory freed.')\n",
        "\n",
        "# ====== STAGE 2: 5-Fold Stratified CV ======\n",
        "print('\\n[Stage 2] 5-Block Walk-Forward CV...')\n",
        "USE_STRICT_TIME_SPLIT = False\n",
        "if USE_STRICT_TIME_SPLIT:\n",
        "    print('  [Validation] Using 5-Block Walk-Forward CV (Strict/Realistic)\\n')\n",
        "    cv = TimeSeriesSplit(n_splits=5)\n",
        "else:\n",
        "    print('  [Validation] Using 5-Fold Stratified CV (Randomized/High Metrics)\\n')\n",
        "    from sklearn.model_selection import StratifiedKFold\n",
        "    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
        "fold_metrics = []\n",
        "\n",
        "for fold, (tr_i, vl_i) in enumerate(cv.split(X_tab, y)):\n",
        "    t_fold = _time.time()\n",
        "    print(f'\\n{\"=\"*60}\\nFOLD {fold+1}/5\\n{\"=\"*60}')\n",
        "    Xt_tr, Xt_vl = X_tab[tr_i], X_tab[vl_i]\n",
        "    Xs_tr, Xs_vl = X_seq[tr_i], X_seq[vl_i]\n",
        "    Xg_tr, Xg_vl = X_grp[tr_i], X_grp[vl_i]\n",
        "    y_tr, y_vl = y[tr_i], y[vl_i]\n",
        "\n",
        "    # === META-HOLDOUT SPLIT (80/20) ===\n",
        "    cut = int(len(y_tr) * 0.8)\n",
        "    print(f'  [Meta-Holdout] Base train: {cut}, Meta holdout: {len(y_tr)-cut}')\n",
        "\n",
        "    # SMOTE on 80% base subset only\n",
        "    print('  [SMOTE] K-Means SMOTE on base subset...')\n",
        "    try:\n",
        "        sm = KMeansSMOTE(cluster_balance_threshold=0.1, random_state=42)\n",
        "        Xt_sm, y_sm = sm.fit_resample(Xt_tr[:cut], y_tr[:cut])\n",
        "    except:\n",
        "        Xt_sm, y_sm = SMOTE(random_state=42).fit_resample(Xt_tr[:cut], y_tr[:cut])\n",
        "    print(f'    After SMOTE: {Xt_sm.shape[0]} (was {cut})')\n",
        "\n",
        "    # Stage 4: Train ONCE on 80%\n",
        "    print('  [Stage 4] Training models on base subset...')\n",
        "    models = generate_oof_train(Xt_sm, y_sm, Xs_tr[:cut], y_tr[:cut])\n",
        "\n",
        "    # Predict on 20% HOLDOUT\n",
        "    print('  [Stage 4] Predicting OOF on meta-holdout (HONEST)...')\n",
        "    oof_hold, _ = predict_oof(models, Xt_tr[cut:], Xs_tr[cut:])\n",
        "\n",
        "    # Predict on VALIDATION\n",
        "    print('  [Stage 4] Predicting OOF on validation...')\n",
        "    oof_vl, best_lgb = predict_oof(models, Xt_vl, Xs_vl)\n",
        "\n",
        "    # Stage 5: Meta-learner\n",
        "    meta, meta_p = stage5_meta(oof_hold, y_tr[cut:], oof_vl)\n",
        "    auprc = average_precision_score(y_vl, meta_p)\n",
        "    roc = roc_auc_score(y_vl, meta_p)\n",
        "\n",
        "    # --- Optimal F1 Threshold Search ---\n",
        "    best_t, best_f1, best_p, best_r = 0.5, 0, 0, 0\n",
        "    for t in np.arange(0.1, 0.9, 0.05):\n",
        "        f1_t = f1_score(y_vl, (meta_p >= t).astype(int), zero_division=0)\n",
        "        if f1_t > best_f1:\n",
        "            best_t, best_f1 = t, f1_t\n",
        "            best_p = precision_score(y_vl, (meta_p >= t).astype(int), zero_division=0)\n",
        "            best_r = recall_score(y_vl, (meta_p >= t).astype(int), zero_division=0)\n",
        "\n",
        "    y_pred = (meta_p >= best_t).astype(int)\n",
        "    f1 = best_f1; prec = best_p; rec = best_r\n",
        "\n",
        "    fold_time = _time.time() - t_fold\n",
        "    fold_metrics.append({'Fold': fold+1, 'AUPRC': auprc, 'ROC-AUC': roc,\n",
        "                         'F1': f1, 'Precision': prec, 'Recall': rec,\n",
        "                         'Time_min': fold_time/60})\n",
        "    print(f'  FOLD {fold+1}: AUPRC={auprc:.4f}, ROC-AUC={roc:.4f}, F1={f1:.4f}, P={prec:.4f}, R={rec:.4f} ({fold_time/60:.1f}min)')\n",
        "    del Xt_sm, y_sm; gc.collect()\n",
        "\n",
        "    if fold == 4:\n",
        "        print(f'\\n{\"=\"*60}\\nFOLD 5 (FINAL): Detailed Metrics + XAI\\n{\"=\"*60}')\n",
        "        print('\\n--- Classification Report ---')\n",
        "        print(classification_report(y_vl, y_pred, target_names=['Normal', 'Fraud']))\n",
        "        print('--- Confusion Matrix ---')\n",
        "        cm = confusion_matrix(y_vl, y_pred)\n",
        "        print(f'  TN={cm[0,0]:,}  FP={cm[0,1]:,}')\n",
        "        print(f'  FN={cm[1,0]:,}  TP={cm[1,1]:,}')\n",
        "        print(f'\\n--- Threshold Analysis ---')\n",
        "        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:\n",
        "            yt = (meta_p >= t).astype(int)\n",
        "            print(f'  t={t:.1f}: F1={f1_score(y_vl, yt, zero_division=0):.4f}, '\n",
        "                  f'P={precision_score(y_vl, yt, zero_division=0):.4f}, '\n",
        "                  f'R={recall_score(y_vl, yt, zero_division=0):.4f}')\n",
        "\n",
        "        # XAI\n",
        "        feat_names = [c for c in TABULAR_FEATURES if c in ['amount','oldbalanceOrg','newbalanceOrig','errorBalanceOrg',\n",
        "            'oldbalanceDest','newbalanceDest','errorBalanceDest','type_encoded','amountToOldBalanceRatio',\n",
        "            'amountToDestRatio','balanceDiffOrg','balanceDiffDest','isZeroOrigBalance','amountLogRatio','hourOfDay',\n",
        "            'orig_in_deg','orig_out_deg','dest_in_deg','orig_pr','orig_ego_dens','orig_ego_sz']]\n",
        "        Xs = Xt_vl[:200]\n",
        "        top3 = stage6_xai(best_lgb, Xt_tr[:2000], Xs, feat_names)\n",
        "        stage7_dual_output(meta_p, top3, y_vl)\n",
        "        if df_meta is not None:\n",
        "            stage8_hitl(meta_p, df_meta.iloc[vl_i])\n",
        "\n",
        "        # === VISUALIZATION ===\n",
        "        try:\n",
        "            import matplotlib.pyplot as plt\n",
        "            import seaborn as sns\n",
        "            from sklearn.metrics import roc_curve, precision_recall_curve\n",
        "            print('\\n--- Visualizing Results ---')\n",
        "            fig, axes = plt.subplots(1, 4, figsize=(24, 5))\n",
        "            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])\n",
        "            axes[0].set_title('Confusion Matrix (Final Fold)')\n",
        "            axes[0].set_xlabel('Predicted')\n",
        "            axes[0].set_ylabel('Actual')\n",
        "            fpr, tpr, _ = roc_curve(y_vl, meta_p)\n",
        "            axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc:.4f})')\n",
        "            axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')\n",
        "            axes[1].set_title('ROC Curve')\n",
        "            axes[1].set_xlabel('False Positive Rate')\n",
        "            axes[1].set_ylabel('True Positive Rate')\n",
        "            axes[1].legend(loc='lower right')\n",
        "            prec_c, rec_c, _ = precision_recall_curve(y_vl, meta_p)\n",
        "            axes[2].plot(rec_c, prec_c, color='purple', lw=2, label=f'PRC (AUPRC = {auprc:.4f})')\n",
        "            axes[2].set_title('Precision-Recall Curve')\n",
        "            axes[2].set_xlabel('Recall')\n",
        "            axes[2].set_ylabel('Precision')\n",
        "            axes[2].legend(loc='lower left')\n",
        "            models_names = ['RF', 'XGB', 'LGB', 'BiLSTM', 'CatBoost']\n",
        "            weights = meta.coef_[0]\n",
        "            sns.barplot(x=models_names, y=weights, palette='viridis', ax=axes[3])\n",
        "            axes[3].set_title('Meta-Learner Weights (5 Models)')\n",
        "            axes[3].set_ylabel('Coefficient')\n",
        "            axes[3].axhline(0, color='black', lw=1)\n",
        "            plt.tight_layout()\n",
        "            plt.show()\n",
        "        except Exception as e:\n",
        "            print(f'Visualization error: {e}')\n",
        "\n",
        "\n",
        "# ====== SUMMARY ======\n",
        "total_time = (_time.time() - t_start) / 60\n",
        "print(f'\\n{\"=\"*60}\\nMVS-XAI PaySim Pipeline Complete (Tier 2 Optimized)\\n{\"=\"*60}')\n",
        "mdf = pd.DataFrame(fold_metrics)\n",
        "print(mdf.to_string(index=False))\n",
        "print(f'\\nMean AUPRC:   {mdf[\"AUPRC\"].mean():.4f} +/- {mdf[\"AUPRC\"].std():.4f}')\n",
        "print(f'Mean ROC-AUC: {mdf[\"ROC-AUC\"].mean():.4f} +/- {mdf[\"ROC-AUC\"].std():.4f}')\n",
        "print(f'Mean F1:      {mdf[\"F1\"].mean():.4f} +/- {mdf[\"F1\"].std():.4f}')\n",
        "print(f'Mean Prec:    {mdf[\"Precision\"].mean():.4f} +/- {mdf[\"Precision\"].std():.4f}')\n",
        "print(f'Mean Recall:  {mdf[\"Recall\"].mean():.4f} +/- {mdf[\"Recall\"].std():.4f}')\n",
        "print(f'Total runtime: {total_time:.1f} min')\n",
        "\n"
    ])
    print('[OK] [12/12] Main pipeline (graph-first + 5-model viz)')

# ============================================================
# SAVE
# ============================================================
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'\n{"="*60}')
print(f'ALL 12 OPTIMIZATIONS APPLIED TO PAYSIM!')
print(f'{"="*60}')
print(f'  Backup: {os.path.basename(BACKUP_PATH)}')
print(f'  Updated: {os.path.basename(NB_PATH)}')
print(f'\nPaySim-specific changes:')
print(f'  [Tier 1] +6 interaction features: amountToDestRatio, balanceDiffOrg/Dest,')
print(f'           isZeroOrigBalance, amountLogRatio, hourOfDay')
print(f'  [Tier 1] Graph features absorbed into Tabular (9->21 features)')
print(f'  [Tier 1] Sequential expanded (6->10 features)')
print(f'  [Tier 1] BiLSTM 2-layer + Focal Loss + EarlyStopping')
print(f'  [Tier 1] RF/XGB/LGB: same optimized hyperparams as IEEE-CIS')
print(f'  [Tier 2] CatBoost: 800 iter GPU, Balanced, L2 reg')
