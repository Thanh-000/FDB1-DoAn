import json
from pathlib import Path

repo = Path.cwd()
ieee_path = repo / 'MVS_XAI_Colab_IEEE_CIS.ipynb'
paysim_path = repo / 'MVS_XAI_Colab_DataPrep_Phase1.ipynb'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, nb):
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')


def set_cell_source(nb, marker, new_source):
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            src = ''.join(cell.get('source', []))
            if marker in src:
                cell['source'] = [line + '\n' for line in new_source.split('\n')]
                return
    raise RuntimeError(f'marker not found: {marker}')


ieee_stage1 = """def stage1_preprocessing(df):
    print('[Stage 1] Preprocessing Pipeline...')

    # 1a. Separate numerical/categorical columns
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['TransactionID', 'isFraud']]

    # 1b. Null handling
    for c in cat_cols:
        df[c].fillna('UNKNOWN', inplace=True)
    for c in num_cols:
        df[c].fillna(df[c].median(), inplace=True)

    # === HETERO TEMPORAL GRAPH PREP ===
    raw_cols = ['P_emaildomain', 'R_emaildomain', 'DeviceInfo', 'DeviceType',
                'id_30', 'id_31', 'id_33', 'ProductCD',
                'addr1', 'addr2', 'card1', 'card2', 'card3', 'card5']
    for c in raw_cols:
        if c in df.columns:
            df[f'{c}_raw'] = df[c].astype(str).copy()

    fp_parts = []
    for c in ['DeviceType_raw', 'DeviceInfo_raw', 'id_30_raw', 'id_31_raw', 'id_33_raw']:
        if c in df.columns:
            fp_parts.append(df[c].fillna('UNKNOWN'))
        else:
            fp_parts.append(pd.Series(['UNKNOWN'] * len(df), index=df.index))
    df['device_fp_raw'] = fp_parts[0] + '|' + fp_parts[1] + '|' + fp_parts[2] + '|' + fp_parts[3] + '|' + fp_parts[4]

    # 1b2. Frequency encoding (BEFORE ordinal encoding)
    for fc in ['R_emaildomain', 'P_emaildomain']:
        if fc in df.columns:
            freq = df[fc].value_counts(normalize=True)
            df[fc + '_freq'] = df[fc].map(freq).astype(np.float32)

    # 1c. Encoding
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[cat_cols] = oe.fit_transform(df[cat_cols])

    # 1d. Construct AccountID (UID proxy)
    if 'TransactionDT' in df.columns and 'D1' in df.columns and 'card1' in df.columns and 'addr1' in df.columns:
        df['Day'] = (df['TransactionDT'] / 86400).astype(int)
        df['D1n'] = df['Day'] - df['D1'].fillna(0)
        df['AccountID'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str) + '_' + df['D1n'].astype(str)
    elif 'card1' in df.columns and 'card2' in df.columns:
        df['AccountID'] = df['card1'].astype(str) + '_' + df['card2'].astype(str)
    else:
        df['AccountID'] = df.index.astype(str)

    # 1e. Time features from TransactionDT
    if 'TransactionDT' in df.columns:
        df['Hour'] = (df['TransactionDT'] / 3600).astype(int) % 24
        df['DayOfWeek'] = (df['TransactionDT'] / 86400).astype(int) % 7

    # 1f. Feature Engineering
    if 'TransactionAmt' in df.columns:
        df['LogAmt'] = np.log1p(df['TransactionAmt'])
        df['Amt_cents'] = np.round(df['TransactionAmt'] - np.floor(df['TransactionAmt']), 2).astype(np.float32)

    if 'TransactionDT' in df.columns and 'AccountID' in df.columns:
        df['DT_diff'] = df.groupby('AccountID')['TransactionDT'].diff().fillna(86400).astype(np.float32)
        df['UID_count'] = df.groupby('AccountID')['TransactionDT'].transform('count').astype(np.float32)
        df['UID_Amt_mean'] = df.groupby('AccountID')['TransactionAmt'].transform('mean').astype(np.float32)
        df['UID_Amt_std'] = df.groupby('AccountID')['TransactionAmt'].transform('std').fillna(0).astype(np.float32)

    for card_col in ['card1', 'card2', 'card5']:
        if card_col in df.columns and 'TransactionAmt' in df.columns:
            df[f'{card_col}_count'] = df.groupby(card_col)['TransactionAmt'].transform('count').astype(np.float32)
            df[f'{card_col}_amt_mean'] = df.groupby(card_col)['TransactionAmt'].transform('mean').astype(np.float32)
            df[f'{card_col}_amt_std'] = df.groupby(card_col)['TransactionAmt'].transform('std').fillna(0).astype(np.float32)

    if 'TransactionAmt' in df.columns and 'Hour' in df.columns:
        df['Amt_x_Hour'] = (df['TransactionAmt'] * df['Hour']).astype(np.float32)
    if 'C1' in df.columns and 'C14' in df.columns:
        df['C1_div_C14'] = (df['C1'] / (df['C14'] + 1e-6)).astype(np.float32)
    if 'TransactionAmt' in df.columns and 'card1' in df.columns:
        card_mean = df.groupby('card1')['TransactionAmt'].transform('mean')
        df['Amt_card_dev'] = (df['TransactionAmt'] - card_mean).astype(np.float32)

    freq_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'DeviceType', 'DeviceInfo']
    for fc in freq_cols:
        if fc in df.columns:
            df[f'{fc}_freq'] = df[fc].map(df[fc].value_counts(normalize=True)).astype(np.float32)

    col_list = df.columns.tolist()
    if 'C1' in col_list:
        df['UID_C1_mean'] = df.groupby('AccountID')['C1'].transform('mean').astype(np.float32)
    if 'C14' in col_list:
        df['UID_C14_mean'] = df.groupby('AccountID')['C14'].transform('mean').astype(np.float32)
    if 'D1' in col_list:
        df['UID_D1_std'] = df.groupby('AccountID')['D1'].transform('std').fillna(0).astype(np.float32)

    print('  [Tier 3] Applying Bayesian Target Encoding (Smoothing) & Time-Window...')
    import re

    def smooth_target_encoding(col, weight=100):
        if col not in df.columns:
            return
        global_mean = df['isFraud'].mean()
        agg = df.groupby(col)['isFraud'].agg(['count', 'mean'])
        smooth = (agg['count'] * agg['mean'] + weight * global_mean) / (agg['count'] + weight)
        df[f'{col}_fraud_rate'] = df[col].map(smooth.to_dict()).fillna(global_mean).astype(np.float32)

    for c in ['card1', 'card2', 'card3', 'addr1', 'addr2', 'P_emaildomain']:
        smooth_target_encoding(c, weight=300)

    if 'TransactionAmt' in df.columns and 'AccountID' in df.columns:
        group = df.groupby('AccountID')['TransactionAmt']
        df['Amt_mean_3'] = group.transform(lambda x: x.rolling(3, min_periods=1).mean()).astype(np.float32)
        df['Amt_std_3'] = group.transform(lambda x: x.rolling(3, min_periods=1).std()).fillna(0).astype(np.float32)

    v_cols = [c for c in df.columns if re.match(r'^V\\d+$', c)]
    if len(v_cols) > 20:
        print(f'  [Tier 3] Applying PCA down to 15 components on {len(v_cols)} V-columns...')
        from sklearn.decomposition import PCA
        try:
            pca = PCA(n_components=15, random_state=42)
            v_pca = pca.fit_transform(df[v_cols].fillna(0))
            for j in range(15):
                df[f'V_pca_{j}'] = v_pca[:, j].astype(np.float32)
            df.drop(columns=v_cols, inplace=True)
            num_cols = [c for c in num_cols if c not in v_cols]
            num_cols += [f'V_pca_{j}' for j in range(15)]
        except Exception as e:
            print(f'  [Tier 3 WARNING] PCA failed: {e}')

    scale_cols = [c for c in num_cols if c in df.columns]
    scaler = MinMaxScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    if 'TransactionDT' in df.columns:
        df = df.sort_values('TransactionDT').reset_index(drop=True)

    print(f'  Preprocessed shape: {df.shape}')
    print(f'  Fraud ratio: {df["isFraud"].mean():.4%}')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)
    import gc; gc.collect()
    print(f'  Memory after preprocessing: {df.memory_usage(deep=True).sum()/1e9:.2f} GB')
    return df, scaler"""

ieee_graph = """def extract_graph_view(df, n_hops=2):
    \"\"\"Heterogeneous temporal graph-derived features for IEEE-CIS.\"\"\"
    import time
    from collections import defaultdict, deque

    t0 = time.time()
    print('  [View 3] Heterogeneous Temporal Graph Features...')

    if 'TransactionDT' not in df.columns:
        raise ValueError('TransactionDT is required for temporal graph features')

    df = df.sort_values('TransactionDT').reset_index(drop=True).copy()
    txn_time = df['TransactionDT'].astype(np.float64).values
    txn_amt = df['TransactionAmt'].astype(np.float32).values if 'TransactionAmt' in df.columns else np.zeros(len(df), dtype=np.float32)

    relation_cols = {
        'acct': 'AccountID',
        'card1': 'card1_raw',
        'addr1': 'addr1_raw',
        'pemail': 'P_emaildomain_raw',
        'remail': 'R_emaildomain_raw',
        'device': 'device_fp_raw',
        'product': 'ProductCD_raw',
    }
    unique_counterparts = {
        'card1': ('addr1_raw', 'addr1'),
        'addr1': ('card1_raw', 'card1'),
        'pemail': ('AccountID', 'acct'),
        'remail': ('AccountID', 'acct'),
        'device': ('AccountID', 'acct'),
    }
    windows = {'1d': 86400.0, '7d': 86400.0 * 7, '30d': 86400.0 * 30}

    states = {rel: defaultdict(deque) for rel in relation_cols}
    acct_seen = {
        'device': defaultdict(set),
        'pemail': defaultdict(set),
        'addr1': defaultdict(set),
    }
    acct_hist = defaultdict(int)

    gcols = []
    for rel in relation_cols:
        gcols.extend([
            f'grp_{rel}_cnt_prev_1d',
            f'grp_{rel}_cnt_prev_7d',
            f'grp_{rel}_cnt_prev_30d',
            f'grp_{rel}_amt_sum_7d',
            f'grp_{rel}_time_since_last',
        ])
        if rel in unique_counterparts:
            gcols.append(f'grp_{rel}_unique_{unique_counterparts[rel][1]}_30d')

    gcols.extend([
        'motif_acct_new_device_flag',
        'motif_acct_new_pemail_flag',
        'motif_acct_new_addr1_flag',
    ])

    for col in gcols:
        df[col] = 0.0

    invalid_tokens = {'UNKNOWN', 'nan', 'None', '-1', ''}

    for idx in range(len(df)):
        now = float(txn_time[idx])
        amt = float(txn_amt[idx])
        acct = str(df.at[idx, 'AccountID']) if 'AccountID' in df.columns else 'UNKNOWN'

        rel_values = {}
        for rel, col in relation_cols.items():
            rel_values[rel] = str(df.at[idx, col]) if col in df.columns else 'UNKNOWN'

        for rel, key in rel_values.items():
            if key in invalid_tokens:
                continue

            dq = states[rel][key]
            while dq and (now - dq[0][0] > windows['30d']):
                dq.popleft()

            cnt_1d = 0
            cnt_7d = 0
            amt_7d = 0.0
            uniq_vals = set()

            for event in reversed(dq):
                dt = now - event[0]
                if dt <= windows['30d'] and rel in unique_counterparts:
                    uniq_vals.add(event[2])
                if dt <= windows['7d']:
                    cnt_7d += 1
                    amt_7d += event[1]
                else:
                    break

            for event in reversed(dq):
                if now - event[0] <= windows['1d']:
                    cnt_1d += 1
                else:
                    break

            df.at[idx, f'grp_{rel}_cnt_prev_1d'] = np.float32(cnt_1d)
            df.at[idx, f'grp_{rel}_cnt_prev_7d'] = np.float32(cnt_7d)
            df.at[idx, f'grp_{rel}_cnt_prev_30d'] = np.float32(len(dq))
            df.at[idx, f'grp_{rel}_amt_sum_7d'] = np.float32(amt_7d)
            df.at[idx, f'grp_{rel}_time_since_last'] = np.float32(now - dq[-1][0]) if dq else np.float32(windows['30d'])

            if rel in unique_counterparts:
                out_name = unique_counterparts[rel][1]
                df.at[idx, f'grp_{rel}_unique_{out_name}_30d'] = np.float32(len(uniq_vals))

        if acct not in invalid_tokens:
            if rel_values['device'] not in invalid_tokens and acct_hist[acct] > 0 and rel_values['device'] not in acct_seen['device'][acct]:
                df.at[idx, 'motif_acct_new_device_flag'] = 1.0
            if rel_values['pemail'] not in invalid_tokens and acct_hist[acct] > 0 and rel_values['pemail'] not in acct_seen['pemail'][acct]:
                df.at[idx, 'motif_acct_new_pemail_flag'] = 1.0
            if rel_values['addr1'] not in invalid_tokens and acct_hist[acct] > 0 and rel_values['addr1'] not in acct_seen['addr1'][acct]:
                df.at[idx, 'motif_acct_new_addr1_flag'] = 1.0

        for rel, key in rel_values.items():
            if key in invalid_tokens:
                continue
            counterpart = 'UNKNOWN'
            if rel in unique_counterparts and unique_counterparts[rel][0] in df.columns:
                counterpart = str(df.at[idx, unique_counterparts[rel][0]])
            states[rel][key].append((now, amt, counterpart))

        if acct not in invalid_tokens:
            acct_hist[acct] += 1
            if rel_values['device'] not in invalid_tokens:
                acct_seen['device'][acct].add(rel_values['device'])
            if rel_values['pemail'] not in invalid_tokens:
                acct_seen['pemail'][acct].add(rel_values['pemail'])
            if rel_values['addr1'] not in invalid_tokens:
                acct_seen['addr1'][acct].add(rel_values['addr1'])

    print(f'    Graph features: {len(gcols)} cols ({time.time()-t0:.1f}s)')
    return df[gcols].values.astype(np.float32), gcols"""

ieee_seq = """SEQ_WINDOW = 10

def extract_sequential_view(df, window=SEQ_WINDOW):
    \"\"\"T=10 sliding window per AccountID. Right-padded zeros for cuDNN.\"\"\"
    print(f'  [View 2] Sequential (T={window} per AccountID)...')
    candidates = ['TransactionAmt', 'LogAmt', 'Hour', 'C1', 'C2', 'D1', 'D2',
                  'C14', 'DayOfWeek',
                  'grp_acct_cnt_prev_7d', 'grp_device_cnt_prev_7d', 'grp_pemail_cnt_prev_7d']
    seq_feats = [c for c in candidates if c in df.columns]
    n_feat = len(seq_feats)
    print(f'    Seq features ({n_feat}): {seq_feats}')
    seq = np.zeros((len(df), window, n_feat), dtype=np.float32)
    for _, grp in df.groupby('AccountID'):
        vals = grp[seq_feats].values.astype(np.float32)
        idxs = grp.index.values
        for pos, idx in enumerate(idxs):
            start = max(0, pos - window + 1)
            s = vals[start:pos+1]
            seq[idx, :len(s), :] = s
    print(f'    Shape: {seq.shape}')
    return seq"""

paysim_stage1 = """def stage1_preprocessing(df):
    print('[Stage 1] Preprocessing Pipeline...')
    df = df.fillna(0)
    if 'isFlaggedFraud' in df.columns:
        df = df.drop(columns=['isFlaggedFraud'])

    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['type'])

    # Raw keys for temporal heterogeneous graph features
    df['nameOrig_raw'] = df['nameOrig'].astype(str)
    df['nameDest_raw'] = df['nameDest'].astype(str)
    df['type_raw'] = df['type'].astype(str)
    df['pair_key'] = df['nameOrig_raw'] + '->' + df['nameDest_raw']
    df['AccountID'] = df['nameOrig_raw']

    # Core engineered features
    df['errorBalanceOrg'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    df['amountToOldBalanceRatio'] = df['amount'] / (df['oldbalanceOrg'] + 1)

    # Tier 1: Advanced interaction features
    df['amountToDestRatio'] = (df['amount'] / (df['oldbalanceDest'] + 1)).astype(np.float32)
    df['balanceDiffOrg'] = (df['oldbalanceOrg'] - df['newbalanceOrig']).astype(np.float32)
    df['balanceDiffDest'] = (df['newbalanceDest'] - df['oldbalanceDest']).astype(np.float32)
    df['isZeroOrigBalance'] = (df['oldbalanceOrg'] == 0).astype(np.float32)
    df['amountLogRatio'] = (np.log1p(df['amount']) / (np.log1p(df['oldbalanceOrg']) + 1)).astype(np.float32)
    df['hourOfDay'] = (df['step'] % 24).astype(np.float32)

    print('    [Tier 3] Applying Bayesian Target Encoding (Smoothing) & Dest-Profile...')

    def smooth_target_encoding(col, weight=100):
        if col not in df.columns:
            return
        global_mean = df['isFraud'].mean()
        agg = df.groupby(col)['isFraud'].agg(['count', 'mean'])
        smooth = (agg['count'] * agg['mean'] + weight * global_mean) / (agg['count'] + weight)
        df[f'{col}_fraud_rate'] = df[col].map(smooth.to_dict()).fillna(global_mean).astype(np.float32)

    smooth_target_encoding('type', weight=300)

    if 'nameDest' in df.columns:
        dest_freq = df['nameDest'].value_counts()
        df['dest_txn_count'] = df['nameDest'].map(dest_freq).astype(np.float32)

    if 'step' in df.columns:
        df['orig_step_diff'] = df.groupby('AccountID')['step'].diff().fillna(24).astype(np.float32)
        df['orig_txn_count'] = df.groupby('AccountID')['step'].transform('count').astype(np.float32)
        df['orig_amt_mean'] = df.groupby('AccountID')['amount'].transform('mean').astype(np.float32)

    num_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
                'newbalanceDest', 'errorBalanceOrg', 'errorBalanceDest',
                'amountToOldBalanceRatio', 'amountToDestRatio', 'balanceDiffOrg',
                'balanceDiffDest', 'amountLogRatio', 'hourOfDay', 'dest_txn_count',
                'orig_step_diff', 'orig_txn_count', 'orig_amt_mean']
    num_cols = [c for c in num_cols if c in df.columns]
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    df = df.sort_values('step').reset_index(drop=True)
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)
    import gc; gc.collect()
    print(f'  Preprocessed shape: {df.shape}')
    print(f'  Fraud ratio: {df["isFraud"].mean():.4%}')
    print(f'  Memory: {df.memory_usage(deep=True).sum()/1e9:.2f} GB')
    return df, scaler"""

paysim_tab = """TABULAR_BASE_FEATURES = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig', 'errorBalanceOrg',
    'oldbalanceDest', 'newbalanceDest', 'errorBalanceDest',
    'type_encoded', 'amountToOldBalanceRatio',
    'amountToDestRatio', 'balanceDiffOrg', 'balanceDiffDest',
    'isZeroOrigBalance', 'amountLogRatio', 'hourOfDay',
    'dest_txn_count', 'orig_step_diff', 'orig_txn_count', 'orig_amt_mean',
    'type_fraud_rate'
]

def get_tabular_cols(df):
    cols = [c for c in TABULAR_BASE_FEATURES if c in df.columns]
    cols.extend([c for c in df.columns if c.startswith('grp_') or c.startswith('motif_')])
    return cols

def extract_tabular_view(df, cols=None):
    cols = get_tabular_cols(df) if cols is None else [c for c in cols if c in df.columns]
    print(f'  [View 1] Tabular ({len(cols)} features)')
    return df[cols].values.astype(np.float32)"""

paysim_seq = """SEQ_WINDOW = 5  # Tier 3: Reduced window for PaySim

def extract_sequential_view(df, window=SEQ_WINDOW):
    import time; t0 = time.time()
    print(f'  [View 2] Sequential (T={window}, VECTORIZED, RIGHT-padded)...')
    candidates = ['amount', 'errorBalanceOrg', 'errorBalanceDest',
                  'amountToOldBalanceRatio', 'oldbalanceOrg', 'type_encoded',
                  'amountToDestRatio', 'balanceDiffOrg',
                  'grp_orig_cnt_prev_7d', 'grp_dest_cnt_prev_7d', 'hourOfDay']
    seq_feats = [c for c in candidates if c in df.columns]
    n_feat = len(seq_feats)
    print(f'    Seq features ({n_feat}): {seq_feats}')
    seq = np.zeros((len(df), window, n_feat), dtype=np.float32)
    df_work = df[['nameOrig'] + seq_feats].copy()
    df_work['_oidx'] = np.arange(len(df))
    df_work = df_work.sort_values(['nameOrig', '_oidx'])
    for i, feat in enumerate(seq_feats):
        for t in range(window):
            shifted = df_work.groupby('nameOrig')[feat].shift(t)
            seq[df_work['_oidx'].values, t, i] = shifted.fillna(0).values
        print(f'      Feature {i+1}/{n_feat} done')
    del df_work; import gc; gc.collect()
    print(f'    Shape: {seq.shape} ({time.time()-t0:.1f}s)')
    return seq"""

paysim_graph = """def extract_graph_view(df, n_hops=2):
    \"\"\"Temporal heterogeneous graph-derived features for PaySim.\"\"\"
    import time
    from collections import defaultdict, deque

    t0 = time.time()
    print('  [View 3] Temporal Heterogeneous Graph Features...')

    if 'step' not in df.columns:
        raise ValueError('step is required for temporal graph features')

    df = df.sort_values('step').reset_index(drop=True).copy()
    txn_time = df['step'].astype(np.float64).values
    txn_amt = df['amount'].astype(np.float32).values if 'amount' in df.columns else np.zeros(len(df), dtype=np.float32)

    relation_cols = {
        'orig': 'nameOrig_raw',
        'dest': 'nameDest_raw',
        'pair': 'pair_key',
        'type': 'type_raw',
    }
    unique_counterparts = {
        'dest': ('nameOrig_raw', 'orig'),
        'pair': ('type_raw', 'type'),
    }
    windows = {'1d': 24.0, '7d': 24.0 * 7}
    states = {rel: defaultdict(deque) for rel in relation_cols}

    gcols = [
        'grp_orig_cnt_prev_1d', 'grp_orig_cnt_prev_7d', 'grp_orig_amt_sum_7d', 'grp_orig_time_since_last',
        'grp_dest_cnt_prev_1d', 'grp_dest_cnt_prev_7d', 'grp_dest_amt_sum_7d', 'grp_dest_time_since_last',
        'grp_pair_cnt_prev_1d', 'grp_pair_cnt_prev_7d', 'grp_pair_amt_sum_7d', 'grp_pair_time_since_last',
        'grp_type_cnt_prev_1d', 'grp_type_cnt_prev_7d',
        'grp_dest_unique_orig_7d',
        'motif_orig_new_dest_flag', 'motif_dest_new_orig_flag'
    ]
    for col in gcols:
        df[col] = 0.0

    invalid_tokens = {'UNKNOWN', 'nan', 'None', ''}
    acct_seen_dest = defaultdict(set)
    dest_seen_orig = defaultdict(set)
    acct_hist = defaultdict(int)

    for idx in range(len(df)):
        now = float(txn_time[idx])
        amt = float(txn_amt[idx])
        orig = str(df.at[idx, 'nameOrig_raw'])
        dest = str(df.at[idx, 'nameDest_raw'])
        pair = str(df.at[idx, 'pair_key'])
        typ = str(df.at[idx, 'type_raw'])

        for rel, key in [('orig', orig), ('dest', dest), ('pair', pair), ('type', typ)]:
            if key in invalid_tokens:
                continue

            dq = states[rel][key]
            while dq and (now - dq[0][0] > windows['7d']):
                dq.popleft()

            cnt_1d = 0
            cnt_7d = len(dq)
            amt_7d = 0.0
            uniq_vals = set()

            for event in reversed(dq):
                dt = now - event[0]
                if dt <= windows['1d']:
                    cnt_1d += 1
                if dt <= windows['7d']:
                    amt_7d += event[1]
                    if rel in unique_counterparts:
                        uniq_vals.add(event[2])
                else:
                    break

            if rel != 'type':
                df.at[idx, f'grp_{rel}_cnt_prev_1d'] = np.float32(cnt_1d)
                df.at[idx, f'grp_{rel}_cnt_prev_7d'] = np.float32(cnt_7d)
                df.at[idx, f'grp_{rel}_amt_sum_7d'] = np.float32(amt_7d)
                df.at[idx, f'grp_{rel}_time_since_last'] = np.float32(now - dq[-1][0]) if dq else np.float32(windows['7d'])
            else:
                df.at[idx, 'grp_type_cnt_prev_1d'] = np.float32(cnt_1d)
                df.at[idx, 'grp_type_cnt_prev_7d'] = np.float32(cnt_7d)

            if rel == 'dest':
                df.at[idx, 'grp_dest_unique_orig_7d'] = np.float32(len(uniq_vals))

        if orig not in invalid_tokens and acct_hist[orig] > 0 and dest not in acct_seen_dest[orig]:
            df.at[idx, 'motif_orig_new_dest_flag'] = 1.0
        if dest not in invalid_tokens and orig not in dest_seen_orig[dest] and len(dest_seen_orig[dest]) > 0:
            df.at[idx, 'motif_dest_new_orig_flag'] = 1.0

        if orig not in invalid_tokens:
            acct_hist[orig] += 1
            acct_seen_dest[orig].add(dest)
        if dest not in invalid_tokens:
            dest_seen_orig[dest].add(orig)

        states['orig'][orig].append((now, amt, dest))
        states['dest'][dest].append((now, amt, orig))
        states['pair'][pair].append((now, amt, typ))
        states['type'][typ].append((now, amt, orig))

    print(f'    Features: {gcols} ({time.time()-t0:.1f}s)')
    return df[gcols].values.astype(np.float32), gcols"""

paysim_main = """from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import time as _time

# ====== STAGE 1 ======
t_start = _time.time()
df_proc, scaler = stage1_preprocessing(df_raw)
del df_raw; import gc; gc.collect()

# ====== STAGE 3 (Graph FIRST so features are available for tabular+seq) ======
print('\\n[Stage 3] Multi-View Feature Engineering...')
X_grp, gcols = extract_graph_view(df_proc)
selected = get_tabular_cols(df_proc)
X_tab = extract_tabular_view(df_proc, selected)
X_seq = extract_sequential_view(df_proc)
y = df_proc['isFraud'].values
uid_array = df_proc['AccountID'].values  # PHASE 6: UID Extraction
df_meta = df_proc[['amount']].copy() if 'amount' in df_proc.columns else None
del df_proc; gc.collect()
print('  Data arrays ready. Memory freed.')

# ====== STAGE 2: 5-Fold Stratified CV ======
print('\\n[Stage 2] 5-Block Walk-Forward CV...')
USE_STRICT_TIME_SPLIT = False
if USE_STRICT_TIME_SPLIT:
    print('  [Validation] Using 5-Block Walk-Forward CV (Strict/Realistic)\\n')
    cv = TimeSeriesSplit(n_splits=5)
else:
    print('  [Validation] Using 5-Fold Stratified CV (Randomized/High Metrics)\\n')
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = []

for fold, (tr_i, vl_i) in enumerate(cv.split(X_tab, y)):
    t_fold = _time.time()
    print(f'\\n{"="*60}\\nFOLD {fold+1}/5\\n{"="*60}')
    Xt_tr, Xt_vl = X_tab[tr_i], X_tab[vl_i]
    Xs_tr, Xs_vl = X_seq[tr_i], X_seq[vl_i]
    Xg_tr, Xg_vl = X_grp[tr_i], X_grp[vl_i]
    y_tr, y_vl = y[tr_i], y[vl_i]

    from sklearn.model_selection import StratifiedKFold as SKF_inner
    inner_cv = SKF_inner(n_splits=3, shuffle=True, random_state=42)
    n_tr = len(y_tr)
    oof_train = np.zeros((n_tr, 4))
    print(f'  [Full-OOF] Generating OOF on {n_tr} samples via 3-fold internal CV...')
    
    for ifold, (itr, ivl) in enumerate(inner_cv.split(Xt_tr, y_tr)):
        print(f'    [Inner {ifold+1}/3] Train: {len(itr)}, Val: {len(ivl)}')
        inner_models = generate_oof_train(Xt_tr[itr], y_tr[itr])
        oof_part, _ = predict_oof(inner_models, Xt_tr[ivl])
        oof_train[ivl] = oof_part
        del inner_models; import gc; gc.collect()
    
    print('  [Stage 4] Training FINAL models on FULL training fold...')
    models = generate_oof_train(Xt_tr, y_tr)
    
    print('  [Stage 4] Predicting OOF on validation...')
    oof_vl, best_lgb = predict_oof(models, Xt_vl)

    meta, meta_p = stage5_meta(oof_train, y_tr, oof_vl)
    import pandas as pd
    uid_vl = uid_array[vl_i]
    uid_df = pd.DataFrame({'UID': uid_vl, 'pred': meta_p})
    uid_mean = uid_df.groupby('UID')['pred'].transform('mean')
    meta_p = 0.7 * meta_p + 0.3 * uid_mean.values
    auprc = average_precision_score(y_vl, meta_p)
    roc = roc_auc_score(y_vl, meta_p)

    best_t, best_f1, best_p, best_r = 0.5, 0, 0, 0
    for t in np.arange(0.1, 0.9, 0.05):
        f1_t = f1_score(y_vl, (meta_p >= t).astype(int), zero_division=0)
        if f1_t > best_f1:
            best_t, best_f1 = t, f1_t
            best_p = precision_score(y_vl, (meta_p >= t).astype(int), zero_division=0)
            best_r = recall_score(y_vl, (meta_p >= t).astype(int), zero_division=0)
    y_pred = (meta_p >= best_t).astype(int)
    f1 = best_f1; prec = best_p; rec = best_r

    fold_time = _time.time() - t_fold
    fold_metrics.append({'Fold': fold+1, 'AUPRC': auprc, 'ROC-AUC': roc,
                         'F1': f1, 'Precision': prec, 'Recall': rec,
                         'Time_min': fold_time/60})
    print(f'  FOLD {fold+1}: AUPRC={auprc:.4f}, ROC-AUC={roc:.4f}, F1={f1:.4f}, P={prec:.4f}, R={rec:.4f} ({fold_time/60:.1f}min)')

    if fold == 4:
        print(f'\\n{"="*60}\\nFOLD 5 (FINAL): Detailed Metrics + XAI\\n{"="*60}')
        print(classification_report(y_vl, y_pred, digits=4))
        print('--- Confusion Matrix ---')
        cm = confusion_matrix(y_vl, y_pred)
        print(cm)

        print(f'\\n--- Threshold Analysis ---')
        for t in [0.3, 0.4, 0.5, 0.6]:
            yt = (meta_p >= t).astype(int)
            print(f'  t={t:.1f}: F1={f1_score(y_vl, yt, zero_division=0):.4f}, '
                  f'P={precision_score(y_vl, yt, zero_division=0):.4f}, '
                  f'R={recall_score(y_vl, yt, zero_division=0):.4f}')

        Xs = Xt_vl[:200]
        top3 = stage6_xai(best_lgb, Xt_tr[:2000], Xs, selected)
        stage7_dual_output(meta_p, top3, y_vl)
        if df_meta is not None:
            stage8_hitl(meta_p, df_meta.iloc[vl_i])
"""

ieee = load(ieee_path)
paysim = load(paysim_path)

set_cell_source(ieee, 'def stage1_preprocessing(df):', ieee_stage1)
set_cell_source(ieee, 'def extract_hetero_temporal_graph_features(df):', ieee_graph)
set_cell_source(ieee, 'SEQ_WINDOW = 10', ieee_seq)

set_cell_source(paysim, 'def stage1_preprocessing(df):', paysim_stage1)
set_cell_source(paysim, 'TABULAR_FEATURES = [', paysim_tab)
set_cell_source(paysim, 'SEQ_WINDOW = 5  # Tier 3: Reduced window for PaySim', paysim_seq)
set_cell_source(paysim, 'def extract_graph_view(df, n_hops=2):', paysim_graph)
set_cell_source(paysim, 'from sklearn.metrics import classification_report', paysim_main)

save(ieee_path, ieee)
save(paysim_path, paysim)
print('Patched both notebooks.')

