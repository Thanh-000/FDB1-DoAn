import json
import re

fname = 'MVS_XAI_Colab_IEEE_CIS.ipynb'
print(f"\n{'='*50}\nPatching {fname} (Phase 3: Kaggle Magic Grandmaster Features)\n{'='*50}")

with open(fname, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb.get('cells', [])):
    src = ''.join(cell.get('source', []))
    new_src = src
    
    # 1. Add UID and Kaggle Features in stage1_preprocessing
    if 'def stage1_preprocessing(df):' in new_src:
        # Patching AccountID definition to create UID (Magic Feature)
        target_account_id = """    # 1d. Construct AccountID proxy
    if 'card1' in df.columns and 'card2' in df.columns and 'addr1' in df.columns:
        df['AccountID'] = df['card1'].astype(str) + '_' + df['card2'].astype(str) + '_' + df['addr1'].astype(str)
    else:
        df['AccountID'] = df.index.astype(str)"""
        
        repl_account_id = """    # 1d. Construct AccountID (UID Magic Proxy using D1)
    if 'TransactionDT' in df.columns and 'D1' in df.columns and 'card1' in df.columns and 'addr1' in df.columns:
        df['Day'] = (df['TransactionDT'] / 86400).astype(int)
        df['D1n'] = df['Day'] - df['D1'].fillna(0)
        df['AccountID'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str) + '_' + df['D1n'].astype(str)
    elif 'card1' in df.columns and 'card2' in df.columns:
        df['AccountID'] = df['card1'].astype(str) + '_' + df['card2'].astype(str)
    else:
        df['AccountID'] = df.index.astype(str)"""
        
        if target_account_id in new_src:
            new_src = new_src.replace(target_account_id, repl_account_id)
            print(" ✅ [1/3] Injected Kaggle Magic UID (AccountID refactoring)")
            
        # Patching Feature Engineering (Fractional cents + TimeDelta)
        target_fe = """    # 1f2. Advanced Interaction Features (Tier 1)"""
        repl_fe = """    # 1f2. Advanced Interaction Features (Tier 1 & Kaggle Grandmaster)
    if 'TransactionAmt' in df.columns:
        df['Amt_cents'] = np.round(df['TransactionAmt'] - np.floor(df['TransactionAmt']), 2).astype(np.float32)
        
    if 'TransactionDT' in df.columns and 'AccountID' in df.columns:
        df['DT_diff'] = df.groupby('AccountID')['TransactionDT'].diff().fillna(86400).astype(np.float32)
        df['UID_count'] = df.groupby('AccountID')['TransactionDT'].transform('count').astype(np.float32)
        df['UID_Amt_mean'] = df.groupby('AccountID')['TransactionAmt'].transform('mean').astype(np.float32)
        df['UID_Amt_std'] = df.groupby('AccountID')['TransactionAmt'].transform('std').fillna(0).astype(np.float32)
"""
        if target_fe in new_src:
            new_src = new_src.replace(target_fe, repl_fe)
            print(" ✅ [2/3] Added Cents, DT_diff (Velocity v2), and UID Aggregations")

    # 2. Update select_tabular_cols
    if 'def select_tabular_cols(df):' in new_src:
        target_extra = """    extra = ['LogAmt', 'Hour', 'DayOfWeek',
             'R_emaildomain_freq', 'P_emaildomain_freq',
             'Amt_x_Hour', 'C1_div_C14', 'Amt_card_dev',
             'grp_degree', 'grp_pagerank', 'grp_ego_dens',
             'Amt_mean_3', 'Amt_std_3']"""
             
        repl_extra = """    extra = ['LogAmt', 'Hour', 'DayOfWeek',
             'R_emaildomain_freq', 'P_emaildomain_freq',
             'Amt_x_Hour', 'C1_div_C14', 'Amt_card_dev',
             'grp_degree', 'grp_pagerank', 'grp_ego_dens',
             'Amt_mean_3', 'Amt_std_3',
             'Amt_cents', 'DT_diff', 'UID_count', 'UID_Amt_mean', 'UID_Amt_std', 'D1n']"""
             
        if target_extra in new_src:
            new_src = new_src.replace(target_extra, repl_extra)
            print(" ✅ [3/3] Updated select_tabular_cols to include all Phase 3 Magic Features")

    if src != new_src:
        nb['cells'][i]['source'] = [l + '\n' for l in new_src.split('\n')][:-1]

with open(fname, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n🚀 Phase 3 Kaggle Magic Patch Complete!")
