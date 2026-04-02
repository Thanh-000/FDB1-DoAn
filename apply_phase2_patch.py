import json

def patch_notebook(fname, is_ieee):
    print(f"\n{'='*50}\nPatching {fname} (Phase 2)\n{'='*50}")
    with open(fname, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    changes = 0
    for i, cell in enumerate(nb.get('cells', [])):
        src = ''.join(cell.get('source', []))
        new_src = src
        
        # 1. Update 1f3: Target Encoding Smoothing (Bayesian Smoothing)
        if '# 1f3. Tier 3: Time-Window Aggregations & Target Encoding' in new_src and is_ieee:
            target_str = """    # 1f3. Tier 3: Time-Window Aggregations & Target Encoding
    print('  [Tier 3] Applying Time-Window (Velocity) & Target Encoding...')
    import re
    if 'card1' in df.columns:
        te_map = df.groupby('card1')['isFraud'].mean().to_dict()
        df['card1_fraud_rate'] = df['card1'].map(te_map).fillna(df['isFraud'].mean()).astype(np.float32)"""
            
            replacement = """    # 1f3. Tier 3: Bayesian Target Encoding (Smoothing) & Time-Window
    print('  [Tier 3] Applying Bayesian Target Encoding (Smoothing) & Time-Window...')
    import re
    
    # Bayesian Smoothing for high-cardinality categoricals
    def smooth_target_encoding(col, weight=100):
        if col not in df.columns: return
        global_mean = df['isFraud'].mean()
        agg = df.groupby(col)['isFraud'].agg(['count', 'mean'])
        smooth = (agg['count'] * agg['mean'] + weight * global_mean) / (agg['count'] + weight)
        df[f'{col}_fraud_rate'] = df[col].map(smooth.to_dict()).fillna(global_mean).astype(np.float32)
        
    for c in ['card1', 'card2', 'card3', 'addr1', 'addr2', 'P_emaildomain']:
        smooth_target_encoding(c, weight=300)"""
            
            if target_str in new_src:
                new_src = new_src.replace(target_str, replacement)
                print(f"  [Cell {i}] ✅ Applied Bayesian Target Encoding Smoothing to 6 features")
                changes += 1

        # IEEE-only: Fix select_tabular_cols to include all engineered columns
        if 'def select_tabular_cols(df):' in new_src and is_ieee:
            extra_str = """    extra = ['LogAmt', 'Hour', 'DayOfWeek',
             'R_emaildomain_freq', 'P_emaildomain_freq',
             'Amt_x_Hour', 'C1_div_C14', 'Amt_card_dev',
             'grp_degree', 'grp_pagerank', 'grp_ego_dens']"""
            
            extra_repl = """    extra = ['LogAmt', 'Hour', 'DayOfWeek',
             'R_emaildomain_freq', 'P_emaildomain_freq',
             'Amt_x_Hour', 'C1_div_C14', 'Amt_card_dev',
             'grp_degree', 'grp_pagerank', 'grp_ego_dens',
             'Amt_mean_3', 'Amt_std_3']
    
    # Add all smoothed target encoded rates
    rate_cols = [c for c in df.columns if c.endswith('_fraud_rate')]
    extra.extend(rate_cols)"""
            
            v_cols_str = """    V_cols = [c for c in df.columns if c.startswith('V') and c[1:].isdigit()]
    V_good = [c for c in V_cols if df[c].isna().mean() < 0.50]
    V_good = sorted(V_good, key=lambda c: df[c].isna().mean())[:70]"""
            
            v_cols_repl = """    # Add PCA components instead of dropping them
    V_good = [c for c in df.columns if c.startswith('V_pca_')]
    if len(V_good) == 0:
        # Fallback to standard V_cols if PCA skipped
        V_cols = [c for c in df.columns if c.startswith('V') and c[1:].isdigit()]
        V_good = [c for c in V_cols if df[c].isna().mean() < 0.50]
        V_good = sorted(V_good, key=lambda c: df[c].isna().mean())[:70]"""

            if extra_str in new_src and v_cols_str in new_src:
                new_src = new_src.replace(extra_str, extra_repl)
                new_src = new_src.replace(v_cols_str, v_cols_repl)
                print(f"  [Cell {i}] ✅ Patched select_tabular_cols to include PCA and new features")
                changes += 1

        # PaySim Alignment: Add smooth_target_encoding helper
        if 'Tier 3: Destination Account Profiling' in new_src and not is_ieee:
            target_str = """    # Tier 3: Destination Account Profiling (Mule Account detection)
    print('    [Tier 3] Adding Dest-Profile Features...')"""
            
            replacement = """    # Tier 3: Bayesian Target Encoding (Smoothing) & Dest-Profile
    print('    [Tier 3] Applying Bayesian Target Encoding (Smoothing) & Dest-Profile...')
    
    def smooth_target_encoding(col, weight=100):
        if col not in df.columns: return
        global_mean = df['isFraud'].mean()
        agg = df.groupby(col)['isFraud'].agg(['count', 'mean'])
        smooth = (agg['count'] * agg['mean'] + weight * global_mean) / (agg['count'] + weight)
        df[f'{col}_fraud_rate'] = df[col].map(smooth.to_dict()).fillna(global_mean).astype(np.float32)
        
    smooth_target_encoding('type', weight=300)

    # Destination Account Profiling (Mule Account detection)"""
            if target_str in new_src:
                new_src = new_src.replace(target_str, replacement)
                print(f"  [Cell {i}] ✅ Aligned Bayesian Target Encoding helper inside PaySim (applied to 'type')")
                changes += 1

        if src != new_src:
            nb['cells'][i]['source'] = [l + '\n' for l in new_src.split('\n')][:-1]

    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"  📊 Total patches applied to {fname}: {changes}")

patch_notebook('MVS_XAI_Colab_IEEE_CIS.ipynb', is_ieee=True)
patch_notebook('MVS_XAI_Colab_DataPrep_Phase1.ipynb', is_ieee=False)
