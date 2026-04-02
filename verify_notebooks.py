import json

for fname in ['MVS_XAI_Colab_IEEE_CIS.ipynb', 'MVS_XAI_Colab_DataPrep_Phase1.ipynb']:
    nb = json.load(open(fname, encoding='utf-8'))
    cells = nb['cells']
    issues = []

    defined_sig = None
    called_sig = None

    for i, cell in enumerate(cells):
        src = ''.join(cell.get('source', []))

        # Check extract_tabular_view definition vs call
        if 'def extract_tabular_view(df)' in src:
            defined_sig = 'df_only'
        elif 'def extract_tabular_view(df, cols)' in src:
            defined_sig = 'df_cols'

        if 'extract_tabular_view(df_proc)' in src and 'def ' not in src:
            called_sig = 'df_only'
        elif 'extract_tabular_view(df_proc, selected)' in src and 'def ' not in src:
            called_sig = 'df_cols'

        # Check pipeline order in main cell
        if 'extract_graph_view(df_proc)' in src and 'extract_tabular_view' in src:
            lines = src.split('\n')
            graph_line = -1
            tab_line = -1
            for j, l in enumerate(lines):
                if 'extract_graph_view' in l and 'def ' not in l:
                    graph_line = j
                if 'extract_tabular_view' in l and 'def ' not in l:
                    tab_line = j
            if graph_line > 0 and tab_line > 0 and graph_line > tab_line:
                issues.append('ORDER: Graph called AFTER tabular!')
            elif graph_line > 0 and tab_line > 0:
                pass  # OK

        # Check XAI feat_names vs X_tab columns mismatch
        if 'stage6_xai' in src and 'Xt_vl' in src and 'def ' not in src:
            if 'feat_names' in src:
                # PaySim uses hardcoded feature list
                pass  
            if 'selected' in src:
                pass  # IEEE uses dynamic selected list

    # Verify signature match
    if defined_sig and called_sig:
        if defined_sig != called_sig:
            issues.append(f'MISMATCH: extract_tabular_view def={defined_sig} call={called_sig}')

    print(f'=== {fname} ===')
    print(f'  Cells: {len(cells)}')
    print(f'  extract_tabular_view: def={defined_sig}, call={called_sig}')
    if defined_sig == called_sig:
        print(f'  Signature match: OK')
    else:
        print(f'  Signature match: MISMATCH!')

    # Check for common issues
    all_src = '\n'.join(''.join(c.get('source', [])) for c in cells)
    checks = {
        'CatBoost import': 'from catboost import CatBoostClassifier' in all_src,
        'Focal Loss def': 'def focal_loss_tf' in all_src,
        'BiLSTM (Bidirectional)': 'Bidirectional' in all_src,
        'OOF [Nx5]': 'np.zeros((n, 5))' in all_src,
        'No GAT class': 'class SimpleGAT' not in all_src,
        'No old OOF [Nx4]': 'np.zeros((n, 4))' not in all_src,
        '5 model names': "['RF','XGB','LGB','BiLSTM','CatBoost']" in all_src,
        'Graph-first pipeline': 'Graph FIRST' in all_src,
        'EarlyStopping': 'EarlyStopping' in all_src,
        'Meta C=0.1': 'C=0.1' in all_src,
    }
    for check, ok in checks.items():
        status = 'OK' if ok else 'MISSING!'
        if not ok:
            issues.append(f'{check}: {status}')
        print(f'  {check}: {status}')

    if issues:
        print(f'\n  ** ISSUES FOUND ({len(issues)}):')
        for iss in issues:
            print(f'    - {iss}')
    else:
        print(f'\n  ** ALL CHECKS PASSED! No issues found.')
    print()
