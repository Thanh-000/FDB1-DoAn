import json

def fix_oof_mismatch(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and cell.get('metadata', {}).get('id') == 'run_pipeline':
            src = ''.join(cell['source'])
            
            # Fix: Use original (pre-SMOTE) data for training OOF to avoid size mismatch
            # Old: oof_tr, _ = generate_oof(Xt_sm, y_sm, Xt_sm, Xs_tr, Xs_tr, y_tr, Xg_tr, Xg_tr)
            # New: oof_tr, _ = generate_oof(Xt_sm, y_sm, Xt_tr, Xs_tr, Xs_tr, y_tr, Xg_tr, Xg_tr)
            #      meta fits on y_tr (original) instead of y_sm (SMOTE'd)
            old_oof_line = "oof_tr, _ = generate_oof(Xt_sm, y_sm, Xt_sm, Xs_tr, Xs_tr, y_tr, Xg_tr, Xg_tr)"
            new_oof_line = "oof_tr, _ = generate_oof(Xt_sm, y_sm, Xt_tr, Xs_tr, Xs_tr, y_tr, Xg_tr, Xg_tr)"
            
            old_meta_line = "meta, meta_p = stage5_meta(oof_tr, y_sm, oof_vl)"
            new_meta_line = "meta, meta_p = stage5_meta(oof_tr, y_tr, oof_vl)"
            
            src = src.replace(old_oof_line, new_oof_line)
            src = src.replace(old_meta_line, new_meta_line)
            
            cell['source'] = [line + '\n' for line in src.split('\n')]
            # Remove trailing empty newline if present
            if cell['source'] and cell['source'][-1].strip() == '':
                cell['source'] = cell['source'][:-1]
            
            print(f"  Fixed OOF mismatch in {fp}")
    
    with open(fp, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

fix_oof_mismatch('MVS_XAI_Colab_IEEE_CIS.ipynb')
fix_oof_mismatch('MVS_XAI_Colab_DataPrep_Phase1.ipynb')
print("Both notebooks fixed!")
