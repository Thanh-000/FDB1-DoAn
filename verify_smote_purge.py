import json
import re

files = ['MVS_XAI_Colab_IEEE_CIS.ipynb', 'MVS_XAI_Colab_DataPrep_Phase1.ipynb']

patterns_to_check = [
    r'\bSMOTE\b',
    r'\bXt_sm\b',
    r'\by_sm\b',
    r'\by_smote\b',
    r'fit_resample',
    r'imblearn',
]

def analyze_purged_notebooks():
    for fname in files:
        print(f"\n{'='*50}\n🔎 DEEP SCAN: {fname}\n{'='*50}")
        with open(fname, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        found_issues = False
        
        for i, cell in enumerate(nb['cells']):
            source = ''.join(cell.get('source', []))
            lines = source.split('\n')
            
            for j, line in enumerate(lines):
                # We skip pure comment lines or markdown cells, but we want to know what's left
                # Actually, let's just flag ANY code line that contains the banned words
                if cell['cell_type'] == 'code' and not line.strip().startswith('#'):
                    for p in patterns_to_check:
                        if re.search(p, line):
                            # Exempt expected print statements
                            if 'print' in line and ('disabled' in line or 'Removed' in line):
                                continue
                            print(f"⚠️ SUSPICIOUS PATTERN '{p}' found in Code Cell {i}, Line {j}:")
                            print(f"   ► {line.strip()}")
                            found_issues = True
                            
                # Let's also verify that `generate_oof_train` is exactly correct
                if 'def generate_oof_train' in line:
                    print(f"✅ Found def: {line.strip()}")
                if 'models = generate_oof_train' in line:
                    print(f"✅ Found call: {line.strip()}")
                    
                # Let's verify `ratio` definition and usage
                if 'ratio =' in line:
                    print(f"✅ Found ratio def: {line.strip()}")
                if 'scale_pos_weight=' in line:
                    print(f"✅ Found scale_pos_weight usage: {line.strip()}")
                    
        if not found_issues:
            print("✅ ZERO leftover SMOTE variables or code detected in active code blocks.")

if __name__ == '__main__':
    analyze_purged_notebooks()
