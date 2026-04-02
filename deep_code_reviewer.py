import json
import ast
import builtins
import sys

files = ['MVS_XAI_Colab_IEEE_CIS.ipynb', 'MVS_XAI_Colab_DataPrep_Phase1.ipynb']

for fname in files:
    print(f"\n{'='*50}\n🔎 LINTING: {fname}\n{'='*50}")
    with open(fname, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    code_lines = []
    line_to_cell = {} # map output lines to notebook cells
    current_line = 1
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = cell.get('source', [])
            for line in source:
                # Comment out magics (like !kaggle, %matplotlib) when doing AST check
                if line.lstrip().startswith('!') or line.lstrip().startswith('%'):
                    code_lines.append('# ' + line)
                else:
                    code_lines.append(line.rstrip('\n'))
                line_to_cell[current_line] = i
                current_line += 1
            code_lines.append("") # padding between cells
            current_line += 1
            
    combined_code = '\n'.join(code_lines)
    
    # Check 1: AST Parse (Catches SyntaxError, IndentationError, incomplete strings)
    try:
        tree = ast.parse(combined_code)
        print("✅ AST (Syntax & Indentation): PASSED (100% Valid Python)")
    except SyntaxError as e:
        print(f"❌ SYNTAX ERROR in {fname}")
        print(f"  Line {e.lineno} (maps to cell {line_to_cell.get(e.lineno, '?')}): {e.text}")
        print(f"  Error details: {e.msg}")
        continue
        
    # Check 2: Very basic undefined variable checking using ast.walk
    # We'll just look for obvious missing imports/names that aren't globals
    class VariableVisitor(ast.NodeVisitor):
        def __init__(self):
            self.assigned = set(dir(builtins))
            self.loaded = set()
            
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Store):
                self.assigned.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                self.loaded.add((node.id, node.lineno))
            self.generic_visit(node)
            
        def visit_Import(self, node):
            for n in node.names:
                name = n.asname if n.asname else n.name.split('.')[0]
                self.assigned.add(name)
            self.generic_visit(node)
            
        def visit_ImportFrom(self, node):
            for n in node.names:
                name = n.asname if n.asname else n.name
                self.assigned.add(name)
            self.generic_visit(node)
            
        def visit_FunctionDef(self, node):
            self.assigned.add(node.name)
            self.generic_visit(node)
            
        def visit_ClassDef(self, node):
            self.assigned.add(node.name)
            self.generic_visit(node)
            
            
    visitor = VariableVisitor()
    visitor.visit(tree)
    
    # Common implicit constants/globals in Colab/Jupyter/DS that might not be declared clearly in ast top-level
    allowed = {'drive', 'print', 'len', 'range', 'enumerate', 'zip', 'max', 'min', 'float', 'int', 'str', 'list', 'dict', 'set', 'tuple', 'bool', 'isinstance', 'type', 'dir', 'hasattr', 'getattr', 'setattr', 'open', 'sum', 'round', 'abs', 'all', 'any'}
    allowed.update(['Model', 'keras', 'Input', 'layers', 'tf', 'np', 'pd', 'plt', 'sns', 'gc', 'xgb', 'lgb', 'RandomForestClassifier', 'LogisticRegression', 'CatBoostClassifier', 'focal_loss_tf', 'EarlyStopping', 'Adam'])
    allowed.update(['warnings', 'train_test_split', 'StandardScaler', 'MinMaxScaler', 'LabelEncoder', 'ROC', 'AUC', 'PCA', 'TargetEncoder'])
    allowed.update(['sys', 'os', 'warnings'])
    
    undefined = []
    for var, lineno in visitor.loaded:
        if var not in visitor.assigned and var not in allowed:
            # Maybe it's defined inside a function argument? 
            # AST is too complex to track scope perfectly, so we'll just log
            undefined.append((var, lineno))
            
    if undefined:
        print(f"⚠️ POTENTIALLY UNDEFINED VARIABLES:")
        shown = set()
        for var, lineno in sorted(undefined, key=lambda x: x[1]):
            if var not in shown:
                print(f"   Line {lineno} (Cell {line_to_cell.get(lineno, '?')}): '{var}'")
                shown.add(var)
    else:
        print("✅ Variable Reference Scan: PASSED")
