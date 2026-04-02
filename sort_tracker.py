import sys, io, openpyxl

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

wb = openpyxl.load_workbook('tracker_fraud_v29_final.xlsx')
ws = wb['TASK_BOARD']

tasks = []
for row_idx, val in enumerate(ws.iter_rows(min_row=7, max_row=45)):
    if not val[0].value:
        continue
    
    tasks.append({
        'id': val[0].value,         # A (1)
        'week': val[1].value,       # B (2)
        'phase': val[2].value,      # C (3)
        'stream': val[3].value,     # D (4)
        'task': val[4].value,       # E (5)
        'deliverable': val[5].value,# F (6)
        'owner': val[6].value,      # G (7)
        'support': val[7].value,    # H (8)
        'reviewer': val[8].value,   # I (9)
        'start_plan': val[9].value, # J (10)
        'due_plan': val[10].value,  # K (11)
        'start_act': val[11].value, # L (12)
        'done_act': val[12].value,  # M (13)
        'priority': val[13].value,  # N (14)
        'pct_comp': val[14].value,  # O (15)
        'last_upd': val[15].value,  # P (16)
        'blocked': val[16].value,   # Q (17)
        # R (18) is Status formula
        'dependency': val[18].value,# S (19)
        'next_action': val[19].value,# T (20)
        'notes': val[20].value       # U (21)
    })

# Sort by Week first, then Phase
tasks.sort(key=lambda x: (x['week'], x['phase']))

id_mapping = {}
for i, t in enumerate(tasks):
    new_id = f"T{i+1:02d}"
    id_mapping[t['id']] = new_id
    t['new_id'] = new_id

def map_deps(dep_str, mapping):
    if not dep_str: return dep_str
    deps = str(dep_str).split(',')
    res = [mapping.get(d.strip(), d.strip()) for d in deps]
    return ','.join(res)

style_row = ws[8]

for i, t in enumerate(tasks):
    r = i + 7
    ws.cell(r, 1).value = t['new_id']
    ws.cell(r, 2).value = t['week']
    ws.cell(r, 3).value = t['phase']
    ws.cell(r, 4).value = t['stream']
    ws.cell(r, 5).value = t['task']
    ws.cell(r, 6).value = t['deliverable']
    ws.cell(r, 7).value = t['owner']
    ws.cell(r, 8).value = t['support']
    ws.cell(r, 9).value = t['reviewer']
    ws.cell(r, 10).value = t['start_plan']
    ws.cell(r, 11).value = t['due_plan']
    ws.cell(r, 12).value = t['start_act']
    ws.cell(r, 13).value = t['done_act']
    ws.cell(r, 14).value = t['priority'] if t['priority'] else 'Cao'
    ws.cell(r, 15).value = t['pct_comp'] if t['pct_comp'] is not None else 0
    ws.cell(r, 16).value = t['last_upd']
    ws.cell(r, 17).value = t['blocked'] if t['blocked'] else 'Không'

    # Formula Status
    ws.cell(r, 18).value = f'=IF($A{r}="","",IF($Q{r}="Có","Blocked",IF(OR($O{r}>=1,$M{r}<>""),"Hoàn thành",IF(AND($K{r}<CONFIG!$B$5,$A{r}<>""),"Trễ hạn",IF(AND($K{r}-CONFIG!$B$5<=2,$K{r}>=CONFIG!$B$5),"Sắp đến hạn",IF(OR($L{r}<>"",$O{r}>0),"Đang thực hiện","Chưa bắt đầu"))))))'
    
    ws.cell(r, 19).value = map_deps(t['dependency'], id_mapping)
    ws.cell(r, 20).value = t['next_action']
    ws.cell(r, 21).value = t['notes']

    for c in range(1, 23):
        src = style_row[c-1]
        tgt = ws.cell(r, c)
        if src.has_style:
            tgt.font = openpyxl.styles.Font(name=src.font.name, bold=src.font.bold, color=src.font.color)
            tgt.border = openpyxl.styles.Border(left=src.border.left, right=src.border.right, top=src.border.top, bottom=src.border.bottom)
            tgt.alignment = openpyxl.styles.Alignment(horizontal=src.alignment.horizontal, vertical=src.alignment.vertical)
            tgt.fill = openpyxl.styles.PatternFill(fill_type=src.fill.fill_type, start_color=src.fill.start_color, end_color=src.fill.end_color)

for r in range(len(tasks) + 7, ws.max_row + 1):
    for c in range(1, 23):
        ws.cell(r, c).value = None

# Update formulas in Deliverables sheet to point to the correct Task ID
ws_del = wb['DELIVERABLES']
for row_idx in range(7, 20):
    linked_task = ws_del.cell(row_idx, 5).value  # Col E is Linked Task ID? Wait, Col 5 in screenshot is Linked Task ID.
    if linked_task and linked_task in id_mapping:
        ws_del.cell(row_idx, 5).value = id_mapping[linked_task]


wb.save('tracker_fraud_v29_final.xlsx')
print("✅ Re-sort and formula fix success!")
