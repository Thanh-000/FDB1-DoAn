import sys, io, openpyxl

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

wb = openpyxl.load_workbook('tracker_fraud_v29_final.xlsx')
ws = wb['TASK_BOARD']

print("Row 8 formulas:")
for col in range(12, 23):
    cell = ws.cell(row=8, column=col)
    print(f"Col {col} ({cell.coordinate}): value='{cell.value}', data_type='{cell.data_type}'")
