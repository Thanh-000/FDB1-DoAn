import openpyxl

wb = openpyxl.load_workbook('tracker_fraud_v29_final.xlsx')
ws = wb['TASK_BOARD']

for col in range(1, 23):
    cell = ws.cell(row=8, column=col)
    print(f"Col {col} ({cell.coordinate}): value='{cell.value}', data_type='{cell.data_type}'")
