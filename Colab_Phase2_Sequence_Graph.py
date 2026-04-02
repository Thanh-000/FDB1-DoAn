import pandas as pd
import numpy as np
import networkx as nx

# --- CELL 1: GRAPH VIEW ENGINEERING ---
print("===== 1. XÂY DỰNG GRAPH VIEW (EGO-NETWORK) =====")
print("Khởi tạo cấu trúc đồ thị luồng tiền...")
# Sử dụng NetworkX phân tích node luồng tiền có hướng
G = nx.from_pandas_edgelist(paysim_df, source='nameOrig', target='nameDest', 
                            edge_attr=['amount', 'step'], create_using=nx.DiGraph())

print(f"Tổng số Node (Tài khoản): {G.number_of_nodes():,}")
print(f"Tổng số Edge (Giao dịch): {G.number_of_edges():,}")

print("\nTrích xuất đặc trưng Graph (Level 1: Centrality)...")
out_degree = dict(G.out_degree())
in_degree = dict(G.in_degree())

# Đưa đặc trưng Graph ngược vào Tabular Data
paysim_df['orig_out_degree'] = paysim_df['nameOrig'].map(out_degree).astype(np.int32)
paysim_df['dest_in_degree'] = paysim_df['nameDest'].map(in_degree).astype(np.int32)
print("Graph Engineering Hoàn Tất!\n")

# --- CELL 2: SEQUENTIAL VIEW ENGINEERING ---
print("===== 2. XÂY DỰNG SEQUENTIAL VIEW =====")
print("Trích xuất tính định danh chuỗi giao dịch liên tiếp...")
# Bắt buộc Sort theo Tên Account rồi mới tới Thời gian
paysim_df.sort_values(by=['nameOrig', 'step'], ascending=[True, True], inplace=True)

# Lịch sử tần suất và cộng dồn ví tiền theo chuỗi
paysim_df['seq_tx_count'] = paysim_df.groupby('nameOrig').cumcount() + 1
paysim_df['seq_cum_amount'] = paysim_df.groupby('nameOrig')['amount'].cumsum().astype(np.float32)

# Reset lại đúng Time Wall (Sort lại theo mốc thời gian tuyệt đối toàn cục)
paysim_df.sort_values(by=['step'], ascending=True, inplace=True)
paysim_df.reset_index(drop=True, inplace=True)
print("Sequential Engineering Hoàn Tất!\n")

# --- CELL 3: TEMPORAL WALK-FORWARD SPLITTING ---
print("===== 3. PHÂN CHIA 5-BLOCK TEMPORAL WALK-FORWARD =====")
total_steps = paysim_df['step'].max()
# Cắt 743 giờ (khoảng 30 ngày) thành 5 khối thời gian nghiêm ngặt
step_thresholds = np.linspace(0, total_steps, 6)

def assign_fold(step):
    for i in range(1, 6):
        if step <= step_thresholds[i]:
            return i
    return 5

paysim_df['cv_fold'] = paysim_df['step'].apply(assign_fold).astype(np.int8)

print("Kích thước dữ liệu tại 5 Block thời gian (Tuyệt đối không rò rỉ tương lai):")
print(paysim_df['cv_fold'].value_counts().sort_index())
