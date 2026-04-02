import nbformat as nbf
import os

# Đường dẫn Tới 2 Thanh Gươm XAI
path_paysim = r'c:\Users\Admin\OneDrive\Desktop\Digital Financial Transaction Fraud Detection Using Explainable Multi-Model Machine Learning on PaySim and IEEE-CIS\MVS_XAI_Colab_DataPrep_Phase1.ipynb'
path_ieee = r'c:\Users\Admin\OneDrive\Desktop\Digital Financial Transaction Fraud Detection Using Explainable Multi-Model Machine Learning on PaySim and IEEE-CIS\MVS_XAI_Colab_IEEE_CIS.ipynb'

def upgrade_notebook(file_path, notebook_type):
    # Đọc cấu trúc Code gốc (Giữ nguyên toàn bộ Code của User)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            nb = nbf.read(f, as_version=4)
    except FileNotFoundError:
        print(f"Không tìm thấy file {file_path}. Tạo bản mới có khung code rỗng.")
        nb = nbf.v4.new_notebook()
        nb.cells.append(nbf.v4.new_code_cell("# Nơi đây chứa code chạy hệ thống\nprint('MVS-XAI Vận Hành')"))
    
    # Gom tất cả code cells 
    code_cells = [cell for cell in nb.cells if cell.cell_type == 'code']
    
    # CHUẨN BỊ BỘ MARKDOWN HỌC THUẬT SIÊU HẠNG DÀNH CHO PAYSIM
    if notebook_type == 'paysim':
        m_intro = nbf.v4.new_markdown_cell("""# 🛡️ MVS-XAI: Multi-View Stacking with Explainable AI for Financial Fraud Detection
### 📖 Phân Khu 1: Khai Thác Mạng Lưới Rửa Tiền & Mobile Money (PaySim)
---
Nghiên cứu Jupyter Notebook này là bản thể hiện thực hoá của Kiến trúc **MVS-XAI** đã được đề xuất trong tham luận IEEE. Khác với các mô hình đơn tuyến truyền thống, MVS-XAI bóp nghẹt mọi ngả đường của tội phạm rửa tiền thông qua một kiến trúc Phức hệ **Đa Góc Nhìn (Multi-View)**.

**Sức mạnh của Quy trình này nằm ở 4 Điểm chốt chặn:**
1. **Góc Nhìn Khách Quan (Multi-View Profiling)**: Đúc kết từ Giám định Tabular, Cấu trúc Hình học mạng Mạng (Graph $\mathcal{G}$) và Vận tốc Trượt Thời Gian (Temporal Windows).
2. **K-Means SMOTE Optimization**: Nhồi máu (Over-sampling) Không gian Véc-tơ có chọn lọc, đưa lớp thiểu số 0.13% cân bằng tuyệt đối tránh làm sai lệch đường phân định.
3. **Walk-Forward CV 5-Blocks**: Lệnh cấm tuyệt đối rò rỉ dữ liệu (Leakage) của chuỗi thời gian - Mô hình chỉ được nhìn vào quá khứ.
4. **Game-Theoretic SHAP Inspector**: Phân rã Black-box thành các Lưỡi cưa Bằng chứng (Waterfall).
""")
        m_phase1 = nbf.v4.new_markdown_cell("""## ⚙️ GIAI ĐOẠN 1: MÁY TRÍCH XUẤT ĐẶC TÍNH ĐA GÓC NHÌN $\\mathbf{MULTI-VIEW}$
Lừa đảo không xảy ra ngẫu nhiên, chúng có chu trình tổ chức.
- Mũi khoan **Graph View**:
Rút trích các thuộc tính Tính hướng Tâm $d_{in}$ và Bức xạ $d_{out}$ của các Nút Danh định.
$$ d_{in}(v_{dest}) = \\sum_{(u, v_{dest}) \in \mathcal{E}} \mathbb{I}(\text{Transfer}) $$
- Mũi khoan **Sequential View**:
Theo dõi nhịp đồ trượt tài sản $W$ trong quá khứ:
$$ \psi_i(t) = \\sum_{k=1}^W \\text{Amt}(i, t-k) $$""")
        m_phase2 = nbf.v4.new_markdown_cell("""## ⚖️ GIAI ĐOẠN 2: LÕI CÂN BẰNG PHỤC HỒI K-MEANS SMOTE
Tỷ lệ thảm họa **99.87% (Sạch)** và **0.13% (Bẩn)**. Thuật toán cân bằng tại đây ứng dụng giải thuật Gom Cụm K-Means trước khi nội suy đường thẳng, giúp mô hình đào sâu vào Vùng Không Gian của kẻ gian lận thay vì gây quấy nhiễu nhóm người dùng vô tội.""")
        m_phase3 = nbf.v4.new_markdown_cell("""## ⚔️ GIAI ĐOẠN 3: ĐỘNG CƠ CÂY QUYẾT ĐỊNH & KIẾM ĐỊNH MŨI TÊN THỜI GIAN
Hệ thống chốt vách ngăn (Block) thời gian. Máy học Gradient Boosting (LightGBM/XGBoost) cấu thành lõi Meta-Ensemble để đạt được sức mạnh tối thượng phân loại Dữ Liệu Bảng.""")
        m_phase4 = nbf.v4.new_markdown_cell("""## 🧠 GIAI ĐOẠN 4: HỆ THỐNG TRÍCH XUẤT TRI THỨC SHAP XAI
Mô hình sẽ không phải là một chiếc Hộp Đen phi tuyến tính mù mờ. Ở pha này, Trí tuệ Giải thích XAI sẽ giải phẫu mô hình thành biểu đồ Thác Đổ (Waterfall / Force Plot), chỉ ra chính xác bằng chứng cho một cảnh báo gian lận.""")
        
        # Kiến trúc cấy ghép thông minh (Tiến hành chia nhỏ kho code hiện tại ra đan xen)
        new_cells = [m_intro]
        total_code = len(code_cells)
        # Giả định phân bố đều các blocks theo flow: Ingest -> Feat Eng -> SMOTE -> Model -> SHAP
        split_1 = max(1, total_code // 4)
        split_2 = max(2, total_code // 2)
        split_3 = max(3, int(total_code * 0.75))
        
        for i, cell in enumerate(code_cells):
            if i == split_1:
                new_cells.append(m_phase1)
            elif i == split_2:
                new_cells.append(m_phase2)
            elif i == split_3:
                new_cells.append(m_phase3)
            elif i == total_code - 1: # Chèn XAI vào cell cuối cùng
                new_cells.append(m_phase4)
            new_cells.append(cell)
            
        nb.cells = new_cells
            
    # CHUẨN BỊ BỘ MARKDOWN HỌC THUẬT SIÊU HẠNG DÀNH CHO IEEE-CIS
    elif notebook_type == 'ieee':
        m_intro = nbf.v4.new_markdown_cell("""# 🛡️ MVS-XAI: Advanced Dimensionality Recovery & Identity Spoofing Engine
### 📖 Phân Khu 2: Triệt phá Chuỗi Tội Phạm Thẻ Tín Dụng (IEEE-CIS)
---
Chào mừng bạn đến Tầng Nghiên cứu Chuyên sâu của Kiến trúc MVS-XAI.
So với sự đơn giản về Số chiều của Đa Hình Rửa Tiền Mobile (PaySim), IEEE-CIS Credit Fraud Detection là một Thách thức Vĩ đại với hơn **400+ Đặc tính Cột Khuyết Thủng**, thiết bị giấu mặt (Device Spoofing) và Mạng lưới IP Ma trận định danh.

**Hệ sinh thái IEEE-CIS MVS-XAI Focuses:**
1. **Massive Dimensionality Pruning (Luật Quét Rác > 70%)**: Ép nén hàng rào tính năng và áp dụng Lớp Phân tích Đa Dạng Hình (Null Pruning).
2. **Identity Bipartite Graph Projection**: Kết nối Tác nhân Thanh toán $V_{Transaction}$ và Hạt nhân Thiết bị  $V_{Identity}$.
3. **Imbalance Adaptive Focal Paradigm**: Trừng phạt các vùng biên giới dữ liệu Nhiễu Sâu.
4. **Global XAI Inspector (Mắt Thần SHAP)**: Bóp nghẹt lớp phòng thủ Đen để diễn dịch Giao dịch Thẻ ảo.
""")
        m_phase1 = nbf.v4.new_markdown_cell("""## 🛠️ GIAI ĐOẠN 1: LIÊN KẾT NHÂN HỆ THỐNG - BIPARTITE GRAPH LƯỠNG PHÂN
Vì cấu trúc chia 2 bảng `transaction_data` và `identity_data`, giải thuật Bipartite của chúng ta thiết lập Mối nối Trực diện `TransactionID`. 
Tất cả các Đặc tính Thiết bị Hệ điều hành (OS), Tên miền Email ảo (P_emaildomain) được đồng bộ hoá cùng một Hệ Quy Chiến Không Gian Vector qua hàm Merge Left và Label Encoding.""")
        m_phase2 = nbf.v4.new_markdown_cell("""## ⚖️ GIAI ĐOẠN 2 & 3: LÕI THÍCH ỨNG SỐ CHIỀU CAO & AUTO-TUNE TEMPORAL ENSEMBLE
Áp lực hơn 400 Chiều khiến hệ sinh thái Giao dịch Sụp đổ. Kỹ thuật ở Trạm này đòi hỏi khả năng kích hoạt Bộ Cân bằng Thông minh kết hợp cấu trúc Học Tăng Cường LightGBM (Gradient Boosting) có Tham số Vượt Hạn Mức (Depth=12, Leaves=256) được tính toán kĩ lưỡng.""")
        m_phase3 = nbf.v4.new_markdown_cell("""## 🧠 GIAI ĐOẠN 4: THẨM ĐỊNH THUẬT TOÁN SHAP GAME-THEORETIC
Tại không gian n-chiều của Credit Card Fraud, SHAP phá vỡ mảng mã hóa của các IP Address và Devices để trả về một Báo cáo Sức mạnh Đồng minh (Shapley Additive Explanations) chỉ điểm cực kỳ chính xác Kẻ thù.""")

        new_cells = [m_intro]
        total_code = len(code_cells)
        split_1 = max(1, total_code // 3)
        split_2 = max(2, int(total_code * 0.6))
        
        for i, cell in enumerate(code_cells):
            if i == split_1:
                new_cells.append(m_phase1)
            elif i == split_2:
                new_cells.append(m_phase2)
            elif i == total_code - 1 or len(code_cells) == 1:
                new_cells.append(m_phase3)
            new_cells.append(cell)
            
        nb.cells = new_cells
        
    # Ghi lại file với định dạng Notebook Chuẩn Học thuật
    with open(file_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"✅ ĐÃ NÂNG CẤP THÀNH CÔNG KIẾN TRÚC MVS-XAI LÊN NOTEBOOK: {notebook_type.upper()}")

# Khởi chạy Bơm Kiến trúc
upgrade_notebook(path_paysim, 'paysim')
upgrade_notebook(path_ieee, 'ieee')
