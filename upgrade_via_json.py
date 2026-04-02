import json
import os

path_paysim = r'c:\Users\Admin\OneDrive\Desktop\Digital Financial Transaction Fraud Detection Using Explainable Multi-Model Machine Learning on PaySim and IEEE-CIS\MVS_XAI_Colab_DataPrep_Phase1.ipynb'
path_ieee = r'c:\Users\Admin\OneDrive\Desktop\Digital Financial Transaction Fraud Detection Using Explainable Multi-Model Machine Learning on PaySim and IEEE-CIS\MVS_XAI_Colab_IEEE_CIS.ipynb'

def create_markdown_cell(source_text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_text.split('\n')]
    }

def upgrade_notebook(file_path, nb_type):
    if not os.path.exists(file_path):
        print(f"File {file_path} không tồn tại!")
        return
        
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    code_cells = [c for c in nb.get('cells', []) if c.get('cell_type') == 'code']
    new_cells = []
    
    if nb_type == 'paysim':
        m_intro = create_markdown_cell(
            "# 🛡️ MVS-XAI: Multi-View Stacking with Explainable AI for Financial Fraud Detection\n"
            "### 📖 Phân Khu 1: Khai Thác Mạng Lưới Rửa Tiền & Mobile Money (PaySim)\n"
            "---\n"
            "Nghiên cứu Jupyter Notebook này là bản thể hiện thực hoá của Kiến trúc **MVS-XAI** đã được đề xuất trong tham luận IEEE. "
            "Khác với các mô hình đơn tuyến truyền thống, MVS-XAI bóp nghẹt mọi ngả đường của tội phạm rửa tiền thông qua một kiến trúc Phức hệ **Đa Góc Nhìn (Multi-View)**.\n\n"
            "**Sức mạnh của Quy trình (Kiến Trúc Tổng Thể) nằm ở 4 Điểm chốt chặn:**\n"
            "1. **Giai đoạn 1 (Multi-View Profiling)**: Đúc kết từ Giám định Tabular, Cấu trúc Hình học mạng Mạng (Graph $\mathcal{G}$) và Vận tốc Trượt Thời Gian (Temporal Windows).\n"
            "2. **Giai đoạn 2 (K-Means SMOTE Optimization)**: Nhồi máu Không gian Véc-tơ có chọn lọc, đưa lớp thiện ác về trạng thái 1:1 tránh làm sai lệch đường phân định.\n"
            "3. **Giai đoạn 3 (Walk-Forward CV 5-Blocks)**: Lệnh cấm tuyệt đối rò rỉ dữ liệu (Leakage) của chuỗi thời gian.\n"
            "4. **Giai đoạn 4 (Game-Theoretic SHAP Inspector)**: Phân rã Black-box thành các Lưỡi cưa Bằng chứng (Waterfall)."
        )
        
        m_phase1 = create_markdown_cell(
            "## ⚙️ GIAI ĐOẠN 1: MÁY TRÍCH XUẤT ĐẶC TÍNH ĐA GÓC NHÌN $\mathbf{MULTI-VIEW}$\n"
            "Lừa đảo không xảy ra ngẫu nhiên, chúng có chu trình tổ chức.\n\n"
            "- **Mũi khoan Graph View**: Rút trích Góc độ Tính hướng Tâm $d_{in}$ của các Nút Danh định.\n"
            "  $$ d_{in}(v_{dest}) = \sum_{(u, v_{dest}) \in \mathcal{E}} \mathbb{I}(\\text{Transfer}) $$\n"
            "- **Mũi khoan Sequential View**: Theo dõi nhịp độ luân chuyển tài sản $W$ trong quá khứ:\n"
            "  $$ \psi_i(t) = \sum_{k=1}^W \\text{Amt}(i, t-k) $$"
        )
        
        m_phase2 = create_markdown_cell(
            "## ⚖️ GIAI ĐOẠN 2 & 3: LÕI K-MEANS SMOTE & WALK-FORWARD CV\n"
            "Tỷ lệ thảm họa **99.87% (Sạch)** và **0.13% (Bẩn)**. Thuật toán cân bằng tại đây ứng dụng giải thuật Gom Cụm K-Means trước khi nội suy đường thẳng, giúp mô hình đào sâu vào Vùng Không Gian của kẻ gian lận thay vì gây quấy nhiễu nhóm người dùng vô tội.\n\n"
            "Đồng thời, hệ thống chốt vách ngăn (Block) thời gian để ngăn rò rỉ (Leakage)."
        )
        
        m_phase4 = create_markdown_cell(
            "## 🧠 GIAI ĐOẠN 4: HỆ THỐNG TRÍCH XUẤT TRI THỨC SHAP XAI\n"
            "Mô hình sẽ không phải là một chiếc Hộp Đen phi tuyến tính mù mờ. Ở pha này, Trí tuệ Giải thích XAI sẽ giải phẫu mô hình thành biểu đồ Thác Đổ (Waterfall / Force Plot), chỉ ra chính xác bằng chứng cho một cảnh báo gian lận."
        )
        
        new_cells.append(m_intro)
        total = len(code_cells)
        for i, cell in enumerate(code_cells):
            if i == max(1, total // 4):
                new_cells.append(m_phase1)
            elif i == max(2, total // 2):
                new_cells.append(m_phase2)
            elif i == total - 1 and total > 1:
                new_cells.append(m_phase4)
            new_cells.append(cell)
            
    elif nb_type == 'ieee':
        m_intro = create_markdown_cell(
            "# 🛡️ MVS-XAI: Advanced Dimensionality Recovery & Identity Spoofing Engine\n"
            "### 📖 Phân Khu 2: Triệt phá Chuỗi Tội Phạm Thẻ Tín Dụng (IEEE-CIS)\n"
            "---\n"
            "Chào mừng bạn đến Tầng Nghiên cứu Chuyên sâu của Kiến trúc MVS-XAI.\n"
            "So với sự đơn giản về Số chiều của Đa Hình Rửa Tiền Mobile (PaySim), IEEE-CIS Credit Fraud Detection là một Thách thức Vĩ đại với hơn **400+ Đặc tính Cột Khuyết Thủng**, thiết bị giấu mặt (Device Spoofing) và Mạng lưới IP Ma trận định danh.\n\n"
            "**Hệ sinh thái IEEE-CIS MVS-XAI Focuses:**\n"
            "1. **Massive Dimensionality Pruning** (Luật Quét Rác > 70%): Ép nén hàng rào tính năng và áp dụng Lớp Phân tích Đa Dạng Hình (Null Pruning).\n"
            "2. **Identity Bipartite Graph Projection**: Kết nối Tác nhân Thanh toán $V_{Transaction}$ và Hạt nhân Thiết bị  $V_{Identity}$.\n"
            "3. **Imbalance Adaptive Focal Paradigm**: Trừng phạt các vùng biên giới dữ liệu Nhiễu Sâu.\n"
            "4. **Global XAI Inspector** (Mắt Thần SHAP): Bóp nghẹt lớp phòng thủ Đen để diễn dịch Giao dịch Thẻ ảo."
        )
        
        m_phase1 = create_markdown_cell(
            "## 🛠️ GIAI ĐOẠN 1: BIPARTITE GRAPH LƯỠNG PHÂN & PCA CẮT CHIỀU\n"
            "Vì cấu trúc chia 2 bảng `transaction` và `identity`, giải thuật Bipartite của chúng ta thiết lập Mối nối Trực diện `TransactionID`. \n"
            "Tất cả các Đặc tính Thiết bị Hệ điều hành (OS), Tên miền Email ảo (P_emaildomain) được đồng bộ hoá cùng một Hệ Quy Chiến Không Gian Vector qua hàm Merge Left và Label Encoding."
        )
        
        m_phase2 = create_markdown_cell(
            "## ⚖️ GIAI ĐOẠN 2 & 3: LÕI THÍCH ỨNG SỐ CHIỀU CAO (SMOTE K-MEANS)\n"
            "Áp lực hơn 400 Chiều khiến hệ sinh thái Giao dịch Sụp đổ. Kỹ thuật ở Trạm này đòi hỏi khả năng kích hoạt Bộ Cân bằng Thông minh kết hợp cấu trúc Học Tăng Cường LightGBM (Gradient Boosting) có Tham số Vượt Hạn Mức (Depth=12, Leaves=256) được tính toán kĩ lưỡng."
        )
        
        m_phase4 = create_markdown_cell(
            "## 🧠 GIAI ĐOẠN 4: THẨM ĐỊNH THUẬT TOÁN SHAP GAME-THEORETIC\n"
            "Tại không gian n-chiều của Credit Card Fraud, SHAP phá vỡ mảng mã hóa của các IP Address và Devices để trả về một Báo cáo Sức mạnh Đồng minh (Shapley Additive Explanations) chỉ điểm cực kỳ chính xác Kẻ thù."
        )
        
        new_cells.append(m_intro)
        total = len(code_cells)
        for i, cell in enumerate(code_cells):
            if i == max(1, total // 4):
                new_cells.append(m_phase1)
            elif i == max(2, total // 2):
                new_cells.append(m_phase2)
            elif i == total - 1 and total > 1:
                new_cells.append(m_phase4)
            new_cells.append(cell)
            
    nb['cells'] = new_cells
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    print(f"✅ Đã nâng cấp Kiến trúc cực khủng cho {nb_type}")

upgrade_notebook(path_paysim, 'paysim')
upgrade_notebook(path_ieee, 'ieee')
