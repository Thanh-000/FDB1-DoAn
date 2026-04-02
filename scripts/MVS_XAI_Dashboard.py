import streamlit as st
import pandas as pd
import numpy as np
import shap
import time
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Cấu hình UI Dashboard tổng quát
st.set_page_config(page_title="MVS-XAI Command Center", layout="wide", page_icon="🛡️")

# Custom CSS cho giao diện Tối Cao - Hacker / Cyberpunk
st.markdown("""
<style>
    .fraud-alert {
        padding: 20px;
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        animation: pulse 1s infinite;
    }
    .safe-alert {
        padding: 20px;
        background-color: #00cc66;
        color: white;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .big-font {
        font-size: 20px !important;
        font-weight: 500;
    }
    .section-title {
        color: #00aaff;
        border-bottom: 2px solid #00aaff;
        padding-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# THEME THỐNG NHẤT
sns.set_theme(style="darkgrid", palette="muted")

st.title("🛡️ TỔNG TƯ LỆNH MVS-XAI: HỆ THỐNG PHÁT HIỆN GIAN LẬN ĐA CHIỀU (IEEE & PAYSIM)")
st.markdown("Hệ thống Mô phỏng Nghiên cứu Đỉnh cao - Kết hợp Mạng Đồ thị (Graph), Chuỗi Thời gian (Sequential) & Trí tuệ Giải thích (SHAP)")
st.markdown("---")

# HỆ THỐNG TABS CHO 4 KIẾN TRÚC LÕI THEO PROPOSAL
tab1, tab2, tab3, tab4 = st.tabs([
    "🏛️ 1. Kiến Trúc Hệ Thống (Architecture)", 
    "🕸️ 2. Kỹ Nghệ Đa Góc Nhìn (Multi-View)", 
    "⚖️ 3. Lõi Cân Bằng (K-Means SMOTE)",
    "🚨 4. Trinh Sát Suy Luận (Real-time XAI)"
])

# ================== TAB 1: KIẾN TRÚC HỆ THỐNG ==================
with tab1:
    st.markdown('<h2 class="section-title">Khung Tham Chiếu Kiến Trúc MVS-XAI</h2>', unsafe_allow_html=True)
    st.markdown("""
    Giải pháp **MVS-XAI** (Multi-View Stacking with Explainable AI) phá vỡ những hạn chế của các hệ thống Phát hiện Gian lận truyền thống bằng cách từ chối việc chỉ nhìn vào dòng tiền cô lập. Chúng tôi hợp nhất **3 góc nhìn (Views)** về dữ liệu để nắm bắt toàn cảnh một vụ rửa tiền hoặc khống thẻ tín dụng.
    """)
    
    colA, colB, colC = st.columns(3)
    
    with colA:
        st.info("📊 **View 1: Tabular Profiling**\n\nKiểm định giá trị lệch tài khoản, mã thiết bị, và email danh tính. (Tập trung vào tính toàn vẹn dư nợ gốc).")
    with colB:
        st.warning("🕸️ **View 2: Graph Topology**\n\nNắm bắt 'Mạng lưới Nhện'. Nếu một thiết bị kết nối đến hàng chục tài khoản nhận tiền lạ, nó sẽ bị lộ diện qua in/out-degree.")
    with colC:
        st.success("🕒 **View 3: Sequential Velocity**\n\nTheo dõi 'Vận tốc luân chuyển'. Chuỗi trượt thời gian đo được tần suất gửi tiền dồn dập trong 24h.")
    
    st.markdown("---")
    st.markdown("### 🧬 Quy Trình Luân Chuyển Lõi Lọc Đa Tầng")
    
    # Biển diễn Pipeline bằng Mã hóa Markdown Process
    st.markdown("""
    1. **Data Ingestion Buffer**: PaySim (Mobile Money) & IEEE-CIS (Credit Card).
    2. **Multi-View Feature Factory**: Xây dựng Ma trận Không gian Đặc trưng $\\mathbb{R}^{d_{tab} + d_{graph} + d_{seq}}$.
    3. **Walk-Forward Temporal CV**: Chia bộ chặn thời gian 5-Blocks để ngăn chặn rò rỉ dòng tiền Tương lai về Quá khứ.
    4. **Structural Imbalance Engine**: Kích hoạt **K-Means SMOTE** để nội suy các vụ lừa đảo trong cụm thiểu số mà không gây nhiễu biên.
    5. **Ensemble Predictor**: Sử dụng LightGBM & Gradient Boosting tối ưu độ sâu cây phân nhánh.
    6. **Game-Theoretic SHAP Inspector**: Xuất xưởng và Diễn dịch giá trị Shapley công khai.
    """)

# ================== TAB 2: KỸ NGHỆ ĐA GÓC NHÌN ==================
with tab2:
    st.markdown('<h2 class="section-title">Trực quan hoá Góc Nhìn Mạng Lưới (Graph View)</h2>', unsafe_allow_html=True)
    st.write("Khái niệm cơ sở: Các tài khoản lừa đảo rải rác thường tập kết dòng tiền về một 'Tài khoản Đích / Thiết Bị Trung Tâm' (Mule Account). Thuật toán Graph trích xuất Node Centrality để bóc trần điều này.")
    
    col2_1, col2_2 = st.columns([1.5, 1])
    
    with col2_1:
        st.write("**Mô phỏng Đồ thị Bipartite / Mạng Tiêu Chuẩn của Mule Accounts**")
        # Vẽ Đồ thị Cụm Lừa đảo
        fig_graph, ax_graph = plt.subplots(figsize=(8, 6))
        G = nx.DiGraph()
        # Thêm node fraud
        frauds = ['C102', 'C103', 'C104', 'C105', 'C106']
        mule = 'MULE_DEST_99X'
        normals = ['N1', 'N2', 'N3']
        
        G.add_node(mule, color='red', size=800)
        for f in frauds:
            G.add_node(f, color='orange', size=300)
            G.add_edge(f, mule, weight=5)
        for n in normals:
            G.add_node(n, color='green', size=300)
            G.add_edge(n, normals[(normals.index(n)+1)%len(normals)], weight=1)

        colors = [node[1]['color'] for node in G.nodes(data=True)]
        sizes = [node[1]['size'] for node in G.nodes(data=True)]
        
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, ax=ax_graph)
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, ax=ax_graph)
        nx.draw_networkx_labels(G, pos, font_size=9, font_color='white', ax=ax_graph)
        fig_graph.patch.set_facecolor('#1e1e1e')
        ax_graph.set_facecolor('#1e1e1e')
        st.pyplot(fig_graph)

    with col2_2:
        st.write("**Góc Nhìn Chuỗi (Sequential View)**")
        st.markdown("""
        > **Cơ sở Toán học:**
        > Tốc độ tích lũy gian lận $\\psi$ được tính qua Sliding Window:
        > $$ \\psi_i(t) = \\sum_{k=1}^{W} Amt(i, t-k) $$
        """)
        seq_data = pd.DataFrame({
            "Thời Gian (Giờ)": [1, 2, 3, 4, 5, 6],
            "Cumulative Amount ($)": [500, 1000, 10500, 40000, 150000, 320000]
        })
        st.line_chart(seq_data.set_index("Thời Gian (Giờ)"))
        st.caption("Biểu đồ vọt biến của $\\psi$ khi chu kỳ lừa đảo cất lưới.")

# ================== TAB 3: K-MEANS SMOTE ==================
with tab3:
    st.markdown('<h2 class="section-title">Phục Hồi Sụp Đổ Cấu Trúc (Imbalance Recovery)</h2>', unsafe_allow_html=True)
    st.write("Tỷ lệ phân tầng Gốc của PaySim là 99.87% (Normal) và 0.13% (Fraud). Trọng số Weight-Balancing thông thường gây nhiễu False Positive. Tại đây chúng ta chọc thủng rào cản đó bằng thuật toán không gian véc-tơ K-Means SMOTE.")
    
    col3_1, col3_2 = st.columns(2)
    
    with col3_1:
        st.write("📉 **Giai đoạn 1: Trước Phục Hồi (Tự Tự nhiên)**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.bar(["Sạch (Normal)", "Bẩn (Fraud)"], [1509658, 1674], color=['#00cc66', '#ff4b4b'])
        ax1.set_title("Vực Thẳm Imbalance Lõi Train")
        ax1.set_yscale("log")
        fig1.patch.set_facecolor('#1e1e1e')
        ax1.set_facecolor('#1e1e1e')
        ax1.tick_params(colors='white')
        ax1.title.set_color('white')
        st.pyplot(fig1)

    with col3_2:
        st.write("📈 **Giai đoạn 2: Quần thể Nhân tạo bằng K-Means SMOTE**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(["Sạch (Normal)", "Nhân Tạo Bẩn (SMOTE Fraud)"], [1509658, 1509659], color=['#00cc66', '#ffaa00'])
        ax2.set_title("Cán cân Cân bằng Tuyệt đối (1:1)")
        fig2.patch.set_facecolor('#1e1e1e')
        ax2.set_facecolor('#1e1e1e')
        ax2.tick_params(colors='white')
        ax2.title.set_color('white')
        st.pyplot(fig2)

# ================== TAB 4: HỆ THỐNG XAI REAL-TIME ==================
with tab4:
    st.markdown('<h2 class="section-title">Động Cơ Kiểm Soát Rủi Ro MVS-XAI (Live Inference)</h2>', unsafe_allow_html=True)
    
    # 1. GENERATE MOCK MODEL NGAY LÚC RUN
    @st.cache_resource
    def load_mock_system():
        from sklearn.ensemble import RandomForestClassifier
        mock_X = pd.DataFrame(np.random.rand(200, 10), columns=[
            'amount', 'oldbalanceOrg', 'newbalanceOrig', 'errorBalanceOrg', 
            'oldbalanceDest', 'newbalanceDest', 'errorBalanceDest', 
            'dest_in_degree', 'seq_cum_amount', 'seq_tx_count'
        ])
        mock_y = np.random.randint(0, 2, 200)
        # Bơm trọng số ảo cho giống thật
        mock_X.loc[mock_y == 1, 'dest_in_degree'] += 150 
        mock_X.loc[mock_y == 1, 'seq_cum_amount'] += 500000 
        mock_X.loc[mock_y == 1, 'amount'] += 800000

        model = RandomForestClassifier(n_estimators=15, max_depth=10, random_state=42)
        model.fit(mock_X, mock_y)
        explainer = shap.TreeExplainer(model)
        return model, explainer, mock_X.columns

    model, explainer, feature_names = load_mock_system()

    # 2. CHỌN LOẠI MÔ PHỎNG TỪ SIDEBAR Ở PHÂN KHU 4
    st.sidebar.markdown("---")
    st.sidebar.header("📡 Điều Khiển Real-time Suy Luận")
    tx_id = st.sidebar.text_input("Gán Mã Bám Đuôi Tracking ID:", value="SYS-IEEE-9002")
    demo_type = st.sidebar.radio("Nạp Hạt Nhân Payload Truy Vết:", ("Giao Dịch Sạch Tuyệt Đối", "Nghi Ngờ Rửa Tiền Đa Điểm"))
    
    if st.sidebar.button("🔥 PHÁT ĐỘNG KIỂM DUYỆT"):
        with st.spinner('Trích xuất Đa chiều & Ép mạng Không Gian...'):
            time.sleep(1.5)
            
            # Khởi tạo ma trận nạp
            if demo_type == "Nghi Ngờ Rửa Tiền Đa Điểm":
                tx_data = pd.DataFrame([[890000, 890000, 0, 0, 10000, 900000, 0, 214, 8500000, 12]], columns=feature_names)
            else:
                tx_data = pd.DataFrame([[2000, 5500, 3500, 0, 15000, 17000, 0, 1, 2000, 1]], columns=feature_names)
            
            # Predict
            prob_fraud = model.predict_proba(tx_data)[0][1]
            
            # HIỂN THỊ ALERT
            if prob_fraud > 0.6:
                st.markdown(f'<div class="fraud-alert">🚨 HỆ THỐNG GAI GÓC BÁO ĐỎ: {tx_id} <br> KHÓA BĂNG TÀI KHOẢN (ĐỘ LỆCH CHUẨN: {prob_fraud*100:.1f}%)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="safe-alert">✅ SẠCH GIAO DỊCH: {tx_id} <br> BỎ QUA KIỂM DUYỆT (TÍN NHIỆM: {(1-prob_fraud)*100:.1f}%)</div>', unsafe_allow_html=True)
                
            st.markdown("---")
            
            # INFOGRAPHIC CHI TIẾT GIAO DỊCH
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.markdown("### 🔍 Phóng to Cột Tính Năng Tabular")
                st.dataframe(tx_data[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'errorBalanceOrg']].T, use_container_width=True)
            with t_col2:
                st.markdown("### 🕸️ Phóng to Mạng Lưới Đa Chiều")
                st.dataframe(tx_data[['dest_in_degree', 'seq_cum_amount', 'seq_tx_count']].T, use_container_width=True)
                
            st.markdown("---")
            st.markdown("### 🧠 Báo cáo Khẩu Y Giải Thích AI (SHAP Waterfall/Force Plot)")
            st.markdown("*Cớ sao Hệ thống AI lại đưa ra kết luận kinh khủng trên? Các yếu tố Màu Đỏ kéo lùi tài khoản vào Rửa tiền, trong khi Màu Xanh Lơ bênh vực nó.*")
            
            # XAI PLOT
            shap_values = explainer.shap_values(tx_data)
            
            # Lấy vị trí 1 cho Positive Class
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
                base_val = explainer.expected_value[1]
            else:
                sv = shap_values[0]
                if isinstance(explainer.expected_value, np.ndarray):
                    base_val = explainer.expected_value[1]
                else:
                    base_val = explainer.expected_value
                    
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            shap.plots._waterfall.waterfall_legacy(base_val, sv, feature_names=feature_names, show=False)
            fig3.patch.set_facecolor('#1e1e1e')
            # Lấy màu trắng cho chữ
            for child in ax3.get_children():
                if isinstance(child, plt.Text):
                    child.set_color('white')
            
            st.pyplot(fig3)
    else:
        st.info("👈 Trinh sát chưa nạp lệnh. Hãy đến Menu điều hướng tại Sidebar!")
