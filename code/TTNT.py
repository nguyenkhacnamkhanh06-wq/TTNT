import pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import deque

# ==========================================
# 1. TẬP DỮ LIỆU (20 CUỐN SÁCH - 5 THAM SỐ CHỦ ĐỀ)
# ==========================================
data = {
    'Ten_Sach': [
        'Đắc Nhân Tâm', 'Nhà Giả Kim', 'Tư Duy Nhanh Và Chậm', 'Hiểu Về Trái Tim', 
        'Kinh Tế Vi Mô', 'Quản Trị Chiến Lược', 'Tâm Lý Học Tội Phạm', 'Khắc Kỷ (Stoicism)',
        'Sapiens: Lược Sử Loài Người', 'Lược Sử Thời Gian', 'Dạy Con Làm Giàu', 
        'Nghĩ Giàu Làm Giàu', 'Lối Sống Tối Giản', 'Sức Mạnh Của Thói Quen', 
        'Hành Trình Về Phương Đông', 'Muôn Kiếp Nhân Sinh', 'Tuổi Trẻ Đáng Giá Bao Nhiêu', 
        'Đọc Vị Bất Kỳ Ai', 'Cà Phê Cùng Tony', 'Khởi Nghiệp Tinh Gọn'
    ],
    'Tam_Ly':     [1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0],
    'Triet_Ly':   [0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
    'Kinh_Doanh': [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
    'Hoc_Thuat':  [0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    'Ky_Nang':    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# ==========================================
# 2. KHỞI TẠO VÀ HUẤN LUYỆN MÔ HÌNH KNN
# ==========================================
features = df[['Tam_Ly', 'Triet_Ly', 'Kinh_Doanh', 'Hoc_Thuat', 'Ky_Nang']].values

# Thiết lập k=6 (1 kết quả là chính cuốn sách đó + 5 gợi ý lân cận)
# Metric cosine đo góc giữa các vector thay vì khoảng cách hình học tuyệt đối
knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)
knn_model.fit(features)

# ==========================================
# TẠO ĐỒ THỊ TƯƠNG ĐỒNG TỪ KNN
# ==========================================
graph = {}

for i in range(len(df)):
    distances, indices = knn_model.kneighbors([features[i]])

    neighbors = []

    for j in range(1, len(indices.flatten())):
        idx = indices.flatten()[j]
        dist = distances.flatten()[j]

        # similarity >= 0.6
        if dist <= 0.4:
            neighbors.append(idx)

    graph[i] = neighbors

# ==========================================
# 3. HÀM GỢI Ý DỰA TRÊN KNN
# ==========================================
def KNN_Tim_Sach_Goi_Y(book_index, model, dataframe):
    # Trả về mảng khoảng cách và index của các cuốn sách gần nhất
    distances, indices = model.kneighbors([features[book_index]])
    
    danh_sach_goi_y = []
    
    # Bỏ qua index 0 vì nó chính là cuốn sách đang xét (khoảng cách = 0)
    for i in range(1, len(distances.flatten())):
        idx = indices.flatten()[i]
        dist = distances.flatten()[i]
        
        # Khoảng cách Cosine trong sklearn: dist = 1 - similarity
        # Điều kiện dist <= 0.4 tương đương với similarity >= 0.6 trong code cũ
        if dist <= 0.4: 
            danh_sach_goi_y.append(dataframe.iloc[idx]['Ten_Sach'])
            
    return danh_sach_goi_y

# ==========================================
# BFS DUYỆT SÁCH LIÊN QUAN
# ==========================================
def BFS_Goi_Y(start_index, graph, dataframe, max_depth=2):

    visited = set()
    queue = deque()

    queue.append((start_index, 0))
    visited.add(start_index)

    ket_qua = []

    while queue:
        current, depth = queue.popleft()

        if depth >= max_depth:
            continue

        for neighbor in graph[current]:

            if neighbor not in visited:
                visited.add(neighbor)

                ket_qua.append(
                    dataframe.iloc[neighbor]['Ten_Sach']
                )

                queue.append((neighbor, depth + 1))

    return ket_qua

# ==========================================
# 4. CHẠY THỬ NGHIỆM THỰC TẾ TRÊN CONSOLE
# ==========================================
if __name__ == "__main__":
    print("="*50)
    print("   HỆ THỐNG GỢI Ý SÁCH THÔNG MINH (k-NN ML)")
    print("="*50)
    
    danh_sach_sach = df['Ten_Sach'].tolist()
    print("\nDANH MỤC SÁCH HIỆN CÓ:")
    for i, sach in enumerate(danh_sach_sach):
        print(f"{i + 1:02d}. {sach}")
        
    print("-" * 50)
    
    book_idx_selected = -1
    sach_nguoi_dung_chon = ""
    
    while True:
        lua_chon = input("Nhập số thứ tự (1-20) hoặc tên sách bạn muốn đọc: ").strip()
        
        if lua_chon.isdigit():
            index = int(lua_chon) - 1
            if 0 <= index < len(danh_sach_sach):
                book_idx_selected = index
                sach_nguoi_dung_chon = danh_sach_sach[index]
                break
            else:
                print("[LỖI] Số thứ tự không hợp lệ. Vui lòng nhập lại!")
                
        else:
            if lua_chon in danh_sach_sach:
                book_idx_selected = danh_sach_sach.index(lua_chon)
                sach_nguoi_dung_chon = lua_chon
                break
            else:
                print("[LỖI] Tên sách không tồn tại trong hệ thống. Vui lòng nhập chính xác từng chữ hoa/thường!")

    print(f"\n>>> Bạn đang đọc: [ {sach_nguoi_dung_chon} ]")
    print(">>> Hệ thống đang phân tích không gian vector kNN...\n")
    
    ket_qua = KNN_Tim_Sach_Goi_Y(book_idx_selected, knn_model, df)
    
    print("\n>>> BFS đang mở rộng mạng lưới liên quan...\n")

    bfs_result = BFS_Goi_Y(book_idx_selected, graph, df)

    print("=> GỢI Ý MỞ RỘNG BẰNG BFS:")

    for i, sach in enumerate(bfs_result):
        print(f"   {i + 1}. {sach}")

    if not ket_qua:
        print("=> Rất tiếc, chưa tìm thấy sách nào đạt ngưỡng tương đồng với lựa chọn của bạn.")
    else:
        print("=> CÓ THỂ BẠN SẼ QUAN TÂM:")
        for i, sach in enumerate(ket_qua):
            print(f"   {i + 1}. {sach}")
    print("="*50)
