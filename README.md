# README - Xử lý Dữ Liệu và Huấn Luyện Mô Hình với PySpark

## Mô tả
Đây là một dự án xử lý dữ liệu và huấn luyện mô hình Machine Learning sử dụng **PySpark**. Chương trình thực hiện các bước tiền xử lý dữ liệu trên tập dữ liệu Zillow Prize, sau đó áp dụng các thuật toán hồi quy để dự đoán giá trị bất động sản.

## Yêu cầu hệ thống
Trước khi chạy chương trình, hãy đảm bảo rằng bạn đã cài đặt:
- Python 3.8 trở lên
- Apache Spark (>= 3.0)
- Thư viện PySpark
- Thư viện hỗ trợ trực quan hóa: Matplotlib, Seaborn
- Bộ dữ liệu từ link: https://www.kaggle.com/competitions/zillow-prize-1

Cài đặt các thư viện cần thiết bằng lệnh:
```bash
pip install pyspark matplotlib seaborn pandas numpy
```

## Cấu trúc dự án
- `f4.py`: File chứa toàn bộ quy trình xử lý dữ liệu và huấn luyện mô hình.
- `zillow-prize-1/`: Thư mục chứa dữ liệu đầu vào (CSV từ Zillow Prize).
- `README.md`: Tài liệu hướng dẫn sử dụng.

## Các bước thực hiện

### 1. Đọc dữ liệu
- Nạp dữ liệu từ các tệp CSV.
- Hợp nhất các tập dữ liệu năm 2016 và 2017 thành một tập huấn luyện chung.

### 2. Tiền xử lý dữ liệu
- **Xử lý giá trị thiếu**: Dùng **Imputer** để thay thế giá trị NaN bằng trung vị.
- **Xử lý ngoại lệ**: Sử dụng phương pháp **IQR** để loại bỏ giá trị bất thường.
- **Chuyển đổi biến phân loại**: Sử dụng **StringIndexer** để chuyển đổi các cột phân loại thành số.
- **Tạo đặc trưng mới**:
  - Đặc trưng tương tác giữa các cột số.
  - Đặc trưng đa thức bậc hai bằng **PolynomialExpansion**.
- **Lựa chọn đặc trưng**: Loại bỏ các cột có phương sai thấp.
- **Chuẩn hóa dữ liệu**: Dùng **StandardScaler** để chuẩn hóa các đặc trưng số.

### 3. Huấn luyện mô hình
- Chia tập dữ liệu thành tập huấn luyện (80%) và kiểm tra (20%).
- Áp dụng các mô hình:
  - **RandomForestRegressor**
  - **Gradient Boosting Trees (GBTRegressor)**

### 4. Đánh giá mô hình
Mô hình được đánh giá bằng các chỉ số:
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score** (Coefficient of Determination)

### 5. Trực quan hóa kết quả
- Vẽ biểu đồ histogram sau từng bước tiền xử lý.
- Vẽ scatter plot để so sánh giá trị thực tế và dự đoán.

## Cách chạy chương trình
Chạy chương trình bằng lệnh:
```bash
python f4.py
```

Dữ liệu đầu ra sẽ bao gồm:
- Kết quả đánh giá mô hình.
- Các biểu đồ trực quan hóa quá trình xử lý dữ liệu.

## Tác giả
Dự án này được thực hiện bởi [Vũ Giang Nam]. Nếu có bất kỳ câu hỏi nào, vui lòng liên hệ qua email hoặc GitHub.

