Nhóm: "Tiếp Theo" ________________
Thành viên: 
- Văn Tiến Hoàng 	- 22520480
- Nguyễn Hoàng Minh 	- 23520937
- Nguyễn Tiến Thành 	- 23521454
- Lê Thành Thắng Đạt 	- 23520251
__________________________________

### Demo Image Deraining System

1. Giới thiệu
- Dự án này là hệ thống demo khử mưa ảnh (Image Deraining), triển khai và so sánh các mô hình học sâu phổ biến gồm:
	+ PReNet
	+ MPRNet
	+ DRT

- Hệ thống được xây dựng dưới dạng ứng dụng Streamlit, cho phép:
	+ Kiểm thử mô hình trên tập ảnh có ground truth
	+ Upload ảnh mưa bất kỳ để suy luận (inference)

2. Cấu trúc thư mục
demo_deraining/
│
├── pretrained_models/     # Chứa trọng số model (.pt, .pth)
├── test_image/            # Ảnh test có ground truth
├── training/              # Code huấn luyện 
├── demo.py                # File chạy Streamlit demo
├── PReNet.py              # Định nghĩa model Preet
├── MPRNet.py              # Định nghĩa model MPRNet
├── DRT.py                 # Định nghĩa model DRT
├── requirements.txt       # Danh sách thư viện
├── .gitignore
└── README.txt

3. Cài đặt môi trường (Build)
3.1. Yêu cầu

- Python >= 3.8
- pip

3.2. Cài đặt thư viện
- Tại thư mục gốc của project, chạy:

pip install -r requirements.txt

4. Tải trọng số mô hình (Model Weights)
- Do giới hạn dung lượng GitHub, các file trọng số .pt / .pth không được lưu trực tiếp trong repo.

4.1. Tải model từ Google Drive
- Tải toàn bộ các file model tại link sau: https://drive.google.com/drive/folders/1JNvZLWa4PMscpx0hg6wb95w9LVOcIOLu?fbclid=IwAR384ZMqBNNqKvDTmaAc4EvRqKQQGQVAj-o5-BifcZ-MUW6D1J4DjUL7Kgw

4.2. Đặt file vào thư mục
- Sau khi tải xong, copy các file .pt / .pth vào: demo_deraining/pretrained_models/

5. Cấu hình đường dẫn model trong demo.py
- Mở file demo.py, tìm biến MODEL_CONFIGS, và thay thế: 

MODEL_CONFIGS = {
    "PReNet": {
	"path":	"pretrained_models/prenet.pth",
	 ...
	}
    "MPRNet": {
	"path":	"pretrained_models/mprnet.pth",
	 ...
	}
    "DRT": {
	"path":	"pretrained_models/drt.pt",
	 ...
	}
}

7. Chạy hệ thống demo (Run)
- Tại thư mục gốc demo_deraining, chạy:
	streamlit run demo.py

- Sau đó mở trình duyệt theo địa chỉ được Streamlit cung cấp (thường là http://localhost:8501).


8. Chức năng của hệ thống Streamlit
- Tab 1: Test trên ảnh có Ground Truth
+ Chọn ảnh trong thư mục test_image
+ So sánh kết quả khử mưa giữa các model
+ Quan sát trực quan chất lượng phục hồi

- Tab 2: Upload ảnh mưa bất kỳ
+ Upload ảnh mưa từ máy cá nhân
+ Chạy suy luận với các model
+ Không yêu cầu ground truth
