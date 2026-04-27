# Anomaly Surveillance System 🚨

Hệ thống phát hiện hành vi bất thường trong video giám sát sử dụng **YOLOv8 + Tracking + Deep Learning (ConvLSTM / AutoEncoder)**.

---

## 🎯 Mục tiêu

Xây dựng pipeline tự động:

Video → Detection → Tracking → Phân tích chuỗi hành vi → Cảnh báo bất thường

Ứng dụng:

* Camera an ninh
* Smart city
* Giám sát hành vi bất thường
* Computer Vision research

---

## 🧠 Công nghệ sử dụng

* Python
* OpenCV
* PyTorch
* YOLOv8 (Ultralytics)
* DeepSORT / BoTSORT
* ConvLSTM / AutoEncoder
* Numpy, Matplotlib

---

## 📂 Cấu trúc thư mục

```
anomaly-surveillance-system/
├── data/                       
│   ├── raw/                    
│   ├── processed/              
│   └── annotations/            
│
├── configs/                    
│   ├── yolov8_config.yaml      
│   ├── convlstm_config.yaml    
│   └── paths.yaml              
│
├── notebooks/                  
│   ├── 01_data_exploration.ipynb  
│   └── 02_test_yolo_tracking.ipynb
│
├── src/                        
│   ├── __init__.py
│   ├── data/                   
│   │   ├── extract_frames.py   
│   │   └── dataset.py          
│
│   ├── models/                 
│   │   ├── autoencoder.py      
│   │   └── convlstm.py         
│
│   ├── tracking/               
│   │   ├── deep_sort.py        
│   │   └── test_yolo.py
│
│   └── utils/                  
│       ├── metrics.py          
│       └── visualization.py    
│
├── weights/                    
│   ├── yolov8n.pt              
│   └── best_autoencoder.pth    
│
├── .gitignore                  
├── requirements.txt            
├── README.md                   
└── main.py                     
```


---

## ⚙️ Cài đặt môi trường

### 1. Clone project

```
git clone <link-github>
cd anomaly-surveillance-system
```

### 2. Tạo môi trường ảo

```
python -m venv venv
venv\Scripts\activate
```

### 3. Cài thư viện

```
pip install -r requirements.txt
```

---

## ▶️ Chạy chương trình

```
python main.py
```

---

## 🔄 Workflow hệ thống

```
Video Input
     ↓
Extract Frames (OpenCV)
     ↓
YOLOv8 Detection
     ↓
Tracking (DeepSORT / BoTSORT)
     ↓
Sequence Creation
     ↓
ConvLSTM / AutoEncoder
     ↓
Anomaly Score
     ↓
Hiển thị cảnh báo
```

---

## ✅ Tiến độ hiện tại

### Đã làm được

* Tiền xử lý dữ liệu video sang frame bằng `src/data/extract_frames.py` (hỗ trợ `.avi`, `.mp4`)
* Demo phát hiện và theo dõi người với YOLOv8 bằng `src/tracking/test_yolo.py`
* Lưu video kết quả tracking tại `results/predictions/demo_tracking.mp4`
* Xây dựng `AnomalySequenceDataset` trong `src/data/dataset.py` để tạo sequence cho mô hình temporal
* Dataset loader hỗ trợ đọc frame dạng `.jpg` (Avenue) và `.tif` (UCSD)
* Đã có script huấn luyện AutoEncoder: `src/train/train_autoencoder.py`

### Chưa làm / đang phát triển

* Chưa hoàn thiện bước tạo `error map -> threshold -> anomaly mask`
* Chưa có module đánh giá định lượng đầy đủ (AUC, AP, EER, IoU)
* `main.py` chưa phải entrypoint hoàn chỉnh cho toàn bộ pipeline

---

## ☁️ Training trên Google Colab

Do yêu cầu GPU cho Deep Learning, quá trình huấn luyện được thực hiện trên **Google Colab**.

### Quy trình đề xuất

1. Mở Colab với runtime GPU
2. Mount Google Drive để lưu dữ liệu và checkpoint
3. Clone project và cài dependencies (`pip install -r requirements.txt`)
4. Chuẩn bị dữ liệu:
   * Avenue: dùng `src/data/extract_frames.py` để cắt video thành `.jpg`
   * UCSD: sử dụng cấu trúc frame `.tif` theo sequence (đã hỗ trợ trong `dataset.py`)
5. Train mô hình (ConvLSTM/AutoEncoder) trên tập train normal-only
6. Lưu trọng số (`.pth`) về Drive và dùng lại cho suy luận/test

### Gợi ý lệnh cơ bản trên Colab

```bash
!git clone <link-github>
%cd anomaly-surveillance-system
!pip install -r requirements.txt
```

---

## 📊 Dataset

Dataset đề xuất:

* UCSD Pedestrian Dataset
* Avenue Dataset

Đặt dữ liệu tại:

```
data/raw/
```

Cấu trúc khuyến nghị:

```
data/raw/
├── ucsd/
│   ├── UCSDped1/
│   │   ├── train/Train001/*.tif
│   │   └── test/Test001/*.tif
│   └── UCSDped2/
│       ├── train/Train001/*.tif
│       └── test/Test001/*.tif
└── avenue/
    ├── training_videos/*.avi
    └── testing_videos/*.avi
```

---

## 📈 Output

* Video có bounding box
* Tracking ID cho từng người
* Cảnh báo bất thường
* Heatmap khu vực nguy hiểm

---

## ▶️ Cách chạy các thành phần hiện có

### 1) Cài môi trường local (Python 3.12)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Tiền xử lý frame từ video (Avenue)

```bash
python src/data/extract_frames.py
```

### 3) Test YOLOv8 + Tracking

```bash
python src/tracking/test_yolo.py
```

### 4) Test Dataset Loader cho chuỗi thời gian

```bash
python src/data/dataset.py
```

### 5) Train ConvLSTM AutoEncoder

```bash
python src/train/train_autoencoder.py --dataset-path data/raw/ucsd/UCSDped1/train --epochs 20
```

Ví dụ train trên Avenue đã cắt frame:

```bash
python src/train/train_autoencoder.py --dataset-path data/processed/avenue --epochs 20
```

Checkpoint sẽ được lưu tại:

```bash
weights/checkpoints/
```

---

## 👨‍💻 Tác giả

Sinh viên ngành Trí tuệ nhân tạo / Computer Vision

---

## 📜 License

Sử dụng cho mục đích học tập và nghiên cứu.
