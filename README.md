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

⚠️ Lưu ý:

* KHÔNG push thư mục **data/**
* KHÔNG push thư mục **weights/**
* KHÔNG push **venv/**

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

## 📊 Dataset

Dataset đề xuất:

* UCSD Pedestrian Dataset
* Avenue Dataset

Đặt dữ liệu tại:

```
data/raw/
```

---

## 📈 Output

* Video có bounding box
* Tracking ID cho từng người
* Cảnh báo bất thường
* Heatmap khu vực nguy hiểm

---

## 👨‍💻 Tác giả

Sinh viên ngành Trí tuệ nhân tạo / Computer Vision

---

## 📜 License

Sử dụng cho mục đích học tập và nghiên cứu.
