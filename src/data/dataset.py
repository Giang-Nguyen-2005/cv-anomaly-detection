import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class AnomalySequenceDataset(Dataset):
    """
    Dataset class cho bài toán Phát hiện hành động bất thường (Anomaly Detection).
    Hỗ trợ tạo chuỗi (sequences) từ các frame ảnh.
    """
    def __init__(self, dataset_path, seq_length=10, resize_shape=(128, 128)):
        """
        Args:
            dataset_path (str): Đường dẫn đến thư mục Train/Test (chứa các thư mục con video).
            seq_length (int): Độ dài chuỗi frame (số bước thời gian).
            resize_shape (tuple): Kích thước (H, W) để resize ảnh, giúp giảm tải GPU.
        """
        self.seq_length = seq_length
        self.resize_shape = resize_shape
        self.sequences = []

        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"⚠️ Đường dẫn không tồn tại: {dataset_path}")

        # Quét các thư mục con (mỗi thư mục tương ứng với 1 video/cảnh)
        video_folders = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])

        for folder in video_folders:
            # Lấy danh sách ảnh, hỗ trợ đa định dạng (.jpg cho Avenue, .tif cho UCSD)
            frames = sorted(
                list(folder.glob("*.jpg")) + 
                list(folder.glob("*.png")) + 
                list(folder.glob("*.tif"))
            )
            
            # Trượt cửa sổ (sliding window) để tạo các chuỗi liên tiếp
            if len(frames) >= self.seq_length:
                for i in range(len(frames) - self.seq_length + 1):
                    seq = frames[i : i + self.seq_length]
                    self.sequences.append(seq)
        
        print(f"✅ Dataset initialized: {len(self.sequences)} sequences found.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_paths = self.sequences[idx]
        seq_frames = []

        for frame_path in seq_paths:
            # Đọc ảnh xám (Grayscale) để tiết kiệm VRAM
            img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            
            # Kiểm tra xem ảnh có đọc được không (phòng trường hợp file lỗi)
            if img is None:
                # resize_shape is (width, height) for cv2.resize, but numpy array is (height, width)
                img = np.zeros((self.resize_shape[0], self.resize_shape[1]), dtype=np.uint8)
            
            # Resize ảnh
            img = cv2.resize(img, self.resize_shape)
            
            # Chuẩn hóa (Normalization) về khoảng [0, 1]
            img = img.astype("float32") / 255.0
            seq_frames.append(img)

        # Chuyển đổi tối ưu: list -> numpy -> tensor (Sửa lỗi UserWarning)
        # Shape gốc: (seq_length, height, width)
        tensor_seq = torch.from_numpy(np.array(seq_frames)).float()
        
        # Thêm chiều Channel (C=1) cho ảnh xám: (seq_length, 1, height, width)
        tensor_seq = tensor_seq.unsqueeze(1) 

        return tensor_seq

# ================= ĐOẠN CODE TEST TRÊN COLAB =================
if __name__ == "__main__":
    # Thay đổi path này tùy theo bộ dữ liệu bạn muốn test
    # Lưu ý: Chú ý chữ 'Train' viết hoa nếu dùng UCSD
    TEST_PATH = "/content/drive/MyDrive/raw/ucsd/UCSDped1/Train"
    
    try:
        dataset = AnomalySequenceDataset(
            dataset_path=TEST_PATH, 
            seq_length=10, 
            resize_shape=(128, 128)
        )
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"🔥 Test thành công!")
            print(f"Kích thước 1 mẫu: {sample.shape}")
            print(f"Giá trị Min/Max: {sample.min():.2f} / {sample.max():.2f}")
    except Exception as e:
        print(e)