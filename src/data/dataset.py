import cv2
import torch
import yaml
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

def load_config(config_path="configs/convlstm_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

class AnomalySequenceDataset(Dataset):
    def __init__(self, dataset_path, seq_length=10, resize_shape=(256, 256)):
        self.seq_length = seq_length
        self.resize_shape = resize_shape
        self.sequences = []

        # Quét tất cả các thư mục con (các video đã được cắt thành frame)
        dataset_dir = Path(dataset_path)
        video_folders = [d for d in dataset_dir.iterdir() if d.is_dir()]

        for folder in video_folders:
            # Lấy tất cả file .jpg và sắp xếp theo thứ tự thời gian
            frames = sorted(list(folder.glob("*.jpg")))
            
            # Trượt cửa sổ để tạo các chuỗi (sequences)
            # Ví dụ có 20 frame, seq=10 -> tạo được 11 chuỗi
            for i in range(len(frames) - self.seq_length + 1):
                seq = frames[i : i + self.seq_length]
                self.sequences.append(seq)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_paths = self.sequences[idx]
        seq_frames = []

        for frame_path in seq_paths:
            # Đọc ảnh dưới dạng ảnh xám (Grayscale) để giảm tải tính toán cho model
            img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.resize_shape)
            
            # Chuẩn hóa giá trị pixel từ 0-255 về 0.0-1.0
            img = img.astype("float32") / 255.0
            seq_frames.append(img)

        # Chuyển list các numpy array thành PyTorch Tensor
        # Shape hiện tại: (seq_length, height, width)
        tensor_seq = torch.tensor(seq_frames, dtype=torch.float32)
        
        # Pytorch yêu cầu shape phải có channel: (seq_length, channels, height, width)
        # Vì là ảnh xám nên channels = 1
        tensor_seq = tensor_seq.unsqueeze(1) 

        return tensor_seq

# ================= CODE TEST NHANH =================
if __name__ == "__main__":
    print("⏳ Đang test thử Dataset Loader...")
    
    # Giả sử test với tập Avenue
    test_path = "data/processed/avenue" 
    
    if Path(test_path).exists():
        dataset = AnomalySequenceDataset(dataset_path=test_path, seq_length=10)
        print(f"✅ Đã tạo được tổng cộng {len(dataset)} chuỗi hành động (sequences).")
        
        # Lấy thử chuỗi đầu tiên ra xem hình thù
        sample_seq = dataset[0]
        print(f"✅ Kích thước của 1 Tensor (Sequence): {sample_seq.shape}")
        print("   -> Ý nghĩa: (10 frames, 1 kênh màu xám, cao 256, rộng 256)")
    else:
        print(f"⚠️ Không tìm thấy đường dẫn {test_path}")