import os
import cv2
import yaml
from pathlib import Path
from tqdm import tqdm

def load_config(config_path="configs/paths.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def extract_video(video_path, output_folder):
    """Trích xuất frame từ 1 video cụ thể"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Lỗi: Không thể mở video {video_path}")
        return 

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = video_path.stem 
    
    video_out_dir = Path(output_folder) / video_name
    video_out_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    with tqdm(total=total_frames, desc=f"Đang xử lý {video_name}", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break # Hết video
            
            # Đặt tên file ảnh: frame_0000.jpg, frame_0001.jpg...
            frame_filename = video_out_dir / f"frame_{frame_count:04d}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            
            frame_count += 1
            pbar.update(1)
            
    cap.release()

def main():
    print("🚀 Bắt đầu quá trình trích xuất dữ liệu...")
    config = load_config()
    
    # Duyệt qua từng tập dữ liệu (ucsd, avenue) được định nghĩa trong config
    for dataset_name, raw_path in config['data']['raw'].items():
        processed_path = config['data']['processed'][dataset_name]
        
        print(f"\n📁 Đang kiểm tra tập dữ liệu: {dataset_name.upper()}")
        raw_dir = Path(raw_path)
        
        if not raw_dir.exists():
            print(f"⚠️ Không tìm thấy thư mục {raw_dir}. Vui lòng kiểm tra lại!")
            continue
            
        # Tìm tất cả các file video (.avi, .mp4) trong thư mục raw
        video_files = list(raw_dir.rglob("*.avi")) + list(raw_dir.rglob("*.mp4"))
        
        if not video_files:
            print(f"⚠️ Không có file video nào trong {raw_dir}.")
            continue
            
        print(f"✅ Tìm thấy {len(video_files)} video. Bắt đầu cắt frame...")
        for video_file in video_files:
            extract_video(video_file, processed_path)
            
    print("\n🎉 Hoàn tất tiền xử lý dữ liệu!")

if __name__ == "__main__":
    main()