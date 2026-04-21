import cv2
from pathlib import Path
from ultralytics import YOLO

def main():
    # 1. Khởi tạo mô hình YOLOv8 (Nó sẽ tự động tải file yolov8n.pt về thư mục gốc)
    print("⏳ Đang tải mô hình YOLOv8 Nano...")
    model = YOLO('yolov8n.pt') 

    # 2. Đường dẫn tới MỘT thư mục chứa ảnh đã cắt (Dũng nhớ thay đổi cho đúng tên thư mục nhé)
    SOURCE_DIR = Path("data/processed/avenue/01") # <--- SỬA CHỖ NÀY NẾU CẦN
    OUTPUT_VIDEO = Path("results/predictions/demo_tracking.mp4")
    
    # Tạo thư mục chứa kết quả nếu chưa có
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    # 3. Đọc và sắp xếp các frame ảnh theo thứ tự
    frames = sorted(list(SOURCE_DIR.glob("*.jpg")))
    if not frames:
        print(f"❌ Không tìm thấy ảnh nào trong {SOURCE_DIR}")
        return

    print(f"✅ Tìm thấy {len(frames)} frames. Bắt đầu tracking...")

    # 4. Setup VideoWriter để lưu thành 1 file video xem cho sướng
    first_frame = cv2.imread(str(frames[0]))
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, 25.0, (width, height))

    # 5. Chạy Tracking qua từng frame
    for frame_path in frames:
        img = cv2.imread(str(frame_path))
        
        # model.track: Vừa phát hiện (detect) vừa theo dõi (gán ID)
        # persist=True: Nhớ ID của người ở frame trước
        # classes=[0]: YOLO có 80 class, số 0 là 'person' (chỉ bắt người, bỏ qua ô tô, chó mèo...)
        results = model.track(img, persist=True, classes=[0], verbose=False)

        # Vẽ bounding box và ID lên ảnh
        annotated_frame = results[0].plot()
        
        # Ghi vào file video kết quả
        out.write(annotated_frame)
        
        # Hiển thị trực tiếp lên màn hình (Nhấn 'q' để thoát sớm)
        cv2.imshow("YOLOv8 Tracking - Nhan 'q' de thoat", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. Dọn dẹp
    out.release()
    cv2.destroyAllWindows()
    print(f"\n🎉 Xong! Hãy mở file {OUTPUT_VIDEO} để xem thành quả nhé!")

if __name__ == "__main__":
    main()