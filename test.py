import numpy as np

def calculate_metrics_strictly(y_true, y_pred, threshold=0.5):
    """
    y_true: Mặt nạ gốc từ Dataset (chứa giá trị 0 và 1)
    y_pred: Mặt nạ dự đoán từ mô hình (đã qua hàm Sigmoid, giá trị từ 0 đến 1)
    """
    # Áp dụng ngưỡng (threshold) để đưa về dạng nhị phân 0 và 1
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    y_true_bin = y_true.astype(np.float32)

    # Làm phẳng mảng (Flatten) để đếm pixel toàn bộ tập dữ liệu (Micro-average)
    y_pred_f = y_pred_bin.flatten()
    y_true_f = y_true_bin.flatten()
    
    # Tính toán TP, FP, FN
    intersection = np.sum(y_true_f * y_pred_f)
    sum_pred = np.sum(y_pred_f)
    sum_true = np.sum(y_true_f)
    union = sum_pred + sum_true - intersection
    
    # Tính công thức chuẩn
    smooth = 1e-6 # Chống lỗi chia cho 0
    dice = (2. * intersection + smooth) / (sum_pred + sum_true + smooth)
    iou = (intersection + smooth) / (union + smooth)
    
    # Kiểm chứng độ khớp công thức toán học (In ra màn hình để tự check)
    math_check_dice = (2 * iou) / (1 + iou)
    
    print(f"=== KẾT QUẢ KIỂM THỬ ĐỘC LẬP ===")
    print(f"Dice Score thực tế:  {dice * 100:.2f}%")
    print(f"IoU Score thực tế:   {iou * 100:.2f}%")
    print(f"[Check Toán học] Dice tính ngược từ IoU: {math_check_dice * 100:.2f}%")
    
    return dice, iou
