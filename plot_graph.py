import pandas as pd
import matplotlib.pyplot as plt

# 1. Đường dẫn tới file log 
log_path = 'models/isic2018_NestedUNet_woDS/log.csv'

# 2. Đọc dữ liệu từ file csv
df = pd.read_csv(log_path)

# 3. Tạo khung tranh gồm 2 biểu đồ (Loss và IoU) nằm ngang nhau
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Vẽ biểu đồ 1: Loss (Mức độ sai sót) ---
ax1.plot(df['epoch'], df['loss'], label='Train Loss', color='blue', linewidth=2)
ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', color='red', linewidth=2)
ax1.set_title('Biểu đồ Loss (Càng giảm càng tốt)', fontsize=14)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# --- Vẽ biểu đồ 2: IoU (Độ chính xác) ---
ax2.plot(df['epoch'], df['iou'], label='Train IoU', color='blue', linewidth=2)
ax2.plot(df['epoch'], df['val_iou'], label='Val IoU', color='red', linewidth=2)
ax2.set_title('Biểu đồ IoU (Càng tăng càng tốt)', fontsize=14)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('IoU', fontsize=12)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

# 4. Căn chỉnh cho đẹp và Lưu thành file ảnh
plt.tight_layout()
plt.savefig('bieu_do_huan_luyen.png', dpi=300) # dpi=300 giúp ảnh cực kỳ sắc nét
print("🎉 Đã vẽ xong! Bạn hãy kiểm tra file 'bieu_do_huan_luyen.png' bên cột trái nhé.")