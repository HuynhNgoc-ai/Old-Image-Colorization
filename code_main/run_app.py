import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Lớp khởi tạo
class ImagePredictionApp:
    def __init__(self, master, model_path):
        self.master = master
        self.master.title("Ứng dụng phục chế màu ảnh cũ")
        self.master.geometry("550x500")

        # Tải mô hình đã được huấn luyện
        self.model = load_model(model_path)

        # Nhãn để hiển thị các hình ảnh
        self.original_image_label = tk.Label(master, text="Ảnh gốc", font=("Arial", 14))
        self.original_image_label.pack()
        self.original_image_canvas = tk.Label(master)
        self.original_image_canvas.pack()

        self.predicted_image_label = tk.Label(master, text="Ảnh dự đoán màu", font=("Arial", 14))
        self.predicted_image_label.pack()
        self.predicted_image_canvas = tk.Label(master)
        self.predicted_image_canvas.pack()

        # Nút để nhập một hình ảnh
        self.import_button = tk.Button(master, text="Tải ảnh lên", command=self.import_image, font=("Arial", 14))
        self.import_button.pack()

    def import_image(self):
        # Mở hộp thoại để chọn một hình ảnh
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not file_path:
            return

        # Tải ảnh bằng thư viện PIL
        original_image = Image.open(file_path).convert("RGB")
        original_image_resized = original_image.resize((128, 128))  #Thay đổi kích thước về 128x128

        # Xử lý ảnh gốc
        original_display = original_image.resize((200, 200)) 
        original_photo = ImageTk.PhotoImage(original_display)
        self.original_image_canvas.config(image=original_photo)
        self.original_image_canvas.image = original_photo

        # Chuyển đổi ảnh sang grayscale
        gray_image = original_image_resized.convert("L")  # Chuyển đổi sang ảnh đen trắng
        gray_array = np.expand_dims(np.array(gray_image) / 255.0, axis=(0, -1))  # Chuẩn hóa dữ liệu

        # Dự đoán ảnh tô màu
        predicted_image = self.model.predict(gray_array)[0]  
        predicted_image = np.clip(predicted_image * 255, 0, 255).astype(np.uint8)  #Chuyển đổi về khoảng [0, 255] và giới hạn giá trị

        # Hiển thị kết quả
        predicted_image_pil = Image.fromarray(predicted_image)
        predicted_display = predicted_image_pil.resize((200, 200))  # Thay đổi kích thước để hiển thị trên giao diện
        predicted_photo = ImageTk.PhotoImage(predicted_display)
        self.predicted_image_canvas.config(image=predicted_photo)
        self.predicted_image_canvas.image = predicted_photo

        # Hiển thị 2 hình và so sánh bằng Matplotlib
        self.show_comparison(original_image, predicted_image_pil)

    def show_comparison(self, original_image, predicted_image):
        plt.figure(figsize=(7, 3))
        plt.subplot(1, 2, 1)
        plt.title("Ảnh gốc")
        plt.imshow(original_image)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.title("Ảnh dự đoán màu")
        plt.imshow(predicted_image)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

# main
if __name__ == "__main__":
    model_path = r"output/autoencoder128.keras" 
    root = tk.Tk()
    app = ImagePredictionApp(root, model_path)
    root.mainloop()
