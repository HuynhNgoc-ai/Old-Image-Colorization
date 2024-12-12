import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Tải và xử lý dữ liệu
class DataLoader:
    def __init__(self, data_dir, img_size=(128, 128)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.X_train_gray, self.X_train_color = [], []
        self.X_test_gray, self.X_test_color = [], []

    def load_data(self):
        #Tải ảnh
        train_gray_path = os.path.join(self.data_dir, 'train_black')
        train_color_path = os.path.join(self.data_dir, 'train_color')

        gray_images = sorted(os.listdir(train_gray_path))
        color_images = sorted(os.listdir(train_color_path))

        for gray_file, color_file in tqdm(zip(gray_images, color_images), total=len(gray_images), desc="Đang tải ảnh"):
            gray_img = load_img(os.path.join(train_gray_path, gray_file), color_mode="grayscale", target_size=self.img_size)
            gray_img = img_to_array(gray_img) / 255.0
            self.X_train_gray.append(gray_img)

            color_img = load_img(os.path.join(train_color_path, color_file), target_size=self.img_size)
            color_img = img_to_array(color_img) / 255.0
            self.X_train_color.append(color_img)
        
        self.X_train_gray = np.array(self.X_train_gray)
        self.X_train_color = np.array(self.X_train_color)

        (self.X_train_gray, self.X_test_gray, 
         self.X_train_color, self.X_test_color) = train_test_split(self.X_train_gray, self.X_train_color, test_size=0.2, random_state=42)

        print('Kích thước X_train_gray:', self.X_train_gray.shape)
        print('Kích thước X_train_color:', self.X_train_color.shape)
        print('Kích thước X_test_gray:', self.X_test_gray.shape)
        print('Kích thước X_test_color:', self.X_test_color.shape)

# Autoencoder Model
class AutoencoderModel:
    def build_model(self):
        """Mô hình autoencoder với đầu vào 128x128."""
        input_ = tf.keras.layers.Input(shape=(128, 128, 3)) 
        #Chuyển đổi ảnh thành mảng số với kích thước (128x128) và chuẩn hóa [0, 1].
        #Encoder
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input_)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        encoder = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        #Encoder
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(encoder)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        decoder = tf.keras.layers.Conv2D(3, (3, 3), padding='same', activation='sigmoid')(x)
        
        self.model = tf.keras.models.Model(inputs=input_, outputs=decoder)
        self.model.summary()

    def compile_model(self):
        #Biên dịch mô hình
        #Sử dụng Adam và hàm mất mát MSE
        checkpoint_cb = ModelCheckpoint(r"output/autoencoder128.keras", save_best_only=True)
        self.model.compile(optimizer='adam', loss='mse')
        return checkpoint_cb

    def train_model(self, X_train_gray, X_train_color, X_test_gray, X_test_color):
        """Huấn luyện mô hình"""
        checkpoint_cb = self.compile_model()
        history = self.model.fit(
            X_train_gray, X_train_color, 
            epochs=30, 
            validation_data=(X_test_gray, X_test_color), 
            callbacks=[checkpoint_cb]
        )
        # Lưu lịch sử huấn luyện vào file npy
        np.save("training_history.npy", history.history)
        print("Lịch sử huấn luyện đã lưu")

        # Vẽ biểu đồ tổn thất
        self.plot_training_history(history.history)

    def plot_training_history(self, history):
        """Vẽ biểu đồ tổn thất với chỉ số epoch."""
        plt.figure(figsize=(8, 5))
        epochs = range(1, len(history['loss']) + 1)
        plt.plot(epochs, history['loss'], label="Tổn thất huấn luyện")
        if 'val_loss' in history:
            plt.plot(epochs, history['val_loss'], label="Tổn thất kiểm tra")
        plt.xlabel("Số epoch")
        plt.ylabel("Tổn thất (loss)")
        plt.title("Biểu đồ tổn thất qua các epoch")
        plt.legend()
        plt.grid()
        plt.show()

    """def evaluate_model(self, X_test_gray, X_test_color):
         self.model.evaluate(X_test_gray, X_test_color)
        predictions = self.model.predict(X_test_gray)

       ssim_scores = [
         ssim(X_test_color[i], predictions[i], data_range=predictions[i].max() - predictions[i].min(), multichannel=True)
         for i in range(len(X_test_color))
        ]
        print(f"Chỉ số SSIM trung bình: {np.mean(ssim_scores):.4f}")

        return predictions"""

# Evaluation class
class Evaluation:
    @staticmethod
    def display_results(X_test_color, predictions, n=10):
        """Hiển thị ảnh gốc và ảnh dự đoán."""
        plt.figure(figsize=(15, 7))
        for i in range(n):
            # Ảnh gốc
            ax = plt.subplot(3, n, i + 1)
            plt.imshow(X_test_color[i])
            ax.axis("off")

            # Ảnh dự đoán
            ax = plt.subplot(3, n, i + 1 + n)
            plt.imshow(predictions[i])
            ax.axis("off")
        plt.show()

    @staticmethod
    def calculate_psnr(X_test_color, predictions):
        psnr_values = [psnr(orig, pred, data_range=1.0) for orig, pred in zip(X_test_color, predictions)]
        print(f"PSNR trung bình: {np.mean(psnr_values):.4f}")

# Chương trình chính
if __name__ == "__main__":
    data_dir = "data"
    data_loader = DataLoader(data_dir)
    data_loader.load_data()

    autoencoder_model = AutoencoderModel()
    autoencoder_model.build_model()
    autoencoder_model.train_model(data_loader.X_train_gray, data_loader.X_train_color, 
                                  data_loader.X_test_gray, data_loader.X_test_color)

    predictions = autoencoder_model.evaluate_model(data_loader.X_test_gray, data_loader.X_test_color)

    evaluation = Evaluation()
    evaluation.display_results(data_loader.X_test_color, predictions)
    evaluation.calculate_psnr(data_loader.X_test_color, predictions)