"""
LSTM Model for Time Series Forecasting
Mô hình LSTM để dự báo chuỗi thời gian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

class LSTMForecast:
    """
    Lớp LSTM để dự báo chuỗi thời gian
    
    LSTM (Long Short-Term Memory) là một loại mạng nơ-ron hồi tiếp (RNN)
    phù hợp cho dữ liệu chuỗi thời gian với mối quan hệ dài hạn
    """
    
    def __init__(self, lookback: int = 10, lstm_units: int = 50, dropout_rate: float = 0.2):
        """
        Khởi tạo mô hình LSTM
        
        Parameters:
        -----------
        lookback : int
            Số bước thời gian nhìn lại (window size)
        lstm_units : int
            Số units trong LSTM layer
        dropout_rate : float
            Tỷ lệ dropout để tránh overfitting
        """
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        self.train_data = None
        
    def _create_sequences(self, data: np.ndarray) -> tuple:
        """
        Tạo sequences cho LSTM
        
        Parameters:
        -----------
        data : np.ndarray
            Dữ liệu đã chuẩn hóa
            
        Returns:
        --------
        tuple: (X, y) sequences
        """
        X, y = [], []
        
        for i in range(len(data) - self.lookback):
            X.append(data[i:(i + self.lookback), 0])
            y.append(data[i + self.lookback, 0])
        
        return np.array(X), np.array(y)
    
    def build_model(self, num_layers: int = 2) -> keras.Model:
        """
        Xây dựng kiến trúc mô hình LSTM
        
        Parameters:
        -----------
        num_layers : int
            Số lượng LSTM layers
            
        Returns:
        --------
        keras.Model: Mô hình LSTM
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units=self.lstm_units, return_sequences=(num_layers > 1),
                      input_shape=(self.lookback, 1)))
        model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for i in range(1, num_layers):
            return_seq = (i < num_layers - 1)
            model.add(LSTM(units=self.lstm_units, return_sequences=return_seq))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error', 
                     metrics=['mae'])
        
        return model
    
    def fit(self, data: np.ndarray, epochs: int = 50, batch_size: int = 32, 
            validation_split: float = 0.1, verbose: int = 1) -> 'LSTMForecast':
        """
        Huấn luyện mô hình LSTM
        
        Parameters:
        -----------
        data : np.ndarray
            Dữ liệu train
        epochs : int
            Số epochs
        batch_size : int
            Kích thước batch
        validation_split : float
            Tỷ lệ dữ liệu validation
        verbose : int
            Mức độ hiển thị log
            
        Returns:
        --------
        self: LSTMForecast
        """
        data = np.asarray(data).flatten().reshape(-1, 1)
        self.train_data = data
        
        print(f"\nDang train mo hinh LSTM...")
        print(f"Lookback: {self.lookback}, LSTM units: {self.lstm_units}")
        
        # Chuẩn hóa dữ liệu
        data_scaled = self.scaler.fit_transform(data)
        
        # Tạo sequences
        X, y = self._create_sequences(data_scaled)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        print(f"Shape X: {X.shape}, Shape y: {y.shape}")
        
        # Build model
        self.model = self.build_model()
        
        print("\nKien truc mo hinh:")
        self.model.summary()
        
        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10, 
                                   restore_best_weights=True, verbose=0)
        
        # Train model
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=verbose
        )
        
        print("\nHoan thanh train!")
        
        return self
    
    def predict_on_train(self) -> np.ndarray:
        """
        Dự báo trên tập train
        
        Returns:
        --------
        np.ndarray: Giá trị dự báo
        """
        if self.model is None:
            raise ValueError("Mo hinh chua duoc train. Goi fit() truoc.")
        
        # Chuẩn hóa
        data_scaled = self.scaler.transform(self.train_data)
        
        # Tạo sequences
        X, _ = self._create_sequences(data_scaled)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Predict
        predictions_scaled = self.model.predict(X, verbose=0)
        
        # Inverse transform
        predictions = self.scaler.inverse_transform(predictions_scaled)
        
        return predictions.flatten()
    
    def predict(self, data: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Dự báo cho steps bước tiếp theo
        
        Parameters:
        -----------
        data : np.ndarray
            Dữ liệu lịch sử (ít nhất lookback điểm)
        steps : int
            Số bước cần dự báo
            
        Returns:
        --------
        np.ndarray: Giá trị dự báo
        """
        if self.model is None:
            raise ValueError("Mo hinh chua duoc train. Goi fit() truoc.")
        
        data = np.asarray(data).flatten().reshape(-1, 1)
        
        if len(data) < self.lookback:
            raise ValueError(f"Can it nhat {self.lookback} diem du lieu")
        
        # Lấy lookback điểm cuối
        last_sequence = data[-self.lookback:]
        
        # Chuẩn hóa
        last_sequence_scaled = self.scaler.transform(last_sequence)
        
        predictions = []
        current_sequence = last_sequence_scaled.copy()
        
        for _ in range(steps):
            # Reshape cho LSTM
            X = current_sequence.reshape((1, self.lookback, 1))
            
            # Predict
            pred_scaled = self.model.predict(X, verbose=0)
            
            # Inverse transform
            pred = self.scaler.inverse_transform(pred_scaled)
            predictions.append(pred[0, 0])
            
            # Update sequence cho lần predict tiếp theo
            current_sequence = np.append(current_sequence[1:], pred_scaled, axis=0)
        
        return np.array(predictions)
    
    def evaluate(self, X_true: np.ndarray, X_pred: np.ndarray) -> dict:
        """
        Đánh giá độ chính xác
        
        Parameters:
        -----------
        X_true : np.ndarray
            Giá trị thực tế
        X_pred : np.ndarray
            Giá trị dự báo
            
        Returns:
        --------
        dict: Các chỉ số đánh giá
        """
        X_true = np.asarray(X_true).flatten()
        X_pred = np.asarray(X_pred).flatten()
        
        # Đảm bảo cùng độ dài
        min_len = min(len(X_true), len(X_pred))
        X_true = X_true[:min_len]
        X_pred = X_pred[:min_len]
        
        # MAPE
        mape = np.mean(np.abs((X_true - X_pred) / X_true)) * 100
        
        # RMSE
        rmse = np.sqrt(np.mean((X_true - X_pred) ** 2))
        
        # MAE
        mae = np.mean(np.abs(X_true - X_pred))
        
        # R-squared
        ss_res = np.sum((X_true - X_pred) ** 2)
        ss_tot = np.sum((X_true - np.mean(X_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return {
            'MAPE': mape,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    
    def plot_training_history(self, save_path: str = None):
        """
        Vẽ biểu đồ loss trong quá trình training
        
        Parameters:
        -----------
        save_path : str
            Đường dẫn lưu hình
        """
        if self.history is None:
            raise ValueError("Mo hinh chua duoc train.")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss (MSE)', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(loc='best', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # MAE
        axes[1].plot(self.history.history['mae'], label='Train MAE', linewidth=2)
        axes[1].plot(self.history.history['val_mae'], label='Val MAE', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
        axes[1].legend(loc='best', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Da luu bieu do tai: {save_path}")
        
        plt.show()
    
    def plot_results(self, X_test: np.ndarray = None, predictions: np.ndarray = None, 
                    save_path: str = None):
        """
        Vẽ biểu đồ kết quả
        
        Parameters:
        -----------
        X_test : np.ndarray
            Dữ liệu test thực tế
        predictions : np.ndarray
            Giá trị dự báo
        save_path : str
            Đường dẫn lưu hình
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot training data
        train_data_flat = self.train_data.flatten()
        ax.plot(range(len(train_data_flat)), train_data_flat, 
               'o-', label='Du lieu train', linewidth=2, markersize=4, alpha=0.7)
        
        # Plot fitted values
        fitted = self.predict_on_train()
        fitted_range = range(self.lookback, self.lookback + len(fitted))
        ax.plot(fitted_range, fitted, 's--', label='Gia tri fitted', 
               linewidth=2, markersize=4, alpha=0.7)
        
        # Plot test data và predictions
        if X_test is not None and predictions is not None:
            X_test = np.asarray(X_test).flatten()
            predictions = np.asarray(predictions).flatten()
            
            n = len(train_data_flat)
            test_range = range(n, n + len(X_test))
            
            ax.plot(test_range, X_test, 'o-', label='Du lieu test', 
                   linewidth=2, markersize=6, color='green')
            ax.plot(test_range, predictions, 's--', label='Du bao LSTM', 
                   linewidth=2, markersize=5, color='red')
        
        ax.set_xlabel('Thoi gian', fontsize=12)
        ax.set_ylabel('Gia tri', fontsize=12)
        ax.set_title('Ket qua mo hinh LSTM', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Da luu bieu do tai: {save_path}")
        
        plt.show()
    
    def get_model_info(self) -> str:
        """
        Trả về thông tin về mô hình
        
        Returns:
        --------
        str: Chuỗi mô tả mô hình
        """
        if self.model is None:
            return "Mo hinh chua duoc train"
        
        info = f"""
        ==============================================
        THONG TIN MO HINH LSTM
        ==============================================
        Lookback (window size): {self.lookback}
        LSTM units: {self.lstm_units}
        Dropout rate: {self.dropout_rate}
        
        So diem du lieu train: {len(self.train_data)}
        Tong so tham so: {self.model.count_params()}
        ==============================================
        """
        return info


def demo_lstm():
    """
    Hàm demo sử dụng LSTM
    """
    print("DEMO MO HINH LSTM")
    print("="*60)
    
    # Tạo dữ liệu mẫu - sine wave với noise
    np.random.seed(42)
    t = np.linspace(0, 100, 200)
    data = np.sin(t * 0.1) * 50 + 100 + np.random.randn(200) * 5
    
    print(f"\nSo diem du lieu: {len(data)}")
    
    # Chia train/test
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    print(f"Tap train: {len(train_data)} diem")
    print(f"Tap test: {len(test_data)} diem")
    
    # Khởi tạo mô hình
    model = LSTMForecast(lookback=10, lstm_units=50, dropout_rate=0.2)
    
    # Train mô hình
    model.fit(train_data, epochs=100, batch_size=16, validation_split=0.1, verbose=0)
    
    # Hiển thị thông tin
    print(model.get_model_info())
    
    # Vẽ training history
    model.plot_training_history(save_path='lstm_training_history.png')
    
    # Đánh giá trên tập train
    fitted = model.predict_on_train()
    train_metrics = model.evaluate(train_data[model.lookback:], fitted)
    
    print("\nDANH GIA TREN TAP TRAIN:")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Dự báo
    predictions = model.predict(train_data, steps=len(test_data))
    
    print(f"\nDu bao {len(test_data)} buoc tiep theo:")
    print(predictions)
    
    # Đánh giá dự báo
    test_metrics = model.evaluate(test_data, predictions)
    print("\nDANH GIA TREN TAP TEST:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # So sánh
    print("\nSo sanh du bao vs thuc te:")
    for i, (pred, true) in enumerate(zip(predictions[:10], test_data[:10])):
        error = abs(pred - true) / abs(true) * 100
        print(f"Buoc {i+1}: Du bao={pred:.2f}, Thuc te={true:.2f}, Sai so={error:.2f}%")
    
    # Vẽ biểu đồ
    model.plot_results(test_data, predictions, save_path='lstm_results.png')
    
    return model, predictions


if __name__ == "__main__":
    model, predictions = demo_lstm()
