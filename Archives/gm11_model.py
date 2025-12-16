"""
Grey Model GM(1,1) for Time Series Forecasting
Mô hình xám GM(1,1) để dự báo chuỗi thời gian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List

class GM11Model:
    """
    Grey Model GM(1,1) - Mô hình xám bậc 1 với 1 biến
    
    GM(1,1) là mô hình dự báo phù hợp với dữ liệu có xu hướng tăng/giảm đều
    và có ít dữ liệu lịch sử (thường >=4 điểm)
    """
    
    def __init__(self):
        """
        Khởi tạo mô hình GM(1,1)
        """
        self.a = None  # Hệ số phát triển
        self.b = None  # Hệ số xám
        self.x0 = None  # Dãy số gốc
        self.x1 = None  # Dãy số tích lũy
        self.fitted_values = None  # Giá trị dự báo trên tập train
        
    def _cumulative_sum(self, X: np.ndarray) -> np.ndarray:
        """
        Tính tổng tích lũy (AGO - Accumulated Generating Operation)
        
        Parameters:
        -----------
        X : np.ndarray
            Dãy số gốc X^(0)
            
        Returns:
        --------
        np.ndarray: Dãy số tích lũy X^(1)
        """
        return np.cumsum(X)
    
    def _mean_generating(self, X1: np.ndarray) -> np.ndarray:
        """
        Tính dãy trung bình liền kề (Mean Generating)
        
        Parameters:
        -----------
        X1 : np.ndarray
            Dãy số tích lũy X^(1)
            
        Returns:
        --------
        np.ndarray: Dãy Z^(1)
        """
        Z = np.zeros(len(X1) - 1)
        for i in range(len(Z)):
            Z[i] = 0.5 * (X1[i] + X1[i + 1])
        return Z
    
    def fit(self, X: np.ndarray) -> 'GM11Model':
        """
        Huấn luyện mô hình GM(1,1)
        
        Các bước:
        1. Tính dãy tích lũy X^(1)
        2. Tạo dãy trung bình Z^(1)
        3. Giải phương trình vi phân để tìm a, b
        4. Tính giá trị fitted
        
        Parameters:
        -----------
        X : np.ndarray
            Dãy số gốc, shape (n,)
            
        Returns:
        --------
        self: GM11Model
        """
        # Chuyển về dạng 1D array
        X = np.asarray(X).flatten()
        
        if len(X) < 4:
            raise ValueError("GM(1,1) yêu cầu ít nhất 4 điểm dữ liệu")
        
        # Lưu dãy gốc
        self.x0 = X.copy()
        
        # Bước 1: Tính tổng tích lũy (AGO)
        self.x1 = self._cumulative_sum(X)
        
        # Bước 2: Tạo dãy trung bình liền kề Z^(1)
        Z = self._mean_generating(self.x1)
        
        # Bước 3: Xây dựng ma trận B và vector Y
        # Phương trình: X^(0)(k) + a*Z^(1)(k) = b
        n = len(X)
        B = np.zeros((n - 1, 2))
        Y = X[1:].reshape(-1, 1)
        
        for i in range(n - 1):
            B[i, 0] = -Z[i]
            B[i, 1] = 1.0
        
        # Bước 4: Giải bằng phương pháp bình phương tối thiểu
        # [a, b]^T = (B^T * B)^(-1) * B^T * Y
        params = np.linalg.lstsq(B, Y, rcond=None)[0]
        self.a = params[0, 0]
        self.b = params[1, 0]
        
        # Bước 5: Tính giá trị fitted
        self.fitted_values = self._calculate_fitted()
        
        return self
    
    def _calculate_fitted(self) -> np.ndarray:
        """
        Tính giá trị fitted cho tập train
        
        Returns:
        --------
        np.ndarray: Giá trị dự báo cho tập train
        """
        n = len(self.x0)
        x1_hat = np.zeros(n)
        x0_hat = np.zeros(n)
        
        # Giá trị đầu tiên giữ nguyên
        x1_hat[0] = self.x0[0]
        x0_hat[0] = self.x0[0]
        
        # Tính các giá trị tiếp theo theo công thức:
        # X^(1)(k+1) = [X^(0)(1) - b/a] * e^(-a*k) + b/a
        for k in range(1, n):
            x1_hat[k] = (self.x0[0] - self.b / self.a) * np.exp(-self.a * k) + self.b / self.a
            x0_hat[k] = x1_hat[k] - x1_hat[k - 1]
        
        return x0_hat
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Dự báo cho steps bước tiếp theo
        
        Parameters:
        -----------
        steps : int
            Số bước cần dự báo
            
        Returns:
        --------
        np.ndarray: Giá trị dự báo
        """
        if self.a is None or self.b is None:
            raise ValueError("Mô hình chưa được train. Gọi fit() trước.")
        
        n = len(self.x0)
        predictions = np.zeros(steps)
        
        # Dự báo dựa trên công thức GM(1,1)
        for k in range(steps):
            k_actual = n + k
            x1_pred = (self.x0[0] - self.b / self.a) * np.exp(-self.a * k_actual) + self.b / self.a
            
            if k == 0:
                x1_prev = self.x1[-1]
            else:
                x1_prev = x1_pred_prev
                
            predictions[k] = x1_pred - x1_prev
            x1_pred_prev = x1_pred
        
        return predictions
    
    def forecast(self, steps: int = 1) -> Tuple[np.ndarray, dict]:
        """
        Dự báo và trả về kèm thông tin chi tiết
        
        Parameters:
        -----------
        steps : int
            Số bước cần dự báo
            
        Returns:
        --------
        predictions : np.ndarray
            Giá trị dự báo
        info : dict
            Thông tin về mô hình (a, b, fitted_values)
        """
        predictions = self.predict(steps)
        
        info = {
            'a': self.a,
            'b': self.b,
            'development_coefficient': self.a,
            'grey_input': self.b,
            'fitted_values': self.fitted_values,
            'original_data': self.x0
        }
        
        return predictions, info
    
    def evaluate(self, X_true: np.ndarray) -> dict:
        """
        Đánh giá độ chính xác của mô hình
        
        Parameters:
        -----------
        X_true : np.ndarray
            Giá trị thực tế
            
        Returns:
        --------
        dict: Các chỉ số đánh giá (MAPE, RMSE, MAE)
        """
        X_true = np.asarray(X_true).flatten()
        X_pred = self.fitted_values
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((X_true - X_pred) / X_true)) * 100
        
        # RMSE (Root Mean Square Error)
        rmse = np.sqrt(np.mean((X_true - X_pred) ** 2))
        
        # MAE (Mean Absolute Error)
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
    
    def plot_results(self, X_test: np.ndarray = None, save_path: str = None):
        """
        Vẽ biểu đồ kết quả fitted và dự báo
        
        Parameters:
        -----------
        X_test : np.ndarray
            Dữ liệu test thực tế (optional)
        save_path : str
            Đường dẫn lưu hình
        """
        if self.fitted_values is None:
            raise ValueError("Mô hình chưa được train.")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot fitted values
        n = len(self.x0)
        ax.plot(range(n), self.x0, 'o-', label='Du lieu goc', linewidth=2, markersize=6)
        ax.plot(range(n), self.fitted_values, 's--', label='Gia tri fitted', linewidth=2, markersize=5)
        
        # Plot predictions nếu có test data
        if X_test is not None:
            X_test = np.asarray(X_test).flatten()
            predictions = self.predict(len(X_test))
            test_range = range(n, n + len(X_test))
            
            ax.plot(test_range, X_test, 'o-', label='Du lieu test', linewidth=2, markersize=6, color='green')
            ax.plot(test_range, predictions, 's--', label='Du bao GM(1,1)', linewidth=2, markersize=5, color='red')
        
        ax.set_xlabel('Thoi gian', fontsize=12)
        ax.set_ylabel('Gia tri', fontsize=12)
        ax.set_title('Ket qua mo hinh GM(1,1)', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Da luu bieu do tai: {save_path}")
        
        plt.show()
    
    def get_model_info(self) -> str:
        """
        Trả về thông tin chi tiết về mô hình
        
        Returns:
        --------
        str: Chuỗi mô tả mô hình
        """
        if self.a is None:
            return "Mo hinh chua duoc train"
        
        info = f"""
        ==============================================
        THONG TIN MO HINH GM(1,1)
        ==============================================
        He so phat trien (a): {self.a:.6f}
        He so xam (b): {self.b:.6f}
        
        Phuong trinh vi phan:
        dx^(1)/dt + a*x^(1) = b
        
        Phuong trinh du bao:
        x^(1)(k+1) = [x^(0)(1) - b/a] * e^(-a*k) + b/a
        
        So diem du lieu train: {len(self.x0)}
        ==============================================
        """
        return info


def demo_gm11():
    """
    Hàm demo sử dụng GM(1,1)
    """
    print("DEMO MO HINH GM(1,1)")
    print("="*60)
    
    # Dữ liệu mẫu - giả sử là giá Bitcoin trong 10 ngày
    data = np.array([45000, 46500, 48000, 47500, 49000, 50500, 51000, 52000, 51500, 53000])
    
    print(f"\nDu lieu goc: {data}")
    print(f"So diem: {len(data)}")
    
    # Chia train/test (80-20)
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    print(f"\nTap train: {train_data}")
    print(f"Tap test: {test_data}")
    
    # Train mô hình
    model = GM11Model()
    model.fit(train_data)
    
    # Hiển thị thông tin mô hình
    print(model.get_model_info())
    
    # Đánh giá trên tập train
    metrics = model.evaluate(train_data)
    print("\nDANH GIA TREN TAP TRAIN:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Dự báo
    predictions, info = model.forecast(steps=len(test_data))
    print(f"\nDu bao {len(test_data)} buoc tiep theo:")
    print(predictions)
    
    # Đánh giá dự báo
    print("\nSo sanh du bao vs thuc te:")
    for i, (pred, true) in enumerate(zip(predictions, test_data)):
        error = abs(pred - true) / true * 100
        print(f"Buoc {i+1}: Du bao={pred:.2f}, Thuc te={true:.2f}, Sai so={error:.2f}%")
    
    # Vẽ biểu đồ
    model.plot_results(test_data, save_path='gm11_demo.png')
    
    return model, predictions


if __name__ == "__main__":
    model, predictions = demo_gm11()
