"""
ARIMA Model for Time Series Forecasting
Mô hình ARIMA để dự báo chuỗi thời gian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

class ARIMAForecast:
    """
    Lớp ARIMA để dự báo chuỗi thời gian
    
    ARIMA(p,d,q):
    - p: Bậc của thành phần tự hồi quy (AR - AutoRegressive)
    - d: Bậc sai phân (I - Integrated) 
    - q: Bậc của trung bình trượt (MA - Moving Average)
    """
    
    def __init__(self, order=(1, 1, 1)):
        """
        Khởi tạo mô hình ARIMA
        
        Parameters:
        -----------
        order : tuple
            (p, d, q) - Bậc của ARIMA model
        """
        self.order = order
        self.model = None
        self.fitted_model = None
        self.train_data = None
        self.fitted_values = None
        
    def check_stationarity(self, data: np.ndarray, significance_level: float = 0.05) -> dict:
        """
        Kiểm tra tính dừng của chuỗi thời gian bằng Augmented Dickey-Fuller test
        
        Parameters:
        -----------
        data : np.ndarray
            Dữ liệu cần kiểm tra
        significance_level : float
            Mức ý nghĩa (default: 0.05)
            
        Returns:
        --------
        dict: Kết quả kiểm định
        """
        data = np.asarray(data).flatten()
        
        result = adfuller(data)
        
        is_stationary = result[1] < significance_level
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': is_stationary,
            'conclusion': 'Chuoi dung' if is_stationary else 'Chuoi khong dung'
        }
    
    def plot_diagnostics(self, data: np.ndarray, lags: int = 40, save_path: str = None):
        """
        Vẽ biểu đồ ACF và PACF để xác định p, q
        
        Parameters:
        -----------
        data : np.ndarray
            Dữ liệu
        lags : int
            Số lags để hiển thị
        save_path : str
            Đường dẫn lưu hình
        """
        data = np.asarray(data).flatten()
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # ACF plot
        plot_acf(data, lags=lags, ax=axes[0])
        axes[0].set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Lag', fontsize=10)
        axes[0].set_ylabel('ACF', fontsize=10)
        
        # PACF plot
        plot_pacf(data, lags=lags, ax=axes[1])
        axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Lag', fontsize=10)
        axes[1].set_ylabel('PACF', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Da luu bieu do tai: {save_path}")
        
        plt.show()
    
    def difference_data(self, data: np.ndarray, order: int = 1) -> np.ndarray:
        """
        Sai phân dữ liệu
        
        Parameters:
        -----------
        data : np.ndarray
            Dữ liệu gốc
        order : int
            Bậc sai phân
            
        Returns:
        --------
        np.ndarray: Dữ liệu sau sai phân
        """
        data = np.asarray(data).flatten()
        
        for _ in range(order):
            data = np.diff(data)
        
        return data
    
    def fit(self, data: np.ndarray) -> 'ARIMAForecast':
        """
        Huấn luyện mô hình ARIMA
        
        Parameters:
        -----------
        data : np.ndarray
            Dữ liệu train
            
        Returns:
        --------
        self: ARIMAForecast
        """
        data = np.asarray(data).flatten()
        self.train_data = data
        
        print(f"\nDang train mo hinh ARIMA{self.order}...")
        
        # Tạo và fit mô hình
        self.model = ARIMA(data, order=self.order)
        self.fitted_model = self.model.fit()
        
        # Lấy fitted values
        self.fitted_values = self.fitted_model.fittedvalues
        
        print("Hoan thanh train!")
        print(self.fitted_model.summary())
        
        return self
    
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
        if self.fitted_model is None:
            raise ValueError("Mo hinh chua duoc train. Goi fit() truoc.")
        
        # Dự báo
        forecast = self.fitted_model.forecast(steps=steps)
        
        return np.array(forecast)
    
    def forecast_with_confidence(self, steps: int = 1, alpha: float = 0.05) -> dict:
        """
        Dự báo kèm khoảng tin cậy
        
        Parameters:
        -----------
        steps : int
            Số bước cần dự báo
        alpha : float
            Mức ý nghĩa (default: 0.05 cho khoảng tin cậy 95%)
            
        Returns:
        --------
        dict: Dự báo và khoảng tin cậy
        """
        if self.fitted_model is None:
            raise ValueError("Mo hinh chua duoc train. Goi fit() truoc.")
        
        # Dự báo với khoảng tin cậy
        forecast_result = self.fitted_model.get_forecast(steps=steps, alpha=alpha)
        
        predictions = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
        
        # Convert to numpy array if needed
        if hasattr(predictions, 'values'):
            predictions = predictions.values
        else:
            predictions = np.asarray(predictions)
        
        if hasattr(conf_int, 'iloc'):
            lower_bound = conf_int.iloc[:, 0].values
            upper_bound = conf_int.iloc[:, 1].values
        else:
            conf_int = np.asarray(conf_int)
            lower_bound = conf_int[:, 0]
            upper_bound = conf_int[:, 1]
        
        return {
            'predictions': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': (1 - alpha) * 100
        }
    
    def evaluate(self, X_true: np.ndarray) -> dict:
        """
        Đánh giá độ chính xác của mô hình
        
        Parameters:
        -----------
        X_true : np.ndarray
            Giá trị thực tế
            
        Returns:
        --------
        dict: Các chỉ số đánh giá
        """
        X_true = np.asarray(X_true).flatten()
        
        # Đảm bảo độ dài khớp
        if len(self.fitted_values) != len(X_true):
            # Bỏ qua giá trị đầu tiên của fitted nếu cần
            X_pred = self.fitted_values[:len(X_true)]
        else:
            X_pred = self.fitted_values
        
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
    
    def plot_results(self, X_test: np.ndarray = None, show_confidence: bool = True, save_path: str = None):
        """
        Vẽ biểu đồ kết quả fitted và dự báo
        
        Parameters:
        -----------
        X_test : np.ndarray
            Dữ liệu test thực tế
        show_confidence : bool
            Hiển thị khoảng tin cậy
        save_path : str
            Đường dẫn lưu hình
        """
        if self.fitted_values is None:
            raise ValueError("Mo hinh chua duoc train.")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        n = len(self.train_data)
        
        # Plot training data và fitted values
        ax.plot(range(n), self.train_data, 'o-', label='Du lieu train', linewidth=2, markersize=6)
        ax.plot(range(len(self.fitted_values)), self.fitted_values, 's--', 
                label='Gia tri fitted', linewidth=2, markersize=5, alpha=0.7)
        
        # Plot predictions nếu có test data
        if X_test is not None:
            X_test = np.asarray(X_test).flatten()
            
            if show_confidence:
                forecast_result = self.forecast_with_confidence(steps=len(X_test))
                predictions = forecast_result['predictions']
                lower = forecast_result['lower_bound']
                upper = forecast_result['upper_bound']
                
                test_range = range(n, n + len(X_test))
                
                ax.plot(test_range, X_test, 'o-', label='Du lieu test', 
                       linewidth=2, markersize=6, color='green')
                ax.plot(test_range, predictions, 's--', label=f'Du bao ARIMA{self.order}', 
                       linewidth=2, markersize=5, color='red')
                
                # Vẽ khoảng tin cậy
                ax.fill_between(test_range, lower, upper, alpha=0.2, color='red',
                               label=f'Khoang tin cay {forecast_result["confidence_level"]}%')
            else:
                predictions = self.predict(len(X_test))
                test_range = range(n, n + len(X_test))
                
                ax.plot(test_range, X_test, 'o-', label='Du lieu test', 
                       linewidth=2, markersize=6, color='green')
                ax.plot(test_range, predictions, 's--', label=f'Du bao ARIMA{self.order}', 
                       linewidth=2, markersize=5, color='red')
        
        ax.set_xlabel('Thoi gian', fontsize=12)
        ax.set_ylabel('Gia tri', fontsize=12)
        ax.set_title(f'Ket qua mo hinh ARIMA{self.order}', fontsize=14, fontweight='bold')
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
        if self.fitted_model is None:
            return "Mo hinh chua duoc train"
        
        p, d, q = self.order
        
        info = f"""
        ==============================================
        THONG TIN MO HINH ARIMA{self.order}
        ==============================================
        p (AR order): {p}
        d (Differencing): {d}
        q (MA order): {q}
        
        So diem du lieu train: {len(self.train_data)}
        AIC: {self.fitted_model.aic:.2f}
        BIC: {self.fitted_model.bic:.2f}
        ==============================================
        """
        return info
    
    def auto_arima_search(self, data: np.ndarray, max_p: int = 5, max_d: int = 2, 
                         max_q: int = 5) -> tuple:
        """
        Tìm kiếm tham số ARIMA tối ưu dựa trên AIC
        
        Parameters:
        -----------
        data : np.ndarray
            Dữ liệu train
        max_p : int
            p tối đa
        max_d : int
            d tối đa
        max_q : int
            q tối đa
            
        Returns:
        --------
        tuple: (p, d, q) tối ưu
        """
        data = np.asarray(data).flatten()
        
        best_aic = np.inf
        best_order = None
        
        print("\nDang tim kiem tham so ARIMA toi uu...")
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted = model.fit()
                        
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                            
                    except:
                        continue
        
        print(f"Tham so toi uu: ARIMA{best_order} voi AIC = {best_aic:.2f}")
        
        return best_order


def demo_arima():
    """
    Hàm demo sử dụng ARIMA
    """
    print("DEMO MO HINH ARIMA")
    print("="*60)
    
    # Dữ liệu mẫu
    np.random.seed(42)
    data = np.cumsum(np.random.randn(100)) + 100
    
    print(f"\nSo diem du lieu: {len(data)}")
    
    # Chia train/test
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    print(f"Tap train: {len(train_data)} diem")
    print(f"Tap test: {len(test_data)} diem")
    
    # Khởi tạo mô hình
    model = ARIMAForecast(order=(2, 1, 2))
    
    # Kiểm tra tính dừng
    print("\nKIEM TRA TINH DUNG:")
    stationarity = model.check_stationarity(train_data)
    for key, value in stationarity.items():
        print(f"{key}: {value}")
    
    # Vẽ ACF, PACF
    model.plot_diagnostics(train_data, lags=20, save_path='arima_diagnostics.png')
    
    # Train mô hình
    model.fit(train_data)
    
    # Hiển thị thông tin
    print(model.get_model_info())
    
    # Đánh giá
    metrics = model.evaluate(train_data)
    print("\nDANH GIA TREN TAP TRAIN:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Dự báo
    forecast_result = model.forecast_with_confidence(steps=len(test_data))
    predictions = forecast_result['predictions']
    
    print(f"\nDu bao {len(test_data)} buoc tiep theo:")
    print(predictions)
    
    # So sánh
    print("\nSo sanh du bao vs thuc te:")
    for i, (pred, true, lower, upper) in enumerate(zip(
        predictions, test_data, 
        forecast_result['lower_bound'], 
        forecast_result['upper_bound']
    )):
        error = abs(pred - true) / abs(true) * 100
        in_range = "✓" if lower <= true <= upper else "✗"
        print(f"Buoc {i+1}: Du bao={pred:.2f}, Thuc te={true:.2f}, " +
              f"Sai so={error:.2f}%, Trong CI={in_range}")
    
    # Vẽ biểu đồ
    model.plot_results(test_data, show_confidence=True, save_path='arima_results.png')
    
    return model, predictions


if __name__ == "__main__":
    model, predictions = demo_arima()
