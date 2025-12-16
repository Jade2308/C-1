"""
Data Preprocessing Module for Cryptocurrency Price Forecasting
Xử lý và chuẩn bị dữ liệu trước khi dự báo
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Cấu hình hiển thị tiếng Việt
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class CryptoDataPreprocessor:
    """
    Lớp xử lý dữ liệu tiền mã hóa
    """
    
    def __init__(self, file_path):
        """
        Khởi tạo với đường dẫn file dữ liệu
        
        Parameters:
        -----------
        file_path : str
            Đường dẫn tới file CSV
        """
        self.file_path = file_path
        self.df = None
        self.scaler = MinMaxScaler()
        
    def load_data(self):
        """
        Đọc dữ liệu từ file CSV
        """
        print(f"Đang đọc dữ liệu từ: {self.file_path}")
        self.df = pd.read_csv(self.file_path)
        print(f"Đã đọc {len(self.df)} dòng dữ liệu")
        print(f"\nCác cột trong dữ liệu: {self.df.columns.tolist()}")
        return self.df
    
    def explore_data(self):
        """
        Khám phá và hiển thị thông tin cơ bản về dữ liệu
        """
        if self.df is None:
            self.load_data()
            
        print("\n" + "="*60)
        print("THÔNG TIN DỮ LIỆU")
        print("="*60)
        print(f"\nKích thước dữ liệu: {self.df.shape}")
        print(f"\nThông tin các cột:")
        print(self.df.info())
        print(f"\nThống kê mô tả:")
        print(self.df.describe())
        print(f"\nGiá trị null:")
        print(self.df.isnull().sum())
        print(f"\n5 dòng đầu tiên:")
        print(self.df.head())
        
    def clean_data(self):
        """
        Làm sạch dữ liệu:
        - Xử lý giá trị null
        - Loại bỏ duplicates
        - Chuyển đổi kiểu dữ liệu
        """
        if self.df is None:
            self.load_data()
            
        print("\n" + "="*60)
        print("LÀM SẠCH DỮ LIỆU")
        print("="*60)
        
        initial_rows = len(self.df)
        
        # Xử lý giá trị null
        print(f"\nGiá trị null trước khi xử lý:")
        null_counts = self.df.isnull().sum()
        print(null_counts[null_counts > 0])
        
        # Loại bỏ các dòng có giá trị null quan trọng
        self.df = self.df.dropna(subset=['close', 'date'])
        
        # Loại bỏ duplicates
        duplicates = self.df.duplicated().sum()
        print(f"\nSố dòng trùng lặp: {duplicates}")
        self.df = self.df.drop_duplicates()
        
        # Chuyển đổi cột date sang datetime
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df = self.df.sort_values('date').reset_index(drop=True)
        
        final_rows = len(self.df)
        print(f"\nĐã xóa {initial_rows - final_rows} dòng dữ liệu")
        print(f"Còn lại {final_rows} dòng dữ liệu sạch")
        
        return self.df
    
    def filter_crypto(self, symbol='BTC'):
        """
        Lọc dữ liệu theo cryptocurrency cụ thể
        
        Parameters:
        -----------
        symbol : str
            Ký hiệu cryptocurrency (default: 'BTC')
        """
        if self.df is None:
            self.load_data()
            
        if 'symbol' in self.df.columns:
            self.df = self.df[self.df['symbol'] == symbol].copy()
            print(f"\nĐã lọc dữ liệu cho {symbol}: {len(self.df)} dòng")
        
        return self.df
    
    def aggregate_to_daily(self):
        """
        Tổng hợp dữ liệu theo ngày (nếu dữ liệu là hourly)
        """
        if self.df is None or 'date' not in self.df.columns:
            return self.df
            
        print("\n" + "="*60)
        print("TỔNG HỢP DỮ LIỆU THEO NGÀY")
        print("="*60)
        
        # Lấy ngày từ timestamp
        self.df['date_only'] = self.df['date'].dt.date
        
        # Tổng hợp theo ngày
        daily_df = self.df.groupby('date_only').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()
        
        daily_df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        
        print(f"Đã tổng hợp từ {len(self.df)} dòng xuống {len(daily_df)} ngày")
        
        self.df = daily_df
        return self.df
    
    def prepare_for_forecasting(self, target_col='close', test_size=30):
        """
        Chuẩn bị dữ liệu cho mô hình dự báo
        
        Parameters:
        -----------
        target_col : str
            Cột cần dự báo (default: 'close')
        test_size : int
            Số điểm dữ liệu cho tập test
            
        Returns:
        --------
        dict: Dictionary chứa train/test data và scaler
        """
        if self.df is None:
            self.load_data()
            
        print("\n" + "="*60)
        print("CHUẨN BỊ DỮ LIỆU CHO DỰ BÁO")
        print("="*60)
        
        # Lấy cột target
        data = self.df[target_col].values.reshape(-1, 1)
        
        # Chia train/test
        train_size = len(data) - test_size
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        # Chuẩn hóa dữ liệu
        train_scaled = self.scaler.fit_transform(train_data)
        test_scaled = self.scaler.transform(test_data)
        
        print(f"\nKích thước tập train: {len(train_data)}")
        print(f"Kích thước tập test: {len(test_data)}")
        print(f"Giá trị min: {data.min():.2f}")
        print(f"Giá trị max: {data.max():.2f}")
        print(f"Giá trị trung bình: {data.mean():.2f}")
        
        return {
            'train': train_data,
            'test': test_data,
            'train_scaled': train_scaled,
            'test_scaled': test_scaled,
            'scaler': self.scaler,
            'dates': self.df['date'].values if 'date' in self.df.columns else None
        }
    
    def visualize_data(self, target_col='close', save_path=None):
        """
        Vẽ biểu đồ dữ liệu
        
        Parameters:
        -----------
        target_col : str
            Cột cần vẽ
        save_path : str
            Đường dẫn lưu hình (optional)
        """
        if self.df is None:
            self.load_data()
            
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Biểu đồ giá theo thời gian
        if 'date' in self.df.columns:
            axes[0].plot(self.df['date'], self.df[target_col], linewidth=1.5)
            axes[0].set_xlabel('Thoi gian', fontsize=12)
        else:
            axes[0].plot(self.df[target_col], linewidth=1.5)
            axes[0].set_xlabel('Index', fontsize=12)
            
        axes[0].set_ylabel(f'{target_col.capitalize()} Price', fontsize=12)
        axes[0].set_title(f'Gia {target_col} theo thoi gian', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Histogram phân phối
        axes[1].hist(self.df[target_col], bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel(f'{target_col.capitalize()} Price', fontsize=12)
        axes[1].set_ylabel('Tan suat', fontsize=12)
        axes[1].set_title(f'Phan phoi gia {target_col}', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nĐã lưu biểu đồ tại: {save_path}")
        
        plt.show()
        
    def get_statistics(self):
        """
        Tính các chỉ số thống kê
        """
        if self.df is None:
            self.load_data()
            
        stats = {
            'count': len(self.df),
            'mean_close': self.df['close'].mean(),
            'std_close': self.df['close'].std(),
            'min_close': self.df['close'].min(),
            'max_close': self.df['close'].max(),
            'median_close': self.df['close'].median()
        }
        
        return stats


def main():
    """
    Hàm chính để test preprocessing
    """
    # Đường dẫn file dữ liệu
    file_path = r"D:\Google Drive\Drive University\Học Kỳ 5\Kho Dữ Liệu\Project\Data\Crypto_Hourly_Refined.csv"
    
    # Khởi tạo preprocessor
    preprocessor = CryptoDataPreprocessor(file_path)
    
    # Load và khám phá dữ liệu
    preprocessor.load_data()
    preprocessor.explore_data()
    
    # Làm sạch dữ liệu
    preprocessor.clean_data()
    
    # Lọc Bitcoin
    preprocessor.filter_crypto('BTC')
    
    # Tổng hợp theo ngày
    preprocessor.aggregate_to_daily()
    
    # Vẽ biểu đồ
    preprocessor.visualize_data(save_path='data_visualization.png')
    
    # Chuẩn bị dữ liệu cho dự báo
    prepared_data = preprocessor.prepare_for_forecasting(test_size=30)
    
    # Thống kê
    stats = preprocessor.get_statistics()
    print("\n" + "="*60)
    print("THỐNG KÊ TỔNG QUAN")
    print("="*60)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    return preprocessor, prepared_data


if __name__ == "__main__":
    preprocessor, prepared_data = main()
