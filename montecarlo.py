import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

# Đọc và chuẩn bị dữ liệu ban đầu
# Chúng ta chỉ quan tâm đến tuổi và tần suất mua hàng
df = pd.read_csv("shopping_trends.csv").filter(['Age', 'Frequency of Purchases'])

# Định nghĩa lớp cơ sở cho tất cả các mô hình Bayesian
# Lớp này chứa các phương thức chung để theo dõi thống kê và cập nhật niềm tin
class BayesianModelBase:
    def __init__(self):
        self.age_stats = {}  # Lưu thống kê về tuổi cho mỗi tần suất
        self.frequency_priors = {}  # Lưu prior probabilities
        self.total_observations = 0
        
    def update_stats(self, age, frequency):
        """
        Cập nhật thống kê cơ bản cho một quan sát mới
        Phương thức này theo dõi các thống kê cần thiết cho việc ước lượng phân phối
        """
        self.total_observations += 1
        
        if frequency not in self.frequency_priors:
            self.frequency_priors[frequency] = 0
            self.age_stats[frequency] = {'sum': 0, 'sum_squared': 0, 'count': 0}
        
        self.frequency_priors[frequency] += 1
        stats = self.age_stats[frequency]
        stats['sum'] += age
        stats['sum_squared'] += age * age
        stats['count'] += 1
        
    def get_distribution_params(self, frequency):
        """
        Tính toán các tham số phân phối (mean và standard deviation) cho một tần suất
        """
        stats = self.age_stats[frequency]
        mean = stats['sum'] / stats['count']
        variance = (stats['sum_squared'] / stats['count']) - (mean * mean)
        std = max(np.sqrt(variance), 1e-6)  # Tránh standard deviation = 0
        return mean, std

# Mô hình sử dụng phân phối chuẩn để mô hình hóa tuổi
class OnlineNormalModel(BayesianModelBase):
    def train(self, age, frequency):
        """Cập nhật mô hình với một quan sát mới"""
        self.update_stats(age, frequency)
        
    def predict(self, age):
        """Dự đoán tần suất dựa trên tuổi sử dụng phân phối chuẩn"""
        if not self.frequency_priors:
            return None
            
        posteriors = {}
        for freq in self.frequency_priors:
            # Tính prior probability từ số lượng quan sát
            prior = self.frequency_priors[freq] / self.total_observations
            
            # Tính likelihood sử dụng phân phối chuẩn
            mean, std = self.get_distribution_params(freq)
            likelihood = norm.pdf(age, mean, std)
            
            # Áp dụng định lý Bayes
            posteriors[freq] = prior * likelihood
            
        # Normalize và chọn tần suất có posterior cao nhất
        total = sum(posteriors.values())
        if total > 0:
            posteriors = {k: v/total for k, v in posteriors.items()}
            return max(posteriors.items(), key=lambda x: x[1])[0]
        return None

# Mô hình phân nhóm tuổi thành các khoảng
class OnlineBatchModel(BayesianModelBase):
    def __init__(self):
        super().__init__()
        self.age_categories = ['Young', 'Middle', 'Senior']
        self.category_stats = {cat: {} for cat in self.age_categories}
        
    def get_age_category(self, age):
        """Xác định nhóm tuổi dựa trên ngưỡng"""
        if age < 30:
            return 'Young'
        elif age < 50:
            return 'Middle'
        return 'Senior'
        
    def train(self, age, frequency):
        """Cập nhật thống kê cho nhóm tuổi và tần suất tương ứng"""
        self.update_stats(age, frequency)
        
        age_cat = self.get_age_category(age)
        if frequency not in self.category_stats[age_cat]:
            self.category_stats[age_cat][frequency] = 0
        self.category_stats[age_cat][frequency] += 1
        
    def predict(self, age):
        """Dự đoán tần suất dựa trên nhóm tuổi"""
        if not self.frequency_priors:
            return None
            
        age_cat = self.get_age_category(age)
        posteriors = {}
        
        for freq in self.frequency_priors:
            prior = self.frequency_priors[freq] / self.total_observations
            
            # Tính likelihood với Laplace smoothing
            cat_count = self.category_stats[age_cat].get(freq, 0)
            total_freq = self.frequency_priors[freq]
            likelihood = (cat_count + 1) / (total_freq + len(self.age_categories))
            
            posteriors[freq] = prior * likelihood
            
        total = sum(posteriors.values())
        if total > 0:
            posteriors = {k: v/total for k, v in posteriors.items()}
            return max(posteriors.items(), key=lambda x: x[1])[0]
        return None

# Mô hình xử lý trực tiếp các giá trị tuổi rời rạc
class OnlineDiscreteModel(BayesianModelBase):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha  # Tham số Laplace smoothing
        self.age_frequency_counts = {}
        
    def train(self, age, frequency):
        """Cập nhật bộ đếm cho cặp (tuổi, tần suất)"""
        self.update_stats(age, frequency)
        
        if age not in self.age_frequency_counts:
            self.age_frequency_counts[age] = {}
        
        if frequency not in self.age_frequency_counts[age]:
            self.age_frequency_counts[age][frequency] = 0
            
        self.age_frequency_counts[age][frequency] += 1
        
    def predict(self, age):
        """Dự đoán tần suất sử dụng các giá trị đếm được"""
        if not self.frequency_priors:
            return None
            
        posteriors = {}
        
        for freq in self.frequency_priors:
            prior = self.frequency_priors[freq] / self.total_observations
            
            if age in self.age_frequency_counts:
                count = self.age_frequency_counts[age].get(freq, 0)
                total = sum(self.age_frequency_counts[age].values())
            else:
                count = 0
                total = 0
                
            likelihood = (count + self.alpha) / (total + self.alpha * len(self.frequency_priors))
            posteriors[freq] = prior * likelihood
            
        total = sum(posteriors.values())
        if total > 0:
            posteriors = {k: v/total for k, v in posteriors.items()}
            return max(posteriors.items(), key=lambda x: x[1])[0]
        return None

# Lớp thực hiện mô phỏng Monte Carlo
class MonteCarloSimulation:
    def __init__(self, initial_data, n_simulations=100, time_periods=10):
        self.initial_data = initial_data
        self.n_simulations = n_simulations
        self.time_periods = time_periods
        
        # Lấy các tham số từ dữ liệu ban đầu
        self.age_mean = initial_data['Age'].mean()
        self.age_std = initial_data['Age'].std()
        self.frequencies = initial_data['Frequency of Purchases'].unique()
        
        # Tham số cho các thay đổi theo thời gian
        self.trend_strength = 0.1
        self.noise_level = 0.05
        
    def generate_time_varying_parameters(self):
        """Tạo các tham số thay đổi theo thời gian cho mô phỏng"""
        time_varying_means = []
        time_varying_stds = []
        time_varying_probs = []
        
        for t in range(self.time_periods):
            # Thêm xu hướng và nhiễu vào mean
            trend = self.trend_strength * t
            noise = np.random.normal(0, self.noise_level)
            mean = self.age_mean + trend + noise
            
            # Thêm biến động vào standard deviation
            std = self.age_std * (1 + np.random.normal(0, 0.1))
            
            # Tạo xác suất mới cho các tần suất
            probs = np.random.dirichlet(np.ones(len(self.frequencies)) * (5 + t))
            
            time_varying_means.append(mean)
            time_varying_stds.append(std)
            time_varying_probs.append(probs)
            
        return time_varying_means, time_varying_stds, time_varying_probs
    
    def simulate_single_path(self):
        """Thực hiện một lần mô phỏng hoàn chỉnh"""
        means, stds, probs = self.generate_time_varying_parameters()
        simulated_data = []
        
        # Phân loại tần suất theo mức độ thường xuyên
        frequent_labels = ['Weekly', 'Bi-Weekly', 'Fortnightly']
        infrequent_labels = ['Monthly', 'Quarterly', 'Every 3 Months', 'Annually']
        
        for t in range(self.time_periods):
            n_samples = len(self.initial_data)
            
            # Tạo tuổi từ phân phối chuẩn
            ages = np.random.normal(means[t], stds[t], n_samples)
            
            # Tạo tần suất mua hàng
            frequencies = np.random.choice(
                self.frequencies, 
                size=n_samples, 
                p=probs[t]
            )
            
            # Điều chỉnh tần suất dựa trên tuổi với các nhãn thực tế
            for i in range(n_samples):
                if ages[i] < 30:  # Khách hàng trẻ
                    if np.random.random() < 0.3:  # 30% cơ hội thay đổi
                        frequencies[i] = np.random.choice(frequent_labels)  # Xu hướng mua sắm thường xuyên hơn
                elif ages[i] > 50:  # Khách hàng lớn tuổi
                    if np.random.random() < 0.3:  # 30% cơ hội thay đổi
                        frequencies[i] = np.random.choice(infrequent_labels)  # Xu hướng mua sắm ít thường xuyên hơn
            
            period_data = pd.DataFrame({
                'Age': ages,
                'Frequency of Purchases': frequencies,
                'Time_Period': t
            })
            
            simulated_data.append(period_data)
        
        return pd.concat(simulated_data, ignore_index=True)
    
    def run_simulation(self):
        """Chạy nhiều lần mô phỏng và tổng hợp kết quả"""
        all_simulations = []
        
        for sim in range(self.n_simulations):
            sim_data = self.simulate_single_path()
            sim_data['Simulation'] = sim
            all_simulations.append(sim_data)
            
        return pd.concat(all_simulations, ignore_index=True)

def evaluate_models_on_simulation(simulation_data, models):
    """Đánh giá hiệu suất của các mô hình trên dữ liệu mô phỏng"""
    results = {name: [] for name in models.keys()}
    
    for sim in simulation_data['Simulation'].unique():
        sim_data = simulation_data[simulation_data['Simulation'] == sim]
        
        for period in sim_data['Time_Period'].unique():
            period_data = sim_data[sim_data['Time_Period'] == period]
            
            for name, model in models.items():
                # Tính độ chính xác trên dữ liệu hiện tại
                predictions = []
                actuals = []
                
                for _, row in period_data.iterrows():
                    pred = model.predict(row['Age'])
                    if pred is not None:
                        predictions.append(pred)
                        actuals.append(row['Frequency of Purchases'])
                
                if predictions:
                    accuracy = np.mean([p == a for p, a in zip(predictions, actuals)])
                    results[name].append(accuracy)
                
                # Cập nhật mô hình với dữ liệu mới
                for _, row in period_data.iterrows():
                    model.train(row['Age'], row['Frequency of Purchases'])
    
    return results

# Thực hiện mô phỏng và đánh giá
mc_sim = MonteCarloSimulation(df, n_simulations=20, time_periods=5)
simulated_data = mc_sim.run_simulation()

# Khởi tạo các mô hình
models = {
    'Normal': OnlineNormalModel(),
    'Batch': OnlineBatchModel(),
    'Discrete': OnlineDiscreteModel()
}

# Đánh giá các mô hình
results = evaluate_models_on_simulation(simulated_data, models)

# Visualize kết quả
plt.figure(figsize=(12, 6))
for name, accuracies in results.items():
    plt.plot(accuracies, label=name, alpha=0.7)
plt.xlabel('Time Steps')
plt.ylabel('Accuracy')
plt.title('Hiệu suất các mô hình qua thời gian (Mô phỏng Monte Carlo)')
plt.legend()
plt.grid(True)
plt.show()

# In kết quả thống kê
for name, accuracies in results.items():
    print(f"\nMô hình {name}:")
    print(f"Độ chính xác trung bình: {np.mean(accuracies):.4f}")
    print(f"Độ lệch chuẩn: {np.std(accuracies):.4f}")
    print(f"Min: {np.min(accuracies):.4f}")
    print(f"Max: {np.max(accuracies):.4f}")