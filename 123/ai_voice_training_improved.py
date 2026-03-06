# AI语音训练模型（改进版）

import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import librosa
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 单类别模型，用于只有一个类别的情况
class SingleClassModel:
    def __init__(self, class_label):
        self.class_label = class_label
    def predict(self, X):
        return np.full(len(X), self.class_label)

class AIVoiceTrainingModel:
    def __init__(self):
        """
        初始化AI语音训练模型
        """
        self.model = SVC(kernel='rbf', C=1.0, gamma='scale')
        self.scaler = StandardScaler()
    
    def train(self, X_train, y_train, use_grid_search=False, epochs=50):
        """
        训练模型
        :param X_train: 训练特征
        :param y_train: 训练标签
        :param use_grid_search: 是否使用网格搜索优化参数
        :param epochs: 训练次数（SVM模型不需要，但保留接口一致性）
        """
        print(f"训练次数: {epochs}")
        # 展平数据，因为SVM需要2D输入
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        # 数据标准化
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        
        # 检查样本数量和类别数量
        n_samples = len(X_train)
        unique_classes = np.unique(y_train)
        n_classes = len(unique_classes)
        
        if n_classes == 1:
            # 只有一个类别时，创建一个简单的模型
            self.model = SingleClassModel(unique_classes[0])
            print("只有一个类别，使用单类别模型")
        elif use_grid_search and n_samples >= 2:
            # 网格搜索优化参数 - 减少参数范围以减少内存使用
            param_grid = {
                'C': [1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf']
            }
            
            # 根据样本数量动态调整交叉验证折数
            cv = min(3, n_samples)  # 减少交叉验证折数
            # 确保交叉验证折数至少为2
            cv = max(2, cv)
            
            # 限制使用的CPU核心数，避免内存过度使用
            n_jobs = min(2, os.cpu_count())
            grid_search = GridSearchCV(SVC(), param_grid, cv=cv, n_jobs=n_jobs)
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
            print(f"最佳参数: {grid_search.best_params_}")
        else:
            # 常规训练
            self.model = SVC()
            self.model.fit(X_train_scaled, y_train)
    
    def evaluate(self, X_test, y_test, plot_confusion_matrix=False):
        """
        评估模型
        :param X_test: 测试数据
        :param y_test: 测试标签
        :param plot_confusion_matrix: 是否绘制混淆矩阵
        """
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        X_test_scaled = self.scaler.transform(X_test_flat)
        
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"测试准确率: {accuracy:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))
        
        if plot_confusion_matrix:
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('混淆矩阵')
            plt.xlabel('预测标签')
            plt.ylabel('真实标签')
            plt.savefig('confusion_matrix.png')
            print("混淆矩阵已保存为 confusion_matrix.png")
        
        return accuracy
    
    def predict(self, audio_file):
        """
        预测单个音频文件
        :param audio_file: 音频文件路径
        """
        try:
            # 加载音频文件
            y_audio, sr = librosa.load(audio_file, sr=None)
            
            # 提取MFCC特征
            mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
            mfccs = mfccs.T  # 转置为 (时间步长, 特征数)
            
            # 调整序列长度
            max_length = 1000
            if len(mfccs) > max_length:
                mfccs = mfccs[:max_length]
            else:
                # 填充零
                padding = max_length - len(mfccs)
                mfccs = np.pad(mfccs, ((0, padding), (0, 0)), 'constant')
            
            # 展平并标准化
            mfccs_flat = mfccs.reshape(1, -1)
            mfccs_scaled = self.scaler.transform(mfccs_flat)
            
            # 预测
            prediction = self.model.predict(mfccs_scaled)
            return prediction[0]
        except Exception as e:
            print(f"预测时出错: {e}")
            return None
    
    def save_model(self, model_path):
        """
        保存模型
        :param model_path: 模型保存路径
        """
        # 保存模型和标准化器
        joblib.dump({'model': self.model, 'scaler': self.scaler}, model_path)
    
    def load_model(self, model_path):
        """
        加载模型
        :param model_path: 模型加载路径
        """
        # 加载模型和标准化器
        data = joblib.load(model_path)
        self.model = data['model']
        self.scaler = data['scaler']

def preprocess_data(audio_files, labels, max_length=1000, augment=False):
    """
    预处理音频数据（加强版）
    :param audio_files: 音频文件列表
    :param labels: 标签列表
    :param max_length: 最大序列长度
    :param augment: 是否进行数据增强
    """
    X = []
    y = []
    
    # 限制处理的文件数量，防止内存不足
    max_files = 100  # 增加最大处理文件数
    processed_files = 0
    
    for audio_file, label in zip(audio_files, labels):
        if processed_files >= max_files:
            break
            
        if os.path.exists(audio_file):
            try:
                # 使用librosa加载音频文件（支持ogg和mp3）
                # 限制音频文件长度，防止内存不足
                y_audio, sr = librosa.load(audio_file, sr=None, duration=15)  # 增加到15秒
                
                # 提取多种特征
                # 1. MFCC特征（主要特征）
                mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=20)  # 增加MFCC数量
                mfccs = mfccs.T
                
                # 2. 色度特征
                chroma = librosa.feature.chroma_stft(y=y_audio, sr=sr)
                chroma = chroma.T
                
                # 3. 光谱对比度特征
                spectral_contrast = librosa.feature.spectral_contrast(y=y_audio, sr=sr)
                spectral_contrast = spectral_contrast.T
                
                # 4. 零交叉率特征
                zcr = librosa.feature.zero_crossing_rate(y_audio)
                zcr = zcr.T
                
                # 5. 谱质心特征
                spectral_centroid = librosa.feature.spectral_centroid(y=y_audio, sr=sr)
                spectral_centroid = spectral_centroid.T
                
                # 合并所有特征
                features = np.concatenate([
                    mfccs,
                    chroma,
                    spectral_contrast,
                    zcr,
                    spectral_centroid
                ], axis=1)
                
                # 调整序列长度
                if len(features) > max_length:
                    features = features[:max_length]
                else:
                    # 填充零
                    padding = max_length - len(features)
                    features = np.pad(features, ((0, padding), (0, 0)), 'constant')
                
                X.append(features)
                y.append(label)
                processed_files += 1
                
                # 数据增强 - 增强数据增强策略
                if augment and processed_files < max_files:
                    # 1. 添加噪声
                    noise = np.random.randn(len(y_audio)) * 0.005
                    y_audio_noisy = y_audio + noise
                    
                    # 提取增强数据的特征
                    mfccs_noisy = librosa.feature.mfcc(y=y_audio_noisy, sr=sr, n_mfcc=20)
                    mfccs_noisy = mfccs_noisy.T
                    
                    chroma_noisy = librosa.feature.chroma_stft(y=y_audio_noisy, sr=sr)
                    chroma_noisy = chroma_noisy.T
                    
                    spectral_contrast_noisy = librosa.feature.spectral_contrast(y=y_audio_noisy, sr=sr)
                    spectral_contrast_noisy = spectral_contrast_noisy.T
                    
                    zcr_noisy = librosa.feature.zero_crossing_rate(y_audio_noisy)
                    zcr_noisy = zcr_noisy.T
                    
                    spectral_centroid_noisy = librosa.feature.spectral_centroid(y=y_audio_noisy, sr=sr)
                    spectral_centroid_noisy = spectral_centroid_noisy.T
                    
                    features_noisy = np.concatenate([
                        mfccs_noisy,
                        chroma_noisy,
                        spectral_contrast_noisy,
                        zcr_noisy,
                        spectral_centroid_noisy
                    ], axis=1)
                    
                    if len(features_noisy) > max_length:
                        features_noisy = features_noisy[:max_length]
                    else:
                        padding = max_length - len(features_noisy)
                        features_noisy = np.pad(features_noisy, ((0, padding), (0, 0)), 'constant')
                    
                    X.append(features_noisy)
                    y.append(label)
                    processed_files += 1
                    
                    # 2. 时间拉伸
                    if processed_files < max_files:
                        y_audio_stretch = librosa.effects.time_stretch(y_audio, rate=1.2)
                        
                        mfccs_stretch = librosa.feature.mfcc(y=y_audio_stretch, sr=sr, n_mfcc=20)
                        mfccs_stretch = mfccs_stretch.T
                        
                        chroma_stretch = librosa.feature.chroma_stft(y=y_audio_stretch, sr=sr)
                        chroma_stretch = chroma_stretch.T
                        
                        spectral_contrast_stretch = librosa.feature.spectral_contrast(y=y_audio_stretch, sr=sr)
                        spectral_contrast_stretch = spectral_contrast_stretch.T
                        
                        zcr_stretch = librosa.feature.zero_crossing_rate(y_audio_stretch)
                        zcr_stretch = zcr_stretch.T
                        
                        spectral_centroid_stretch = librosa.feature.spectral_centroid(y=y_audio_stretch, sr=sr)
                        spectral_centroid_stretch = spectral_centroid_stretch.T
                        
                        features_stretch = np.concatenate([
                            mfccs_stretch,
                            chroma_stretch,
                            spectral_contrast_stretch,
                            zcr_stretch,
                            spectral_centroid_stretch
                        ], axis=1)
                        
                        if len(features_stretch) > max_length:
                            features_stretch = features_stretch[:max_length]
                        else:
                            padding = max_length - len(features_stretch)
                            features_stretch = np.pad(features_stretch, ((0, padding), (0, 0)), 'constant')
                        
                        X.append(features_stretch)
                        y.append(label)
                        processed_files += 1
                    
            except Exception as e:
                print(f"处理文件 {audio_file} 时出错: {e}")
        else:
            print(f"文件不存在: {audio_file}")
    
    if len(X) == 0:
        # 返回空数组
        return np.array([]), np.array([])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"预处理完成，共处理 {len(X)} 个样本，特征维度: {X.shape}")
    
    return X, y

def main():
    """
    主函数
    """
    # 模拟数据
    audio_files = [f"audio_{i}.wav" for i in range(100)]
    labels = np.random.randint(0, 10, 100)  # 10个类别
    
    # 预处理数据
    X, y = preprocess_data(audio_files, labels, augment=True)
    
    if len(X) == 0:
        print("没有有效的音频文件，使用模拟数据进行测试")
        # 使用模拟数据
        X = np.random.rand(100, 1000, 13)
        y = np.random.randint(0, 10, 100)
    
    # 划分训练集、验证集和测试集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 初始化模型
    model = AIVoiceTrainingModel()
    
    # 训练模型
    print("开始训练模型...")
    model.train(X_train, y_train, use_grid_search=True)
    
    # 评估模型
    print("\n评估模型...")
    accuracy = model.evaluate(X_test, y_test, plot_confusion_matrix=True)
    
    # 保存模型
    model.save_model("ai_voice_model_improved.pkl")
    print("\n模型保存成功！")

if __name__ == "__main__":
    main()
