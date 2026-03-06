# AI语音训练模型（轻量级版本）

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import librosa

class AIVoiceTrainingModel:
    def __init__(self):
        """
        初始化AI语音训练模型
        """
        self.model = SVC(kernel='rbf', C=1.0, gamma='scale')
    
    def train(self, X_train, y_train):
        """
        训练模型
        :param X_train: 训练数据
        :param y_train: 训练标签
        """
        # 展平数据，因为SVM需要2D输入
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        self.model.fit(X_train_flat, y_train)
    
    def evaluate(self, X_test, y_test):
        """
        评估模型
        :param X_test: 测试数据
        :param y_test: 测试标签
        """
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        y_pred = self.model.predict(X_test_flat)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    
    def save_model(self, model_path):
        """
        保存模型
        :param model_path: 模型保存路径
        """
        import joblib
        joblib.dump(self.model, model_path)
    
    def load_model(self, model_path):
        """
        加载模型
        :param model_path: 模型加载路径
        """
        import joblib
        self.model = joblib.load(model_path)

def preprocess_data(audio_files, labels, max_length=1000):
    """
    预处理音频数据
    :param audio_files: 音频文件列表
    :param labels: 标签列表
    :param max_length: 最大序列长度
    """
    X = []
    y = []
    
    for audio_file, label in zip(audio_files, labels):
        if os.path.exists(audio_file):
            try:
                # 使用librosa加载音频文件（支持ogg和mp3）
                y_audio, sr = librosa.load(audio_file, sr=None)
                
                # 提取MFCC特征
                mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
                mfccs = mfccs.T  # 转置为 (时间步长, 特征数)
                
                # 调整序列长度
                if len(mfccs) > max_length:
                    mfccs = mfccs[:max_length]
                else:
                    # 填充零
                    padding = max_length - len(mfccs)
                    mfccs = np.pad(mfccs, ((0, padding), (0, 0)), 'constant')
                
                X.append(mfccs)
                y.append(label)
            except Exception as e:
                print(f"处理文件 {audio_file} 时出错: {e}")
        else:
            print(f"文件不存在: {audio_file}")
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def main():
    """
    主函数
    """
    # 模拟数据
    audio_files = [f"audio_{i}.wav" for i in range(100)]
    labels = np.random.randint(0, 10, 100)  # 10个类别
    
    # 预处理数据
    X, y = preprocess_data(audio_files, labels)
    
    if len(X) == 0:
        print("没有有效的音频文件，使用模拟数据进行测试")
        # 使用模拟数据
        X = np.random.rand(100, 1000, 13)
        y = np.random.randint(0, 10, 100)
    
    # 划分训练集、验证集和测试集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # 初始化模型
    model = AIVoiceTrainingModel()
    
    # 训练模型
    model.train(X_train, y_train)
    
    # 评估模型
    accuracy = model.evaluate(X_test, y_test)
    print(f"测试准确率: {accuracy:.4f}")
    
    # 保存模型
    model.save_model("ai_voice_model.pkl")
    print("模型保存成功！")

if __name__ == "__main__":
    main()
