# AI语音训练模型

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import librosa
import logging
from datetime import datetime

# 设置日志记录
def setup_logging():
    """设置日志记录"""
    log_dir = os.path.expanduser("~/Desktop")
    log_file = os.path.join(log_dir, "voice_training.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class AIVoiceTrainingModel:
    def __init__(self, input_shape, output_shape):
        """
        初始化AI语音训练模型
        :param input_shape: 输入数据形状 (时间步长, 特征数)
        :param output_shape: 输出数据形状
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = self._build_model()
    
    def _build_model(self):
        """
        构建模型架构
        """
        model = Sequential()
        
        # 卷积层提取特征
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))
        
        # LSTM层处理序列数据
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(64))
        model.add(Dropout(0.2))
        
        # 全连接层
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.output_shape, activation='softmax'))
        
        # 编译模型
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        训练模型
        :param X_train: 训练数据
        :param y_train: 训练标签
        :param X_val: 验证数据
        :param y_val: 验证标签
        :param epochs: 训练轮数
        :param batch_size: 批次大小
        """
        logger.info(f"开始训练: epochs={epochs}, batch_size={batch_size}")
        logger.info(f"训练数据: X_train={X_train.shape}, y_train={y_train.shape}")
        logger.info(f"验证数据: X_val={X_val.shape}, y_val={y_val.shape}")
        
        history = self.model.fit(X_train, y_train,
                               validation_data=(X_val, y_val),
                               epochs=epochs,
                               batch_size=batch_size,
                               verbose=1)
        
        logger.info("训练完成")
        return history
    
    def evaluate(self, X_test, y_test):
        """
        评估模型
        :param X_test: 测试数据
        :param y_test: 测试标签
        """
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return loss, accuracy
    
    def save_model(self, model_path):
        """
        保存模型
        :param model_path: 模型保存路径
        """
        self.model.save(model_path)
    
    def load_model(self, model_path):
        """
        加载模型
        :param model_path: 模型加载路径
        """
        self.model = tf.keras.models.load_model(model_path)

def preprocess_data(audio_files, labels, max_length=1000):
    """
    预处理音频数据
    :param audio_files: 音频文件列表
    :param labels: 标签列表
    :param max_length: 最大序列长度
    """
    X = []
    y = []
    processed_count = 0
    error_count = 0
    
    logger.info(f"开始预处理数据，共 {len(audio_files)} 个音频文件")
    
    for idx, (audio_file, label) in enumerate(zip(audio_files, labels)):
        if os.path.exists(audio_file):
            try:
                # 使用librosa加载音频文件（支持ogg和mp3）
                logger.info(f"正在处理文件 {idx+1}/{len(audio_files)}: {audio_file}")
                y_audio, sr = librosa.load(audio_file, sr=None)
                
                # 检查音频是否为空
                if len(y_audio) == 0:
                    logger.warning(f"音频文件为空: {audio_file}")
                    error_count += 1
                    continue
                
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
                processed_count += 1
                logger.info(f"成功处理文件 {audio_file}, 特征形状: {mfccs.shape}")
                
            except Exception as e:
                logger.error(f"处理文件 {audio_file} 时出错: {e}")
                error_count += 1
        else:
            logger.warning(f"文件不存在: {audio_file}")
            error_count += 1
    
    logger.info(f"预处理完成: 成功处理 {processed_count} 个文件，失败 {error_count} 个文件")
    
    if len(X) == 0:
        logger.error("没有成功处理任何音频文件！")
        return np.array([]), np.array([])
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"数据形状: X={X.shape}, y={y.shape}")
    
    # 标签独热编码
    try:
        y = tf.keras.utils.to_categorical(y)
        logger.info(f"标签独热编码完成，形状: {y.shape}")
    except Exception as e:
        logger.error(f"标签独热编码失败: {e}")
        return np.array([]), np.array([])
    
    return X, y

def main():
    """
    主函数
    """
    logger.info("=" * 50)
    logger.info("开始AI语音训练模型")
    logger.info("=" * 50)
    
    # 检查是否有真实的音频文件
    audio_dir = "audio_data"
    if not os.path.exists(audio_dir):
        logger.warning(f"音频数据目录不存在: {audio_dir}")
        logger.info("将使用模拟数据进行测试...")
        
        # 使用模拟数据
        logger.info("生成模拟数据...")
        num_samples = 100
        num_classes = 10
        max_length = 1000
        n_mfcc = 13
        
        # 生成模拟的MFCC特征
        X = np.random.rand(num_samples, max_length, n_mfcc)
        y = np.random.randint(0, num_classes, num_samples)
        
        logger.info(f"模拟数据生成完成: X={X.shape}, y={y.shape}")
        
        # 标签独热编码
        y = tf.keras.utils.to_categorical(y)
        logger.info(f"标签独热编码完成: {y.shape}")
        
        # 划分训练集、验证集和测试集
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        logger.info(f"数据集划分完成:")
        logger.info(f"  训练集: {X_train.shape}")
        logger.info(f"  验证集: {X_val.shape}")
        logger.info(f"  测试集: {X_test.shape}")
    else:
        # 使用真实音频文件
        logger.info(f"从目录 {audio_dir} 加载音频文件...")
        
        # 获取所有音频文件
        audio_files = []
        labels = []
        
        for label, class_name in enumerate(os.listdir(audio_dir)):
            class_dir = os.path.join(audio_dir, class_name)
            if os.path.isdir(class_dir):
                for file in os.listdir(class_dir):
                    if file.endswith(('.wav', '.mp3', '.ogg', '.flac')):
                        audio_files.append(os.path.join(class_dir, file))
                        labels.append(label)
        
        logger.info(f"找到 {len(audio_files)} 个音频文件")
        
        if len(audio_files) == 0:
            logger.error("没有找到任何音频文件！")
            return
        
        # 预处理数据
        X, y = preprocess_data(audio_files, labels)
        
        if len(X) == 0:
            logger.error("预处理失败，没有可用的数据！")
            return
        
        # 划分训练集、验证集和测试集
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        logger.info(f"数据集划分完成:")
        logger.info(f"  训练集: {X_train.shape}")
        logger.info(f"  验证集: {X_val.shape}")
        logger.info(f"  测试集: {X_test.shape}")
    
    # 初始化模型
    input_shape = X_train.shape[1:]
    output_shape = y_train.shape[1]
    
    logger.info(f"初始化模型: input_shape={input_shape}, output_shape={output_shape}")
    model = AIVoiceTrainingModel(input_shape, output_shape)
    
    # 训练模型
    logger.info("开始训练模型...")
    logger.info(f"训练参数: epochs=10, batch_size=16")
    
    try:
        history = model.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=16)
        logger.info("模型训练完成")
        
        # 评估模型
        logger.info("评估模型...")
        loss, accuracy = model.evaluate(X_test, y_test)
        logger.info(f"测试损失: {loss:.4f}, 测试准确率: {accuracy:.4f}")
        
        # 保存模型
        model_path = "ai_voice_model.h5"
        logger.info(f"保存模型到: {model_path}")
        model.save_model(model_path)
        logger.info("模型保存成功！")
        
        logger.info("=" * 50)
        logger.info("训练流程完成")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
