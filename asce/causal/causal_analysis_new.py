import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime

# 删除了不再使用的因果发现算法导入
# 只保留必要的基础库

# 非线性模型导入
try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn未安装，非线性模型将不可用")

# 配置日志
import os
import time

# 创建日志目录
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)

# 配置日志
log_file = os.path.join(log_dir, "causal_analysis.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()  # 保留控制台输出，但设置为较高的级别
    ]
)
logger = logging.getLogger('causal_analysis')

# 设置控制台输出级别为WARNING，减少终端输出
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setLevel(logging.WARNING)

# 全局变量，用于存储DBN模式的输出格式
dbn_output_format = ""

# 全局变量，用于存储模型计算耗时
model_timing_log = []

# 定义映射字典
# 情感类型映射：积极情感数值越大，避免使用0
MOOD_TYPE_MAP = {
    "positive":  3,
    "neutral":   2,
    "negative": 1
}

# 情感值映射：积极情感数值越大，避免使用0
MOOD_VALUE_MAP = {
    "Optimistic": 10, "Confident": 9, "Passionate": 8, "Empathetic": 7, "Grateful": 6,
    "Realistic": 5,  "Rational": 4,   "Prudent": 3,     "Detached": 2,    "Objective": 1,
    "Pessimistic": -1, "Apathetic": -2, "Distrustful": -3, "Cynical": -4, "Resentful": -5
}

# 情绪类型映射：积极情绪数值越大，避免使用0
EMOTION_TYPE_MAP = {
    "positive":  3,
    "complex":   2,
    "negative": 1
}

# 情绪值映射：积极情绪数值越大，避免使用0
EMOTION_VALUE_MAP = {
    "Excited": 10, "Satisfied": 9, "Joyful": 8, "Touched": 7, "Calm": 6,
    "Conflicted": 5, "Doubtful": 4, "Hesitant": 3, "Surprised": 2, "Helpless": 1,
    "Angry": -5, "Anxious": -4, "Depressed": -3, "Fearful": -2, "Disgusted": -1
}

# 认知类型映射：越有逻辑数值越大，避免使用0
THINKING_TYPE_MAP = {
    "intuitive":1,
    "analytical":2,
    "authority_dependent":3,
    "critical":4
}
# 认知值映射：越有逻辑数值越大，避免使用0
THINKING_VALUE_MAP = {
    "Subjective":1,      "Gut Feeling":2,     "Experience-based":3,
    "Logical":4,         "Evidence-based":5,   "Data-driven":6,
    "Follow Mainstream":7,"Trust Experts":8,    "Obey Authority":9,
    "Skeptical":10,      "Questioning":11,     "Identifying Flaws":12
}

# 立场类型映射：越保守数值越大，避免使用0
STANCE_TYPE_MAP = {
    "conservative": 3,
    "neutral":      2,
    "radical":      1
}

# 立场值映射：越保守数值越大，避免使用0
STANCE_VALUE_MAP = {
    "Respect Authority": 9,
    "Emphasize Stability": 8,
    "Preserve Traditions": 7,
    "Compromise": 6,
    "Balance Perspectives": 5,
    "Pragmatic": 4,
    "Promote Change": 3,
    "Break Conventions": 2,
    "Challenge Authority": 1
}

# 意图类型映射：越温和数值越大，避免使用0
INTENTION_TYPE_MAP = {
    "observant": 4,
    "expressive": 3,
    "active":    2,
    "resistant": 1
}

# 意图值映射：越温和数值越大，避免使用0，修复空格问题
INTENTION_VALUE_MAP = {
    "Remaining Silent": 12,
    "Observing": 11,
    "Recording": 10,
    "Voting": 9,
    "Commenting": 8,
    "Writing Articles": 7,
    "Joining Discussions": 6,
    "Organizing Events": 5,
    "Advocating Actions": 4,  # 修复空格问题
    "Opposing": 3,
    "Arguing": 2,
    "Protesting": 1
}

def safe_map(value, mapping, default=0):
    """
    安全地将值映射到数字，如果找不到映射则返回默认值
    
    参数:
        value: 要映射的值
        mapping: 映射字典
        default: 默认值
        
    返回:
        int: 映射后的数值
    """
    if pd.isna(value) or value is None:
        return default
    
    # 转换为字符串进行匹配，处理可能的空格和大小写问题
    if isinstance(value, str):
        # 尝试原值匹配
        if value in mapping:
            return mapping[value]
        
        # 尝试大小写不敏感匹配
        value_lower = value.lower()
        for k, v in mapping.items():
            if k.lower() == value_lower:
                return v
        
        # 尝试部分匹配
        for k, v in mapping.items():
            if k.lower() in value_lower or value_lower in k.lower():
                return v
    
    # 如果是数字，直接返回
    try:
        return float(value)
    except (ValueError, TypeError):
        pass
    
    logger.warning(f"无法映射的值: {value}, 使用默认值: {default}")
    return default

def get_user_data_from_csv(csv_path, user_id, num_steps, filter_inactive=True):
    """
    从CSV文件中获取指定用户的数据
    
    参数:
        csv_path: CSV文件路径
        user_id: 用户ID
        num_steps: 轮次数
        filter_inactive: 是否过滤非激活状态的数据
    
    返回:
        DataFrame: 用户数据
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"成功读取CSV文件: {csv_path}, 总记录数: {len(df)}")
        
        # 过滤用户ID和轮次
        if 'user_id' in df.columns:
            df = df[df['user_id'] == user_id]
        
        if 'timestep' in df.columns:
            df = df[df['timestep'] <= num_steps]
        
        # 过滤非激活状态（如果需要）
        if filter_inactive and 'is_active' in df.columns:
            df = df[df['is_active'] != 0]
            logger.info(f"过滤非激活状态后，剩余记录数: {len(df)}")
        
        logger.info(f"用户 {user_id} 在轮次 1-{num_steps} 的数据: {len(df)} 条记录")
        
        return df
        
    except Exception as e:
        logger.error(f"读取CSV文件时发生错误: {str(e)}")
        return pd.DataFrame()

def process_cognitive_data(df):
    """
    处理用户行为数据，提取认知状态信息
    
    参数:
        df: 包含用户行为数据的DataFrame（认知状态数据直接存储在各列中）
        
    返回:
        DataFrame: 处理后的认知数据
    """
    if df.empty:
        logger.warning("输入的DataFrame为空")
        return pd.DataFrame()
    
    try:
        logger.info(f"开始处理认知数据，原始数据形状: {df.shape}")
        
        # 检查必需的认知状态列是否存在
        required_cols = ['user_id', 'timestep', 'is_active', 'mood_type', 'mood_value', 
                        'emotion_type', 'emotion_value', 'thinking_type', 'thinking_value',
                        'stance_type', 'stance_value', 'intention_type', 'intention_value']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"CSV文件中缺少必需的认知状态列: {missing_cols}")
            return pd.DataFrame()
        
        cognitive_rows = []
        
        # 遍历每一行数据，直接从相关列读取认知状态数据
        for idx, row in df.iterrows():
            try:
                # 检查是否为激活状态（如果需要过滤的话）
                if pd.isna(row.get('is_active', 1)) or row.get('is_active', 1) == 0:
                    continue
                
                # 提取认知维度（直接从CSV列读取）
                cognitive_row = {}
                
                # 基本信息
                cognitive_row['user_id'] = row.get('user_id', 0)
                cognitive_row['timestep'] = row.get('timestep', 0)
                cognitive_row['is_active'] = row.get('is_active', 1)
                
                # 情感(mood) - 直接从CSV列读取
                cognitive_row['mood_type'] = row.get('mood_type', 'neutral')
                cognitive_row['mood_value'] = safe_map(row.get('mood_value', 'Realistic'), MOOD_VALUE_MAP, 0)
                
                # 情绪(emotion) - 直接从CSV列读取
                cognitive_row['emotion_type'] = row.get('emotion_type', 'complex')
                cognitive_row['emotion_value'] = safe_map(row.get('emotion_value', 'Calm'), EMOTION_VALUE_MAP, 0)
                
                # 思维(thinking) - 直接从CSV列读取
                cognitive_row['thinking_type'] = row.get('thinking_type', 'analytical')
                cognitive_row['thinking_value'] = safe_map(row.get('thinking_value', 'Logical'), THINKING_VALUE_MAP, 0)
                
                # 立场(stance) - 直接从CSV列读取
                cognitive_row['stance_type'] = row.get('stance_type', 'neutral')
                cognitive_row['stance_value'] = safe_map(row.get('stance_value', 'Balance Perspectives'), STANCE_VALUE_MAP, 0)
                
                # 意图(intention) - 直接从CSV列读取
                cognitive_row['intention_type'] = row.get('intention_type', 'observant')
                cognitive_row['intention_value'] = safe_map(row.get('intention_value', 'Observing'), INTENTION_VALUE_MAP, 0)
                
                cognitive_rows.append(cognitive_row)
                
            except Exception as e:
                logger.warning(f"处理第 {idx} 行数据时出错: {str(e)}")
                continue
        
        # 创建DataFrame
        result_df = pd.DataFrame(cognitive_rows)
        logger.info(f"成功处理认知数据，输出数据形状: {result_df.shape}")
        
        return result_df
        
    except Exception as e:
        logger.error(f"处理认知数据时发生错误: {str(e)}")
        return pd.DataFrame()

def discover_causal_relations(cognitive_df, method="dbn_custom", user_id=0, num_steps=5, dynamic_window=None):
    """
    发现因果关系，根据指定的方法选择不同的算法
    
    参数:
        cognitive_df: 包含认知数据的DataFrame
        method: 因果发现方法 ("dbn_custom", "dbn_neural", "dbn_forest")
        user_id: 用户ID
        num_steps: 轮次数
        dynamic_window: 动态时间窗口大小（如果为None则自动计算）
    
    返回:
        tuple: (矩阵, 变量列表, 因果关系字典)
    """
    global dbn_output_format  # 声明全局变量
    
    # 确保参数类型正确
    try:
        user_id = int(user_id)
        num_steps = int(num_steps)
    except (ValueError, TypeError):
        logger.error(f"用户ID或轮次数类型错误: user_id={user_id}, num_steps={num_steps}")
        user_id = 0 if user_id is None else user_id
        num_steps = 5 if num_steps is None else num_steps
        
    if cognitive_df.empty:
        logger.warning("认知数据为空，无法执行因果分析")
        return None, None, {}

    # 对数据进行数值化处理
    numeric_df = cognitive_df.copy()

    # 确保不重复列名，使用认知维度的value列而不是type列，因为value列有更多的变化
    column_names = ["mood_value", "emotion_value", "thinking_value", "stance_value", "intention_value"]

    # 检查所有需要的列是否存在
    missing_cols = [col for col in column_names if col not in numeric_df.columns]
    if missing_cols:
        logger.error(f"缺少必要的列: {missing_cols}")
        return None, None, {}

    # 保留需要的列和元数据列
    meta_cols = []
    if 'timestep' in numeric_df.columns:
        meta_cols.append('timestep')
    if 'is_active' in numeric_df.columns:
        meta_cols.append('is_active')
    if 'user_id' in numeric_df.columns:
        meta_cols.append('user_id')

    # 保留认知列和元数据列
    numeric_df = numeric_df[column_names + meta_cols]

    # 检查列数和行数是否足够
    if len(column_names) < 2:
        logger.error(f"变量数量不足 ({len(column_names)} < 2)，无法执行因果分析")
        return None, None, {}

    if len(numeric_df) < 3:
        logger.warning(f"样本量过小 ({len(numeric_df)})，因果分析结果可能不可靠")

    # 自动计算动态时间窗口（如果没有提供）
    if dynamic_window is None:
        if 'timestep' in numeric_df.columns:
            unique_timesteps = numeric_df['timestep'].nunique()
            dynamic_window = min(10, max(3, unique_timesteps))
        else:
            dynamic_window = min(10, max(3, len(numeric_df) // 10))
    
    logger.info(f"使用动态时间窗口: {dynamic_window}")

    # 根据指定的方法选择DBN算法
    if method.lower() == "dbn" or method.lower() == "dbn_custom":
        # DBN模式下返回4个值，包括输出格式
        result = discover_causal_relations_dbn_custom(numeric_df, column_names, user_id, num_steps, dynamic_window)
        # 将结果保存到全局变量中，以便其他函数使用
        dbn_output_format = result[3] if len(result) > 3 else ""
        # 只返回前3个值，与其他方法保持一致
        return result[0], result[1], result[2]
    elif method.lower() == "dbn_neural":
        # 神经网络DBN模式
        result = discover_causal_relations_dbn_neural(numeric_df, column_names, user_id, num_steps, dynamic_window)
        dbn_output_format = result[3] if len(result) > 3 else ""
        return result[0], result[1], result[2]
    elif method.lower() == "dbn_forest":
        # 随机森林DBN模式
        result = discover_causal_relations_dbn_forest(numeric_df, column_names, user_id, num_steps, dynamic_window)
        dbn_output_format = result[3] if len(result) > 3 else ""
        return result[0], result[1], result[2]
    else:
        # 不支持的方法，默认使用dbn_custom
        logger.warning(f"不支持的方法: {method}，使用默认的dbn_custom方法")
        result = discover_causal_relations_dbn_custom(numeric_df, column_names, user_id, num_steps, dynamic_window)
        dbn_output_format = result[3] if len(result) > 3 else ""
        return result[0], result[1], result[2]

def discover_causal_relations_dbn_custom(numeric_df, column_names, user_id=0, num_steps=5, dynamic_window=None):
    """
    使用自定义动态贝叶斯网络(DBN)进行因果关系发现
    采用线性回归方法建模认知状态间的动态关系
    
    参数:
        numeric_df: 包含数值化认知数据的DataFrame
        column_names: 列名列表
        user_id: 用户ID
        num_steps: 轮次数
        dynamic_window: 动态时间窗口大小
    
    返回:
        tuple: (矩阵, 变量列表, 因果关系字典, 输出格式信息)
    """
    try:
        start_time = time.time()
        logger.info(f"使用自定义DBN算法进行因果发现 (样本量: {len(numeric_df)}, 变量: {len(column_names)})")
        
        if len(numeric_df) < 3:
            logger.warning(f"样本量过小 ({len(numeric_df)})，DBN结果可能不可靠")
        
        # 准备时间序列数据
        df_sorted = numeric_df.sort_values('timestep') if 'timestep' in numeric_df.columns else numeric_df
        
        # 构建认知状态矩阵
        cognitive_matrix = df_sorted[column_names].values
        
        n_vars = len(column_names)
        causal_matrix = np.zeros((n_vars, n_vars))
        
        # 针对小样本量的特殊处理
        if len(df_sorted) < 4:
            logger.warning(f"样本量过小 ({len(df_sorted)} < 4)，使用简化的相关性分析")
            # 对于小样本，使用简化的相关性分析
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j:
                        try:
                            # 检查是否存在变化（避免方差为0的情况）
                            x_values = cognitive_matrix[:, j]
                            y_values = cognitive_matrix[:, i]
                            
                            # 如果没有变化，使用基于数值差异的简单因果评估
                            if np.var(x_values) == 0 or np.var(y_values) == 0:
                                # 计算基于数值差异的因果强度
                                value_diff = abs(np.mean(x_values) - np.mean(y_values))
                                # 标准化到0-1范围
                                max_possible_diff = 12  # 基于我们的映射范围
                                normalized_diff = min(value_diff / max_possible_diff, 1.0)
                                
                                # 如果差异较大，认为可能存在因果关系
                                if normalized_diff > 0.2:
                                    causal_matrix[i, j] = normalized_diff
                            else:
                                # 计算皮尔逊相关系数
                                corr_coeff = np.corrcoef(x_values, y_values)[0, 1]
                                if not np.isnan(corr_coeff):
                                    # 将相关系数的绝对值作为因果强度
                                    causal_strength = abs(corr_coeff)
                                    # 对小样本使用更低的阈值
                                    if causal_strength > 0.3:  # 降低小样本的阈值
                                        causal_matrix[i, j] = causal_strength
                        except Exception as e:
                            logger.warning(f"计算小样本相关性 {j} -> {i} 时出错: {str(e)}")
                            continue
        else:
            # 使用动态时间窗口
            if dynamic_window is not None:
                window_size = min(dynamic_window, len(df_sorted) - 1)
            else:
                window_size = min(3, len(df_sorted) - 1)  # 原始的动态调整窗口大小
            
            logger.info(f"DBN自定义算法使用时间窗口大小: {window_size}")
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j:  # 避免自回归
                        try:
                            # 构建预测模型：使用前一时刻的状态预测当前时刻的状态
                            X_data = []
                            y_data = []
                            
                            for t in range(window_size, len(cognitive_matrix)):
                                # 特征：前window_size个时刻的j维度状态
                                features = cognitive_matrix[t-window_size:t, j].flatten()
                                X_data.append(features)
                                
                                # 目标：当前时刻的i维度状态
                                y_data.append(cognitive_matrix[t, i])
                            
                            if len(X_data) >= 2:  # 降低样本要求
                                X = np.array(X_data)
                                y = np.array(y_data)
                                
                                # 使用最小二乘法计算因果强度
                                try:
                                    # 添加偏置项
                                    X_with_bias = np.column_stack([np.ones(len(X)), X])
                                    
                                    # 计算回归系数
                                    coeffs = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
                                    
                                    # 计算R²作为因果强度
                                    y_pred = X_with_bias @ coeffs
                                    ss_res = np.sum((y - y_pred) ** 2)
                                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                                    
                                    # 根据样本量动态调整阈值
                                    min_threshold = 0.05 if len(df_sorted) < 10 else 0.1
                                    if r_squared > min_threshold:
                                        causal_matrix[i, j] = r_squared
                                        
                                except np.linalg.LinAlgError:
                                    # 处理矩阵奇异的情况，使用相关性作为后备
                                    try:
                                        corr_coeff = np.corrcoef(X.flatten(), y)[0, 1]
                                        if not np.isnan(corr_coeff) and abs(corr_coeff) > 0.3:
                                            causal_matrix[i, j] = abs(corr_coeff)
                                    except:
                                        continue
                                        
                        except Exception as e:
                            logger.warning(f"计算 {j} -> {i} 因果关系时出错: {str(e)}")
                            continue
        
        # 根据样本量动态调整阈值过滤
        if len(df_sorted) < 4:
            threshold = 0.2  # 小样本降低阈值，包括数值差异分析
        elif len(df_sorted) < 10:
            threshold = 0.08  # 中等样本降低阈值
        else:
            threshold = 0.12  # 大样本使用原阈值
        
        causal_matrix[causal_matrix < threshold] = 0
        
        # 构建因果关系描述
        causal_relations = {}
        dimension_names = ['Mood', 'Emotion', 'Thinking', 'Stance', 'Intention']
        
        for i in range(n_vars):
            for j in range(n_vars):
                if causal_matrix[i, j] > 0:
                    cause = dimension_names[j]
                    effect = dimension_names[i]
                    strength = causal_matrix[i, j]
                    
                    # 根据样本量和强度值调整分类标准
                    if len(df_sorted) < 4:
                        # 小样本基于相关系数的分类
                        if strength > 0.7:
                            strength_desc = "强"
                        elif strength > 0.5:
                            strength_desc = "中等"
                        else:
                            strength_desc = "弱"
                    else:
                        # 正常样本基于R²值的分类
                        if strength > 0.5:
                            strength_desc = "强"  # 高R²值表示强因果关系
                        elif strength > 0.25:
                            strength_desc = "中等"  # 中等R²值
                        else:
                            strength_desc = "弱"  # 低R²值但高于阈值
                    
                    key = f"{cause}→{effect}"
                    causal_relations[key] = {
                        'strength': strength,
                        'description': f"{cause}对{effect}有{strength_desc}的因果影响 (强度: {strength:.3f})"
                    }
        
        # 记录计算时间
        computation_time = time.time() - start_time
        model_timing_log.append({
            'model': 'dbn_custom',
            'time': computation_time,
            'timestamp': datetime.now(),
            'sample_size': len(numeric_df)
        })
        
        logger.info(f"DBN自定义算法完成，发现 {len(causal_relations)} 个因果关系，耗时: {computation_time:.3f}秒")
        
        # 生成输出格式信息
        output_format = f"使用自定义DBN模型分析了 {len(numeric_df)} 个样本，发现 {len(causal_relations)} 个因果关系"
        
        return causal_matrix, column_names, causal_relations, output_format
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"DBN自定义算法执行出错: {str(e)}")
        logger.error(f"错误详情:\n{error_trace}")
        
        # 返回空结果
        n_vars = len(column_names)
        empty_matrix = np.zeros((n_vars, n_vars))
        return empty_matrix, column_names, {}, "执行出错"

def discover_causal_relations_dbn_neural(numeric_df, column_names, user_id=0, num_steps=5, dynamic_window=None):
    """
    使用神经网络动态贝叶斯网络(Neural DBN)进行因果关系发现
    采用多层感知机(MLP)建模认知状态间的非线性动态关系
    
    参数:
        numeric_df: 包含数值化认知数据的DataFrame
        column_names: 列名列表
        user_id: 用户ID
        num_steps: 轮次数
        dynamic_window: 动态时间窗口大小
    
    返回:
        tuple: (矩阵, 变量列表, 因果关系字典, 输出格式信息)
    """
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn未安装，无法使用神经网络模型")
        return discover_causal_relations_dbn_custom(numeric_df, column_names, user_id, num_steps)
    
    try:
        start_time = time.time()
        logger.info(f"使用神经网络DBN算法进行因果发现 (样本量: {len(numeric_df)}, 变量: {len(column_names)})")
        
        # 检查样本量，降低神经网络的最小样本要求
        if len(numeric_df) < 5:
            logger.warning(f"样本量过小 ({len(numeric_df)})，神经网络无法训练")
            # 返回空结果而不是回退到其他模型
            n_vars = len(column_names)
            empty_matrix = np.zeros((n_vars, n_vars))
            return empty_matrix, column_names, {}, f"样本量不足({len(numeric_df)})，神经网络未训练"
        
        # 准备时间序列数据
        df_sorted = numeric_df.sort_values('timestep') if 'timestep' in numeric_df.columns else numeric_df
        cognitive_matrix = df_sorted[column_names].values
        
        n_vars = len(column_names)
        causal_matrix = np.zeros((n_vars, n_vars))
        
        # 根据样本量动态调整神经网络参数
        sample_size = len(df_sorted)
        if sample_size < 10:
            hidden_layer_sizes = (3,)  # 小样本使用单层简单网络
            max_iter = 500
            alpha = 0.1  # 增加正则化
        elif sample_size < 20:
            hidden_layer_sizes = (5, 3)  # 中等样本使用较小的双层网络
            max_iter = 800
            alpha = 0.05
        else:
            hidden_layer_sizes = (10, 5)  # 大样本使用原网络结构
            max_iter = 1000
            alpha = 0.01
        
        # 使用动态时间窗口
        if dynamic_window is not None:
            window_size = min(dynamic_window, len(df_sorted) - 2)
        else:
            window_size = min(3, len(df_sorted) - 2)  # 减小窗口大小以适应小样本
        
        logger.info(f"神经网络DBN算法使用时间窗口大小: {window_size}")
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:  # 避免自回归
                    try:
                        # 构建训练数据
                        X_data = []
                        y_data = []
                        
                        for t in range(window_size, len(cognitive_matrix)):
                            # 特征：包含多个维度的历史状态
                            features = []
                            for k in range(window_size):
                                features.extend(cognitive_matrix[t-window_size+k, :])  # 所有维度的历史
                            features.append(cognitive_matrix[t-1, j])  # 重点关注j维度的前一时刻
                            X_data.append(features)
                            
                            # 目标：当前时刻的i维度状态
                            y_data.append(cognitive_matrix[t, i])
                        
                        if len(X_data) >= 3:  # 降低神经网络样本要求
                            X = np.array(X_data)
                            y = np.array(y_data)
                            
                            # 改进的数据预处理和标准化
                            # 检查数据变化情况
                            X_var = np.var(X, axis=0)
                            y_var = np.var(y)
                            
                            # 如果输入特征方差过小，添加微小的噪声
                            if np.any(X_var < 1e-8):
                                noise_scale = 1e-6
                                X = X + np.random.normal(0, noise_scale, X.shape)
                            
                            # 如果输出变量方差过小，添加微小的噪声
                            if y_var < 1e-8:
                                noise_scale = 1e-6
                                y = y + np.random.normal(0, noise_scale, y.shape)
                            
                            # 使用 RobustScaler 对异常值更鲁棒
                            try:
                                from sklearn.preprocessing import RobustScaler
                                scaler_X = RobustScaler()
                                scaler_y = RobustScaler()
                            except ImportError:
                                scaler_X = StandardScaler()
                                scaler_y = StandardScaler()
                            
                            # 标准化数据
                            try:
                                X_scaled = scaler_X.fit_transform(X)
                                y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
                                
                                # 检查标准化后的数据
                                if np.any(np.isnan(X_scaled)) or np.any(np.isnan(y_scaled)):
                                    logger.warning(f"数据标准化后包含NaN值，跳过 {j} -> {i}")
                                    continue
                                    
                            except Exception as scale_error:
                                logger.warning(f"数据标准化失败 {j} -> {i}: {str(scale_error)}")
                                continue
                            
                            # 根据样本量和特征数选择合适的求解器
                            n_features = X.shape[1]
                            n_samples = X.shape[0]
                            
                            # 动态选择求解器
                            if n_samples < 10 or n_features > n_samples:
                                # 小样本或高维特征使用 adam
                                solver = 'adam'
                                learning_rate_init = 0.01
                                early_stopping = True
                                validation_fraction = 0.2 if n_samples > 5 else 0.1
                            elif n_samples < 100:
                                # 中等样本使用 lbfgs 但调整参数
                                solver = 'lbfgs'
                                learning_rate_init = 0.001  # lbfgs不使用这个参数，但为了统一
                                early_stopping = False  # lbfgs不支持early_stopping
                                validation_fraction = 0.1
                            else:
                                # 大样本使用原设置
                                solver = 'lbfgs'
                                learning_rate_init = 0.001
                                early_stopping = True
                                validation_fraction = 0.1
                            
                            # 训练神经网络
                            mlp_params = {
                                'hidden_layer_sizes': hidden_layer_sizes,
                                'activation': 'relu',  # 改用relu激活函数，收敛更稳定
                                'solver': solver,
                                'alpha': alpha,
                                'max_iter': max_iter,
                                'random_state': 42,
                                'tol': 1e-4  # 增加容忍度
                            }
                            
                            # 根据求解器添加特定参数
                            if solver == 'adam':
                                mlp_params.update({
                                    'learning_rate_init': learning_rate_init,
                                    'early_stopping': early_stopping,
                                    'validation_fraction': validation_fraction,
                                    'n_iter_no_change': 10,
                                    'beta_1': 0.9,
                                    'beta_2': 0.999
                                })
                            elif solver == 'lbfgs' and early_stopping:
                                mlp_params['early_stopping'] = early_stopping
                                mlp_params['validation_fraction'] = validation_fraction
                            
                            mlp = MLPRegressor(**mlp_params)
                            
                            # 训练神经网络，包含多重回退机制
                            training_success = False
                            current_mlp = None
                            
                            # 尝试训练，如果收敛失败则尝试简化模型
                            for attempt in range(3):  # 最多尝试3次
                                try:
                                    if attempt == 0:
                                        # 第一次尝试：使用设定的参数
                                        current_mlp = MLPRegressor(**mlp_params)
                                    elif attempt == 1:
                                        # 第二次尝试：简化网络结构
                                        simplified_params = mlp_params.copy()
                                        simplified_params['hidden_layer_sizes'] = (3,)  # 更简单的网络
                                        simplified_params['alpha'] = alpha * 2  # 增加正则化
                                        simplified_params['max_iter'] = max_iter // 2  # 减少迭代次数
                                        current_mlp = MLPRegressor(**simplified_params)
                                        logger.info(f"尝试简化网络结构: {j} -> {i}")
                                    else:
                                        # 第三次尝试：使用adam求解器
                                        fallback_params = {
                                            'hidden_layer_sizes': (2,),
                                            'activation': 'relu',
                                            'solver': 'adam',
                                            'alpha': 0.1,
                                            'max_iter': 200,
                                            'learning_rate_init': 0.01,
                                            'random_state': 42,
                                            'tol': 1e-3,
                                            'early_stopping': True,
                                            'validation_fraction': 0.2,
                                            'n_iter_no_change': 5
                                        }
                                        current_mlp = MLPRegressor(**fallback_params)
                                        logger.info(f"使用回退参数: {j} -> {i}")
                                    
                                    # 捕获警告并训练
                                    import warnings
                                    with warnings.catch_warnings():
                                        warnings.filterwarnings('ignore', category=UserWarning)
                                        warnings.filterwarnings('ignore', message='.*ConvergenceWarning.*')
                                        current_mlp.fit(X_scaled, y_scaled)
                                    
                                    training_success = True
                                    break
                                    
                                except Exception as train_error:
                                    logger.warning(f"神经网络训练尝试 {attempt + 1} 失败 {j} -> {i}: {str(train_error)}")
                                    if attempt == 2:  # 最后一次尝试失败
                                        logger.warning(f"所有训练尝试都失败，跳过 {j} -> {i}")
                                        continue  # 跳到下一个变量对
                            
                            if not training_success or current_mlp is None:
                                continue  # 训练失败，跳过这个变量对
                                
                            # 使用训练成功的模型进行预测
                            mlp = current_mlp
                            
                            # 计算预测性能
                            y_pred_scaled = mlp.predict(X_scaled)
                            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                            
                            # 计算R²作为因果强度
                            ss_res = np.sum((y - y_pred) ** 2)
                            ss_tot = np.sum((y - np.mean(y)) ** 2)
                            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                            
                            # 使用特征重要性分析增强因果强度评估
                            # 通过扰动j维度特征来评估重要性
                            X_perturbed = X_scaled.copy()
                            X_perturbed[:, -1] = np.random.permutation(X_perturbed[:, -1])  # 随机打乱j维度特征
                            y_pred_perturbed_scaled = mlp.predict(X_perturbed)
                            y_pred_perturbed = scaler_y.inverse_transform(y_pred_perturbed_scaled.reshape(-1, 1)).flatten()
                            
                            # 计算扰动后的性能下降
                            ss_res_perturbed = np.sum((y - y_pred_perturbed) ** 2)
                            r_squared_perturbed = 1 - (ss_res_perturbed / ss_tot) if ss_tot > 0 else 0
                            
                            importance_score = max(0, r_squared - r_squared_perturbed)
                            
                            # 综合评分
                            final_score = (r_squared + importance_score) / 2
                            
                            # 根据样本量动态调整阈值
                            if sample_size < 10:
                                # 小样本使用更低的阈值
                                min_threshold = 0.3
                                r2_threshold = 0.05
                            elif sample_size < 20:
                                # 中等样本使用中等阈值
                                min_threshold = 0.4
                                r2_threshold = 0.06
                            else:
                                # 大样本使用原阈值
                                min_threshold = 0.505
                                r2_threshold = 0.08
                            
                            if final_score > min_threshold and r_squared > r2_threshold:
                                causal_matrix[i, j] = final_score
                                
                    except Exception as e:
                        logger.warning(f"计算神经网络因果关系 {j} -> {i} 时出错: {str(e)}")
                        continue
        
        # 根据样本量动态调整全局阈值过滤
        if sample_size < 10:
            threshold = 0.3  # 小样本使用更低的阈值
        elif sample_size < 20:
            threshold = 0.4  # 中等样本使用中等阈值
        else:
            threshold = 0.505  # 大样本使用原阈值
        
        causal_matrix[causal_matrix < threshold] = 0
        
        # 构建因果关系描述
        causal_relations = {}
        dimension_names = ['Mood', 'Emotion', 'Thinking', 'Stance', 'Intention']
        
        for i in range(n_vars):
            for j in range(n_vars):
                if causal_matrix[i, j] > 0:
                    cause = dimension_names[j]
                    effect = dimension_names[i]
                    strength = causal_matrix[i, j]
                    
                    # 基于实际测试数据分布的神经网络因果强度分类
                    # 根据dbn_neural_causal_results.json数据优化分类标准
                    if strength > 0.56:  # 约3-5%的关系为很强影响 (如0.566)
                        strength_desc = "很强"
                    elif strength > 0.535:  # 约8-12%的关系为强影响 (如0.545-0.549)
                        strength_desc = "强"
                    elif strength > 0.505:  # 约75-80%的关系为中等影响 (如0.506-0.535)
                        strength_desc = "中等"
                    else:
                        strength_desc = "弱"  # 其余为弱影响
                    
                    key = f"{cause}→{effect}"
                    causal_relations[key] = {
                        'strength': strength,
                        'description': f"{cause}对{effect}有{strength_desc}的非线性因果影响 (强度: {strength:.3f})"
                    }
        
        # 记录计算时间
        computation_time = time.time() - start_time
        model_timing_log.append({
            'model': 'dbn_neural',
            'time': computation_time,
            'timestamp': datetime.now(),
            'sample_size': len(numeric_df)
        })
        
        logger.info(f"神经网络DBN算法完成，发现 {len(causal_relations)} 个因果关系，耗时: {computation_time:.3f}秒")
        
        output_format = f"使用神经网络DBN模型分析了 {len(numeric_df)} 个样本，发现 {len(causal_relations)} 个非线性因果关系"
        
        return causal_matrix, column_names, causal_relations, output_format
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"神经网络DBN算法执行出错: {str(e)}")
        logger.error(f"错误详情:\n{error_trace}")
        
        # 回退到线性模型
        logger.info("回退到线性DBN模型")
        return discover_causal_relations_dbn_custom(numeric_df, column_names, user_id, num_steps)

def discover_causal_relations_dbn_forest(numeric_df, column_names, user_id=0, num_steps=5, dynamic_window=None):
    """
    使用随机森林动态贝叶斯网络(Random Forest DBN)进行因果关系发现
    采用随机森林回归建模认知状态间的非线性动态关系
    
    参数:
        numeric_df: 包含数值化认知数据的DataFrame
        column_names: 列名列表
        user_id: 用户ID
        num_steps: 轮次数
        dynamic_window: 动态时间窗口大小
    
    返回:
        tuple: (矩阵, 变量列表, 因果关系字典, 输出格式信息)
    """
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn未安装，无法使用随机森林模型")
        return discover_causal_relations_dbn_custom(numeric_df, column_names, user_id, num_steps)
    
    try:
        start_time = time.time()
        logger.info(f"使用随机森林DBN算法进行因果发现 (样本量: {len(numeric_df)}, 变量: {len(column_names)})")
        
        # 检查样本量
        if len(numeric_df) < 3:
            logger.warning(f"样本量过小 ({len(numeric_df)})，回退到线性DBN模型")
            return discover_causal_relations_dbn_custom(numeric_df, column_names, user_id, num_steps)
        
        # 准备时间序列数据
        df_sorted = numeric_df.sort_values('timestep') if 'timestep' in numeric_df.columns else numeric_df
        cognitive_matrix = df_sorted[column_names].values
        
        n_vars = len(column_names)
        causal_matrix = np.zeros((n_vars, n_vars))
        
        # 随机森林参数
        n_estimators = 100
        max_depth = 10
        
        # 使用动态时间窗口
        if dynamic_window is not None:
            window_size = min(dynamic_window, len(df_sorted) - 2)
        else:
            window_size = min(3, len(df_sorted) - 2)
        
        logger.info(f"随机森林DBN算法使用时间窗口大小: {window_size}")
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:  # 避免自回归
                    try:
                        # 构建训练数据
                        X_data = []
                        y_data = []
                        
                        for t in range(window_size, len(cognitive_matrix)):
                            # 特征：历史状态窗口 + 当前其他维度状态
                            features = []
                            
                            # 添加j维度的历史状态
                            for k in range(window_size):
                                features.append(cognitive_matrix[t-window_size+k, j])
                            
                            # 添加当前时刻其他维度的状态（除了目标维度i）
                            for dim in range(n_vars):
                                if dim != i and dim != j:
                                    features.append(cognitive_matrix[t-1, dim])
                            
                            X_data.append(features)
                            
                            # 目标：当前时刻的i维度状态
                            y_data.append(cognitive_matrix[t, i])
                        
                        if len(X_data) >= 6:  # 随机森林需要足够样本
                            X = np.array(X_data)
                            y = np.array(y_data)
                            
                            # 训练随机森林
                            rf = RandomForestRegressor(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                random_state=42,
                                n_jobs=-1
                            )
                            
                            try:
                                rf.fit(X, y)
                                
                                # 计算预测性能
                                y_pred = rf.predict(X)
                                
                                # 计算R²作为基础因果强度
                                ss_res = np.sum((y - y_pred) ** 2)
                                ss_tot = np.sum((y - np.mean(y)) ** 2)
                                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                                
                                # 使用特征重要性增强因果强度评估
                                feature_importances = rf.feature_importances_
                                
                                # j维度特征的重要性（前window_size个特征）
                                j_importance = np.sum(feature_importances[:window_size])
                                
                                # 综合评分：结合预测性能和特征重要性
                                combined_score = (r_squared + j_importance) / 2
                                
                                # 使用Gini重要性作为额外验证
                                gini_importance = j_importance
                                
                                # 最终评分
                                final_score = (combined_score + gini_importance) / 2
                                
                                # 只保留显著的因果关系
                                # 随机森林模型使用相对宽松的阈值（保留更多非线性关系）
                                if final_score > 0.08 and r_squared > 0.05:
                                    causal_matrix[i, j] = final_score
                                    
                            except Exception as rf_error:
                                logger.warning(f"随机森林训练失败 {j} -> {i}: {str(rf_error)}")
                                continue
                                
                    except Exception as e:
                        logger.warning(f"计算随机森林因果关系 {j} -> {i} 时出错: {str(e)}")
                        continue
        
        # 应用阈值过滤（随机森林能够捉捉非线性关系，阈值相对较低）
        threshold = 0.1  # 适当降低阈值以保留更多非线性因果关系
        causal_matrix[causal_matrix < threshold] = 0
        
        # 构建因果关系描述
        causal_relations = {}
        dimension_names = ['Mood', 'Emotion', 'Thinking', 'Stance', 'Intention']
        
        for i in range(n_vars):
            for j in range(n_vars):
                if causal_matrix[i, j] > 0:
                    cause = dimension_names[j]
                    effect = dimension_names[i]
                    strength = causal_matrix[i, j]
                    
                    # 随机森林DBN模型的因果强度分类（能捉捉非线性关系）
                    if strength > 0.5:
                        strength_desc = "很强"  # 高非线性因果强度
                    elif strength > 0.3:
                        strength_desc = "强"  # 中等非线性因果强度
                    elif strength > 0.15:
                        strength_desc = "中等"  # 较弱但显著的非线性关系
                    else:
                        strength_desc = "弱"  # 弱但高于阈值的关系
                    
                    key = f"{cause}→{effect}"
                    causal_relations[key] = {
                        'strength': strength,
                        'description': f"{cause}对{effect}有{strength_desc}的决策树因果影响 (强度: {strength:.3f})"
                    }
        
        # 记录计算时间
        computation_time = time.time() - start_time
        model_timing_log.append({
            'model': 'dbn_forest',
            'time': computation_time,
            'timestamp': datetime.now(),
            'sample_size': len(numeric_df)
        })
        
        logger.info(f"随机森林DBN算法完成，发现 {len(causal_relations)} 个因果关系，耗时: {computation_time:.3f}秒")
        
        output_format = f"使用随机森林DBN模型分析了 {len(numeric_df)} 个样本，发现 {len(causal_relations)} 个基于决策树的因果关系"
        
        return causal_matrix, column_names, causal_relations, output_format
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"随机森林DBN算法执行出错: {str(e)}")
        logger.error(f"错误详情:\n{error_trace}")
        
        # 回退到线性模型
        logger.info("回退到线性DBN模型")
        return discover_causal_relations_dbn_custom(numeric_df, column_names, user_id, num_steps)

def generate_causal_descriptions(relations_data, user_id, num_steps):
    """
    生成因果关系的自然语言描述
    
    参数:
        relations_data: 因果关系字典
        user_id: 用户ID
        num_steps: 轮次数
    
    返回:
        list: 因果关系描述的列表
    """
    descriptions = []
    
    if not relations_data:
        descriptions.append(f"用户 {user_id} 在轮次 1-{num_steps} 期间，各认知维度之间未发现显著的因果关系。")
        return descriptions
    
    descriptions.append(f"用户 {user_id} 在轮次 1-{num_steps} 期间的认知因果关系分析：")
    
    # 按因果强度排序
    sorted_relations = sorted(relations_data.items(), 
                            key=lambda x: x[1]['strength'] if isinstance(x[1], dict) else 0, 
                            reverse=True)
    
    for relation_key, relation_info in sorted_relations:
        if isinstance(relation_info, dict) and 'description' in relation_info:
            descriptions.append(f"- {relation_info['description']}")
        else:
            descriptions.append(f"- {relation_key}: {relation_info}")
    
    return descriptions

def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    
    参数:
        vec1, vec2: 输入向量
    
    返回:
        float: 余弦相似度值
    """
    try:
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        return dot_product / (norm_vec1 * norm_vec2)
    except:
        return 0.0

def enhanced_cognitive_similarity(user1_data, user2_data, weights=None):
    """
    增强的认知相似度计算，基于认知状态的标签相似性
    
    参数:
        user1_data: 用户1的认知数据DataFrame
        user2_data: 用户2的认知数据DataFrame  
        weights: 各维度权重 {'mood': w1, 'emotion': w2, 'thinking': w3, 'stance': w4, 'intention': w5}
    
    返回:
        float: 综合相似度分数 (0-1)
    """
    if weights is None:
        weights = {'mood': 0.2, 'emotion': 0.2, 'thinking': 0.2, 'stance': 0.2, 'intention': 0.2}
    
    try:
        # 提取认知标签
        dimensions = ['mood', 'emotion', 'thinking', 'stance', 'intention']
        similarities = []
        
        for dim in dimensions:
            type_col = f"{dim}_type"
            value_col = f"{dim}_value"
            
            # 计算类型相似度
            type_similarity = 0.0
            if type_col in user1_data.columns and type_col in user2_data.columns:
                user1_types = user1_data[type_col].value_counts(normalize=True)
                user2_types = user2_data[type_col].value_counts(normalize=True)
                
                # 计算类型分布的重叠度
                common_types = set(user1_types.index) & set(user2_types.index)
                for ctype in common_types:
                    type_similarity += min(user1_types[ctype], user2_types[ctype])
            
            # 计算数值相似度
            value_similarity = 0.0
            if value_col in user1_data.columns and value_col in user2_data.columns:
                user1_values = user1_data[value_col].dropna()
                user2_values = user2_data[value_col].dropna()
                
                if len(user1_values) > 0 and len(user2_values) > 0:
                    # 使用数值化的认知状态计算相似度
                    u1_mean = user1_values.mean() if isinstance(user1_values.iloc[0], (int, float)) else \
                             safe_map(user1_values.mode().iloc[0] if len(user1_values.mode()) > 0 else user1_values.iloc[0], 
                                    eval(f"{dim.upper()}_VALUE_MAP"), 0)
                    u2_mean = user2_values.mean() if isinstance(user2_values.iloc[0], (int, float)) else \
                             safe_map(user2_values.mode().iloc[0] if len(user2_values.mode()) > 0 else user2_values.iloc[0], 
                                    eval(f"{dim.upper()}_VALUE_MAP"), 0)
                    
                    # 标准化相似度计算
                    max_diff = 20  # 基于映射范围的最大差异
                    value_similarity = max(0, 1 - abs(u1_mean - u2_mean) / max_diff)
            
            # 综合类型和数值相似度
            dim_similarity = (type_similarity * 0.3 + value_similarity * 0.7)
            similarities.append(dim_similarity * weights[dim])
        
        return sum(similarities)
        
    except Exception as e:
        logger.warning(f"计算认知相似度时出错: {str(e)}")
        return 0.0

def get_enhanced_similar_users_data(csv_path, user_id, num_steps, target_samples=200, min_window=3, max_window=10, filter_inactive=True):
    """
    增强的相似用户数据获取，实现动态样本扩展
    
    参数:
        csv_path: CSV文件路径
        user_id: 目标用户ID
        num_steps: 轮次数
        target_samples: 目标样本总数
        min_window: 最小时间窗口
        max_window: 最大时间窗口
        filter_inactive: 是否过滤非激活状态的数据
    
    返回:
        tuple: (合并后的相似用户数据DataFrame, 实际样本数, 时间窗口大小)
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"读取CSV文件，总记录数: {len(df)}")
        
        # 过滤非激活状态
        if filter_inactive and 'is_active' in df.columns:
            df = df[df['is_active'] != 0]
            logger.info(f"过滤非激活状态后，剩余记录数: {len(df)}")
        
        # 获取目标用户的数据
        target_user_data = df[df['user_id'] == user_id] if 'user_id' in df.columns else df
        
        if target_user_data.empty:
            logger.warning(f"目标用户 {user_id} 没有数据")
            return pd.DataFrame(), 0, min_window
        
        # 动态确定时间窗口大小
        user_data_by_step = target_user_data.groupby('timestep').size() if 'timestep' in target_user_data.columns else {}
        available_steps = len(user_data_by_step)
        
        # 根据可用数据动态调整时间窗口
        if available_steps >= max_window:
            time_window = max_window
        elif available_steps >= min_window:
            time_window = available_steps
        else:
            time_window = min_window
        
        # 过滤时间步（使用动态窗口）
        if 'timestep' in df.columns:
            df = df[df['timestep'] <= num_steps]
        
        # 提取目标用户的认知数据用于相似度计算
        target_cognitive = process_cognitive_data(target_user_data)
        
        if target_cognitive.empty:
            logger.warning(f"目标用户 {user_id} 没有认知状态数据")
            return pd.DataFrame(), 0, time_window
        
        # 使用增强的相似度计算
        user_similarities = []
        unique_users = df['user_id'].unique() if 'user_id' in df.columns else []
        
        logger.info(f"开始计算与 {len(unique_users)} 个用户的相似度")
        
        for other_user_id in unique_users:
            if other_user_id != user_id:
                other_user_data = df[df['user_id'] == other_user_id]
                other_cognitive = process_cognitive_data(other_user_data)
                
                if not other_cognitive.empty:
                    # 使用增强的认知相似度计算
                    similarity = enhanced_cognitive_similarity(target_cognitive, other_cognitive)
                    if similarity > 0.1:  # 过滤相似度过低的用户
                        user_similarities.append((other_user_id, similarity, len(other_cognitive)))
        
        # 按相似度排序
        user_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 动态选择相似用户，直到达到目标样本数
        selected_users = [user_id]  # 始终包含目标用户
        current_samples = len(target_cognitive)
        
        logger.info(f"目标用户 {user_id} 有 {current_samples} 个样本")
        
        for other_user_id, similarity, user_sample_count in user_similarities:
            if current_samples >= target_samples:
                break
            
            selected_users.append(other_user_id)
            current_samples += user_sample_count
            logger.debug(f"添加相似用户 {other_user_id} (相似度: {similarity:.3f}, 样本数: {user_sample_count})")
            
            # 如果相似用户太多，限制数量以避免计算过慢
            if len(selected_users) >= 50:  # 最多50个用户
                break
        
        # 合并选定用户的数据
        similar_users_data = df[df['user_id'].isin(selected_users)]
        
        final_sample_count = len(process_cognitive_data(similar_users_data))
        
        logger.info(f"选择了 {len(selected_users)} 个相似用户，最终样本数: {final_sample_count}, 时间窗口: {time_window}")
        
        return similar_users_data, final_sample_count, time_window
        
    except Exception as e:
        logger.error(f"获取增强相似用户数据时发生错误: {str(e)}")
        return pd.DataFrame(), 0, min_window

def get_similar_users_data(csv_path, user_id, num_steps, top_n=20, filter_inactive=True):
    """
    保持向后兼容的相似用户数据获取函数
    """
    similar_data, sample_count, _ = get_enhanced_similar_users_data(csv_path, user_id, num_steps, 
                                                                  target_samples=top_n*3, 
                                                                  filter_inactive=filter_inactive)
    return similar_data

def get_all_users_data(csv_path, num_steps, filter_inactive=True):
    """
    获取所有用户的数据，用于全局分析
    
    参数:
        csv_path: CSV文件路径
        num_steps: 轮次数
        filter_inactive: 是否过滤非激活状态的数据
    
    返回:
        DataFrame: 所有用户的数据
    """
    try:
        df = pd.read_csv(csv_path)
        
        # 过滤时间步
        if 'timestep' in df.columns:
            df = df[df['timestep'] <= num_steps]
        
        # 过滤非激活状态
        if filter_inactive and 'is_active' in df.columns:
            df = df[df['is_active'] != 0]
        
        logger.info(f"获取所有用户数据，包含 {len(df)} 条记录")
        
        return df
        
    except Exception as e:
        logger.error(f"获取所有用户数据时发生错误: {str(e)}")
        return pd.DataFrame()

def get_causal_relations_for_simulation(user_id, num_steps, csv_path, method="dbn_custom", merge=False, merge_mode="similar", top_n=20, filter_inactive=True, enable_smart_merge=True, target_samples=200, min_steps_for_causal=3):
    """
    为模拟过程提供的函数，返回用户在指定轮次的因果关系描述

    参数:
        user_id: 用户ID
        num_steps: 轮次数
        csv_path: CSV文件路径
        method: 因果发现方法 ("dbn_custom", "dbn_neural", "dbn_forest")
        merge: 是否使用合并模式
        merge_mode: 合并模式 ("all" 或 "similar")
        top_n: 合并模式下，选取的相似用户数量
        filter_inactive: 是否过滤未激活状态的数据
        target_samples: 目标样本数量
        min_steps_for_causal: 开始因果分析的最小轮次数

    返回:
        list: 因果关系描述的列表，每个元素是一个字符串
    """
    try:
        logger.info(f"模拟过程调用: 获取用户 {user_id} 在轮次 {num_steps} 的因果关系")

        # 检查是否达到最小轮次要求
        if num_steps < min_steps_for_causal:
            logger.info(f"用户 {user_id} 轮次 {num_steps} 小于最小要求 {min_steps_for_causal}，跳过因果分析")
            return [f"用户 {user_id} 轮次不足，需要至少 {min_steps_for_causal} 轮数据才能进行因果分析。"]

        # 始终使用增强的相似用户数据获取（除非明确指定不使用）
        if merge and merge_mode == "all":
            # 获取所有用户的数据
            merged_df = get_all_users_data(csv_path, num_steps, filter_inactive)
            cognitive_df = process_cognitive_data(merged_df)
            sample_count = len(cognitive_df) if not cognitive_df.empty else 0
            time_window = min(10, num_steps)
            
            if cognitive_df.empty:
                logger.info(f"全局模式下没有找到符合条件的用户数据，用户{user_id}第{num_steps}轮")
                return []
                
            logger.info(f"全局模式下共有 {sample_count} 条认知数据")
            
        else:
            # 使用增强的相似用户数据获取
            merged_df, sample_count, time_window = get_enhanced_similar_users_data(
                csv_path, user_id, num_steps, 
                target_samples=target_samples, 
                min_window=3, max_window=10, 
                filter_inactive=filter_inactive
            )

            if merged_df.empty:
                logger.warning(f"用户 {user_id} 在轮次 1 到 {num_steps} 之间没有数据")
                return []

            # 处理并提取认知状态数据
            cognitive_df = process_cognitive_data(merged_df)

            if cognitive_df.empty:
                logger.warning(f"用户 {user_id} 在轮次 1 到 {num_steps} 之间没有认知状态数据")
                return []
            
            # 更新实际样本数
            actual_sample_count = len(cognitive_df)
            logger.info(f"实际获得样本数: {actual_sample_count}, 动态时间窗口: {time_window}")
            
            # 如果样本数仍然不足，尝试扩大搜索范围
            if actual_sample_count < 50 and enable_smart_merge:
                logger.info(f"样本数不足({actual_sample_count})，尝试扩大搜索范围")
                
                # 尝试更大的目标样本数
                expanded_df, expanded_count, expanded_window = get_enhanced_similar_users_data(
                    csv_path, user_id, num_steps, 
                    target_samples=target_samples * 2,  # 扩大搜索范围
                    min_window=3, max_window=10, 
                    filter_inactive=filter_inactive
                )
                
                if expanded_count > actual_sample_count:
                    cognitive_df = process_cognitive_data(expanded_df)
                    time_window = expanded_window
                    logger.info(f"扩大搜索后获得样本数: {len(cognitive_df)}")

        # 执行因果发现，传递动态时间窗口信息
        causal_matrix, variables, relations = discover_causal_relations(
            cognitive_df, method, user_id, num_steps, dynamic_window=time_window
        )

        if causal_matrix is None:
            logger.warning(f"用户 {user_id} 在轮次 1 到 {num_steps} 之间没有有效的因果关系")
            return []

        # 生成描述性文本
        descriptions = generate_causal_descriptions(relations, user_id, num_steps)

        # 构建数值结果字典（用于可视化和分析）
        numerical_results = {
            'user_id': user_id,
            'num_steps': num_steps,
            'method': method,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'merge_mode': 'enhanced_similar' if not merge else merge_mode,
            'sample_size': len(cognitive_df),
            'time_window': time_window,
            'target_samples': target_samples,
            'variables': variables.tolist() if hasattr(variables, 'tolist') else variables,
            'causal_matrix': causal_matrix.tolist() if hasattr(causal_matrix, 'tolist') else causal_matrix,
            'relations': relations,
            'descriptions': descriptions
        }

        # 为了保持向后兼容性，仍然返回描述列表
        # 但同时将数值结果存储在全局变量中供后续保存使用
        global_results_key = f"{method}_numerical_results"
        if not hasattr(get_causal_relations_for_simulation, global_results_key):
            setattr(get_causal_relations_for_simulation, global_results_key, [])
        
        results_list = getattr(get_causal_relations_for_simulation, global_results_key)
        results_list.append(numerical_results)

        # 返回描述列表（保持向后兼容性）
        return descriptions

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"获取因果关系时发生错误: {str(e)}")
        logger.error(f"错误详情:\n{error_trace}")
        return []

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASCE系统因果分析工具（简化版）")
    parser.add_argument("--user_id", type=int, help="用户ID")
    parser.add_argument("--num_steps", type=int, default=5, help="轮次数")
    parser.add_argument("--csv_path", type=str, help="CSV文件路径")
    parser.add_argument("--method", type=str, default="dbn_custom", 
                       choices=["dbn_custom", "dbn_neural", "dbn_forest"], 
                       help="因果发现方法")
    parser.add_argument("--merge", action="store_true", help="是否使用合并模式")
    parser.add_argument("--merge_mode", type=str, default="similar", 
                       choices=["all", "similar"], help="合并模式")
    parser.add_argument("--top_n", type=int, default=20, help="相似用户数量")
    parser.add_argument("--no_filter_inactive", action="store_true", help="不过滤未激活状态")

    args = parser.parse_args()

    # 参数验证
    if not args.csv_path:
        print("错误: 必须指定CSV文件路径")
        exit(1)

    filter_inactive = not args.no_filter_inactive

    # 执行因果分析
    try:
        descriptions = get_causal_relations_for_simulation(
            user_id=args.user_id or 0,
            num_steps=args.num_steps,
            csv_path=args.csv_path,
            method=args.method,
            merge=args.merge,
            merge_mode=args.merge_mode,
            top_n=args.top_n,
            filter_inactive=filter_inactive
        )

        print(f"\n=== 因果关系分析结果 ({args.method}) ===")
        for desc in descriptions:
            print(desc)

    except Exception as e:
        print(f"执行过程中发生错误: {str(e)}")
        exit(1)