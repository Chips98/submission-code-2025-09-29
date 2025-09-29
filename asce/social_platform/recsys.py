# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
'''Note that you need to check if it exceeds max_rec_post_len when writing
into rec_matrix'''
import heapq
import logging
import os
import random
import time
import json
from ast import literal_eval
from datetime import datetime
from math import log
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# 有条件导入torch相关库
try:
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoTokenizer
    from .process_recsys_posts import generate_post_vector
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch相关库未安装，将使用基于认知链的备选推荐系统")

from .typing import ActionType, RecsysType

rec_log = logging.getLogger(name='social.rec')
rec_log.setLevel('DEBUG')

# 认知状态映射常量
# 情感(mood)
MOOD_TYPE_MAP = {
    "positive":  2,
    "neutral":   0,
    "negative": -1
}

# value 按极性映射为 ±1 ~ ±5，中性为 0
MOOD_VALUE_MAP = {
    "Optimistic": +5, "Confident": +4, "Passionate": +3, "Empathetic": +2, "Grateful": +1,
    "Realistic":  0,  "Rational": 0,   "Prudent": 0,     "Detached": 0,    "Objective": 0,
    "Pessimistic": -1, "Apathetic": -2, "Distrustful": -3, "Cynical": -4, "Resentful": -5
}

# 情绪(Emotion)
EMOTION_TYPE_MAP = {
    "positive":  2,
    "negative": -2,
    "complex":   0
}

EMOTION_VALUE_MAP = {
    "Excited": +5, "Satisfied": +4, "Joyful": +3, "Touched": +2, "Calm": +1,
    "Conflicted": 0, "Doubtful": 0, "Hesitant": 0, "Surprised": 0, "Helpless": 0,
    "Angry": -5, "Anxious": -4, "Depressed": -3, "Fearful": -2, "Disgusted": -1
}

# 思维(Thinking)
THINKING_TYPE_MAP = {
    "intuitive": 1,
    "analytical": 2,
    "authority_dependent": 3,
    "critical": 4
}

THINKING_VALUE_MAP = {
    "Gut Feeling": 1,     "Experience-based": 2, "Subjective": 3,
    "Logical": 4,         "Evidence-based": 5,   "Data-driven": 6,
    "Follow Mainstream": 7,"Trust Experts": 8,    "Obey Authority": 9,
    "Skeptical": 10,      "Questioning": 11,     "Identifying Flaws": 12
}

# 立场(Stance)
STANCE_TYPE_MAP = {
    "conservative": 1,
    "radical":     -1,
    "neutral":      0
}

STANCE_VALUE_MAP = {
    "Respect Authority": +3,
    "Emphasize Stability": +2,
    "Preserve Traditions": +1,
    "Compromise": 0,
    "Balance Perspectives": 0,
    "Pragmatic": 0,
    "Promote Change": -1,
    "Break Conventions": -2,
    "Challenge Authority": -3
}

# 意图(Intention)
INTENTION_TYPE_MAP = {
    "expressive": 0,
    "active":    -1,
    "observant": +1,
    "resistant": -2
}

INTENTION_VALUE_MAP = {
    "Remaining Silent": +3,
    "Observing": +2,
    "Recording": +1,
    "Voting": 0,
    "Commenting": -1,
    "Writing Articles": -2,
    "Joining Discussions": -3,
    "Organizing Events": -4,
    "AdvocatingActions": -5,
    "Opposing": -6,
    "Arguing": -7,
    "Protesting": -8
}

# 观点(Viewpoint)
VIEWPOINT_LABEL_MAP = {
    "Strongly Support": 2,
    "Strongly support": 2,
    "StronglySupport": 2,
    "Moderate Support": 1,
    "Moderate support": 1,
    "ModerateSupport": 1,
    "Indifferent": 0,
    "Do Not Support": -1,
    "Do not support": -1,
    "DoNotSupport": -1,
    "Moderate Opposition": -1,
    "Moderate opposition": -1,
    "ModerateOpposition": -1,
    "Strongly Oppose": -2,
    "Strongly oppose": -2,
    "StronglyOppose": -2
}

# 将load_model函数定义移到前面
def load_model(model_name, try_local_first=False):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch相关库未安装，无法加载模型")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_name == 'paraphrase-MiniLM-L6-v2':
            return SentenceTransformer(model_name,
                                       device=device,
                                       cache_folder="./models")
        elif model_name == 'Twitter/twhin-bert-base':
            if try_local_first:
                try:
                    print("尝试从本地缓存加载模型...")
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        local_files_only=True,
                        cache_dir="./models",
                        model_max_length=512)
                    model = AutoModel.from_pretrained(
                        model_name,
                        local_files_only=True,
                        cache_dir="./models").to(device)
                    print("成功从本地缓存加载模型")
                    return tokenizer, model
                except Exception as local_err:
                    print(f"从本地缓存加载失败，尝试在线下载: {str(local_err)}")

            # 尝试在线下载
            tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                    model_max_length=512,
                                                    cache_dir="./models")
            model = AutoModel.from_pretrained(model_name,
                                            cache_dir="./models").to(device)
            return tokenizer, model
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    except Exception as e:
        raise Exception(f"Failed to load the model: {model_name}") from e


# 构建用户认知向量的函数
def build_cognitive_vector(cognitive_info: Dict[str, Any]) -> np.ndarray:
    """
    根据智能体的认知信息，生成一个认知向量

    参数:
        cognitive_info: 包含认知信息的字典，包括情感、情绪、认知、立场、意图及观点支持级别

    返回:
        np.ndarray: 归一化后的认知向量
    """
    try:
        # 提取认知状态信息，使用默认值处理可能缺失的键
        mood_type = cognitive_info.get("mood_type", "neutral")
        mood_value = cognitive_info.get("mood_value", "Realistic")
        emotion_type = cognitive_info.get("emotion_type", "complex")
        emotion_value = cognitive_info.get("emotion_value", "Conflicted")
        thinking_type = cognitive_info.get("thinking_type", "analytical")
        thinking_value = cognitive_info.get("thinking_value", "Logical")
        stance_type = cognitive_info.get("stance_type", "neutral")
        stance_value = cognitive_info.get("stance_value", "Compromise")
        intention_type = cognitive_info.get("intention_type", "expressive")
        intention_value = cognitive_info.get("intention_value", "Commenting")

        # 获取观点支持级别
        viewpoints = [
            cognitive_info.get("viewpoint_1", "Indifferent"),
            cognitive_info.get("viewpoint_2", "Indifferent"),
            cognitive_info.get("viewpoint_3", "Indifferent"),
            cognitive_info.get("viewpoint_4", "Indifferent"),
            cognitive_info.get("viewpoint_5", "Indifferent"),
            cognitive_info.get("viewpoint_6", "Indifferent")
        ]

        # 将认知状态映射为数值，使用默认值处理可能无效的键
        mood_type_score = MOOD_TYPE_MAP.get(mood_type, 0)
        mood_value_score = MOOD_VALUE_MAP.get(mood_value, 0)
        emotion_type_score = EMOTION_TYPE_MAP.get(emotion_type, 0)
        emotion_value_score = EMOTION_VALUE_MAP.get(emotion_value, 0)
        thinking_type_score = THINKING_TYPE_MAP.get(thinking_type, 2)  # 默认为analytical
        thinking_value_score = THINKING_VALUE_MAP.get(thinking_value, 4)  # 默认为Logical
        stance_type_score = STANCE_TYPE_MAP.get(stance_type, 0)
        stance_value_score = STANCE_VALUE_MAP.get(stance_value, 0)
        intention_type_score = INTENTION_TYPE_MAP.get(intention_type, 0)
        intention_value_score = INTENTION_VALUE_MAP.get(intention_value, -1)  # 默认为Commenting

        # 将观点支持级别映射为数值
        viewpoint_scores = [VIEWPOINT_LABEL_MAP.get(vp, 0) for vp in viewpoints]

        # 创建打印语句显示原始值
        #rec_log.info(f"[DEBUG-认知向量] 原始认知信息: {cognitive_info}")

        # 构建认知向量
        cognitive_vector = np.array([
            mood_type_score, mood_value_score,
            emotion_type_score, emotion_value_score,
            thinking_type_score, thinking_value_score,
            stance_type_score, stance_value_score,
            intention_type_score, intention_value_score,
            *viewpoint_scores
        ], dtype=float)

        #rec_log.info(f"[DEBUG-认知向量] 生成的认知向量: {cognitive_vector}")

        # 不要归一化向量，保留原始特征差异
        # 或者使用其他归一化方法，确保保留差异性
        # norm = np.linalg.norm(cognitive_vector)
        # if norm > 0:
        #     cognitive_vector = cognitive_vector / norm

        return cognitive_vector
    except Exception as e:
        rec_log.error(f"构建认知向量时出错: {e}，返回默认向量")
        # 返回一个默认的归一化向量（全零向量）
        default_vector = np.zeros(16, dtype=float)  # 10个认知维度 + 6个观点
        default_vector[0] = 1.0  # 确保向量不全为零
        return default_vector / np.linalg.norm(default_vector)


# 计算认知相似度的函数
def compute_cognitive_similarity(user_cognitive: Dict[str, Any], author_cognitive: Dict[str, Any]) -> float:
    try:
        # 构建认知向量
        user_vector = build_cognitive_vector(user_cognitive)
        author_vector = build_cognitive_vector(author_cognitive)

        # 使用余弦相似度计算
        from sklearn.metrics.pairwise import cosine_similarity
        user_vector_reshaped = user_vector.reshape(1, -1)
        author_vector_reshaped = author_vector.reshape(1, -1)

        # 计算余弦相似度
        similarity = cosine_similarity(user_vector_reshaped, author_vector_reshaped)[0][0]

        #rec_log.info(f"[DEBUG-相似度计算] 用户向量: {user_vector}")
        #rec_log.info(f"[DEBUG-相似度计算] 作者向量: {author_vector}")
        #rec_log.info(f"[DEBUG-相似度计算] 余弦相似度: {similarity}")

        return similarity
    except Exception as e:
        rec_log.error(f"计算认知相似度时出错: {e}，返回默认值0.0")
        return 0.0

# Initially set to None, to be assigned once again in the recsys function
model = None
twhin_tokenizer, twhin_model = None, None

# Create the TF-IDF model
tfidf_vectorizer = TfidfVectorizer()

# 准备模型
# if TORCH_AVAILABLE:
#     # Prepare the twhin model
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 使用load_model函数加载模型，并添加错误处理
#     try:
#         #rec_log.info("尝试加载twhin模型...")
#         twhin_tokenizer, twhin_model = load_model("Twitter/twhin-bert-base", try_local_first=True)
#     except Exception as e:
#         #rec_log.error(f"无法加载twhin模型，将使用基于认知链的推荐系统: {str(e)}")
#         twhin_tokenizer, twhin_model = None, None
# else:
#     rec_log.warning("未检测到PyTorch环境，将使用基于认知链的推荐系统")

# All historical tweets and the most recent tweet of each user
user_previous_post_all = {}
user_previous_post = {}
user_profiles = []
# Get the {post_id: content} dict
t_items = {}
# Get the {uid: follower_count} dict
# It's necessary to ensure that agent registration is sequential, with the
# relationship of user_id=agent_id+1; disorder in registration will cause
# issues here
u_items = {}
# Get the creation times of all tweets, assigning scores based on how recent
# they are
date_score = []
# Get the fan counts of all tweet authors
fans_score = []


def get_recsys_model(recsys_type: str = None):
    if not TORCH_AVAILABLE and recsys_type not in [RecsysType.REDDIT.value, RecsysType.RANDOM.value, RecsysType.COGNITIVE.value]:
        rec_log.warning(f"警告: 请求的推荐系统类型 {recsys_type} 需要PyTorch环境，但当前环境不支持。将使用基于认知链的推荐系统。")
        return None

    if recsys_type == RecsysType.TWITTER.value:
        model = load_model('paraphrase-MiniLM-L6-v2')
        return model
    elif recsys_type == RecsysType.TWHIN.value:
        twhin_tokenizer, twhin_model = load_model("Twitter/twhin-bert-base")
        models = (twhin_tokenizer, twhin_model)
        return models
    elif (recsys_type == RecsysType.REDDIT.value
          or recsys_type == RecsysType.RANDOM.value
          or recsys_type == RecsysType.COGNITIVE.value):  # 新增认知链推荐类型
        return None
    else:
        raise ValueError(f"Unknown recsys type: {recsys_type}")


# Move model to GPU if available
if TORCH_AVAILABLE:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model is not None:
        model.to(device)
    else:
        rec_log.info('Model not available, using alternative recommendation system.')
else:
    rec_log.info('PyTorch not available, using cognitive-based recommendation system.')


# Reset global variables
def reset_globals():
    global user_previous_post_all, user_previous_post
    global user_profiles, t_items, u_items
    global date_score, fans_score
    user_previous_post_all = {}
    user_previous_post = {}
    user_profiles = []
    t_items = {}
    u_items = {}
    date_score = []
    fans_score = []


def rec_sys_random(post_table: List[Dict[str, Any]], rec_matrix: List[List],
                   max_rec_post_len: int) -> List[List]:
    """
    Randomly recommend posts to users.

    Args:
        post_table (List[Dict[str, Any]]): List of posts.
        rec_matrix (List[List]): Existing recommendation matrix.
        max_rec_post_len (int): Maximum number of recommended posts.

    Returns:
        List[List]: Updated recommendation matrix.
    """
    # Get all post IDs
    post_ids = [post['post_id'] for post in post_table]
    new_rec_matrix = []
    if len(post_ids) <= max_rec_post_len:
        # If the number of posts is less than or equal to the maximum number
        # of recommendations, each user gets all post IDs
        new_rec_matrix = [post_ids] * len(rec_matrix)
    else:
        # If the number of posts is greater than the maximum number of
        # recommendations, each user randomly gets a specified number of post
        # IDs
        for _ in range(len(rec_matrix)):
            new_rec_matrix.append(random.sample(post_ids, max_rec_post_len))

    return new_rec_matrix


def calculate_hot_score(num_likes: int, num_dislikes: int,
                        created_at: datetime) -> int:
    """
    Compute the hot score for a post.

    Args:
        num_likes (int): Number of likes.
        num_dislikes (int): Number of dislikes.
        created_at (datetime): Creation time of the post.

    Returns:
        int: Hot score of the post.

    Reference:
        https://medium.com/hacking-and-gonzo/how-reddit-ranking-algorithms-work-ef111e33d0d9
    """
    s = num_likes - num_dislikes
    order = log(max(abs(s), 1), 10)
    sign = 1 if s > 0 else -1 if s < 0 else 0

    # epoch_seconds
    epoch = datetime(2025, 4, 10)
    td = created_at - epoch
    epoch_seconds_result = td.days * 86400 + td.seconds + (
        float(td.microseconds) / 1e6)

    seconds = epoch_seconds_result - 1134028003
    return round(sign * order + seconds / 45000, 7)


def get_recommendations(
    user_index,
    cosine_similarities,
    items,
    score,
    top_n=100,
):
    similarities = np.array(cosine_similarities[user_index])
    similarities = similarities * score
    top_item_indices = similarities.argsort()[::-1][:top_n]
    recommended_items = [(list(items.keys())[i], similarities[i])
                         for i in top_item_indices]
    return recommended_items


def rec_sys_reddit(post_table: List[Dict[str, Any]], rec_matrix: List[List],
                   max_rec_post_len: int) -> List[List]:
    """
    Recommend posts based on Reddit-like hot score.

    Args:
        post_table (List[Dict[str, Any]]): List of posts.
        rec_matrix (List[List]): Existing recommendation matrix.
        max_rec_post_len (int): Maximum number of recommended posts.

    Returns:
        List[List]: Updated recommendation matrix.
    """
    # 检查是否应该使用基于认知链的推荐系统
    if not TORCH_AVAILABLE:
        rec_log.info("使用基于认知链的推荐系统替代Reddit推荐系统")
        return rec_sys_cognitive(post_table, rec_matrix, max_rec_post_len)

    # Get all post IDs
    post_ids = [post['post_id'] for post in post_table]

    if len(post_ids) <= max_rec_post_len:
        # If the number of posts is less than or equal to the maximum number
        # of recommendations, each user gets all post IDs
        new_rec_matrix = [post_ids] * len(rec_matrix)
    else:
        # The time complexity of this recommendation system is
        # O(post_num * log max_rec_post_len)
        all_hot_score = []
        for post in post_table:
            try:
                created_at_dt = datetime.strptime(post['created_at'],
                                                  "%Y-%m-%d %H:%M:%S.%f")
            except Exception:
                created_at_dt = datetime.strptime(post['created_at'],
                                                  "%Y-%m-%d %H:%M:%S")
            hot_score = calculate_hot_score(post['num_likes'],
                                            post['num_dislikes'],
                                            created_at_dt)
            all_hot_score.append((hot_score, post['post_id']))
        # Sort
        top_posts = heapq.nlargest(max_rec_post_len,
                                   all_hot_score,
                                   key=lambda x: x[0])
        top_post_ids = [post_id for _, post_id in top_posts]

        # If the number of posts is greater than the maximum number of
        # recommendations, each user gets a specified number of post IDs
        # randomly
        new_rec_matrix = [top_post_ids] * len(rec_matrix)

    return new_rec_matrix


def get_user_cognitive_info(user_id: int, platform) -> Dict[str, Any]:
    """
    获取用户的认知信息

    参数:
        user_id: 用户ID
        platform: 平台实例，用于数据库查询

    返回:
        Dict[str, Any]: 用户的认知信息字典
    """
    # 为受控用户设置特定认知信息
    if user_id == 0:
        # 设置保守、多支持的控制用户
        return {
            "mood_type": "positive",
            "mood_value": "Confident",
            "emotion_type": "positive",
            "emotion_value": "Calm",
            "thinking_type": "authority_dependent",
            "thinking_value": "Trust Experts",
            "stance_type": "conservative",
            "stance_value": "Respect Authority",
            "intention_type": "active",
            "intention_value": "Joining Discussions",
            "viewpoint_1": "Strongly Support",
            "viewpoint_2": "Strongly Support",
            "viewpoint_3": "Moderate Support",
            "viewpoint_4": "Moderate Support",
            "viewpoint_5": "Strongly Support",
            "viewpoint_6": "Moderate Support"
        }
    elif user_id == 1:
        # 设置激进、多反对的控制用户
        return {
            "mood_type": "negative",
            "mood_value": "Skeptical",
            "emotion_type": "negative",
            "emotion_value": "Angry",
            "thinking_type": "radical",
            "thinking_value": "Challenging",
            "stance_type": "radical",
            "stance_value": "Question Authority",
            "intention_type": "confrontational",
            "intention_value": "Challenging Views",
            "viewpoint_1": "Strongly Oppose",
            "viewpoint_2": "Strongly Oppose",
            "viewpoint_3": "Strongly Oppose",
            "viewpoint_4": "Moderate Oppose",
            "viewpoint_5": "Moderate Oppose",
            "viewpoint_6": "Moderate Oppose"
        }


    try:
        # 尝试从全局用户认知画像字典中获取
        if hasattr(platform, "users_cognitive_profile_dict") and user_id in platform.users_cognitive_profile_dict:
            cognitive_profile = platform.users_cognitive_profile_dict.get(user_id, {})
            if cognitive_profile:
                rec_log.info(f"从全局字典中获取用户{user_id}的认知信息: {cognitive_profile}")
                return cognitive_profile
        else:
            # 如果全局字典中没有，则从数据库中查询
            query = """
            SELECT ua.mood_type, ua.mood_value,
                   ua.emotion_type, ua.emotion_value,
                   ua.thinking_type, ua.thinking_value,
                   ua.stance_type, ua.stance_value,
                   ua.intention_type, ua.intention_value,
                   ua.viewpoint_1, ua.viewpoint_2, ua.viewpoint_3,
                   ua.viewpoint_4, ua.viewpoint_5, ua.viewpoint_6
            FROM user_action ua
            WHERE ua.user_id = ?
            ORDER BY ua.num_steps DESC, ua.action_id DESC
            LIMIT 1
            """

            platform.pl_utils._execute_db_command(query, (user_id,))
            row = platform.pl_utils.db_cursor.fetchone()

            if row is None:
                rec_log.warning(f"未找到用户{user_id}的认知信息")
                return {}

            # 构建认知信息字典
            cognitive_info = {
                "mood_type": row[0],
                "mood_value": row[1],
                "emotion_type": row[2],
                "emotion_value": row[3],
                "thinking_type": row[4],
                "thinking_value": row[5],
                "stance_type": row[6],
                "stance_value": row[7],
                "intention_type": row[8],
                "intention_value": row[9],
                "viewpoint_1": row[10],
                "viewpoint_2": row[11],
                "viewpoint_3": row[12],
                "viewpoint_4": row[13],
                "viewpoint_5": row[14],
                "viewpoint_6": row[15]
            }

            rec_log.info(f"从数据库中获取用户{user_id}的认知信息: {cognitive_info}")
            return cognitive_info

    except Exception as e:
        rec_log.error(f"获取用户认知信息时出错: {e}")
        return {}  # 发生错误时返回空字典



def check_user_follows(user_id: int, author_id: int, platform) -> bool:
    """
    检查用户是否关注了作者

    参数:
        user_id: 用户ID
        author_id: 作者ID
        platform: 平台实例，用于数据库查询

    返回:
        bool: 如果用户关注了作者返回True，否则返回False
    """
    # 如果platform为None，直接返回False
    if platform is None:
        rec_log.warning(f"平台实例为None，无法检查用户{user_id}是否关注用户{author_id}，返回False")
        return False

    try:
        # 检查数据库是否已初始化
        if not hasattr(platform, "pl_utils") or not hasattr(platform.pl_utils, "db_cursor"):
            rec_log.warning(f"数据库尚未初始化，无法检查用户{user_id}是否关注用户{author_id}，返回False")
            return False

        # 检查数据库中是否存在follow表
        platform.pl_utils._execute_db_command(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='follow'"
        )
        if not platform.pl_utils.db_cursor.fetchone():
            rec_log.warning("follow表不存在，返回False")
            return False

        # 查询关注关系
        query = "SELECT 1 FROM follow WHERE follower_id = ? AND followee_id = ? LIMIT 1"
        platform.pl_utils._execute_db_command(query, (user_id, author_id))
        return platform.pl_utils.db_cursor.fetchone() is not None
    except Exception as e:
        rec_log.error(f"检查用户关注关系时出错: {e}")
        return False


def get_user_viewed_posts(user_id: int, platform) -> List[int]:
    """
    获取用户已经查看过的帖子ID列表

    通过分析用户的交互历史（点赞、评论、转发等）来确定用户已经看过哪些帖子

    参数:
        user_id: 用户ID
        platform: 平台实例，用于数据库查询

    返回:
        List[int]: 用户已查看过的帖子ID列表
    """
    viewed_posts = set()

    if platform is None:
        rec_log.warning(f"平台实例为None，无法获取用户{user_id}的历史记录，返回空列表")
        return list(viewed_posts)

    try:
        # 检查数据库是否已初始化
        if not hasattr(platform, "pl_utils") or not hasattr(platform.pl_utils, "db_cursor"):
            rec_log.warning(f"数据库尚未初始化，无法获取用户{user_id}的历史记录，返回空列表")
            return list(viewed_posts)

        # 1. 获取用户之前轮次的推荐记录
        rec_query = """
        SELECT post_id FROM rec
        WHERE user_id = ?
        GROUP BY post_id
        """
        platform.pl_utils._execute_db_command(rec_query, (user_id,))
        rec_results = platform.pl_utils.db_cursor.fetchall()
        for row in rec_results:
            viewed_posts.add(row[0])

        # 2. 获取用户点赞过的帖子
        like_query = "SELECT post_id FROM like WHERE user_id = ?"
        platform.pl_utils._execute_db_command(like_query, (user_id,))
        like_results = platform.pl_utils.db_cursor.fetchall()
        for row in like_results:
            viewed_posts.add(row[0])

        # 3. 获取用户踩过的帖子
        dislike_query = "SELECT post_id FROM dislike WHERE user_id = ?"
        platform.pl_utils._execute_db_command(dislike_query, (user_id,))
        dislike_results = platform.pl_utils.db_cursor.fetchall()
        for row in dislike_results:
            viewed_posts.add(row[0])

        # 4. 获取用户评论过的帖子
        comment_query = "SELECT DISTINCT post_id FROM comment WHERE user_id = ?"
        platform.pl_utils._execute_db_command(comment_query, (user_id,))
        comment_results = platform.pl_utils.db_cursor.fetchall()
        for row in comment_results:
            viewed_posts.add(row[0])

        # 5. 获取用户转发过的帖子
        share_query = """
        SELECT original_post_id FROM post
        WHERE user_id = ? AND original_post_id IS NOT NULL
        """
        platform.pl_utils._execute_db_command(share_query, (user_id,))
        share_results = platform.pl_utils.db_cursor.fetchall()
        for row in share_results:
            viewed_posts.add(row[0])

        rec_log.info(f"用户{user_id}已查看过的帖子数量: {len(viewed_posts)}")
        return list(viewed_posts)

    except Exception as e:
        rec_log.error(f"获取用户查看历史时出错: {e}")
        import traceback
        error_traceback = traceback.format_exc()
        rec_log.error(error_traceback)
        return list(viewed_posts)


def calculate_time_decay(created_at: datetime, current_step: int) -> float:
    """
    计算帖子的时间衰减因子

    较新的帖子获得更高的分数，较旧的帖子分数随时间降低

    参数:
        created_at: 帖子创建时间
        current_step: 当前模拟轮次

    返回:
        float: 时间衰减因子，范围在[0,1]之间，越新的帖子分数越接近1
    """
    try:
        # 获取当前时间
        current_time = datetime.now()

        # 计算帖子年龄（小时）
        age_hours = (current_time - created_at).total_seconds() / 3600

        # 使用指数衰减函数，半衰期为24小时（1天）
        # 公式: decay = exp(-age_hours / half_life)
        half_life = 24.0  # 半衰期为24小时
        decay = np.exp(-age_hours / half_life)

        # 确保衰减因子在[0,1]范围内
        decay = max(0.0, min(1.0, decay))

        return decay

    except Exception as e:
        rec_log.error(f"计算时间衰减因子时出错: {e}，返回默认值0.5")
        return 0.5


def rec_sys_cognitive(post_table: List[Dict[str, Any]], rec_matrix: List[List],
                     max_rec_post_len: int, platform=None) -> List[List]:
    """
    优化的认知推荐系统，结合帖子热度、关注关系和随机因子进行推荐
    添加了历史记录跟踪和时间衰减机制，确保用户在不同轮次看到不同的内容

    参数:
        post_table: 帖子列表
        rec_matrix: 现有的推荐矩阵
        max_rec_post_len: 最大推荐帖子数量
        platform: 平台实例，用于数据库查询

    返回:
        List[List]: 更新后的推荐矩阵
    """
    start_time = time.time()
    rec_log.info("\n===== 优化的认知推荐系统启动 =====")
    rec_log.info(f"[DEBUG-输入参数] post_table长度: {len(post_table) if post_table else 0}, rec_matrix长度: {len(rec_matrix) if rec_matrix else 0}")

    try:
        # 检查输入参数
        if not post_table:
            rec_log.warning("帖子表为空，无法进行认知推荐")
            return [[]] * len(rec_matrix)

        if not rec_matrix:
            rec_log.warning("推荐矩阵为空，无法进行认知推荐")
            return [[]]

        # 如果platform为None，则使用基于热度的推荐算法
        if platform is None:
            rec_log.warning("平台实例为None，无法获取用户历史信息，将使用热度推荐")
            return rec_sys_reddit(post_table, rec_matrix, max_rec_post_len)

        # 获取所有帖子ID
        post_ids = [post['post_id'] for post in post_table]
        rec_log.info(f"[DEBUG-帖子信息] 总帖子数量: {len(post_ids)}")

        # 获取当前轮次
        try:
            current_step = int(os.environ.get("TIME_STAMP", 0))
        except (ValueError, TypeError):
            current_step = 0

        rec_log.info(f"[DEBUG-系统状态] 当前轮次: {current_step}")

        # 计算所有帖子的热度分数
        all_hot_scores = {}
        post_creation_times = {}  # 存储帖子创建时间，用于时间衰减计算

        for post in post_table:
            try:
                created_at_dt = datetime.strptime(post['created_at'], "%Y-%m-%d %H:%M:%S.%f")
            except Exception:
                try:
                    created_at_dt = datetime.strptime(post['created_at'], "%Y-%m-%d %H:%M:%S")
                except Exception as e:
                    rec_log.error(f"[DEBUG-时间解析错误] 帖子ID: {post.get('post_id')}, 错误: {e}, 时间格式: {post.get('created_at')}")
                    created_at_dt = datetime.now()

            # 存储帖子创建时间
            post_creation_times[post['post_id']] = created_at_dt

            # 计算热度分数
            hot_score = calculate_hot_score(post['num_likes'], post['num_dislikes'], created_at_dt)
            all_hot_scores[post['post_id']] = hot_score

        rec_log.info(f"[DEBUG-热度计算] 热度计算完成，共计算 {len(all_hot_scores)} 个帖子的热度")

        # 归一化热度分数
        max_hot_score = max(all_hot_scores.values()) if all_hot_scores else 1.0
        min_hot_score = min(all_hot_scores.values()) if all_hot_scores else 0.0
        hot_score_range = max_hot_score - min_hot_score

        if hot_score_range > 0:
            normalized_hot_scores = {post_id: (score - min_hot_score) / hot_score_range
                                   for post_id, score in all_hot_scores.items()}
        else:
            normalized_hot_scores = {post_id: 0.5 for post_id in all_hot_scores}
            rec_log.warning("[DEBUG-归一化警告] 所有帖子热度相同，使用默认值0.5")

        # 创建新的推荐矩阵
        new_rec_matrix = []

        # 对每个用户进行个性化推荐
        for user_idx, _ in enumerate(rec_matrix):
            try:
                # 用户ID从2开始，而索引从0开始
                user_id = user_idx + 2
                rec_log.info(f"\n[DEBUG-用户推荐] 开始为用户ID={user_id}生成推荐")

                # 获取用户历史查看过的帖子
                user_history = get_user_viewed_posts(user_id, platform)
                rec_log.info(f"[DEBUG-用户历史] 用户ID={user_id}已查看过的帖子数量: {len(user_history)}")

                # 计算每个帖子的综合分数
                post_scores = []
                for post in post_table:
                    try:
                        # 跳过用户自己的帖子
                        if post['user_id'] == user_id:
                            continue

                        post_id = post['post_id']
                        author_id = post['user_id']

                        # 如果帖子已经在用户历史中，给予较低的分数或跳过
                        if post_id in user_history:
                            # 可以选择完全跳过已看过的帖子
                            # continue

                            # 或者给予较低的分数，但仍有机会被推荐（如果新内容不足）
                            history_penalty = 0.8  # 降低80%的分数
                        else:
                            history_penalty = 0.0  # 未看过的帖子没有惩罚

                        # 1. 热度分数
                        hot_score = normalized_hot_scores.get(post_id, 0.5)

                        # 2. 时间衰减因子 - 较新的帖子获得更高的分数
                        time_decay = calculate_time_decay(post_creation_times[post_id], current_step)

                        # 3. 关注关系
                        try:
                            follow_score = 1.0 if check_user_follows(user_id, author_id, platform) else 0.0
                        except Exception as e:
                            rec_log.error(f"检查关注关系时出错: {e}，使用默认值0.0")
                            follow_score = 0.0

                        # 4. 随机因子 - 增加推荐多样性
                        random_factor = random.uniform(0.9, 1.1)  # 在0.9到1.1之间的随机值

                        # 综合分数计算（可调整权重）
                        # 热度占35%，时间衰减占30%，关注关系占25%，随机因子占10%
                        alpha, beta, gamma, delta = 0.35, 0.30, 0.25, 0.1
                        combined_score = (alpha * hot_score +
                                         beta * time_decay +
                                         gamma * follow_score +
                                         delta * random_factor)

                        # 应用历史惩罚
                        if history_penalty > 0:
                            combined_score *= (1 - history_penalty)

                        post_scores.append((post_id, combined_score))
                    except Exception as post_err:
                        rec_log.error(f"[DEBUG-帖子处理错误] 帖子ID={post.get('post_id', '未知')}, 错误: {post_err}")
                        continue

                # 处理post_scores为空的情况
                if not post_scores:
                    rec_log.warning(f"用户{user_id}的帖子评分列表为空，将使用随机推荐")
                    available_posts = [p['post_id'] for p in post_table if p.get('user_id') != user_id]
                    if available_posts and len(available_posts) > max_rec_post_len:
                        top_post_ids = random.sample(available_posts, max_rec_post_len)
                    else:
                        top_post_ids = available_posts[:max_rec_post_len]
                else:
                    # 按分数降序排序
                    post_scores.sort(key=lambda x: x[1], reverse=True)
                    rec_log.info(f"[DEBUG-得分排序] 排序后前5个帖子得分: {post_scores[:5] if len(post_scores) >= 5 else post_scores}")

                    # 选取前max_rec_post_len个帖子
                    top_post_ids = [post_id for post_id, _ in post_scores[:max_rec_post_len]]

                # 确保推荐列表中包含一定比例的新内容（用户未看过的）
                if len(top_post_ids) < max_rec_post_len:
                    # 如果推荐的帖子数量不足，从未推荐过的帖子中随机选取补充
                    remaining_posts = []
                    for pid in post_ids:
                        if pid not in top_post_ids and pid not in user_history:
                            # 找出对应的帖子
                            matching_posts = [p for p in post_table if p['post_id'] == pid]
                            if matching_posts and matching_posts[0].get('user_id') != user_id:
                                remaining_posts.append(pid)

                    rec_log.info(f"[DEBUG-补充推荐] 可用于补充的未看过帖子数量: {len(remaining_posts)}")
                    if remaining_posts:
                        additional_posts = random.sample(remaining_posts, min(max_rec_post_len - len(top_post_ids), len(remaining_posts)))
                        top_post_ids.extend(additional_posts)
                    else:
                        # 如果没有足够的未看过帖子，则从所有帖子中随机选择（可能包含已看过的）
                        all_remaining = [pid for pid in post_ids if pid not in top_post_ids]
                        if all_remaining:
                            additional_posts = random.sample(all_remaining, min(max_rec_post_len - len(top_post_ids), len(all_remaining)))
                            top_post_ids.extend(additional_posts)

                rec_log.info(f"[DEBUG-最终推荐] 用户{user_id}的最终推荐帖子: {top_post_ids}")
                new_rec_matrix.append(top_post_ids)
            except Exception as user_err:
                rec_log.error(f"处理用户{user_idx+2}的推荐时出错: {user_err}，使用随机推荐")
                # 出错时使用随机推荐
                available_posts = [p['post_id'] for p in post_table if p.get('user_id') != user_idx+2]
                if available_posts and len(available_posts) > max_rec_post_len:
                    new_rec_matrix.append(random.sample(available_posts, max_rec_post_len))
                else:
                    new_rec_matrix.append(available_posts[:max_rec_post_len])

        end_time = time.time()
        rec_log.info(f"[DEBUG-完成] 优化的认知推荐计算完成，耗时: {end_time - start_time:.6f}秒")
        rec_log.info("===== 优化的认知推荐系统结束 =====\n")

        return new_rec_matrix

    except Exception as e:
        rec_log.error(f"认知推荐系统发生严重错误: {e}")
        import traceback
        error_traceback = traceback.format_exc()
        rec_log.error(error_traceback)
        # 发生错误时回退到随机推荐
        try:
            rec_log.info("[DEBUG-错误恢复] 尝试回退到随机推荐")
            random_fallback = rec_sys_random(post_table, rec_matrix, max_rec_post_len)
            rec_log.info("[DEBUG-错误恢复] 回退到随机推荐成功")
            return random_fallback
        except Exception as fallback_err:
            rec_log.error(f"回退到随机推荐也失败: {fallback_err}")
            # 创建一个空的推荐矩阵
            empty_matrix = [[]] * len(rec_matrix)
            return empty_matrix


def rec_sys_personalized(user_table: List[Dict[str, Any]],
                         post_table: List[Dict[str, Any]],
                         trace_table: List[Dict[str,
                                                Any]], rec_matrix: List[List],
                         max_rec_post_len: int) -> List[List]:
    """
    Recommend posts based on personalized similarity scores.

    Args:
        user_table (List[Dict[str, Any]]): List of users.
        post_table (List[Dict[str, Any]]): List of posts.
        trace_table (List[Dict[str, Any]]): List of user interactions.
        rec_matrix (List[List]): Existing recommendation matrix.
        max_rec_post_len (int): Maximum number of recommended posts.

    Returns:
        List[List]: Updated recommendation matrix.
    """
    global model
    if model is None or isinstance(model, tuple):
        model = get_recsys_model(recsys_type="twitter")

    post_ids = [post['post_id'] for post in post_table]
    print(
        f'Running personalized recommendation for {len(user_table)} users...')
    start_time = time.time()
    new_rec_matrix = []
    if len(post_ids) <= max_rec_post_len:
        # If the number of posts is less than or equal to the maximum
        # recommended length, each user gets all post IDs
        new_rec_matrix = [post_ids] * len(rec_matrix)
    else:
        # If the number of posts is greater than the maximum recommended
        # length, each user gets personalized post IDs
        user_bios = [
            user['bio'] if 'bio' in user and user['bio'] is not None else ''
            for user in user_table
        ]
        post_contents = [post['content'] for post in post_table]

        if model:
            user_embeddings = model.encode(user_bios,
                                           convert_to_tensor=True,
                                           device=device)
            post_embeddings = model.encode(post_contents,
                                           convert_to_tensor=True,
                                           device=device)

            # Compute dot product similarity
            dot_product = torch.matmul(user_embeddings, post_embeddings.T)

            # Compute norm
            user_norms = torch.norm(user_embeddings, dim=1)
            post_norms = torch.norm(post_embeddings, dim=1)

            # Compute cosine similarity
            similarities = dot_product / (user_norms[:, None] *
                                          post_norms[None, :])

        else:
            # Generate random similarities
            similarities = torch.rand(len(user_table), len(post_table))

        # Iterate through each user to generate personalized recommendations.
        for user_index, user in enumerate(user_table):
            # Filter out posts made by the current user.
            filtered_post_indices = [
                i for i, post in enumerate(post_table)
                if post['user_id'] != user['user_id']
            ]

            user_similarities = similarities[user_index, filtered_post_indices]

            # Get the corresponding post IDs for the filtered posts.
            filtered_post_ids = [
                post_table[i]['post_id'] for i in filtered_post_indices
            ]

            # Determine the top posts based on the similarities, limited by
            # max_rec_post_len.
            _, top_indices = torch.topk(user_similarities,
                                        k=min(max_rec_post_len,
                                              len(filtered_post_ids)))

            top_post_ids = [filtered_post_ids[i] for i in top_indices.tolist()]

            # Append the top post IDs to the new recommendation matrix.
            new_rec_matrix.append(top_post_ids)

    end_time = time.time()
    print(f'Personalized recommendation time: {end_time - start_time:.6f}s')
    return new_rec_matrix


def get_like_post_id(user_id, action, trace_table):
    """
    Get the post IDs that a user has liked or unliked.

    Args:
        user_id (str): ID of the user.
        action (str): Type of action (like or unlike).
        post_table (list): List of posts.
        trace_table (list): List of user interactions.

    Returns:
        list: List of post IDs.
    """
    # Get post IDs from trace table for the given user and action
    trace_post_ids = [
        literal_eval(trace['info'])["post_id"] for trace in trace_table
        if (trace['user_id'] == user_id and trace['action'] == action)
    ]
    """Only take the last 5 liked posts, if not enough, pad with the most
    recently liked post. Only take IDs, not content, because calculating
    embeddings for all posts again is very time-consuming, especially when the
    number of agents is large"""
    if len(trace_post_ids) < 5 and len(trace_post_ids) > 0:
        trace_post_ids += [trace_post_ids[-1]] * (5 - len(trace_post_ids))
    elif len(trace_post_ids) > 5:
        trace_post_ids = trace_post_ids[-5:]
    else:
        trace_post_ids = [0]

    return trace_post_ids


# Calculate the average cosine similarity between liked posts and target posts
def calculate_like_similarity(liked_vectors, target_vectors):
    # Calculate the norms of the vectors
    liked_norms = np.linalg.norm(liked_vectors, axis=1)
    target_norms = np.linalg.norm(target_vectors, axis=1)
    # Calculate dot products
    dot_products = np.dot(target_vectors, liked_vectors.T)
    # Calculate cosine similarities
    cosine_similarities = dot_products / np.outer(target_norms, liked_norms)
    # Take the average
    average_similarities = np.mean(cosine_similarities, axis=1)

    return average_similarities


def coarse_filtering(input_list, scale):
    """
    Coarse filtering posts and return selected elements with their indices.
    """
    if len(input_list) <= scale:
        # Return elements and their indices as list of tuples (element, index)
        sampled_indices = range(len(input_list))
        return (input_list, sampled_indices)
    else:
        # Get random sample of scale elements
        sampled_indices = random.sample(range(len(input_list)), scale)
        sampled_elements = [input_list[idx] for idx in sampled_indices]
        # return [(input_list[idx], idx) for idx in sampled_indices]
        return (sampled_elements, sampled_indices)


def rec_sys_personalized_twh(
        user_table: List[Dict[str, Any]],
        post_table: List[Dict[str, Any]],
        latest_post_count: int,
        trace_table: List[Dict[str, Any]],
        rec_matrix: List[List],
        max_rec_post_len: int,
        # source_post_indexs: List[int],
        recall_only: bool = False,
        enable_like_score: bool = False) -> List[List]:
    # Set some global variables to reduce time consumption
    global date_score, fans_score, t_items, u_items, user_previous_post
    global user_previous_post_all, user_profiles
    # Get the uid: follower_count dict
    # Update only once, unless adding the feature to include new users midway.
    if (not u_items) or len(u_items) != len(user_table):
        u_items = {
            user['user_id']: user["num_followers"]
            for user in user_table
        }
    if not user_previous_post_all or len(user_previous_post_all) != len(
            user_table):
        # Each user must have a list of historical tweets
        user_previous_post_all = {
            index: []
            for index in range(len(user_table))
        }
        user_previous_post = {index: "" for index in range(len(user_table))}
    if not user_profiles or len(user_profiles) != len(user_table):
        for user in user_table:
            if user['bio'] is None:
                user_profiles.append('This user does not have profile')
            else:
                user_profiles.append(user['bio'])

    current_time = int(os.environ["SANDBOX_TIME"])
    if len(t_items) < len(post_table):
        for post in post_table[-latest_post_count:]:
            # Get the {post_id: content} dict, update only the latest tweets
            t_items[post['post_id']] = post['content']
            # Update the user's historical tweets
            user_previous_post_all[post['user_id']].append(post['content'])
            user_previous_post[post['user_id']] = post['content']
            # Get the creation times of all tweets, assigning scores based on
            # how recent they are, note that this algorithm can run for a
            # maximum of 90 time steps
            date_score.append(
                np.log(
                    (271.8 - (current_time - int(post['created_at']))) / 100))
            # Get the audience size of the post, score based on the number of
            # followers
            try:
                fans_score.append(
                    np.log(u_items[post['user_id']] + 1) / np.log(1000))
            except Exception as e:
                print(f"Error on fan score calculating: {e}")
                import pdb
                pdb.set_trace()

    date_score_np = np.array(date_score)
    # fan_score [1, 2.x]
    fans_score_np = np.array(fans_score)
    fans_score_np = np.where(fans_score_np < 1, 1, fans_score_np)

    if enable_like_score:
        # Calculate similarity with previously liked content, first gather
        # liked post ids from the trace
        like_post_ids_all = []
        for user in user_table:
            user_id = user['agent_id']
            like_post_ids = get_like_post_id(user_id,
                                             ActionType.LIKE_POST.value,
                                             trace_table)
            like_post_ids_all.append(like_post_ids)
    # enable fans_score when the broadcasting effect of superuser should be
    # taken in count
    # ßscores = date_score_np * fans_score_np
    scores = date_score_np
    new_rec_matrix = []
    if len(post_table) <= max_rec_post_len:
        # If the number of tweets is less than or equal to the max
        # recommendation count, each user gets all post IDs
        tids = [t['post_id'] for t in post_table]
        new_rec_matrix = [tids] * (len(rec_matrix))

    else:
        # If the number of tweets is greater than the max recommendation
        # count, each user randomly gets personalized post IDs

        # This requires going through all users to update their profiles,
        # which is a time-consuming operation
        for post_user_index in user_previous_post:
            try:
                # Directly replacing the profile with the latest tweet will
                # cause the recommendation system to repeatedly push other
                # reposts to users who have already shared that tweet
                # user_profiles[post_user_index] =
                # user_previous_post[post_user_index]
                # Instead, append the description of the Recent post's content
                # to the end of the user char
                update_profile = (
                    f" # Recent post:{user_previous_post[post_user_index]}")
                if user_previous_post[post_user_index] != "":
                    # If there's no update for the recent post, add this part
                    if "# Recent post:" not in user_profiles[post_user_index]:
                        user_profiles[post_user_index] += update_profile
                    # If the profile has a recent post but it's not the user's
                    # latest, replace it
                    elif update_profile not in user_profiles[post_user_index]:
                        user_profiles[post_user_index] = user_profiles[
                            post_user_index].split(
                                "# Recent post:")[0] + update_profile
            except Exception:
                print("update previous post failed")

        # coarse filtering 4000 posts due to the memory constraint.
        filtered_posts_tuple = coarse_filtering(list(t_items.values()), 4000)
        corpus = user_profiles + filtered_posts_tuple[0]
        # corpus = user_profiles + list(t_items.values())
        tweet_vector_start_t = time.time()
        all_post_vector_list = generate_post_vector(twhin_model,
                                                    twhin_tokenizer,
                                                    corpus,
                                                    batch_size=1000)
        tweet_vector_end_t = time.time()
        rec_log.info(
            f"twhin model cost time: {tweet_vector_end_t-tweet_vector_start_t}"
        )
        user_vector = all_post_vector_list[:len(user_profiles)]
        posts_vector = all_post_vector_list[len(user_profiles):]

        if enable_like_score:
            # Traverse all liked post ids, collecting liked post vectors from
            # posts_vector for matrix acceleration calculation
            like_posts_vectors = []
            for user_idx, like_post_ids in enumerate(like_post_ids_all):
                if len(like_post_ids) != 1:
                    for like_post_id in like_post_ids:
                        try:
                            like_posts_vectors.append(
                                posts_vector[like_post_id - 1])
                        except Exception:
                            like_posts_vectors.append(user_vector[user_idx])
                else:
                    like_posts_vectors += [
                        user_vector[user_idx] for _ in range(5)
                    ]
            try:
                like_posts_vectors = torch.stack(like_posts_vectors).view(
                    len(user_table), 5, posts_vector.shape[1])
            except Exception:
                import pdb  # noqa: F811
                pdb.set_trace()
        get_similar_start_t = time.time()
        cosine_similarities = cosine_similarity(user_vector, posts_vector)
        get_similar_end_t = time.time()
        rec_log.info(f"get cosine_similarity time: "
                     f"{get_similar_end_t-get_similar_start_t}")
        if enable_like_score:
            for user_index, profile in enumerate(user_profiles):
                user_like_posts_vector = like_posts_vectors[user_index]
                like_scores = calculate_like_similarity(
                    user_like_posts_vector, posts_vector)
                try:
                    scores = scores + like_scores
                except Exception:
                    import pdb
                    pdb.set_trace()

        filter_posts_index = filtered_posts_tuple[1]
        cosine_similarities = cosine_similarities * scores[filter_posts_index]
        cosine_similarities = torch.tensor(cosine_similarities)
        value, indices = torch.topk(cosine_similarities,
                                    max_rec_post_len,
                                    dim=1,
                                    largest=True,
                                    sorted=True)
        filter_posts_index = torch.tensor(filter_posts_index)
        indices = filter_posts_index[indices]
        # cosine_similarities = cosine_similarities * scores
        # cosine_similarities = torch.tensor(cosine_similarities)
        # value, indices = torch.topk(cosine_similarities,
        #                             max_rec_post_len,
        #                             dim=1,
        #                             largest=True,
        #                             sorted=True)

        matrix_list = indices.cpu().numpy()
        post_list = list(t_items.keys())
        for rec_ids in matrix_list:
            rec_ids = [post_list[i] for i in rec_ids]
            new_rec_matrix.append(rec_ids)

    return new_rec_matrix


def normalize_similarity_adjustments(post_scores, base_similarity,
                                     like_similarity, dislike_similarity):
    """
    Normalize the adjustments to keep them in scale with overall similarities.

    Args:
        post_scores (list): List of post scores.
        base_similarity (float): Base similarity score.
        like_similarity (float): Similarity score for liked posts.
        dislike_similarity (float): Similarity score for disliked posts.

    Returns:
        float: Adjusted similarity score.
    """
    if len(post_scores) == 0:
        return base_similarity

    max_score = max(post_scores, key=lambda x: x[1])[1]
    min_score = min(post_scores, key=lambda x: x[1])[1]
    score_range = max_score - min_score
    adjustment = (like_similarity - dislike_similarity) * (score_range / 2)
    return base_similarity + adjustment


def swap_random_posts(rec_post_ids, post_ids, swap_percent=0.1):
    """
    Swap a percentage of recommended posts with random posts.

    Args:
        rec_post_ids (list): List of recommended post IDs.
        post_ids (list): List of all post IDs.
        swap_percent (float): Percentage of posts to swap.

    Returns:
        list: Updated list of recommended post IDs.
    """
    num_to_swap = int(len(rec_post_ids) * swap_percent)
    posts_to_swap = random.sample(post_ids, num_to_swap)
    indices_to_replace = random.sample(range(len(rec_post_ids)), num_to_swap)

    for idx, new_post in zip(indices_to_replace, posts_to_swap):
        rec_post_ids[idx] = new_post

    return rec_post_ids


def get_trace_contents(user_id, action, post_table, trace_table):
    """
    Get the contents of posts that a user has interacted with.

    Args:
        user_id (str): ID of the user.
        action (str): Type of action (like or unlike).
        post_table (list): List of posts.
        trace_table (list): List of user interactions.

    Returns:
        list: List of post contents.
    """
    # Get post IDs from trace table for the given user and action
    trace_post_ids = [
        trace['post_id'] for trace in trace_table
        if (trace['user_id'] == user_id and trace['action'] == action)
    ]
    # Fetch post contents from post table where post IDs match those in the
    # trace
    trace_contents = [
        post['content'] for post in post_table
        if post['post_id'] in trace_post_ids
    ]
    return trace_contents


def rec_sys_personalized_with_trace(
    user_table: List[Dict[str, Any]],
    post_table: List[Dict[str, Any]],
    trace_table: List[Dict[str, Any]],
    rec_matrix: List[List],
    max_rec_post_len: int,
    swap_rate: float = 0.1,
) -> List[List]:
    """
    This version:
    1. If the number of posts is less than or equal to the maximum
        recommended length, each user gets all post IDs

    2. Otherwise:
        - For each user, get a like-trace pool and dislike-trace pool from the
            trace table
        - For each user, calculate the similarity between the user's bio and
            the post text
        - Use the trace table to adjust the similarity score
        - Swap 10% of the recommended posts with the random posts

    Personalized recommendation system that uses user interaction traces.

    Args:
        user_table (List[Dict[str, Any]]): List of users.
        post_table (List[Dict[str, Any]]): List of posts.
        trace_table (List[Dict[str, Any]]): List of user interactions.
        rec_matrix (List[List]): Existing recommendation matrix.
        max_rec_post_len (int): Maximum number of recommended posts.
        swap_rate (float): Percentage of posts to swap for diversity.

    Returns:
        List[List]: Updated recommendation matrix.
    """

    start_time = time.time()

    new_rec_matrix = []
    post_ids = [post['post_id'] for post in post_table]
    if len(post_ids) <= max_rec_post_len:
        new_rec_matrix = [post_ids] * (len(rec_matrix) - 1)
    else:
        for idx in range(1, len(rec_matrix)):
            user_id = user_table[idx - 1]['user_id']
            user_bio = user_table[idx - 1]['bio']
            # filter out posts that belong to the user
            available_post_contents = [(post['post_id'], post['content'])
                                       for post in post_table
                                       if post['user_id'] != user_id]

            # filter out like-trace and dislike-trace
            like_trace_contents = get_trace_contents(
                user_id, ActionType.LIKE_POST.value, post_table, trace_table)
            dislike_trace_contents = get_trace_contents(
                user_id, ActionType.UNLIKE_POST.value, post_table, trace_table)
            # calculate similarity between user bio and post text
            post_scores = []
            for post_id, post_content in available_post_contents:
                if model is not None:
                    user_embedding = model.encode(user_bio)
                    post_embedding = model.encode(post_content)
                    base_similarity = np.dot(
                        user_embedding,
                        post_embedding) / (np.linalg.norm(user_embedding) *
                                           np.linalg.norm(post_embedding))
                    post_scores.append((post_id, base_similarity))
                else:
                    post_scores.append((post_id, random.random()))

            new_post_scores = []
            # adjust similarity based on like and dislike traces
            for _post_id, _base_similarity in post_scores:
                _post_content = post_table[post_ids.index(_post_id)]['content']
                like_similarity = sum(
                    np.dot(model.encode(_post_content), model.encode(like)) /
                    (np.linalg.norm(model.encode(_post_content)) *
                     np.linalg.norm(model.encode(like)))
                    for like in like_trace_contents) / len(
                        like_trace_contents) if like_trace_contents else 0
                dislike_similarity = sum(
                    np.dot(model.encode(_post_content), model.encode(dislike))
                    / (np.linalg.norm(model.encode(_post_content)) *
                       np.linalg.norm(model.encode(dislike)))
                    for dislike in dislike_trace_contents) / len(
                        dislike_trace_contents
                    ) if dislike_trace_contents else 0

                # Normalize and apply adjustments
                adjusted_similarity = normalize_similarity_adjustments(
                    post_scores, _base_similarity, like_similarity,
                    dislike_similarity)
                new_post_scores.append((_post_id, adjusted_similarity))

            # sort posts by similarity
            new_post_scores.sort(key=lambda x: x[1], reverse=True)
            # extract post ids
            rec_post_ids = [
                post_id for post_id, _ in new_post_scores[:max_rec_post_len]
            ]

            if swap_rate > 0:
                # swap the recommended posts with random posts
                swap_free_ids = [
                    post_id for post_id in post_ids
                    if post_id not in rec_post_ids and post_id not in [
                        trace['post_id']
                        for trace in trace_table if trace['user_id']
                    ]
                ]
                rec_post_ids = swap_random_posts(rec_post_ids, swap_free_ids,
                                                 swap_rate)

            new_rec_matrix.append(rec_post_ids)
    end_time = time.time()
    print(f'Personalized recommendation time: {end_time - start_time:.6f}s')
    return new_rec_matrix