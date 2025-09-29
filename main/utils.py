"""
辅助函数模块，用于Reddit模拟脚本

此模块包含各种辅助函数，用于支持Reddit模拟脚本的运行。
"""

import asyncio
import json
import logging
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Tuple, Set
from colorama import Fore, Style
from collections import defaultdict

# 配置日志记录器
social_log = logging.getLogger(name="social")


def load_user_profiles(profile_path):
    """
    从指定路径加载用户画像数据

    参数:
        profile_path: 用户画像文件路径

    返回:
        用户画像字典，以用户名为键
    """
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            user_profiles = json.load(f)

        # 将用户画像转换为以用户名为键的字典
        profiles_dict = {}
        for profile in user_profiles:
            username = profile.get("username", "")
            if username:
                profiles_dict[username] = profile
                # 如果有realname且与username不同，也添加realname作为键
                realname = profile.get("realname", "")
                if realname and realname != username:
                    profiles_dict[realname] = profile

        social_log.info(f"已从{profile_path}加载{len(profiles_dict)}个用户画像")
        return profiles_dict
    except Exception as e:
        social_log.error(f"加载用户画像时出错: {str(e)}")
        return {}

def match_user_context(labeled_data_path, user_profiles):
    """
    匹配用户画像和上下文数据

    参数:
        labeled_data_path: 带标签的数据文件路径
        user_profiles: 用户画像字典

    返回:
        用户上下文字典，格式为 {username: [context1, context2, ...]}
        用户认知状态字典，格式为 {username: {timestep: cognitive_state}}
    """
    try:
        # 加载带标签的数据
        with open(labeled_data_path, "r", encoding="utf-8") as f:
            labeled_data = json.load(f)

        # 按用户分组数据
        user_contexts = {}
        user_cog_states = {}

        # 用于记录匹配情况的统计
        matched_users = set()
        unmatched_users = set()
        context_count = 0

        # 首先按用户名整理所有上下文
        user_data_dict = {}
        for item in labeled_data:
            user = item.get("user", "")
            if not user:
                continue

            if user not in user_data_dict:
                user_data_dict[user] = []

            user_data_dict[user].append(item)

        social_log.info(f"数据中共有{len(user_data_dict)}个用户的上下文数据")

        # 匹配用户并处理上下文
        for username, profile in user_profiles.items():
            if username in user_data_dict:
                matched_users.add(username)
                user_data = user_data_dict[username]

                # 按顺序处理该用户的所有上下文数据
                user_contexts[username] = []
                user_cog_states[username] = {}

                for i, item in enumerate(user_data):
                    # 构建上下文对象
                    context = {
                        "current_time": item.get("current_time", ""),
                        "trigger_news": item.get("trigger_news", ""),
                        "tweet_page": item.get("tweet_page", ""),
                        "gt_text": item.get("gt_text", ""),
                        "gt_tweet_id": item.get("gt_tweet_id", ""),
                        "gt_msg_type": item.get("gt_msg_type", ""),
                        "env_prompt": item.get("env_prompt", "") or f"{item.get('trigger_news', '')}\n\n{item.get('tweet_page', '')}"
                    }

                    # 提取认知状态
                    cognitive_state = item.get("cognitive_state", {})

                    # 添加上下文和认知状态
                    user_contexts[username].append(context)
                    user_cog_states[username][i] = cognitive_state
                    context_count += 1
            else:
                unmatched_users.add(username)

        # 记录匹配结果
        social_log.info(f"用户匹配结果: 匹配成功 {len(matched_users)}个，匹配失败 {len(unmatched_users)}个")
        social_log.info(f"共加载了{context_count}条上下文数据")

        # 计算最长的上下文序列长度
        max_context_length = max([len(contexts) for contexts in user_contexts.values()]) if user_contexts else 0
        social_log.info(f"最长上下文序列: {max_context_length}轮")

        # 打印每个用户的上下文长度
        user_context_lengths = {user: len(contexts) for user, contexts in user_contexts.items()}
        social_log.info(f"用户上下文长度: {user_context_lengths}")

        # 打印未匹配的用户名，以便调试
        if unmatched_users:
            social_log.warning(f"未匹配的用户名前10个：{list(unmatched_users)[:10]}")

        return user_contexts, user_cog_states
    except Exception as e:
        social_log.error(f"匹配用户上下文时出错: {str(e)}")
        return {}, {}

def load_user_contexts(json_file_path, user_profiles_path=None):
    """
    从JSON文件中加载用户上下文序列

    参数:
        json_file_path: 包含用户数据的JSON文件路径
        user_profiles_path: 可选的用户画像文件路径

    返回:
        用户上下文字典，格式为 {user_id: [context1, context2, ...]}，
        以及用户认知状态字典，格式为 {user_id: {timestep: cognitive_state}}
    """
    try:
        # 检查是否提供了用户画像路径，如果提供了，优先使用匹配函数
        if user_profiles_path and os.path.exists(user_profiles_path):
            social_log.info(f"使用用户画像路径：{user_profiles_path} 匹配上下文数据")
            user_profiles = load_user_profiles(user_profiles_path)
            if user_profiles:
                return match_user_context(json_file_path, user_profiles)
            else:
                social_log.warning(f"未能加载用户画像，将使用传统方式加载上下文")

        # 如果没有提供用户画像或加载失败，使用传统方式加载
        # 加载数据
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 按用户分组数据
        user_data = {}
        user_cognitive_states = {}

        # 检测数据格式 - 是用户配置文件还是上下文数据
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            # 判断是否为用户配置文件格式
            if "username" in data[0] or "realname" in data[0]:
                social_log.info(f"检测到用户配置文件格式，将为每个用户创建模拟上下文")

                # 为每个用户创建一个简单的上下文
                for user_info in data:
                    # 同时使用username和realname作为标识，增强匹配能力
                    username = user_info.get("username", "")
                    realname = user_info.get("realname", "")

                    # 如果没有username，使用realname
                    if not username:
                        if not realname:
                            continue
                        username = realname

                    # 从用户个人资料创建上下文
                    bio = user_info.get("bio", "无个人简介")
                    persona = user_info.get("persona", "无个性描述")

                    # 构建上下文对象，包含用户配置文件信息
                    context = {
                        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "trigger_news": f"用户简介：{bio}",
                        "tweet_page": f"人物描述：{persona}",
                        "gt_text": "",
                        "gt_tweet_id": "",
                        "gt_msg_type": "profile"
                    }

                    # 提取认知状态
                    cognitive_state = user_info.get("cognitive_profile", {})

                    # 为每个用户创建简单的上下文序列
                    if username not in user_data:
                        user_data[username] = []
                        user_cognitive_states[username] = {}

                    # 为了测试，添加多个上下文
                    for i in range(3):  # 每个用户添加3个上下文
                        context_copy = context.copy()
                        context_copy["current_time"] = datetime.now().strftime(f"%Y-%m-%d %H:%M:{i:02d}")
                        context_index = len(user_data[username])
                        user_data[username].append(context_copy)
                        user_cognitive_states[username][context_index] = cognitive_state

                    # 如果realname与username不同，也为realname创建相同的上下文序列
                    if realname and realname != username and realname not in user_data:
                        user_data[realname] = user_data[username].copy()
                        user_cognitive_states[realname] = user_cognitive_states[username].copy()
                        social_log.info(f"为用户{username}的真实名称{realname}创建了相同的上下文")
            else:
                # 处理传统上下文数据格式
                for item in data:
                    user = item.get("user", "")
                    if not user:
                        continue

                    # 提取内容字段构建上下文
                    current_time = item.get("current_time", "")
                    trigger_news = item.get("trigger_news", "")
                    tweet_page = item.get("tweet_page", "")

                    # 构建上下文对象
                    context = {
                        "current_time": current_time,
                        "trigger_news": trigger_news,
                        "tweet_page": tweet_page,
                        "gt_text": item.get("gt_text", ""),
                        "gt_tweet_id": item.get("gt_tweet_id", ""),
                        "gt_msg_type": item.get("gt_msg_type", "")
                    }

                    # 提取认知状态
                    cognitive_state = item.get("cognitive_state", {})

                    # 将数据添加到用户词典中
                    if user not in user_data:
                        user_data[user] = []
                        user_cognitive_states[user] = {}

                    # 添加上下文和认知状态
                    context_index = len(user_data[user])
                    user_data[user].append(context)
                    user_cognitive_states[user][context_index] = cognitive_state

        # 检查是否成功加载用户上下文
        if not user_data:
            social_log.warning(f"从{json_file_path}加载的用户上下文序列为空")
            return {}, {}

        # 对每个用户的上下文按时间排序（假设已按时间顺序排列）
        social_log.info(f"已从{json_file_path}加载{len(user_data)}个用户的上下文序列")

        # 记录每个用户的上下文数量
        for user, contexts in user_data.items():
            social_log.info(f"用户 {user} 有 {len(contexts)} 个上下文")

        return user_data, user_cognitive_states

    except Exception as e:
        social_log.error(f"加载用户上下文序列时出错: {str(e)}")
        return {}, {}

def generate_control_user_profiles(num_agents, real_user_path, agent_group_view, agent_group_list, cognition_space_dict):
    """
    根据认知维度对用户进行分组选择，生成控制用户配置文件

    参数:
        num_agents: 总智能体数量
        real_user_path: 真实用户数据路径
        agent_group_view: 分组维度（如"mood", "emotion", "viewpoint_1"等）
        agent_group_list: 各组选取的用户数量列表
        cognition_space_dict: 认知空间字典

    返回:
        生成的用户配置文件路径
    """
    try:
        # 加载真实用户数据
        with open(real_user_path, "r", encoding="utf-8") as f:
            real_users = json.load(f)
        
        social_log.info(f"已加载{len(real_users)}个真实用户数据")
        
        # 根据分组维度对用户进行分类
        grouped_users = {}
        
        if agent_group_view in ["mood", "emotion", "cognition", "stance", "intention"]:
            # 对于基础认知维度，按type分组
            if agent_group_view in cognition_space_dict:
                group_types = list(cognition_space_dict[agent_group_view]["type_list"])
                social_log.info(f"按{agent_group_view}维度分组，类型: {group_types}")
                
                # 初始化分组
                for group_type in group_types:
                    grouped_users[group_type] = []
                
                # 对用户进行分组
                for user in real_users:
                    if "cognitive_profile" in user and agent_group_view in user["cognitive_profile"]:
                        user_type = user["cognitive_profile"][agent_group_view].get("type", "")
                        if user_type in grouped_users:
                            grouped_users[user_type].append(user)
                        else:
                            social_log.warning(f"用户{user.get('username', 'unknown')}的{agent_group_view}类型'{user_type}'不在认知空间中")
                    else:
                        social_log.warning(f"用户{user.get('username', 'unknown')}缺少{agent_group_view}认知档案")
            else:
                raise ValueError(f"认知空间中不存在维度: {agent_group_view}")
                
        elif agent_group_view.startswith("viewpoint_"):
            # 对于观点维度，按支持级别分组
            # 获取所有可能的支持级别
            support_levels = cognition_space_dict.get("opinion_support_levels", [])
            social_log.info(f"按{agent_group_view}观点分组，支持级别: {support_levels}")
            
            # 初始化分组
            for level in support_levels:
                grouped_users[level] = []
            
            # 对用户进行分组
            for user in real_users:
                if "cognitive_profile" in user and "opinion" in user["cognitive_profile"]:
                    opinions = user["cognitive_profile"]["opinion"]
                    for opinion in opinions:
                        if agent_group_view in opinion:
                            support_level = opinion.get("type_support_levels", "")
                            if support_level in grouped_users:
                                grouped_users[support_level].append(user)
                                break  # 找到匹配的观点后跳出
                    else:
                        social_log.warning(f"用户{user.get('username', 'unknown')}没有{agent_group_view}观点")
                else:
                    social_log.warning(f"用户{user.get('username', 'unknown')}缺少观点认知档案")
        else:
            raise ValueError(f"不支持的分组维度: {agent_group_view}")
        
        # 打印各组用户数量
        for group_name, users in grouped_users.items():
            social_log.info(f"分组'{group_name}': {len(users)}个用户")
        
        # 检查agent_group_list长度是否与分组数量匹配
        if len(agent_group_list) != len(grouped_users):
            social_log.warning(f"agent_group_list长度({len(agent_group_list)})与分组数量({len(grouped_users)})不匹配")
            # 调整agent_group_list长度
            if len(agent_group_list) < len(grouped_users):
                agent_group_list.extend([0] * (len(grouped_users) - len(agent_group_list)))
            else:
                agent_group_list = agent_group_list[:len(grouped_users)]
        
        # 从各组中选择指定数量的用户
        selected_users = []
        group_names = list(grouped_users.keys())
        
        for i, count in enumerate(agent_group_list):
            if i < len(group_names):
                group_name = group_names[i]
                group_users = grouped_users[group_name]
                
                if count > len(group_users):
                    social_log.warning(f"分组'{group_name}'只有{len(group_users)}个用户，但需要{count}个，将使用所有可用用户")
                    selected_count = len(group_users)
                else:
                    selected_count = count
                
                # 随机选择用户
                if selected_count > 0:
                    selected_from_group = random.sample(group_users, selected_count)
                    selected_users.extend(selected_from_group)
                    social_log.info(f"从分组'{group_name}'选择了{selected_count}个用户")
        
        # 随机打乱选中的用户顺序
        random.shuffle(selected_users)
        
        # 生成输出文件路径
        base_dir = os.path.dirname(real_user_path)
        base_filename = os.path.basename(real_user_path).replace('.json', '')
        agent_group_list_str = '_'.join(map(str, agent_group_list))
        output_filename = f"{base_filename}_{agent_group_view}_{agent_group_list_str}.json"
        output_path = os.path.join(base_dir, output_filename)
        
        # 确保输出目录存在
        os.makedirs(base_dir, exist_ok=True)
        
        # 保存选择的用户
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(selected_users, f, ensure_ascii=False, indent=2)
        
        social_log.info(f"已成功生成控制用户配置文件，包含{len(selected_users)}个用户")
        social_log.info(f"控制用户配置文件保存至: {output_path}")
        
        return output_path
        
    except Exception as e:
        social_log.error(f"生成控制用户配置文件时出错：{str(e)}")
        raise e

def generate_hybrid_user_profiles(real_user_ratio, num_agents, real_user_path, random_user_path, hybrid_user_profiles_path):
    """
    根据真实用户比例和总智能体数量，生成混合用户配置文件

    参数:
        real_user_ratio: 真实用户占总用户的比例
        num_agents: 总智能体数量
        real_user_path: 真实用户数据路径
        random_user_path: 随机用户数据路径
        hybrid_user_profiles_path: 混合用户数据保存路径

    返回:
        混合用户配置文件的路径
    """
    # 使用传入的混合用户文件路径
    hybrid_user_path = hybrid_user_profiles_path
    social_log.info(f"混合用户配置文件将保存至: {hybrid_user_path}")

    # 计算真实用户和随机用户的数量
    real_user_count = int(num_agents * real_user_ratio)
    random_user_count = num_agents - real_user_count
    social_log.info(f"计划生成混合用户配置文件，包含{real_user_count}个真实用户和{random_user_count}个随机用户")

    # 读取真实用户和随机用户数据
    try:
        # 检查文件是否存在
        if not os.path.exists(real_user_path):
            social_log.error(f"真实用户数据文件不存在: {real_user_path}")
            raise FileNotFoundError(f"真实用户数据文件不存在: {real_user_path}")

        if not os.path.exists(random_user_path):
            social_log.error(f"随机用户数据文件不存在: {random_user_path}")
            raise FileNotFoundError(f"随机用户数据文件不存在: {random_user_path}")

        # 读取用户数据
        with open(real_user_path, "r", encoding="utf-8") as f:
            real_users = json.load(f)
            social_log.info(f"已加载{len(real_users)}个真实用户数据")

        with open(random_user_path, "r", encoding="utf-8") as f:
            random_users = json.load(f)
            social_log.info(f"已加载{len(random_users)}个随机用户数据")

        # 确保有足够的用户数据
        if len(real_users) < real_user_count:
            social_log.warning(f"警告：真实用户数据不足，需要{real_user_count}个，但只有{len(real_users)}个")
            real_user_count = len(real_users)
            random_user_count = num_agents - real_user_count

        if len(random_users) < random_user_count:
            social_log.warning(f"警告：随机用户数据不足，需要{random_user_count}个，但只有{len(random_users)}个")
            random_user_count = len(random_users)

        social_log.info(f"调整后将包含{real_user_count}个真实用户和{random_user_count}个随机用户")

        # 随机选择用户
        selected_real_users = random.sample(real_users, real_user_count)
        selected_random_users = random.sample(random_users, random_user_count)

        # 合并用户数据
        hybrid_users = selected_real_users + selected_random_users

        # 随机打乱用户顺序，使真实用户和随机用户混合在一起
        random.shuffle(hybrid_users)

        # 确保输出目录存在
        hybrid_dir = os.path.dirname(hybrid_user_path)
        if not os.path.exists(hybrid_dir):
            os.makedirs(hybrid_dir, exist_ok=True)
            social_log.info(f"创建目录: {hybrid_dir}")

        # 保存混合用户数据
        with open(hybrid_user_path, "w", encoding="utf-8") as f:
            json.dump(hybrid_users, f, ensure_ascii=False, indent=2)

        social_log.info(f"已成功生成混合用户配置文件，包含{real_user_count}个真实用户和{random_user_count}个随机用户")
        social_log.info(f"混合用户配置文件保存至: {hybrid_user_path}")
        return hybrid_user_path

    except Exception as e:
        social_log.error(f"生成混合用户配置文件时出错：{str(e)}")
        # 如果出错，仍然返回混合用户路径，但会在调用处检查文件是否存在
        return hybrid_user_path

def generate_sub_post_data(post_path, agent_group_view, cognition_space_dict):
    """
    根据认知维度对帖子进行分组，生成子帖子数据文件
    
    参数:
        post_path: 原始帖子文件路径
        agent_group_view: 分组维度 (mood, emotion, cognition, stance, intention, viewpoint_1-6)
        cognition_space_dict: 认知空间字典
    
    返回:
        dict: 包含各分组帖子文件路径的字典
    """
    import json
    import os
    from collections import defaultdict
    
    social_log = logging.getLogger(name="social")
    
    # 创建子文件夹用于存储分组数据
    base_dir = os.path.dirname(post_path)
    grouped_data_dir = os.path.join(base_dir, "grouped_posts")
    os.makedirs(grouped_data_dir, exist_ok=True)
    
    # 生成分组文件的标识符，避免重复分组
    base_name = os.path.splitext(os.path.basename(post_path))[0]
    group_marker_file = os.path.join(grouped_data_dir, f"{base_name}_{agent_group_view}_grouped.marker")
    
    # 检查是否已经分组过
    if os.path.exists(group_marker_file):
        social_log.info(f"检测到已存在的分组标记文件，直接加载已分组的数据")
        # 读取已存在的分组文件路径
        try:
            with open(group_marker_file, "r", encoding="utf-8") as f:
                existing_group_paths = json.load(f)
            
            # 验证所有分组文件是否存在
            all_files_exist = all(os.path.exists(path) for path in existing_group_paths.values())
            if all_files_exist:
                social_log.info(f"所有分组文件都存在，跳过重新分组")
                return existing_group_paths
            else:
                social_log.warning("部分分组文件缺失，将重新进行分组")
        except Exception as e:
            social_log.warning(f"读取分组标记文件失败: {e}，将重新进行分组")
    
    # 读取原始帖子数据
    try:
        with open(post_path, "r", encoding="utf-8") as f:
            posts = json.load(f)
        social_log.info(f"成功加载{len(posts)}个帖子")
    except Exception as e:
        social_log.error(f"加载帖子数据失败: {e}")
        raise
    
    # 根据分组维度确定分组类型
    group_types = []
    if agent_group_view in ["mood", "emotion", "cognition", "stance", "intention"]:
        # 基础认知维度，按type分组
        # 检查cognition_space_dict的格式
        if agent_group_view in cognition_space_dict:
            # 新格式：直接从cognition_space_dict中获取type_list
            dimension_data = cognition_space_dict.get(agent_group_view, {})
            if "type_list" in dimension_data:
                group_types = dimension_data["type_list"]
            else:
                # 如果没有type_list，尝试获取键
                group_types = list(dimension_data.keys()) if isinstance(dimension_data, dict) else []
        else:
            # 旧格式：从Cognitive_State_CUT中获取
            cognitive_state = cognition_space_dict.get("Cognitive_State_CUT", {})
            dimension_data = cognitive_state.get(agent_group_view, {})
            group_types = list(dimension_data.keys())
        
        social_log.info(f"认知维度 {agent_group_view} 的分组类型: {group_types}")
    elif agent_group_view.startswith("viewpoint_"):
        # 观点维度，按支持级别分组
        # 检查是否有观点数据
        opinion_data = None
        if "opinion" in cognition_space_dict:
            opinion_data = cognition_space_dict["opinion"]
        else:
            # 尝试从Cognitive_State_CUT中获取
            cognitive_state = cognition_space_dict.get("Cognitive_State_CUT", {})
            opinion_data = cognitive_state.get("opinion", [])
        
        if opinion_data and len(opinion_data) > 0:
            # 获取第一个观点的支持级别作为模板
            first_viewpoint = opinion_data[0]
            type_support_levels = first_viewpoint.get("type_support_levels", {})
            group_types = list(type_support_levels.keys())
            social_log.info(f"观点维度 {agent_group_view} 的分组类型: {group_types}")
        else:
            social_log.error("认知空间中未找到观点数据")
            raise ValueError("认知空间中未找到观点数据")
    else:
        social_log.error(f"不支持的分组维度: {agent_group_view}")
        raise ValueError(f"不支持的分组维度: {agent_group_view}")
    
    # 按分组类型对帖子进行分组
    grouped_posts = defaultdict(list)
    unmatched_posts = []
    
    for post in posts:
        if agent_group_view in ["mood", "emotion", "cognition", "stance", "intention"]:
            # 基础认知维度使用 {dimension}_type 字段
            group_key = post.get(f"{agent_group_view}_type")
        elif agent_group_view.startswith("viewpoint_"):
            # 观点维度直接使用字段值
            group_key = post.get(agent_group_view)
        else:
            group_key = None
        
        if group_key and group_key in group_types:
            grouped_posts[group_key].append(post)
        else:
            # 记录未匹配的帖子，但不中断处理
            unmatched_posts.append({
                "post_id": post.get("post_id", "unknown"),
                "group_key": group_key,
                "expected_types": group_types
            })
    
    # 报告未匹配的帖子统计
    if unmatched_posts:
        social_log.warning(f"共有 {len(unmatched_posts)} 个帖子未能匹配到预期分组类型")
        # 统计未匹配的分组键
        unmatched_keys = {}
        for item in unmatched_posts:
            key = item["group_key"]
            if key:
                unmatched_keys[key] = unmatched_keys.get(key, 0) + 1
        
        social_log.warning(f"未匹配的分组键统计: {unmatched_keys}")
        social_log.info(f"预期的分组类型: {group_types}")
    
    # 生成子文件路径并保存分组数据
    group_file_paths = {}
    
    for group_type in group_types:
        # 生成子文件名，保存到子文件夹中
        sub_file_name = f"{base_name}_{agent_group_view}_{group_type}.json"
        sub_file_path = os.path.join(grouped_data_dir, sub_file_name)
        
        # 保存分组数据
        group_posts = grouped_posts.get(group_type, [])
        try:
            with open(sub_file_path, "w", encoding="utf-8") as f:
                json.dump(group_posts, f, ensure_ascii=False, indent=2)
            social_log.info(f"保存 {group_type} 分组帖子到 {sub_file_path}，共 {len(group_posts)} 个帖子")
            group_file_paths[group_type] = sub_file_path
        except Exception as e:
            social_log.error(f"保存分组帖子文件失败 {sub_file_path}: {e}")
            raise
    
    # 保存分组标记文件
    try:
        with open(group_marker_file, "w", encoding="utf-8") as f:
            json.dump(group_file_paths, f, ensure_ascii=False, indent=2)
        social_log.info(f"保存分组标记文件: {group_marker_file}")
    except Exception as e:
        social_log.error(f"保存分组标记文件失败: {e}")
    
    social_log.info(f"帖子分组完成，共生成 {len(group_file_paths)} 个分组文件")
    return group_file_paths

async def export_data(post_agent, rate_agent, pairs, timestep, round_post_num, data_format, init_post_score):
    """
    导出数据到平台

    参数:
        post_agent: 发帖代理
        rate_agent: 评分代理
        pairs: 帖子对数据
        timestep: 当前时间步
        round_post_num: 每轮发布的帖子数量
        data_format: 数据格式
        init_post_score: 初始帖子分数

    返回:
        None
    """
    tasks = []

    async def process_single_post(i):
        try:
            # 计算帖子索引,允许重复使用
            rs_rc_index = (i + timestep * round_post_num) % len(pairs)

            # 根据数据格式处理内容
            if data_format == "twitter_raw":
                # 对于twitter_raw格式，检查是否使用"Original Post"格式
                if "Original Post" in pairs[rs_rc_index]:
                    # 使用新格式
                    content = pairs[rs_rc_index]["Original Post"]["text"]
                    social_log.info(f"使用twitter_raw新格式处理帖子 {i}")
                else:
                    # 尝试使用旧格式
                    rs_content = pairs[rs_rc_index]["RS"]["selftext"]
                    rc1_content = pairs[rs_rc_index]["RC_1"]["body"]
                    content = rs_content
                    if rc1_content.strip():
                        content += f"\n\n{rc1_content}"
                    social_log.info(f"使用twitter_raw旧格式处理帖子 {i}")
            else:
                # 拼接RS和RC1的内容，并添加英文语句做区分
                rs_content = pairs[rs_rc_index]["RS"]["selftext"]
                rc1_content = pairs[rs_rc_index]["RC_1"]["body"]

                if data_format == "reddit":
                    content = f"Original Post: {rs_content}\n\nResponse: {rc1_content}"
                else:  # 普通twitter格式
                    content = f"News: {rs_content}\n\nTweets: {rc1_content}"

            # 让发帖代理发布帖子
            response = await post_agent.perform_action_by_data(
                "create_post", content=content)
            post_id = response["post_id"]

            # 根据初始帖子分数参数对帖子进行评分
            if init_post_score == 1:
                await rate_agent.perform_action_by_data(
                    "like_post", post_id)  # 点赞帖子
            elif init_post_score == -1:
                await rate_agent.perform_action_by_data(
                    "dislike_post", post_id)  # 踩帖子
            elif init_post_score == 0:
                pass  # 不做任何操作
            else:
                raise ValueError(f"Unsupported value of init_post_score: "
                                f"{init_post_score}")  # 抛出异常

            # 返回成功信息
            return {"success": True, "post_id": post_id}
        except Exception as e:
            social_log.error(f"导出数据错误，索引{i}: {e}")
            # 记录错误并返回异常
            return Exception(f"导出数据错误，索引{i}: {e}")

    # 创建导出数据任务列表
    social_log.info(f"准备导出{round_post_num}个帖子，数据格式为{data_format}，总帖子数量为{len(pairs)}")
    for i in range(round_post_num):
        tasks.append(process_single_post(i))

    # 并行执行所有任务
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 检查任务执行结果
    success_count = sum(1 for r in results if r is not None and not isinstance(r, Exception))
    error_count = sum(1 for r in results if isinstance(r, Exception))

    if error_count > 0:
        # 记录错误详情
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                social_log.error(f"帖子{i}导出失败: {str(result)}")

    social_log.info(f"帖子导出完成: 成功{success_count}个，失败{error_count}个")

async def export_user_context_data(post_agent, rate_agent, context, init_post_score=0):
    """
    导出用户上下文数据，让智能体发布帖子

    参数:
        post_agent: 发帖代理
        rate_agent: 评分代理
        context: 上下文数据
        init_post_score: 初始帖子分数

    返回:
        post_id: 发布的帖子ID或None(失败时)
    """
    try:
        # 从上下文对象中提取信息
        if isinstance(context, dict):
            # 处理字典格式的上下文
            current_time = context.get("current_time", "")
            trigger_news = context.get("trigger_news", "")
            tweet_page = context.get("tweet_page", "")
            gt_text = context.get("gt_text", "")

            # 优先使用预先生成的env_prompt
            if "env_prompt" in context and context["env_prompt"]:
                detailed_context = context["env_prompt"]
                social_log.debug(f"为用户 {post_agent.user_info.name} 使用预设的env_prompt")
            else:
                # 构建完整上下文字符串，包含更详细的信息
                detailed_context = f"{current_time}\n\n新闻: {trigger_news}\n\n推文页面: {tweet_page}"
                if gt_text:
                    detailed_context += f"\n\n用户回应: {gt_text}"

            # 生成标题（优先使用特定标记，若没有则使用首行文本）
            title = context.get("title", "") or tweet_page[:100] or "个体模拟帖子"

            # 添加用户标识信息，方便跟踪
            detailed_context = f"[用户: {post_agent.user_info.name}]\n{detailed_context}"

            # 添加一些调试信息
            social_log.debug(f"为用户 {post_agent.user_info.name} 导出上下文: {current_time[:10]}...")

            # 发布帖子
            post_id = await post_agent.action.post(
                detailed_context,
                title,  # 使用设置的标题
                init_post_score
            )

            return post_id
        else:
            # 处理字符串格式的上下文（兼容旧版本）
            post_id = await post_agent.action.post(
                context,
                context.split("\n")[0][:100] if "\n" in context else "No Title",
                init_post_score
            )
            return post_id
    except Exception as e:
        social_log.error(f"导出用户上下文数据失败: {str(e)}")
        return None

def load_cognition_space(data_name: str = None, cognitive_space_path: str= None):
    """根据数据集类型加载相应的认知空间

    Args:
        data_name: 数据集类型，如 'metoo', 'blm', 'roe'

    Returns:
        加载的认知空间字典，经过处理后的格式，包含type_list和value_list
    """
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 尝试两种可能的路径
    cognition_space_dir = os.path.join(project_root, "cognition_space")

    # 加载认知空间
    raw_cognition_space_dict = None
    try:
        # 检查文件是否存在
        if not os.path.exists(cognitive_space_path):
            social_log.error(f"认知空间文件不存在: {cognitive_space_path}")
            # 列出目录中的文件，帮助调试
            if os.path.exists(cognition_space_dir):
                social_log.info(f"目录 {cognition_space_dir} 中的文件: {os.listdir(cognition_space_dir)}")
            raise FileNotFoundError(f"认知空间文件不存在: {cognitive_space_path}")

        # 加载JSON文件
        with open(cognitive_space_path, 'r', encoding='utf-8') as f:
            raw_cognition_space_dict = json.load(f)
            social_log.info(f"成功加载认知空间: {cognitive_space_path}")
            # 输出认知空间的顶层键，帮助调试
            if raw_cognition_space_dict:
                social_log.info(f"认知空间顶层键: {list(raw_cognition_space_dict.keys())}")
            else:
                social_log.warning(f"认知空间为空")
                raise ValueError("加载的认知空间为空")
    except Exception as e:
        social_log.error(f"加载认知空间出错: {str(e)}")
        raise e

    # 处理认知空间数据，转换为更有用的格式
    processed_cognition_space = {}
    if raw_cognition_space_dict:
        # 获取第一个顶层键（通常是"Cognitive_State_PlanA_meToo_Optimized"或类似的）
        top_level_key = list(raw_cognition_space_dict.keys())[0]
        cognition_data = raw_cognition_space_dict[top_level_key]

        # 处理mood
        if "mood" in cognition_data:
            mood_types = list(cognition_data["mood"].keys())
            mood_values = []

            # 遍历每个类型下的所有值并收集
            for type_key in mood_types:
                if isinstance(cognition_data["mood"][type_key], list):
                    # 直接处理字符串列表形式的值
                    mood_values.extend(cognition_data["mood"][type_key])

            processed_cognition_space["mood"] = {
                "type_list": mood_types,
                "value_list": mood_values
            }
            social_log.debug(f"处理后的mood类型: {mood_types}")
            social_log.debug(f"处理后的mood值: {mood_values}")
        else:
            raise ValueError("认知空间中缺少mood维度")

        # 处理emotion
        if "emotion" in cognition_data:
            emotion_types = list(cognition_data["emotion"].keys())
            emotion_values = []

            # 遍历每个类型下的所有值并收集
            for type_key in emotion_types:
                if isinstance(cognition_data["emotion"][type_key], list):
                    # 直接处理字符串列表形式的值
                    emotion_values.extend(cognition_data["emotion"][type_key])

            processed_cognition_space["emotion"] = {
                "type_list": emotion_types,
                "value_list": emotion_values
            }
            social_log.debug(f"处理后的emotion类型: {emotion_types}")
            social_log.debug(f"处理后的emotion值: {emotion_values}")
        else:
            raise ValueError("认知空间中缺少emotion维度")

        # 处理cognition
        if "thinking" in cognition_data:
            cognition_types = list(cognition_data["thinking"].keys())
            cognition_values = []

            # 遍历每个类型下的所有值并收集
            for type_key in cognition_types:
                if isinstance(cognition_data["thinking"][type_key], list):
                    # 直接处理字符串列表形式的值
                    cognition_values.extend(cognition_data["thinking"][type_key])

            processed_cognition_space["thinking"] = {
                "type_list": cognition_types,
                "value_list": cognition_values
            }
            social_log.debug(f"处理后的cognition类型: {cognition_types}")
            social_log.debug(f"处理后的cognition值: {cognition_values}")
        else:
            raise ValueError("认知空间中缺少cognition维度")

        # 处理stance
        if "stance" in cognition_data:
            stance_types = list(cognition_data["stance"].keys())
            stance_values = []

            # 遍历每个类型下的所有值并收集
            for type_key in stance_types:
                if isinstance(cognition_data["stance"][type_key], list):
                    # 直接处理字符串列表形式的值
                    stance_values.extend(cognition_data["stance"][type_key])

            processed_cognition_space["stance"] = {
                "type_list": stance_types,
                "value_list": stance_values
            }
            social_log.debug(f"处理后的stance类型: {stance_types}")
            social_log.debug(f"处理后的stance值: {stance_values}")
        else:
            raise ValueError("认知空间中缺少stance维度")

        # 处理intention
        if "intention" in cognition_data:
            intention_types = list(cognition_data["intention"].keys())
            intention_values = []

            # 遍历每个类型下的所有值并收集
            for type_key in intention_types:
                if isinstance(cognition_data["intention"][type_key], list):
                    # 直接处理字符串列表形式的值
                    intention_values.extend(cognition_data["intention"][type_key])

            processed_cognition_space["intention"] = {
                "type_list": intention_types,
                "value_list": intention_values
            }
            social_log.debug(f"处理后的intention类型: {intention_types}")
            social_log.debug(f"处理后的intention值: {intention_values}")
        else:
            raise ValueError("认知空间中缺少intention维度")

        # 处理opinion（特殊处理，确保所有6个观点都被添加）
        if "opinion" in cognition_data:
            viewpoints = []
            support_levels = []

            for opinion_item in cognition_data["opinion"]:
                for key in opinion_item:
                    if key.startswith("viewpoint_"):
                        viewpoints.append(opinion_item[key])
                # 获取支持级别
                if "type_support_levels" in opinion_item:
                    support_levels = list(opinion_item["type_support_levels"].keys())

            processed_cognition_space["opinion_list"] = viewpoints
            processed_cognition_space["opinion_support_levels"] = support_levels
            social_log.debug(f"处理后的观点列表: {viewpoints}")
            social_log.debug(f"处理后的支持级别: {support_levels}")
        else:
            raise ValueError("认知空间中缺少opinion维度")

        # 确保所有必需的维度都已加载
        required_dimensions = ["mood", "emotion", "thinking", "stance", "intention", "opinion_list", "opinion_support_levels"]
        for dim in required_dimensions:
            if dim not in processed_cognition_space:
                raise ValueError(f"处理后的认知空间缺少必需的维度: {dim}")

        social_log.info(f"认知空间处理完成，包含以下维度: {list(processed_cognition_space.keys())}")

    return processed_cognition_space

class AsyncTqdmWrapper:
    """异步进度条封装类"""

    def __init__(self, total, desc="模拟进度", colour="green"):
        """初始化进度条"""
        from tqdm import tqdm
        self.pbar = tqdm(total=total, desc=desc, colour=colour, position=0, leave=True)
        self.lock = asyncio.Lock()

    async def update(self, n=1):
        """更新进度条"""
        async with self.lock:
            self.pbar.update(n)

    async def close(self):
        """异步关闭进度条"""
        async with self.lock:
            self.pbar.close()

    def set_description(self, desc):
        """设置进度条描述"""
        self.pbar.set_description(desc)

def get_individual_simulation_progress(user_context_index, user_contexts):
    """
    计算个体模拟的进度

    参数:
        user_context_index: 用户当前处理到的上下文索引
        user_contexts: 所有用户的上下文数据

    返回:
        总体进度百分比, 已完成用户数, 已完成上下文数量, 总用户数, 总上下文数量
    """
    if not user_context_index or not user_contexts:
        return 0.0, 0, 0, 0, 0

    total_users = len(user_contexts)
    completed_users = 0
    total_contexts = 0
    completed_contexts = 0

    # 统计每个用户的总上下文数和已完成数
    for user, contexts in user_contexts.items():
        idx = user_context_index.get(user, 0)
        total_contexts += len(contexts)
        completed_contexts += min(idx, len(contexts))

        # 如果该用户的所有上下文都已处理完，计入已完成用户数
        if idx >= len(contexts):
            completed_users += 1

    # 计算总体进度百分比
    progress_percentage = (completed_contexts / total_contexts * 100) if total_contexts > 0 else 0.0

    return progress_percentage, completed_users, completed_contexts, total_users, total_contexts

def check_all_contexts_done(user_context_index, user_contexts):
    """
    检查所有用户的上下文是否已处理完毕

    参数:
        user_context_index: 用户当前处理到的上下文索引
        user_contexts: 所有用户的上下文数据

    返回:
        bool: 如果所有用户的上下文都已处理完毕，返回True；否则返回False
    """
    if not user_context_index or not user_contexts:
        return False

    # 检查每个用户的上下文处理情况
    for user, contexts in user_contexts.items():
        idx = user_context_index.get(user, 0)
        # 如果有任何用户的上下文未处理完，返回False
        if idx < len(contexts):
            return False

    # 所有用户的上下文都已处理完毕
    return True

def load_user_contexts(username: str, context_file_path: str) -> List[Dict[str, Any]]:
    """
    加载特定用户名的交互上下文列表

    Args:
        username: 用户名
        context_file_path: 交互数据文件路径

    Returns:
        包含用户交互上下文的列表，按时间排序
    """
    if not os.path.exists(context_file_path):
        logging.error(f"交互数据文件不存在: {context_file_path}")
        return []

    try:
        with open(context_file_path, "r", encoding="utf-8") as f:
            context_data = json.load(f)

        # 筛选出与用户名匹配的交互项
        user_contexts = []
        for item in context_data:
            if item.get("user", "") == username:
                # 提取trigger_news和tweet_page，拼接为一个字符串
                trigger_news = item.get("trigger_news", "")
                tweet_page = item.get("tweet_page", "")
                current_time = item.get("current_time", "")

                # 创建上下文项
                context_item = {
                    "time": current_time,
                    "content": f"{trigger_news}\n\n{tweet_page}"
                }
                user_contexts.append(context_item)

        # 按时间排序
        if user_contexts:
            user_contexts.sort(key=lambda x: x.get("time", ""))

        return user_contexts
    except Exception as e:
        logging.error(f"获取用户{username}上下文时出错: {e}")
        return []

def export_user_context_data(user_id_to_contexts: Dict[int, List[Dict[str, Any]]], timestep: int) -> Dict[int, str]:
    """
    导出指定时间步的用户上下文数据

    Args:
        user_id_to_contexts: 用户ID到上下文列表的映射
        timestep: 当前时间步

    Returns:
        包含当前时间步用户上下文的字典
    """
    current_contexts = {}

    for user_id, contexts in user_id_to_contexts.items():
        if timestep <= len(contexts):
            context_index = timestep - 1  # 索引从0开始，时间步从1开始
            current_contexts[user_id] = contexts[context_index]["content"]

    return current_contexts

def get_individual_simulation_progress(user_id_to_contexts: Dict[int, List[Dict[str, Any]]], timestep: int) -> Tuple[int, int]:
    """
    获取个体模拟的进度信息

    Args:
        user_id_to_contexts: 用户ID到上下文列表的映射
        timestep: 当前时间步

    Returns:
        (活跃用户数, 总用户数)元组
    """
    active_users = 0
    total_users = len(user_id_to_contexts)

    for user_id, contexts in user_id_to_contexts.items():
        if timestep <= len(contexts):
            active_users += 1

    return active_users, total_users

async def save_user_actions_to_csv(all_agents, csv_path: str, csv_think_path: str, timestep: int, include_initial_state: bool = False) -> bool:
    """
    将所有智能体的user_action_dict和思考内容保存到CSV文件

    Args:
        all_agents: 所有智能体的列表，格式为[(agent_id, agent), ...]
        csv_path: 常规用户行为数据CSV文件保存路径
        csv_think_path: 思考内容数据CSV文件保存路径
        timestep: 当前时间步
        include_initial_state: 是否包含初始状态（轮次为0）

    Returns:
        bool: 保存是否成功
    """
    import csv
    import os

    try:
        # 收集所有智能体的user_action_dict和user_action_dict_think
        all_user_actions = []
        all_user_actions_think = []

        # 如果需要包含初始状态，先为每个智能体生成并保存初始状态
        if include_initial_state:
            for agent_id, agent in all_agents:
                # 生成并保存初始状态
                initial_state = await agent.save_initial_state()
                if initial_state:
                    all_user_actions.append(initial_state)
                    social_log.info(f"已为智能体{agent_id}生成初始状态记录")

        # 处理所有智能体的行为数据
        for agent_id, agent in all_agents:
            # 处理常规用户行为数据
            if hasattr(agent, 'user_action_dict') and agent.user_action_dict:
                # 确保user_action_dict中包含所有必要字段
                user_action = agent.user_action_dict.copy()
                # 确保timestep字段存在
                if 'timestep' not in user_action:
                    user_action['timestep'] = timestep
                # 确保user_id字段存在
                if 'user_id' not in user_action:
                    user_action['user_id'] = agent_id
                all_user_actions.append(user_action)
                social_log.debug(f"收集到智能体{agent_id}的行为数据: {user_action.get('action', 'unknown')}")
            else:
                social_log.warning(f"智能体{agent_id}没有user_action_dict或为空")
                # 如果智能体没有user_action_dict，创建一个默认的
                if agent.is_active:
                    # 只为活跃智能体创建默认行为
                    default_action = {
                        'user_id': agent_id,
                        'user_name': getattr(agent, 'user_name', f"user_{agent_id}"),
                        'timestep': timestep,
                        'is_active': agent.is_active,
                        'post_id': 0,
                        'action': 'no_action',  # 默认行为
                        'action_info': {},
                        'reason': 'No action recorded'
                    }

                    # 添加认知状态字段（如果存在）
                    if hasattr(agent, 'cognitive_profile') and agent.cognitive_profile:
                        for key, value in agent.cognitive_profile.items():
                            if isinstance(value, dict):
                                for subkey, subvalue in value.items():
                                    default_action[f"{key}_{subkey}"] = subvalue
                            else:
                                default_action[key] = value

                    all_user_actions.append(default_action)
                    social_log.info(f"为智能体{agent_id}创建了默认行为数据")

            # 处理思考内容数据
            if hasattr(agent, 'user_action_dict_think') and agent.user_action_dict_think:
                # 确保user_action_dict_think中包含所有必要字段
                user_action_think = agent.user_action_dict_think.copy()
                # 确保timestep字段存在
                if 'timestep' not in user_action_think:
                    user_action_think['timestep'] = timestep
                # 确保user_id字段存在
                if 'user_id' not in user_action_think:
                    user_action_think['user_id'] = agent_id
                all_user_actions_think.append(user_action_think)

        # 保存常规用户行为数据
        success_main = True
        if all_user_actions:
            # 确保目录存在
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)

            # 确定CSV文件的字段名
            # 合并所有行为的字段，确保包含所有可能的字段
            all_fields = set()
            for action in all_user_actions:
                all_fields.update(action.keys())
            fieldnames = sorted(list(all_fields))

            # 检查CSV文件是否已存在
            file_exists = os.path.exists(csv_path)

            # 读取现有数据（如果文件存在）
            existing_data = []
            existing_fieldnames = []
            if file_exists:
                try:
                    with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
                        reader = csv.DictReader(csvfile)
                        existing_fieldnames = reader.fieldnames or []
                        existing_data = list(reader)
                except Exception as e:
                    social_log.error(f"读取现有CSV数据时出错: {str(e)}")

            # 合并现有字段和新字段
            if existing_fieldnames:
                all_fields.update(existing_fieldnames)
                fieldnames = sorted(list(all_fields))

            # 创建一个字典，用于检查重复数据
            existing_entries = {(int(row.get('timestep', 0)), int(row.get('user_id', 0))): True
                               for row in existing_data if 'timestep' in row and 'user_id' in row}

            # 过滤掉已存在的数据
            new_entries = []
            for action in all_user_actions:
                key = (int(action.get('timestep', 0)), int(action.get('user_id', 0)))
                if key not in existing_entries:
                    # 确保所有字段都存在
                    for field in fieldnames:
                        if field not in action:
                            action[field] = ""
                    new_entries.append(action)
                    existing_entries[key] = True
                else:
                    social_log.warning(f"跳过重复条目: timestep={key[0]}, user_id={key[1]}")

            # 按照timestep和user_id排序
            new_entries.sort(key=lambda x: (x.get('timestep', 0), x.get('user_id', 0)))

            # 打开CSV文件进行追加
            with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # 如果文件不存在，写入表头
                if not file_exists:
                    writer.writeheader()

                # 写入新数据
                writer.writerows(new_entries)

            social_log.info(f"已成功将{len(new_entries)}条新的user_action_dict数据保存到CSV文件: {csv_path}")
        else:
            social_log.warning(f"没有收集到任何user_action_dict数据，跳过CSV保存")

        # 保存思考内容数据
        success_think = True
        if all_user_actions_think:
            # 确保目录存在
            os.makedirs(os.path.dirname(csv_think_path), exist_ok=True)

            # 确定CSV文件的字段名
            # 合并所有思考内容的字段
            all_think_fields = set()
            for think in all_user_actions_think:
                all_think_fields.update(think.keys())
            think_fieldnames = sorted(list(all_think_fields))

            # 检查CSV文件是否已存在
            think_file_exists = os.path.exists(csv_think_path)

            # 读取现有数据（如果文件存在）
            existing_think_data = []
            existing_think_fieldnames = []
            if think_file_exists:
                try:
                    with open(csv_think_path, 'r', newline='', encoding='utf-8') as csvfile:
                        reader = csv.DictReader(csvfile)
                        existing_think_fieldnames = reader.fieldnames or []
                        existing_think_data = list(reader)
                except Exception as e:
                    social_log.error(f"读取现有思考内容CSV数据时出错: {str(e)}")

            # 合并现有字段和新字段
            if existing_think_fieldnames:
                all_think_fields.update(existing_think_fieldnames)
                think_fieldnames = sorted(list(all_think_fields))

            # 创建一个字典，用于检查重复数据
            existing_think_entries = {(int(row.get('timestep', 0)), int(row.get('user_id', 0))): True
                                    for row in existing_think_data if 'timestep' in row and 'user_id' in row}

            # 过滤掉已存在的数据
            new_think_entries = []
            for think in all_user_actions_think:
                key = (int(think.get('timestep', 0)), int(think.get('user_id', 0)))
                if key not in existing_think_entries:
                    # 确保所有字段都存在
                    for field in think_fieldnames:
                        if field not in think:
                            think[field] = ""
                    new_think_entries.append(think)
                    existing_think_entries[key] = True

            # 按照timestep和user_id排序
            new_think_entries.sort(key=lambda x: (x.get('timestep', 0), x.get('user_id', 0)))

            # 打开CSV文件进行追加
            with open(csv_think_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=think_fieldnames)

                # 如果文件不存在，写入表头
                if not think_file_exists:
                    writer.writeheader()

                # 写入新数据
                writer.writerows(new_think_entries)

            social_log.info(f"已成功将{len(new_think_entries)}条思考内容数据保存到CSV文件: {csv_think_path}")
        else:
            social_log.warning(f"没有收集到任何思考内容数据，跳过CSV保存")

        return success_main

    except Exception as e:
        social_log.error(f"保存数据到CSV文件时出错: {str(e)}")
        import traceback
        social_log.error(traceback.format_exc())
        return False