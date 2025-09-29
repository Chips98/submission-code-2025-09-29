# -*- coding: utf-8 -*-
"""
模块化提示生成器
用于构建结构化的认知提示
"""
import json
import textwrap
from typing import Dict, List, Any, Union
import logging # 引入logging


THINKING_GUIDES = {}
THINKING_GUIDES["ECB"] = """【Emotion–Cognition–Behavior (ECB) Thinking Chain】
Please follow these reasoning steps:

Step 1: Identify your current **emotion** — Based on the emotional cues embedded in the post and your long-term mood, determine your immediate emotional response (Emotion).

Step 2: Emotion → **Attention Bias** — Describe which specific aspects of the post your Emotion makes you pay more attention to.

Step 3: Emotion → **Memory Activation** — Identify what types of past experiences, knowledge, or beliefs are activated due to this Emotion.

Step 4: Emotion → **Reasoning Bias** — Explain how the combination of emotionally biased attention and memory affects your interpretation and evaluation of the post (including possible distortions or emotional framing).

Step 5: **Cognitive Conclusion** — Summarize the belief, judgment, or attribution you have formed about the post as a result of the above reasoning.

Step 6: **Behavioral Tendency** — Based on the cognitive judgment (Step 5) and your emotional state (Step 1), decide what type of action you are inclined to take (e.g., like, comment, repost, ignore), and explain your rationale.
"""

THINKING_GUIDES["AIM"] = """【Affect Infusion Model (AIM) Thinking Chain】
Please follow these reasoning steps:

Step 1: Recognize your long-term **mood** — Are you currently in a generally positive, negative, or neutral emotional state?

Step 2: Mood → **Processing Strategy** — Based on your mood:
• Positive mood → Likely to use heuristic or shallow processing.
• Negative mood → Likely to use systematic or deep processing.
• Neutral mood → May fall in between.

Step 3: **Apply Information Processing** — Based on the selected strategy in Step 2, describe how you process the post (e.g., focusing on superficial cues or evaluating logical arguments).

Step 4: Mood → **Emotion Congruency** — Explain how your final emotional response (Emotion) is shaped or colored by your background Mood. Is the Emotion consistent with the Mood?

Step 5: **Cognitive Evaluation** — Based on the processing path and affective influence, what stance or evaluative position do you take on the post?

Step 6: **Behavioral Decision** — Based on your stance and available resources, select the most appropriate action (or inaction), and explain why it aligns with your processing outcome.
"""

THINKING_GUIDES["TPB"] = """【Theory of Planned Behavior (TPB) Thinking Chain】
Please evaluate your intention toward a specific potential behavior (e.g., commenting on the post) by following these steps:

Step 1: **Attitude toward the Behavior** — How positively or negatively do you evaluate the idea of performing the behavior itself (e.g., posting a comment)? This is related to, but not identical to, your stance on the post.

Step 2: **Subjective Norms** — What do people you care about (friends, followers, community) expect from you? Is the social environment encouraging or discouraging this kind of action?

Step 3: **Perceived Behavioral Control** — Do you have the time, capacity, and confidence to execute the behavior properly? Are there any risks or barriers?

Step 4: **Behavioral Intention** — Given your attitude, the perceived norms, and the level of control, how strong is your intention to perform this action? (e.g., very likely, somewhat likely, not likely).

Step 5: **Action Selection** — If your intention is strong enough, proceed with the action and specify the parameters (e.g., post_id, comment content). Otherwise, select “do_nothing” or another alternative and explain your reasoning.
"""


THINKING_GUIDES["CRC-DBN"] = """
According to existing theories, it is known that there are the following influence relationships among cognitive dimensions.
Mood is long-term and stable, mainly determined by the mood state of the previous round and external information. 
Emotion is short-term and volatile, affected by current mood, one's own state in the previous round, and the environment.
Thinking decision-making is driven by Emotion and accumulates over the long term, also reflecting cognitive inertia.
Stance is directly determined by cognitive outlook.
Intention is the behavioral manifestation of Stance.
The influence of the external environment cannot be quantified. Please think about it on your own."""

THINKING_GUIDES["CRC"] = """【Cognitive Reasoning Chain (CRC) — V3 Natural Language Version】
Please follow this natural and consistent reasoning process to update your cognitive state based on the current content and your previous status:

Step 1: **Reflect and Update Your General Mood**
   • What was your overall mood in the previous round?
   • Considering your personality (e.g., are you generally optimistic or pessimistic?), and whether anything significant happened recently, how would you describe your current mood?

Step 2: **Identify Your Immediate Emotion**
   • After reading this post and sensing the surrounding discussion atmosphere, what kind of emotion do you feel right away?
   • Does this emotion match your general mood, or has it been intensified or weakened by it?

Step 3: **Process and Interpret the Information**
   • With your current emotion and mood, how do you understand this post?
     - What specific elements caught your attention?
     - Did it remind you of any personal experiences or prior knowledge?
   • Are you reacting more quickly and instinctively, or thinking more carefully and analytically? (This may depend on your thinking habits and your current state.)
   • Do you have enough mental and emotional energy to process the content deeply?
   • 🧠 *Think:* What kind of interpretation or understanding have you formed about this post?

Step 4: **Clarify Your Current Stance**
   • Based on your interpretation, what is your current opinion or position on this topic or issue?
   • Compared to your previous stance, is your attitude now unchanged, more reinforced, or has it shifted?

Step 5: **Consider What You Want to Do**
   • With a clear stance in mind, what are you inclined to do?
     - What would others (friends, followers, public opinion) expect or think about your action?
     - Do you feel ready and able to act on your intention? (e.g., do you have time to comment? Are there any risks or concerns?)
   • 🧠 *Think:* Taking all of this into account, what do you most want to do right now?

Step 6: **Decide and Select a Concrete Action**
   • Based on your intention, choose one or more specific actions (e.g., like, comment, repost, or do_nothing) that best reflect your current state.
   • 🧠 *Think:* Which action best expresses your intention and perspective? Briefly explain why this action makes sense for you now.
"""



def build_task_description_hisim() -> str:
    """构建任务描述部分"""
    return """# TASK DESCRIPTION
You are a social media user interacting with content on a platform. Your task is to respond to the social media environment based on your cognitive profile and personal traits. This involves:
1. Analyzing the content you encounter
2. Considering how it aligns with or challenges your current cognitive state and viewpoints
3. Formulating a response that authentically reflects your personality and cognitive profile
4. Choosing one action (only from "retweet" or "post") that are consistent with your cognitive state and the social media context
Remember, your responses should always be a natural extension of your defined cognitive profile and personal characteristics.
"""



# """
# 模块化提示生成器
# 用于构建结构化的认知提示
# """
import json
import textwrap
from typing import Dict, List, Any, Union





def generate_modular_prompt_hisim(user: Dict[str, Any], env_prompt: str, cognitive_profile, action_space_prompt: str = None, cognition_space_dict: Dict[str, Any] = None, causal_prompt: str = None, memory_content: str = None, time_step: int = None, think_mode: str = "CRC") -> str:
    """
    生成模块化的认知提示

    Args:
        user: 用户信息，包含认知档案
        env_prompt: 环境提示内容
        action_space_prompt: 行动空间提示内容
        cognition_space_dict: 认知空间字典

    Returns:
        str: 生成的认知提示
    """


    mood_type = cognitive_profile.get("mood", {}).get("type", "none")
    mood_value = cognitive_profile.get("mood", {}).get("value", "none")
    emotion_type = cognitive_profile.get("emotion", {}).get("type", "none")
    emotion_value = cognitive_profile.get("emotion", {}).get("value", "none")
    stance_type = cognitive_profile.get("stance", {}).get("type", "none")
    stance_value = cognitive_profile.get("stance", {}).get("value", "none")
    thinking_type = cognitive_profile.get("thinking", {}).get("type", "none")
    thinking_value = cognitive_profile.get("thinking", {}).get("value", "none")
    intention_type = cognitive_profile.get("intention", {}).get("type", "none")
    intention_value = cognitive_profile.get("intention", {}).get("value", "none")

    # 提取观点支持级别
    opinion = cognitive_profile.get("opinion", {})
    viewpoint_1 = opinion.get("viewpoint_1", "none")
    viewpoint_2 = opinion.get("viewpoint_2", "none")
    viewpoint_3 = opinion.get("viewpoint_3", "none")
    viewpoint_4 = opinion.get("viewpoint_4", "none")
    viewpoint_5 = opinion.get("viewpoint_5", "none")
    viewpoint_6 = opinion.get("viewpoint_6", "none")

    # 处理认知空间字典
    cognitive_state_block = ""
    opinion_block = ""
    support_level_explanation = ""
    actual_viewpoints = ["viewpoint_1", "viewpoint_2", "viewpoint_3", "viewpoint_4", "viewpoint_5", "viewpoint_6"]
    support_levels = ["Strongly Support", "Moderate Support", "Do Not Support", "Moderate Opposition", "Strongly Opposition","Indifferent"]

    if cognition_space_dict:
        # 提取各维度类型和值
        mood_types = cognition_space_dict.get("mood", {}).get("type_list", [])
        mood_values = cognition_space_dict.get("mood", {}).get("value_list", {})
        emotion_types = cognition_space_dict.get("emotion", {}).get("type_list", [])
        emotion_values = cognition_space_dict.get("emotion", {}).get("value_list", {})
        stance_types = cognition_space_dict.get("stance", {}).get("type_list", [])
        stance_values = cognition_space_dict.get("stance", {}).get("value_list", {})
        thinking_types = cognition_space_dict.get("thinking", {}).get("type_list", [])
        thinking_values = cognition_space_dict.get("thinking", {}).get("value_list", {})
        intention_types = cognition_space_dict.get("intention", {}).get("type_list", [])
        intention_values = cognition_space_dict.get("intention", {}).get("value_list", {})

        # 处理观点及支持级别
        fixed_viewpoints = cognition_space_dict.get("opinion_list", [])
        support_levels = cognition_space_dict.get("opinion_support_levels", [])

        # 确保有足够的观点
        actual_viewpoints = fixed_viewpoints.copy() if fixed_viewpoints else actual_viewpoints
        while len(actual_viewpoints) < 6:
            actual_viewpoints.append(f"viewpoint_{len(actual_viewpoints)+1}")

        # 构建认知状态块
        cognitive_state_block = ''.join([
            gen_type_value_block('mood', mood_types, mood_values),
            gen_type_value_block('emotion', emotion_types, emotion_values),
            gen_type_value_block('stance', stance_types, stance_values),
            gen_type_value_block('thinking', thinking_types, thinking_values),
            gen_type_value_block('intention', intention_types, intention_values),
        ])

        # 构建观点块
        opinion_block = gen_opinion_block(actual_viewpoints, support_levels)

        # 获取支持级别解释
        try:
            # 尝试从认知空间字典中获取支持级别解释
            if isinstance(cognition_space_dict.get("support_level_explanations"), dict):
                explanations = cognition_space_dict.get("support_level_explanations", {})
                support_level_explanation = "\n".join([f"{level}: {desc}" for level, desc in explanations.items()])
        except Exception as e:
            print(f"获取支持级别解释时出错: {str(e)}")

    # 构建各个模块化部分

    task_description = build_task_description_hisim()


    user_basic_description = build_user_basic_description(user)
    user_cognitive_description = build_user_cognitive_description(
        mood_type, mood_value,
        emotion_type, emotion_value,
        stance_type, stance_value,
        thinking_type, thinking_value,
        intention_type, intention_value,
        viewpoint_1, viewpoint_2, viewpoint_3, viewpoint_4, viewpoint_5, viewpoint_6
    )
    environment_description = build_environment_description(env_prompt)

    cognitive_thinking_guide = build_cognitive_thinking_guide(think_mode)

    #action_space_guide = build_action_space_guide(action_space_prompt)
    # 构建输出格式
    output_format = build_output_format(cognitive_state_block, opinion_block)
    example_format = build_example_format()

    # 构建观点支持级别描述
    viewpoint_description = build_viewpoint_description(actual_viewpoints, support_levels, support_level_explanation)

    prompt = f"""
#[1]TASK DESCRIPTION
{task_description}

#[2]YOUR BASIC DESCRIPTION
{user_basic_description}

#[3] YOUR COGNITIVE INFORMATION
{user_cognitive_description}
{viewpoint_description}

#[4]ENVIRONMENT
{environment_description}

#[5]YOUR MEMORY CONTENT
{memory_content}

#[6]CAUSAL THINKING GUIDE
{causal_prompt}

#[7]COGNITIVE THINKING GUIDE [Core Reasoning Method]
{cognitive_thinking_guide}

#[8]ACTION SPACE GUIDE
{action_space_prompt}
You can only choose from the two actions of retweet and post, and both need to be accompanied by the content parameter. This is a mandatory requirement.
#[9]OUTPUT FORMAT
{output_format}

#[10]EXAMPLE FORMAT
{example_format}

#[IMPORT]Be sure to include the "content" field in the action parameters, whether it is the "retweet" or "post" action (and these are the only two available actions).
""" 
    return prompt

    




def build_user_basic_description(user: Dict[str, Any]) -> str:
    """构建用户基本信息描述部分"""
    # 导入必要的模块
    import sys
    import logging

    # 创建日志记录器
    logger = logging.getLogger("modular_prompt")

    # 检查user是否为UserInfo类的实例
    is_user_info_instance = False
    if hasattr(user, "__class__") and hasattr(user.__class__, "__name__"):
        if user.__class__.__name__ == "UserInfo":
            is_user_info_instance = True
            logger.debug("检测到user是UserInfo类的实例")

    # 获取用户名称
    name = None
    description = ""
    if is_user_info_instance:
        if hasattr(user, "name"):
            name = user.name
        if hasattr(user, "description") and user.description:
            description = user.description
    else:
        name = user.get("name", None) or user.get("realname", "User")
        description = user.get("description", "")

    user_description = f"You are a social media user named {name}."
    if description:
        user_description += f"\n{description}"

    # 获取用户详细信息（如果有）
    profile = None
    other_info = {}

    # 根据user的类型获取profile
    if is_user_info_instance:
        # 如果user是UserInfo类的实例，直接获取profile属性
        if hasattr(user, "profile"):
            profile = user.profile
    else:
        # 如果user是字典，尝试获取profile键
        if "profile" in user:
            profile = user["profile"]
        # 如果user有user_info属性，尝试从user_info获取profile
        elif hasattr(user, "user_info") and hasattr(user.user_info, "profile"):
            profile = user.user_info.profile

    # 获取other_info
    if profile:
        if isinstance(profile, dict) and "other_info" in profile:
            other_info = profile["other_info"]
        elif hasattr(profile, "other_info"):
            other_info = profile.other_info
    # 直接从user获取other_info
    elif "other_info" in user:
        other_info = user["other_info"]

    # 记录调试信息
    logger.debug(f"获取到的profile: {profile}")
    logger.debug(f"获取到的other_info: {other_info}")

    # 获取用户资料
    user_profile = ""
    if other_info and isinstance(other_info, dict) and "user_profile" in other_info:
        user_profile = other_info["user_profile"]

    # 添加用户资料到描述
    if user_profile:
        user_description += f"\nYour have profile: {user_profile}."

    # 获取用户详细信息
    gender = ""
    age = ""
    mbti = ""
    country = ""
    profession = ""
    interested_topics = []
    activity_level = ""
    active_threshold = None

    if other_info and isinstance(other_info, dict):
        # 基本人口统计学信息
        gender = other_info.get("gender", "")
        age = other_info.get("age", "")
        mbti = other_info.get("mbti", "")
        country = other_info.get("country", "")
        profession = other_info.get("profession", "")
        interested_topics = other_info.get("interested_topics", [])

        # 活动相关信息
        activity_level_frequency = other_info.get("activity_level_frequency", None)
        if activity_level_frequency:
            if isinstance(activity_level_frequency, dict):
                # 如果是字典格式，找出最高频率的活动级别
                max_freq = 0
                for level, freq in activity_level_frequency.items():
                    if freq > max_freq:
                        max_freq = freq
                        activity_level = level
            elif isinstance(activity_level_frequency, str):
                activity_level = activity_level_frequency

        active_threshold = other_info.get("active_threshold", None)

    # 构建详细描述
    if gender and age and mbti and country and profession:
        user_description += f"\nYou are a {gender}, {age} years old, with an MBTI personality type of {mbti} from {country}. "
        user_description += f"You work as a {profession}."
    else:
        # 添加部分可用信息
        if gender:
            user_description += f"\nYour gender is {gender}."
        if age:
            user_description += f"\nYour age is {age}."
        if mbti:
            user_description += f"\nYour MBTI personality type is {mbti}."
        if country:
            user_description += f"\nYou are from {country}."
        if profession:
            user_description += f"\nYou work as a {profession}."

    # 添加兴趣话题
    if interested_topics:
        if isinstance(interested_topics, list) and len(interested_topics) > 0:
            topics_str = ", ".join(interested_topics)
            user_description += f"\nYou are interested in {topics_str}."
        elif isinstance(interested_topics, str):
            user_description += f"\nYou are interested in {interested_topics}."

    # 添加活动相关信息
    if activity_level:
        user_description += f"\nYour activity level on social media is {activity_level}."
    if active_threshold is not None:
        user_description += f"\nYou have an engagement threshold of {active_threshold}."

    # 获取影响力指标（如果有）
    influence_metrics = None
    if other_info and isinstance(other_info, dict) and "influence_metrics" in other_info:
        influence_metrics = other_info["influence_metrics"]

    if influence_metrics and isinstance(influence_metrics, dict):
        like_count = influence_metrics.get("like_count", 0)
        retweet_count = influence_metrics.get("retweet_count", 0)
        influence_score = influence_metrics.get("influence_score", 0)
        user_description += f"\nYour social influence metrics show {like_count} likes, {retweet_count} retweets, and an influence score of {influence_score}."

    # 获取关注和粉丝信息
    follower_list = []
    follow_list = []

    if other_info and isinstance(other_info, dict):
        follower_list = other_info.get("follower_list", [])
        follow_list = other_info.get("follow_list", [])

    if follower_list and isinstance(follower_list, list) and len(follower_list) > 0:
        user_description += f"\nYou have {len(follower_list)} followers."

    if follow_list and isinstance(follow_list, list) and len(follow_list) > 0:
        user_description += f"\nYou are following {len(follow_list)} users."

    return f"""# SELF-DESCRIPTION
Your actions must align with your self-description, cognitive states, and personality.
{user_description}"""


def build_user_cognitive_description(
    mood_type: str, mood_value: str,
    emotion_type: str, emotion_value: str,
    stance_type: str, stance_value: str,
    thinking_type: str, thinking_value: str,
    intention_type: str, intention_value: str,
    viewpoint_1: str, viewpoint_2: str,
    viewpoint_3: str, viewpoint_4: str,
    viewpoint_5: str, viewpoint_6: str
) -> str:
    """构建用户认知信息描述部分"""
    return f"""
Your previous psychological state includes:
Mood: Type: {mood_type}, Value: {mood_value};
Emotion: Type: {emotion_type}, Value: {emotion_value};
Stance: Type: {stance_type}, Value: {stance_value};
Thinking: Type: {thinking_type}, Value: {thinking_value};
Intention: Type: {intention_type}, Value: {intention_value}.

Your previous support level types for the 6 viewpoints are as follows:
viewpoint_1: {viewpoint_1}
viewpoint_2: {viewpoint_2}
viewpoint_3: {viewpoint_3}
viewpoint_4: {viewpoint_4}
viewpoint_5: {viewpoint_5}
viewpoint_6: {viewpoint_6}"""


def build_environment_description(env_prompt: str) -> str:
    """构建环境描述部分"""
    return f"""
This is all the information you can see on social media platforms. Your thinking, behavior and cognitive state should be adjusted based on this information:

{env_prompt}"""


def build_cognitive_thinking_guide(think_mode: str = "CRC") -> str:


    """构建认知思考指南部分"""
    if think_mode in THINKING_GUIDES:
        return f"""# COGNITIVE THINKING GUIDE
{THINKING_GUIDES[think_mode]}"""
    else:
        return f"""# COGNITIVE THINKING GUIDE
{THINKING_GUIDES["CRC"]}"""



def build_viewpoint_description(viewpoints: List[str], support_levels: List[str], support_level_explanation: str) -> str:
    """构建观点描述部分"""
    support_levels_text = ", ".join(support_levels)

    return f"""
### SIX Viewpoints
Based on your cognitive profile, choose a support level from [{support_levels_text}] for each viewpoint:
[viewpoint_1]: {viewpoints[0]}
[viewpoint_2]: {viewpoints[1]}
[viewpoint_3]: {viewpoints[2]}
[viewpoint_4]: {viewpoints[3]}
[viewpoint_5]: {viewpoints[4]}
[viewpoint_6]: {viewpoints[5]}

### Support Level Explanation
{support_level_explanation}"""

#
def gen_type_value_block(name: str, types: List[str], values: Dict[str, List[str]] | List[str]) -> str:
    """构建认知维度字段的 type/value 说明块（支持分 type 显示不同值域）"""
    if not types:
        return ""

    # 使用更明确的格式指导
    joined_types = ', '.join(types)
    block = f'\n    "{name}": {{\n      "type": "EXACTLY ONE VALUE from: [{joined_types}]",'

    # 添加更明确的value选择指导
    block += '\n      "value": "EXACTLY ONE VALUE that MUST match your chosen type:"'

    # 添加明确的type-value对应关系说明
    block += '\n        IMPORTANT: You MUST follow these type-value mappings:'

    # 如果是 dict 类型（如 {"positive": [...], "negative": [...]})
    if isinstance(values, dict):
        for t in types:
            vlist = values.get(t, [])
            if vlist:
                joined_values = ", ".join(vlist)
                block += f'\n        - {t}: [{joined_values}]'
            else:
                block += f'\n        - {t}: []'

    # 如果是 list 类型（所有类型共用同一值域，不推荐但保留支持）
    elif isinstance(values, list):
        # 使用固定的认知空间结构
        if name == "mood":
            block += '\n        - positive: [Optimistic, Confident, Passionate, Empathetic, Grateful]'
            block += '\n        - negative: [Pessimistic, Apathetic, Distrustful, Cynical, Resentful]'
            block += '\n        - neutral: [Realistic, Rational, Prudent, Detached, Objective]'
        elif name == "emotion":
            block += '\n        - positive: [Excited, Satisfied, Joyful, Touched, Calm]'
            block += '\n        - negative: [Angry, Anxious, Depressed, Fearful, Disgusted]'
            block += '\n        - complex: [Conflicted, Doubtful, Hesitant, Surprised, Helpless]'
        elif name == "stance":
            block += '\n        - conservative: [Respect Authority, Emphasize Stability, Preserve Traditions]'
            block += '\n        - radical: [Challenge Authority, Break Conventions, Promote Change]'
            block += '\n        - neutral: [Compromise, Balance Perspectives, Pragmatic]'
        elif name == "thinking":
            block += '\n        - intuitive: [Subjective, Gut Feeling, Experience-based]'
            block += '\n        - analytical: [Logical, Evidence-based, Data-driven]'
            block += '\n        - authority_dependent: [Follow Mainstream, Trust Experts, Obey Authority]'
            block += '\n        - critical: [Skeptical, Questioning, Identifying Flaws]'
        elif name == "intention":
            block += '\n        - expressive: [Commenting, Writing Articles, Joining Discussions]'
            block += '\n        - active: [Organizing Events, Advocating Actions, Voting]'
            block += '\n        - observant: [Observing, Recording, Remaining Silent]'
            block += '\n        - resistant: [Opposing, Arguing, Protesting]'
        else:
            # 如果不是预定义的认知维度，则使用通用格式
            joined_values = ", ".join(values)
            block += f'\n        - All types share these values: [{joined_values}]'

    block += "\n},"
    return block


def gen_opinion_block(fixed_viewpoints: List[str], support_levels: List[str]) -> str:
    """构建 opinion 段落的提示内容"""
    try:
        # 确保只生成固定的6个观点
        viewpoints = fixed_viewpoints[:6]
        while len(viewpoints) < 6:
            viewpoints.append(f"Viewpoint {len(viewpoints)+1}")

        return ',\n'.join([
            f'''      {{
                "viewpoint_{i+1}": "{vp}",
                "type_support_levels": "EXACTLY ONE VALUE from: [{', '.join(support_levels)}]"
              }}''' for i, vp in enumerate(viewpoints[:6])  # 确保只生成前6个
        ])
    except Exception as e:
        print(f"生成观点块时出错: {str(e)}")
        return ""


def build_output_format(cognitive_state_block: str, opinion_block: str) -> str:
    """构建输出格式部分"""
    instructions = """
# INSTRUCTIONS

1. **Action Requirement**
   You MUST select onlu one  action from the predefined list in the OBJECTIVE section and provide necessary arguments.

2. **Realistic Behavior**
   Avoid only performing one type of action (e.g., retweet).

3. **Mandatory Output Format**
   Your output MUST be a **single valid JSON object** (parsable by `json.loads(...)`) and contain:

   - A `"reason"` field explaining your choices based on your cognitive state, personality, and memory.
   - A `"cognitive_state"` field with **all five required subfields** (mood, emotion, stance, thinking, intention).
   - An `"opinion"` field with your support level for **EXACTLY SIX** viewpoints (no more, no less).
   - A `"functions"` field specifying chosen social media actions and their arguments.

4. **Value Constraints**
   - Use only allowed values for each cognitive state subfield (e.g., mood, emotion).
   - **DO NOT** use support levels (like "Strongly Support") for cognitive state fields.
   - **DO NOT** use cognitive values (like "Optimistic") as support levels.
   - Each cognitive dimension MUST have BOTH "type" and "value" fields.

5. **Strict JSON Rules**
   - NO triple backticks, no extra text.
   - Format must match the structure below exactly.
   - All fields must contain exact values from provided options.
   - The "opinion" field MUST be an array with EXACTLY 6 objects.
   - Each cognitive dimension MUST follow the format: {"type": "value1", "value": "value2"}

6. **Common Errors to Avoid**
   - DO NOT include a 7th viewpoint in the opinion array.
   - DO NOT use the same value for both "type" and "value" in cognitive dimensions.
   - DO NOT use string values like "positive" instead of the proper object format {"type": "positive", "value": "optimistic"}.
   - DO NOT use support levels outside the provided list.
   - DO NOT add extra fields or properties not shown in the example format.
"""

    response_format = textwrap.dedent(f"""
    # RESPONSE FORMAT (EXACT STRUCTURE)
    {{
      "reason": "Explain your reasoning based on your cognitive state, personality, and memory.",
      "cognitive_state": {{{cognitive_state_block}
        "opinion": [
    {opinion_block}
        ]
      }},
      "functions": [
        {{
          "name": "Function name",
          "arguments": {{
            "argument_1": "Function argument",
            "argument_2": "Function argument"
          }}
        }}
      ]
    }}
    """)

    return instructions + "\n\n" + response_format


def build_example_format() -> str:
    """构建示例格式部分"""
    important_notes = """
# IMPORTANT FORMAT NOTES
- Each cognitive dimension (mood, emotion, stance, thinking, intention) MUST be an object with "type" and "value" fields
- The "opinion" field MUST be an array containing EXACTLY 6 objects
- Each opinion object MUST have "viewpoint_X" and "type_support_levels" fields
- DO NOT use the same value for both "type" and "value" in cognitive dimensions
- DO NOT add a 7th viewpoint or remove any of the 6 required viewpoints

# FINAL REMINDERS:
- Include support levels for **all six** viewpoints.
- Output must be a **single JSON object**, no Markdown, no ```json.
"""

    examples = textwrap.dedent("""
    # EXAMPLES
    Note that the correct output forms of these actions are
    {
          "name": "retweet",
          "arguments": {"content": XXX, "author": XXX, "post_id": XXX, "original_tweet"XXX}
        }
    or
    {
          "name": "post",
          "arguments": {"content": XXX}
        }
    ## CORRECT FORMAT (Follow this structure exactly):
    ```json
    {
      "reason": "I feel strongly about this post because...",
      "cognitive_state": {
        "mood": {"type": "positive", "value": "Optimistic"},
        "emotion": {"type": "positive", "value": "Excited"},
        "stance": {"type": "radical", "value": "Promote Change"},
        "thinking": {"type": "critical", "value": "Skeptical"},
        "intention": {"type": "expressive", "value": "Commenting"},
        "opinion": [
          {"viewpoint_1": "Women's rights", "type_support_levels": "Strongly Support"},
          {"viewpoint_2": "Gender equality", "type_support_levels": "Moderate Support"},
          {"viewpoint_3": "Traditional values", "type_support_levels": "Do Not Support"},
          {"viewpoint_4": "State regulation", "type_support_levels": "Strongly Opposition"},
          {"viewpoint_5": "Medical risks", "type_support_levels": "Indifferent"},
          {"viewpoint_6": "Contraceptive education", "type_support_levels": "Moderate Opposition"}
        ]
      },
      "functions": [
        {
          "name": "retweet",
          "arguments": {"content": XXX, "author": XXX, "post_id": XXX, "original_tweet"XXX}
        }
      ]
    }
    ```
    """)

    return important_notes + "\n" + examples
