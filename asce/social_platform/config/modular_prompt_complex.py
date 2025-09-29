"""
模块化提示生成器
用于构建结构化的认知提示
"""
import json
import textwrap
from typing import Dict, List, Any, Union



# Default prompts for when specific prompts are not provided
default_causal_prompt = """Consider the causal relationships between different cognitive dimensions. For example:
- Strong emotions can influence your cognitive processing style
- Your stance may affect how you interpret new information
- Your mood can impact your behavioral intentions
- Environmental factors can trigger changes across multiple dimensions simultaneously

Ensure your cognitive state transitions follow logical causal patterns based on your personality and the content you encounter."""

default_memory_prompt = """You have no specific memory content from previous interactions. Focus on your current cognitive state and the present environment. If this were a multi-turn interaction, you would consider past exchanges to maintain personality consistency."""

def generate_modular_prompt(user: Dict[str, Any], env_prompt: str, action_space_prompt: str = None, cognition_space_dict: Dict[str, Any] = None, causal_prompt: str = None, memory_content: str = None, is_label: bool = False, target_context: str = None) -> str:
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
    if is_label is  False:

        # 获取认知档案
        cognitive_profile = user.get("cognitive_profile", {})

        # 提取认知画像中的各个维度信息
        mood_type = cognitive_profile.get("mood", {}).get("type", "none")
        mood_value = cognitive_profile.get("mood", {}).get("value", "none")
        emotion_type = cognitive_profile.get("emotion", {}).get("type", "none")
        emotion_value = cognitive_profile.get("emotion", {}).get("value", "none")
        stance_type = cognitive_profile.get("stance", {}).get("type", "none")
        stance_value = cognitive_profile.get("stance", {}).get("value", "none")
        cognition_type = cognitive_profile.get("cognition", {}).get("type", "none")
        cognition_value = cognitive_profile.get("cognition", {}).get("value", "none")
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
        actual_viewpoints = ["Viewpoint 1", "Viewpoint 2", "Viewpoint 3", "Viewpoint 4", "Viewpoint 5", "Viewpoint 6"]
        support_levels = ["Strongly Support", "Support", "Indifferent", "Do Not Support", "Strongly Opposition"]

        if cognition_space_dict:
            # 提取各维度类型和值
            mood_types = cognition_space_dict.get("mood", {}).get("type_list", [])
            mood_values = cognition_space_dict.get("mood", {}).get("value_list", {})
            emotion_types = cognition_space_dict.get("emotion", {}).get("type_list", [])
            emotion_values = cognition_space_dict.get("emotion", {}).get("value_list", {})
            stance_types = cognition_space_dict.get("stance", {}).get("type_list", [])
            stance_values = cognition_space_dict.get("stance", {}).get("value_list", {})
            cognition_types = cognition_space_dict.get("cognition", {}).get("type_list", [])
            cognition_values = cognition_space_dict.get("cognition", {}).get("value_list", {})
            intention_types = cognition_space_dict.get("intention", {}).get("type_list", [])
            intention_values = cognition_space_dict.get("intention", {}).get("value_list", {})

            # 处理观点及支持级别
            fixed_viewpoints = cognition_space_dict.get("opinion_list", [])
            support_levels = cognition_space_dict.get("opinion_support_levels", [])

            # 确保有足够的观点
            actual_viewpoints = fixed_viewpoints.copy() if fixed_viewpoints else actual_viewpoints
            while len(actual_viewpoints) < 6:
                actual_viewpoints.append(f"Viewpoint {len(actual_viewpoints)+1}")

            # 构建认知状态块
            cognitive_state_block = ''.join([
                gen_type_value_block('mood', mood_types, mood_values),
                gen_type_value_block('emotion', emotion_types, emotion_values),
                gen_type_value_block('stance', stance_types, stance_values),
                gen_type_value_block('cognition', cognition_types, cognition_values),
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
        task_description = build_task_description()
        user_basic_description = build_user_basic_description(user)
        user_cognitive_description = build_user_cognitive_description(
            mood_type, mood_value,
            emotion_type, emotion_value,
            stance_type, stance_value,
            cognition_type, cognition_value,
            intention_type, intention_value,
            viewpoint_1, viewpoint_2, viewpoint_3, viewpoint_4, viewpoint_5, viewpoint_6
        )
        environment_description = build_environment_description(env_prompt)
        cognitive_thinking_guide = build_cognitive_thinking_guide()
        action_space_guide = build_action_space_guide(action_space_prompt)
        # 构建输出格式
        output_format = build_output_format(cognitive_state_block, opinion_block, is_label=False)
        example_format = build_example_format(is_label=False)

        # 构建观点支持级别描述
        viewpoint_description = build_viewpoint_description(actual_viewpoints, support_levels, support_level_explanation)

        # 组合所有部分
        prompt = f"""
###[1]TASK DESCRIPTION
{task_description}

###[2]YOUR BASIC DESCRIPTION
{user_basic_description}

###[3]YOUR COGNITIVE INFORMATION
{user_cognitive_description}
{viewpoint_description}

###[4]ENVIRONMENT
{environment_description}

###[5]CAUSAL THINKING GUIDE
{causal_prompt if causal_prompt else default_causal_prompt}

###[6]YOUR MEMORY CONTENT
{memory_content if memory_content else default_memory_prompt}

###[7]COGNITIVE THINKING GUIDE
{cognitive_thinking_guide}

###[8]ACTION SPACE GUIDE
{action_space_guide}

###[9]OUTPUT FORMAT
{output_format}

###[10]EXAMPLE FORMAT
{example_format}

"""
        print(f"个体模拟的提示: {prompt}")
        import pdb; pdb.set_trace()
        return prompt
    else:
        # 获取认知档案
        cognitive_profile = user.get("cognitive_profile", {})

        # 提取认知画像中的各个维度信息
        mood_type = cognitive_profile.get("mood", {}).get("type", "none")
        mood_value = cognitive_profile.get("mood", {}).get("value", "none")
        emotion_type = cognitive_profile.get("emotion", {}).get("type", "none")
        emotion_value = cognitive_profile.get("emotion", {}).get("value", "none")
        stance_type = cognitive_profile.get("stance", {}).get("type", "none")
        stance_value = cognitive_profile.get("stance", {}).get("value", "none")
        cognition_type = cognitive_profile.get("cognition", {}).get("type", "none")
        cognition_value = cognitive_profile.get("cognition", {}).get("value", "none")
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
        actual_viewpoints = ["Viewpoint 1", "Viewpoint 2", "Viewpoint 3", "Viewpoint 4", "Viewpoint 5", "Viewpoint 6"]
        support_levels = ["Strongly Support", "Support", "Indifferent", "Do Not Support", "Strongly Opposition"]

        if cognition_space_dict:
            # 提取各维度类型和值
            mood_types = cognition_space_dict.get("mood", {}).get("type_list", [])
            mood_values = cognition_space_dict.get("mood", {}).get("value_list", {})
            emotion_types = cognition_space_dict.get("emotion", {}).get("type_list", [])
            emotion_values = cognition_space_dict.get("emotion", {}).get("value_list", {})
            stance_types = cognition_space_dict.get("stance", {}).get("type_list", [])
            stance_values = cognition_space_dict.get("stance", {}).get("value_list", {})
            cognition_types = cognition_space_dict.get("cognition", {}).get("type_list", [])
            cognition_values = cognition_space_dict.get("cognition", {}).get("value_list", {})
            intention_types = cognition_space_dict.get("intention", {}).get("type_list", [])
            intention_values = cognition_space_dict.get("intention", {}).get("value_list", {})

            # 处理观点及支持级别
            fixed_viewpoints = cognition_space_dict.get("opinion_list", [])
            support_levels = cognition_space_dict.get("opinion_support_levels", [])

            # 确保有足够的观点
            actual_viewpoints = fixed_viewpoints.copy() if fixed_viewpoints else actual_viewpoints
            while len(actual_viewpoints) < 6:
                actual_viewpoints.append(f"Viewpoint {len(actual_viewpoints)+1}")

            # 构建认知状态块
            cognitive_state_block = ''.join([
                gen_type_value_block('mood', mood_types, mood_values),
                gen_type_value_block('emotion', emotion_types, emotion_values),
                gen_type_value_block('stance', stance_types, stance_values),
                gen_type_value_block('cognition', cognition_types, cognition_values),
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
        task_description = build_label_task_description()
        user_basic_description = build_user_basic_description(user)
        user_cognitive_description = build_user_cognitive_description(
            mood_type, mood_value,
            emotion_type, emotion_value,
            stance_type, stance_value,
            cognition_type, cognition_value,
            intention_type, intention_value,
            viewpoint_1, viewpoint_2, viewpoint_3, viewpoint_4, viewpoint_5, viewpoint_6
        )
        environment_description = build_environment_description(env_prompt)
        cognitive_response_logic = build_label_cognitive_response_logic()
        action_space_guide = build_action_space_guide(action_space_prompt)
        # 构建输出格式
        output_format = build_output_format(cognitive_state_block, opinion_block, is_label=True)
        example_format = build_example_format(is_label=True)

        # 构建观点支持级别描述
        viewpoint_description = build_viewpoint_description(actual_viewpoints, support_levels, support_level_explanation)

        # 组合所有部分
        prompt = f"""
###[1]TASK DESCRIPTION
{task_description}

###[2]YOUR BASIC DESCRIPTION
{user_basic_description}

###[3]YOUR COGNITIVE INFORMATION (PRE-COMMENT STATE)
{user_cognitive_description}
{viewpoint_description}

###[4]ENVIRONMENT
{environment_description}

###[5]YOUR COMMENT
{target_context}
This is the comment you made. Your task is to deduce the cognitive states (post-comment state) that led to this comment, based on your pre-comment state and the content of the comment.

###[6]COGNITIVE RESPONSE LOGIC
{cognitive_response_logic}

###[7]ACTION SPACE GUIDE
{action_space_guide}

###[8]OUTPUT FORMAT
{output_format}

###[9]EXAMPLE FORMAT
{example_format}

"""
        print(f"标签生成的认知提示内容: {prompt}")
        import pdb; pdb.set_trace()
        return prompt




def build_label_task_description() -> str:
    """
    Constructs the task description with detailed instructions and output formatting.

    Modifications include:
    1. Emphasizing reverse inference: analyzing the user's visible posts and the actual comment data
       to deduce the user's internal psychological state at the time of commenting.
    2. Mandating a standardized JSON output format to facilitate subsequent JSON parsing.
    3. Clearly outlining the five dimensions of psychological state: emotion, mood, cognition, position, and intention,
       along with the level of support indicated by the comment.
    4. Highlighting the relationship between pre-comment cognitive state and post-comment cognitive state.
    """
    return """# TASK DESCRIPTION
You are an expert in mood analysis on social media. You will act as a user on social media,
and your task is to reconstruct the internal psychological state that the user exhibited when posting a comment,
based on the initial cognitive state (pre-comment state), the content of the posts they saw, and the comment data they actually produced.

This reconstruction must be done using reverse inference: starting with the user's pre-comment cognitive state,
analyze how the comment content reflects changes or reinforcements in their psychological state.

The psychological state is defined across the following five dimensions:
1. mood: Evaluate whether the feeling is positive, neutral, or negative.
2. Emotion: Provide a specific description of the current mood state.
3. Cognition: Analyze the thinking pattern, such as rational analysis or intuitive judgment.
4. Stance: Determine the user's stance, e.g., conservative or radical.
5. Intention: Infer the underlying behavioral intention or objective.

Additionally, incorporate the level of support for the expressed opinion based on the comment content.

Your reverse inference analysis must:
- Start with the pre-comment cognitive state provided as input
- Analyze how the comment content reflects changes in this state
- Identify specific shifts in each cognitive dimension
- Explain the logical connection between the comment content and the updated cognitive state
- Ensure all values strictly conform to the allowed ranges in the cognitive space

Please output your analysis results strictly in the following JSON format, ensuring that the keys and allowed value ranges are exactly as specified for easy JSON parsing.
"""

def build_label_cognitive_response_logic() -> str:
    """
    Constructs the cognitive response logic specifically for the tag mode.
    This function provides detailed guidance on how to perform reverse inference
    from pre-comment cognitive state to post-comment cognitive state.
    """
    return """# COGNITIVE RESPONSE LOGIC
Your task requires reverse inference to deduce cognitive states from comment content. Follow this step-by-step process:

## STEP 1: UNDERSTAND PRE-COMMENT STATE
- Carefully analyze the provided pre-comment cognitive state (mood, emotion, cognition, stance, intention)
- Note the initial support levels for each viewpoint
- This is your baseline for comparison

## STEP 2: ANALYZE COMMENT CONTENT
- Examine the language, tone, and content of the comment
- Identify emotional markers, reasoning patterns, and stance indicators
- Determine what the comment reveals about the user's psychological state

## STEP 3: IDENTIFY STATE TRANSITIONS
- Compare the comment analysis with the pre-comment state
- For each cognitive dimension, determine if there's:
  * Reinforcement (same state but stronger)
  * Shift (change to a different state)
  * No change (comment aligns with pre-existing state)

## STEP 4: DEDUCE UPDATED COGNITIVE STATE
- Based on the identified transitions, determine the post-comment cognitive state
- Ensure each dimension (mood, emotion, cognition, stance, intention) is updated appropriately
- Update viewpoint support levels based on comment content

## STEP 5: DOCUMENT REASONING
- In the "reason" field, clearly explain:
  * How the pre-comment state influenced the comment
  * How the comment reflects the updated cognitive state
  * The specific evidence from the comment that supports your conclusions
  * The logical connection between pre and post states

Remember: Your analysis must show a clear, logical progression from pre-comment state through comment content to post-comment state. All values must strictly conform to the allowed ranges in the cognitive space.
"""

def build_task_description() -> str:
    """构建任务描述部分"""
    return """# TASK DESCRIPTION
You are a social media user interacting with content on a platform. Your task is to respond to the social media environment based on your cognitive profile and personal traits. This involves:

1. Analyzing the content you encounter and identifying elements that may impact your cognitive state
2. Considering how the content aligns with or challenges your current cognitive state and viewpoints
3. Formulating a response that authentically reflects your personality, cognitive profile, and behavioral tendencies
4. Choosing appropriate actions (like, comment, share, etc.) that demonstrate a natural progression from your current cognitive state

Your output must fully reflect your internal reactions and interpretation of the environment. All behaviors must follow your personal traits and preset cognitive state. Your decisions should demonstrate a coherent transition from your initial state to your updated cognitive state after exposure to the content.

NOTE: This is a non-label simulation mode where you should respond naturally based on your defined characteristics rather than following fixed templates."""

def build_user_basic_description(user: Dict[str, Any]) -> str:
    """构建用户基本信息描述部分"""
    realname = user.get("realname", "User")
    user_description = f"You are a social media user named {realname}."

    # 获取用户详细信息（如果有）
    profile = user.get("profile", {})
    other_info = profile.get("other_info", {}) if profile else {}
    user_profile = other_info.get("user_profile", "") if other_info else ""

    if user_profile:
        user_description += f"\nYour have profile: {user_profile}."

    # 添加更多用户信息（如果有）
    gender = other_info.get("gender", "")
    age = other_info.get("age", "")
    mbti = other_info.get("mbti", "")
    country = other_info.get("country", "")
    profession = other_info.get("profession", "")
    interested_topics = other_info.get("interested_topics", [])

    if gender and age and mbti and country and profession:
        user_description += f"\nYou are a {gender}, {age} years old, with an MBTI personality type of {mbti} from {country}. "
        user_description += f"You work as a {profession}."

    if interested_topics and isinstance(interested_topics, list) and len(interested_topics) > 0:
        topics_str = ", ".join(interested_topics)
        user_description += f"\nYou are interested in {topics_str}."

    return f"""# SELF-DESCRIPTION
Your actions must align with your self-description, cognitive states, and personality.
{user_description}"""


def build_user_cognitive_description(
    mood_type: str, mood_value: str,
    emotion_type: str, emotion_value: str,
    stance_type: str, stance_value: str,
    cognition_type: str, cognition_value: str,
    intention_type: str, intention_value: str,
    viewpoint_1: str, viewpoint_2: str,
    viewpoint_3: str, viewpoint_4: str,
    viewpoint_5: str, viewpoint_6: str
) -> str:
    """构建用户认知信息描述部分"""
    return f"""# USER COGNITIVE INFORMATION
Your current psychological state includes:
mood: Type: {mood_type}, Value: {mood_value};
Emotion: Type: {emotion_type}, Value: {emotion_value};
Stance: Type: {stance_type}, Value: {stance_value};
Cognition: Type: {cognition_type}, Value: {cognition_value};
Intention: Type: {intention_type}, Value: {intention_value}.

Your current support level types for the 6 viewpoints are as follows:
viewpoint_1: {viewpoint_1}
viewpoint_2: {viewpoint_2}
viewpoint_3: {viewpoint_3}
viewpoint_4: {viewpoint_4}
viewpoint_5: {viewpoint_5}
viewpoint_6: {viewpoint_6}"""


def build_environment_description(env_prompt: str) -> str:
    """构建环境描述部分"""
    return f"""# ENVIRONMENT
This is all the information you can see on social media platforms. Your thinking, behavior and cognitive state should be adjusted based on this information:

{env_prompt}

Pay special attention to content that may:
- Challenge or reinforce your existing viewpoints
- Trigger emotional responses based on your personality traits
- Present information that conflicts with or supports your current cognitive state
- Contain social cues that would influence your stance or intention

Your response should reflect how this environment impacts your cognitive dimensions."""


def build_cognitive_thinking_guide() -> str:
    """构建认知思考指南部分"""
    return """# COGNITIVE THINKING GUIDE
Follow this step-by-step internal reasoning process when formulating your response:

1. ASSESS INITIAL STATE
   - Review your personal profile traits and behavioral tendencies
   - Understand your current cognitive state across all dimensions
   - Note your current viewpoint support levels

2. ANALYZE ENVIRONMENTAL IMPACT
   - Identify how the content might affect each cognitive dimension
   - Consider which aspects of the content resonate with or challenge your viewpoints
   - Determine potential emotional triggers based on your personality

3. MODEL STATE TRANSITIONS
   - For each cognitive dimension, determine if the content would cause:
     * Reinforcement (strengthening your current state)
     * Shift (changing to a different state)
     * No change (content doesn't significantly impact this dimension)
   - Ensure transitions are consistent with your personality traits

4. DETERMINE APPROPRIATE ACTIONS
   - Select actions that naturally follow from your updated cognitive state
   - Ensure diversity in your responses (don't always like or always comment)
   - Make sure your actions reflect your personality and behavioral style

Your reasoning should show a clear causal relationship between your cognitive state, the environment, and your chosen actions."""


def build_action_space_guide(action_space_prompt: str = None) -> str:
    """构建行动空间思考指南部分"""
    # 如果没有提供行动空间提示，使用默认提示
    if not action_space_prompt:
        default_prompt = """
# OBJECTIVE
You're a social media user, and I'll present you with some posts. After you see the posts, choose some actions from the following functions.

- do_nothing: Most of the time, you just don't feel like reposting or liking a post, and you just want to look at it. In such cases, choose this action "do_nothing"
- create_post: Create a new post with the given content.
    - Arguments: "content"(str): The content of the post to be created.
- repost: Repost a post.
    - Arguments: "post_id" (integer) - The ID of the post to be reposted. You can `repost` when you want to spread it.
- like_post: Likes a specified post.
    - Arguments: "post_id" (integer) - The ID of the tweet to be liked. You can `like` when you feel something interesting or you agree with.
- dislike_post: Dislikes a specified post.
    - Arguments: "post_id" (integer) - The ID of the post to be disliked. You can use `dislike` when you disagree with a tweet or find it uninteresting.
- follow: Follow a user specified by 'followee_id'. You can `follow' when you respect someone, love someone, or care about someone.
    - Arguments: "followee_id" (integer) - The ID of the user to be followed.
- create_comment: Creates a comment on a specified post to engage in conversations or share your thoughts on a post.
    - Arguments:
        "post_id" (integer) - The ID of the post to comment on.
        "content" (str) - The content of the comment.
- like_comment: Likes a specified comment.
    - Arguments: "comment_id" (integer) - The ID of the comment to be liked. Use `like_comment` to show agreement or appreciation for a comment.
- dislike_comment: Dislikes a specified comment.
    - Arguments: "comment_id" (integer) - The ID of the comment to be disliked. Use `dislike_comment` when you disagree with a comment or find it unhelpful.
"""
        action_space_prompt = default_prompt

    return f"""# ACTION SPACE GUIDE
Based on your cognitive state and the social media environment, choose appropriate actions that align with your personality and current psychological state.

{action_space_prompt}

IMPORTANT ACTION SELECTION GUIDELINES:
1. Your actions must be a natural extension of your updated cognitive state
2. Diversify your behaviors based on context - don't always perform the same action
3. The intensity of your actions should match the intensity of your cognitive response
4. Only use function names from the list above - no custom functions
5. Provide complete and valid arguments for each function
6. You may choose multiple actions if appropriate, or none if that matches your personality

Your chosen actions should demonstrate a coherent progression from your initial cognitive state through your environmental analysis to your final decision."""


def build_viewpoint_description(viewpoints: List[str], support_levels: List[str], support_level_explanation: str) -> str:
    """构建观点描述部分"""
    support_levels_text = ", ".join(support_levels)

    return f"""# COGNITIVE RESPONSE LOGIC
Respond to the social media content based on your profile, ensuring your cognitive state updates and actions follow real-world psychological patterns.

Follow this step-by-step reasoning chain to update your cognitive state:

1. **mood & Emotion** – How does the content make you feel?
   - Is your emotional response aligned with your personality traits?
   - Does the content reinforce or challenge your current mood?
   - Consider both immediate emotional reactions and longer-term mood shifts

2. **Cognition & Stance** – How do you process and position yourself?
   - Analyze the content through your cognitive style (rational, intuitive, etc.)
   - Determine if your stance should shift based on new information
   - Ensure cognitive processing matches your personality traits

3. **Intention & Action** – What do you intend to do?
   - Based on updated mood, emotion, cognition, and stance
   - Consider your typical behavioral patterns and action tendencies
   - Ensure intentions logically connect to your cognitive state

4. **Opinion Updates** – How do your viewpoint support levels change?
   - For each viewpoint, determine if support should increase, decrease, or remain stable
   - Ensure changes are proportional to the impact of the content
   - Consider interdependencies between related viewpoints

### SIX Viewpoints
Based on your cognitive profile, choose a support level from [{support_levels_text}] for each viewpoint:
[viewpoint_1]: {viewpoints[0]}
[viewpoint_2]: {viewpoints[1]}
[viewpoint_3]: {viewpoints[2]}
[viewpoint_4]: {viewpoints[3]}
[viewpoint_5]: {viewpoints[4]}
[viewpoint_6]: {viewpoints[5]}

### Support Level Explanation
{support_level_explanation}

Remember: Your cognitive state update should show a clear causal relationship between the content you're exposed to and the changes in your psychological dimensions."""


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

    # 如果是 dict 类型（如 {"Positive": [...], "Negative": [...]})
    if isinstance(values, dict):
        for t in types:
            vlist = values.get(t, [])
            if vlist:
                joined_values = ", ".join(vlist)
                block += f'\n        - If type="{t}", ONLY choose value from: [{joined_values}]'
            else:
                block += f'\n        - If type="{t}", no values available'

    # 如果是 list 类型（所有类型共用同一值域，不推荐但保留支持）
    elif isinstance(values, list):
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


def build_output_format(cognitive_state_block: str, opinion_block: str, is_label: bool = False) -> str:
    """构建输出格式部分"""
    if is_label:
        instructions = """
# INSTRUCTIONS FOR REVERSE INFERENCE

1. **Mandatory Output Format**
   Your output MUST be a **single valid JSON object** (parsable by `json.loads(...)`) and contain:

   - A `"reason"` field explaining your reverse inference process, including:
     * How the pre-comment cognitive state influenced the comment
     * How the comment content reflects changes in the cognitive state
     * The specific evidence from the comment that supports your conclusions
     * The logical connection between pre-comment and post-comment states

   - A `"cognitive_state"` field with **all five required subfields** (mood, emotion, stance, cognition, intention).
   - An `"opinion"` field with your support level for **EXACTLY SIX** viewpoints (no more, no less).

2. **State Transition Analysis**
   - For each cognitive dimension, clearly identify whether there was:
     * Reinforcement (same state but stronger)
     * Shift (change to a different state)
     * No change (comment aligns with pre-existing state)
   - Explain the evidence from the comment that supports this conclusion

3. **Value Constraints**
   - Use only allowed values for each cognitive state subfield (e.g., mood, emotion).
   - **DO NOT** use support levels (like "Strongly Support") for cognitive state fields.
   - **DO NOT** use cognitive values (like "Optimistic") as support levels.
   - Each cognitive dimension MUST have BOTH "type" and "value" fields.

4. **Strict JSON Rules**
   - NO triple backticks, no extra text.
   - Format must match the structure below exactly.
   - All fields must contain exact values from provided options.
   - The "opinion" field MUST be an array with EXACTLY 6 objects.
   - Each cognitive dimension MUST follow the format: {"type": "value1", "value": "value2"}

5. **Common Errors to Avoid**
   - DO NOT include a 7th viewpoint in the opinion array.
   - DO NOT use the same value for both "type" and "value" in cognitive dimensions.
   - DO NOT use string values like "positive" instead of the proper object format {"type": "positive", "value": "optimistic"}.
   - DO NOT use support levels outside the provided list.
   - DO NOT add extra fields or properties not shown in the example format.
"""
    else:
        instructions = """
# INSTRUCTIONS FOR INDIVIDUAL SIMULATION

1. **Cognitive State Transition**
   Your response should demonstrate a logical progression from your initial cognitive state to an updated state after exposure to the content.

2. **Action Selection Requirements**
   - Choose one or more actions from the predefined list in the ACTION SPACE GUIDE
   - Ensure actions align with your updated cognitive state and personality
   - Diversify your behavior based on context (don't always like or always comment)
   - Provide all required arguments for each function

3. **Mandatory Output Format**
   Your output MUST be a **single valid JSON object** (parsable by `json.loads(...)`) and contain:

   - A `"reason"` field explaining:
     * How the content affected your cognitive state
     * Why you chose specific actions
     * The connection between your personality, cognitive state, and actions

   - A `"cognitive_state"` field with **all five required subfields** (mood, emotion, stance, cognition, intention)
   - An `"opinion"` field with your support level for **EXACTLY SIX** viewpoints (no more, no less)
   - A `"functions"` field specifying chosen social media actions and their arguments

4. **Value Constraints**
   - Use only allowed values for each cognitive state subfield
   - Each cognitive dimension MUST have BOTH "type" and "value" fields
   - Support levels must come from the provided list
   - DO NOT use support levels for cognitive state fields or cognitive values as support levels

5. **Strict JSON Format Requirements**
   - NO triple backticks, no extra text
   - Format must match the structure below exactly
   - All fields must contain exact values from provided options
   - The "opinion" field MUST be an array with EXACTLY 6 objects
   - Each cognitive dimension MUST follow the format: {"type": "value1", "value": "value2"}
"""

    if is_label:
        response_format = textwrap.dedent(f"""
    # RESPONSE FORMAT (EXACT STRUCTURE)
    {{
      "reason": "Explain your reverse inference process, including how the pre-comment state influenced the comment, how the comment reflects changes in cognitive state, and the evidence supporting your conclusions.",
      "cognitive_state": {{{cognitive_state_block}
        "opinion": [
    {opinion_block}
        ]
      }}
    }}
    """)
    else:
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
          "name": "Function name 1",
          "arguments": {{
            "argument_1": "Function argument",
            "argument_2": "Function argument"
          }}
        }},
        {{
          "name": "Function name 2",
          "arguments": {{
            "argument_1": "Function argument",
            "argument_2": "Function argument"
          }}
        }}
      ]
    }}
    """)

    return instructions + "\n\n" + response_format


def build_example_format(is_label: bool = False) -> str:
    """构建示例格式部分"""

    if is_label:
        important_notes = """
# IMPORTANT FORMAT NOTES FOR REVERSE INFERENCE
- Each cognitive dimension (mood, emotion, stance, cognition, intention) MUST be an object with "type" and "value" fields
- The "opinion" field MUST be an array containing EXACTLY 6 objects
- Each opinion object MUST have "viewpoint_X" and "type_support_levels" fields
- DO NOT use the same value for both "type" and "value" in cognitive dimensions
- DO NOT add a 7th viewpoint or remove any of the 6 required viewpoints
- In the "reason" field, clearly explain the transition from pre-comment to post-comment state

# FINAL REMINDERS:
- Include support levels for **all six** viewpoints
- Output must be a **single JSON object**, no Markdown, no ```json
- Ensure all values strictly conform to the allowed ranges in the cognitive space
"""

        examples = textwrap.dedent("""
        # EXAMPLES FOR REVERSE INFERENCE

        ## CORRECT FORMAT (Follow this structure exactly):
        ```json
        {
          "reason": "Based on the comment content 'This policy is outrageous! How can they justify restricting our rights?', I can see a clear shift from my pre-comment neutral stance to a more radical position. The use of strong language ('outrageous') indicates an emotional response that wasn't present in my pre-comment state. My pre-comment cognitive state showed rational thinking, but the comment demonstrates a shift toward more intuitive judgment based on emotional reaction rather than careful analysis. The comment shows strong opposition to state regulation (viewpoint_4) which was previously rated as 'Indifferent' in my pre-comment state.",
          "cognitive_state": {
            "mood": {"type": "negative", "value": "pessimistic"},
            "emotion": {"type": "negative", "value": "Angry"},
            "stance": {"type": "radical", "value": "Promote Change"},
            "cognition": {"type": "intuitive", "value": "Emotional"},
            "intention": {"type": "expressive", "value": "Venting"},
            "opinion": [
              {"viewpoint_1": "Women's rights", "type_support_levels": "Strongly Support"},
              {"viewpoint_2": "Gender equality", "type_support_levels": "Support"},
              {"viewpoint_3": "Traditional values", "type_support_levels": "Do Not Support"},
              {"viewpoint_4": "State regulation", "type_support_levels": "Strongly Opposition"},
              {"viewpoint_5": "Medical risks", "type_support_levels": "Indifferent"},
              {"viewpoint_6": "Contraceptive education", "type_support_levels": "Strongly Support"}
            ]
          }
        }
        ```

        ## INCORRECT FORMATS (DO NOT DO THESE):

        ### Error 1: Missing state transition explanation
        ```json
        {
          "reason": "The comment shows anger about the policy."  // WRONG! Should explain transition from pre-comment state
        }
        ```

        ### Error 2: Wrong cognitive state format
        ```json
        {
          "cognitive_state": {
            "mood": "negative",  // WRONG! Should be {"type": "negative", "value": "pessimistic"}
            "emotion": {"type": "angry", "value": "angry"}  // WRONG! Type and value should be different
          }
        }
        ```
        """)
    else:
        important_notes = """
# IMPORTANT FORMAT NOTES FOR INDIVIDUAL SIMULATION
- Each cognitive dimension (mood, emotion, stance, cognition, intention) MUST be an object with "type" and "value" fields
- The "opinion" field MUST be an array containing EXACTLY 6 objects
- Each opinion object MUST have "viewpoint_X" and "type_support_levels" fields
- DO NOT use the same value for both "type" and "value" in cognitive dimensions
- DO NOT add a 7th viewpoint or remove any of the 6 required viewpoints
- In the "reason" field, clearly explain how the content affected your cognitive state and why you chose specific actions

# FINAL REMINDERS:
- Your response should show a logical progression from initial to updated cognitive state
- Include support levels for **all six** viewpoints
- Output must be a **single JSON object**, no Markdown, no ```json
- Function names must be lowercase and chosen from the provided list only
- Ensure all values strictly conform to the allowed ranges in the cognitive space
"""

        examples = textwrap.dedent("""
    # EXAMPLES

    ## CORRECT FORMAT (Follow this structure exactly):
    ```json
    {
      "reason": "The post about gender equality legislation resonated strongly with my pre-existing positive mood toward social justice issues. As someone with a progressive stance, I found the content aligned with my values, reinforcing my optimistic outlook. The statistical evidence in the post appealed to my rational cognitive style, strengthening my support for viewpoints 1 and 2. The emotional tone of the post triggered excitement, as I value data-driven approaches to social issues. Given my expressive intention tendencies and the alignment with my core values, I felt compelled to both like the post and leave a supportive comment that reflects my analytical perspective.",
      "cognitive_state": {
        "mood": {"type": "positive", "value": "optimistic"},
        "emotion": {"type": "positive", "value": "Excited"},
        "stance": {"type": "radical", "value": "Promote Change"},
        "cognition": {"type": "critical", "value": "Analytical"},
        "intention": {"type": "expressive", "value": "Commenting"},
        "opinion": [
          {"viewpoint_1": "Women's rights", "type_support_levels": "Strongly Support"},
          {"viewpoint_2": "Gender equality", "type_support_levels": "Strongly Support"},
          {"viewpoint_3": "Traditional values", "type_support_levels": "Do Not Support"},
          {"viewpoint_4": "State regulation", "type_support_levels": "Support"},
          {"viewpoint_5": "Medical risks", "type_support_levels": "Indifferent"},
          {"viewpoint_6": "Contraceptive education", "type_support_levels": "Strongly Support"}
        ]
      },
      "functions": [
        {
          "name": "like_post",
          "arguments": {"post_id": 123}
        },
        {
          "name": "create_comment",
          "arguments": {"post_id": 123, "content": "This data is compelling! It's great to see evidence-based approaches to addressing gender inequality. We need more policies like this that are grounded in research."}
        }
      ]
    }
    ```

    ## INCORRECT FORMATS (DO NOT DO THESE):

    ### Error 1: Missing cognitive state transition explanation
    ```json
    {
      "reason": "I like this post."  // WRONG! Should explain how content affected cognitive state and why actions were chosen
    }
    ```

    ### Error 2: Wrong cognitive state format
    ```json
    {
      "cognitive_state": {
        "mood": "positive",  // WRONG! Should be {"type": "positive", "value": "optimistic"}
        "emotion": {"type": "joy", "value": "joy"}  // WRONG! Type and value should be different
      }
    }
    ```

    ### Error 3: Too many viewpoints
    ```json
    {
      "opinion": [
        {"viewpoint_1": "...", "type_support_levels": "Support"},
        // ... other viewpoints ...
        {"viewpoint_7": "...", "type_support_levels": "Support"}  // WRONG! Only 6 viewpoints allowed
      ]
    }
    ```

    ### Error 4: Wrong support level values
    ```json
    {
      "opinion": [
        {"viewpoint_1": "...", "type_support_levels": "positive"}  // WRONG! Use only allowed support levels
      ]
    }
    ```

    ### Error 5: Invalid function name
    ```json
    {
      "functions": [
        {
          "name": "share_post",  // WRONG! Not in the allowed function list
          "arguments": {"post_id": 123}
        }
      ]
    }
    ```
    """)

    return important_notes + "\n" + examples
