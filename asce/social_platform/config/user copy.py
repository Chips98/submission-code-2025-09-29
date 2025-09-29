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
# flake8: noqa: E501
from dataclasses import dataclass
from typing import Any
import json
import textwrap


@dataclass
class UserInfo:
    name: str | None = None
    description: str | None = None
    profile: dict[str, Any] | None = None
    recsys_type: str = "twitter",
    is_controllable: bool = False

    def to_system_message(self, dataset_name: str, action_space_prompt: str = None, cognition_space_dict: dict = None) -> str:
        if self.recsys_type != "reddit":
            return self.to_twitter_system_message(dataset_name, action_space_prompt, cognition_space_dict)
        else:
            return self.to_reddit_system_message(dataset_name, action_space_prompt, cognition_space_dict)

    def to_twitter_system_message(self, dataset_name,
                                  action_space_prompt: str = None,
                                  cognition_space_dict: dict = None) -> str:
        name_string = ""
        description_string = ""
        if self.name is not None:
            name_string = f"Your name is {self.name}."
        if self.profile is None:
            description = name_string
        elif "other_info" not in self.profile:
            description = name_string
        elif "user_profile" in self.profile["other_info"]:
            if self.profile["other_info"]["user_profile"] is not None:
                user_profile = self.profile["other_info"]["user_profile"]
                description_string = f"Your have profile: {user_profile}."
                description = f"{name_string}\n{description_string}"

        if not action_space_prompt:
            action_space_prompt = """
# OBJECTIVE
You're a Twitter user, and I'll present you with some posts. After you see the posts, choose some actions from the following functions.

- do_nothing: Most of the time, you just don't feel like reposting or liking a post, and you just want to look at it. In such cases, choose this action "do_nothing"
- create_post:Create a new post with the given content.
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
        system_content = action_space_prompt + f"""
# SELF-DESCRIPTION
Your actions should be consistent with your self-description and personality.

{description}



# RESPONSE FORMAT
Your answer should follow the response format:

{{
    "reason": "your feeling about these tweets and users, then choose some functions based on the feeling. Reasons and explanations can only appear here.",
    "functions": [{{
        "name": "Function name 1",
        "arguments": {{
            "argument_1": "Function argument",
            "argument_2": "Function argument"
        }}
    }}, {{
        "name": "Function name 2",
        "arguments": {{
            "argument_1": "Function argument",
            "argument_2": "Function argument"
        }}
    }}]
}}

Ensure that your output can be directly converted into **JSON format**, and avoid outputting anything unnecessary! Don't forget the key `name`.
        """

        return system_content

    def to_reddit_system_message(self, dataset_name, action_space_prompt: str = None, cognition_space_dict: dict = None) -> str:
        name_string = ""
        description_string = ""
        if self.name is not None:
            name_string = f"Your name is {self.name}."
        if self.profile is None:
            description = name_string
        elif "other_info" not in self.profile:
            description = name_string
        elif "user_profile" in self.profile["other_info"]:
            if self.profile["other_info"]["user_profile"] is not None:
                user_profile = self.profile["other_info"]["user_profile"]
                description_string = f"Your have profile: {user_profile}."
                description = f"{name_string}\n{description_string}"
                # 使用get方法安全地获取字段，并提供默认值
                other_info = self.profile["other_info"]
                gender = other_info.get("gender", "None")
                age = other_info.get("age", "None")
                mbti = other_info.get("mbti", "None")
                country = other_info.get("country", "None")
                profession = other_info.get("profession", "None")
                interested_topics = other_info.get("interested_topics", ["None"])
                # 获取影响力指标，如果不存在则使用默认值
                influence_metrics = other_info.get("influence_metrics", {
                    "like_count": 0,
                    "retweet_count": 0,
                    "influence_score": 0
                })
                like_count = influence_metrics.get("like_count", 0)
                retweet_count = influence_metrics.get("retweet_count", 0)
                influence_score = influence_metrics.get("influence_score", 0)
                
                # 构建描述文本
                description += (
                    f"You are a {gender}, "
                    f"{age} years old, with an MBTI "
                    f"personality type of {mbti} from "
                    f"{country}. "
                    f"You work as a {profession} and are interested in "
                    f"{', '.join(interested_topics)}. "
                    f"Your social influence metrics show {like_count} likes, "
                    f"{retweet_count} retweets, and an influence score of "
                    f"{influence_score}.")
        
        cognitive_description = self._build_cognitive_description(dataset_name, cognition_space_dict)
        
        # 确保action_space_prompt不为None，避免TypeError
        if action_space_prompt is None:
            action_space_prompt = ""
            print(f"警告: action_space_prompt为None，已使用空字符串替代")
        else:
            print(f"收到action_space_prompt，长度: {len(action_space_prompt)}")
        
        #print(f"认知描述长度: {len(cognitive_description)}")
        
        system_content = action_space_prompt + cognitive_description
        return system_content
    
    def gen_type_value_block(self, name: str, types: list, values: dict | list) -> str:
        """构建认知维度字段的 type/value 说明块（支持分 type 显示不同值域）"""
        if not types:
            return ""

        # 使用更明确的格式指导
        block = f'\n    "{name}": {{\n      "type": "EXACTLY ONE VALUE from: [{", ".join(types)}]",\n      "value": "EXACTLY ONE VALUE based on your chosen type:"'
        
        # 如果是 dict 类型（如 {"Positive": [...], "Negative": [...]})        
        if isinstance(values, dict):
            for t in types:
                vlist = values.get(t, [])
                if vlist:
                    block += f'\nIf type="{t}", ONLY USE ONE VALUE from: [{", ".join(vlist)}]'
                else:
                    block += f'\nIf type="{t}", ONLY USE ONE VALUE from: []'

        # 如果是 list 类型（所有类型共用同一值域，不推荐但保留支持）
        elif isinstance(values, list):
            block += f'\n(Shared across all types) ONLY USE ONE VALUE from: [{", ".join(values)}]'

        block += "\n},"
        return block

    def get_support_level_explanations(self, json_path: str, selected_viewpoints: list[str]) -> str:
        """获取支持级别解释说明"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 尝试找到包含opinion的键
            opinion_key = None
            for key in data.keys():
                if isinstance(data[key], dict) and "opinion" in data[key]:
                    opinion_key = key
                    break
            
            if not opinion_key:
                print(f"警告: 在{json_path}中找不到包含opinion的键")
                print(f"可用的顶级键: {', '.join(data.keys())}")
                return ""
            
            opinions = data[opinion_key]["opinion"]
        
            explanation_blocks = []
            for i, vp in enumerate(opinions):
                vp_key = f"viewpoint_{i+1}"
                if vp_key in selected_viewpoints or vp.get(vp_key, "") in selected_viewpoints:
                    levels = vp.get("type_support_levels", {})
                    # 确保levels是字典
                    if not isinstance(levels, dict):
                        print(f"警告: viewpoint_{i+1}的support_levels不是字典: {levels}")
                        continue
                    level_lines = [f"{level}: {desc}" for level, desc in levels.items()]
                    explanation_blocks.append(f"[{vp.get(vp_key, vp_key)} Support Level Explanation]:\n" + "\n".join(level_lines))
            
            return "\n\n".join(explanation_blocks)
        except Exception as e:
            print(f"获取支持级别解释时出错: {str(e)}")
            import traceback
            print(f"错误详情: {traceback.format_exc()}")
            return ""

    def gen_opinion_block(self, fixed_viewpoints: list, support_levels: list) -> str:
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

    def _build_cognitive_description(self, dataset_name, cognitive_space: dict = None) -> str:
        """重构后的认知描述构建函数，集成模块化认知维度提示构建逻辑"""
        try:
            # 检查认知空间有效性
            if not cognitive_space or not isinstance(cognitive_space, dict):
                raise ValueError("认知空间无效或为空，请确保已正确加载认知空间数据")

            # 所需字段
            required_fields = ["mood", "emotion", "stance", "cognition", "intention", "opinion_list", "opinion_support_levels"]
            for field in required_fields:
                if field not in cognitive_space:
                    raise ValueError(f"认知空间缺少必要字段: {field}")

            # 提取各维度类型和值
            mood_types = cognitive_space["mood"]["type_list"]
            mood_values = cognitive_space["mood"]["value_list"]
            emotion_types = cognitive_space["emotion"]["type_list"]
            emotion_values = cognitive_space["emotion"]["value_list"]
            stance_types = cognitive_space["stance"]["type_list"]
            stance_values = cognitive_space["stance"]["value_list"]
            cognition_types = cognitive_space["cognition"]["type_list"]
            cognition_values = cognitive_space["cognition"]["value_list"]
            intention_types = cognitive_space["intention"]["type_list"]
            intention_values = cognitive_space["intention"]["value_list"]

            # 处理观点及支持级别
            fixed_viewpoints = cognitive_space["opinion_list"]
            support_levels = cognitive_space["opinion_support_levels"]

            actual_viewpoints = fixed_viewpoints.copy()
            while len(actual_viewpoints) < 6:
                actual_viewpoints.append(f"Viewpoint {len(actual_viewpoints)+1}")

            json_path = f'/Users/zl_24/Documents/Codes/2025-2/ASCE-main/cognition_space/combined_{dataset_name}_cognition_space.json'
            print(f"使用认知空间文件: {json_path}")
            
            support_level_explanation = self.get_support_level_explanations(
                json_path=json_path,
                selected_viewpoints=[f"viewpoint_{i+1}" for i in range(6)])

            user_description = getattr(self, 'description', '') or ""

            # 使用模块化函数构建各认知字段块
            cognitive_state_block = ''.join([
                self.gen_type_value_block('mood', mood_types, mood_values),
                self.gen_type_value_block('emotion', emotion_types, emotion_values),
                self.gen_type_value_block('stance', stance_types, stance_values),
                self.gen_type_value_block('cognition', cognition_types, cognition_values),
                self.gen_type_value_block('intention', intention_types, intention_values),
            ])

            opinion_block = self.gen_opinion_block(actual_viewpoints, support_levels)


            part1 = f"""

# SELF-DESCRIPTION
Your actions must align with your self-description, cognitive states, and personality.
{user_description}

# COGNITIVE RESPONSE LOGIC
Respond to the post or social interaction based on your profile, ensuring your actions follow real-world logic. Then, update your cognitive state accordingly.

Follow this step-by-step reasoning chain:
1. **Emotion** – How do you feel about the content? Is it spontaneous or aligned with your long-term mood?
2. **Cognition** – How do you interpret the content based on your cognitive style?
3. **Stance** – What is your position on this issue?
4. **Intention** – What actions are you planning to take?
5. **Opinion** – Express your support level for each of the six viewpoints below.

### SIX Viewpoints
Based on your cognitive profile, choose a support level from [{', '.join(support_levels)}] for each viewpoint:
[viewpoint_1]: {actual_viewpoints[0]}
[viewpoint_2]: {actual_viewpoints[1]}
[viewpoint_3]: {actual_viewpoints[2]}
[viewpoint_4]: {actual_viewpoints[3]}
[viewpoint_5]: {actual_viewpoints[4]}
[viewpoint_6]: {actual_viewpoints[5]}

### Support Level Explanation
{support_level_explanation}
"""

            # 第2部分：指令
            part2 = """
# INSTRUCTIONS

1. **Action Requirement**  
   You MUST select one or more actions from the predefined list in the OBJECTIVE section and provide necessary arguments.

2. **Realistic Behavior**  
   Avoid only performing one type of action (e.g., like). Diversify your behavior (e.g., comment, follow) to enrich your social interaction.

3. **Mandatory Output Format**  
   Your output MUST be a **single valid JSON object** (parsable by `json.loads(...)`) and contain:

   - A `"reason"` field explaining your choices based on your cognitive state, personality, and memory.
   - A `"cognitive_state"` field with **all five required subfields** (mood, emotion, stance, cognition, intention).
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

            # 第3部分：响应格式 - 使用三重引号避免嵌套的大括号问题
            part3 = textwrap.dedent("""
            # RESPONSE FORMAT (EXACT STRUCTURE)
            {
              "reason": "Explain your reasoning based on your cognitive state, personality, and memory.",
              "cognitive_state": {""") + cognitive_state_block + textwrap.dedent("""
                "opinion": [
            """) + opinion_block + textwrap.dedent("""
                ]
              },
              "functions": [
                {
                  "name": "Function name 1",
                  "arguments": {
                    "argument_1": "Function argument",
                    "argument_2": "Function argument"
                  }
                },
                {
                  "name": "Function name 2",
                  "arguments": {
                    "argument_1": "Function argument",
                    "argument_2": "Function argument"
                  }
                }
              ]
            }
            """)

            # 第4部分：重要格式注意事项
            part4 = """
# IMPORTANT FORMAT NOTES
- Each cognitive dimension (mood, emotion, stance, cognition, intention) MUST be an object with "type" and "value" fields
- The "opinion" field MUST be an array containing EXACTLY 6 objects
- Each opinion object MUST have "viewpoint_X" and "type_support_levels" fields
- DO NOT use the same value for both "type" and "value" in cognitive dimensions
- DO NOT add a 7th viewpoint or remove any of the 6 required viewpoints

# FINAL REMINDERS:
- Include support levels for **all six** viewpoints.
- Output must be a **single JSON object**, no Markdown, no ```json.
- Function names must be lowercase and chosen from: like_comment, dislike_comment, like_post, dislike_post, search_posts, search_user, refresh, do_nothing.
"""

            # 第5部分：示例 - 使用三重引号避免嵌套的大括号问题
            part5 = textwrap.dedent("""
            # EXAMPLES

            ## CORRECT FORMAT (Follow this structure exactly):
            ```json
            {
              "reason": "I feel strongly about this post because...",
              "cognitive_state": {
                "mood": {"type": "positive", "value": "optimistic"},
                "emotion": {"type": "positive", "value": "Excited"},
                "stance": {"type": "radical", "value": "Promote Change"},
                "cognition": {"type": "critical", "value": "Skeptical"},
                "intention": {"type": "expressive", "value": "Commenting"},
                "opinion": [
                  {"viewpoint_1": "Women's rights", "type_support_levels": "Strongly Support"},
                  {"viewpoint_2": "Gender equality", "type_support_levels": "Moderate Support"},
                  {"viewpoint_3": "Traditional values", "type_support_levels": "Do Not Support"},
                  {"viewpoint_4": "State regulation", "type_support_levels": "Strongly Opposition"},
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
                  "arguments": {"post_id": 123, "content": "I strongly agree with this!"}
                }
              ]
            }
            ```

            ## INCORRECT FORMATS (DO NOT DO THESE):

            ### Error 1: Wrong cognitive state format
            ```json
            {
              "cognitive_state": {
                "mood": "positive",  // WRONG! Should be {"type": "positive", "value": "optimistic"}
                "emotion": {"type": "joy", "value": "joy"}  // WRONG! Type and value should be different
              }
            }
            ```

            ### Error 2: Too many viewpoints
            ```json
            {
              "opinion": [
                {"viewpoint_1": "...", "type_support_levels": "Support"},
                // ... other viewpoints ...
                {"viewpoint_7": "...", "type_support_levels": "Support"}  // WRONG! Only 6 viewpoints allowed
              ]
            }
            ```

            ### Error 3: Wrong support level values
            ```json
            {
              "opinion": [
                {"viewpoint_1": "...", "type_support_levels": "positive"}  // WRONG! Use only allowed support levels
              ]
            }
            ```
            """)

            # 将所有部分组合在一起
            cognitive_description = part1 + part2 + part3 + part4 + part5
            
            return cognitive_description
        except Exception as e:
            print(f"构建认知描述时出错: {str(e)}")
            return ""
