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

    def to_system_message(self, dataset_name: str, action_space_prompt: str = None, cognition_space_dict: dict = None, use_camel: bool = False) -> str:
        if self.recsys_type != "reddit":
            return self.to_twitter_system_message(dataset_name, action_space_prompt, cognition_space_dict)
        else:
            return self.to_reddit_system_message(dataset_name, action_space_prompt, cognition_space_dict, use_camel)

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

    def to_reddit_system_message(self, dataset_name, action_space_prompt: str = None, cognition_space_dict: dict = None, use_camel: bool = False) -> str:
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

        # 确保action_space_prompt不为None，避免TypeError
        if action_space_prompt is None:
            action_space_prompt = ""
            print(f"警告: action_space_prompt为None，已使用空字符串替代")
        else:
            print(f"收到action_space_prompt，长度: {len(action_space_prompt)}")

        # 根据是否使用Camel格式选择不同的认知描述构建方法
        if use_camel:
            cognitive_description = self._build_cognitive_description_camel(dataset_name, cognition_space_dict)
            print(f"使用Camel格式的认知描述，长度: {len(cognitive_description)}")
        else:
            cognitive_description = self._build_cognitive_description(dataset_name, cognition_space_dict)
            print(f"使用标准格式的认知描述，长度: {len(cognitive_description)}")

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
        """构建认知描述

        参数:
            dataset_name: 数据集名称
            cognitive_space: 认知空间字典

        返回:
            认知描述字符串
        """
        if not cognitive_space or not isinstance(cognitive_space, dict):
            return ""

        # 获取认知空间
        cognitive_space_data = cognitive_space.get("cognitive_space", {})

        # Build cognitive description
        cognitive_description = "\n\n# Cognitive Space\n"

        # Add cognitive states description
        cognitive_description += "## Cognitive States\n"
        cognitive_states = cognitive_space_data.get("cognitive_states", {})
        for state_name, state_values in cognitive_states.items():
            cognitive_description += f"- {state_name}: {', '.join(state_values)}\n"

        # Add opinions description
        cognitive_description += "\n## Opinions\n"
        opinions = cognitive_space_data.get("opinions", {})
        for opinion_name, opinion_values in opinions.items():
            cognitive_description += f"- {opinion_name}: {', '.join(opinion_values)}\n"

        # Add support levels description
        cognitive_description += "\n## Support Levels\n"
        support_levels = cognitive_space_data.get("support_levels", {})
        for level_name, level_values in support_levels.items():
            cognitive_description += f"- {level_name}: {', '.join(level_values)}\n"

        return cognitive_description

    def _build_cognitive_description_camel(self, dataset_name, cognitive_space: dict = None) -> str:
        """使用Camel框架格式构建认知描述函数

        注意：此方法生成的提示内容与原始提示完全一致，只是添加了Camel解析所需的标记
        """
        try:
            # 首先获取原始的认知描述
            original_description = self._build_cognitive_description(dataset_name, cognitive_space)

            # Add Camel parsing guidance
            camel_guide = """
# CAMEL RESPONSE FORMAT
When generating a response, please ensure that your cognitive state and function calls are wrapped in an <asce_response> tag, formatted as follows:

<asce_response>
{
  "reason": "Your reasoning process",
  "cognitive_state": {
    "mood": {"type": "[type]", "value": "[value]"},
    "emotion": {"type": "[type]", "value": "[value]"},
    "stance": {"type": "[type]", "value": "[value]"},
    "cognition": {"type": "[type]", "value": "[value]"},
    "intention": {"type": "[type]", "value": "[value]"}
  },
  "opinion": {
    "viewpoint_1": "[support level]",
    "viewpoint_2": "[support level]",
    "viewpoint_3": "[support level]",
    "viewpoint_4": "[support level]",
    "viewpoint_5": "[support level]",
    "viewpoint_6": "[support level]"
  },
  "functions": [
    {
      "name": "[function name]",
      "arguments": {
        "[parameter name]": "[parameter value]"
      }
    }
  ]
}
</asce_response>

After the <asce_response> tag, you can add additional explanations or responses.
Note: The format above is completely consistent with the original format, just wrapped in an <asce_response> tag.
"""

            # 将原始描述和Camel指导组合在一起
            return original_description + camel_guide
        except Exception as e:
            print(f"构建Camel认知描述时出错: {str(e)}")
            return ""
