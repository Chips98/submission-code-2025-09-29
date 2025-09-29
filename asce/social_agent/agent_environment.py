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
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from string import Template

from asce.social_agent.agent_action import SocialAction


class Environment(ABC):

    @abstractmethod
    def to_text_prompt(self) -> str:
        r"""Convert the environment to text prompt."""
        raise NotImplementedError


class SocialEnvironment(Environment):
    followers_env_template = Template("I have $num_followers followers.")
    follows_env_template = Template("I have $num_follows follows.")

    posts_env_template = Template(
        "After refreshing, you see some posts $posts")
    trending_comments_env_template = Template(
        "Here are some trending comments: $trending_comments")
    env_template = Template(
        "$refresh_env\n$posts_env\n$trending_comments_env\n$followers_env\n$follows_env")

    def __init__(self, action: SocialAction):
        self.action = action
        self.platform = None  # 添加平台引用

    async def get_posts_env(self) -> str:
        posts = await self.action.refresh()
        # TODO: Replace posts json format string to other formats
        if posts["success"]:
            posts_env = json.dumps(posts["posts"], indent=4)
            posts_env = self.posts_env_template.substitute(posts=posts_env)
        else:
            posts_env = "After refreshing, there are no existing posts."
        return posts_env

    async def get_trending_comments_env(self) -> str:
        trending_comments = await self.action.get_trending_comments()
        if trending_comments["success"]:
            trending_comments_env = json.dumps(trending_comments["comments"], indent=4)
            trending_comments_env = self.trending_comments_env_template.substitute(
                trending_comments=trending_comments_env)
        else:
            trending_comments_env = "No trending comments available."
        return trending_comments_env

    async def get_followers_env(self) -> str:
        # TODO: Implement followers env
        return self.followers_env_template.substitute(num_followers=0)

    async def get_follows_env(self) -> str:
        # TODO: Implement follows env
        return self.follows_env_template.substitute(num_follows=0)

    async def to_text_prompt(
        self,
        include_posts: bool = True,
        include_followers: bool = False,
        include_follows: bool = False,
        include_trending_comments: bool = True,
    ) -> str:
        followers_env = (await self.get_followers_env()
                         if include_follows else "No followers.")
        follows_env = (await self.get_follows_env()
                       if include_followers else "No follows.")
        posts_env = await self.get_posts_env() if include_posts else ""
        trending_comments_env = (await self.get_trending_comments_env()
                                if include_trending_comments else "")
        refresh_env = "You have refreshed your timeline."

        return self.env_template.substitute(
            refresh_env=refresh_env,
            followers_env=followers_env,
            follows_env=follows_env,
            posts_env=posts_env,
            trending_comments_env=trending_comments_env,
        )
