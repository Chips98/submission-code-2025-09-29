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
import json
import os
from datetime import datetime


class PlatformUtils:
    """平台工具类，提供各种数据库操作和辅助功能"""

    def __init__(self, db, db_cursor, start_time, sandbox_clock, show_score):
        """初始化函数，设置平台工具类的基本属性

        参数:
        db - 数据库连接对象
        db_cursor - 数据库游标对象，用于执行SQL命令
        start_time - 平台开始时间
        sandbox_clock - 沙盒时钟对象，用于时间模拟
        show_score - 布尔值，决定是否显示分数而不是单独显示赞和踩的数量
        """
        self.db = db                      # 存储数据库连接
        self.db_cursor = db_cursor        # 存储数据库游标
        self.start_time = start_time      # 存储平台开始时间
        self.sandbox_clock = sandbox_clock # 存储沙盒时钟对象
        self.show_score = show_score      # 存储是否显示分数的设置

    @staticmethod
    def _not_signup_error_message(agent_id):
        """生成用户未注册的错误消息

        这是一个静态方法（@staticmethod装饰器表示不需要实例即可调用），
        用于生成标准化的错误消息，当代理未注册时使用。

        参数:
        agent_id - 未注册的代理ID

        返回:
        包含错误信息的字典，success字段为False
        """
        return {
            "success":
            False,
            "error": (f"Agent {agent_id} has not signed up and does not have "
                      f"a user id."),
        }

    def _execute_db_command(self, command, params=None, commit=False):
        """
        执行单个数据库命令

        参数:
            command (str): SQL命令
            params (tuple, optional): SQL参数
            commit (bool, optional): 是否提交事务

        返回:
            cursor: 数据库游标
        """
        try:

            if params:
                self.db_cursor.execute(command, params)
            else:
                self.db_cursor.execute(command)

            if commit:
                self.db.commit()  # 修正: 使用self.db而不是self.conn
            return self.db_cursor
        except Exception as e:
            #print(f"==DEBUG== 数据库命令执行失败: {str(e)}")
            raise e

    def _execute_many_db_command(self, command, args_list, commit=False):
        """执行多条数据库命令

        用于批量执行相同的SQL命令但参数不同的情况，提高效率。

        参数:
        command - SQL命令字符串
        args_list - 参数列表，每个元素是一个参数元组
        commit - 布尔值，是否在执行后提交事务，默认为False

        返回:
        数据库游标对象
        """
        self.db_cursor.executemany(command, args_list)  # 批量执行SQL命令
        if commit:
            self.db.commit()  # 如果需要，提交事务
        return self.db_cursor  # 返回游标对象

    def _check_agent_userid(self, agent_id):
        """检查代理ID对应的用户ID

        通过查询数据库，获取指定代理ID对应的用户ID。

        参数:
        agent_id - 要查询的代理ID

        返回:
        如果找到对应的用户ID，返回该ID；否则返回None
        """
        try:
            # 构建SQL查询语句
            user_query = "SELECT user_id FROM user WHERE agent_id = ?"
            # 执行查询
            results = self._execute_db_command(user_query, (agent_id, ))
            # 获取查询结果的第一行
            first_row = results.fetchone()
            if first_row:
                # 如果有结果，返回用户ID（第一列）
                user_id = first_row[0]
                return user_id
            else:
                # 如果没有结果，返回None
                return None
        except Exception as e:
            # 捕获并处理可能的异常
            # 打印错误信息
            #print(f"Error querying user_id for agent_id {agent_id}: {e}")
            return None

    def _add_comments_to_posts(self, posts_results):
        """为帖子添加评论信息

        此函数处理从数据库查询到的帖子结果，为每个帖子添加相关评论，
        并根据帖子类型（普通帖子、转发、引用）格式化内容。
        支持嵌套评论结构，父评论和子评论通过parent_comment_id关联。

        参数:
        posts_results - 从数据库查询到的帖子结果列表

        返回:
        包含完整帖子信息（包括评论）的列表
        """
        # 初始化返回的帖子列表
        posts = []
        # 遍历查询结果中的每一行
        for row in posts_results:
            # 解包查询结果，获取帖子的各个字段
            (post_id, user_id, original_post_id, content, quote_content,
             created_at, num_likes, num_dislikes, num_shares) = row
            # 获取帖子类型信息（原始帖子、转发帖子、引用帖子）
            post_type_result = self._get_post_type(post_id)
            # 如果无法获取帖子类型，跳过当前帖子
            if post_type_result is None:
                continue
            # 构建查询原始用户ID的SQL语句
            original_user_id_query = (
                "SELECT user_id FROM post WHERE post_id = ?")

            # 处理不同类型的帖子
            # 如果是转发类型的帖子
            if post_type_result["type"] == "repost":
                # 查询原始帖子的用户ID
                self.db_cursor.execute(original_user_id_query,
                                       (original_post_id, ))
                original_user_id = self.db_cursor.fetchone()[0]
                # 保存转发帖子的ID
                original_post_id = post_id
                # 获取原始帖子的ID
                post_id = post_type_result["root_post_id"]
                # 查询原始帖子的详细信息
                self.db_cursor.execute(
                    "SELECT content, quote_content, created_at, num_likes, "
                    "num_dislikes, num_shares FROM post WHERE post_id = ?",
                    (post_id, ))
                original_post_result = self.db_cursor.fetchone()
                # 解包原始帖子的查询结果
                (content, quote_content, created_at, num_likes, num_dislikes,
                 num_shares) = original_post_result
                # 格式化转发帖子的内容
                post_content = (
                    f"User {user_id} reposted （转发） a post from User "
                    f"{original_user_id}. Repost content （转发内容）: {content}. ")

            # 如果是引用类型的帖子
            elif post_type_result["type"] == "quote":
                # 查询被引用帖子的用户ID
                self.db_cursor.execute(original_user_id_query,
                                       (original_post_id, ))
                original_user_id = self.db_cursor.fetchone()[0]
                # 格式化引用帖子的内容
                post_content = (
                    f"User {user_id} quoted a post from User "
                    f"{original_user_id}. Quote content: {quote_content}. "
                    f"Original Content: {content}")

            # 如果是普通帖子
            elif post_type_result["type"] == "common":
                # 直接使用帖子内容
                post_content = content

            # 对每个帖子，查询其对应的评论（包含嵌套评论结构）
            # 获取平台设置的最大可见评论数量，如果未设置则默认为所有评论
            # 注意：max_visible_comments控制每个帖子显示的顶级评论数量，同时会获取这些顶级评论的所有回复
            # 这样可以保持评论的上下文完整性，同时限制总体评论数量，防止上下文溢出
            max_visible_comments = getattr(self, 'max_visible_comments', None)

            # 构建SQL查询，根据是否有评论数量限制添加LIMIT子句
            if max_visible_comments is not None:
                # 首先获取顶级评论（没有父评论的评论）
                query = (
                    "SELECT comment_id, post_id, user_id, parent_comment_id, content, created_at, "
                    "num_likes, num_dislikes, num_replies, num_shares FROM comment "
                    "WHERE post_id = ? AND parent_comment_id IS NULL ORDER BY num_likes DESC LIMIT ?"
                )
                self.db_cursor.execute(query, (post_id, max_visible_comments))

                # 获取顶级评论的结果
                top_level_results = self.db_cursor.fetchall()

                # 如果有顶级评论，获取它们的回复
                if top_level_results:
                    # 提取顶级评论的ID
                    top_level_ids = [comment[0] for comment in top_level_results]

                    # 构建IN子句的参数占位符
                    placeholders = ','.join(['?'] * len(top_level_ids))

                    # 获取这些顶级评论的回复
                    reply_query = (
                        f"SELECT comment_id, post_id, user_id, parent_comment_id, content, created_at, "
                        f"num_likes, num_dislikes, num_replies, num_shares FROM comment "
                        f"WHERE post_id = ? AND parent_comment_id IN ({placeholders})"
                    )

                    # 构建参数列表：post_id和所有顶级评论ID
                    params = [post_id] + top_level_ids
                    self.db_cursor.execute(reply_query, params)

                    # 获取回复结果
                    reply_results = self.db_cursor.fetchall()

                    # 合并顶级评论和回复
                    comments_results = top_level_results + reply_results
                else:
                    comments_results = top_level_results
            else:
                # 如果没有设置评论数量限制，获取所有评论
                query = (
                    "SELECT comment_id, post_id, user_id, parent_comment_id, content, created_at, "
                    "num_likes, num_dislikes, num_replies, num_shares FROM comment "
                    "WHERE post_id = ?"
                )
                self.db_cursor.execute(query, (post_id,))
                comments_results = self.db_cursor.fetchall()

            # 将评论组织成嵌套结构
            comments_dict = {}  # 用于临时存储所有评论
            top_level_comments = []  # 存储顶级评论

            # 第一遍循环：创建所有评论对象
            for comment_row in comments_results:
                (comment_id, comment_post_id, comment_user_id, parent_comment_id, comment_content,
                 comment_created_at, comment_num_likes, comment_num_dislikes,
                 comment_num_replies, comment_num_shares) = comment_row

                # 创建评论字典
                comment_dict = {
                    "comment_id": comment_id,
                    "post_id": comment_post_id,
                    "user_id": comment_user_id,
                    "parent_comment_id": parent_comment_id,
                    "content": comment_content,
                    "created_at": comment_created_at,
                    "num_replies": comment_num_replies,
                    "num_shares": comment_num_shares,
                    "replies": []  # 用于存储回复
                }

                # 根据show_score决定显示分数还是赞踩数量
                if self.show_score:
                    comment_dict["score"] = comment_num_likes - comment_num_dislikes
                else:
                    comment_dict["num_likes"] = comment_num_likes
                    comment_dict["num_dislikes"] = comment_num_dislikes

                # 将评论添加到字典中
                comments_dict[comment_id] = comment_dict

                # 如果是顶级评论，添加到顶级评论列表
                if parent_comment_id is None:
                    top_level_comments.append(comment_dict)

            # 第二遍循环：构建评论树
            for comment_id, comment in comments_dict.items():
                parent_id = comment.get("parent_comment_id")
                if parent_id and parent_id in comments_dict:
                    # 将当前评论添加到父评论的回复列表中
                    comments_dict[parent_id]["replies"].append(comment)

            # 将帖子信息和对应的评论添加到帖子列表中
            post_dict = {}
            # 帖子ID，如果是转发则使用原始帖子ID
            post_dict["post_id"] = post_id if post_type_result["type"] != "repost" else original_post_id
            post_dict["user_id"] = user_id        # 用户ID
            post_dict["content"] = post_content   # 帖子内容
            post_dict["created_at"] = created_at  # 创建时间

            # 与评论类似，根据show_score决定显示方式
            if self.show_score:
                post_dict["score"] = num_likes - num_dislikes
            else:
                post_dict["num_likes"] = num_likes
                post_dict["num_dislikes"] = num_dislikes

            post_dict["num_shares"] = num_shares  # 分享数量
            post_dict["comments"] = top_level_comments  # 只包含顶级评论，回复包含在每个评论的replies字段中

            # 将处理好的帖子字典添加到帖子列表中
            posts.append(post_dict)
        # 返回处理后的帖子列表
        return posts

    def _record_trace(self,
                      user_id,
                      action_type,
                      action_info,
                      current_time=None,
                      step_number=None):
        r"""如果除了跟踪记录外，操作函数还在数据库的其他表中记录时间，
        为了保持一致性，应使用进入操作函数的时间。

        传入current_time参数，例如，可以使post表中的created_at
        与trace表中的时间完全相同。

        如果只有trace表需要记录时间，则使用进入_record_trace函数的时间
        作为跟踪记录的时间。

        参数:
            user_id: 用户ID
            action_type: 操作类型
            action_info: 操作信息
            current_time: 当前时间，如果为None则自动获取
            step_number: 当前轮次，如果为None则从环境变量获取
        """
        # 判断是否存在沙盒时钟
        if self.sandbox_clock:
            # 如果存在沙盒时钟，使用时钟的time_transfer方法转换当前时间
            # datetime.now()获取当前的真实时间
            # self.start_time是平台的起始时间
            # 这里的time_transfer方法可能是将真实时间映射到模拟环境中的时间
            current_time = self.sandbox_clock.time_transfer(
                datetime.now(), self.start_time)
        else:
            # 如果没有沙盒时钟，从环境变量中获取沙盒时间
            # os.environ是一个字典，包含所有环境变量
            # 这里从名为"SANDBOX_TIME"的环境变量中获取时间
            current_time = os.environ["SANDBOX_TIME"]

        # 获取当前轮次信息
        if step_number is None:
            # 如果未提供轮次信息，尝试从环境变量获取
            try:
                step_number = int(os.environ.get("TIME_STAMP", 0))
            except (ValueError, TypeError):
                step_number = 0

        # 构建SQL插入语句，用于向trace表中插入用户行为记录
        # 这是一个多行字符串，使用括号连接
        trace_insert_query = (
            "INSERT INTO trace (user_id, created_at, action, info, step_number) "
            "VALUES (?, ?, ?, ?, ?)")

        # 将action_info（可能是字典或其他复杂数据结构）转换为JSON字符串
        # json.dumps()函数将Python对象转换为JSON字符串格式
        action_info_str = json.dumps(action_info)

        # 执行数据库命令，插入跟踪记录
        # self._execute_db_command是一个辅助方法，用于执行SQL命令
        # 参数包括：SQL查询语句、参数元组、是否提交事务
        # commit=True表示立即提交这个事务到数据库
        self._execute_db_command(
            trace_insert_query,
            (user_id, current_time, action_type, action_info_str, step_number),
            commit=True,
        )

    def _check_self_post_rating(self, post_id, user_id):
        # 定义SQL查询语句，用于从post表中查询指定post_id的帖子作者
        self_like_check_query = "SELECT user_id FROM post WHERE post_id = ?"
        # 执行SQL查询，传入post_id作为参数
        # 这里的问号是SQL参数占位符，(post_id,)是一个单元素元组
        self._execute_db_command(self_like_check_query, (post_id, ))
        # 获取查询结果的第一行（如果有的话）
        result = self.db_cursor.fetchone()
        # 检查结果是否存在且帖子作者是否就是当前用户
        if result and result[0] == user_id:
            # 如果是同一个用户，创建错误信息
            error_message = ("Users are not allowed to like/dislike their own "
                             "posts.")
            # 返回包含错误信息的字典，表示操作失败
            return {"success": False, "error": error_message}
        else:
            # 如果不是同一个用户或帖子不存在，返回None表示检查通过
            return None

    def _check_self_comment_rating(self, comment_id, user_id):
        # 定义SQL查询语句，用于从comment表中查询指定comment_id的评论作者
        self_like_check_query = ("SELECT user_id FROM comment WHERE "
                                 "comment_id = ?")
        # 执行SQL查询，传入comment_id作为参数
        self._execute_db_command(self_like_check_query, (comment_id, ))
        # 获取查询结果的第一行
        result = self.db_cursor.fetchone()
        # 检查结果是否存在且评论作者是否就是当前用户
        if result and result[0] == user_id:
            # 如果是同一个用户，创建错误信息
            error_message = ("Users are not allowed to like/dislike their "
                             "own comments.")
            # 返回包含错误信息的字典，表示操作失败
            return {"success": False, "error": error_message}
        else:
            # 如果不是同一个用户或评论不存在，返回None表示检查通过
            return None

    def _get_post_type(self, post_id: int):
        """获取帖子类型的函数

        这个函数用于确定给定帖子ID的帖子类型（普通帖子、转发帖子或引用帖子）
        并返回相关信息。

        参数:
            post_id: int - 要查询的帖子ID

        返回:
            字典包含帖子类型和原始帖子ID，或者如果帖子不存在则返回None
        """
        # 构建SQL查询语句，从post表中选择original_post_id和quote_content字段
        # 其中post_id等于传入的参数值
        query = (
            "SELECT original_post_id, quote_content FROM post WHERE post_id "
            "= ?")
        # 执行SQL查询，传入post_id作为参数
        # 这里的问号是SQL参数占位符，可以防止SQL注入攻击
        self._execute_db_command(query, (post_id, ))
        # 获取查询结果的第一行（如果有的话）
        result = self.db_cursor.fetchone()

        # 如果没有找到对应的帖子，返回None
        if not result:
            return None

        # 解包查询结果，获取原始帖子ID和引用内容
        # 这是Python的元组解包语法，将result元组的两个元素分别赋值给两个变量
        original_post_id, quote_content = result

        # 根据查询结果判断帖子类型
        if original_post_id is None:
            # 如果原始帖子ID为空，说明这是一个普通帖子（既不是转发也不是引用）
            return {"type": "common", "root_post_id": None}
        elif quote_content is None:
            # 如果原始帖子ID不为空但引用内容为空，说明这是一个转发帖子
            return {"type": "repost", "root_post_id": original_post_id}
        else:
            # 如果原始帖子ID和引用内容都不为空，说明这是一个引用帖子
            return {"type": "quote", "root_post_id": original_post_id}
