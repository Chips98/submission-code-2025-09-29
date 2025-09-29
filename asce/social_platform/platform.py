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

import asyncio
import logging
import os
import random
import sqlite3
import sys
from datetime import datetime, timedelta
from typing import Any
import json
import hashlib

from asce.clock.clock import Clock
# 从数据库模块导入函数：
# - create_db：创建数据库连接和游标
# - fetch_rec_table_as_matrix：将推荐表数据以矩阵形式获取
# - fetch_table_from_db：从数据库获取表格数据
from asce.social_platform.database import (create_db,
                                            fetch_rec_table_as_matrix,
                                            fetch_table_from_db)
# 导入平台工具类，提供各种平台操作的辅助功能
from asce.social_platform.platform_utils import PlatformUtils
# 导入各种推荐系统算法：
# - rec_sys_personalized_twh：个性化推荐系统（带时间权重）
# - rec_sys_personalized_with_trace：带用户行为追踪的个性化推荐
# - rec_sys_random：随机推荐系统
# - rec_sys_reddit：类Reddit风格的推荐系统
from asce.social_platform.recsys import (rec_sys_personalized_twh,
                                          rec_sys_personalized_with_trace,
                                          rec_sys_random, rec_sys_reddit,
                                          rec_sys_cognitive)
# 导入操作类型和推荐系统类型的枚举定义
from asce.social_platform.typing import ActionType, RecsysType

# 检查是否在Sphinx文档生成环境中运行
# sys.modules是一个字典，包含所有已导入的模块
# 这个条件判断的意思是：如果不是在Sphinx文档环境中运行，则执行下面的日志配置
if "sphinx" not in sys.modules:
    # 创建一个名为"social.twitter"的日志记录器
    twitter_log = logging.getLogger(name="social.twitter")
    # 设置日志级别为DEBUG（最详细的日志级别，会记录所有信息）
    twitter_log.setLevel("DEBUG")
    # 获取当前时间并格式化为字符串，用于日志文件名
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 创建一个文件处理器，将日志写入到特定文件中
    # f"./log/social.twitter-{now}.log" 是一个f-string（格式化字符串），会把变量now的值插入到字符串中
    file_handler = logging.FileHandler(f"./log/social.twitter-{now}.log")
    # 设置文件处理器的日志级别为DEBUG
    file_handler.setLevel("DEBUG")
    # 设置日志格式，包含日志级别、时间、日志名称和消息内容
    file_handler.setFormatter(
        logging.Formatter(
            "%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
    # 将文件处理器添加到日志记录器，这样日志就会被写入到文件中
    twitter_log.addHandler(file_handler)



class Platform:
    r"""平台类。"""

    def __init__(
        self,
        db_path: str,                          # 数据库路径
        channel: Any,                          # 通信通道
        sandbox_clock: Clock | None = None,    # 沙盒时钟，默认为None
        start_time: datetime | None = None,    # 开始时间，默认为None
        show_score: bool = False,              # 是否显示分数，默认为False
        allow_self_rating: bool = True,        # 是否允许自评，默认为True
        recsys_type: str | RecsysType = "reddit",  # 推荐系统类型，默认为"reddit"，其他选项包括：
                                                   # - "personalized_twh"：带时间权重的个性化推荐系统
                                                   # - "personalized_with_trace"：带用户行为追踪的个性化推荐系统
                                                   # - "random"：随机推荐系统
        refresh_rec_post_count: int = 1,       # 每次刷新推荐的帖子数量，默认为1
        max_rec_post_len: int = 2,             # 推荐表中每个用户的最大帖子数，默认为2
        following_post_count=3,                # 关注用户的帖子数量，默认为3
        activate_prob: float = 0.1,            # 用户激活概率，默认为0.1
        max_visible_comments: int = 5,         # 每个帖子最多显示的评论数量，默认为5
        max_total_comments: int = 10,          # 用户上下文中显示的最大评论总数，默认为10
    ):
        # 初始化函数，设置平台的各种参数和状态

        self.db_path = db_path                 # 存储数据库路径
        self.recsys_type = recsys_type         # 存储推荐系统类型
        self.activate_prob = activate_prob     # 存储用户激活概率
        self.max_visible_comments = max_visible_comments  # 存储每个帖子最多显示的评论数量
        self.max_total_comments = max_total_comments  # 存储用户上下文中显示的最大评论总数
        # import pdb; pdb.set_trace()          # 调试代码（已注释）

        # 添加当前时间步属性和第一轮标记
        self.timestep = 0                      # 当前时间步，初始为0
        self.first_round = True                # 标记是否是第一轮模拟

        # 初始化用户认知信息字典
        self.users_cognitive_profile_dict = {} # 存储用户认知信息，用于认知推荐系统

        # 如果推荐系统类型是"reddit"
        if self.recsys_type in ["reddit", "cognitive"]:
            # 如果没有指定时钟，默认平台的时间放大因子为60
            if sandbox_clock is None:
                sandbox_clock = Clock(60)      # 创建一个放大因子为60的时钟
            if start_time is None:
                start_time = datetime.now()    # 如果没有指定开始时间，使用当前时间
            self.start_time = start_time       # 设置开始时间
            self.sandbox_clock = sandbox_clock # 设置沙盒时钟
        else:
            # 如果不是"reddit"类型，则设置不同的默认值
            self.start_time = 0                # 开始时间设为0
            self.sandbox_clock = None          # 沙盒时钟设为None

        # 创建数据库连接和游标
        self.db, self.db_cursor = create_db(self.db_path)

        # 优化SQLite配置以提高写入性能
        # 使用WAL模式，提供更好的并发性能
        self.db.execute("PRAGMA journal_mode = WAL")
        # 关闭同步模式以提高性能（最快但最不安全的设置）
        self.db.execute("PRAGMA synchronous = OFF")
        # 增加缓存大小以减少磁盘访问
        self.db.execute("PRAGMA cache_size = -50000")  # 约50MB缓存
        # 设置较大的页面大小
        self.db.execute("PRAGMA page_size = 4096")
        # 启用内存映射，减少系统调用
        self.db.execute("PRAGMA mmap_size = 30000000000")

        # 存储通信通道
        self.channel = channel

        # 将平台实例设置到通道中，以便通道可以调用平台方法
        channel.set_platform(self)

        # 将推荐系统类型转换为枚举类型
        self.recsys_type = RecsysType(recsys_type)

        # 是否像Reddit那样显示分数（赞减踩）
        # 而不是分别显示赞和踩的数量
        self.show_score = show_score

        # 是否允许用户对自己的帖子和评论进行点赞或点踩
        self.allow_self_rating = allow_self_rating

        # 社交媒体内部推荐系统每次刷新返回的帖子数量
        self.refresh_rec_post_count = refresh_rec_post_count

        # 一次返回的关注用户发布的帖子数量，按点赞数排序
        self.following_post_count = following_post_count

        # 推荐表（缓冲区）中每个用户的最大帖子数
        self.max_rec_post_len = max_rec_post_len

        # 随机推荐和个性化推荐之间的概率比例
        self.rec_prob = 0.7

        # 平台内部趋势规则的参数
        self.trend_num_days = 7               # 趋势计算的天数
        self.trend_top_k = 1                  # 趋势取前k个

        # 创建平台工具类实例，用于辅助数据库操作和其他功能
        self.pl_utils = PlatformUtils(
            self.db,                          # 数据库连接
            self.db_cursor,                   # 数据库游标
            self.start_time,                  # 开始时间
            self.sandbox_clock,               # 沙盒时钟
            self.show_score,                  # 是否显示分数
        )

        # 将最大可见评论数设置到平台工具类中
        self.pl_utils.max_visible_comments = self.max_visible_comments
        # 将用户上下文中显示的最大评论总数设置到平台工具类中
        self.pl_utils.max_total_comments = self.max_total_comments

        # 初始化完成后，检查并创建额外的表
        self._init_additional_tables()

    def _init_additional_tables(self):
        """初始化额外的数据库表，如user_information和user_action"""
        try:
            # 创建平台日志记录器
            platform_log = logging.getLogger(name="social.platform")

            platform_log.info("检查并创建额外的数据库表")

            # 检查user_information表是否存在，如果不存在则创建
            self.db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_information'")
            if not self.db_cursor.fetchone():
                platform_log.info("正在创建user_information表...")
                # 读取并执行SQL文件
                try:
                    sql_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema", "user_information.sql")
                    with open(sql_file_path, "r") as sql_file:
                        sql_script = sql_file.read()
                        self.db.executescript(sql_script)
                    platform_log.info("成功创建user_information表")
                except Exception as e:
                    platform_log.error(f"创建user_information表失败: {str(e)}")
            else:
                platform_log.info("user_information表已存在")

            # 检查user_action表是否存在，如果不存在则创建
            self.db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_action'")
            if not self.db_cursor.fetchone():
                platform_log.info("正在创建user_action表...")
                # 读取并执行SQL文件
                try:
                    sql_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema", "user_action.sql")
                    with open(sql_file_path, "r") as sql_file:
                        sql_script = sql_file.read()
                        self.db.executescript(sql_script)
                    platform_log.info("成功创建user_action表")
                except Exception as e:
                    platform_log.error(f"创建user_action表失败: {str(e)}")
            else:
                platform_log.info("user_action表已存在")

        except Exception as e:
            platform_log.error(f"初始化额外表时出错: {str(e)}")
            import traceback
            platform_log.error(f"错误详情: {traceback.format_exc()}")

    async def running(self):
        # 平台主运行循环函数，使用异步方式处理消息
        while True:
            # 异步等待从通道接收消息，返回消息ID和数据
            message_id, data = await self.channel.receive_from()

            # 解析接收到的数据，包含代理ID、消息内容和动作类型
            agent_id, message, action = data
            # 将动作字符串转换为ActionType枚举类型，
            # 如果action不是ActionType枚举类型中的一个，会抛出ValueError异常
            action = ActionType(action)

            # 如果收到退出动作，则关闭数据库连接并退出循环
            if action == ActionType.EXIT:
                # 先将缓存中的数据写入数据库
                await self._flush_all_caches()

                # 如果数据库是内存数据库，在退出前将其保存到文件
                # ":memory:"是SQLite的特殊标识符，表示数据库在内存中而非磁盘上
                if self.db_path == ":memory:":
                    # 创建一个新的数据库连接到mock.db文件
                    dst = sqlite3.connect("mock.db")
                    # 使用with语句确保即使发生异常也能正确关闭连接
                    with dst:
                        # 将内存数据库备份到文件数据库
                        self.db.backup(dst)

                # 关闭数据库游标和连接
                self.db_cursor.close()
                self.db.close()
                # 跳出循环，结束运行
                break

            # 使用getattr函数根据动作类型获取对应的处理函数
            # getattr(对象, 属性名, 默认值)：获取对象的属性，如果不存在则返回默认值
            # action.value是枚举值，对应Platform类中的方法名
            action_function = getattr(self, action.value, None)
            if action_function:
                # 如果找到对应的处理函数，获取函数参数信息
                # __code__是函数的代码对象，包含函数的各种元数据
                func_code = action_function.__code__
                # co_varnames包含函数的局部变量名，前co_argcount个是参数名
                # 这里获取函数的所有参数名
                param_names = func_code.co_varnames[:func_code.co_argcount]

                # 获取参数数量
                len_param_names = len(param_names)
                # 如果参数超过3个，可能需要特殊处理
                # 构建参数字典，用于后续调用函数
                params = {}
                # 如果函数至少有2个参数，将agent_id作为第一个参数
                if len_param_names >= 2:
                    params["agent_id"] = agent_id

                # 处理特殊情况：关注操作可能带有step_number参数
                if action.value == "follow" and isinstance(message, tuple) and len(message) == 2:
                    # 如果message是一个包含两个元素的元组，表示是(followee_id, step_number)格式
                    followee_id, step_number = message
                    params["followee_id"] = followee_id
                    params["step_number"] = step_number
                # 如果函数有更多参数但不是特殊处理的情况
                elif len_param_names >= 3:
                    # 获取第三个参数的名称
                    # param_names[2]表示参数列表中的第三个参数名
                    second_param_name = param_names[2]
                    # 将消息内容赋值给对应名称的参数
                    params[second_param_name] = message

                # 使用**params语法展开字典作为关键字参数调用函数
                # await表示这是异步调用，等待函数执行完成
                result = await action_function(**params)
                # 将处理结果通过通道发送回去
                # 发送的是一个元组，包含消息ID、代理ID和处理结果
                await self.channel.send_to((message_id, agent_id, result))
            else:
                # 如果找不到对应的处理函数，抛出异常
                raise ValueError(f"Action {action} is not supported")

    def run(self):
        """运行平台的主函数"""
        # 这个方法是平台的入口点，用于启动整个平台的运行
        # asyncio.run() 是Python的异步IO库中的函数，用于运行异步函数并等待其完成
        # 它接收一个协程对象(coroutine)作为参数，并负责创建事件循环、运行协程、关闭事件循环
        # self.running() 是一个异步方法(使用async定义的方法)，包含了平台的主要运行逻辑
        # 这里的写法相当于创建一个新的事件循环，运行self.running()协程直到完成，然后关闭事件循环
        # 这是启动异步程序的标准方式，使平台能够高效处理并发操作
        asyncio.run(self.running())

    async def sign_up(self, agent_id, user_message):
        # 用户注册函数，使用异步方式定义，接收代理ID和用户消息作为参数
        # async def 定义一个异步函数，可以在函数内使用await关键字

        # 从用户消息中解包三个变量：用户名、姓名和个人简介
        # 这里使用了Python的元组解包语法，假设user_message是一个包含三个元素的元组或列表
        user_name, name, bio = user_message

        # 根据推荐系统类型确定当前时间
        # 条件判断：如果推荐系统类型是Reddit
        if self.recsys_type in [RecsysType.REDDIT, RecsysType.COGNITIVE]:
            # 使用沙盒时钟将当前时间转换为相对于开始时间的时间
            # datetime.now()获取当前系统时间，然后通过time_transfer方法转换
            current_time = self.sandbox_clock.time_transfer(
                datetime.now(), self.start_time)
        else:
            # 如果不是Reddit类型，则从环境变量中获取沙盒时间
            # os.environ是一个字典，包含所有环境变量
            current_time = os.environ["SANDBOX_TIME"]

        # 使用try-except结构捕获可能发生的异常
        try:
            # 构建SQL插入语句，用于向user表中添加新用户
            # 问号(?)是SQL参数占位符，防止SQL注入攻击
            user_insert_query = (
                "INSERT INTO user (user_id, agent_id, user_name, name, bio, "
                "created_at, num_followings, num_followers) VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?)")

            # 执行SQL命令，插入新用户数据
            # _execute_db_command是一个辅助方法，用于执行SQL命令
            # commit=True表示立即提交事务，将更改永久保存到数据库
            self.pl_utils._execute_db_command(
                user_insert_query,
                (agent_id, agent_id, user_name, name, bio, current_time, 0, 0),
                commit=True,
            )

            # 将agent_id赋值给user_id变量（在这个上下文中它们是相同的）
            user_id = agent_id

            # 创建一个包含用户注册信息的字典，用于记录操作
            # 这是Python字典的字面量语法，创建键值对集合
            action_info = {"name": name, "user_name": user_name, "bio": bio}

            # 记录用户注册操作到跟踪表中
            # ActionType.SIGNUP.value获取枚举值SIGNUP的字符串表示
            self.pl_utils._record_trace(user_id, ActionType.SIGNUP.value,
                                        action_info, current_time)

            # 使用日志记录器记录跟踪信息
            # f-string是Python 3.6+的格式化字符串语法，可以在字符串中嵌入变量
            # 格式为f"文本{变量}更多文本"
            twitter_log.info(f"Trace inserted: user_id={user_id}, "
                             f"current_time={current_time}, "
                             f"action={ActionType.SIGNUP.value}, "
                             f"info={action_info}")

            # 返回成功结果和用户ID
            # 返回一个字典，表示操作成功并包含用户ID
            return {"success": True, "user_id": user_id}

        # 捕获任何可能发生的异常
        except Exception as e:
            # 返回失败结果和错误信息
            # str(e)将异常对象转换为字符串，获取错误消息
            return {"success": False, "error": str(e)}

    async def sign_up_product(self, product_id: int, product_name: str):
        """注册产品函数

        这个函数用于在系统中注册新产品，将产品信息添加到数据库中。

        参数:
            product_id: int - 产品ID，用于唯一标识产品
            product_name: str - 产品名称

        返回:
            字典 - 包含操作结果的信息
        """
        # 注意：不要使用相同的产品名称注册产品
        try:
            # 构建SQL插入语句，用于向product表中添加新产品
            # 这是一个字符串，定义了SQL命令的结构
            # (?, ?) 是参数占位符，防止SQL注入攻击
            product_insert_query = (
                "INSERT INTO product (product_id, product_name) VALUES (?, ?)")

            # 执行SQL命令，插入新产品数据
            # self.pl_utils._execute_db_command是一个辅助方法，用于执行SQL命令
            # 参数包括：SQL查询语句、参数元组、是否提交事务
            # (product_id, product_name)是一个元组，包含要插入的值
            # commit=True表示立即提交事务，将更改永久保存到数据库
            self.pl_utils._execute_db_command(product_insert_query,
                                              (product_id, product_name),
                                              commit=True)

            # 操作成功，返回成功状态和产品ID
            # 返回一个字典，包含两个键值对：success表示操作是否成功，product_id是注册的产品ID
            return {"success": True, "product_id": product_id}

        # 使用try-except结构捕获可能发生的任何异常
        except Exception as e:
            # 如果发生异常，返回失败状态和错误信息
            # Exception as e 捕获所有类型的异常，并将异常对象赋值给变量e
            # str(e)将异常对象转换为字符串，获取错误消息
            return {"success": False, "error": str(e)}

    async def purchase_product(self, agent_id, purchase_message):
        """购买产品函数

        这个函数用于处理用户购买产品的请求，更新产品销售数量并记录购买行为。

        参数:
            agent_id: int - 代理ID，用于标识购买者
            purchase_message: tuple - 包含产品名称和购买数量的元组

        返回:
            字典 - 包含操作结果的信息
        """
        # 从purchase_message元组中解包出产品名称和购买数量
        # 这是Python的元组解包语法，将元组的两个元素分别赋值给两个变量
        product_name, purchase_num = purchase_message

        # 根据推荐系统类型确定当前时间
        # 如果是Reddit类型的推荐系统，使用沙盒时钟转换时间
        if self.recsys_type in [RecsysType.REDDIT, RecsysType.COGNITIVE]:
            # 使用沙盒时钟将真实时间转换为模拟环境中的时间
            # datetime.now()获取当前的真实时间
            # self.start_time是平台的起始时间
            current_time = self.sandbox_clock.time_transfer(
                datetime.now(), self.start_time)
        else:
            # 如果不是Reddit类型，从环境变量中获取沙盒时间
            # os.environ是一个字典，包含所有环境变量
            current_time = os.environ["SANDBOX_TIME"]

        # try:  # 注释掉的异常处理开始部分

        # 将agent_id赋值给user_id变量
        user_id = agent_id

        # 检查产品是否存在
        # 构建SQL查询语句，从product表中查询指定产品名称的记录
        product_check_query = (
            "SELECT * FROM 'product' WHERE product_name = ?")

        # 执行SQL查询，传入产品名称作为参数
        # 这里的问号是SQL参数占位符，(product_name,)是一个单元素元组
        self.pl_utils._execute_db_command(product_check_query,
                                          (product_name, ))

        # 获取查询结果的第一行
        # fetchone()方法返回查询结果的第一行，如果没有结果则返回None
        check_result = self.db_cursor.fetchone()

        # 如果没有找到产品记录
        if not check_result:
            # 返回失败信息，表示没有找到该产品
            return {"success": False, "error": "No such product."}
        else:
            # 如果找到产品记录，获取产品ID（结果的第一列）
            product_id = check_result[0]

        # 构建SQL更新语句，增加产品的销售数量
        # SET sales = sales + ? 表示将当前销售量加上购买数量
        product_update_query = (
            "UPDATE product SET sales = sales + ? WHERE product_name = ?")

        # 执行SQL更新，传入购买数量和产品名称作为参数
        # commit=True表示立即提交事务，将更改永久保存到数据库
        self.pl_utils._execute_db_command(product_update_query,
                                          (purchase_num, product_name),
                                          commit=True)

        # 准备记录购买行为的信息
        # 创建一个字典，包含产品名称和购买数量
        action_info = {
            "product_name": product_name,
            "purchase_num": purchase_num
        }

        # 调用_record_trace方法记录用户的购买行为
        # ActionType.PURCHASE_PRODUCT.value是一个枚举值，表示购买产品的行为类型
        # action_info是包含详细信息的字典
        # current_time是操作发生的时间
        self.pl_utils._record_trace(user_id, ActionType.PURCHASE_PRODUCT.value,
                                    action_info, current_time)

        # 返回成功信息和产品ID
        return {"success": True, "product_id": product_id}

        # except Exception as e:  # 注释掉的异常处理结束部分
        #     return {"success": False, "error": str(e)}

    async def refresh(self, agent_id: int):
        """刷新用户的推荐内容，获取推荐帖子和关注用户的帖子"""
        # 根据特定ID从推荐表中检索帖子
        if self.recsys_type in [RecsysType.REDDIT, RecsysType.COGNITIVE]:
            # 如果推荐系统类型是Reddit，使用沙盒时钟转换当前时间
            current_time = self.sandbox_clock.time_transfer(
                datetime.now(), self.start_time)
        else:
            # 否则，从环境变量中获取沙盒时间
            current_time = os.environ["SANDBOX_TIME"]
        try:
            # 将agent_id赋值给user_id变量
            user_id = agent_id
            # 获取当前轮次信息
            try:
                current_step = int(os.environ.get("TIME_STAMP", 0))
            except (ValueError, TypeError):
                current_step = 0

            # 从rec表中检索给定user_id和当前轮次的所有post_id
            # 构建SQL查询语句，查询推荐表中特定用户和轮次的所有帖子ID
            rec_query = "SELECT post_id FROM rec WHERE user_id = ? AND step_number = ?"
            # 执行SQL查询，传入用户ID和当前轮次作为参数
            self.pl_utils._execute_db_command(rec_query, (user_id, current_step))
            # 获取所有查询结果
            rec_results = self.db_cursor.fetchall()

            # 从查询结果中提取所有帖子ID，形成一个列表
            # row[0]表示每行结果的第一个元素，即post_id
            post_ids = [row[0] for row in rec_results]
            # 初始化选定的帖子ID列表
            selected_post_ids = post_ids
            # 如果帖子ID数量大于等于刷新推荐帖子数量，随机选择指定数量的帖子ID
            if len(selected_post_ids) >= self.refresh_rec_post_count:
                # random.sample函数用于从列表中随机选择指定数量的元素，不重复
                selected_post_ids = random.sample(selected_post_ids,
                                                  self.refresh_rec_post_count)

            # 如果推荐系统类型不是Reddit
            if self.recsys_type != RecsysType.REDDIT:
                # 检索来自关注用户的帖子（网络内）
                # 修改SQL查询，使刷新获取用户关注的人的帖子，按Twitter上的点赞数排序
                query_following_post = (
                    "SELECT post.post_id, post.user_id, post.content, "
                    "post.created_at, post.num_likes FROM post "
                    "JOIN follow ON post.user_id = follow.followee_id "
                    "WHERE follow.follower_id = ? "
                    "ORDER BY post.num_likes DESC  "
                    "LIMIT ?")
                # 执行SQL查询，获取用户关注的人的帖子
                # 参数包括用户ID和要获取的关注帖子数量
                self.pl_utils._execute_db_command(
                    query_following_post,
                    (
                        user_id,
                        self.following_post_count,
                    ),
                )

                # 获取查询结果，即关注用户的帖子
                following_posts = self.db_cursor.fetchall()
                # 从结果中提取帖子ID列表
                following_posts_ids = [row[0] for row in following_posts]

                # 将关注用户的帖子ID添加到选定的帖子ID列表中
                selected_post_ids = following_posts_ids + selected_post_ids
                # 使用set()去除可能的重复帖子ID，然后转回列表
                selected_post_ids = list(set(selected_post_ids))

            # 为SQL查询创建占位符字符串，每个帖子ID对应一个"?"
            # 例如，如果有3个帖子ID，则生成"?, ?, ?"
            placeholders = ", ".join("?" for _ in selected_post_ids)

            # 构建SQL查询，获取选定帖子ID的详细信息
            post_query = (
                f"SELECT post_id, user_id, original_post_id, content, "
                f"quote_content, created_at, num_likes, num_dislikes, "
                f"num_shares FROM post WHERE post_id IN ({placeholders})")
            # 执行SQL查询，传入所有选定的帖子ID作为参数
            self.pl_utils._execute_db_command(post_query, selected_post_ids)
            # 获取查询结果，即所有选定帖子的详细信息
            results = self.db_cursor.fetchall()
            # 如果没有找到帖子，返回失败信息
            if not results:
                return {"success": False, "message": "No posts found."}
            # 为每个帖子添加评论信息
            results_with_comments = self.pl_utils._add_comments_to_posts(
                results)

            # 准备记录刷新行为的信息
            action_info = {"posts": results_with_comments}
            # 记录日志信息
            twitter_log.info(action_info)
            # 调用_record_trace方法记录用户的刷新行为
            self.pl_utils._record_trace(user_id, ActionType.REFRESH.value,
                                        action_info, current_time)

            # 返回成功信息和带有评论的帖子列表
            return {"success": True, "posts": results_with_comments}
        except Exception as e:
            # 如果发生异常，返回失败信息和错误详情
            return {"success": False, "error": str(e)}

    async def update_rec_table(self):
        platform_log = logging.getLogger(name="social.platform")

        """更新推荐表格的异步方法"""
        platform_log.info("\n===== 开始更新推荐表 =====")
        platform_log.info(f"[DEBUG-平台] 当前推荐系统类型: {self.recsys_type}")

        # 从数据库获取用户表、帖子表和跟踪表
        user_table = fetch_table_from_db(self.db_cursor, "user")  # 获取用户表数据
        post_table = fetch_table_from_db(self.db_cursor, "post")  # 获取帖子表数据
        trace_table = fetch_table_from_db(self.db_cursor, "trace")  # 获取用户行为跟踪表数据
        rec_matrix = fetch_rec_table_as_matrix(self.db_cursor)  # 获取当前推荐表并转换为矩阵形式
        platform_log = logging.getLogger(name="social.platform")

        platform_log.info(f"[DEBUG-平台] 从数据库获取的数据 - 用户: {len(user_table)}人, 帖子: {len(post_table)}条, 当前推荐矩阵尺寸: {len(rec_matrix)}x{len(rec_matrix[0]) if rec_matrix and rec_matrix[0] else 0}")

        # 更新时间步
        self.timestep += 1
        platform_log.info(f"当前时间步: {self.timestep}")
        platform_log.info(f"[DEBUG-平台] 当前时间步: {self.timestep}")

        # 根据不同的推荐系统类型选择不同的推荐算法
        if self.recsys_type == RecsysType.RANDOM:
            # 使用随机推荐算法生成新的推荐矩阵
            platform_log.info("[DEBUG-平台] 使用随机推荐算法")
            new_rec_matrix = rec_sys_random(post_table, rec_matrix,
                                            self.max_rec_post_len)
        elif self.recsys_type == RecsysType.TWITTER:
            # 使用类似Twitter的个性化推荐算法，考虑用户行为
            platform_log.info("[DEBUG-平台] 使用Twitter个性化推荐算法")
            new_rec_matrix = rec_sys_personalized_with_trace(
                user_table, post_table, trace_table, rec_matrix,
                self.max_rec_post_len)
        elif self.recsys_type == RecsysType.TWHIN:
            # 使用TWHIN模型的推荐算法，需要获取最新帖子进行增量更新
            platform_log.info("[DEBUG-平台] 使用TWHIN模型推荐算法")
            latest_post_time = post_table[-1]["created_at"]  # 获取最新帖子的创建时间
            post_query = "SELECT COUNT(*) " "FROM post " "WHERE created_at = ?"  # SQL查询语句，计算特定时间创建的帖子数量

            # 执行SQL查询，获取最新时间点创建的帖子数量
            self.pl_utils._execute_db_command(post_query, (latest_post_time, ))
            result = self.db_cursor.fetchone()  # 获取查询结果的第一行
            latest_post_count = result[0]  # 提取结果中的计数值
            platform_log.info(f"[DEBUG-平台] 最新帖子时间: {latest_post_time}, 帖子数量: {latest_post_count}")

            # 如果没有获取到最新帖子数量，返回失败信息
            if not latest_post_count:
                platform_log.error("[DEBUG-平台错误] 无法获取最新帖子数量")
                return {
                    "success": False,
                    "message": "Fail to get latest posts count"
                }

            # 使用TWHIN个性化推荐算法生成新的推荐矩阵
            new_rec_matrix = rec_sys_personalized_twh(
                user_table,
                post_table,
                latest_post_count,
                trace_table,
                rec_matrix,
                self.max_rec_post_len,
            )
        elif self.recsys_type == RecsysType.REDDIT:
            # 使用类似Reddit的推荐算法生成新的推荐矩阵
            platform_log.info("[DEBUG-平台] 使用Reddit热度推荐算法")
            new_rec_matrix = rec_sys_reddit(post_table, rec_matrix,
                                            self.max_rec_post_len)
        elif self.recsys_type == RecsysType.COGNITIVE:
            # 使用基于认知链的推荐算法
            platform_log.info("[DEBUG-平台] 开始认知推荐算法流程")
            try:
                # 检查是否是第一轮模拟
                platform_log.info(f"[DEBUG-认知推荐] 检查是否是第一轮模拟: {self.first_round}")
                if self.first_round:
                    platform_log.warning("认知推荐模式：检测到第一轮模拟，使用随机推荐")
                    platform_log.info("[DEBUG-认知推荐] 第一轮模拟，使用随机推荐")
                    # 第一轮使用随机推荐，以避免数据库尚未初始化的问题
                    new_rec_matrix = rec_sys_random(post_table, rec_matrix,
                                              self.max_rec_post_len)
                    platform_log.info(f"[DEBUG-认知推荐] 随机推荐结果大小: {len(new_rec_matrix)}x{len(new_rec_matrix[0]) if new_rec_matrix and new_rec_matrix[0] else 0}")
                    # 更新标记，下一轮将使用认知推荐
                    self.first_round = False
                    platform_log.info("[DEBUG-认知推荐] 已更新first_round标记为False，下一轮将使用认知推荐")
                else:
                    # 从第二轮开始使用认知推荐
                    platform_log.info("[DEBUG-认知推荐] 非首轮，开始执行认知推荐系统")
                    platform_log.info(f"[DEBUG-认知推荐] 传递给认知推荐系统的参数 - post_table: {len(post_table)}条, rec_matrix: {len(rec_matrix)}行, max_rec_post_len: {self.max_rec_post_len}, platform实例已传递")

                    # 检查数据库表状态
                    platform_log.info("[DEBUG-认知推荐] 检查数据库表状态")
                    self.pl_utils._execute_db_command("SELECT name FROM sqlite_master WHERE type='table' AND name='user_action'")
                    has_user_action = self.pl_utils.db_cursor.fetchone() is not None
                    platform_log.info(f"[DEBUG-认知推荐] user_action表存在: {has_user_action}")

                    # 如果存在user_action表，检查数据
                    if has_user_action:
                        self.pl_utils._execute_db_command("SELECT COUNT(*) FROM user_action")
                        user_action_count = self.pl_utils.db_cursor.fetchone()[0]
                        platform_log.info(f"[DEBUG-认知推荐] user_action表中有 {user_action_count} 条记录")

                    # 调用认知推荐系统
                    new_rec_matrix = rec_sys_cognitive(post_table, rec_matrix,
                                                  self.max_rec_post_len, self)
                    platform_log.info(f"[DEBUG-认知推荐] 认知推荐系统返回结果大小: {len(new_rec_matrix)}x{len(new_rec_matrix[0]) if new_rec_matrix and new_rec_matrix[0] else 0}")
            except Exception as e:
                # 如果认知推荐出错，回退到随机推荐
                platform_log.error(f"认知推荐系统出错，回退到随机推荐: {e}")
                import traceback
                error_traceback = traceback.format_exc()
                platform_log.error(error_traceback)
                platform_log.error("[DEBUG-认知推荐错误恢复] 回退到随机推荐")
                new_rec_matrix = rec_sys_random(post_table, rec_matrix,
                                          self.max_rec_post_len)
                platform_log.info(f"[DEBUG-认知推荐] 随机推荐回退结果大小: {len(new_rec_matrix)}x{len(new_rec_matrix[0]) if new_rec_matrix and new_rec_matrix[0] else 0}")
        else:
            # 如果推荐系统类型不支持，抛出ValueError异常
            error_msg = f"不支持的推荐系统类型: {self.recsys_type}, 请检查 RecsysType。"
            platform_log.error(f"[DEBUG-平台错误] {error_msg}")
            raise ValueError(error_msg)

        platform_log.info(f"[DEBUG-平台] 推荐矩阵生成完成，准备更新数据库")
        platform_log.info(f"[DEBUG-平台] 新推荐矩阵大小: {len(new_rec_matrix)}x{len(new_rec_matrix[0]) if new_rec_matrix and new_rec_matrix[0] else 0}")

        # 获取当前轮次信息
        try:
            current_step = int(os.environ.get("TIME_STAMP", 0))
        except (ValueError, TypeError):
            current_step = 0

        # 清空当前轮次的推荐表记录（保留其他轮次的记录）
        sql_query = "DELETE FROM rec WHERE step_number = ?"
        # 执行SQL语句，删除rec表中当前轮次的记录，并提交事务
        self.pl_utils._execute_db_command(sql_query, (current_step,), commit=True)
        platform_log.info(f"[DEBUG-平台] 已清空轮次 {current_step} 的推荐表记录")

        # 在每轮模拟结束时，将缓存中的数据写入数据库
        await self._flush_all_caches()
        platform_log.info("[DEBUG-平台] 已将缓存数据写入数据库")

        # 获取当前轮次信息
        try:
            current_step = int(os.environ.get("TIME_STAMP", 0))
        except (ValueError, TypeError):
            current_step = 0

        platform_log.info(f"[DEBUG-平台] 当前轮次: {current_step}")

        # 批量插入更高效
        # 创建要插入的值列表
        # 这是一个列表推导式，遍历新推荐矩阵中的每个用户和他们的推荐帖子
        # 对于每个用户ID和推荐给该用户的每个帖子ID，创建一个元组(user_id, post_id, step_number)
        insert_values = [(user_id, post_id, current_step)
                         for user_id in range(len(new_rec_matrix))  # 遍历所有用户ID
                         for post_id in new_rec_matrix[user_id]]  # 遍历该用户的所有推荐帖子ID

        platform_log.info(f"[DEBUG-平台] 准备插入 {len(insert_values)} 条推荐记录")

        # 执行批量插入操作
        # 使用_execute_many_db_command方法一次性插入多条记录
        # 第一个参数是SQL插入语句
        # 第二个参数是要插入的值列表
        # commit=True表示执行后立即提交事务
        self.pl_utils._execute_many_db_command(
            "INSERT INTO rec (user_id, post_id, step_number) VALUES (?, ?, ?)",  # SQL插入语句，使用问号作为参数占位符
            insert_values,  # 要插入的值列表
            commit=True,  # 执行后立即提交事务
        )
        platform_log.info("[DEBUG-平台] 推荐表更新完成")
        platform_log.info("===== 推荐表更新结束 =====\n")

    async def create_post(self, agent_id: int, content: str):
        """创建帖子的异步方法，允许用户发布新内容

        参数:
        agent_id: int - 发帖用户的ID
        content: str - 帖子内容

        返回:
        字典 - 包含操作成功状态和帖子ID或错误信息
        """
        # 根据推荐系统类型确定当前时间
        if self.recsys_type in [RecsysType.REDDIT, RecsysType.COGNITIVE]:
            # 如果是Reddit类型的推荐系统，使用沙盒时钟转换时间
            # 将当前真实时间(datetime.now())转换为沙盒内的模拟时间
            current_time = self.sandbox_clock.time_transfer(
                datetime.now(), self.start_time)
        else:
            # 否则，从环境变量中获取沙盒时间
            current_time = os.environ["SANDBOX_TIME"]

        try:
            # 将agent_id赋值给user_id（在这个函数中它们是相同的）
            user_id = agent_id

            # 获取当前轮次信息
            try:
                current_step = int(os.environ.get("TIME_STAMP", 0))
            except (ValueError, TypeError):
                current_step = 0

            # 构建SQL插入语句，用于创建新帖子
            # 这是一个多行字符串，使用括号连接各行
            # 问号(?)是SQL参数占位符，防止SQL注入攻击
            post_insert_query = (
                "INSERT INTO post (user_id, content, created_at, step_number, num_likes, "
                "num_dislikes, num_shares) VALUES (?, ?, ?, ?, ?, ?, ?)")

            # 执行SQL命令，插入新帖子记录
            # 参数依次是：用户ID、帖子内容、创建时间、轮次、点赞数(0)、踩数(0)、分享数(0)
            # commit=True表示立即提交事务到数据库
            self.pl_utils._execute_db_command(
                post_insert_query, (user_id, content, current_time, current_step, 0, 0, 0),
                commit=True)

            # 获取刚插入记录的ID（即新帖子的ID）
            # lastrowid是cursor对象的属性，返回最后插入行的ID
            post_id = self.db_cursor.lastrowid

            # 创建操作信息字典，包含帖子内容和ID
            # 这将用于记录用户行为跟踪
            action_info = {"content": content, "post_id": post_id}

            # 记录用户创建帖子的行为
            # ActionType.CREATE_POST.value获取枚举值对应的字符串
            self.pl_utils._record_trace(user_id, ActionType.CREATE_POST.value,
                                        action_info, current_time)

            # 记录日志，使用f-string格式化字符串
            # f-string是Python 3.6+的特性，允许在字符串中嵌入表达式
            # 格式为f"文本{表达式}"，表达式会被求值并转换为字符串
            twitter_log.info(f"Trace inserted: user_id={user_id}, "
                             f"current_time={current_time}, "
                             f"action={ActionType.CREATE_POST.value}, "
                             f"info={action_info}")

            # 返回成功信息和新帖子ID
            return {"success": True, "post_id": post_id}

        except Exception as e:
            # 捕获所有可能的异常
            # str(e)将异常对象转换为字符串，获取错误信息
            # 返回失败信息和错误详情
            return {"success": False, "error": str(e)}

    async def repost(self, agent_id: int, post_id: int):
        """转发帖子的方法

        参数:
        agent_id: 执行转发操作的用户ID
        post_id: 被转发的帖子ID

        返回:
        包含操作结果的字典
        """
        # 根据推荐系统类型确定当前时间
        if self.recsys_type in [RecsysType.REDDIT, RecsysType.COGNITIVE]:
            # 如果是Reddit类型，使用沙盒时钟转换当前时间
            current_time = self.sandbox_clock.time_transfer(
                datetime.now(), self.start_time)
        else:
            # 否则从环境变量获取沙盒时间
            current_time = os.environ["SANDBOX_TIME"]
        try:
            # 将agent_id赋值给user_id（在这个函数中它们是相同的）
            user_id = agent_id

            # 检查该用户是否已经转发过这个帖子
            # 构建SQL查询语句，查找该用户是否已转发过指定帖子
            repost_check_query = (
                "SELECT * FROM 'post' WHERE original_post_id = ? AND "
                "user_id = ?")
            # 执行查询，传入帖子ID和用户ID作为参数
            self.pl_utils._execute_db_command(repost_check_query,
                                              (post_id, user_id))
            # 如果查询结果不为空，说明已经转发过
            if self.db_cursor.fetchone():
                # 对于普通帖子和引用帖子，检查是否已被转发
                # 返回错误信息
                return {
                    "success": False,
                    "error": "Repost record already exists."
                }

            # 获取帖子类型信息
            post_type_result = self.pl_utils._get_post_type(post_id)
            # 构建插入转发帖子的SQL语句
            post_insert_query = ("INSERT INTO post (user_id, original_post_id"
                                 ", created_at) VALUES (?, ?, ?)")
            # 构建更新原帖子分享数的SQL语句
            update_shares_query = (
                "UPDATE post SET num_shares = num_shares + 1 WHERE post_id = ?"
            )

            # 如果找不到帖子，返回错误
            if not post_type_result:
                return {"success": False, "error": "Post not found."}
            # 如果是普通帖子或引用帖子
            elif (post_type_result['type'] == 'common'
                  or post_type_result['type'] == 'quote'):
                # 插入新的转发帖子记录
                self.pl_utils._execute_db_command(
                    post_insert_query, (user_id, post_id, current_time),
                    commit=True)
                # 更新原帖子的分享数
                self.pl_utils._execute_db_command(update_shares_query,
                                                  (post_id, ),
                                                  commit=True)
            # 如果是转发帖子
            elif post_type_result['type'] == 'repost':
                # 检查用户是否已经转发过原始帖子
                repost_check_query = (
                    "SELECT * FROM 'post' WHERE original_post_id = ? AND "
                    "user_id = ?")
                self.pl_utils._execute_db_command(
                    repost_check_query,
                    (post_type_result['root_post_id'], user_id))

                # 如果已经转发过原始帖子，返回错误
                if self.db_cursor.fetchone():
                    # 对于转发帖子，检查是否已经转发过原始帖子
                    return {
                        "success": False,
                        "error": "Repost record already exists."
                    }

                # 插入新的转发帖子记录，但使用原始帖子的ID
                self.pl_utils._execute_db_command(
                    post_insert_query,
                    (user_id, post_type_result['root_post_id'], current_time),
                    commit=True)
                # 更新原始帖子的分享数
                self.pl_utils._execute_db_command(
                    update_shares_query, (post_type_result['root_post_id'], ),
                    commit=True)

            # 获取新插入的转发帖子ID
            new_post_id = self.db_cursor.lastrowid

            # 创建操作信息字典，包含被转发的帖子ID和新帖子ID
            action_info = {"reposted_id": post_id, "new_post_id": new_post_id}
            # 记录用户转发帖子的行为
            self.pl_utils._record_trace(user_id, ActionType.REPOST.value,
                                        action_info, current_time)

            # 返回成功信息和新帖子ID
            return {"success": True, "post_id": new_post_id}
        except Exception as e:
            # 捕获所有可能的异常，返回失败信息和错误详情
            return {"success": False, "error": str(e)}

    async def quote_post(self, agent_id: int, quote_message: tuple):
        """引用帖子的方法，允许用户引用其他帖子并添加自己的评论内容"""
        # 解析引用消息，获取被引用的帖子ID和引用内容
        post_id, quote_content = quote_message
        # 根据推荐系统类型确定当前时间
        if self.recsys_type in [RecsysType.REDDIT, RecsysType.COGNITIVE]:
            # 如果是Reddit类型，使用沙盒时钟转换当前时间
            current_time = self.sandbox_clock.time_transfer(
                datetime.now(), self.start_time)
        else:
            # 否则从环境变量中获取沙盒时间
            current_time = os.environ["SANDBOX_TIME"]
        try:
            # 设置用户ID为代理ID
            user_id = agent_id

            # 允许多次引用同一帖子，因为每次引用内容可能不同

            # SQL查询语句，用于获取指定帖子ID的内容
            post_query = "SELECT content FROM post WHERE post_id = ?"

            # 获取帖子类型信息
            post_type_result = self.pl_utils._get_post_type(post_id)
            # SQL插入语句，用于创建新的引用帖子
            post_insert_query = (
                "INSERT INTO post (user_id, original_post_id, "
                "content, quote_content, created_at) VALUES (?, ?, ?, ?, ?)")
            # SQL更新语句，用于增加原帖子的分享数
            update_shares_query = (
                "UPDATE post SET num_shares = num_shares + 1 WHERE post_id = ?"
            )

            # 检查帖子是否存在
            if not post_type_result:
                # 如果帖子不存在，返回错误信息
                return {"success": False, "error": "Post not found."}
            elif post_type_result['type'] == 'common':
                # 如果是普通帖子，获取其内容
                self.pl_utils._execute_db_command(post_query, (post_id, ))
                # fetchone()[0]获取查询结果的第一行第一列，即帖子内容
                post_content = self.db_cursor.fetchone()[0]
                # 插入新的引用帖子记录
                self.pl_utils._execute_db_command(
                    post_insert_query, (user_id, post_id, post_content,
                                        quote_content, current_time),
                    commit=True)
                # 更新原帖子的分享数
                self.pl_utils._execute_db_command(update_shares_query,
                                                  (post_id, ),
                                                  commit=True)
            elif (post_type_result['type'] == 'repost'
                  or post_type_result['type'] == 'quote'):
                # 如果是转发或引用帖子，获取原始帖子的内容
                self.pl_utils._execute_db_command(
                    post_query, (post_type_result['root_post_id'], ))
                post_content = self.db_cursor.fetchone()[0]
                # 插入新的引用帖子记录，使用原始帖子ID
                self.pl_utils._execute_db_command(
                    post_insert_query,
                    (user_id, post_type_result['root_post_id'], post_content,
                     quote_content, current_time),
                    commit=True)
                # 更新原始帖子的分享数
                self.pl_utils._execute_db_command(
                    update_shares_query, (post_type_result['root_post_id'], ),
                    commit=True)

            # 获取新插入的引用帖子ID
            new_post_id = self.db_cursor.lastrowid

            # 创建操作信息字典，包含被引用的帖子ID和新帖子ID
            action_info = {"quoted_id": post_id, "new_post_id": new_post_id}
            # 记录用户引用帖子的行为
            self.pl_utils._record_trace(user_id, ActionType.QUOTE_POST.value,
                                        action_info, current_time)

            # 返回成功信息和新帖子ID
            return {"success": True, "post_id": new_post_id}
        except Exception as e:
            # 捕获所有可能的异常，返回失败信息和错误详情
            return {"success": False, "error": str(e)}

    async def like_post(self, agent_id: int, post_id: int):
        if self.recsys_type in [RecsysType.REDDIT, RecsysType.COGNITIVE]:
            current_time = self.sandbox_clock.time_transfer(
                datetime.now(), self.start_time)
        else:
            current_time = os.environ["SANDBOX_TIME"]
        try:
            post_type_result = self.pl_utils._get_post_type(post_id)
            if post_type_result['type'] == 'repost':
                post_id = post_type_result['root_post_id']
            user_id = agent_id
            # Check if a like record already exists
            like_check_query = ("SELECT * FROM 'like' WHERE post_id = ? AND "
                                "user_id = ?")
            self.pl_utils._execute_db_command(like_check_query,
                                              (post_id, user_id))
            if self.db_cursor.fetchone():
                # Like record already exists
                return {
                    "success": False,
                    "error": "Like record already exists."
                }

            # Check if the post to be liked is self-posted
            if self.allow_self_rating is False:
                check_result = self.pl_utils._check_self_post_rating(
                    post_id, user_id)
                if check_result:
                    return check_result

            # Update the number of likes in the post table
            post_update_query = (
                "UPDATE post SET num_likes = num_likes + 1 WHERE post_id = ?")
            self.pl_utils._execute_db_command(post_update_query, (post_id, ),
                                              commit=True)

            # 获取当前轮次信息
            try:
                current_step = int(os.environ.get("TIME_STAMP", 0))
            except (ValueError, TypeError):
                current_step = 0

            # Add a record in the like table
            like_insert_query = (
                "INSERT INTO 'like' (post_id, user_id, created_at, step_number) "
                "VALUES (?, ?, ?, ?)")
            self.pl_utils._execute_db_command(like_insert_query,
                                              (post_id, user_id, current_time, current_step),
                                              commit=True)
            # Get the ID of the newly inserted like record
            like_id = self.db_cursor.lastrowid

            # Record the action in the trace table
            # if post has been reposted, record the root post id into trace
            action_info = {"post_id": post_id, "like_id": like_id}
            self.pl_utils._record_trace(user_id, ActionType.LIKE_POST.value,
                                        action_info, current_time)
            return {"success": True, "like_id": like_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def unlike_post(self, agent_id: int, post_id: int):
        try:
            post_type_result = self.pl_utils._get_post_type(post_id)
            if post_type_result['type'] == 'repost':
                post_id = post_type_result['root_post_id']
            user_id = agent_id

            # Check if a like record already exists
            like_check_query = ("SELECT * FROM 'like' WHERE post_id = ? AND "
                                "user_id = ?")
            self.pl_utils._execute_db_command(like_check_query,
                                              (post_id, user_id))
            result = self.db_cursor.fetchone()

            if not result:
                # No like record exists
                return {
                    "success": False,
                    "error": "Like record does not exist."
                }

            # Get the `like_id`
            like_id, _, _, _ = result

            # Update the number of likes in the post table
            post_update_query = (
                "UPDATE post SET num_likes = num_likes - 1 WHERE post_id = ?")
            self.pl_utils._execute_db_command(
                post_update_query,
                (post_id, ),
                commit=True,
            )

            # Delete the record in the like table
            like_delete_query = "DELETE FROM 'like' WHERE like_id = ?"
            self.pl_utils._execute_db_command(
                like_delete_query,
                (like_id, ),
                commit=True,
            )

            # Record the action in the trace table
            action_info = {"post_id": post_id, "like_id": like_id}
            self.pl_utils._record_trace(user_id, ActionType.UNLIKE_POST.value,
                                        action_info)
            return {"success": True, "like_id": like_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def dislike_post(self, agent_id: int, post_id: int):
        if self.recsys_type in [RecsysType.REDDIT, RecsysType.COGNITIVE]:
            current_time = self.sandbox_clock.time_transfer(
                datetime.now(), self.start_time)
        else:
            current_time = os.environ["SANDBOX_TIME"]
        try:
            post_type_result = self.pl_utils._get_post_type(post_id)
            if post_type_result['type'] == 'repost':
                post_id = post_type_result['root_post_id']
            user_id = agent_id
            # Check if a dislike record already exists
            like_check_query = (
                "SELECT * FROM 'dislike' WHERE post_id = ? AND user_id = ?")
            self.pl_utils._execute_db_command(like_check_query,
                                              (post_id, user_id))
            if self.db_cursor.fetchone():
                # Dislike record already exists
                return {
                    "success": False,
                    "error": "Dislike record already exists."
                }

            # Check if the post to be disliked is self-posted
            if self.allow_self_rating is False:
                check_result = self.pl_utils._check_self_post_rating(
                    post_id, user_id)
                if check_result:
                    return check_result

            # Update the number of dislikes in the post table
            post_update_query = (
                "UPDATE post SET num_dislikes = num_dislikes + 1 WHERE "
                "post_id = ?")
            self.pl_utils._execute_db_command(post_update_query, (post_id, ),
                                              commit=True)

            # 获取当前轮次信息
            try:
                current_step = int(os.environ.get("TIME_STAMP", 0))
            except (ValueError, TypeError):
                current_step = 0

            # Add a record in the dislike table
            dislike_insert_query = (
                "INSERT INTO 'dislike' (post_id, user_id, created_at, step_number) "
                "VALUES (?, ?, ?, ?)")
            self.pl_utils._execute_db_command(dislike_insert_query,
                                              (post_id, user_id, current_time, current_step),
                                              commit=True)
            # Get the ID of the newly inserted dislike record
            dislike_id = self.db_cursor.lastrowid

            # Record the action in the trace table
            action_info = {"post_id": post_id, "dislike_id": dislike_id}
            self.pl_utils._record_trace(user_id, ActionType.DISLIKE_POST.value,
                                        action_info, current_time)
            return {"success": True, "dislike_id": dislike_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def undo_dislike_post(self, agent_id: int, post_id: int):
        try:
            post_type_result = self.pl_utils._get_post_type(post_id)
            if post_type_result['type'] == 'repost':
                post_id = post_type_result['root_post_id']
            user_id = agent_id

            # Check if a dislike record already exists
            like_check_query = (
                "SELECT * FROM 'dislike' WHERE post_id = ? AND user_id = ?")
            self.pl_utils._execute_db_command(like_check_query,
                                              (post_id, user_id))
            result = self.db_cursor.fetchone()

            if not result:
                # No dislike record exists
                return {
                    "success": False,
                    "error": "Dislike record does not exist."
                }

            # Get the `dislike_id`
            dislike_id, _, _, _ = result

            # Update the number of dislikes in the post table
            post_update_query = (
                "UPDATE post SET num_dislikes = num_dislikes - 1 WHERE "
                "post_id = ?")
            self.pl_utils._execute_db_command(
                post_update_query,
                (post_id, ),
                commit=True,
            )

            # Delete the record in the dislike table
            like_delete_query = "DELETE FROM 'dislike' WHERE dislike_id = ?"
            self.pl_utils._execute_db_command(
                like_delete_query,
                (dislike_id, ),
                commit=True,
            )

            # Record the action in the trace table
            action_info = {"post_id": post_id, "dislike_id": dislike_id}
            self.pl_utils._record_trace(user_id,
                                        ActionType.UNDO_DISLIKE_POST.value,
                                        action_info)
            return {"success": True, "dislike_id": dislike_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def search_posts(self, agent_id: int, query: str):
        try:
            user_id = agent_id
            # Update the SQL query to search by content, post_id, and user_id
            # simultaneously
            sql_query = (
                "SELECT post_id, user_id, original_post_id, content, "
                "quote_content, created_at, num_likes, num_dislikes, "
                "num_shares FROM post WHERE content LIKE ? OR CAST(post_id AS "
                "TEXT) LIKE ? OR CAST(user_id AS TEXT) LIKE ?")
            # Note: CAST is necessary because post_id and user_id are integers,
            # while the search query is a string type
            self.pl_utils._execute_db_command(
                sql_query,
                ("%" + query + "%", "%" + query + "%", "%" + query + "%"),
                commit=True,
            )
            results = self.db_cursor.fetchall()

            # Record the operation in the trace table
            action_info = {"query": query}
            self.pl_utils._record_trace(user_id, ActionType.SEARCH_POSTS.value,
                                        action_info)

            # If no results are found, return a dictionary indicating failure
            if not results:
                return {
                    "success": False,
                    "message": "No posts found matching the query.",
                }
            results_with_comments = self.pl_utils._add_comments_to_posts(
                results)

            return {"success": True, "posts": results_with_comments}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def search_user(self, agent_id: int, query: str):
        try:
            user_id = agent_id
            sql_query = (
                "SELECT user_id, user_name, name, bio, created_at, "
                "num_followings, num_followers "
                "FROM user "
                "WHERE user_name LIKE ? OR name LIKE ? OR bio LIKE ? OR "
                "CAST(user_id AS TEXT) LIKE ?")
            # Rewrite to use the execute_db_command method
            self.pl_utils._execute_db_command(
                sql_query,
                (
                    "%" + query + "%",
                    "%" + query + "%",
                    "%" + query + "%",
                    "%" + query + "%",
                ),
                commit=True,
            )
            results = self.db_cursor.fetchall()

            # Record the operation in the trace table
            action_info = {"query": query}
            self.pl_utils._record_trace(user_id, ActionType.SEARCH_USER.value,
                                        action_info)

            # If no results are found, return a dict indicating failure
            if not results:
                return {
                    "success": False,
                    "message": "No users found matching the query.",
                }

            # Convert each tuple in results into a dictionary
            users = [{
                "user_id": user_id,
                "user_name": user_name,
                "name": name,
                "bio": bio,
                "created_at": created_at,
                "num_followings": num_followings,
                "num_followers": num_followers,
            } for user_id, user_name, name, bio, created_at, num_followings,
                     num_followers in results]
            return {"success": True, "users": users}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def follow(self, agent_id: int, followee_id: int, step_number: int = None):
        if self.recsys_type in [RecsysType.REDDIT, RecsysType.COGNITIVE]:
            current_time = self.sandbox_clock.time_transfer(
                datetime.now(), self.start_time)
        else:
            current_time = os.environ["SANDBOX_TIME"]
        try:
            user_id = agent_id
            # Check if a follow record already exists
            follow_check_query = ("SELECT * FROM follow WHERE follower_id = ? "
                                  "AND followee_id = ?")
            self.pl_utils._execute_db_command(follow_check_query,
                                              (user_id, followee_id))
            if self.db_cursor.fetchone():
                # Follow record already exists
                return {
                    "success": False,
                    "error": "Follow record already exists."
                }

            # 获取当前轮次数，如果未提供则使用环境变量中的值
            if step_number is None:
                # 尝试从环境变量获取时间步
                try:
                    step_number = int(os.environ.get("TIME_STAMP", 0))
                except (ValueError, TypeError):
                    step_number = 0

            # Add a record in the follow table
            follow_insert_query = (
                "INSERT INTO follow (follower_id, followee_id, created_at, step_number) "
                "VALUES (?, ?, ?, ?)")
            self.pl_utils._execute_db_command(
                follow_insert_query, (user_id, followee_id, current_time, step_number),
                commit=True)
            # Get the ID of the newly inserted follow record
            follow_id = self.db_cursor.lastrowid

            # Update the following field in the user table
            user_update_query1 = (
                "UPDATE user SET num_followings = num_followings + 1 "
                "WHERE user_id = ?")
            self.pl_utils._execute_db_command(user_update_query1, (user_id, ),
                                              commit=True)

            # Update the follower field in the user table
            user_update_query2 = (
                "UPDATE user SET num_followers = num_followers + 1 "
                "WHERE user_id = ?")
            self.pl_utils._execute_db_command(user_update_query2,
                                              (followee_id, ),
                                              commit=True)

            # Record the operation in the trace table
            action_info = {"follow_id": follow_id, "step_number": step_number}
            self.pl_utils._record_trace(user_id, ActionType.FOLLOW.value,
                                        action_info, current_time)
            twitter_log.info(f"Trace inserted: user_id={user_id}, "
                             f"current_time={current_time}, "
                             f"action={ActionType.FOLLOW.value}, "
                             f"info={action_info}, step_number={step_number}")
            return {"success": True, "follow_id": follow_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def unfollow(self, agent_id: int, followee_id: int):
        try:
            user_id = agent_id
            # Check for the existence of a follow record and get its ID
            follow_check_query = (
                "SELECT follow_id FROM follow WHERE follower_id = ? AND "
                "followee_id = ?")
            self.pl_utils._execute_db_command(follow_check_query,
                                              (user_id, followee_id))
            follow_record = self.db_cursor.fetchone()
            if not follow_record:
                return {
                    "success": False,
                    "error": "Follow record does not exist."
                }
            # Assuming ID is in the first column of the result
            follow_id = follow_record[0]

            # Delete the record in the follow table
            follow_delete_query = "DELETE FROM follow WHERE follow_id = ?"
            self.pl_utils._execute_db_command(follow_delete_query,
                                              (follow_id, ),
                                              commit=True)

            # Update the following field in the user table
            user_update_query1 = (
                "UPDATE user SET num_followings = num_followings - 1 "
                "WHERE user_id = ?")
            self.pl_utils._execute_db_command(user_update_query1, (user_id, ),
                                              commit=True)

            # Update the follower field in the user table
            user_update_query2 = (
                "UPDATE user SET num_followers = num_followers - 1 "
                "WHERE user_id = ?")
            self.pl_utils._execute_db_command(user_update_query2,
                                              (followee_id, ),
                                              commit=True)

            # Record the operation in the trace table
            action_info = {"followee_id": followee_id}
            self.pl_utils._record_trace(user_id, ActionType.UNFOLLOW.value,
                                        action_info)
            return {
                "success": True,
                "follow_id": follow_id,
            }  # Return the ID of the deleted follow record
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def mute(self, agent_id: int, mutee_id: int):
        if self.recsys_type in [RecsysType.REDDIT, RecsysType.COGNITIVE]:
            current_time = self.sandbox_clock.time_transfer(
                datetime.now(), self.start_time)
        else:
            current_time = os.environ["SANDBOX_TIME"]
        try:
            user_id = agent_id
            # Check if a mute record already exists
            mute_check_query = ("SELECT * FROM mute WHERE muter_id = ? AND "
                                "mutee_id = ?")
            self.pl_utils._execute_db_command(mute_check_query,
                                              (user_id, mutee_id))
            if self.db_cursor.fetchone():
                # Mute record already exists
                return {
                    "success": False,
                    "error": "Mute record already exists."
                }
            # 获取当前轮次信息
            try:
                current_step = int(os.environ.get("TIME_STAMP", 0))
            except (ValueError, TypeError):
                current_step = 0

            # Add a record in the mute table
            mute_insert_query = (
                "INSERT INTO mute (muter_id, mutee_id, created_at, step_number) "
                "VALUES (?, ?, ?, ?)")
            self.pl_utils._execute_db_command(
                mute_insert_query, (user_id, mutee_id, current_time, current_step),
                commit=True)
            # Get the ID of the newly inserted mute record
            mute_id = self.db_cursor.lastrowid

            # Record the operation in the trace table
            action_info = {"mutee_id": mutee_id}
            self.pl_utils._record_trace(user_id, ActionType.MUTE.value,
                                        action_info, current_time)
            return {"success": True, "mute_id": mute_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def unmute(self, agent_id: int, mutee_id: int):
        """取消屏蔽用户"""
        # 检查用户是否已注册
        if not await self.check_user_registered(agent_id):
            return {"status": "error", "message": "User not registered"}
        if not await self.check_user_registered(mutee_id):
            return {"status": "error", "message": "Target user not registered"}

        # 构建SQL语句并执行
        sql = """DELETE FROM mute WHERE muter_id = ? AND mutee_id = ?"""
        self.pl_utils._execute_db_command(sql, (agent_id, mutee_id), commit=True)

        # 返回成功信息
        return {"status": "success", "unmuted_id": mutee_id}

    # 移除update_think_table方法

    async def check_database_state(self):
        #print(f"==DEBUG== 检查数据库状态")
        try:
            # 检查think表
            self.db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='think'")
            if self.db_cursor.fetchone():
                #print(f"==DEBUG== think表存在")
                self.db_cursor.execute("PRAGMA table_info(think)")
                columns = self.db_cursor.fetchall()
                #print(f"==DEBUG== think表结构: {columns}")
            else:
                #print(f"==DEBUG== think表不存在")
                pass

            # 检查trace表
            self.db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trace'")
            if self.db_cursor.fetchone():
                #print(f"==DEBUG== trace表存在")
                self.db_cursor.execute("PRAGMA table_info(trace)")
                columns = self.db_cursor.fetchall()
                #print(f"==DEBUG== trace表结构: {columns}")
            else:
                #print(f"==DEBUG== trace表不存在")
                pass

            # 检查数据库中的表
            self.db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = self.db_cursor.fetchall()
            #print(f"==DEBUG== 数据库中的表: {tables}")

            return {"success": True}
        except Exception as e:
            print(f"==ERROR== 检查数据库状态时出错: {str(e)}")
            return {"success": False, "error": str(e)}

    async def save_user_information(self, user_id, persona, age, gender, mbti, country, profession, interested_topics):
        """
        保存用户扩展信息到user_information表

        Args:
            user_id: 用户ID
            persona: 用户画像描述
            age: 年龄段
            gender: 性别
            mbti: MBTI人格类型
            country: 国家
            profession: 职业
            interested_topics: 用户感兴趣话题(JSON字符串)

        Returns:
            字典: 包含操作是否成功的信息
        """
        try:
            # 创建平台日志记录器
            platform_log = logging.getLogger(name="social.platform")
            platform_log.debug(f"save_user_information被调用: user_id={user_id}")

            # 检查用户是否存在
            user_check_query = "SELECT * FROM user WHERE user_id = ?"
            self.pl_utils._execute_db_command(user_check_query, (user_id, ))
            if not self.pl_utils.db_cursor.fetchone():
                platform_log.warning(f"用户不存在: user_id={user_id}")
                return {"success": False, "error": "User not found."}

            # 检查user_information表是否存在，如果不存在则创建
            self.pl_utils.db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_information'")
            if not self.pl_utils.db_cursor.fetchone():
                # 读取并执行SQL文件
                try:
                    with open("asce/social_platform/schema/user_information.sql", "r") as sql_file:
                        sql_script = sql_file.read()
                        self.pl_utils.db_connection.executescript(sql_script)
                    platform_log.info("成功创建user_information表")
                except Exception as e:
                    platform_log.error(f"创建user_information表失败: {str(e)}")
                    return {"success": False, "error": f"Failed to create user_information table: {str(e)}"}

            # 检查是否已存在相同用户ID的记录
            check_existing_query = "SELECT info_id FROM user_information WHERE user_id = ?"
            self.pl_utils._execute_db_command(check_existing_query, (user_id,))
            existing_record = self.pl_utils.db_cursor.fetchone()

            # 准备参数
            params = (
                user_id,
                persona,
                age,
                gender,
                mbti,
                country,
                profession,
                interested_topics
            )

            # 根据是否存在记录决定更新或插入
            if existing_record:
                platform_log.debug(f"发现相同user_id的记录，将进行更新")
                info_update_query = (
                    "UPDATE user_information SET persona = ?, age = ?, gender = ?, "
                    "mbti = ?, country = ?, profession = ?, interested_topics = ? "
                    "WHERE user_id = ?")

                # 为更新操作调整参数顺序（最后一个参数是WHERE条件）
                update_params = params[1:] + (user_id,)

                self.pl_utils._execute_db_command(
                    info_update_query,
                    update_params,
                    commit=True)
            else:
                platform_log.debug(f"未发现相同记录，将插入新记录")
                info_insert_query = (
                    "INSERT INTO user_information (user_id, persona, age, gender, "
                    "mbti, country, profession, interested_topics) VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?)")

                self.pl_utils._execute_db_command(
                    info_insert_query,
                    params,
                    commit=True)

            platform_log.debug(f"SQL执行成功")
            return {"success": True, "user_id": user_id}
        except Exception as e:
            platform_log.error(f"保存用户信息时出错: {str(e)}")
            import traceback
            platform_log.error(f"错误详情: {traceback.format_exc()}")
            return {"success": False, "error": str(e)}

    # 缓存用户行为数据，用于批量保存
    _user_action_cache = []
    _user_action_cache_lock = asyncio.Lock()
    _user_action_cache_size_limit = 1000  # 增大缓存大小限制，减少批量保存频率

    # 生产者-消费者模式相关变量
    _data_save_queue = asyncio.Queue()
    _consumer_task = None
    _is_consumer_running = False

    async def save_user_action(
            self, user_id, timestep, post_id, action, reason, mood_type,
            mood_value, emotion_type, emotion_value, stance_type, stance_value,
            thinking_type, thinking_value, intention_type, intention_value,
            is_active, action_info, viewpoint_1="Indifferent", viewpoint_2="Indifferent",
            viewpoint_3="Indifferent", viewpoint_4="Indifferent", viewpoint_5="Indifferent",
            viewpoint_6="Indifferent"):
            """保存用户行为到user_action表

            Args:
                user_id: 用户ID
                timestep: 时间步，在数据库中对应num_steps列
                post_id: 帖子ID
                action: 行为名称
                reason: 行为原因
                mood_type: 情感类型
                mood_value: 情感值
                emotion_type: 情绪类型
                emotion_value: 情绪值
                stance_type: 立场类型
                stance_value: 立场值
                thinking_type: 认知类型
                thinking_value: 认知值
                intention_type: 意图类型
                intention_value: 意图值
                is_active: 用户是否激活(true/false)
                action_info: 行为详细信息(JSON)
                viewpoint_1: 观点1的支持级别，默认为"Indifferent"
                viewpoint_2: 观点2的支持级别，默认为"Indifferent"
                viewpoint_3: 观点3的支持级别，默认为"Indifferent"
                viewpoint_4: 观点4的支持级别，默认为"Indifferent"
                viewpoint_5: 观点5的支持级别，默认为"Indifferent"
                viewpoint_6: 观点6的支持级别，默认为"Indifferent"

            Returns:
                dict: 包含操作结果的字典
            """
            # 获取平台日志记录器
            platform_log = logging.getLogger(name="social.platform")
            platform_log.debug(f"save_user_action被调用: user_id={user_id}, action={action}")

            # 如果用户未激活，不保存数据
            if is_active == "false":
                platform_log.debug(f"用户{user_id}未激活，跳过保存")
                return {"success": True, "action": "skipped"}

            try:
                # 将action_info转换为JSON字符串
                action_info_json = json.dumps(action_info) if action_info else "{}"

                # 检查用户是否存在
                user_check_query = "SELECT * FROM user WHERE user_id = ?"
                self.pl_utils._execute_db_command(user_check_query, (user_id, ))
                if not self.pl_utils.db_cursor.fetchone():
                    platform_log.warning(f"用户不存在: user_id={user_id}")
                    return {"success": False, "error": "User not found."}

                # 计算哈希值作为记录唯一标识（实际未使用）
                record_hash = hashlib.md5(f"{user_id}_{timestep}_{post_id}_{action}".encode()).hexdigest()

                # 检查该记录是否已存在
                existing_query = "SELECT action_id FROM user_action WHERE user_id = ? AND num_steps = ?"
                self.pl_utils._execute_db_command(existing_query, (user_id, timestep))
                existing_record = self.pl_utils.db_cursor.fetchone()

                # 日志记录观点支持级别
                platform_log.debug(f"保存的观点支持级别: viewpoint_1={viewpoint_1}, viewpoint_2={viewpoint_2}, viewpoint_3={viewpoint_3}, viewpoint_4={viewpoint_4}, viewpoint_5={viewpoint_5}, viewpoint_6={viewpoint_6}")

                # 准备数据记录
                if existing_record:
                    # 更新现有记录
                    action_record = {
                        "type": "update",
                        "params": (
                            post_id, action, reason, action_info_json,
                            mood_type, mood_value,
                            emotion_type, emotion_value,
                            stance_type, stance_value,
                            thinking_type, thinking_value,
                            intention_type, intention_value,
                            is_active,
                            viewpoint_1, viewpoint_2,
                            viewpoint_3, viewpoint_4,
                            viewpoint_5, viewpoint_6,
                            user_id, timestep
                        )
                    }
                else:
                    # 插入新记录
                    action_record = {
                        "type": "insert",
                        "params": (
                            user_id, timestep, post_id, action, reason, action_info_json,
                            mood_type, mood_value,
                            emotion_type, emotion_value,
                            stance_type, stance_value,
                            thinking_type, thinking_value,
                            intention_type, intention_value,
                            viewpoint_1, viewpoint_2, viewpoint_3,
                            viewpoint_4, viewpoint_5, viewpoint_6,
                            is_active
                        )
                    }

                # 使用锁来保护缓存操作
                async with self._user_action_cache_lock:
                    # 将记录添加到缓存
                    self._user_action_cache.append(action_record)

                    # 确保消费者协程正在运行
                    if not self._is_consumer_running:
                        self._start_consumer()

                return {"success": True, "action": action_record["type"]}
            except Exception as e:
                platform_log.error(f"保存用户行为出错: {str(e)}")
                import traceback
                platform_log.error(traceback.format_exc())
                return {"success": False, "error": str(e)}

    def _start_consumer(self):
        """启动数据保存消费者协程"""
        platform_log = logging.getLogger(name="social.platform")
        if not self._is_consumer_running:
            platform_log.info("启动数据保存消费者协程")
            self._is_consumer_running = True
            self._consumer_task = asyncio.create_task(self._data_save_consumer())

    async def _data_save_consumer(self):
        """数据保存消费者协程，定期将缓存中的数据保存到数据库"""
        platform_log = logging.getLogger(name="social.platform")
        platform_log.info("数据保存消费者协程已启动")

        try:
            while True:
                # 每5秒检查一次缓存
                await asyncio.sleep(5)

                # 如果缓存中有数据，批量保存
                async with self._user_action_cache_lock:
                    if self._user_action_cache:
                        cache_size = len(self._user_action_cache)
                        platform_log.info(f"消费者发现{cache_size}条缓存数据，开始批量保存")
                        await self._batch_save_user_actions()
        except asyncio.CancelledError:
            platform_log.info("数据保存消费者协程被取消")
            self._is_consumer_running = False
        except Exception as e:
            platform_log.error(f"数据保存消费者协程出错: {str(e)}")
            import traceback
            platform_log.error(traceback.format_exc())
            self._is_consumer_running = False

    async def _batch_save_user_actions(self):
        """批量保存用户行为数据"""
        platform_log = logging.getLogger(name="social.platform")

        if not self._user_action_cache:
            return

        try:
            # 开始事务
            self.db.execute("BEGIN TRANSACTION")

            # 准备SQL语句
            update_query = """
            UPDATE user_action
            SET post_id = ?, action = ?, reason = ?, info = ?,
                mood_type = ?, mood_value = ?,
                emotion_type = ?, emotion_value = ?,
                stance_type = ?, stance_value = ?,
                thinking_type = ?, thinking_value = ?,
                intention_type = ?, intention_value = ?,
                is_active = ?,
                viewpoint_1 = ?, viewpoint_2 = ?,
                viewpoint_3 = ?, viewpoint_4 = ?,
                viewpoint_5 = ?, viewpoint_6 = ?
            WHERE user_id = ? AND num_steps = ?
            """

            insert_query = """
            INSERT INTO user_action (
                user_id, num_steps, post_id, action, reason, info,
                mood_type, mood_value,
                emotion_type, emotion_value,
                stance_type, stance_value,
                thinking_type, thinking_value,
                intention_type, intention_value,
                viewpoint_1, viewpoint_2, viewpoint_3,
                viewpoint_4, viewpoint_5, viewpoint_6,
                is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            # 分类处理更新和插入操作
            update_records = [record["params"] for record in self._user_action_cache if record["type"] == "update"]
            insert_records = [record["params"] for record in self._user_action_cache if record["type"] == "insert"]

            # 批量执行更新
            if update_records:
                self.db.executemany(update_query, update_records)
                platform_log.debug(f"批量更新{len(update_records)}条user_action记录")

            # 批量执行插入
            if insert_records:
                self.db.executemany(insert_query, insert_records)
                platform_log.debug(f"批量插入{len(insert_records)}条user_action记录")

            # 提交事务
            self.db.commit()
            platform_log.info(f"批量保存完成: 更新{len(update_records)}条，插入{len(insert_records)}条")

            # 清空缓存
            self._user_action_cache.clear()

        except Exception as e:
            # 回滚事务
            self.db.rollback()
            platform_log.error(f"批量保存user_action表出错: {str(e)}")
            import traceback
            platform_log.error(traceback.format_exc())

    async def _flush_all_caches(self):
        """将所有缓存中的数据写入数据库"""
        platform_log = logging.getLogger(name="social.platform")
        platform_log.info("模拟结束，将所有缓存数据写入数据库")

        # 保存user_action缓存
        if hasattr(self, "_user_action_cache") and self._user_action_cache:
            platform_log.info(f"正在保存{len(self._user_action_cache)}条user_action缓存数据")
            await self._batch_save_user_actions()

        # 如果将来有其他缓存，可以在这里添加对应的保存操作

        platform_log.info("所有缓存数据已写入数据库")

    async def trend(self, agent_id: int):
        """获取热门话题"""
        # 根据推荐系统类型确定当前时间
        if self.recsys_type in [RecsysType.REDDIT, RecsysType.COGNITIVE]:
            # 如果是Reddit类型的推荐系统，使用沙盒时钟转换当前时间
            # sandbox_clock.time_transfer方法将真实时间转换为模拟环境中的时间
            current_time = self.sandbox_clock.time_transfer(
                datetime.now(), self.start_time)
        else:
            # 否则从环境变量中获取沙盒时间
            current_time = os.environ["SANDBOX_TIME"]
        try:
            # 将agent_id赋值给user_id变量，便于后续使用
            user_id = agent_id
            # 计算查询的起始时间
            if self.recsys_type in [RecsysType.REDDIT, RecsysType.COGNITIVE]:
                # 对于Reddit类型，使用datetime的timedelta对象计算过去几天的时间
                # timedelta是Python中表示时间间隔的类，days参数指定天数
                start_time = current_time - timedelta(days=self.trend_num_days)
            else:
                # 对于其他类型，假设current_time是整数时间戳（以分钟为单位）
                # 计算过去self.trend_num_days天的时间（转换为分钟）
                start_time = int(current_time) - self.trend_num_days * 24 * 60

            # 构建SQL查询语句，使用三引号可以写多行字符串
            # 查询帖子表中在指定时间范围内的帖子，按点赞数降序排序，并限制返回数量
            sql_query = """
                SELECT post_id, user_id, original_post_id, content,
                quote_content, created_at, num_likes, num_dislikes,
                num_shares FROM post
                WHERE created_at >= ?
                ORDER BY num_likes DESC
                LIMIT ?
            """
            # 执行数据库查询，传入起始时间和要获取的热门帖子数量作为参数
            # commit=True表示立即提交事务
            self.pl_utils._execute_db_command(sql_query,
                                              (start_time, self.trend_top_k),
                                              commit=True)
            # 获取查询结果的所有行
            results = self.db_cursor.fetchall()

            # 如果没有找到结果，返回一个表示失败的字典
            if not results:
                return {
                    "success": False,
                    "message": "No trending posts in the specified period.",
                }
            # 为查询结果添加评论信息
            # _add_comments_to_posts是一个辅助方法，用于获取每个帖子的评论并添加到结果中
            results_with_comments = self.pl_utils._add_comments_to_posts(
                results)

            # 创建操作信息字典，包含带有评论的帖子列表
            action_info = {"posts": results_with_comments}
            # 记录用户查看热门帖子的行为
            # ActionType.TREND.value是一个枚举值，表示查看热门帖子的行为类型
            self.pl_utils._record_trace(user_id, ActionType.TREND.value,
                                        action_info, current_time)

            # 返回成功信息和带有评论的帖子列表
            return {"success": True, "posts": results_with_comments}
        except Exception as e:
            # 捕获所有可能的异常，返回失败信息和错误详情
            # str(e)将异常对象转换为字符串
            return {"success": False, "error": str(e)}

    async def create_comment(self, agent_id: int, comment_message: tuple):
        post_id, content = comment_message
        if self.recsys_type in [RecsysType.REDDIT, RecsysType.COGNITIVE]:
            current_time = self.sandbox_clock.time_transfer(
                datetime.now(), self.start_time)
        else:
            current_time = os.environ["SANDBOX_TIME"]
        try:
            post_type_result = self.pl_utils._get_post_type(post_id)
            if post_type_result['type'] == 'repost':
                post_id = post_type_result['root_post_id']
            user_id = agent_id

            # 获取当前轮次信息
            try:
                current_step = int(os.environ.get("TIME_STAMP", 0))
            except (ValueError, TypeError):
                current_step = 0

            # Insert the comment record
            comment_insert_query = (
                "INSERT INTO comment (post_id, user_id, content, created_at, step_number) "
                "VALUES (?, ?, ?, ?, ?)")
            self.pl_utils._execute_db_command(
                comment_insert_query,
                (post_id, user_id, content, current_time, current_step),
                commit=True,
            )
            comment_id = self.db_cursor.lastrowid

            # Prepare information for the trace record
            action_info = {"content": content, "comment_id": comment_id}
            self.pl_utils._record_trace(user_id,
                                        ActionType.CREATE_COMMENT.value,
                                        action_info, current_time)

            return {"success": True, "comment_id": comment_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def like_comment(self, agent_id: int, comment_id: int):
        # 这是一个异步函数，用于处理用户对评论的点赞操作
        # agent_id是点赞用户的ID，comment_id是被点赞评论的ID
        if self.recsys_type in [RecsysType.REDDIT, RecsysType.COGNITIVE]:
            # 如果推荐系统类型是REDDIT，则使用sandbox_clock来转换时间
            current_time = self.sandbox_clock.time_transfer(
                datetime.now(), self.start_time)
        else:
            # 否则，直接从环境变量中获取沙盒时间
            current_time = os.environ["SANDBOX_TIME"]
        try:
            # 尝试执行以下代码块，如果出现异常会被捕获
            user_id = agent_id  # 将agent_id赋值给user_id，表示当前操作的用户

            # 检查是否已经存在点赞记录
            # 这是一个SQL查询语句，用于从comment_like表中查找特定评论和用户的点赞记录
            like_check_query = (
                "SELECT * FROM comment_like WHERE comment_id = ? AND "
                "user_id = ?")
            # 执行SQL查询，传入评论ID和用户ID作为参数
            self.pl_utils._execute_db_command(like_check_query,
                                              (comment_id, user_id))
            # 获取查询结果的第一行
            if self.db_cursor.fetchone():
                # 如果查询结果不为空，说明已经存在点赞记录
                # 返回失败信息，表示不能重复点赞
                return {
                    "success": False,
                    "error": "Comment like record already exists.",
                }

            # 检查要点赞的评论是否是用户自己发布的（如果系统不允许自我评价）
            if self.allow_self_rating is False:
                # 调用工具函数检查是否是自己的评论
                check_result = self.pl_utils._check_self_comment_rating(
                    comment_id, user_id)
                if check_result:
                    # 如果检查结果不为空，说明是自己的评论，返回检查结果（通常是错误信息）
                    return check_result

            # 更新评论表中的点赞数量
            # 这是一个SQL更新语句，将指定评论的点赞数加1
            comment_update_query = (
                "UPDATE comment SET num_likes = num_likes + 1 WHERE "
                "comment_id = ?")
            # 执行SQL更新，并提交事务
            self.pl_utils._execute_db_command(comment_update_query,
                                              (comment_id, ),
                                              commit=True)

            # 获取当前轮次信息
            try:
                current_step = int(os.environ.get("TIME_STAMP", 0))
            except (ValueError, TypeError):
                current_step = 0

            # 在comment_like表中添加一条点赞记录
            # 这是一个SQL插入语句，记录谁对哪条评论进行了点赞以及点赞时间和轮次
            like_insert_query = (
                "INSERT INTO comment_like (comment_id, user_id, created_at, step_number) "
                "VALUES (?, ?, ?, ?)")
            # 执行SQL插入，并提交事务
            self.pl_utils._execute_db_command(
                like_insert_query, (comment_id, user_id, current_time, current_step),
                commit=True)
            # 获取新插入的点赞记录的ID
            # lastrowid是数据库游标的属性，返回最后插入行的ID
            comment_like_id = self.db_cursor.lastrowid

            # 在trace表中记录此操作
            # 创建包含操作信息的字典
            action_info = {
                "comment_id": comment_id,
                "comment_like_id": comment_like_id
            }
            # 调用工具函数记录用户的点赞评论行为
            # ActionType.LIKE_COMMENT.value是一个枚举值，表示点赞评论的行为类型
            self.pl_utils._record_trace(user_id, ActionType.LIKE_COMMENT.value,
                                        action_info, current_time)
            # 返回成功信息和点赞记录ID
            return {"success": True, "comment_like_id": comment_like_id}
        except Exception as e:
            # 捕获所有可能的异常，返回失败信息和错误详情
            # str(e)将异常对象转换为字符串
            return {"success": False, "error": str(e)}

    async def unlike_comment(self, agent_id: int, comment_id: int):
        try:
            user_id = agent_id

            # Check if a like record already exists
            like_check_query = (
                "SELECT * FROM comment_like WHERE comment_id = ? AND "
                "user_id = ?")
            self.pl_utils._execute_db_command(like_check_query,
                                              (comment_id, user_id))
            result = self.db_cursor.fetchone()

            if not result:
                # No like record exists
                return {
                    "success": False,
                    "error": "Comment like record does not exist.",
                }
            # Get the `comment_like_id`
            comment_like_id = result[0]

            # Update the number of likes in the comment table
            comment_update_query = (
                "UPDATE comment SET num_likes = num_likes - 1 WHERE "
                "comment_id = ?")
            self.pl_utils._execute_db_command(
                comment_update_query,
                (comment_id, ),
                commit=True,
            )
            # Delete the record in the comment_like table
            like_delete_query = ("DELETE FROM comment_like WHERE "
                                 "comment_like_id = ?")
            self.pl_utils._execute_db_command(
                like_delete_query,
                (comment_like_id, ),
                commit=True,
            )
            # Record the operation in the trace table
            action_info = {
                "comment_id": comment_id,
                "comment_like_id": comment_like_id
            }
            self.pl_utils._record_trace(user_id,
                                        ActionType.UNLIKE_COMMENT.value,
                                        action_info)
            return {"success": True, "comment_like_id": comment_like_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def dislike_comment(self, agent_id: int, comment_id: int):
        if self.recsys_type in [RecsysType.REDDIT, RecsysType.COGNITIVE]:
            current_time = self.sandbox_clock.time_transfer(
                datetime.now(), self.start_time)
        else:
            current_time = os.environ["SANDBOX_TIME"]
        try:
            user_id = agent_id

            # Check if a dislike record already exists
            dislike_check_query = (
                "SELECT * FROM comment_dislike WHERE comment_id = ? AND "
                "user_id = ?")
            self.pl_utils._execute_db_command(dislike_check_query,
                                              (comment_id, user_id))
            if self.db_cursor.fetchone():
                # Dislike record already exists
                return {
                    "success": False,
                    "error": "Comment dislike record already exists.",
                }

            # Check if the comment to be disliked was posted by oneself
            if self.allow_self_rating is False:
                check_result = self.pl_utils._check_self_comment_rating(
                    comment_id, user_id)
                if check_result:
                    return check_result

            # Update the number of dislikes in the comment table
            comment_update_query = (
                "UPDATE comment SET num_dislikes = num_dislikes + 1 WHERE "
                "comment_id = ?")
            self.pl_utils._execute_db_command(comment_update_query,
                                              (comment_id, ),
                                              commit=True)

            # 获取当前轮次信息
            try:
                current_step = int(os.environ.get("TIME_STAMP", 0))
            except (ValueError, TypeError):
                current_step = 0

            # Add a record in the comment_dislike table
            dislike_insert_query = (
                "INSERT INTO comment_dislike (comment_id, user_id, "
                "created_at, step_number) VALUES (?, ?, ?, ?)")
            self.pl_utils._execute_db_command(
                dislike_insert_query, (comment_id, user_id, current_time, current_step),
                commit=True)
            # Get the ID of the newly inserted dislike record
            comment_dislike_id = (self.db_cursor.lastrowid)

            # Record the operation in the trace table
            action_info = {
                "comment_id": comment_id,
                "comment_dislike_id": comment_dislike_id,
            }
            self.pl_utils._record_trace(user_id,
                                        ActionType.DISLIKE_COMMENT.value,
                                        action_info, current_time)
            return {"success": True, "comment_dislike_id": comment_dislike_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def undo_dislike_comment(self, agent_id: int, comment_id: int):
        try:
            user_id = agent_id

            # Check if a dislike record already exists
            dislike_check_query = (
                "SELECT comment_dislike_id FROM comment_dislike WHERE "
                "comment_id = ? AND user_id = ?")
            self.pl_utils._execute_db_command(dislike_check_query,
                                              (comment_id, user_id))
            dislike_record = self.db_cursor.fetchone()
            if not dislike_record:
                # No dislike record exists
                return {
                    "success": False,
                    "error": "Comment dislike record does not exist.",
                }
            comment_dislike_id = dislike_record[0]

            # Delete the record from the comment_dislike table
            dislike_delete_query = (
                "DELETE FROM comment_dislike WHERE comment_id = ? AND "
                "user_id = ?")
            self.pl_utils._execute_db_command(dislike_delete_query,
                                              (comment_id, user_id),
                                              commit=True)

            # Update the number of dislikes in the comment table
            comment_update_query = (
                "UPDATE comment SET num_dislikes = num_dislikes - 1 WHERE "
                "comment_id = ?")
            self.pl_utils._execute_db_command(comment_update_query,
                                              (comment_id, ),
                                              commit=True)

            # Record the operation in the trace table
            action_info = {
                "comment_id": comment_id,
                "comment_dislike_id": comment_dislike_id,
            }
            self.pl_utils._record_trace(user_id,
                                        ActionType.UNDO_DISLIKE_COMMENT.value,
                                        action_info)
            return {"success": True, "comment_dislike_id": comment_dislike_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def do_nothing(self, agent_id: int):
        try:
            user_id = agent_id

            action_info = {}
            self.pl_utils._record_trace(user_id, ActionType.DO_NOTHING.value,
                                        action_info)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def process_agent_step(self, agent_id, step_info):
        action_type = step_info.pop("action_type")

        # 记录所有动作
        step_info["user_id"] = agent_id
        self.pl_utils._record_trace(agent_id, action_type, step_info)

        if action_type == ActionType.SIGN_UP.value:
            return await self.register_user(
                agent_id, step_info["username"], step_info.get("bio", ""))

        elif action_type == ActionType.CREATE_POST.value:
            ret_val = await self.create_post(agent_id, step_info["content"],
                                            step_info.get("media", ""))
            return ret_val

        elif action_type == ActionType.REPOST.value:
            if step_info.get("post_id") is None:
                return {"success": False, "message": "No post specified!"}
            ret_val = await self.repost_post(agent_id, step_info["post_id"])
            return ret_val

        # ... existing action processing ...

        elif action_type == ActionType.UPDATE_THINK.value:
            # 处理思考表更新动作
            return await self.process_update_think(agent_id, step_info)

        elif action_type == ActionType.DO_NOTHING.value:
            return {"success": True, "message": "Do nothing, successfully"}

    async def process_update_think(self, agent_id, step_info):
        """
        处理思考表更新动作

        参数:
        - agent_id: 智能体ID
        - step_info: 包含思考记录信息的字典，应包含以下字段:
            - step_number: 步骤编号
            - sub_step_number: 子步骤编号（可选，默认为0）
            - post_id: 相关帖子ID (可选)
            - action_name: 动作名称
            - content: 思考内容
            - reason: 思考原因
            - cognitive_state: 认知状态（情感、情绪等）包含viewpoint_1到viewpoint_6的观点支持级别

        返回:
        - 包含操作结果的字典
        """
        # 创建平台日志记录器
        platform_log = logging.getLogger(name="social.platform")
        platform_log.debug(f"process_update_think被调用: agent_id={agent_id}")

        # 提取必要参数
        step_number = step_info.get("step_number")
        sub_step_number = step_info.get("sub_step_number", 0)  # 默认子步骤为0
        post_id = step_info.get("post_id")
        action_name = step_info.get("action_name", "initial")  # 设置默认值为"initial"
        content = step_info.get("content", "")  # 设置默认内容
        reason = step_info.get("reason", "")  # 设置默认原因
        cognitive_state = step_info.get("cognitive_state")

        # 记录完整接收到的参数
        platform_log.debug(f"思考表更新接收到的参数: {step_info}")

        # 验证必要参数
        if None in [step_number, cognitive_state]:
            platform_log.warning(f"思考表更新参数不完整: step_number或cognitive_state缺失")
            return {"success": False, "error": "Missing required parameters (step_number or cognitive_state)"}

        # 确保action_name不为空
        if not action_name or action_name.strip() == "":
            action_name = "initial"
            platform_log.debug(f"action_name为空，使用默认值: {action_name}")

        # 确保认知状态格式正确
        if isinstance(cognitive_state, dict):
            # 检查是否需要增加opinions字段
            if "opinion" not in cognitive_state:
                # 收集现有的viewpoint字段
                opinion = {}
                for i in range(1, 7):
                    vp_key = f"viewpoint_{i}"
                    if vp_key in cognitive_state:
                        if isinstance(cognitive_state[vp_key], dict):
                            support_level = cognitive_state[vp_key].get("type_support_levels", "Indifferent")
                        else:
                            support_level = cognitive_state[vp_key]
                        opinion[vp_key] = support_level
                    else:
                        opinion[vp_key] = "Indifferent"

                # 确保每个viewpoint都有值
                for i in range(1, 7):
                    vp_key = f"viewpoint_{i}"
                    if vp_key not in opinion:
                        opinion[vp_key] = "Indifferent"

                # 更新认知状态中的观点
                cognitive_state["opinion"] = opinion
                platform_log.debug(f"添加了opinions字段到认知状态: {opinion}")

        # 调用现有的更新方法
        return await self.update_think_table(
            agent_id, step_number, sub_step_number, post_id, action_name, content, reason, cognitive_state)

    async def update_think(self, agent_id, step_info):
        """
        处理思考表更新动作

        参数:
        - agent_id: 智能体ID
        - step_info: 包含思考记录信息的字典，应包含以下字段:
            - step_number: 步骤编号
            - sub_step_number: 子步骤编号
            - post_id: 相关帖子ID (可选)
            - action_name: 动作名称
            - content: 思考内容
            - reason: 思考原因
            - cognitive_state: 认知状态（情感、情绪等）

        返回:
        - 包含操作结果的字典
        """
        #print(f"==DEBUG== update_think被调用: agent_id={agent_id}")
        #print(f"==DEBUG== 传入的step_info: {step_info}")
        return await self.process_update_think(agent_id, step_info)

    async def reply_to_comment(self, agent_id: int, reply_message: tuple):
        """回复评论

        此方法允许用户回复特定的评论，创建嵌套的讨论结构。

        参数:
            agent_id: 用户ID
            reply_message: 包含评论ID和回复内容的元组

        返回:
            dict: 包含操作结果的字典
        """
        comment_id, content = reply_message
        if self.recsys_type in [RecsysType.REDDIT, RecsysType.COGNITIVE]:
            current_time = self.sandbox_clock.time_transfer(
                datetime.now(), self.start_time)
        else:
            current_time = os.environ["SANDBOX_TIME"]
        try:
            user_id = agent_id

            # 获取原评论所属的帖子ID
            comment_query = "SELECT post_id FROM comment WHERE comment_id = ?"
            self.pl_utils._execute_db_command(comment_query, (comment_id,))
            result = self.db_cursor.fetchone()
            if not result:
                return {"success": False, "error": "Comment not found."}

            post_id = result[0]

            # 获取当前轮次信息
            try:
                current_step = int(os.environ.get("TIME_STAMP", 0))
            except (ValueError, TypeError):
                current_step = 0

            # 插入回复评论记录
            reply_insert_query = (
                "INSERT INTO comment (post_id, user_id, parent_comment_id, content, created_at, step_number) "
                "VALUES (?, ?, ?, ?, ?, ?)")
            self.pl_utils._execute_db_command(
                reply_insert_query,
                (post_id, user_id, comment_id, content, current_time, current_step),
                commit=True,
            )

            # 获取新插入的评论ID
            new_comment_id = self.db_cursor.lastrowid

            # 更新原评论的回复数
            update_replies_query = (
                "UPDATE comment SET num_replies = num_replies + 1 WHERE comment_id = ?")
            self.pl_utils._execute_db_command(
                update_replies_query, (comment_id,), commit=True)

            # 记录操作
            action_info = {
                "parent_comment_id": comment_id,
                "new_comment_id": new_comment_id,
                "content": content
            }
            self.pl_utils._record_trace(user_id, ActionType.REPLY_COMMENT.value,
                                    action_info, current_time)

            return {"success": True, "comment_id": new_comment_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def share_comment(self, agent_id: int, comment_id: int):
        """分享评论

        此方法允许用户分享特定的评论，增加评论的可见性。

        参数:
            agent_id: 用户ID
            comment_id: 要分享的评论ID

        返回:
            dict: 包含操作结果的字典
        """
        if self.recsys_type in [RecsysType.REDDIT, RecsysType.COGNITIVE]:
            current_time = self.sandbox_clock.time_transfer(
                datetime.now(), self.start_time)
        else:
            current_time = os.environ["SANDBOX_TIME"]
        try:
            user_id = agent_id

            # 检查评论是否存在
            comment_check_query = "SELECT * FROM comment WHERE comment_id = ?"
            self.pl_utils._execute_db_command(comment_check_query, (comment_id,))
            if not self.db_cursor.fetchone():
                return {"success": False, "error": "Comment not found."}

            # 获取当前轮次信息
            try:
                current_step = int(os.environ.get("TIME_STAMP", 0))
            except (ValueError, TypeError):
                current_step = 0

            # 插入评论分享记录
            share_insert_query = (
                "INSERT INTO comment_share (user_id, comment_id, created_at, step_number) "
                "VALUES (?, ?, ?, ?)")
            self.pl_utils._execute_db_command(
                share_insert_query, (user_id, comment_id, current_time, current_step),
                commit=True)

            # 获取新插入的分享ID
            share_id = self.db_cursor.lastrowid

            # 更新评论的分享数
            update_shares_query = (
                "UPDATE comment SET num_shares = num_shares + 1 WHERE comment_id = ?")
            self.pl_utils._execute_db_command(
                update_shares_query, (comment_id,), commit=True)

            # 记录操作
            action_info = {"comment_id": comment_id, "share_id": share_id}
            self.pl_utils._record_trace(user_id, ActionType.SHARE_COMMENT.value,
                                    action_info, current_time)

            return {"success": True, "share_id": share_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_trending_comments(self, agent_id: int, limit: int = 2):
        """获取热门评论

        此方法获取系统中的热门评论，基于点赞数排序。

        参数:
            agent_id: 用户ID
            limit: 要获取的评论数量上限，默认为2

        返回:
            dict: 包含热门评论的字典
        """
        if self.recsys_type in [RecsysType.REDDIT, RecsysType.COGNITIVE]:
            current_time = self.sandbox_clock.time_transfer(
                datetime.now(), self.start_time)
        else:
            current_time = os.environ["SANDBOX_TIME"]
        try:
            user_id = agent_id

            # 获取热门评论（基于点赞数）
            trending_query = """
                SELECT c.comment_id, c.post_id, c.user_id, c.parent_comment_id,
                       c.content, c.created_at, c.num_likes, c.num_dislikes,
                       c.num_replies, c.num_shares, p.content as post_content
                FROM comment c
                JOIN post p ON c.post_id = p.post_id
                ORDER BY c.num_likes DESC
                LIMIT ?
            """
            self.pl_utils._execute_db_command(trending_query, (limit,))
            results = self.db_cursor.fetchall()

            if not results:
                return {"success": False, "message": "No trending comments found."}

            # 格式化评论结果
            trending_comments = []
            for row in results:
                comment_id, post_id, comment_user_id, parent_comment_id, content, created_at, \
                num_likes, num_dislikes, num_replies, num_shares, post_content = row

                # 获取父评论内容（如果有）
                parent_content = None
                if parent_comment_id:
                    parent_query = "SELECT content FROM comment WHERE comment_id = ?"
                    self.pl_utils._execute_db_command(parent_query, (parent_comment_id,))
                    parent_result = self.db_cursor.fetchone()
                    if parent_result:
                        parent_content = parent_result[0]

                comment_dict = {
                    "comment_id": comment_id,
                    "post_id": post_id,
                    "user_id": comment_user_id,
                    "parent_comment_id": parent_comment_id,
                    "parent_content": parent_content,
                    "content": content,
                    "created_at": created_at,
                    "post_content": post_content,
                    "num_replies": num_replies,
                    "num_shares": num_shares
                }

                if self.show_score:
                    comment_dict["score"] = num_likes - num_dislikes
                else:
                    comment_dict["num_likes"] = num_likes
                    comment_dict["num_dislikes"] = num_dislikes

                trending_comments.append(comment_dict)

            # 记录操作
            action_info = {"trending_comments_count": len(trending_comments)}
            self.pl_utils._record_trace(user_id, ActionType.VIEW_TRENDING_COMMENTS.value,
                                    action_info, current_time)

            return {"success": True, "comments": trending_comments}
        except Exception as e:
            return {"success": False, "error": str(e)}
