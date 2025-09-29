# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import asyncio
import uuid


class AsyncSafeDict:
    # 异步安全字典类，用于在异步环境中安全地操作字典数据
    def __init__(self):
        # 初始化方法，创建一个空字典和一个异步锁
        self.dict = {}  # 创建一个空字典用于存储数据
        self.lock = asyncio.Lock()  # 创建一个异步锁对象，用于确保字典操作的线程安全

    async def put(self, key, value):
        # 异步方法，用于向字典中添加或更新键值对
        async with self.lock:  # 使用异步上下文管理器获取锁，确保同一时间只有一个协程可以修改字典
            self.dict[key] = value  # 将键值对存入字典

    async def get(self, key, default=None):
        # 异步方法，用于从字典中获取指定键的值，如果键不存在则返回默认值
        async with self.lock:  # 使用锁确保读取操作的安全
            return self.dict.get(key, default)  # 使用字典的get方法获取值，如果键不存在则返回default

    async def pop(self, key, default=None):
        # 异步方法，用于从字典中移除指定键并返回其值，如果键不存在则返回默认值
        async with self.lock:  # 使用锁确保操作安全
            return self.dict.pop(key, default)  # 使用字典的pop方法移除键并返回其值

    async def keys(self):
        # 异步方法，用于获取字典中所有键的列表
        async with self.lock:  # 使用锁确保操作安全
            return list(self.dict.keys())  # 将字典的键转换为列表并返回


class Channel:
    # 通道类，用于异步消息传递
    def __init__(self):
        # 初始化方法，创建接收队列和发送字典
        self.receive_queue = asyncio.Queue()  # 创建一个异步队列用于存储接收到的消息
        # 使用前面定义的异步安全字典来存储要发送的消息
        self.send_dict = AsyncSafeDict()
        # 平台实例引用，初始为None
        self.platform = None

    def set_platform(self, platform):
        """
        设置平台引用
        
        参数:
            platform: 平台实例
        """
        self.platform = platform
        
    async def receive_from(self):
        # 异步方法，用于从接收队列中获取消息
        message = await self.receive_queue.get()  # 等待并获取队列中的下一个消息
        # await关键字表示这是一个异步操作，会暂停当前协程直到操作完成
        return message  # 返回获取到的消息

    async def send_to(self, message):
        # 异步方法，用于发送消息
        # message_id是消息的第一个元素
        message_id = message[0]  # 从消息元组中提取消息ID
        await self.send_dict.put(message_id, message)  # 将消息存入发送字典，以消息ID为键

    async def write_to_receive_queue(self, action_info):
        # 异步方法，用于将操作信息写入接收队列
        message_id = str(uuid.uuid4())  # 生成一个唯一的UUID作为消息ID
        # uuid.uuid4()生成一个随机UUID，str()将其转换为字符串
        await self.receive_queue.put((message_id, action_info))  # 将消息ID和操作信息作为元组放入接收队列
        return message_id  # 返回生成的消息ID

    async def read_from_send_queue(self, message_id):
        # 异步方法，用于从发送字典中读取指定ID的消息
        while True:  # 无限循环，直到找到消息或被中断
            if message_id in await self.send_dict.keys():  # 检查消息ID是否在发送字典的键中
                # 尝试获取并移除消息
                message = await self.send_dict.pop(message_id, None)  # 从发送字典中弹出消息
                if message:  # 如果成功获取到消息
                    return message  # 返回找到的消息
            # 暂时挂起以避免紧密循环
            await asyncio.sleep(0.1)  # 暂停0.1秒，减轻CPU负担
            # asyncio.sleep()是一个异步睡眠函数，不会阻塞整个程序
