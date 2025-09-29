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
# 从未来导入annotations特性，使得类型注解中可以引用自身类型
from __future__ import annotations

# 导入操作系统相关功能的模块
import os
# 导入处理文件路径的模块，并将其重命名为osp以便简化使用
import os.path as osp
# 导入SQLite数据库操作模块
import sqlite3
# 从typing模块导入类型提示相关的类，用于函数参数和返回值的类型注解
#from typing import Any, Dict, List
import sys
sys.path = [p for p in sys.path if 'Oasis-main/asce/social_platform' not in p]
from typing import Any, Dict, List
# 定义数据库模式文件所在的目录路径
SCHEMA_DIR = "social_platform/schema"
# 定义数据库文件存储的目录名
DB_DIR = "db"
# 定义数据库文件的名称
DB_NAME = "social_media.db"

# 以下是各个数据表的SQL模式文件名称
# 用户表的SQL模式文件
USER_SCHEMA_SQL = "user.sql"
# 帖子表的SQL模式文件
POST_SCHEMA_SQL = "post.sql"
# 关注关系表的SQL模式文件
FOLLOW_SCHEMA_SQL = "follow.sql"
# 静音/屏蔽用户表的SQL模式文件
MUTE_SCHEMA_SQL = "mute.sql"
# 点赞表的SQL模式文件
LIKE_SCHEMA_SQL = "like.sql"
# 点踩表的SQL模式文件
DISLIKE_SCHEMA_SQL = "dislike.sql"
# 用户行为追踪表的SQL模式文件
TRACE_SCHEMA_SQL = "trace.sql"
# 推荐内容表的SQL模式文件
REC_SCHEMA_SQL = "rec.sql"
# 评论表的SQL模式文件
COMMENT_SCHEMA_SQL = "comment.sql"
# 评论点赞表的SQL模式文件
COMMENT_LIKE_SCHEMA_SQL = "comment_like.sql"
# 评论点踩表的SQL模式文件
COMMENT_DISLIKE_SCHEMA_SQL = "comment_dislike.sql"
# 产品表的SQL模式文件
PRODUCT_SCHEMA_SQL = "product.sql"
# 用户思考表的SQL模式文件
THINK_SCHEMA_SQL = "think.sql"


# 定义一个集合，包含所有数据表的名称
# 使用集合（set）数据结构可以确保元素的唯一性，并且支持高效的成员检测
# 注意：这里有几个表名包含了.sql后缀，可能是代码中的错误
TABLE_NAMES = {
    "user",
    "post",
    "follow",
    "mute",
    "like",
    "dislike",
    "trace",
    "rec",
    "comment.sql",
    "comment_like.sql",
    "comment_dislike.sql",
    "product.sql",
    "think.sql",
}


def get_db_path() -> str:
    # 获取当前文件的绝对路径
    curr_file_path = osp.abspath(__file__)
    # 获取当前文件的父目录
    parent_dir = osp.dirname(osp.dirname(curr_file_path))
    # 获取数据库文件存储的目录路径
    db_dir = osp.join(parent_dir, DB_DIR)
    # 确保数据库文件存储的目录存在
    os.makedirs(db_dir, exist_ok=True)
    # 获取数据库文件的完整路径
    db_path = osp.join(db_dir, DB_NAME)
    return db_path


def get_schema_dir_path() -> str:
    # 获取当前文件的绝对路径
    curr_file_path = osp.abspath(__file__)
    # 获取当前文件的父目录
    parent_dir = osp.dirname(osp.dirname(curr_file_path))
    # 获取数据库模式文件所在的目录路径
    schema_dir = osp.join(parent_dir, SCHEMA_DIR)
    return schema_dir


def create_db(db_path: str | None = None):
    r"""Create the database if it does not exist. A :obj:`twitter.db`
    file will be automatically created  in the :obj:`data` directory.
    """
    # 获取数据库模式文件所在的目录路径
    schema_dir = get_schema_dir_path()
    # 如果数据库文件路径未提供，则使用默认路径
    if db_path is None:
        db_path = get_db_path()

    # 连接到数据库：
    #print("db_path", db_path)  # 打印数据库路径，便于调试
    conn = sqlite3.connect(db_path)  # 创建与数据库的连接
    cursor = conn.cursor()  # 创建一个游标对象，用于执行SQL命令

    try:
        # 读取并执行用户表的SQL脚本：
        user_sql_path = osp.join(schema_dir, USER_SCHEMA_SQL)  # 构建用户表SQL文件的完整路径
        with open(user_sql_path, "r") as sql_file:  # 以只读模式打开SQL文件
            user_sql_script = sql_file.read()  # 读取文件内容到变量中
        cursor.executescript(user_sql_script)  # 执行整个SQL脚本，可能包含多条SQL语句

        # 读取并执行帖子表的SQL脚本：
        post_sql_path = osp.join(schema_dir, POST_SCHEMA_SQL)  # 构建帖子表SQL文件的完整路径
        with open(post_sql_path, "r") as sql_file:  # 以只读模式打开SQL文件
            post_sql_script = sql_file.read()  # 读取文件内容
        cursor.executescript(post_sql_script)  # 执行SQL脚本创建帖子表

        # 读取并执行关注关系表的SQL脚本：
        follow_sql_path = osp.join(schema_dir, FOLLOW_SCHEMA_SQL)  # 构建关注表SQL文件的完整路径
        with open(follow_sql_path, "r") as sql_file:  # 打开文件
            follow_sql_script = sql_file.read()  # 读取内容
        cursor.executescript(follow_sql_script)  # 执行脚本

        # 读取并执行静音/屏蔽用户表的SQL脚本：
        mute_sql_path = osp.join(schema_dir, MUTE_SCHEMA_SQL)  # 构建静音表SQL文件的完整路径
        with open(mute_sql_path, "r") as sql_file:  # 打开文件
            mute_sql_script = sql_file.read()  # 读取内容
        cursor.executescript(mute_sql_script)  # 执行脚本

        # 读取并执行点赞表的SQL脚本：
        like_sql_path = osp.join(schema_dir, LIKE_SCHEMA_SQL)  # 构建点赞表SQL文件的完整路径
        with open(like_sql_path, "r") as sql_file:  # 打开文件
            like_sql_script = sql_file.read()  # 读取内容
        cursor.executescript(like_sql_script)  # 执行脚本

        # 读取并执行点踩表的SQL脚本：
        dislike_sql_path = osp.join(schema_dir, DISLIKE_SCHEMA_SQL)  # 构建点踩表SQL文件的完整路径
        with open(dislike_sql_path, "r") as sql_file:  # 打开文件
            dislike_sql_script = sql_file.read()  # 读取内容
        cursor.executescript(dislike_sql_script)  # 执行脚本

        # 读取并执行用户行为追踪表的SQL脚本：
        trace_sql_path = osp.join(schema_dir, TRACE_SCHEMA_SQL)  # 构建追踪表SQL文件的完整路径
        with open(trace_sql_path, "r") as sql_file:  # 打开文件
            trace_sql_script = sql_file.read()  # 读取内容
        cursor.executescript(trace_sql_script)  # 执行脚本

        # 读取并执行推荐内容表的SQL脚本：
        rec_sql_path = osp.join(schema_dir, REC_SCHEMA_SQL)  # 构建推荐表SQL文件的完整路径
        with open(rec_sql_path, "r") as sql_file:  # 打开文件
            rec_sql_script = sql_file.read()  # 读取内容
        cursor.executescript(rec_sql_script)  # 执行脚本

        # 读取并执行评论表的SQL脚本：
        comment_sql_path = osp.join(schema_dir, COMMENT_SCHEMA_SQL)  # 构建评论表SQL文件的完整路径
        with open(comment_sql_path, "r") as sql_file:  # 打开文件
            comment_sql_script = sql_file.read()  # 读取内容
        cursor.executescript(comment_sql_script)  # 执行脚本

        # 读取并执行评论点赞表的SQL脚本：
        comment_like_sql_path = osp.join(schema_dir, COMMENT_LIKE_SCHEMA_SQL)  # 构建评论点赞表SQL文件的完整路径
        with open(comment_like_sql_path, "r") as sql_file:  # 打开文件
            comment_like_sql_script = sql_file.read()  # 读取内容
        cursor.executescript(comment_like_sql_script)  # 执行脚本

        # 读取并执行评论点踩表的SQL脚本：
        comment_dislike_sql_path = osp.join(schema_dir,
                                            COMMENT_DISLIKE_SCHEMA_SQL)  # 构建评论点踩表SQL文件的完整路径
        with open(comment_dislike_sql_path, "r") as sql_file:  # 打开文件
            comment_dislike_sql_script = sql_file.read()  # 读取内容
        cursor.executescript(comment_dislike_sql_script)  # 执行脚本

        # 读取并执行产品表的SQL脚本：
        product_sql_path = osp.join(schema_dir, PRODUCT_SCHEMA_SQL)  # 构建产品表SQL文件的完整路径
        with open(product_sql_path, "r") as sql_file:  # 打开文件
            product_sql_script = sql_file.read()  # 读取内容
        cursor.executescript(product_sql_script)  # 执行脚本

        # 读取并执行用户思考表的SQL脚本：
        think_sql_path = osp.join(schema_dir, THINK_SCHEMA_SQL)  # 构建思考表SQL文件的完整路径
        with open(think_sql_path, "r") as sql_file:  # 打开文件
            think_sql_script = sql_file.read()  # 读取内容
        cursor.executescript(think_sql_script)  # 执行脚本
        

        # 提交更改：
        conn.commit()  # 将所有更改永久保存到数据库文件中
        # 注意：在SQLite中，如果不调用commit()，所有更改在连接关闭时会被丢弃

    except sqlite3.Error as e:  # 捕获SQLite操作过程中可能发生的任何错误
        print(f"An error occurred while creating tables: {e}")  # 打印错误信息
        # 这里使用了f-string格式化，是Python 3.6+的新特性，可以在字符串中直接嵌入变量

    return conn, cursor  # 返回数据库连接和游标对象，供调用者使用
    # 这个函数的主要作用是初始化数据库，创建所有必要的表结构
    # 它通过读取预定义的SQL脚本文件来创建各种表，确保数据库结构的一致性


def print_db_tables_summary():
    # 连接到SQLite数据库
    db_path = get_db_path()  # 获取数据库文件路径
    conn = sqlite3.connect(db_path)  # 创建与数据库的连接
    cursor = conn.cursor()  # 创建一个游标对象，用于执行SQL命令

    # 获取数据库中所有表的列表
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")  # 执行SQL查询语句，从sqlite_master系统表中获取所有表名
    tables = cursor.fetchall()  # 获取查询结果的所有行，返回一个列表，每个元素是一个元组

    # 打印每个表的摘要信息
    for table in tables:  # 遍历所有表
        table_name = table[0]  # 获取表名（表名存储在元组的第一个位置）
        if table_name not in TABLE_NAMES:  # 如果表名不在预定义的TABLE_NAMES列表中，则跳过
            continue
        print(f"Table: {table_name}")  # 打印表名（f-string是Python 3.6+的字符串格式化方法）

        # 获取表的结构信息
        cursor.execute(f"PRAGMA table_info({table_name})")  # 使用SQLite的PRAGMA命令获取表的列信息
        columns = cursor.fetchall()  # 获取所有列信息
        column_names = [column[1] for column in columns]  # 使用列表推导式提取每列的名称（列名在结果的第二个位置）
        print("- Columns:", column_names)  # 打印列名列表

        # 获取并打印外键信息
        cursor.execute(f"PRAGMA foreign_key_list({table_name})")  # 使用SQLite的PRAGMA命令获取表的外键信息
        foreign_keys = cursor.fetchall()  # 获取所有外键信息
        if foreign_keys:  # 如果存在外键
            print("- Foreign Keys:")  # 打印外键标题
            for fk in foreign_keys:  # 遍历每个外键
                print(f"    {fk[2]} references {fk[3]}({fk[4]}) on update "
                      f"{fk[5]} on delete {fk[6]}")  # 打印外键详细信息，包括引用的列、表和更新/删除规则
        else:
            print("  No foreign keys.")  # 如果没有外键，打印提示信息

        # 打印表中的前几行数据
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")  # 执行SQL查询，获取表中最多5行数据
        rows = cursor.fetchall()  # 获取查询结果
        for row in rows:  # 遍历每一行
            print(row)  # 打印行数据（以元组形式）
        print()  # 添加一个空行，使不同表之间的输出更易读

    # 关闭数据库连接
    conn.close()  # 关闭数据库连接，释放资源

# 这个函数的作用是打印数据库中所有表的摘要信息，包括表结构、外键关系和示例数据。
# 它首先连接到数据库，然后获取所有表的列表，对于每个表，它会：
# 1. 打印表名
# 2. 获取并打印表的列名
# 3. 获取并打印表的外键关系
# 4. 获取并打印表中的前5行数据
# 这个函数对于调试和了解数据库结构非常有用。


def fetch_table_from_db(cursor: sqlite3.Cursor,
                        table_name: str) -> List[Dict[str, Any]]:
    # 从数据库中获取指定表的所有数据
    cursor.execute(f"SELECT * FROM {table_name}")  # 执行SQL查询，获取表中所有数据
    # 获取查询结果的列名，cursor.description返回查询结果的元数据，每个元素的第一个值是列名
    columns = [description[0] for description in cursor.description]  # 使用列表推导式提取所有列名
    # 将查询结果转换为字典列表，每行数据变成一个字典，键是列名，值是对应的数据
    # zip()函数将columns和row中的元素一一对应，dict()将这些键值对转换为字典
    data_dicts = [dict(zip(columns, row)) for row in cursor.fetchall()]  # 使用列表推导式创建字典列表
    return data_dicts  # 返回包含所有行数据的字典列表


def fetch_rec_table_as_matrix(cursor: sqlite3.Cursor) -> List[List[int]]:
    # 将推荐表转换为矩阵形式
    # 首先，查询用户表中的所有用户ID，假设它们从1开始且连续
    cursor.execute("SELECT user_id FROM user ORDER BY user_id")  # 执行SQL查询，按用户ID排序获取所有用户ID
    # 使用列表推导式提取查询结果中的用户ID
    user_ids = [row[0] for row in cursor.fetchall()]  # row[0]表示结果中的第一个字段，即user_id

    # 然后，查询推荐表中的所有记录
    cursor.execute(
        "SELECT user_id, post_id FROM rec ORDER BY user_id, post_id")  # 执行SQL查询，获取推荐表中的用户ID和帖子ID
    rec_rows = cursor.fetchall()  # 获取所有查询结果
    # 初始化一个字典，为每个用户ID分配一个空列表
    # 字典推导式：创建一个字典，键是user_ids中的每个用户ID，值是空列表
    user_posts = {user_id: [] for user_id in user_ids}  
    # 用查询到的记录填充字典
    for user_id, post_id in rec_rows:  # 遍历查询结果中的每一行
        if user_id in user_posts:  # 如果用户ID在字典中存在
            user_posts[user_id].append(post_id)  # 将帖子ID添加到对应用户的列表中
    # 将字典转换为矩阵形式
    # 列表推导式：按照user_ids的顺序，从user_posts字典中提取每个用户的帖子列表
    matrix = [user_posts[user_id] for user_id in user_ids]  
    return matrix  # 返回矩阵形式的数据


def insert_matrix_into_rec_table(cursor: sqlite3.Cursor,
                                 matrix: List[List[int]]) -> None:
    # 将矩阵形式的数据插入到推荐表中
    # 遍历矩阵，从索引1开始（跳过索引0的占位符）
    # enumerate()函数返回索引和值的对，start=1表示索引从1开始计数
    for user_id, post_ids in enumerate(matrix, start=1):  
        # 调整为从1开始计数，user_id是当前索引，post_ids是对应的帖子ID列表
        for post_id in post_ids:  # 遍历当前用户的所有帖子ID
            # 将每个用户ID和帖子ID的组合插入到推荐表中
            # 使用参数化查询防止SQL注入，(?, ?)是占位符，(user_id, post_id)是要插入的值
            cursor.execute("INSERT INTO rec (user_id, post_id) VALUES (?, ?)",
                           (user_id, post_id))


if __name__ == "__main__":
    create_db()
    print_db_tables_summary()
