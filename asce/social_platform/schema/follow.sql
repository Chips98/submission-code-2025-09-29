-- 这是关注表的模式定义
-- 关注表(follow)用于存储用户之间的关注关系，记录谁关注了谁以及关注的时间
CREATE TABLE follow (
    follow_id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 关注ID：整数类型，作为表的主键，会自动增长，每添加一条记录会自动加1，确保每个关注关系有唯一标识
    follower_id INTEGER,                          -- 关注者ID：整数类型，标识执行关注操作的用户，关联到user表的user_id
    followee_id INTEGER,                          -- 被关注者ID：整数类型，标识被关注的用户，关联到user表的user_id
    created_at DATETIME,                          -- 创建时间：日期时间类型，记录关注关系建立的具体时间点，格式为年-月-日 时:分:秒
    step_number INTEGER DEFAULT 0,                -- 轮次数：整数类型，记录关注关系创建时的模拟轮次，默认为0（表示初始化阶段）
    FOREIGN KEY(follower_id) REFERENCES user(user_id),  -- 外键约束：确保follower_id必须存在于user表中的user_id字段，维护数据完整性
    FOREIGN KEY(followee_id) REFERENCES user(user_id)   -- 外键约束：确保followee_id必须存在于user表中的user_id字段，维护数据完整性
);
