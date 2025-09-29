-- 这是不喜欢表的模式定义
-- 不喜欢表(dislike)用于存储用户对帖子的不喜欢信息，记录谁对哪篇帖子表示不喜欢以及操作的时间
CREATE TABLE dislike (
    dislike_id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 不喜欢ID：整数类型，作为表的主键，会自动增长，每添加一条记录会自动加1，确保每个不喜欢操作有唯一标识
    user_id INTEGER,                               -- 用户ID：整数类型，标识进行不喜欢操作的用户，关联到user表的user_id
    post_id INTEGER,                               -- 帖子ID：整数类型，标识被不喜欢的帖子，关联到tweet表的post_id
    created_at DATETIME,                           -- 创建时间：日期时间类型，记录不喜欢操作发生的具体时间点，格式为年-月-日 时:分:秒
    step_number INTEGER DEFAULT 0,                 -- 轮次数：整数类型，记录不喜欢操作发生时的模拟轮次，默认为0（表示初始化阶段）
    FOREIGN KEY(user_id) REFERENCES user(user_id),    -- 外键约束：确保user_id必须存在于user表中的user_id字段，维护数据完整性
    FOREIGN KEY(post_id) REFERENCES post(post_id)     -- 外键约束：确保post_id必须存在于post表中的post_id字段，维护数据完整性
);
