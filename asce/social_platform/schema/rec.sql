-- 这是推荐表的模式定义
-- 推荐表(rec)用于存储系统为用户推荐的帖子信息，建立用户与推荐内容之间的关联
CREATE TABLE rec (
    user_id INTEGER,                           -- 用户ID：整数类型，标识接收推荐的用户，关联到user表
    post_id INTEGER,                           -- 帖子ID：整数类型，标识被推荐的帖子，关联到post表
    step_number INTEGER DEFAULT 0,             -- 轮次数：整数类型，记录推荐发生时的模拟轮次，默认为0（表示初始化阶段）
    PRIMARY KEY(user_id, post_id, step_number),-- 复合主键：由用户ID、帖子ID和轮次组成，允许记录每轮的推荐结果
    FOREIGN KEY(user_id) REFERENCES user(user_id),  -- 外键约束：确保user_id必须存在于user表中，维护数据完整性
    FOREIGN KEY(post_id) REFERENCES post(post_id)   -- 外键约束：确保post_id必须存在于post表中，维护数据完整性
);
