-- 这是评论分享表的模式定义
-- 评论分享表(comment_share)用于存储用户分享评论的记录，记录谁分享了哪条评论以及分享的时间
CREATE TABLE comment_share (
    comment_share_id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 评论分享ID：整数类型，作为表的主键，会自动增长，每添加一条记录会自动加1，确保每个分享记录有唯一标识
    user_id INTEGER,                                     -- 用户ID：整数类型，标识分享评论的用户，关联到user表的user_id
    comment_id INTEGER,                                  -- 评论ID：整数类型，标识被分享的评论，关联到comment表的comment_id
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,       -- 创建时间：日期时间类型，记录分享的具体时间点，默认为当前时间戳，即分享创建时的系统时间
    step_number INTEGER DEFAULT 0,                       -- 轮次数：整数类型，记录分享发生时的模拟轮次，默认为0（表示初始化阶段）
    FOREIGN KEY(user_id) REFERENCES user(user_id),       -- 外键约束：确保user_id必须存在于user表中的user_id字段，维护数据完整性
    FOREIGN KEY(comment_id) REFERENCES comment(comment_id) -- 外键约束：确保comment_id必须存在于comment表中的comment_id字段，维护数据完整性
);
