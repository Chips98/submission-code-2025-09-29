-- 这是评论表的模式定义
-- 评论表(comment)用于存储用户对帖子的评论信息，记录谁对哪篇帖子发表了什么评论以及评论的时间和互动数据
CREATE TABLE comment (
    comment_id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 评论ID：整数类型，作为表的主键，会自动增长，每添加一条记录会自动加1，确保每个评论有唯一标识
    post_id INTEGER,                               -- 帖子ID：整数类型，标识被评论的帖子，关联到post表的post_id
    user_id INTEGER,                               -- 用户ID：整数类型，标识发表评论的用户，关联到user表的user_id
    parent_comment_id INTEGER DEFAULT NULL,        -- 父评论ID：整数类型，标识回复的父评论，如果为NULL则表示是顶级评论
    content TEXT,                                  -- 内容：文本类型，存储评论的具体内容
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP, -- 创建时间：日期时间类型，记录评论发布的具体时间点，默认为当前时间戳，即评论创建时的系统时间
    step_number INTEGER DEFAULT 0,                 -- 轮次数：整数类型，记录评论发布时的模拟轮次，默认为0（表示初始化阶段）
    num_likes INTEGER DEFAULT 0,                   -- 点赞数：整数类型，记录该评论获得的点赞数量，默认值为0，表示新评论初始没有点赞
    num_dislikes INTEGER DEFAULT 0,                -- 不喜欢数：整数类型，记录该评论获得的不喜欢数量，默认值为0，表示新评论初始没有不喜欢
    num_replies INTEGER DEFAULT 0,                 -- 回复数：整数类型，记录该评论获得的回复数量，默认值为0，表示新评论初始没有回复
    num_shares INTEGER DEFAULT 0,                  -- 分享数：整数类型，记录该评论被分享的次数，默认值为0，表示新评论初始没有被分享
    FOREIGN KEY(post_id) REFERENCES post(post_id), -- 外键约束：确保post_id必须存在于post表中的post_id字段，维护数据完整性
    FOREIGN KEY(user_id) REFERENCES user(user_id), -- 外键约束：确保user_id必须存在于user表中的user_id字段，维护数据完整性
    FOREIGN KEY(parent_comment_id) REFERENCES comment(comment_id) -- 外键约束：确保parent_comment_id必须存在于comment表中的comment_id字段，维护数据完整性
);
