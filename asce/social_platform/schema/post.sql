-- 这是帖子表的模式定义
-- 可以考虑添加图片、位置等功能？
CREATE TABLE post (
    post_id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 帖子ID：整数类型，作为表的主键，会自动增长，每添加一条记录会自动加1，确保每个帖子有唯一标识
    user_id INTEGER,                            -- 用户ID：整数类型，标识发布帖子的用户，关联到user表
    original_post_id INTEGER,                   -- 原始帖子ID：整数类型，如果这是一个原创帖子则为NULL，用于标识转发或引用的原始帖子
    content TEXT DEFAULT '',                    -- 内容：文本类型，存储帖子的主要内容，对于初始帖子默认为空字符串
    quote_content TEXT,                         -- 引用内容：文本类型，如果这是原创帖子或纯转发则为NULL，存储引用时添加的评论内容
    created_at DATETIME,                        -- 创建时间：日期时间类型，记录帖子发布的具体时间点，格式为年-月-日 时:分:秒
    step_number INTEGER DEFAULT 0,              -- 轮次数：整数类型，记录帖子发布时的模拟轮次，默认为0（表示初始化阶段）
    num_likes INTEGER DEFAULT 0,                -- 点赞数：整数类型，默认值为0，记录帖子获得的点赞总数
    num_dislikes INTEGER DEFAULT 0,             -- 踩数：整数类型，默认值为0，记录帖子获得的踩（不喜欢）总数
    num_shares INTEGER DEFAULT 0,               -- 分享数：整数类型，默认值为0，记录帖子被分享的总次数（等于转发数加引用数）
    FOREIGN KEY(user_id) REFERENCES user(user_id),           -- 外键约束：确保user_id必须存在于user表中，维护数据完整性
    FOREIGN KEY(original_post_id) REFERENCES post(post_id)   -- 外键约束：确保original_post_id必须存在于post表中，维护数据完整性
);
