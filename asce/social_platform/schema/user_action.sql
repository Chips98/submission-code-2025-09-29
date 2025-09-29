-- 这是用户行为表的模式定义
-- 用户行为表(user_action)用于记录用户在每个时间步的行为、理由及认知状态
CREATE TABLE user_action (
    action_id INTEGER PRIMARY KEY AUTOINCREMENT,   -- 行为ID：整数类型，作为表的主键，自动增长，确保每条记录有唯一标识
    user_id INTEGER,                               -- 用户ID：整数类型，标识执行行为的用户
    num_steps INTEGER,                             -- 时间步：整数类型，记录当前的时间步
    post_id INTEGER,                               -- 帖子ID：整数类型，与该行为相关的帖子ID
    action TEXT,                                   -- 行为：文本类型，记录用户执行的行为类型
    reason TEXT,                                   -- 理由：文本类型，记录执行该行为的原因
    info TEXT,                                     -- 附加信息：文本类型，存储与操作相关的额外详细信息，JSON格式
    mood_type TEXT,                           -- 情感类型：文本类型，记录智能体的情感类型
    mood_value TEXT,                          -- 情感值：文本类型，记录智能体的情感具体值
    emotion_type TEXT,                             -- 情绪类型：文本类型，记录智能体的情绪类型
    emotion_value TEXT,                            -- 情绪值：文本类型，记录智能体的情绪具体值
    stance_type TEXT,                              -- 立场类型：文本类型，记录智能体的立场类型
    stance_value TEXT,                             -- 立场值：文本类型，记录智能体的立场具体值
    thinking_type TEXT,                           -- 认知类型：文本类型，记录智能体的认知类型
    thinking_value TEXT,                          -- 认知值：文本类型，记录智能体的认知具体值
    intention_type TEXT,                           -- 意图类型：文本类型，记录智能体的意图类型
    intention_value TEXT,                          -- 意图值：文本类型，记录智能体的意图具体值
    viewpoint_1 TEXT,                              -- 观点1支持级别：文本类型，记录智能体对观点1的支持级别
    viewpoint_2 TEXT,                              -- 观点2支持级别：文本类型，记录智能体对观点2的支持级别
    viewpoint_3 TEXT,                              -- 观点3支持级别：文本类型，记录智能体对观点3的支持级别
    viewpoint_4 TEXT,                              -- 观点4支持级别：文本类型，记录智能体对观点4的支持级别
    viewpoint_5 TEXT,                              -- 观点5支持级别：文本类型，记录智能体对观点5的支持级别
    viewpoint_6 TEXT,                              -- 观点6支持级别：文本类型，记录智能体对观点6的支持级别
    is_active TEXT,                                -- 是否激活：文本类型，标记用户在该轮次是否被激活
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP, -- 创建时间：日期时间类型，记录行为发生的具体时间点，默认为当前时间戳
    FOREIGN KEY(user_id) REFERENCES user(user_id), -- 外键约束：确保user_id必须存在于user表中
    FOREIGN KEY(post_id) REFERENCES post(post_id)  -- 外键约束：确保post_id必须存在于post表中
);

-- 创建联合主键索引
CREATE UNIQUE INDEX idx_user_action_user_step ON user_action(user_id, num_steps);

-- 创建其他索引以提高查询性能
CREATE INDEX idx_user_action_user_id ON user_action(user_id);
CREATE INDEX idx_user_action_num_steps ON user_action(num_steps);
CREATE INDEX idx_user_action_post_id ON user_action(post_id);
CREATE INDEX idx_user_action_action ON user_action(action);
CREATE INDEX idx_user_action_info ON user_action(info);
CREATE INDEX idx_user_action_is_active ON user_action(is_active);
CREATE INDEX idx_user_action_created_at ON user_action(created_at);

-- 添加注释
PRAGMA table_info(user_action); 