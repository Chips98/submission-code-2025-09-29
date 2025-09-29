-- 这是用户思考表的模式定义
-- 用户思考表(think)用于记录用户或AI代理在每个步骤中的思考过程和内容
CREATE TABLE think (
    think_id INTEGER PRIMARY KEY AUTOINCREMENT,    -- 思考ID：整数类型，作为表的主键，自动增长，确保每条思考记录有唯一标识
    user_id INTEGER,                               -- 用户ID：整数类型，标识进行思考的用户，关联到user表的user_id
    step_number INTEGER,                           -- 步骤序号：整数类型，记录思考在用户行动序列中的顺序
    sub_step_number INTEGER DEFAULT 0,             -- 子步骤序号：整数类型，记录同一步骤内的多个行为，默认为0
    post_id INTEGER,                               -- 帖子ID：整数类型，与该思考相关的帖子ID
    action_name TEXT,                              -- 行动名称：文本类型，记录用户执行的行动名称
    content TEXT,                                  -- 思考内容：文本类型，存储用户或AI代理的具体思考内容
    reason TEXT,                                   -- 推理原因：文本类型，记录智能体的推理过程
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
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP, -- 创建时间：日期时间类型，记录思考发生的具体时间点，默认为当前时间戳
    FOREIGN KEY(user_id) REFERENCES user(user_id),  -- 外键约束：确保user_id必须存在于user表中的user_id字段，维护数据完整性
    FOREIGN KEY(post_id) REFERENCES post(post_id)   -- 外键约束：确保post_id必须存在于post表中的post_id字段
);

-- 创建联合主键索引（修改为包含sub_step_number）
CREATE UNIQUE INDEX idx_think_user_step ON think(user_id, step_number, sub_step_number);

-- 创建其他索引以提高查询性能
CREATE INDEX idx_think_user_id ON think(user_id);
CREATE INDEX idx_think_created_at ON think(created_at);
CREATE INDEX idx_think_step_number ON think(step_number);
CREATE INDEX idx_think_post_id ON think(post_id);
CREATE INDEX idx_think_action_name ON think(action_name);

-- 添加注释
PRAGMA table_info(think); 