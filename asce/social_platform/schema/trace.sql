-- 这是跟踪表的模式定义
-- 跟踪表用于记录用户在平台上的各种活动和行为
CREATE TABLE trace (
    user_id INTEGER,                           -- 用户ID：整数类型，用于标识执行操作的用户，关联到user表
    created_at DATETIME,                       -- 创建时间：日期时间类型，记录操作发生的具体时间点，格式为年-月-日 时:分:秒
    action TEXT,                               -- 操作类型：文本类型，描述用户执行的具体操作，如"登录"、"发帖"、"点赞"等
    info TEXT,                                 -- 附加信息：文本类型，存储与操作相关的额外详细信息，可能包含操作对象ID或其他上下文
    step_number INTEGER DEFAULT 0,             -- 轮次数：整数类型，记录操作发生时的模拟轮次，默认为0（表示初始化阶段）
    PRIMARY KEY(user_id, created_at, action, info),  -- 复合主键：由四个字段组成，确保每条记录的唯一性，防止重复记录
    FOREIGN KEY(user_id) REFERENCES user(user_id)    -- 外键约束：确保user_id必须存在于user表中，维护数据完整性
);
