-- 这是屏蔽表的模式定义
-- 屏蔽表(mute)用于存储用户之间的屏蔽关系，记录谁屏蔽了谁以及屏蔽的时间
CREATE TABLE mute (
    mute_id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 屏蔽ID：整数类型，作为表的主键，会自动增长，每添加一条记录会自动加1，确保每个屏蔽关系有唯一标识
    muter_id INTEGER,                           -- 屏蔽者ID：整数类型，标识执行屏蔽操作的用户，关联到user表的user_id
    mutee_id INTEGER,                           -- 被屏蔽者ID：整数类型，标识被屏蔽的用户，关联到user表的user_id
    created_at DATETIME,                        -- 创建时间：日期时间类型，记录屏蔽关系建立的具体时间点，格式为年-月-日 时:分:秒
    step_number INTEGER DEFAULT 0,              -- 轮次数：整数类型，记录屏蔽操作发生时的模拟轮次，默认为0（表示初始化阶段）
    FOREIGN KEY(muter_id) REFERENCES user(user_id),  -- 外键约束：确保muter_id必须存在于user表中的user_id字段，维护数据完整性
    FOREIGN KEY(mutee_id) REFERENCES user(user_id)   -- 外键约束：确保mutee_id必须存在于user表中的user_id字段，维护数据完整性
);
