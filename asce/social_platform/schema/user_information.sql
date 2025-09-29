-- 这是用户信息表的模式定义
-- 用户信息表(user_information)用于记录用户的详细信息
CREATE TABLE user_information (
    info_id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 信息ID：整数类型，作为表的主键，自动增长
    user_id INTEGER,                            -- 用户ID：整数类型，关联到user表的user_id
    persona TEXT,                               -- 用户画像描述：文本类型
    age TEXT,                                   -- 年龄段：文本类型
    gender TEXT,                                -- 性别：文本类型
    mbti TEXT,                                  -- MBTI人格类型：文本类型
    country TEXT,                               -- 国家：文本类型
    profession TEXT,                            -- 职业：文本类型
    interested_topics TEXT,                     -- 用户感兴趣话题：文本类型(JSON格式)
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP, -- 创建时间：日期时间类型，默认为当前时间戳
    FOREIGN KEY(user_id) REFERENCES user(user_id) -- 外键约束：确保user_id必须存在于user表中
);

-- 创建索引以提高查询性能
CREATE UNIQUE INDEX idx_user_information_user_id ON user_information(user_id);
CREATE INDEX idx_user_information_created_at ON user_information(created_at);

-- 添加注释
PRAGMA table_info(user_information); 