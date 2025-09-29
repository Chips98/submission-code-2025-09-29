-- 这是用户表的模式定义
-- 这是用户表的模式定义
CREATE TABLE user (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 用户ID：整数类型，作为表的主键，会自动增长，每添加一条记录会自动加1，确保每个用户有唯一标识
    agent_id INTEGER,                           -- 代理ID：整数类型，用于关联到系统中的AI代理或其他实体，便于识别用户类型或来源
    user_name TEXT,                             -- 用户名：文本类型，用户登录系统的唯一标识符，类似用户账号
    name TEXT,                                  -- 姓名：文本类型，用户在平台上展示的名称，可以更改，不同于用户名
    bio TEXT,                                   -- 个人简介：文本类型，用户可以自定义的个人描述信息，展示在个人资料页面
    created_at DATETIME,                        -- 创建时间：日期时间类型，记录用户账户创建的具体时间，格式为年-月-日 时:分:秒
    num_followings INTEGER DEFAULT 0,           -- 关注数量：整数类型，默认值为0，记录该用户关注了多少其他用户，当用户关注他人时此值增加
    num_followers INTEGER DEFAULT 0             -- 粉丝数量：整数类型，默认值为0，记录有多少用户关注了该用户，当被他人关注时此值增加
);
