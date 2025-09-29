-- This is the schema definition for the comment_dislike table
CREATE TABLE comment_dislike (
    comment_dislike_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    comment_id INTEGER,
    created_at DATETIME,
    step_number INTEGER DEFAULT 0,             -- 轮次数：整数类型，记录不喜欢评论操作发生时的模拟轮次，默认为0（表示初始化阶段）
    FOREIGN KEY(user_id) REFERENCES user(user_id),
    FOREIGN KEY(comment_id) REFERENCES comment(comment_id)
);
