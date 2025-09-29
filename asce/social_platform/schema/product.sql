-- 这是产品表的模式定义
-- 产品表(product)用于存储平台上的产品信息，包括产品ID、名称和销售数量
CREATE TABLE product (
    product_id INTEGER PRIMARY KEY,            -- 产品ID：整数类型，作为表的主键，用于唯一标识每个产品
    product_name TEXT,                         -- 产品名称：文本类型，存储产品的名称信息
    sales INTEGER DEFAULT 0                    -- 销售数量：整数类型，记录产品的销售数量，默认值为0，表示新添加的产品初始销售量为零
);
