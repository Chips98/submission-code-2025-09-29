# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# 代理关系图可视化模块

import os
import traceback
from datetime import datetime

def visualize_agent_graph(agent_graph, timestamp=None, prefix="agent_graph"):
    """
    可视化代理关系图，并保存为图像文件
    
    Args:
        agent_graph: 代理关系图对象
        timestamp: 可选的时间戳标识（如果为None则使用当前时间）
        prefix: 输出文件名前缀
    
    Returns:
        str: 保存的图像文件路径，如果失败则返回None
    """
    try:
        import os
        from datetime import datetime
        
        # 创建输出目录
        output_dir = "./visualization"
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用时间戳创建唯一文件名
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_path = os.path.join(output_dir, f"{prefix}_{timestamp}.png")
        
        # 检查后端类型
        if agent_graph.backend == "neo4j":
            print("Neo4j后端不支持可视化功能")
            return None
        else:
            try:
                # 使用更丰富的参数调用可视化方法
                agent_graph.visualize(
                    path=output_path,
                    vertex_size=30,                # 节点大小
                    edge_arrow_size=1.0,           # 边箭头大小
                    with_labels=True,              # 显示标签
                    vertex_color="#4287f5",        # 节点颜色（蓝色）
                    vertex_frame_width=1,          # 节点边框宽度
                    width=1500,                    # 图像宽度
                    height=1200,                   # 图像高度
                    show_attributes=True,          # 显示节点属性
                    show_edge_labels=True,         # 显示边标签
                    layout_name="fruchterman_reingold"  # 使用力导向布局算法
                )
                print(f"代理关系图已成功保存至: {output_path}")
                
                # 获取边信息
                edges = agent_graph.get_edges()
                
                # 节点数量 - 由于没有直接的get_nodes()方法，我们可以通过其他方式获取节点数
                # 这里假设agent_graph有vertices属性或类似方法
                try:
                    # 尝试不同的可能属性或方法获取节点数
                    if hasattr(agent_graph, 'vertices'):
                        nodes_count = len(agent_graph.vertices)
                    elif hasattr(agent_graph, 'nodes'):
                        nodes_count = len(agent_graph.nodes)
                    elif hasattr(agent_graph, 'get_vertices'):
                        nodes_count = len(agent_graph.get_vertices())
                    else:
                        # 如果都不存在，则只显示边的信息
                        print(f"图包含 {len(edges)} 条关系边")
                        nodes_count = None
                    
                    if nodes_count is not None:
                        print(f"图包含 {nodes_count} 个代理节点和 {len(edges)} 条关系边")
                    
                except Exception as e:
                    # 如果获取节点数失败，只显示边的信息
                    print(f"图包含 {len(edges)} 条关系边")
                
                # 打印边信息
                if len(edges) > 0:
                    print("边信息:")
                    for edge in edges:
                        print(f"  {edge}")
                
                return output_path
                
            except AttributeError as e:
                if "Plotting not available" in str(e):
                    print("缺少绘图依赖，请安装pycairo或cairocffi:")
                    print("pip install pycairo 或 pip install cairocffi")
                else:
                    raise
                return None
    except Exception as e:
        print(f"可视化代理关系图时出错: {str(e)}")
        traceback.print_exc()
        return None 