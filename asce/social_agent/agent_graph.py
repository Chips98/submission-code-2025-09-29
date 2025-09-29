# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from __future__ import annotations
import matplotlib.pyplot as plt
from typing import Any, Literal
from neo4j import GraphDatabase
import igraph as ig
from asce.social_agent.agent import SocialAgent
from asce.social_platform.config import Neo4jConfig


class Neo4jHandler:

    def __init__(self, nei4j_config: Neo4jConfig):
        self.driver = GraphDatabase.driver(
            nei4j_config.uri,
            auth=(nei4j_config.username, nei4j_config.password),
        )
        self.driver.verify_connectivity()

    def close(self):
        self.driver.close()

    def create_agent(self, agent_id: int):
        with self.driver.session() as session:
            session.write_transaction(self._create_and_return_agent, agent_id)

    def delete_agent(self, agent_id: int):
        with self.driver.session() as session:
            session.write_transaction(
                self._delete_agent_and_relationships,
                agent_id,
            )

    def get_number_of_nodes(self) -> int:
        with self.driver.session() as session:
            return session.read_transaction(self._get_number_of_nodes)

    def get_number_of_edges(self) -> int:
        with self.driver.session() as session:
            return session.read_transaction(self._get_number_of_edges)

    def add_edge(self, src_agent_id: int, dst_agent_id: int):
        with self.driver.session() as session:
            session.write_transaction(
                self._add_and_return_edge,
                src_agent_id,
                dst_agent_id,
            )

    def remove_edge(self, src_agent_id: int, dst_agent_id: int):
        with self.driver.session() as session:
            session.write_transaction(
                self._remove_and_return_edge,
                src_agent_id,
                dst_agent_id,
            )

    def get_all_nodes(self) -> list[int]:
        with self.driver.session() as session:
            return session.read_transaction(self._get_all_nodes)

    def get_all_edges(self) -> list[tuple[int, int]]:
        with self.driver.session() as session:
            return session.read_transaction(self._get_all_edges)

    def reset_graph(self):
        with self.driver.session() as session:
            session.write_transaction(self._reset_graph)

    @staticmethod
    def _create_and_return_agent(tx: Any, agent_id: int):
        query = """
        CREATE (a:Agent {id: $agent_id})
        RETURN a
        """
        result = tx.run(query, agent_id=agent_id)
        return result.single()

    @staticmethod
    def _delete_agent_and_relationships(tx: Any, agent_id: int):
        query = """
        MATCH (a:Agent {id: $agent_id})
        DETACH DELETE a
        RETURN count(a) AS deleted
        """
        result = tx.run(query, agent_id=agent_id)
        return result.single()

    @staticmethod
    def _add_and_return_edge(tx: Any, src_agent_id: int, dst_agent_id: int):
        query = """
        MATCH (a:Agent {id: $src_agent_id}), (b:Agent {id: $dst_agent_id})
        CREATE (a)-[r:FOLLOW]->(b)
        RETURN r
        """
        result = tx.run(query,
                        src_agent_id=src_agent_id,
                        dst_agent_id=dst_agent_id)
        return result.single()

    @staticmethod
    def _remove_and_return_edge(tx: Any, src_agent_id: int, dst_agent_id: int):
        query = """
        MATCH (a:Agent {id: $src_agent_id})
        MATCH (b:Agent {id: $dst_agent_id})
        MATCH (a)-[r:FOLLOW]->(b)
        DELETE r
        RETURN count(r) AS deleted
        """
        result = tx.run(query,
                        src_agent_id=src_agent_id,
                        dst_agent_id=dst_agent_id)
        return result.single()

    @staticmethod
    def _get_number_of_nodes(tx: Any) -> int:
        query = """
        MATCH (n)
        RETURN count(n) AS num_nodes
        """
        result = tx.run(query)
        return result.single()["num_nodes"]

    @staticmethod
    def _get_number_of_edges(tx: Any) -> int:
        query = """
        MATCH ()-[r]->()
        RETURN count(r) AS num_edges
        """
        result = tx.run(query)
        return result.single()["num_edges"]

    @staticmethod
    def _get_all_nodes(tx: Any) -> list[int]:
        query = """
        MATCH (a:Agent)
        RETURN a.id AS agent_id
        """
        result = tx.run(query)
        return [record["agent_id"] for record in result]

    @staticmethod
    def _get_all_edges(tx: Any) -> list[tuple[int, int]]:
        query = """
        MATCH (a:Agent)-[r:FOLLOW]->(b:Agent)
        RETURN a.id AS src_agent_id, b.id AS dst_agent_id
        """
        result = tx.run(query)
        return [(record["src_agent_id"], record["dst_agent_id"])
                for record in result]

    @staticmethod
    def _reset_graph(tx: Any):
        query = """
        MATCH (n)
        DETACH DELETE n
        """
        tx.run(query)

class AgentGraph:
    r"""代理图类，用于管理代理之间的社交关系图。"""

    def __init__(
        self,
        backend: Literal["igraph", "neo4j"] = "igraph",  # 后端类型，可以是"igraph"或"neo4j"
        neo4j_config: Neo4jConfig | None = None,  # Neo4j配置，如果使用neo4j后端则需要提供
    ):
        # 初始化代理图的后端类型
        self.backend = backend
        if self.backend == "igraph":  # 如果使用igraph作为后端
            self.graph = ig.Graph(directed=True)  # 创建一个有向图
        else:  # 如果使用neo4j作为后端
            assert neo4j_config is not None  # 断言neo4j配置不为空
            assert neo4j_config.is_valid()  # 断言neo4j配置有效
            self.graph = Neo4jHandler(neo4j_config)  # 创建Neo4j处理器
        self.agent_mappings: dict[int, SocialAgent] = {}  # 初始化代理ID到代理对象的映射字典

    def reset(self):
        """重置代理图，清除所有节点和边"""
        if self.backend == "igraph":  # 如果使用igraph作为后端
            self.graph = ig.Graph(directed=True)  # 重新创建一个空的有向图
        else:  # 如果使用neo4j作为后端
            self.graph.reset_graph()  # 调用neo4j处理器的重置方法
        self.agent_mappings: dict[int, SocialAgent] = {}  # 重置代理映射字典

    def add_agent(self, agent: SocialAgent):
        """添加一个代理到图中
        
        参数:
            agent: 要添加的社交代理对象
        """
        if self.backend == "igraph":  # 如果使用igraph作为后端
            self.graph.add_vertex(agent.agent_id)  # 添加一个以代理ID为标识的顶点
        else:  # 如果使用neo4j作为后端
            self.graph.create_agent(agent.agent_id)  # 调用neo4j处理器创建代理
        self.agent_mappings[agent.agent_id] = agent  # 将代理添加到映射字典中

    def add_edge(self, agent_id_0: int, agent_id_1: int):
        """在两个代理之间添加一条边（表示关注关系）
        
        参数:
            agent_id_0: 源代理ID
            agent_id_1: 目标代理ID
        """
        try:
            self.graph.add_edge(agent_id_0, agent_id_1)  # 尝试添加一条从agent_id_0到agent_id_1的边
        except Exception:
            pass  # 如果添加失败（例如边已存在），则忽略异常

    def remove_agent(self, agent: SocialAgent):
        """从图中移除一个代理
        
        参数:
            agent: 要移除的社交代理对象
        """
        if self.backend == "igraph":  # 如果使用igraph作为后端
            self.graph.delete_vertices(agent.agent_id)  # 删除对应的顶点
        else:  # 如果使用neo4j作为后端
            self.graph.delete_agent(agent.agent_id)  # 调用neo4j处理器删除代理
        del self.agent_mappings[agent.agent_id]  # 从映射字典中删除代理

    def remove_edge(self, agent_id_0: int, agent_id_1: int):
        """移除两个代理之间的边（取消关注关系）
        
        参数:
            agent_id_0: 源代理ID
            agent_id_1: 目标代理ID
        """
        if self.backend == "igraph":  # 如果使用igraph作为后端
            if self.graph.are_connected(agent_id_0, agent_id_1):  # 如果两个代理之间有连接
                self.graph.delete_edges([(agent_id_0, agent_id_1)])  # 删除这条边
        else:  # 如果使用neo4j作为后端
            self.graph.remove_edge(agent_id_0, agent_id_1)  # 调用neo4j处理器移除边

    def get_agent(self, agent_id: int) -> SocialAgent:
        """根据代理ID获取代理对象
        
        参数:
            agent_id: 代理ID
        返回:
            对应的社交代理对象
        """
        return self.agent_mappings[agent_id]  # 从映射字典中获取代理对象

    def get_agents(self) -> list[tuple[int, SocialAgent]]:
        """获取图中所有代理
        
        返回:
            包含(代理ID, 代理对象)元组的列表
        """
        if self.backend == "igraph":  # 如果使用igraph作为后端
            # 列表推导式：遍历图中所有顶点，返回(顶点索引, 对应的代理对象)元组列表
            return [(node.index, self.agent_mappings[node.index])
                    for node in self.graph.vs]
        else:  # 如果使用neo4j作为后端
            # 列表推导式：遍历所有代理ID，返回(代理ID, 对应的代理对象)元组列表
            return [(agent_id, self.agent_mappings[agent_id])
                    for agent_id in self.graph.get_all_nodes()]

    def get_edges(self) -> list[tuple[int, int]]:
        """获取图中所有边（关注关系）
        
        返回:
            包含(源代理ID, 目标代理ID)元组的列表
        """
        if self.backend == "igraph":  # 如果使用igraph作为后端
            # 列表推导式：遍历图中所有边，返回(源顶点, 目标顶点)元组列表
            return [(edge.source, edge.target) for edge in self.graph.es]
        else:  # 如果使用neo4j作为后端
            return self.graph.get_all_edges()  # 调用neo4j处理器获取所有边

    def get_num_nodes(self) -> int:
        """获取图中节点（代理）的数量
        
        返回:
            节点数量
        """
        if self.backend == "igraph":  # 如果使用igraph作为后端
            return self.graph.vcount()  # 返回图中顶点数量
        else:  # 如果使用neo4j作为后端
            return self.graph.get_number_of_nodes()  # 调用neo4j处理器获取节点数量

    def get_num_edges(self) -> int:
        """获取图中边（关注关系）的数量
        
        返回:
            边的数量
        """
        if self.backend == "igraph":  # 如果使用igraph作为后端
            return self.graph.ecount()  # 返回图中边的数量
        else:  # 如果使用neo4j作为后端
            return self.graph.get_number_of_edges()  # 调用neo4j处理器获取边的数量

    def close(self) -> None:
        """关闭图的连接（仅适用于neo4j后端）"""
        if self.backend == "neo4j":  # 如果使用neo4j作为后端
            self.graph.close()  # 关闭neo4j连接

    def visualize(
        self,
        path: str,  # 保存可视化图像的路径
        vertex_size: int = 30,  # 顶点大小
        edge_arrow_size: float = 0.8,  # 边箭头大小
        with_labels: bool = True,  # 是否显示标签
        vertex_color: str = "#4287f5",  # 顶点颜色（十六进制颜色代码）
        vertex_frame_width: int = 1,  # 顶点边框宽度
        width: int = 1200,  # 图像宽度
        height: int = 1000,  # 图像高度
        show_attributes: bool = True,  # 是否显示节点属性
        show_edge_labels: bool = True,  # 是否显示边标签
        layout_name: str = "fruchterman_reingold"  # 布局算法
    ):
        """将代理图可视化并保存为图像，展示更丰富的信息
        
        参数:
            path: 保存图像的文件路径
            vertex_size: 顶点大小
            edge_arrow_size: 边箭头大小
            with_labels: 是否显示顶点标签
            vertex_color: 顶点颜色
            vertex_frame_width: 顶点边框宽度
            width: 图像宽度
            height: 图像高度
            show_attributes: 是否显示节点属性
            show_edge_labels: 是否显示边标签
            layout_name: 布局算法名称
        """

        
        if self.backend == "neo4j":  # 如果使用neo4j后端
            raise ValueError("Neo4j backend does not support visualization.")
        
        # 为图添加可视化属性
        graph = self.graph.copy()  # 复制图以避免修改原图
        
        # 添加节点属性
        if show_attributes:
            for i, (agent_id, agent) in enumerate(self.get_agents()):
                # 设置节点属性
                graph.vs[i]["label"] = f"{agent_id}"
                graph.vs[i]["username"] = agent.user_info.name
                graph.vs[i]["color"] = vertex_color
                
                # 增加活跃节点的大小
                if hasattr(agent.user_info.profile["other_info"], "activity_level_frequency"):
                    activity = sum(agent.user_info.profile["other_info"].get("activity_level_frequency", [0]))
                    graph.vs[i]["size"] = vertex_size + min(activity * 0.5, 20)  # 限制最大增加值
                else:
                    graph.vs[i]["size"] = vertex_size
        
        # 添加边标签和样式
        if show_edge_labels and graph.ecount() > 0:
            for i, edge in enumerate(graph.es):
                graph.es[i]["label"] = "follows"
                graph.es[i]["width"] = 1
                graph.es[i]["arrow_size"] = edge_arrow_size
                graph.es[i]["curved"] = 0.3  # 弯曲边以避免重叠
        
        # 选择布局算法
        layout_options = {
            "fruchterman_reingold": graph.layout_fruchterman_reingold,
            "kamada_kawai": graph.layout_kamada_kawai,
            "circle": graph.layout_circle,
            "grid": graph.layout_grid,
            "random": graph.layout_random,
            "auto": graph.layout
        }
        
        layout = layout_options.get(layout_name, graph.layout)(dim=2)
        
        # 准备节点标签 - 修复属性访问方式
        if with_labels:
            if show_attributes:
                # 修复：使用正确的属性访问方式
                vertex_labels = [f"ID:{v.index}\n{v['username'] if 'username' in v.attribute_names() else ''}" for v in graph.vs]
            else:
                # 简单标签：只显示ID
                vertex_labels = [str(node_id) for node_id, _ in self.get_agents()]
        else:
            vertex_labels = None
        
        # 创建自定义样式 - 修复属性访问方式
        visual_style = {
            "vertex_size": [v["size"] if "size" in v.attribute_names() else vertex_size for v in graph.vs] if show_attributes else vertex_size,
            "vertex_color": [v["color"] if "color" in v.attribute_names() else vertex_color for v in graph.vs] if show_attributes else vertex_color,
            "vertex_label": vertex_labels,
            "vertex_label_dist": 1.5,
            "vertex_label_size": 12,
            "vertex_frame_width": vertex_frame_width,
            "edge_width": [e["width"] if "width" in e.attribute_names() else 1 for e in graph.es] if graph.ecount() > 0 and show_edge_labels else 1,
            "edge_arrow_size": [e["arrow_size"] if "arrow_size" in e.attribute_names() else edge_arrow_size for e in graph.es] if graph.ecount() > 0 and show_edge_labels else edge_arrow_size,
            "edge_label": [e["label"] if "label" in e.attribute_names() else "" for e in graph.es] if graph.ecount() > 0 and show_edge_labels else None,
            "edge_curved": [e["curved"] if "curved" in e.attribute_names() else 0 for e in graph.es] if graph.ecount() > 0 and show_edge_labels else 0,
            "layout": layout,
            "bbox": (width, height),
            "margin": 100
        }
        
        # 绘制图像
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        
        # 使用igraph的plot函数
        ig.plot(
            graph,
            target=path,
            **visual_style
        )
        
        # 添加标题和说明
        plt.figtext(0.02, 0.02, f"Total Agents: {graph.vcount()}, Relationships: {graph.ecount()}", 
                   fontsize=10, ha='left')
        
        plt.close(fig)
        
        print(f"代理关系图已保存至: {path}")
        print(f"图包含 {graph.vcount()} 个代理节点和 {graph.ecount()} 条关系边")
