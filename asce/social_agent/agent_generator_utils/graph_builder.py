"""图构建模块,用于处理代理之间的关系图结构。

包含图结构构建、边关系管理、节点操作等功能。
"""
import networkx as nx
from typing import Any, Dict, List, Optional, Set, Tuple

class SocialGraph:
    """社交关系图类"""
    
    def __init__(self):
        """初始化社交图"""
        self.graph = nx.DiGraph()
        
    def add_node(self, node_id: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """添加节点"""
        self.graph.add_node(node_id, **attributes or {})
        
    def add_edge(
        self, 
        source: str, 
        target: str, 
        attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """添加边"""
        self.graph.add_edge(source, target, **attributes or {})
        
    def remove_node(self, node_id: str) -> None:
        """删除节点"""
        if self.graph.has_node(node_id):
            self.graph.remove_node(node_id)
            
    def remove_edge(self, source: str, target: str) -> None:
        """删除边"""
        if self.graph.has_edge(source, target):
            self.graph.remove_edge(source, target)
            
    def get_neighbors(self, node_id: str) -> Set[str]:
        """获取节点的邻居"""
        return set(self.graph.neighbors(node_id))
        
    def get_predecessors(self, node_id: str) -> Set[str]:
        """获取指向该节点的节点集合"""
        return set(self.graph.predecessors(node_id))
        
    def get_node_degree(self, node_id: str) -> Tuple[int, int]:
        """获取节点的入度和出度"""
        in_degree = self.graph.in_degree(node_id)
        out_degree = self.graph.out_degree(node_id)
        return in_degree, out_degree

def build_follow_matrix(
    agent_ids: List[str],
    follow_probability: float = 0.3
) -> List[List[int]]:
    """构建关注关系矩阵"""
    import random
    import numpy as np
    
    n = len(agent_ids)
    matrix = np.random.random((n, n)) < follow_probability
    np.fill_diagonal(matrix, False)
    return matrix.astype(int).tolist()

def analyze_graph_metrics(graph: SocialGraph) -> Dict[str, Any]:
    """分析图的统计指标"""
    metrics = {
        "node_count": graph.graph.number_of_nodes(),
        "edge_count": graph.graph.number_of_edges(),
        "density": nx.density(graph.graph),
        "average_clustering": nx.average_clustering(graph.graph),
        "strongly_connected_components": len(list(nx.strongly_connected_components(graph.graph)))
    }
    return metrics

def find_influential_nodes(
    graph: SocialGraph,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """找出最具影响力的节点"""
    pagerank = nx.pagerank(graph.graph)
    sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    return sorted_nodes[:top_k]

def get_community_structure(graph: SocialGraph) -> List[Set[str]]:
    """获取社区结构"""
    communities = nx.community.greedy_modularity_communities(graph.graph.to_undirected())
    return [set(community) for community in communities]

def calculate_shortest_paths(
    graph: SocialGraph,
    source: str,
    targets: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """计算最短路径"""
    targets = targets or list(graph.graph.nodes())
    paths = {}
    for target in targets:
        try:
            path = nx.shortest_path(graph.graph, source, target)
            paths[target] = path
        except nx.NetworkXNoPath:
            paths[target] = []
    return paths 