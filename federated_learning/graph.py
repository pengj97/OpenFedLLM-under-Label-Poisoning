import random
import torch

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

            
def is_valid(nx_graph):
    return nx.connected.is_connected(nx_graph)

def is_honest_graph_connected(nx_graph, byzantine_nodes):
    G = nx_graph.copy()
    for i in byzantine_nodes:
        G.remove_node(i)
    
    num_connected = 0
    for _ in nx.connected_components(G):
        num_connected += 1

    if num_connected == 1:
        return True
    else:
        return False

def MH_rule(graph):
    # Metropolis-Hastings rule
    node_size = graph.number_of_nodes()
    W = torch.eye(node_size, dtype=torch.float)

    for i in range(node_size):
        for j in range(node_size):
            if i == j or not graph.has_edge(j, i):
                continue
            i_n = graph.neighbor_sizes[i] + 1
            j_n = graph.neighbor_sizes[j] + 1
            W[i][j] = 1 / max(i_n, j_n)
            W[i][i] -= W[i][j]

    # return W
    return W

class Graph():
    def __init__(self, name, nx_graph, honest_nodes, byzantine_nodes):
        self.init(name, nx_graph, honest_nodes, byzantine_nodes)
        
    def init(self, name, nx_graph, honest_nodes, byzantine_nodes):
        self.name = name
        self.nx_graph = nx_graph
        self.honest_nodes = honest_nodes
        self.byzantine_nodes = byzantine_nodes
        # node counting
        self.node_size = nx_graph.number_of_nodes()
        self.honest_size = len(honest_nodes)
        self.byzantine_size = len(byzantine_nodes)
        # neighbor list
        self.neighbors = [
            list(nx_graph.neighbors(node)) for node in nx_graph.nodes()
        ]
        self.honest_neighbors = [
            [j for j in nx_graph.nodes() if nx_graph.has_edge(j, i)
                and j in honest_nodes]
            for i in nx_graph.nodes()
        ]
        self.byzantine_neighbors = [
            [j for j in nx_graph.nodes() if nx_graph.has_edge(j, i)
                and j in byzantine_nodes] 
            for i in nx_graph.nodes()
        ]
        self.honest_neighbors_and_itself = [
            neighbors + [node] for node, neighbors in enumerate(self.honest_neighbors)
        ]
        self.neighbors_and_itself = [
            neighbors + [node] for node, neighbors in enumerate(self.neighbors)
        ]
        # neighbor size list
        self.honest_sizes = [
            len(node_list) for node_list in self.honest_neighbors
        ]

        self.byzantine_sizes = []
        for node, node_list in enumerate(self.byzantine_neighbors):
            if node in self.byzantine_nodes:
                self.byzantine_sizes.append(len(node_list)+1)
            else:
                self.byzantine_sizes.append(len(node_list))

        self.neighbor_sizes = [
            len(node_list) for node_list in self.neighbors
        ]
        # self.byzantine_sizes = [
        #     len(node_list) for node_list in self.byzantine_neighbors
        # ]
        # self.neighbor_sizes = [
        #     self.honest_sizes[node] + self.byzantine_sizes[node] 
        #     for node in nx_graph.nodes()
        # ]
        
        # lost node refers to the the node has more than 1/2 byzantine neighbors
        self.lost_nodes = [
            node for node in self.honest_nodes
            if self.honest_sizes[node] <= 2 * self.byzantine_sizes[node]
        ]
        
    def honest_subgraph(self, name='', relabel=True):
        nx_subgraph = self.subgraph(self.honest_nodes)
        if name == '':
            name = self.name.replace(f'n={self.node_size}', f'n={self.honest_size}').replace(f'b={self.byzantine_size}', 'b=0')
        if relabel:
            nx_subgraph = nx.convert_node_labels_to_integers(nx_subgraph)
        return Graph(name=name, nx_graph=nx_subgraph, 
                     honest_nodes=list(nx_subgraph.nodes()),
                     byzantine_nodes=[])
    
    def __getattr__(self, attr):
        '''
        inherit the properties of 'nx_graph'
        '''
        return getattr(self.nx_graph, attr)

    def compute_lollipop_layout(self, G, head_radius=1.0, tail_length=0.2, center=(0.0, 0.0)):
        """
        计算 Lollipop 结构的布局：假设 G 是一个 lollipop 图，
        即有若干度 > 1 的“头部”节点（完全子图），以及一些度 = 1 的“挂叶”节点。
        head_radius: 头部圆的半径（可以调大使头看起来更“松散更大”）
        tail_length: 挂叶节点距离其邻居沿径向的移动距离（设小一些）
        center: 头部圆心坐标
        返回 pos: dict: node -> np.array([x, y])
        """
        # 找出度为1的节点（pendant），以及头部节点
        pendant_nodes = [n for n, d in G.degree() if d == 1]
        head_nodes = [n for n in G.nodes() if n not in pendant_nodes]
        # 若头部节点<1或只有1个节点，无法做圆，退回到spring/circular：
        if len(head_nodes) <= 1:
            # fallback: circular layout on all nodes
            pos = nx.circular_layout(G)
            return pos
    
        # 1) 为头部节点计算圆周均匀分布
        # 先用 circular_layout 得到初始角度顺序，也可以直接手动
        # 方式A：直接取 circular_layout，并缩放到 head_radius
        pos_head = nx.circular_layout(G.subgraph(head_nodes))  # dict: node -> np.array([x,y])
        # 将 pos_head 缩放到指定 radius，保持相对角度
        # 先计算当前质心（理论应接近0），然后计算每个点与质心方向归一化，再乘 head_radius
        # 这样保证头部节点在半径为 head_radius 的圆上
        # 计算质心（在 circular_layout 下，一般质心约在原点）
        pts = np.stack([pos_head[n] for n in head_nodes], axis=0)
        centroid = pts.mean(axis=0)
        pos = {}
        for n in head_nodes:
            vec = pos_head[n] - centroid
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                # 如果某个点碰巧在质心（非常罕见），给一个默认方向
                # 例如放在 (head_radius, 0)
                vec_norm = np.array([1.0, 0.0])
            else:
                vec_norm = vec / norm
            pos[n] = np.array(center) + vec_norm * head_radius
    
        # 2) 为挂叶节点计算位置：沿着与中心的径向方向，从其邻居位置移动 tail_length
        for p in pendant_nodes:
            # 取唯一邻居
            neighs = list(G.neighbors(p))
            if len(neighs) != 1:
                # 如果出现度>1的挂叶节点，跳过或fallback
                continue
            u = neighs[0]
            # u 应该在 pos 中，否则说明头部划分有问题
            if u not in pos:
                # fallback：直接放在 origin
                pos[p] = np.array(center) + np.array([0.0, -tail_length])
                continue
            # 计算方向：从中心到 u 的方向
            dir_vec = pos[u] - np.array(center)
            norm = np.linalg.norm(dir_vec)
            if norm < 1e-6:
                # u 恰好在 center？此时随机给一个方向
                dir_norm = np.array([0.0, -1.0])
            else:
                dir_norm = dir_vec / norm
            # 将 p 放在 u 位置再沿方向移动 tail_length
            pos[p] = pos[u] + dir_norm * tail_length
    
        return pos
    
    
    def show(self, reverse=False, rotate=False, show_label=False, show_lost=False, as_subplot=False,
             label_dict=None, node_size=400, font_size=12, layout='default', angle_degrees=0):
        NODE_COLOR_HONEST = '#99CCCC'
        NODE_COLOR_BYZANTINE = '#FF6666'
        NODE_COLOR_LOST = '#CCCCCC'
        EDGE_WIDTH = 2
                
        # layout
        if layout == 'default':
            pos = nx.kamada_kawai_layout(self.nx_graph)
        elif layout == 'circular':
            pos = nx.circular_layout(self.nx_graph)
        elif layout == 'lollipop':
            # 你可以调整 head_radius 和 tail_length。比如 head_radius=1.0、tail_length=0.2，可根据节点数和画布大小微调。
            pos = self.compute_lollipop_layout(self.nx_graph, head_radius=1.0, tail_length=0.5, center=(0.0, 0.0))
        
        if rotate:
            angle = np.radians(angle_degrees)
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle),  np.cos(angle)]
            ])
            pos = {k: rotation_matrix @ v for k, v in pos.items()}
        if reverse:
            pos = {n: (-x, -y) for n, (x, y) in pos.items()}

        # honest nodes
        nx.draw_networkx_nodes(self.nx_graph, pos, 
            node_size = node_size, 
            nodelist = self.honest_nodes,
            node_color = NODE_COLOR_HONEST,
        )
        # Byzantine nodes
        nx.draw_networkx_nodes(self.nx_graph, pos, 
            node_size = node_size,
            nodelist = self.byzantine_nodes,
            node_color = NODE_COLOR_BYZANTINE,
        )
        # Lost nodes
        if show_lost:
            nx.draw_networkx_nodes(self.nx_graph, pos, 
                node_size = node_size,
                nodelist = self.lost_nodes,
                node_color = NODE_COLOR_LOST,
            )

        nx.draw_networkx_edges(self.nx_graph, pos, alpha=0.5, width=EDGE_WIDTH)

        if show_label:
            if label_dict == None:
                label_dict = {
                    i: str(i) for i in range(self.nx_graph.number_of_nodes())
                }
            nx.draw_networkx_labels(self.nx_graph, pos, label_dict,
                                    font_size=font_size)
        
        if not as_subplot:
            plt.savefig(f'{self.name}.pdf')
            plt.show()
    
   

    def __getstate__(self):
        state = {
            'name': self.name,
            'nx_graph': self.nx_graph,
            'honest_nodes': self.honest_nodes,
            'byzantine_nodes': self.byzantine_nodes
        }
        return state
    
    def __setstate__(self, state):
        name = state['name']
        nx_graph = state['nx_graph']
        honest_nodes = state['honest_nodes']
        byzantine_nodes = state['byzantine_nodes']
        self.init(name, nx_graph, honest_nodes, byzantine_nodes)

class CompleteGraph(Graph):
    def __init__(self, node_size, byzantine_size):
        assert node_size > byzantine_size
        graph = nx.complete_graph(node_size)
        
        honest_nodes = list(range(node_size-byzantine_size))
        byzantine_nodes = list(range(node_size-byzantine_size, node_size))

        # all_nodes = list(range(node_size))
        # byzantine_nodes = list(range(7, 7 + byzantine_size))
        # honest_nodes = [node for node in all_nodes if node not in byzantine_nodes]

        # all_nodes = list(range(node_size))
        # byzantine_nodes = [0, 1, 5]
        # # byzantine_nodes = [5]
        # honest_nodes = [node for node in all_nodes if node not in byzantine_nodes]

        assert byzantine_size == len(byzantine_nodes)

        name = f'Complete_n={node_size}_b={byzantine_size}'
        super().__init__(name=name, nx_graph=graph,
                                            honest_nodes=honest_nodes,
                                            byzantine_nodes=byzantine_nodes)
        self.show(show_label=True)


class LollipopGraph(Graph):
    def __init__(self, node_size, byzantine_size):
        # assert node_size > byzantine_size and node_size == 2 * byzantine_size
        graph = nx.complete_graph(node_size - byzantine_size)

        all_nodes = list(range(node_size))
        honest_nodes = list(range(node_size - byzantine_size))
        byzantine_nodes = [node for node in all_nodes if node not in honest_nodes]
        for i, j in zip(honest_nodes[node_size - 2 * byzantine_size: ], byzantine_nodes):
            graph.add_edge(i, j)

        assert byzantine_size == len(byzantine_nodes)

        name = f'Lollipop_n={node_size}_b={byzantine_size}'
        super().__init__(name=name, nx_graph=graph,
                                            honest_nodes=honest_nodes,
                                            byzantine_nodes=byzantine_nodes)
        # self.show(show_label=True)
        
        
class LineGraph(Graph):
    def __init__(self, node_size, byzantine_size):
        assert node_size > byzantine_size
        graph = nx.path_graph(node_size)
        
        honest_nodes = list(range(node_size-byzantine_size))
        byzantine_nodes = list(range(node_size-byzantine_size, node_size))
        name = f'Line_n={node_size}_b={byzantine_size}'
        super().__init__(name=name, nx_graph=graph,
                                            honest_nodes=honest_nodes,
                                            byzantine_nodes=byzantine_nodes)



class TwoHeadLineGraph(Graph):
    def __init__(self, node_size, byzantine_size):
        assert node_size > byzantine_size
        honest_size = node_size - byzantine_size

        honest_nodes = list(range(honest_size))
        byzantine_nodes = list(range(honest_size, node_size))

        graph = nx.path_graph(honest_size)
        for i in byzantine_nodes:
            graph.add_edge(honest_size-1, i)

        name = f'TwoHeadLine_n={node_size}_b={byzantine_size}'
        super().__init__(name=name, nx_graph=graph,
                                            honest_nodes=honest_nodes,
                                            byzantine_nodes=byzantine_nodes)
        self.show(show_label=True)


class UnconnectedRegularLineGraph(Graph):
    def __init__(self, node_size, byzantine_size):
        assert node_size > byzantine_size
        graph = nx.path_graph(node_size)
        
        all_nodes = list(range(node_size))
        byzantine_nodes = list(range((node_size-byzantine_size) // 2, (node_size-byzantine_size) // 2 + byzantine_size))
        honest_nodes = [node for node in all_nodes if node not in byzantine_nodes]
        name = f'UnconnectedRegularLine_n={node_size}_b={byzantine_size}'
        super().__init__(name=name, nx_graph=graph,
                                            honest_nodes=honest_nodes,
                                            byzantine_nodes=byzantine_nodes)
        # self.show()



class FanGraph(Graph):
    def __init__(self, node_size, byzantine_size):
        assert node_size > byzantine_size
        honest_size = node_size - byzantine_size

        honest_nodes = list(range(honest_size))
        byzantine_nodes = list(range(honest_size, node_size))

        graph = nx.path_graph(honest_size)
        for i in honest_nodes:
            for j in byzantine_nodes:
                graph.add_edge(i, j)

        name = f'Fan_n={node_size}_b={byzantine_size}'
        super().__init__(name=name, nx_graph=graph,
                                            honest_nodes=honest_nodes,
                                            byzantine_nodes=byzantine_nodes)
        # self.show_reverse()


class FanBaselineGraph(Graph):
    def __init__(self, node_size, byzantine_size=0):
        assert byzantine_size == 0
        line_size = node_size - 1

        line_nodes = list(range(line_size))
        head_nodes = list(range(line_size, node_size))

        graph = nx.path_graph(line_size)
        for i in line_nodes:
            for j in head_nodes:
                graph.add_edge(i, j)

        name = f'FanBaseline_n={node_size}_b={byzantine_size}'
        super().__init__(name=name, nx_graph=graph,
                                            honest_nodes=line_nodes + head_nodes,
                                            byzantine_nodes=[])
        self.show()

class LowerBoundGraph(Graph):
    def __init__(self, node_size, byzantine_size):
        assert node_size == 2 * byzantine_size
        honest_size = node_size - byzantine_size
        honest_nodes = list(range(honest_size))
        byzantine_nodes = list(range(honest_size, node_size))
        
        graph = nx.complete_graph(honest_size)
        for i in byzantine_nodes:
            graph.add_edge(i%honest_size, i)
            graph.add_edge((i+1)%honest_size, i)
        
        name = f'LowerBoundGraph_n={node_size}_b={byzantine_size}'
        super().__init__(name=name, nx_graph=graph,
                                            honest_nodes=honest_nodes,
                                            byzantine_nodes=byzantine_nodes)
        # self.show(show_label=True, label_dict=label_dict)
        
        
class RingGraph(Graph):
    def __init__(self, node_size, byzantine_size):
        assert node_size > byzantine_size
        graph = nx.cycle_graph(node_size)
        
        honest_nodes = list(range(node_size-byzantine_size))
        byzantine_nodes = list(range(node_size-byzantine_size, node_size))
        name = f'Ring_n={node_size}_b={byzantine_size}'
        super().__init__(name=name, nx_graph=graph,
                                            honest_nodes=honest_nodes,
                                            byzantine_nodes=byzantine_nodes)
        # self.show()

class StarGraph(Graph):
    def __init__(self, node_size, byzantine_size):
        assert node_size > byzantine_size
        graph = nx.star_graph(node_size-1)
        
        honest_nodes = list(range(node_size-byzantine_size))
        byzantine_nodes = list(range(node_size-byzantine_size, node_size))
        name = f'Star_n={node_size}_b={byzantine_size}'
        super().__init__(name=name, nx_graph=graph,
                                            honest_nodes=honest_nodes,
                                            byzantine_nodes=byzantine_nodes)
        self.show(show_label=True)

class ErdosRenyi(Graph):
    def __init__(self, node_size, byzantine_size, connected_p=0.7, seed=None):
        rng = random if seed is None else random.Random(seed)
        valid = False
        # while not valid:
        #     graph = nx.fast_gnp_random_graph(node_size, connected_p, seed=rng)
        #     valid = is_valid(graph)
        
        # byzantine_nodes = rng.sample(list(graph.nodes()), byzantine_size)
        # honest_nodes = [i for i in graph.nodes() if i not in byzantine_nodes]

        while not valid:
            graph = nx.fast_gnp_random_graph(node_size, connected_p, seed=rng)
            byzantine_nodes = rng.sample(list(graph.nodes()), byzantine_size)
            valid = is_honest_graph_connected(graph, byzantine_nodes)
        
        honest_nodes = [i for i in graph.nodes() if i not in byzantine_nodes]
        name = f'ER_n={node_size}_b={byzantine_size}_p={connected_p}'
        if seed is not None:
            name = name + f'_seed={seed}'
        super().__init__(name = name, nx_graph = graph,
                                         honest_nodes=honest_nodes,
                                         byzantine_nodes=byzantine_nodes)
        self.show()
        
class RandomGeometricGraph(Graph):
    def __init__(self, node_size, byzantine_size, radius, seed=None):
        rng = random if seed is None else random.Random(seed)
        valid = False
        while not valid:
            graph = nx.random_geometric_graph(node_size, radius, seed=rng)
            valid = is_valid(graph)
        
        # byzantine_nodes = rng.sample(list(graph.nodes()), byzantine_size)
        # honest_nodes = [i for i in graph.nodes() if i not in byzantine_nodes]
        honest_nodes = list(range(node_size-byzantine_size))
        byzantine_nodes = list(range(node_size-byzantine_size, node_size))
        name = f'RGG_n={node_size}_b={byzantine_size}_radius={radius}'
        if seed is not None:
            name = name + f'_seed={seed}'
        super().__init__(name = name, nx_graph = graph,
                                         honest_nodes=honest_nodes,
                                         byzantine_nodes=byzantine_nodes)
        self.radius = radius
        self.P = self.Prob_matrix() 
        self.neighbors, self.honest_neighbors, self.byzantine_neighbors, self.neighbor_sizes, self.honest_sizes, self.byzantine_sizes = self.time_varying_graph()

    def Prob_matrix(self, k=0.7):
        nodes = self.nx_graph.nodes(data=True)
        P = torch.zeros((self.node_size, self.node_size))   
        for u, du in nodes:
            pu = du['pos']
            for v, dv in nodes:
                pv = dv['pos']
                d = sum((a - b) ** 2 for a, b in zip(pu, pv))
                prob = k ** ((d / self.radius ** 2) ** 2)
                P[u][v] = prob
        return P
    
    def time_varying_graph(self):
        neighbors = self.neighbors.copy()
        honest_neighbors = self.honest_neighbors.copy()
        byzantine_neighbors = self.byzantine_neighbors.copy()
        # neighbor_sizes = []
        # honest_sizes = []
        # byzantine_sizes = []
        for i, j in self.nx_graph.edges():
            if random.random() >= self.P[i][j]:
                neighbors[i].remove(j)
                neighbors[j].remove(i)
                if j in self.honest_nodes:
                    honest_neighbors[i].remove(j)
                elif j in self.byzantine_nodes:
                    byzantine_neighbors[i].remove(j)
                if i in self.honest_nodes:
                    honest_neighbors[j].remove(i)
                elif i in self.byzantine_nodes:
                    byzantine_neighbors[j].remove(i)
        
        # neighbor size list
        honest_sizes = [
            len(node_list) for node_list in honest_neighbors
        ]
        byzantine_sizes = [
            len(node_list) for node_list in byzantine_neighbors
        ]
        neighbor_sizes = [
            honest_sizes[node] + byzantine_sizes[node] 
            for node in self.nx_graph.nodes()
        ]

        return neighbors, honest_neighbors, byzantine_neighbors, neighbor_sizes, honest_sizes, byzantine_sizes      
    
class TwoCastle(Graph):
    '''
    There are 2k nodes in the netword totally
    '''
    def __init__(self, k=3, byzantine_size=1, seed=None):
        '''k >= 3, byzantine_size <= k-2'''
        assert k >= 3, 'k must be greater than or equal to 3'
        assert byzantine_size <= k - 2, 'byzantine_size must be less than or equal to k - 2'
        node_size = 2 * k
        rng = random if seed is None else random.Random(seed)
        graph = nx.Graph()
        graph.add_nodes_from(range(node_size))
        # inner edges
        for castle in range(2):
            edges_list = [(i, j) for i in range(k*castle, k*castle+k)
                          for j in range(i+1, k*castle+k)]
            graph.add_edges_from(edges_list)
        # outer edges
        edges_list = [(i, j) for i in range(k)
                      for j in range(k, 2*k) if i + k != j]
        graph.add_edges_from(edges_list)
        # byzantine_nodes = rng.sample(list(graph.nodes()), byzantine_size)
        byzantine_nodes = [4]
        # byzantine_nodes = [4, 5]
        honest_nodes = [i for i in graph.nodes() if i not in byzantine_nodes]
        name = f'TwoCastle_k={k}_b={byzantine_size}'
        if seed is not None:
            name = name + f'_seed={seed}'
        super().__init__(name = name, nx_graph = graph,
                                        honest_nodes=honest_nodes,
                                        byzantine_nodes=byzantine_nodes)
        # self.show()

class RingCastle(Graph):
    def __init__(self, castle_cnt, byzantine_size, seed=None):
        node_size = 4 * castle_cnt
        
        rng = random if seed is None else random.Random(seed)
        graph = nx.Graph()
        graph.add_nodes_from(range(node_size))
        
        # inner edges
        for castle in range(castle_cnt):
            edges_list = [(i, j) for i in range(4*castle, 4*castle+4)
                          for j in range(i+1, 4*castle+4)]
            graph.add_edges_from(edges_list)
        # outer edges
        for castle in range(castle_cnt):
            next_castle = (castle+1) % castle_cnt
            graph.add_edges_from([
                (4*castle+2, 4*next_castle+0),
                (4*castle+3, 4*next_castle+1),
            ])
            byzantine_nodes = rng.sample(graph.nodes(), byzantine_size)
            honest_nodes = [i for i in graph.nodes() if i not in byzantine_nodes]
            name = f'RingCastle_castle={castle_cnt}_b={byzantine_size}'
            if seed is not None:
                name = name + f'_seed={seed}'
        
        super().__init__(name, graph, honest_nodes, byzantine_nodes)
        

class OctopusGraph(Graph):
    def __init__(self, head_cnt, head_byzantine_cnt, hand_byzantine_cnt):
        assert head_cnt > head_byzantine_cnt
        # assert head_cnt > hand_byzantine_cnt
        # head
        nx_graph = nx.complete_graph(head_cnt)
        # hands
        nx_graph.add_nodes_from(range(head_cnt, 2*head_cnt))
        nx_graph.add_edges_from([
            (i, i+head_cnt) for i in range(head_cnt)
        ])
        
        honest_nodes = list(range(head_byzantine_cnt, head_cnt)) \
            + list(range(head_cnt+hand_byzantine_cnt, 2*head_cnt))
        byzantine_nodes = list(range(head_byzantine_cnt)) \
            + list(range(head_cnt, head_cnt+hand_byzantine_cnt))
        
        name = f'Octopus_head={head_cnt}_headb={head_byzantine_cnt}_handb={hand_byzantine_cnt}'
        super().__init__(name, nx_graph, honest_nodes, byzantine_nodes)
        self.show()


class  Graph_byz_nodes_on_shortest_paths(Graph):
    def __init__(self, node_size):
        byzantine_size = 1
        honest_size = node_size - byzantine_size
        graph = nx.path_graph(honest_size)

        if byzantine_size > 0 :
            byzantine_index = honest_size
            graph.add_node(byzantine_index)

            byzantine_edges = [(0, byzantine_index), (honest_size - 1, byzantine_index)]
            graph.add_edges_from(byzantine_edges)

        honest_nodes = list(range(node_size-byzantine_size))
        byzantine_nodes = list(range(node_size-byzantine_size, node_size))
        name = f'Graph_byz_on_shortest_n={node_size}_b={byzantine_size}'
        super().__init__(name=name, nx_graph=graph,
                                            honest_nodes=honest_nodes,
                                            byzantine_nodes=byzantine_nodes)
        self.show()
        

class  Graph_byz_nodes_Not_on_shortest_paths(Graph):
    def __init__(self, node_size):
        byzantine_size = 1
        honest_size = node_size - byzantine_size
        graph = nx.path_graph(honest_size)

        byzantine_index = honest_size
        graph.add_node(byzantine_index)

        byzantine_edges = [(honest_size - 2, byzantine_index), (honest_size - 1, byzantine_index)]
        graph.add_edges_from(byzantine_edges)

        honest_nodes = list(range(node_size-byzantine_size))
        byzantine_nodes = list(range(node_size-byzantine_size, node_size))
        name = f'Graph_byz_Not_on_shortest_n={node_size}_b={byzantine_size}'
        super().__init__(name=name, nx_graph=graph,
                                            honest_nodes=honest_nodes,
                                            byzantine_nodes=byzantine_nodes)
        self.show()        

def create_graph(graph: str, node_size: int, byzantine_size: int, **kwargs):
    """Factory helper that constructs a graph object from a simple type string.

    ``graph`` may be one of the names of the classes defined above.
    The lookup is case-insensitive and matches prefixes.  ``None`` or the
    strings ``'none'``/``'central'`` return ``None`` which indicates that a
    centralized aggregation should be used instead of a decentralized protocol.
    Additional keyword arguments are passed through to the constructor to
    support graphs such as ``ErdosRenyi`` or ``RandomGeometricGraph`` that
    accept parameters like ``connected_p`` or ``radius``.
    """
    if graph is None or graph.lower() in ["none", "central", ""]:
        return None
    name = graph.lower()
    mapping = {
        'complete': CompleteGraph,
        'line': LineGraph,
        'lollipop': LollipopGraph,
        'unconnectedregularline': UnconnectedRegularLineGraph,
        'fan': FanGraph,
        'fanbaseline': FanBaselineGraph,
        'lowerbound': LowerBoundGraph,
        'ring': RingGraph,
        'star': StarGraph,
        'erdosrenyi': ErdosRenyi,
        'er': ErdosRenyi,
        'randomgeometric': RandomGeometricGraph,
        'rgg': RandomGeometricGraph,
        'twocastle': TwoCastle,
        'ringcastle': RingCastle,
        'octopus': OctopusGraph,
        'byz_shortest': Graph_byz_nodes_on_shortest_paths,
        'byz_not_shortest': Graph_byz_nodes_Not_on_shortest_paths,
    }
    for key, cls in mapping.items():
        if name.startswith(key):
            if key == 'twocastle':
                node_size = node_size // 2
            try:
                return cls(node_size, byzantine_size, **kwargs)
            except TypeError:
                return cls(node_size, byzantine_size)
    raise ValueError(f"unrecognized graph type '{graph}'")        