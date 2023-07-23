"""
DYNAMIC_RRT_CONNECT_2D
@author: Dodge Ho (asdsay@gmail.com)
2023.04.09
"""
import math, time, copy
from scipy.interpolate import BSpline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import env, plotting, utils, QueueFIFO


# 点结构
class Node:
    def __init__(self, n):
        self.x = n[0] # x,y为坐标
        self.y = n[1]
        self.parent = None # parent,child为路径中的上一点和下一点
        self.child = None
        self.flag = "VALID" # 无效标记
        self.edge = [] # 拥有的边
        self.visit = False # 遍历用

class Edge:
    def __init__(self, n_p, n_c):
        self.n1 = n_p # n1, n2 边的两个点
        self.n2 = n_c
        self.flag = "VALID" # 无效标记


# 双向生成动态避障算法
class DynamicRrtConnect:

    # 初始化函数：初始化类成员变量及需要的环境
    def __init__(self, s_start, s_goal,
                step_len = 0.8, goal_sample_rate = 0.05, waypoint_sample_rate = 0.65, iter_max = 5000, bs_degree = 3):
        # 将参数初始化到self中
        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.waypoint_sample_rate = waypoint_sample_rate
        self.iter_max = iter_max
        self.bs_degree = bs_degree
        
        # 初始化过程变量
        self.V1 = [self.s_start]
        self.V2 = [self.s_goal]
        self.vertex = []
        self.vertex_old = []
        self.vertex_new = []
        self.E1 = []
        self.E2 = []
        self.edges = []
        self.edges_coor = []

        # 初始化作图环境和变量
        self.env = env.Env()
        self.plotting = plotting.Plotting(s_start, s_goal)
        self.utils = utils.Utils()
        self.fig, self.ax = plt.subplots()
        self.x_range = self.env.x_range # 注意此处的range与输入的起点终点无关，它被env.py中的init函数所规定的了。
        self.y_range = self.env.y_range # 如果想要修改，请注意同时修改env中的绘图区域和on_press函数中的判断范围
        self.obs_circle = self.env.obs_circle  #circle， rectangle， boundary同样如此
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary
        self.obs_add = [0, 0, 0]
        self.path = []
        self.waypoint = []

    # 首次绘制函数：首次绘制rrt图，第一次绘制和rrt_connect双向生成算法无结果区别
    def planning(self):
        t1 = time.time() #计时器
        for i in range(self.iter_max):
            last_node, node_v1_new = self.rrt_connect_inuse()

            if self.is_node_same(last_node, node_v1_new):
                #连线完成，提取路径，绘图。
                self.vertex, self.edges = list(self.V1 + self.V2), list(self.E1 + self.E2)
                self.path, self.waypoint = self.extract_path(node_v1_new, last_node)
                t2 = time.time() #计时器
                print('The program runs: %s seconds' % (t2 - t1))

                self.plot_grid("Dynamic_RRT_CONNECT")
                self.plot_visited()
                self.plot_path(self.path, 'lightcoral')
                self.plot_path_in_BSline(self.path, self.bs_degree, 'red')
                self.fig.canvas.mpl_connect('button_press_event', self.on_press)
                plt.pause(0.01)
                plt.show()
                return

            if len(self.V2) < len(self.V1):
                self.V1, self.V2 = self.V2, self.V1
                self.E1, self.E2 = self.E2, self.E1

        # 计算iter_max次仍无结果返回None
        return None

    # 点击事件函数：处理鼠标单击屏幕区域后的函数
    def on_press(self, event):
        t1 = time.time() #计时器
        x, y = event.xdata, event.ydata
        if x < 0 or x > 50 or y < 0 or y > 30:
            print("Please choose right area!")
        else:
            x, y = int(x), int(y)
            print("Add circle obstacle at: s =", x, ",", "y =", y)
            self.obs_add = [x, y, 2]
            self.obs_circle.append([x, y, 2])
            self.utils.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle)
            self.InvalidateNodes()

            if self.is_path_invalid():
  
                print("Path is Replanning ...")
                self.replanning()
                t2 = time.time() #计时器
                print('The program runs: %s seconds' % (t2 - t1))
                print("len_vertex: ", len(self.vertex))
                print("len_vertex_old: ", len(self.vertex_old))
                print("len_vertex_new: ", len(self.vertex_new))

                # 清空所有内容，重新画图
                plt.cla()
                self.utils = utils.Utils()
                self.plot_grid("Dynamic_RRT_CONNECT")
                self.plotting.plot_visited_connect(self.V1, self.V2)
                self.plot_vertex_new()
                self.plot_path(self.path, 'lightcoral')
                self.plot_path_in_BSline(self.path, self.bs_degree, 'gold')
                plt.show()
            else:
                print("Trimming Invalid Nodes ...")
                self.TrimRRT()

                plt.cla()
                self.plot_grid("Dynamic_RRT_CONNECT")
                self.plot_visited(animation=False)
                self.plot_path(self.path, 'lightcoral')
                self.plot_path_in_BSline(self.path, self.bs_degree, 'gold')

            self.fig.canvas.draw_idle()

    # 无效点函数：发现无效边和无效点，并给予INVALID标记
    def InvalidateNodes(self):
        for edge in self.edges:
            if (not edge.n1 or edge.n1.flag == "INVALID") or (not edge.n2 or edge.n2.flag == "INVALID"): # 有任一点无效则边无效
                edge.flag = "INVALID"
                continue
            if (self.is_collision_obs_add(edge.n1, edge.n2)):
                edge.flag = "INVALID"
                if (edge.n1.parent == edge.n2 or edge.n2.child == edge.n1):
                    edge.n1.parent = None
                    edge.n2.child = None
                elif (edge.n1.child == edge.n2 or edge.n2.parent == edge.n1):
                    edge.n1.child = None
                    edge.n2.parent = None

        for i in range(1, len(self.vertex)):
            node = self.vertex[i]
            node.edge = []
            if (node.parent and node.parent.flag == "INVALID"):
                node.parent = None
            if (node.child and node.child.flag == "INVALID"):
                node.child = None
            if (not node.parent) and (not node.child): # 既无父节点又无子节点的点无效
                node.flag = "INVALID"
                

    # 路径有效检查：如果过程点有无效的，则路径无效
    def is_path_invalid(self):
        for node in self.waypoint:
            if node.flag == "INVALID":
                return True

    # 障碍共线检查：检查添加的obs_add是否与 start/end两点组成的线段区域重合。
    def is_collision_obs_add(self, start, end):
        delta = self.utils.delta
        obs_add = self.obs_add

        if math.hypot(start.x - obs_add[0], start.y - obs_add[1]) <= obs_add[2] + delta:
            return True     

        if math.hypot(end.x - obs_add[0], end.y - obs_add[1]) <= obs_add[2] + delta:
            return True     

        o, d = self.utils.get_ray(start, end)
        if self.utils.is_intersect_circle(o, d, [obs_add[0], obs_add[1]], obs_add[2]):
            return True     

        return False

    # 重新绘制函数：算法和首次绘制时基本相同，不再重复注释。重新绘制时不同的是：首次绘制V1和V2分别只有起点终点，
    #              而重新绘制时的V1 V2来自于TrimRRT函数回收的从起点终点分别出发的搜索树。
    #              （此函数无绘图，绘图在on_press完成）
    def replanning(self):
        
        self.TrimRRT()

        for i in range(self.iter_max):
            last_node, node_v1_new = self.rrt_connect_inuse()

            if self.is_node_same(last_node, node_v1_new):
                path, waypoint = self.extract_path(node_v1_new, last_node)
                self.path = path
                self.extract_waypoint = waypoint
                self.vertex = list(self.V1 + self.V2)
                self.edges = list(self.E1 + self.E2)
                print("path: ", len(path))
                print("waypoint: ", len(waypoint))
                return

            if len(self.V2) < len(self.V1):
                self.V1, self.V2 = self.V2, self.V1
                self.E1, self.E2 = self.E2, self.E1

        return None

    # 剪枝函数：去除存储的所有无效边和点，重新记录有效边和点。
    def TrimRRT(self):
        numNodeEdge = len(self.edges) + len(self.vertex);
        # 清理无效点
        for i in range(1, len(self.vertex)):
            node = self.vertex[i]
            node.edge = []
            if (node.parent and node.parent.flag == "INVALID"):
                node.parent = None
            if (node.child and node.child.flag == "INVALID"):
                node.child = None
            if (not node.parent) and (not node.child): # 既无父节点又无子节点的点无效
                node.flag = "INVALID"
  
        self.vertex = [node for node in self.vertex if node.flag == "VALID"]

        # 清理无效边
        for edge in self.edges:
            if (edge.n1 and edge.n1.flag == "INVALID"):
                edge.n1 = None
            if (edge.n2 and edge.n2.flag == "INVALID"):
                edge.n2 = None
            if (not edge.n1) or (not edge.n2): # 有任一点无效则边无效
                edge.flag = "INVALID"

        self.edges = [edge for edge in self.edges if edge.flag == "VALID"]

        if (numNodeEdge == len(self.edges) + len(self.vertex)): #无新增无效点边
            return

        #重新绘制树 并且根据起点和终点得到两棵不相连的搜索树（因为被障碍物阻断。）
        for edge in self.edges:
            edge.n1.edge.append(edge)
            edge.n2.edge.append(edge)

        for node in self.vertex:
            node.visit = False

        self.edges_coor.clear()
        self.gatherNodeTree(self.s_start, self.V1, self.E1, self.edges_coor)
        self.gatherNodeTree(self.s_goal, self.V2, self.E2, self.edges_coor)
        self.vertex = [node for node in self.vertex if node.visit == True]
        self.vertex_old = copy.deepcopy(self.vertex) #vertex_old/vertex_new/edges_coor仅用于绘图
        self.vertex_new = []
      

        if len(self.V2) < len(self.V1):
            self.V1, self.V2 = self.V2, self.V1
            self.E1, self.E2 = self.E2, self.E1

    # 搜集节点树：从search_node出发，搜索可以到达的节点存到vertex_list中，而边存放到edge_list中。
    @staticmethod
    def gatherNodeTree(search_node, vertex_list, edge_list, edges_coor):
        vertex_list.clear()
        edge_list.clear()
        qNode = QueueFIFO.QueueFIFO()
        qNode.put(search_node)
        while(not qNode.empty()):
            # 根据队列里弹出的点，遍历它的边，边上另一个点加入队列。
            node = qNode.get()
            node.visit = True
            vertex_list.append(node)
            for edge in node.edge:
                if edge.flag != "VALID":
                    continue

                collectNode = None
                if edge.n1 == node:
                    collectNode = edge.n2
                elif edge.n2 == node:
                    collectNode = edge.n1
                else:
                    continue

                if collectNode.flag == "VALID" and collectNode.visit == False:
                    node.child = collectNode
                    collectNode.parent = node
                    qNode.put(collectNode)
                    edge_list.append(edge)
                    edges_coor.append([[node.x, collectNode.x], [node.y, collectNode.y]])

        return

    # 生成随机点：生成区域范围内一个随机点
    def generate_random_node(self, goal_sample_rate):
        delta = self.utils.delta #来自utils.py里的delta值（现为0.5）

        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return self.s_goal

    # 重新生成路径上的随机点：生成区域范围内一个随机点，但是有一定概率随机取路径上已有的点
    def generate_random_node_replanning(self, goal_sample_rate, waypoint_sample_rate):
        delta = self.utils.delta
        p = np.random.random()

        if p < goal_sample_rate:
            return self.s_goal
        elif goal_sample_rate < p < goal_sample_rate + waypoint_sample_rate:
            return self.waypoint[np.random.randint(0, len(self.waypoint) - 1)]
        else:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

    # 核心搜索算法：
    def rrt_connect_inuse(self):
        #根据V1点集和最近邻的方法，随机生成新点node_v1_new
        node_rand = self.generate_random_node(self.goal_sample_rate)
        node_v1_near = self.nearest_neighbor(self.V1, node_rand)
        node_v1_new = self.new_state(node_v1_near, node_rand)
        last_node = Node([0,0])

        if node_v1_new and not self.utils.is_collision(node_v1_near, node_v1_new):
            #若新点node_v1_new没有被障碍物阻挡，则加入V1中
            self.V1.append(node_v1_new)
            self.E1.append(Edge(node_v1_near, node_v1_new))
            self.vertex_new.append(node_v1_new)

            #根据V2点集和最近邻的方法，随机生成新点node_v2_prim
            node_nearest_v2tov1_new = self.nearest_neighbor(self.V2, node_v1_new)
            node_v2_prim = self.new_state(node_nearest_v2tov1_new, node_v1_new)
            last_node = node_v2_prim
            if node_v2_prim and not self.utils.is_collision(node_v2_prim, node_nearest_v2tov1_new):
                #若新点node_v2_prim没有被障碍物阻挡，加入V2中
                self.V2.append(node_v2_prim)
                self.E2.append(Edge(node_nearest_v2tov1_new, node_v2_prim))
                self.vertex_new.append(node_v2_prim)
                #接下来尝试直接从node_v2_prim向node_v1_new连线，
                #若连线没有被阻挡，我们可以通过许多个该方向的线段完成任务，每个节点都会临时成为node_v2_prim_2
                    
                while last_node:
                    saved_child_last_node = last_node.child
                    node_v2_prim_iter = self.new_state(last_node, node_v1_new)
                    if node_v2_prim_iter and not self.utils.is_collision(node_v2_prim_iter, node_v2_prim):
                        node_v2_prim_for_path = Node((node_v2_prim_iter.x, node_v2_prim_iter.y))
                        self.V2.append(node_v2_prim_for_path)
                        self.E2.append(Edge(last_node, node_v2_prim_for_path))
                        self.vertex_new.append(node_v2_prim_for_path)
                        node_v2_prim_for_path.parent, last_node.child = last_node, node_v2_prim_for_path
                        last_node = node_v2_prim_for_path # 注意我们不能用node_v2_prim_iter，它是循环变量，下次循环需要从node_v2_prim_for_path开始
                    else:
                        last_node.child = saved_child_last_node
                        break
                    #如果刚刚画出来的node_v2_prim_iter （node_v2_prim_iter=node_v2_prim）就是node_v1_new， 意味着任务完成
                    if self.is_node_same(last_node, node_v1_new):
                        break

        return last_node, node_v1_new

    # 最近邻点函数：返回node_list中离点n最近的点
    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    # 同点函数：检查node1m和node2是否坐标相同
    @staticmethod
    def is_node_same(node1, node2):
        if node1.x == node2.x and node1.y == node2.y:
            return True
        return False

    # 生成新点：有时候随机生成的点会远于step_len，
    #          这样我们就要在这个方向上走step_len距离，取得新点，而不是直接取随机生成的点。
    def new_state(self, node_start, node_end):
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start
        node_new.parent.child = node_new
        return node_new

    # 提取路径函数：获取从node_v1_new和node_v2_prim，从起点到终点的整条路径。
    @staticmethod
    def extract_path(node_v1_new, node_v2_prim):
        # node_v1_new和node_v2_prim是坐标相同的两个点。
        # 正常的路径应为：
        # [s_start, ..., node_v1_new] + [node_v2_prim, ..., s_goal]
        # 或者是（V1和V2有可能颠倒）：
        # [s_start, ..., node_v2_prim] + [node_v1_new, ..., s_goal]
        # 由于node_v1_new和 node_v2_prim到起点和终点有父子关系，通过child-parent指针可以获得链表。
        waypoint1 = [node_v1_new]
        path1 = [(node_v1_new.x, node_v1_new.y)]
        node_now = node_v1_new
        while node_now.parent is not None: #往父节点遍历，获得路径
            node_now = node_now.parent
            path1.append((node_now.x, node_now.y))
            waypoint1.append(node_now)

        #node_v1_new和node_v2_prim是坐标相同的两个点，去除重复的首项
        #waypoint2 = [node_v2_prim]
        #path2 = [(node_v2_prim.x, node_v2_prim.y)]
        waypoint2 = []
        path2 = []
        node_now = node_v2_prim
        while node_now.parent is not None: #往父节点遍历，获得路径
            node_now = node_now.parent
            path2.append((node_now.x, node_now.y))
            waypoint2.append(node_now)

        #首尾相接
        waypoint1[-1].child = waypoint2[0]
        waypoint2[0].parent = waypoint1[-1]

        return list(list(reversed(path1)) + path2), list(list(reversed(waypoint1)) + waypoint2)
        #注意返回值有两个，返回第一项path是坐标值的list，第二项waypoint的Node对象的List

    # 距离与角度函数：计算从点node_start到node_end的欧氏距离和角度
    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    # 绘制网格函数
    def plot_grid(self, name):
        for (ox, oy, w, h) in self.obs_boundary:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )

        for (ox, oy, w, h) in self.obs_rectangle:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )

        for (ox, oy, r) in self.obs_circle:
            self.ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )

        plt.plot(self.s_start.x, self.s_start.y, "bs", linewidth=3)
        plt.plot(self.s_goal.x, self.s_goal.y, "gs", linewidth=3)

        plt.title(name)
        plt.axis("equal")
        '''
    def plot_visited(self, animation=True):
        if animation:
            count = 0
            for node in self.vertex:
                count += 1
                for nextNode in [node.parent, node.child]:
                    if nextNode:
                        plt.plot([nextNode.x, node.x], [nextNode.y, node.y], "-g")
                        plt.gcf().canvas.mpl_connect('key_release_event',
                                                     lambda event:
                                                     [exit(0) if event.key == 'escape' else None])
                        if count % 10 == 0:
                            plt.pause(0.001)
        else:
            for node in self.vertex:
                for nextNode in [node.parent, node.child]:
                    if nextNode:
                        plt.plot([nextNode.x, node.x], [nextNode.y, node.y], "-g")
        '''
    def plot_visited(self, animation=True):
        if animation:
            count = 0
            for node in self.vertex:
                count += 1
                if node.parent:
                    if (self.is_node_same(node.parent, self.s_start) or self.is_node_same(node.parent, self.s_goal)):
                        continue
                    plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g")
                    plt.gcf().canvas.mpl_connect('key_release_event',
                                                    lambda event:
                                                    [exit(0) if event.key == 'escape' else None])
                    if count % 10 == 0:
                        plt.pause(0.001)
        else:
            for node in self.vertex:
                if node.parent:
                    if (self.is_node_same(node.parent, self.s_start) or self.is_node_same(node.parent, self.s_goal)):
                        continue
                    plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g")

    # 绘制旧节点函数
    def plot_vertex_old(self):
        for node in self.vertex_old:
            for nextNode in [node.parent, node.child]:
                if nextNode:
                    if (self.is_node_same(nextNode, self.s_start) or self.is_node_same(nextNode, self.s_goal)):
                        continue
                    plt.plot([nextNode.x, node.x], [nextNode.y, node.y], "-b")

    # 绘制新节点函数
    def plot_vertex_new(self):
        count = 0
        for node in self.vertex_new:
            count += 1
            if node.parent:
                if (self.is_node_same(node.parent, self.s_start) or self.is_node_same(node.parent, self.s_goal)):
                    continue
                plt.plot([node.parent.x, node.x], [node.parent.y, node.y], color='darkorange')
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event:
                                             [exit(0) if event.key == 'escape' else None])
                if count % 10 == 0:
                    plt.pause(0.001)

    # 绘制路径函数
    @staticmethod
    def plot_path(path, color='red', linewidth=2):
        plt.plot([x[0] for x in path], [x[1] for x in path], linewidth=linewidth, color=color)
        plt.pause(0.01)

    # 以B样条曲线绘制路径函数
    @staticmethod
    def plot_path_in_BSline(path, bs_degree, color='red', linewidth=2):
        d = bs_degree    #degree, k越大，曲线越逼近原始控制点
        t = []    #knots vector
        num = len(path)

        for i in range(num+d+1):
            if i <= d:
                t.append(0)
            elif i >= num:
                t.append(num-d)
            else:
                t.append(i-d)

        c1 = [x[0] for x in path]
        c2 = [x[1] for x in path]

        spl_x = BSpline(t, c1, d)
        spl_y = BSpline(t, c2, d)
        xx = np.linspace(0.0, num-d, 100)
        plt.plot(spl_x(xx), spl_y(xx), linewidth=linewidth, color=color)
        plt.pause(0.01)


def main():
    x_start = (2, 2)  # 起点
    x_goal = (49, 24)  # 终点

    drrt = DynamicRrtConnect(x_start, x_goal,
                             step_len = 0.8, goal_sample_rate = 0.05,
                             waypoint_sample_rate = 0.65, iter_max = 5000,
                             bs_degree = 10)
    path = drrt.planning()
    

if __name__ == '__main__':
    main()
