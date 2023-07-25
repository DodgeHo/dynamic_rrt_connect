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


# Node and Edge Structure
class Node:
    def __init__(self, n):
        self.x = n[0] # x,y coordinate
        self.y = n[1]
        self.parent = None # parent,child are the last node and next node 
        self.child = None
        self.flag = "VALID" # Validation Mark
        self.edge = [] # the edges own by this node
        self.visit = False # Traversal Mark

class Edge:
    def __init__(self, n_p, n_c):
        self.n1 = n_p # n1, n2 are two nodes of edge
        self.n2 = n_c
        self.flag = "VALID" # Validation Mark


# Dynamic Bidirectional Generation Obstacle Avoidance Algorithm
class DynamicRrtConnect:

    # Initialize class member variables and the required environment
    def __init__(self, s_start, s_goal,
                step_len = 0.8, goal_sample_rate = 0.05, waypoint_sample_rate = 0.65, iter_max = 5000, bs_degree = 3):
        # Initializes parameters
        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.waypoint_sample_rate = waypoint_sample_rate
        self.iter_max = iter_max
        self.bs_degree = bs_degree
        
        # Initialize in-using variables
        self.V1 = [self.s_start]
        self.V2 = [self.s_goal]
        self.vertex = []
        self.vertex_old = []
        self.vertex_new = []
        self.E1 = []
        self.E2 = []
        self.edges = []
        self.edges_coor = []

        # Initialize the graph environment and variables
        self.env = env.Env()
        self.plotting = plotting.Plotting(s_start, s_goal)
        self.utils = utils.Utils()
        self.fig, self.ax = plt.subplots()
        self.x_range = self.env.x_range 
        # Please note that the range here is specified by the init function in env.py.
        # It does not matter what's the start point or end point in the input.
        # If you want to change this, be careful to change both the drawing area in env and the judgment range in the on_press function.
        self.y_range = self.env.y_range
        # So does circle, rectangle, boundary
        self.obs_circle = self.env.obs_circle 
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary
        self.obs_add = [0, 0, 0]
        self.path = []
        self.waypoint = []

    # First-time draw function: The first draw rrt graph, the first-time draw and rrt_connect bidirectional generation algorithm has no result difference
    def planning(self):
        t1 = time.time() # set a timer
        for i in range(self.iter_max):
            last_node, node_v1_new = self.rrt_connect_inuse()

            if self.is_node_same(last_node, node_v1_new):
                # Line complete, extract path, plot.
                self.vertex, self.edges = list(self.V1 + self.V2), list(self.E1 + self.E2)
                self.path, self.waypoint = self.extract_path(node_v1_new, last_node)
                t2 = time.time() # set another timer
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

        # None is returned when iter_max times have been computed
        return None

    # Click event function: function that handles mouse clicks on an area of the screen
    def on_press(self, event):
        t1 = time.time() # timer
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
                t2 = time.time() #timer
                print('The program runs: %s seconds' % (t2 - t1))
                print("len_vertex: ", len(self.vertex))
                print("len_vertex_old: ", len(self.vertex_old))
                print("len_vertex_new: ", len(self.vertex_new))

                # clear everything, re-plot
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

    # INVALID point function: Finds invalid edges and invalid points and marks them invalid
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
            if (not node.parent) and (not node.child):
               # a node is invalid when it has no parents nor children
                node.flag = "INVALID"
                

    # Path validity check: If any of the process points are invalid, the path is invalid
    def is_path_invalid(self):
        for node in self.waypoint:
            if node.flag == "INVALID":
                return True

    # obstacles collinear check: check whether obs_add coincides with the line segment consisting of two points: start and end.
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

    # Re-Plan function: The algorithm is basically the same as when it was first drawn, no more repeated comments. The difference when redrawing is that V1 and V2 only have the starting point and the end point respectively for the first time.
    # And the V1 V2 in redrawing comes from the search tree recovered by the TrimRRT function from the starting point and the end point respectively.
    # (This function does not actually draw, the drawing is done in on_press)
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

    # Pruning function: removes all invalid edges and points stored and re-records valid edges and points.
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

    # Collect node tree: Starting from search_node, search reachable nodes are stored in vertex_list, and edges are stored in edge_list.
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

    # Generate random point
    def generate_random_node(self, goal_sample_rate):
        delta = self.utils.delta #来自utils.py里的delta值（现为0.5）

        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return self.s_goal

    # Regenerate random points on the path: Generate a random point in the range of the region, but there is a certain probability of randomly taking the existing points on the path
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

    # Core search algorithm:
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

    # Nearest neighbor function: Returns the point closest to point n in the node list
    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    # Same point function: check whether node1 and node2 have the same coordinates
    @staticmethod
    def is_node_same(node1, node2):
        if node1.x == node2.x and node1.y == node2.y:
            return True
        return False

    # Generate new points: Sometimes randomly generated points will be far away from step_len,
    # So we have to go step_len distance in this direction to get new points, instead of directly taking randomly generated points.
    def new_state(self, node_start, node_end):
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start
        node_new.parent.child = node_new
        return node_new

    # Extract the path function: Get the entire path from the start to the end from node_v1_new and node_v2_prim.
    @staticmethod
    def extract_path(node_v1_new, node_v2_prim):
        # node_v1_new and node_v2_prim are two points with same coordinates.
        # The correct path is:
        # [s_start, ..., node_v1_new] + [node_v2_prim, ..., s_goal]
        # Or（V1 and V2 may be in reversed order）：
        # [s_start, ..., node_v2_prim] + [node_v1_new, ..., s_goal]
        # Since node_v1_new and node_v2_prim have parent-child relationships to the start and end points, a linked list can be obtained through the child-parent pointer.
        waypoint1 = [node_v1_new]
        path1 = [(node_v1_new.x, node_v1_new.y)]
        node_now = node_v1_new
        while node_now.parent is not None: # Walk through the parent node to get the path
            node_now = node_now.parent
            path1.append((node_now.x, node_now.y))
            waypoint1.append(node_now)

        # node_v1_new and node_v2_prim are two points with the same coordinates, removing the duplicate first term
        # waypoint2 = [node_v2_prim]
        # path2 = [(node_v2_prim.x, node_v2_prim.y)]
        waypoint2 = []
        path2 = []
        node_now = node_v2_prim
        while node_now.parent is not None: # Walk through the parent node to get the path
            node_now = node_now.parent
            path2.append((node_now.x, node_now.y))
            waypoint2.append(node_now)

        # nose to tail
        waypoint1[-1].child = waypoint2[0]
        waypoint2[0].parent = waypoint1[-1]

        return list(list(reversed(path1)) + path2), list(list(reversed(waypoint1)) + waypoint2)
        # Note that there are two return values, the first list where path is the coordinate value, and the second List where waypoint is the Node object

    # Distance and Angle function: Calculates the Euclidean distance and Angle from node_start to node_end
    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    # Draw grid function
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

    # Draw old nodes function
    def plot_vertex_old(self):
        for node in self.vertex_old:
            for nextNode in [node.parent, node.child]:
                if nextNode:
                    if (self.is_node_same(nextNode, self.s_start) or self.is_node_same(nextNode, self.s_goal)):
                        continue
                    plt.plot([nextNode.x, node.x], [nextNode.y, node.y], "-b")

    # Draw new nodes function
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

    # Draw path function
    @staticmethod
    def plot_path(path, color='red', linewidth=2):
        plt.plot([x[0] for x in path], [x[1] for x in path], linewidth=linewidth, color=color)
        plt.pause(0.01)

    # Plot the path function with a B-spline curve
    @staticmethod
    def plot_path_in_BSline(path, bs_degree, color='red', linewidth=2):
        d = bs_degree    # degree, The larger k is,
                         # the closer the curve is to the original control point
        t = []           # knots vector
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
    x_start = (2, 2)  # start point
    x_goal = (49, 24)  # end point

    drrt = DynamicRrtConnect(x_start, x_goal,
                             step_len = 0.8, goal_sample_rate = 0.05,
                             waypoint_sample_rate = 0.65, iter_max = 5000,
                             bs_degree = 10)
    path = drrt.planning()
    

if __name__ == '__main__':
    main()
