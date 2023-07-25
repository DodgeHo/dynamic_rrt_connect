# Dynamic_RRT_Connect

# Python3 implementation of Bidirectional Generation Dynamic Obstacle Avoidance Algorithm



Written by: Dodge(Lang HE) asdsay@gmail.com
Updated date: 2023-07-24

The (dynamic_rrt_connect) is a fusion of dynamic-RRT algorithm and RT-CONNECT algorithm.
It implements:
(1) Bidirectional growth of branches, while retaining the original function of dynamic-rrt to dynamically add obstacles and plan the path again.
(2) After the completion of the first path planning plot display, you can add obstacles by clicking the mouse, and then re-plan the path;
(3) The artificial potential field enhances the target bias of path planning and speeds up path planning; B-spline curve is used to optimize the path curve.
An adaptive step strategy based on artificial potential field is designed to divide space. After selecting the growth direction of the expanded tree, the step value can be adjusted adaptively according to the spatial position of the artificial potential field where the sampling point is located, so as to efficiently grow the expanded tree and reduce the planning time.



Sample:


![](.\Sample.gif)
