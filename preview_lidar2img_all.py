import numpy as np
import cv2
import rosbag
from sensor_msgs.msg import PointCloud2, CompressedImage
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import matplotlib.cm as cm

def find_closest_msg(target_time, msgs, max_diff=0.05):
    """
    找到与目标时间最接近的消息
    :param target_time: 目标时间（秒）
    :param msgs: 消息队列 [(msg, time), ...]
    :param max_diff: 最大允许的时间差（秒）
    :return: 最接近的消息或 None
    """
    closest_msg = None
    min_diff = float('inf')
    for msg, time in msgs:
        diff = abs((time - target_time).to_sec())
        if diff < min_diff and diff <= max_diff:
            min_diff = diff
            closest_msg = msg
    return closest_msg

# 读取 rosbag 文件
bag_path = "../sparkal2_10_14_crop.bag"
lidar_topic = "/sparkal2/lidar_points"
image_topic = "/sparkal2/forward/color/image_raw/compressed"

# 初始化 ROS bag 和 CvBridge
bag = rosbag.Bag(bag_path)
bridge = CvBridge()

# # 相机内参和畸变参数（示例数据）
# K = np.array([
#     [377.229220831, 0.0, 326.351864976],
#     [0.0, 377.486565843, 239.659665361],
#     [0.0, 0.0, 1.0]
# ])
# D = np.array([-0.00439906, -0.00467669,  0.00017386,  0.00324217, 0.0])

# # lidar坐标到相机坐标的外参矩阵
# t_lidar_to_cam = np.array([
#     #origin
#     # [-0.02904653, -0.99957706, -0.00171542,  0.10047405],
#     # [-0.06927801,  0.00372514, -0.99759064, -0.10898666],
#     # [ 0.99717459, -0.02885769, -0.06935687, -0.06544693],
#     # [ 0.,          0.,          0.,          1.        ]
#     [-0.04997608, -0.99874977, -0.0011341 , 0.10047405],
#     [-0.05185702,  0.00372884, -0.99864756, -0.10898666],
#     [ 0.99740325, -0.04984968, -0.05197854, -0.06544693],
#     [ 0.,          0.,          0.,          1.        ]
# ])

# # sparkal1
# K = np.array([
#     [380.8096923828125, 0.0, 315.84698486328125], 
#     [0.0, 380.5378723144531, 238.04495239257812], 
#     [0.0, 0.0, 1.0]
# ])
# D = np.array([-0.054963257163763046, 0.06448927521705627, 0.00020229471556376666, 0.00045873370254412293, -0.02038593403995037])

# # lidar坐标到相机坐标的外参矩阵
# t_lidar_to_cam = np.array([
#     [-0.01917113, -0.99968097, -0.01644461, 0.02],
#     [-0.05233596,  0.01742848, -0.99847744, 0.1],
#     [ 0.9984455 , -0.0182813 , -0.05265338, -0.09],
#     [ 0.,    0.,    0.,    1.  ]
# ])

#sparkal2
K = np.array([
    [380.8096923828125, 0.0, 315.84698486328125], 
    [0.0, 380.5378723144531, 238.04495239257812], 
    [0.0, 0.0, 1.0]
])
D = np.array([-0.054963257163763046, 0.06448927521705627, 0.00020229471556376666, 0.00045873370254412293, -0.02038593403995037])

# lidar坐标到相机坐标的外参矩阵
t_lidar_to_cam = np.array([
    [-0.01912439, -0.99969264, -0.01577626, 0.02],
    [-0.08715574,  0.01738598, -0.99604297, 0.1],
    [ 0.99601111, -0.01767372, -0.08746145, -0.09],
    [ 0.,    0.,    0.,    1.  ]
])

# 创建 colormap 映射深度值到颜色
colormap = cm.get_cmap('jet')

# 读取所有消息
lidar_msgs = [(msg, t) for _, msg, t in bag.read_messages(topics=[lidar_topic])]
image_msgs = [(msg, t) for _, msg, t in bag.read_messages(topics=[image_topic])]

# 显示消息总数
print(f"Lidar messages: {len(lidar_msgs)}")
print(f"Image messages: {len(image_msgs)}")

# 遍历点云消息
for lidar_msg, lidar_time in lidar_msgs:
    # 查找与点云时间戳最接近的图像消息
    image_msg = find_closest_msg(lidar_time, image_msgs)
    if image_msg is None:
        continue  # 如果没有找到匹配的图像，跳过

    # 处理点云数据
    points = []
    for point in pc2.read_points(lidar_msg, skip_nans=True):
        points.append([point[0], point[1], point[2]])
    point_cloud = np.array(points)

    # 处理图像数据
    if image_msg._type == 'sensor_msgs/CompressedImage':
        image = bridge.compressed_imgmsg_to_cv2(image_msg)
    else:
        image = bridge.imgmsg_to_cv2(image_msg)

    # 图像去畸变
    h, w = image.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(image, K, D, None, new_camera_matrix)

    # 转换灰度图像为彩色图像（3 通道）
    # undistorted_image = cv2.cvtColor(undistorted_image, cv2.COLOR_GRAY2BGR)

    # 点云投影到图像平面，并将深度信息显示在 undistorted_image 上
    for point in point_cloud:
        # 转换到齐次坐标
        point_h = np.append(point, 1)
        cam_coords = np.dot(t_lidar_to_cam, point_h)

        # 投影到图像平面
        if cam_coords[2] > 0:  # 深度值为正
            img_coords = np.dot(K, cam_coords[:3])
            img_coords /= img_coords[2]
            u, v = int(img_coords[0]), int(img_coords[1])

            # 检查是否在图像范围内
            if 0 <= u < w and 0 <= v < h:
                depth = cam_coords[2]  # 提取深度值
                normalized_depth = np.clip(depth / 20.0, 0, 1)  # 深度归一化到 [0, 1]
                color = np.array(colormap(normalized_depth)[:3]) * 255  # 使用 colormap 映射为彩色
                color = color.astype(np.uint8)
                undistorted_image[v, u] = color  # 将深度颜色叠加到图像上

    # 展示叠加了深度投影的图像
    cv2.imshow("Projected Depth on Image", undistorted_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
        break

# 关闭资源
cv2.destroyAllWindows()
bag.close()
