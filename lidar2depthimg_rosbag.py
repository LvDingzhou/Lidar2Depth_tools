import numpy as np
import cv2
from tqdm import tqdm
import rosbag
from sensor_msgs.msg import PointCloud2, CompressedImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import matplotlib.cm as cm
import rospy

def find_closest_msg(target_time, msgs, max_diff=0.05):
    """
    找到与目标时间最接近的消息
    :param target_time: 目标时间（秒）
    :param msgs: 消息队列 [(msg, time), ...]
    :param max_diff: 最大允许的时间差（秒）
    :return: 最接近的消息或 None
    """
    closest_msg = None
    closest_time = None
    min_diff = float('inf')
    for msg,time in msgs:
        diff = abs((msg.header.stamp - target_time).to_sec())
        # print(f"Comparing: target={target_time.to_sec()}, msg={time.to_sec()}, diff={diff}")
        if diff < min_diff and diff <= max_diff:
            min_diff = diff
            closest_msg = msg
            closest_time = time
    return closest_msg, closest_time
    # return closest_msg

# ROS bag 文件路径和话题
bag_path = "../sparkal1_10_14_crop.bag"
lidar_topic = "/sparkal1/lidar_points"
image_topic = "/sparkal1/forward/color/image_raw/compressed"

# 初始化 ROS bag 和 CvBridge
bag = rosbag.Bag(bag_path)
bridge = CvBridge()

# # acl_jackal2
# # 相机内参和畸变参数（示例数据）
# K = np.array([
#     [377.229220831, 0.0, 326.351864976],
#     [0.0, 377.486565843, 239.659665361],
#     [0.0, 0.0, 1.0]
# ])
# D = np.array([-0.00439906, -0.00467669,  0.00017386,  0.00324217, 0.0])

# # lidar坐标到相机坐标的外参矩阵
# t_lidar_to_cam = np.array([
#     [-0.04997608, -0.99874977, -0.0011341 , 0.10047405],
#     [-0.05185702,  0.00372884, -0.99864756, -0.10898666],
#     [ 0.99740325, -0.04984968, -0.05197854, -0.06544693],
#     [ 0.,          0.,          0.,          1.        ]
# ])

# sparkal1
K = np.array([
    [380.8096923828125, 0.0, 315.84698486328125], 
    [0.0, 380.5378723144531, 238.04495239257812], 
    [0.0, 0.0, 1.0]
])
D = np.array([-0.054963257163763046, 0.06448927521705627, 0.00020229471556376666, 0.00045873370254412293, -0.02038593403995037])

# lidar坐标到相机坐标的外参矩阵
t_lidar_to_cam = np.array([
    [-0.01917113, -0.99968097, -0.01644461, 0.02],
    [-0.05233596,  0.01742848, -0.99847744, 0.1],
    [ 0.9984455 , -0.0182813 , -0.05265338, -0.09],
    [ 0.,    0.,    0.,    1.  ]
])

# #sparkal2
# K = np.array([
#     [380.8096923828125, 0.0, 315.84698486328125], 
#     [0.0, 380.5378723144531, 238.04495239257812], 
#     [0.0, 0.0, 1.0]
# ])
# D = np.array([-0.054963257163763046, 0.06448927521705627, 0.00020229471556376666, 0.00045873370254412293, -0.02038593403995037])

# # lidar坐标到相机坐标的外参矩阵
# t_lidar_to_cam = np.array([
#     [-0.01912439, -0.99969264, -0.01577626, 0.02],
#     [-0.08715574,  0.01738598, -0.99604297, 0.1],
#     [ 0.99601111, -0.01767372, -0.08746145, -0.09],
#     [ 0.,    0.,    0.,    1.  ]
# ])

# 读取所有消息
lidar_msgs = [(msg, t) for _, msg, t in bag.read_messages(topics=[lidar_topic])]
image_msgs = [(msg, t) for _, msg, t in bag.read_messages(topics=[image_topic])]

# 显示消息总数
print(f"Lidar messages: {len(lidar_msgs)}")
print(f"Image messages: {len(image_msgs)}")

# 创建新bag文件路径
output_bag_path = "output_with_depth_and_original.bag"

# 打开新文件并写入数据
with rosbag.Bag(output_bag_path, 'w') as out_bag:
    # 复制原始bag文件中的所有话题
    for topic, msg, t in tqdm(bag.read_messages(), desc="read bag msgs"):
        if topic == '/sparkal1/forward/depth/image_rect_raw':
            continue  # 跳过该话题
        out_bag.write(topic, msg, t)  # 写入其他话题和消息

    # 添加新的深度图话题
    prev_depth_msg = None
    prev_time = None
    for idx, (lidar_msg, lidar_time) in enumerate(tqdm(lidar_msgs, desc="Processing LiDAR messages")):
        # 查找与点云时间戳最接近的图像消息
        image_msg, image_time = find_closest_msg(lidar_msg.header.stamp, image_msgs)
        # image_msg = find_closest_msg(lidar_msg.header.stamp, image_msgs)
        if image_msg is None:
            print(f"No matching image for LiDAR time {lidar_time.to_sec()}")
            continue  # 如果没有找到匹配的图像，跳过

        # 处理点云数据和生成深度图
        points = []
        for point in pc2.read_points(lidar_msg, skip_nans=True):
            points.append([point[0], point[1], point[2]])
        point_cloud = np.array(points)

        if image_msg._type == 'sensor_msgs/CompressedImage':
            image = bridge.compressed_imgmsg_to_cv2(image_msg)
        else:
            image = bridge.imgmsg_to_cv2(image_msg)

        h, w = image.shape[:2]
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
        undistorted_image = cv2.undistort(image, K, D, None, new_camera_matrix)

        depth_map = np.zeros((h, w), dtype=np.float32)
        for point in point_cloud:
            point_h = np.append(point, 1)
            cam_coords = np.dot(t_lidar_to_cam, point_h)
            if cam_coords[2] > 0:
                img_coords = np.dot(K, cam_coords[:3])
                img_coords /= img_coords[2]
                u, v = int(np.round(img_coords[0])), int(np.round(img_coords[1]))
                if 0 <= u < w and 0 <= v < h:
                    depth = cam_coords[2]
                    if depth_map[v, u] == 0 or depth < depth_map[v, u]:
                        depth_map[v, u] = depth

        depth_msg = Image()
        # depth_msg.header.stamp = image_time
        depth_msg.header = image_msg.header
        # depth_msg.header.frame_id = "camera_depth_frame"
        depth_msg.height = depth_map.shape[0]
        depth_msg.width = depth_map.shape[1]
        depth_msg.encoding = "32FC1"
        depth_msg.is_bigendian = 0
        depth_msg.step = depth_map.shape[1] * 4
        depth_msg.data = depth_map.tobytes()

        # 写入当前帧深度图
        out_bag.write('/sparkal1/forward/depth/image_rect_raw', depth_msg, image_time)
        print(f"Writing depth message to bag at {depth_msg.header.stamp.to_sec()}")
        print(f"Writing depth timestamp to bag at {image_time.to_sec()}")

# <<<如果需要插帧操作<<<
        # # 如果存在上一帧，插入两帧中间插值
        # if prev_depth_msg is not None:
        #     time_diff = (image_time - prev_time).to_sec() #如果需要更改插值
        #     for i in range(1, 3):  # 插入两帧
        #         interpolated_time = prev_time + rospy.Duration(i * time_diff / 3)
        #         interpolated_msg = prev_depth_msg  # 沿用前一帧数据
        #         interpolated_msg.header.stamp = prev_depth_msg.header.stamp
        #         out_bag.write('/sparkal1/forward/depth/image_rect_raw', interpolated_msg, interpolated_time)
        #         print(f"Writing depth message to bag at {interpolated_msg.header.stamp.to_sec()}")
        #         print(f"Inserted interpolated depth timestamp to bag at {interpolated_time.to_sec()}")

        # # 更新上一帧数据
        # prev_depth_msg = depth_msg
        # prev_time = image_time #如果需要更改插值
# >>>如果需要插帧操作>>>

print(f"新ROS bag文件已创建: {output_bag_path}")

# 关闭资源
cv2.destroyAllWindows()
bag.close()
