import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import open3d as o3d
import rosbag
from sensor_msgs.msg import PointCloud2, Image, CompressedImage
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import matplotlib.cm as cm

def read_first_message(bag, topic):
    """读取 ROS bag 中指定话题的第一条消息"""
    for topic, msg, t in bag.read_messages(topics=[topic]):
        return msg

# 读取 rosbag 文件
bag_path = "../sparkal2_10_14_crop.bag"
lidar_topic = "/sparkal2/lidar_points"
image_topic = "/sparkal2/forward/color/image_raw/compressed"

is_colorful = 0  # 图像为彩色图像
is_compressed = 1  # 图像为压缩格式

bag = rosbag.Bag(bag_path)
lidar_msg = read_first_message(bag, lidar_topic)
image_msg = read_first_message(bag, image_topic)

# 将点云消息转换为 numpy 数组
points = []
for point in pc2.read_points(lidar_msg, skip_nans=True):
    points.append([point[0], point[1], point[2]])
point_cloud = np.array(points)

# 将图像消息转换为 OpenCV 格式
bridge = CvBridge()
if is_compressed:
    image = bridge.compressed_imgmsg_to_cv2(image_msg)
else:
    image = bridge.imgmsg_to_cv2(image_msg)

# # 相机内参和畸变参数（示例数据）
# K = np.array([
#     [377.229220831, 0.0, 326.351864976],
#     [0.0, 377.486565843, 239.659665361],
#     [0.0, 0.0, 1.0]
# ])
# D = np.array([-0.00439906, -0.00467669,  0.00017386,  0.00324217, 0.0])

# # lidar坐标到相机坐标的外参矩阵
# t_lidar_to_cam = np.array([
#     [-0.02904653, -0.99957706, -0.00171542,  0.10047405],
#     [-0.06927801,  0.00372514, -0.99759064, -0.10898666],
#     [ 0.99717459, -0.02885769, -0.06935687, -0.06544693],
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

# 提取旋转矩阵（3x3部分）
rotation_matrix = t_lidar_to_cam[:3, :3]
r = R.from_matrix(rotation_matrix)
euler_angles = r.as_euler('xzy', degrees=True)

# 输出原始的欧拉角
print("原始欧拉角 (Roll, Pitch, Yaw)：")
print(euler_angles)

# 图像去畸变
h, w = image.shape[:2]
new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
undistorted_image = cv2.undistort(image, K, D, None, new_camera_matrix)

#undistorted_image = image

# 转换灰度图像为彩色图像（3 通道）
# undistorted_image = cv2.cvtColor(undistorted_image, cv2.COLOR_GRAY2BGR)

# 创建 colormap 映射深度值到颜色
colormap = cm.get_cmap('jet')

# 点云投影到图像平面，并将深度信息显示在 undistorted_image 上
def project_lidar_to_image():
    global t_lidar_to_cam, undistorted_image  # 使用全局变量
    refined_img = undistorted_image.copy()
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
                refined_img[v, u] = color  # 将深度颜色叠加到图像上

    # 展示叠加了深度投影的图像
    cv2.imshow("Projected Depth on Image", refined_img)
    cv2.waitKey(0)  # 等待1ms并刷新图像
    #cv2.destroyAllWindows()

# 初始图像投影
project_lidar_to_image()

# 通过键盘调整欧拉角
print("\n请通过键盘输入想要调整的欧拉角：xzy")
print("输入 'q' 来调整 [0]，'w' 来调整 [1],'e' 来调整 [2]，输入 'd' 退出调整模式")

while True:
    key = input("请输入要调整的轴 (q/w/e):xzy,d退出")
    if key.lower() == 'q':
        delta = float(input("请输入[0]调整的角度（单位：度）: "))
        euler_angles[0] += delta
    elif key.lower() == 'w':
        delta = float(input("请输入[1]调整的角度（单位：度）: "))
        euler_angles[1] += delta
    elif key.lower() == 'e':
        delta = float(input("请输入[2]调整的角度（单位：度）: "))
        euler_angles[2] += delta
    elif key.lower() == 'd':
        print("退出调整模式")
        break
    else:
        print("无效输入，请输入 'q', 'w', 'e' 或 'd' 来退出。")

    # 更新旋转矩阵
    r_adjusted = R.from_euler('xzy', euler_angles, degrees=True)
    rotation_matrix_adjusted = r_adjusted.as_matrix()

    # 更新外参矩阵
    t_lidar_to_cam[:3, :3] = rotation_matrix_adjusted

    # 输出更新后的欧拉角和旋转矩阵
    print("\n更新后的欧拉角 (Roll, Pitch, Yaw)：")
    print(euler_angles)
    print("\n更新后的旋转矩阵：")
    print(rotation_matrix_adjusted)

    # 每次调整后立即重新进行点云投影
    project_lidar_to_image()

# 关闭 rosbag
bag.close()
