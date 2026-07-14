# JetCarCloud API

默认地址：

```text
HTTP: http://<cloud-ip>:8000
WS:   ws://<cloud-ip>:8000
```

WSL2 场景下，手机和 Jetson 通常填写 Windows 主机的局域网 IP。

## 视频与算法

Edge 只在手机打开功能后才向 Cloud 推流：

```text
WS /ws/video/{car_id}/{stream_id}/edge?algorithm_ids=<ids>&include_image=true
```

常用算法：

```text
yolov5-manhole-detect
yolov8-road-damage
yolov5-similarity
```

处理后画面：

```text
GET /api/video/streams/{car_id}/{stream_id}/algorithms/{algorithm_id}/mjpeg?fps=5
```

手机接收结果：

```text
WS /ws/inference/{car_id}/app
```

## Edge 控制

Edge AI 控制端口默认：

```text
tcp://<edge-ip>:6001
```

路面巡检开关使用 mask：

```json
{"type":"jetcar_ai_control","mode":"road_inspection","car_id":"car_001","stream_id":"camera_front","mask":"TF"}
```

`TF` 表示井盖开、路面缺陷关；`TT` 两个都开；`FF` 全关并停止推流。

相似度寻物：

```json
{"type":"jetcar_ai_control","mode":"similarity","car_id":"car_001","stream_id":"camera_front","algorithm_ids":["yolov5-similarity"]}
```

关闭：

```json
{"type":"jetcar_ai_control","mode":"off","car_id":"car_001","stream_id":"camera_front","algorithm_ids":[]}
```

## 自动任务控制

Edge 自动任务端口默认：

```text
tcp://<edge-ip>:6002
```

该端口由 `jetcar_edge task_orchestrator_node` 提供。它假设 Nav2 已启动并暴露：

```text
Action: /navigate_to_pose
Pose:   /amcl_pose
Cmd:    /cmd_vel
```

前往目标点：

```json
{"type":"jetcar_task_control","mode":"navigate_to_point","car_id":"car_001","stream_id":"camera_front","x":1.0,"y":0.5,"yaw":0.0}
```

自动巡检。Edge 会按 `waypoints.yaml` 的 `inspection` 路径导航，并打开井盖+路面缺陷检测：

```json
{"type":"jetcar_task_control","mode":"inspection_task","car_id":"car_001","stream_id":"camera_front"}
```

地图寻物。手机需先上传目标图到 Cloud，然后发该命令。Edge 会按 `waypoints.yaml` 的 `search` 路径导航，并打开 similarity 算法；Cloud 匹配后 Edge 再执行最后对准/靠近：

```json
{"type":"jetcar_task_control","mode":"similarity_search_task","car_id":"car_001","stream_id":"camera_front"}
```

停止任务：

```json
{"type":"jetcar_task_control","mode":"stop_task","car_id":"car_001","stream_id":"camera_front"}
```

临时更新 Edge 内存中的 waypoint，不需要重新编译：

```json
{
  "type": "jetcar_task_control",
  "mode": "set_waypoints",
  "car_id": "car_001",
  "stream_id": "camera_front",
  "name": "inspection",
  "waypoints": [
    {"label":"p1","x":0.0,"y":0.0,"yaw":0.0,"hold_seconds":1.0},
    {"label":"p2","x":1.0,"y":0.0,"yaw":0.0,"hold_seconds":1.0}
  ]
}
```

查看 Edge 当前内存 waypoint：

```json
{"type":"jetcar_task_control","mode":"list_waypoints","car_id":"car_001","stream_id":"camera_front"}
```

任务状态由 Edge 发布到 `/jetcar/task_status`，再由上传节点上报 Cloud：

```text
POST /api/edge/events
```

Cloud 查询最新任务状态：

```text
GET /api/tasks/{car_id}/{stream_id}/latest
```

## Similarity Session

上传目标图并生成目标特征：

```text
POST /api/similarity/search/start
```

请求：

```json
{
  "car_id": "car_001",
  "stream_id": "camera_front",
  "algorithm_id": "yolov5-similarity",
  "threshold": 0.70,
  "image": {"encoding":"jpeg","width":1280,"height":720,"data":"base64"}
}
```

停止 session：

```text
POST /api/similarity/search/stop
```

## 报告与调试

生成当前缓存结果报告：

```text
POST /api/tasks/report
```

请求：

```json
{"car_id":"car_001","stream_id":"camera_front","task_id":"inspection-demo","mode":"inspection_task","summary":{}}
```

Cloud 会保存到 `.jetcar_reports/`。

列出报告：

```text
GET /api/tasks/reports?car_id=car_001&stream_id=camera_front
```

报告响应会返回 `report_url`，例如：

```text
/api/tasks/reports/car_001/camera_front/inspection_task-123/index.html
```

报告目录中会保存：

```text
report.json
index.html
images/*.jpg
```

地图文件列表：

```text
GET /api/maps
GET /api/maps/{map_id}
GET /api/maps/{map_id}/image
```

调试页面：

```text
GET /dashboard
GET /unicorn
GET /api/dashboard/state
```

调试目录：

```text
.jetcar_debug
.jetcar_algorithm_runs
.jetcar_reports
.jetcar_maps
```

## 上车后必须确认

在 Jetson 容器中确认 Nav2 名称：

```bash
source /opt/ros/foxy/setup.bash
source /workspace/install/setup.bash
ros2 action list
ros2 action info /navigate_to_pose
ros2 topic list | grep -E 'amcl|pose|map|tf|goal|cmd_vel'
```

如果 action/topic 名称不同，只改 `edge.yaml` 或 launch 参数：

```text
navigate_action
amcl_pose_topic
cmd_vel_topic
waypoints_file
```
