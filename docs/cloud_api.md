# JetCarCloud 接口说明

局域网地址：

```text
HTTP: http://<cloud-ip>:8000
WS:   ws://<cloud-ip>:8000
```

WSL2 运行时，`<cloud-ip>` 通常填写 Windows 主机的局域网 IP。

## 1. 路面巡检链路

路面巡检由手机端两个开关联动控制：

```text
mask[0] = 井盖检测 yolov5-manhole-detect
mask[1] = 路面缺陷 yolov8-road-damage
```

示例：

```text
TF -> 只开井盖检测
FT -> 只开路面缺陷检测
TT -> 两个模型都开
FF -> 全部关闭，Edge 断开 Cloud 推流，Cloud 不再触发算法
```

手机端把 mask 发给小车端 Edge 的 AI 控制端口。Edge 根据 mask 更新它连接 Cloud 的 WebSocket：

```text
WS /ws/video/{car_id}/{stream_id}/edge?algorithm_ids=yolov5-manhole-detect&include_image=true
WS /ws/video/{car_id}/{stream_id}/edge?algorithm_ids=yolov8-road-damage&include_image=true
WS /ws/video/{car_id}/{stream_id}/edge?algorithm_ids=yolov5-manhole-detect,yolov8-road-damage&include_image=true
```

Edge 发给 Cloud 的每帧 JSON：

```json
{
  "car_id": "car_001",
  "image": {
    "encoding": "jpeg",
    "width": 640,
    "height": 480,
    "data": "base64-encoded-jpeg"
  }
}
```

Cloud 会按 `VIDEO_PUSH_MIN_INTERVAL_MS`、`ALGORITHM_MIN_INTERVAL_MS` 和
`ALGORITHM_MAX_CONCURRENT_TASKS` 限流。结果通过手机端结果 WebSocket 返回。

## 2. Edge AI 控制端口

JetCarEdge 额外提供一个轻量 TCP 控制端口，默认：

```text
tcp://<edge-ip>:6001
```

每条消息是一行 JSON，Edge 返回一行 JSON。路面巡检示例：

```json
{"type":"jetcar_ai_control","mode":"road_inspection","car_id":"car_001","stream_id":"camera_front","mask":"TF"}
```

关闭所有 AI：

```json
{"type":"jetcar_ai_control","mode":"off","car_id":"car_001","stream_id":"camera_front","mask":"FF","algorithm_ids":[]}
```

相似度寻物：

```json
{"type":"jetcar_ai_control","mode":"similarity","car_id":"car_001","stream_id":"camera_front","algorithm_ids":["yolov5-similarity"]}
```

Edge 收到空算法列表或 `FF` 时应停止上传并断开 Cloud WebSocket，用于省电和避免无意义推理。

## 3. 手机端结果通道

手机端连接：

```text
WS /ws/inference/{car_id}/app
```

主要事件：

```json
{
  "type": "algorithm_result",
  "ok": true,
  "algorithm_id": "yolov8-road-damage",
  "car_id": "car_001",
  "stream_id": "camera_front",
  "runner": "local",
  "latency_ms": 1234.5,
  "result": {
    "detection_count": 1,
    "detections": []
  },
  "annotated_image": null,
  "error": ""
}
```

显示处理后画面：

```text
GET /api/video/streams/{car_id}/{stream_id}/algorithms/{algorithm_id}/mjpeg?fps=5
```

该接口只输出最新缓存的处理后画面，不主动触发算法。算法由 Edge 推帧触发。

## 4. 相似度寻物链路

相似度寻物和路面巡检是独立逻辑：

1. 手机端上传目标图片到 Cloud。
2. Cloud 保存本次寻物会话的目标图。
3. 手机端通知 Edge 开启 `yolov5-similarity`。
4. Edge 持续推摄像头帧到 Cloud。
5. Cloud 用每帧和目标图做相似度判断。
6. 一旦 `matched=true`，手机端结束任务，并通知 Edge/Cloud 停止。

启动会话：

```text
POST /api/similarity/search/start
```

请求：

```json
{
  "car_id": "car_001",
  "stream_id": "camera_front",
  "algorithm_id": "yolov5-similarity",
  "threshold": 0.45,
  "image": {
    "encoding": "jpeg",
    "width": 1280,
    "height": 720,
    "data": "base64-encoded-jpeg"
  }
}
```

响应：

```json
{
  "ok": true,
  "type": "similarity_search_session",
  "car_id": "car_001",
  "stream_id": "camera_front",
  "algorithm_id": "yolov5-similarity",
  "threshold": 0.45,
  "template_path": ".jetcar_algorithm_runs/similarity_sessions/car_001/camera_front/yolov5-similarity/target.jpg",
  "active": true,
  "edge_mask": "similarity",
  "edge_algorithm_ids": ["yolov5-similarity"]
}
```

停止会话：

```text
POST /api/similarity/search/stop
```

请求：

```json
{
  "car_id": "car_001",
  "stream_id": "camera_front",
  "algorithm_id": "yolov5-similarity"
}
```

相似度结果示例：

```json
{
  "type": "algorithm_result",
  "ok": true,
  "algorithm_id": "yolov5-similarity",
  "car_id": "car_001",
  "stream_id": "camera_front",
  "result": {
    "task": "similarity",
    "matched": true,
    "similarity": 0.72,
    "threshold": 0.45
  }
}
```

如果 Edge 在没有启动相似度会话时请求 `yolov5-similarity`，Cloud 会跳过该算法，避免拿旧模板误判。

## 5. 算法管理

查看算法：

```text
GET /api/algorithms
```

修改 `algorithms.json` 后重载：

```text
POST /api/algorithms/reload
```

当前默认算法：

```text
yolov5-similarity
yolov5-manhole-detect
yolov8-road-damage
```

## 6. 调试页面

```text
GET /dashboard
GET /unicorn
GET /api/dashboard/state
```

页面展示视频流、算法表、活跃任务、相似度会话、最近处理结果和 `.jetcar_debug` 目录摘要。

## 7. 健康检查

```text
GET /health
```
