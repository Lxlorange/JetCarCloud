# JetCarCloud 接口简表

局域网地址：

```text
HTTP: http://<cloud-ip>:8000
WS:   ws://<cloud-ip>:8000
```

Cloud 跑在 WSL2 时，`<cloud-ip>` 一般填写 Windows 主机的局域网 IP。

## 主流程

1. 手机连接结果 WebSocket。
2. 手机打开处理后图像 MJPEG 流。
3. 小车持续向 Cloud 推送 JPEG 帧。
4. Cloud 收到帧后按 `algorithm_id` 调用本地 Python runner。
5. Cloud 把 JSON 识别结果和处理后画面发给手机。

## 小车端接口

HTTP 推送单帧：

```text
POST /api/video/streams/{car_id}/{stream_id}/frames?algorithm_id={algorithm_id}
POST /api/video/streams/{car_id}/{stream_id}/frames?algorithm_ids=yolov5-manhole-detect,yolov8-road-damage
```

WebSocket 连续推帧：

```text
WS /ws/video/{car_id}/{stream_id}/edge?algorithm_ids=yolov5-manhole-detect,yolov8-road-damage
```

请求体：

```json
{
  "car_id": "car_001",
  "image": {
    "encoding": "jpeg",
    "width": 1280,
    "height": 720,
    "data": "base64-encoded-jpeg"
  }
}
```

响应里 `frame_accepted=false` 表示输入帧率过高，本帧被限流丢弃；`algorithms_skipped` 表示模型忙或算法限速。

## 手机端接口

接收识别结果：

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
  "result": {},
  "annotated_image": null,
  "error": ""
}
```

显示处理后画面：

```text
GET /api/video/streams/{car_id}/{stream_id}/algorithms/{algorithm_id}/mjpeg?fps=5
```

返回：

```text
multipart/x-mixed-replace; boundary=frame
```

该接口只输出最新处理后画面，不主动触发算法；算法由小车推帧或后台采样触发。

## 算法管理

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

算法配置示例：

```json
{
  "algorithms": {
    "yolov8-road-damage": {
      "name": "YOLOv8 Road Damage Detection",
      "runner": "local",
      "image": "",
      "inputs": ["frame.jpg", "request.json"],
      "outputs": ["result.json", "annotated.jpg"],
      "timeout_seconds": 60,
      "enabled": true,
      "metadata": {
        "task": "road_damage_detection",
        "model_path": "models/yolov8-road-damage/YOLOv8_Small_2nd_Model.pt",
        "imgsz": 640,
        "conf": 0.2,
        "iou": 0.45,
        "device": "cpu"
      }
    }
  }
}
```

## 健康检查

```text
GET /health
```
