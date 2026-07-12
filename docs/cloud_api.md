# JetCarCloud 接口简表

局域网地址：

```text
HTTP: http://<cloud-ip>:8000
WS:   ws://<cloud-ip>:8000
```

Cloud 跑在 WSL2 时，`<cloud-ip>` 一般填 Windows 主机的局域网 IP。

## 主流程

1. 手机连接结果 WebSocket。
2. 手机打开处理后图像 MJPEG 流。
3. 小车持续向 Cloud 推送 JPEG 帧。
4. Cloud 收到帧后自动调用 `algorithm_id` 对应的 Docker 模型。
5. Cloud 把 JSON 识别结果和处理后画面发给手机。

## 基础数据格式

所有图片都用 JPEG base64：

```json
{
  "encoding": "jpeg",
  "width": 1280,
  "height": 720,
  "data": "base64-encoded-jpeg"
}
```

## 小车端接口

### HTTP 推送单帧

```text
POST /api/video/streams/{car_id}/{stream_id}/frames?algorithm_id={algorithm_id}
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

说明：

- `car_id`：小车编号，例如 `car_001`
- `stream_id`：摄像头编号，例如 `camera_front`
- `algorithm_id`：算法编号，例如 `yolov5-similarity`
- Cloud 收到后会缓存最新帧，并异步触发算法

### WebSocket 连续推帧

```text
WS /ws/video/{car_id}/{stream_id}/edge?algorithm_id={algorithm_id}
```

小车每帧发送一次同样的 JSON：

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

这是小车视频流的推荐入口。

## 手机端接口

### 接收识别结果

```text
WS /ws/inference/{car_id}/app
```

主要事件：

```json
{
  "type": "algorithm_result",
  "ok": true,
  "algorithm_id": "yolov5-similarity",
  "car_id": "car_001",
  "stream_id": "camera_front",
  "latency_ms": 1234.5,
  "result": {},
  "annotated_image": null,
  "error": ""
}
```

`result` 是模型容器输出的 `result.json` 内容，Cloud 不解析模型内部字段。

### 显示处理后画面

```text
GET /api/video/streams/{car_id}/{stream_id}/algorithms/{algorithm_id}/mjpeg?fps=5
```

返回：

```text
multipart/x-mixed-replace; boundary=frame
```

手机可以把这个 URL 当作 MJPEG 图像流显示。该接口只输出最新处理后画面，不主动触发算法；算法由小车推帧触发。

## 算法管理

查看算法：

```text
GET /api/algorithms
```

修改 `algorithms.json` 后重载：

```text
POST /api/algorithms/reload
```

算法配置示例：

```json
{
  "algorithms": {
    "yolov5-similarity": {
      "name": "YOLOv5 Similarity",
      "runner": "docker",
      "image": "model-yolov5-similarity:v1",
      "inputs": ["frame.jpg", "request.json"],
      "outputs": ["result.json", "annotated.jpg"],
      "timeout_seconds": 60,
      "enabled": true,
      "metadata": {}
    }
  }
}
```

## 模型容器约定

Cloud 调用容器：

```bash
docker run --rm \
  -v "<input-dir>:/app/data/input" \
  -v "<output-dir>:/app/data/output" \
  <image>
```

容器读取：

```text
/app/data/input/frame.jpg
/app/data/input/request.json
```

容器输出：

```text
/app/data/output/result.json
/app/data/output/annotated.jpg
```

`annotated.jpg` 用于手机端处理后画面流；没有它也可以返回 JSON 识别结果。

## 健康检查

```text
GET /health
```
