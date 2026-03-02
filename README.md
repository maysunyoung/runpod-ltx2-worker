# LTX-2 Video Generation - RunPod Serverless Worker

Generate high-quality AI videos using LTX-2 on RunPod Serverless with H100 GPU.

## Features

- **Fast**: ~60-80 seconds for a 4-second 720p video on H100
- **Pay-per-use**: Only charged when generating (scales to zero)
- **Simple API**: Just send a prompt, get a video
- **Cost-effective**: ~$0.05-0.10 per 4-second video

---

## 构建镜像（线上构建推荐）

在 **GitHub 上自动构建**，无需本地 Docker/GPU：

1. 将本仓库推送到你的 GitHub（fork 或 clone 后新建 repo）。
2. 打开 **Actions**，推送 `main` 分支或手动运行 **“Build and Push Docker Image”**。
3. 镜像会推送到 **GitHub Container Registry**：
   - 地址：`ghcr.io/<你的用户名>/ltx2-worker:latest`
   - 无需配置密钥即可使用 GHCR。

### 可选：同时推送到 Docker Hub

在 GitHub 仓库 **Settings → Secrets and variables → Actions** 添加：

| Secret 名 | 说明 |
|-----------|------|
| `DOCKERHUB_USERNAME` | Docker Hub 用户名 |
| `DOCKERHUB_TOKEN` | Docker Hub Access Token（[创建](https://hub.docker.com/settings/security)） |

保存后，同一 workflow 会多推一份到 `docker.io/<DOCKERHUB_USERNAME>/ltx2-worker:latest`。

### 本地构建（可选）

```bash
docker build -t your-dockerhub/ltx2-worker:latest .
docker push your-dockerhub/ltx2-worker:latest
```

---

## 密钥与环境变量

| 位置 | 变量 | 必填 | 说明 |
|------|------|------|------|
| **RunPod 控制台** | `HF_TOKEN` | 可选 | 若使用 HuggingFace 需授权模型，在 [HuggingFace Tokens](https://huggingface.co/settings/tokens) 创建后填入 RunPod 端点环境变量。 |
| **RunPod 控制台** | - | - | RunPod API Key 在 [RunPod API Keys](https://www.runpod.io/console/user/settings) 获取，用于调用接口。 |

LTX-2 公开模型一般不需要 `HF_TOKEN`；若遇到 gated 或 403，再配置即可。

---

## Deployment

### Option 1: Deploy via GitHub (Recommended)

1. Push this repo to GitHub
2. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
3. Click "New Endpoint" → "Deploy from GitHub"
4. Connect your repo
5. Configure:
   - **GPU**: H100 SXM (80GB) - fastest
   - **Min Workers**: 0 (scales to zero)
   - **Max Workers**: 3
   - **Idle Timeout**: 5 seconds
   - **Volume**: 100GB (for model cache)

### Option 2: Deploy via Docker image (GHCR / Docker Hub)

1. 在 GitHub Actions 或本地构建好镜像（见上方「构建镜像」）。
2. RunPod 控制台 → New Endpoint → 选择 **Custom Container**。
3. 镜像地址示例：
   - GHCR: `ghcr.io/<你的用户名>/ltx2-worker:latest`（若私有，需在 RunPod 配置 GHCR 拉取权限）
   - Docker Hub: `docker.io/<用户名>/ltx2-worker:latest`
4. GPU 选 H100 SXM，Volume 建议 100GB。

### Option 3: Deploy via RunPod CLI

```bash
brew install runpod/tap/runpodctl
runpodctl login
runpodctl deploy --name ltx2-worker --image ghcr.io/YOUR_USER/ltx2-worker:latest --gpu H100
```

---

## API Usage

### Endpoint URL

```
https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/runsync
```

### Request Format

```bash
curl -X POST \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A cat playing piano, studio lighting, 4K",
      "width": 1280,
      "height": 720,
      "num_frames": 97
    }
  }' \
  "https://api.runpod.ai/v2/$RUNPOD_LTX_ENDPOINT_ID/runsync"
```

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | *required* | Text description of the video |
| `negative_prompt` | string | "blurry, low quality..." | What to avoid |
| `width` | int | 1280 | Video width (720, 1280, 1920) |
| `height` | int | 720 | Video height (480, 720, 1080) |
| `num_frames` | int | 97 | Number of frames (~4 sec at 24fps) |
| `fps` | int | 24 | Frames per second |
| `guidance_scale` | float | 7.5 | Prompt adherence (5-15) |
| `num_inference_steps` | int | 30 | Quality steps (20-50) |
| `seed` | int | random | Reproducibility seed |

### Response Format

```json
{
  "id": "job-uuid",
  "status": "COMPLETED",
  "output": {
    "video": "data:video/mp4;base64,AAAA...",
    "duration_seconds": 4.04,
    "resolution": "1280x720",
    "fps": 24,
    "seed": 12345,
    "generation_time_seconds": 72.5
  }
}
```

### Async Usage

```bash
# Submit job
curl -X POST ... "https://api.runpod.ai/v2/$ENDPOINT_ID/run"
# Returns: {"id": "job-123", "status": "IN_QUEUE"}

# Check status
curl -H "Authorization: Bearer $API_KEY" \
  "https://api.runpod.ai/v2/$ENDPOINT_ID/status/job-123"
```

---

## Performance Guide

| Use Case | Resolution | Frames | Time | Cost |
|----------|------------|--------|------|------|
| Quick preview | 720p | 49 (~2s) | ~30s | ~$0.02 |
| Social media | 720p | 97 (~4s) | ~60s | ~$0.05 |
| High quality | 1080p | 97 (~4s) | ~90s | ~$0.08 |
| Long form | 720p | 241 (~10s) | ~4min | ~$0.18 |

| GPU | 4s 720p | 4s 1080p | Cost/hr |
|-----|---------|----------|---------|
| RTX 4090 | ~45s | ~90s | $0.44 |
| A40 | ~60s | ~120s | $0.79 |
| H100 SXM | ~30s | ~60s | $2.71 |

---

## Troubleshooting

- **Cold start slow?** 首次请求需 2–5 分钟拉模型，可设 Min Workers: 1 保活。
- **OOM?** 降低 `width`/`height` 或 `num_frames`。
- **Quality?** 提高 `num_inference_steps`、`guidance_scale`。

## License

LTX-2 is Apache 2.0 licensed. This worker code is provided as-is.
