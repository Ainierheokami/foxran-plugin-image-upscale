# 图片放大（foxran-plugin-image-upscale）

Foxran Agent 工具插件：输入一张图，输出高清大图。适合 GPT Image 2 生成后排版合适、但单图分辨率低导致细节发糊、文字乱码的图片。

## 原理

1. **甜点尺寸切块**：按放大倍率算出输出尺寸，从 `tile_sizes`（默认 1024x1024 / 1536x1024 / 1024x1536）里挑块数最少的一种，相邻块保留约 15% 冗余。
2. **逐块重绘**：每块先插值放大到该块的请求尺寸，再交给图像编辑模型（推荐 `gpt-image-2`），要求保持构图不变、画清细节、修正乱码文字。可以传 `text_hint` 告诉模型文字的正确内容。
3. **对齐与校色**：用 ECC 把重绘块的轻微平移/缩放对齐回插值参考图；只替换低频分量来消除各块色差，不影响重绘出的锐利细节和文字。
4. **最佳拼缝拼接**：在冗余区里用动态规划找两块差异最小的路径作为拼缝，再轻微羽化。两块对同一段文字画得不一样时，拼缝会绕开，不会叠出重影。

某块重试后仍失败时用普通插值顶上，结果里会注明；透明通道单独插值后保留。

## 工具

`image_hd_upscale(path, scale=2, text_hint="")`

- `path`：本地路径、http(s) URL 或 `url-<uuid>` 引用
- `scale`：1~4；只修乱码不放大时传 1
- `text_hint`：图中文字的正确内容（可选）

块数超过 `max_tiles` 或输出像素超过 `max_output_pixels` 时会自动降低倍率。

## 配置

首次加载时生成 `config/image_hd_upscale.yaml`，各项说明见 [config_template.yaml](config_template.yaml)。最常改的是 `model_name`：填 models.yaml 里 gpt-image-2 的模型名；留空则按 `image_edit` 标签轮询。

## 依赖

宿主自带的 Pillow、numpy、opencv-python；宿主版本 >= 4.4.0。

## 测试

在宿主仓库根目录运行：

```bash
.venv/bin/python -m pytest --import-mode=importlib <插件目录>/tests
```
