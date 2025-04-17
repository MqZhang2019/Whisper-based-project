# 实时语音转写翻译工具

这是一个基于 Whisper 的实时语音转写和翻译工具，可以将实时音频转换为文本并翻译成中文。

## 功能特点

- 实时音频捕获和处理
- 使用 OpenAI Whisper 进行语音识别
- 支持两种翻译模式：
  - 本地翻译模型（Helsinki-NLP/opus-mt-en-zh）
  - Kimi API 在线翻译（可选）
- 自动静音检测和分段处理
- 支持 GPU 加速（如果可用）
- 实时显示转写和翻译结果

## 环境要求

- Python 3.8+
- CUDA 支持（可选，用于 GPU 加速）
- 音频设备（支持系统音频捕获）

## 依赖安装

```bash
pip install numpy torch whisper soundcard transformers openai
```

## 配置说明

1. 语音识别模型大小选择：
   - tiny: 最快但准确度较低
   - base: 平衡速度和准确度（默认）
   - small/medium/large: 更高准确度但需要更多资源

2. 翻译模型选择：
   ```python
   TRANSLATION_MODEL = "local"  # 使用本地翻译模型
   # 或
   TRANSLATION_MODEL = "kimi"   # 使用 Kimi API（需要配置 API 密钥）
   ```

3. 音频参数配置：
   ```python
   RATE = 16000          # 采样率
   CHUNK = 4096          # 缓冲区大小
   SILENCE_THRESHOLD = 0.03  # 静音检测阈值
   ```

## 使用方法

1. 运行程序：
   ```bash
   python main.py
   ```

2. 程序会自动：
   - 加载必要的模型
   - 检测并选择默认音频设备
   - 开始捕获系统音频
   - 实时显示转写和翻译结果

3. 按 Ctrl+C 停止程序

## 输出示例
#==================================================

English: This is a test message.
Chinese: 这是一条测试消息。

#==================================================

## 高级功能

- 自动处理长文本分段
- 智能文本拼接，避免重复
- GPU 内存自动管理
- 异步处理保证实时性能

## 注意事项

1. 首次运行会下载必要的模型文件
2. 使用 Kimi API 需要配置有效的 API 密钥
3. GPU 模式需要足够的显存
4. 建议使用耳机避免音频回环

## 许可证

Apache-2.0 license

## 贡献

欢迎提交 Issues 和 Pull Requests 来改进这个项目。

## 致谢

- OpenAI Whisper
- Helsinki-NLP
- Moonshot AI
