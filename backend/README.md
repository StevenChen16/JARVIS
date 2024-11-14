# 项目名称

## 依赖项

本项目依赖于 [GLM-4-Voice](https://github.com/THUDM/GLM-4-Voice.git) 项目。为确保代码能够正确运行，请将 GLM-4-Voice 项目的代码克隆到本项目目录中，并确保以下目录结构：

```
├── 项目目录/
│   ├── GLM-4-Voice/      # 克隆的 GLM-4-Voice 代码放置在这里
│   ├── cosyvoice/        # 其他依赖项目或自定义代码
│   ├── third_party/      # 其他依赖项目
│   ├── main_script.py    # 主脚本文件
│   ├── README.md         # 本说明文件
```

## 设置步骤

1. 克隆当前项目：
   ```bash
   git clone https://github.com/StevenChen16/JARVIS.git
   cd ./backend/voice-rag
   ```
```
2. 克隆 GLM-4-Voice 代码到项目目录下：
   git clone https://github.com/THUDM/GLM-4-Voice.git
```

3. 按照 GLM-4-Voice 仓库的说明安装必要的依赖。
4. 运行 `model_server.py` 文件。
5. 运行 `web_demo_rag.py` 文件。

## 运行方法

```bash
python web_demo.py --host 0.0.0.0 --port 7860 --flow-path ./GLM-4-Voice --model-path ./weights --tokenizer-path ./GLM-4-Voice
```

