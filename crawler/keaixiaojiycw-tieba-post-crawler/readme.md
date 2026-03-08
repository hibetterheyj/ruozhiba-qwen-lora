# keaixiaojiycw-tieba-post-crawler 使用指南

## 库概述

这是一个**贴吧一体化异步爬取与数据处理工具箱**，基于 `aiotieba` 构建。

| 功能     | 描述                              |
| -------- | --------------------------------- |
| 异步爬虫 | 高并发爬取帖子内容和图片          |
| 断点续爬 | 自动保存进度，中断后可恢复        |
| 图片下载 | 多线程异步下载帖子图片            |
| HTML转换 | 生成 Apple 风格的离线 HTML 查看器 |
| LLM数据  | 输出 JSONL 格式，便于大模型训练   |

**版本**: 0.7.1
**依赖**: aiotieba, aiohttp, aiofiles
**Python**: 3.8+

---

## 安装

```bash
pip install keaixiaojiycw-tieba-post-crawler
```

---

## 使用方法

### 1. 设置 BDUSS（必须）

首先需要获取贴吧的登录凭证 BDUSS：

```bash
# macOS/Linux
export TIEBA_BDUSS="你的BDUSS值"
```

**如何获取 BDUSS：**

1. 登录百度贴吧网页版
2. 按 F12 打开开发者工具 → Application → Cookies
3. 找到 `BDUSS` 的值并复制

---

### 2. 命令行使用 (CLI)

安装后会提供 `tieba-crawler` 命令：

```bash
# 爬取帖子 (tid 是帖子ID，在帖子URL中可以找到)
tieba-crawler crawl 1234567890

# 列出已爬取的所有帖子
tieba-crawler list

# 将爬取数据转换为 HTML 查看器
tieba-crawler convert 1234567890

# 指定页码范围转换
tieba-crawler convert 1234567890 --start 1 --end 50
```

---

### 3. Python API 使用

```python
import asyncio
import os
from keaixiaojiycw_tieba_post_crawler import TiebaCrawler, convert_to_html, list_crawled_threads

# 设置 BDUSS
bduss = os.environ.get("TIEBA_BDUSS") or "你的BDUSS"

async def main():
    # 创建爬虫实例
    crawler = TiebaCrawler(bduss=bduss, max_consecutive_failures=3)

    # 爬取帖子 (tid = 帖子ID)
    await crawler.crawl_thread(tid=1234567890)

    # 列出已爬取的帖子
    threads = list_crawled_threads()
    for t in threads:
        print(f"TID: {t['tid']}, 标题: {t['title']}, 已爬页数: {t['last_page']}")

    # 转换为 HTML 查看器
    convert_to_html(
        tid=1234567890,
        thread_title="帖子标题",
        start_page=1,
        end_page=50,
        max_page=50
    )

asyncio.run(main())
```

---

## 输出目录结构

爬取后会在当前目录生成 `threads/` 文件夹：

```
threads/
├── index.json              # 帖子ID-标题映射
├── 目录.txt                # 可读的帖子目录
└── 1234567890/             # 以帖子ID命名的文件夹
    ├── checkpoint.json     # 断点续爬检查点
    ├── 1234567890_log.txt  # 爬取日志
    ├── 1234567890_dump.jsonl  # LLM格式数据(含原始图片URL)
    ├── 1234567890_metadata.json  # HTML查看器元数据
    ├── 1234567890_apple_p1_to_p50.html  # HTML查看器
    ├── images/             # 下载的图片
    │   ├── 楼层_1_12345_图片1.jpg
    │   └── ...
    └── posts/              # HTML格式的分页JSON
        ├── page_0001.json
        ├── page_0002.json
        └── ...
```

---

## 数据格式

### JSONL 格式 (用于 LLM 训练)

每行一个 JSON 对象：

```json
{
  "pid": 123456,
  "tid": 7890123,
  "user_name": "用户名",
  "user_id": 12345678,
  "text": "帖子内容...",
  "time": 1700000000,
  "img_urls_original": ["https://..."],
  "is_comment": false,
  "parent_pid": null,
  "floor": 1
}
```

### 字段说明

| 字段                  | 类型 | 说明                    |
| --------------------- | ---- | ----------------------- |
| `pid`               | int  | 帖子/评论ID             |
| `tid`               | int  | 主题帖ID                |
| `user_name`         | str  | 用户名                  |
| `user_id`           | int  | 用户ID                  |
| `text`              | str  | 文本内容                |
| `time`              | int  | 发布时间戳              |
| `img_urls_original` | list | 原始图片URL列表         |
| `is_comment`        | bool | 是否为楼中楼评论        |
| `parent_pid`        | int  | 父帖子ID (楼中楼时有效) |
| `floor`             | int  | 楼层号                  |

---

## 核心参数

| 参数                   | 默认值 | 说明                     |
| ---------------------- | ------ | ------------------------ |
| `IMAGE_WORKERS`      | 8      | 图片下载并发数           |
| `REQUEST_RETRIES`    | 5      | HTTP 请求重试次数        |
| `RETRY_BASE_DELAY`   | 1.0    | 重试基础延迟(秒)         |
| `REVISIT_LAST_PAGES` | 3      | 断点续爬时重爬的最后页数 |

---

## 注意事项

1. **必须设置 BDUSS** - 未登录无法爬取
2. **HTML 查看器需要本地服务器** - 直接打开 HTML 会因 CORS 无法加载数据，需使用 Live Server 或：
   ```bash
   python -m http.server 8000
   # 然后访问 http://localhost:8000
   ```
3. **断点续爬** - 中断后再次运行相同命令会自动从上次位置继续
4. **速率限制** - 库内置了重试和退避机制，但请勿过度请求

---

## API 参考

### TiebaCrawler

```python
class TiebaCrawler:
    def __init__(self, bduss: str, max_consecutive_failures: int = 3):
        """
        初始化爬虫。

        Args:
            bduss: 贴吧 BDUSS 登录凭证
            max_consecutive_failures: 连续失败上限，超过则停止爬取
        """

    async def crawl_thread(self, tid: int):
        """
        爬取指定帖子。

        Args:
            tid: 帖子ID
        """
```

### convert_to_html

```python
def convert_to_html(
    tid: int,
    thread_title: str,
    start_page: int,
    end_page: int,
    max_page: int
):
    """
    将已爬取数据转换为 HTML 查看器。

    Args:
        tid: 帖子ID
        thread_title: 帖子标题
        start_page: 起始页码
        end_page: 结束页码
        max_page: 总页数
    """
```

### list_crawled_threads

```python
def list_crawled_threads() -> List[Dict[str, Any]]:
    """
    列出所有已爬取帖子的信息。

    Returns:
        包含 tid, title, last_page, path 的字典列表
    """
```

---

## 源码位置

```
/envs/sft/lib/python3.12/site-packages/keaixiaojiycw-tieba-post-crawler/
├── __init__.py    # 模块导出
├── cli.py         # 命令行接口
├── converter.py   # HTML 转换器
├── crawler.py     # 核心爬虫类
└── utils.py       # 工具函数
```

---

## 实际使用示例

### 从 Cookies 文件提取 BDUSS

如果你有从浏览器导出的 cookies 文件，可以用以下命令提取 BDUSS：

```bash
# 从 cookies 文件中提取 BDUSS
cat tieba.baidu.com_cookies.txt | grep -o 'BDUSS=[^;]*' | cut -d'=' -f2
```

### 完整爬取流程

```bash
# 1. 进入工作目录
cd /path/to/your/project

# 2. 设置 BDUSS 环境变量
export TIEBA_BDUSS="你的BDUSS值"

# 3. 爬取帖子 (帖子ID可在帖子URL中找到)
# 例如: https://tieba.baidu.com/p/10130417881 中的 10130417881 就是帖子ID
tieba-crawler crawl 10130417881
tieba-crawler crawl 10354221105

# 4. 查看已爬取的帖子列表
tieba-crawler list
```

### 爬取结果示例

```
TID          | Title                              | Pages
------------------------------------------------------------
10130417881  | 弱智吧21岁生日快乐！盘点弱智吧最出圈的... | 9     (PIDs:340)
10354221105  | 弱智吧2025年度365佳贴               | 58    (PIDs:2847)
------------------------------------------------------------
```

### 输出文件说明

爬取完成后，会在当前目录生成 `threads/` 文件夹：

```
threads/
├── index.json                    # 帖子索引
├── 目录.txt                      # 可读目录
├── 10130417881/
│   ├── 10130417881_dump.jsonl    # LLM格式数据 (用于训练)
│   ├── 10130417881_dump.txt      # 文本日志
│   ├── checkpoint.json           # 断点续爬检查点
│   ├── images/                   # 下载的图片
│   └── posts/                    # 分页JSON数据
└── 10354221105/
    ├── 10354221105_dump.jsonl
    ├── 10354221105_dump.txt
    ├── checkpoint.json
    ├── images/
    └── posts/
```

### 常见问题

**Q: 如何获取帖子ID？**

A: 帖子URL格式为 `https://tieba.baidu.com/p/xxxxxxxxx`，其中 `xxxxxxxxx` 就是帖子ID。

**Q: 爬取中断了怎么办？**

A: 直接再次运行相同的爬取命令，爬虫会自动从断点继续。

**Q: BDUSS 过期怎么办？**

A: 重新登录百度贴吧，按上述方法获取新的 BDUSS 并更新环境变量。

---
