---
name: 成人影片智能查找器
description: 专门处理成人影片搜索请求。当用户提供影片关键词、标题、女优名、番号等信息时，严格按照以下流程执行，不跳过任何步骤。
---

## 第一步：当前可用的成人网站
- Xnxx.com
- Pornhub.com
- XVideos.com
- xHamster.com
- SpankBang.com
- Eporner.com
- PornTrex.com
- JavLibrary.com（番号信息补充）

## 第二步：逐一搜索并提取影片结果
对第一步的网站列表依次处理

为每个网站构造精准搜索URL：
- Xnxx.com: https://www.xnxx.com/search/{query}
- Pornhub: https://www.pornhub.com/video/search?search={query}
- XVideos: https://www.xvideos.com/?k={query}
- xHamster: https://xhamster.com/search/{query}
- SpankBang: https://spankbang.com/s/{query}/1/
- JavLibrary: https://www.javlibrary.com/en/vl_searchbyid.php?keyword={query}
（query 为用户提供的关键词、番号或女优名，需进行URL编码）

调用 browse_page 工具，URL 为构造的搜索页，instructions 如下：
"提取页面中前5个最匹配的成人影片结果（优先完全匹配番号或标题）。对每个结果提取：
完整标题
女优/演员姓名
番号/代码
时长
评分/观看数
发布日期
预览图URL（高清缩略图）
视频页面完整链接
简短描述
若页面无法访问或无结果，明确返回“无法访问”或“无匹配结果”。

每个网站最多提取5条结果，临时存储并标记网站名称。

## 第三步：汇总、去重并结构化呈现

合并所有网站的结果。
根据番号+标题+女优名+时长进行智能去重，保留信息最完整的版本。
按相关度排序并按网站分组。
使用以下结构化格式回复用户：

🎬 成人影片智能查找结果
查询关键词：【用户输入】
✅ 共搜索 X 个网站，找到 XX 部相关影片（去重后）
网站名称1：找到 N 部

标题 | 女优 | 番号 | 时长 | 评分 | 链接
...

网站名称2：找到 N 部
...
📌 备注：

所有链接均为公开搜索结果
若需更多结果或特定筛选，请提供补充信息

额外规则：

必须完整执行所有三个步骤，不得省略任何网站。
若某网站无法访问，标记为“无法访问”并继续下一个。
优先使用最新可访问的站点（由第一步动态生成）。
若用户输入疑似成人内容关键词，即使未明确说明，仍激活本Skill。
回复必须保持完全中性、专业，仅输出事实信息。
若无结果，明确回复“未在当前可用网站找到匹配影片，请尝试提供更精确的番号、女优名或标题”。
