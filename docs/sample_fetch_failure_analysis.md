# Sample Fetch 失败分析

## 失败分类

| 类型 | 数量 | 可避免 | 说明 |
|------|------|--------|------|
| **501 Job manager crashed** | 6 | ✅ 重试 | 服务端瞬时故障，重试可能成功 |
| **500 ReadTimeout/Connection** | 5 | ✅ 重试+超时 | 网络超时或断开，增加 timeout 或重试 |
| **422 Parameter 'split' required** | 1 | ✅ URL 编码 | config 含 `#` 等特殊字符，需 URL 编码 |
| **501 response exceeds size** | 1 | ✅ 降级 | 5 行过大，改用 length=1 重试 |
| **401 Unauthorized** | 2 | ❌ | 私有/需认证，无法获取 |
| **500 数据集格式/空** | 7 | ❌ | EmptyDataset、格式错误、不支持等 |

## 可改进项

1. **URL 编码**：config/split 含 `#`、`+` 等需 `urllib.parse.quote`
2. **重试**：对 501、500 瞬时错误增加重试
3. **超时**：rows 请求超时从 30s 提到 60s
4. **响应过大**：遇到 "exceeds size" 时用 length=1 重试
