"""压缩引擎的提示词与常量文案。

集中管理硬编码的提示词模板，便于统一调整，避免散落在引擎代码里。
"""
from __future__ import annotations


# --- 摘要消息前缀模板 ---

#: 简单摘要（Level 2 工具裁剪后仍超限时的摘要）的前缀
SUMMARY_PREFIX = "[历史上下文摘要]:{}"

#: 上下文合并摘要（Level 1/3 折叠旧轮次）的前缀
MERGE_SUMMARY_PREFIX = "[历史上下文合并摘要]:{}"

#: 折叠后的压缩轮次消息前缀
COMPRESSED_TURN_PREFIX = "[历史上下文合并摘要]:{}"


# --- 错误消息 ---

ERROR_COLLAPSE_BATCH_SIZE = "collapse_batch_size 必须大于等于 2"
ERROR_RETRY_LEVEL = "不支持的压缩重试等级：{level}，只允许 1、2、3"
ERROR_NO_SUMMARIZE_FN = "MessageManage 没有配置摘要模型"
ERROR_NO_ASUMMARIZE_FN = "MessageManage 没有配置异步摘要模型"
