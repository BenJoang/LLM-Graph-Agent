"""上下文压缩阈值判断（纯函数）。

把原来散落在 engine.py 里的 "max_tokens * 0.5 / 0.7 / 0.85" 阈值计算
与判断收拢成一组无状态函数，便于单独测试，也方便以后按模型调节比例。

职责边界：只做"估算值 vs 阈值"的比较，不做任何消息改写。
"""
from __future__ import annotations

from dataclasses import dataclass


# 各阶段占 max_tokens 的默认比例
SNIP_RATIO = 0.50        # 触发工具输出裁剪
SUMMARIZE_RATIO = 0.70   # 触发折叠摘要
RETRY_TARGET_RATIO = 0.85  # 降级压缩的目标上限


@dataclass(frozen=True)
class CompactThresholds:
    """一次压缩决策用到的全部阈值。"""

    max_tokens: int
    snip_tokens: int
    summarize_tokens: int
    retry_target_tokens: int


def compute_thresholds(
    max_tokens: int,
    *,
    snip_ratio: float = SNIP_RATIO,
    summarize_ratio: float = SUMMARIZE_RATIO,
    retry_target_ratio: float = RETRY_TARGET_RATIO,
) -> CompactThresholds:
    """由 max_tokens 推导出各阶段阈值。"""
    return CompactThresholds(
        max_tokens=max_tokens,
        snip_tokens=int(max_tokens * snip_ratio),
        summarize_tokens=int(max_tokens * summarize_ratio),
        retry_target_tokens=int(max_tokens * retry_target_ratio),
    )


def should_snip(estimate: int, thresholds: CompactThresholds) -> bool:
    """估算值超过裁剪阈值时，需要先裁剪历史工具输出。"""
    return estimate > thresholds.snip_tokens


def should_summarize(estimate: int, thresholds: CompactThresholds) -> bool:
    """估算值超过摘要阈值时，需要折叠旧轮次。"""
    return estimate > thresholds.summarize_tokens


def still_over_max(estimate: int, thresholds: CompactThresholds) -> bool:
    """估算值仍在 max_tokens 之上，需要继续折叠。"""
    return estimate > thresholds.max_tokens


def at_or_below_retry_target(estimate: int, thresholds: CompactThresholds) -> bool:
    """降级压缩中，达到目标上限即可停止，不必压缩到最优。"""
    return estimate <= thresholds.retry_target_tokens
