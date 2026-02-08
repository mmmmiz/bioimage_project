# src/bioimage_qc/judge.py
# Day12 判定ロジック（OK/NG + 理由付与）

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping
import math


class Verdict(str, Enum):
    OK = "OK"
    NG = "NG"


@dataclass(frozen=True)
class Range:
    """
    しきい値レンジ（min/max は None なら無制限）
    例：
      Range(min=100, max=200)  -> 100〜200 の範囲内ならOK
      Range(min=2000, max=None)-> 2000以上ならOK
    """
    min: float | None = None
    max: float | None = None

    def contains(self, x: float) -> bool:
        if self.min is not None and x < self.min:
            return False
        if self.max is not None and x > self.max:
            return False
        return True


@dataclass(frozen=True)
class JudgeConfig:
    """
    Day11で決めた「しきい値」をここに入れる想定。
    いまは“動線確認用”に広めのデフォルト値にしています。
    """
    brightness_mean: Range = field(default_factory=lambda: Range(0.0, 255.0))
    contrast_std: Range = field(default_factory=lambda: Range(0.0, 255.0))
    sharpness_lap_var: Range = field(default_factory=lambda: Range(0.0, None))


@dataclass(frozen=True)
class FailedCheck:
    metric: str
    value: float | None
    expected: Range
    message: str


@dataclass(frozen=True)
class JudgeResult:
    verdict: Verdict
    reasons: tuple[str, ...] = ()
    failed: tuple[FailedCheck, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "reasons": list(self.reasons),
            "failed": [
                {
                    "metric": f.metric,
                    "value": f.value,
                    "expected": {"min": f.expected.min, "max": f.expected.max},
                    "message": f.message,
                }
                for f in self.failed
            ],
        }


_RULE_ORDER: tuple[str, ...] = (
    "brightness_mean",
    "contrast_std",
    "sharpness_lap_var",
)


def judge_metrics(metrics: Mapping[str, float], config: JudgeConfig) -> JudgeResult:
    """
    metrics（指標）と config（しきい値）から OK/NG 判定と理由を返す。
    """
    failed: list[FailedCheck] = []

    for name in _RULE_ORDER:
        expected: Range = getattr(config, name)

        # 1) 指標がそもそも存在しない
        if name not in metrics:
            failed.append(
                FailedCheck(
                    metric=name,
                    value=None,
                    expected=expected,
                    message=f"{name} が metrics に存在しません",
                )
            )
            continue

        # 2) 数値に変換できる前提だが、安全のため float 化してチェック
        try:
            value = float(metrics[name])
        except (TypeError, ValueError):
            failed.append(
                FailedCheck(
                    metric=name,
                    value=None,
                    expected=expected,
                    message=f"{name} が数値として扱えません: {metrics[name]!r}",
                )
            )
            continue

        # 3) NaN / inf はNG扱い（比較できない or 異常値）
        if math.isnan(value) or math.isinf(value):
            failed.append(
                FailedCheck(
                    metric=name,
                    value=value,
                    expected=expected,
                    message=f"{name} が異常値です（NaN/inf）: {value}",
                )
            )
            continue

        # 4) しきい値判定
        if not expected.contains(value):
            failed.append(
                FailedCheck(
                    metric=name,
                    value=value,
                    expected=expected,
                    message=_build_out_of_range_message(name, value, expected),
                )
            )

    if failed:
        return JudgeResult(
            verdict=Verdict.NG,
            reasons=tuple(f.message for f in failed),
            failed=tuple(failed),
        )

    return JudgeResult(verdict=Verdict.OK)


def _build_out_of_range_message(name: str, value: float, expected: Range) -> str:
    lo = "-∞" if expected.min is None else f"{expected.min:g}"
    hi = "∞" if expected.max is None else f"{expected.max:g}"
    return f"{name}={value:g} は許容範囲[{lo}, {hi}]から外れています"
