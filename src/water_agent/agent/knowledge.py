from __future__ import annotations

from water_agent.labels import LABEL_DESCRIPTIONS, WaterLabel
from water_agent.schemas import LabelGuidance

KNOWLEDGE_VERSION = "water-label-guidance-v1"

_GUIDANCE: dict[WaterLabel, tuple[list[str], list[str]]] = {
    WaterLabel.ILLEGAL_MINING: (
        ["可能与岸线施工、砂石临时堆放或普通作业设施混淆。"],
        ["核实是否存在采挖、装卸或运输活动。", "结合现场位置与管理范围进行人工确认。"],
    ),
    WaterLabel.ILLEGAL_BUILDING: (
        ["可能与乱占、岸边既有设施或施工临时构筑物混淆。"],
        ["核实构筑物性质、位置及管理范围。", "当前模型对该类训练证据较少，建议人工复核。"],
    ),
    WaterLabel.ILLEGAL_PILING: (
        ["可能与乱建、乱占或普通岸边物料堆放混淆。"],
        ["核实堆放物性质、持续时间和是否占用管理范围。", "关注岸线小目标与遮挡情况。"],
    ),
    WaterLabel.ILLEGAL_OCCUPATION: (
        ["可能与乱建、普通耕作或合法岸线利用场景混淆。"],
        ["核实水域、滩涂或岸线边界。", "结合现场资料确认是否存在违规占用。"],
    ),
    WaterLabel.FLOATING_DEBRIS: (
        ["可能与水面反光、漂浮植被、船只尾迹或阴影混淆。"],
        ["核实漂浮物是否持续存在及影响范围。", "必要时结合现场或连续监控画面确认。"],
    ),
    WaterLabel.NORMAL: (
        ["可能与远距离、小尺度或被遮挡的异常场景混淆。"],
        ["当前模型对正常类训练证据较少，建议人工复核。", "不能将该结果视为不存在任何风险的证明。"],
    ),
}

_LABEL_ALIASES: dict[WaterLabel, tuple[str, ...]] = {
    WaterLabel.ILLEGAL_MINING: ("乱采", "非法采砂", "采砂", "采挖"),
    WaterLabel.ILLEGAL_BUILDING: ("乱建", "违建", "违规建筑", "违规构筑物"),
    WaterLabel.ILLEGAL_PILING: ("乱堆", "违规堆放", "堆放", "堆料"),
    WaterLabel.ILLEGAL_OCCUPATION: ("乱占", "违规占用", "占用"),
    WaterLabel.FLOATING_DEBRIS: ("有漂浮物", "漂浮物", "水面垃圾", "漂浮垃圾"),
    WaterLabel.NORMAL: ("正常", "无异常", "没有异常"),
}


def lookup_label_guidance(label: WaterLabel) -> LabelGuidance:
    confusions, review_checks = _GUIDANCE[label]
    return LabelGuidance(
        label=label,
        definition=LABEL_DESCRIPTIONS[label],
        common_confusions=confusions,
        review_checks=review_checks,
        knowledge_version=KNOWLEDGE_VERSION,
    )


def resolve_label_from_text(prompt: str) -> WaterLabel | None:
    """Resolve the six official labels from common, non-authoritative user wording."""
    normalized = prompt.strip()
    for label, aliases in _LABEL_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            return label
    return None
