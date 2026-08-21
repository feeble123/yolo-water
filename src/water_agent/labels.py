from enum import StrEnum


class WaterLabel(StrEnum):
    ILLEGAL_MINING = "乱采"
    ILLEGAL_BUILDING = "乱建"
    ILLEGAL_PILING = "乱堆"
    ILLEGAL_OCCUPATION = "乱占"
    FLOATING_DEBRIS = "有漂浮物"
    NORMAL = "正常"


LABEL_DESCRIPTIONS: dict[WaterLabel, str] = {
    WaterLabel.ILLEGAL_MINING: "水域或岸线附近存在疑似非法采砂、采石或装卸作业。",
    WaterLabel.ILLEGAL_BUILDING: "河湖管理范围内存在疑似违规建筑或构筑物。",
    WaterLabel.ILLEGAL_PILING: "岸线或水域附近存在疑似违规堆放砂石、建材或废弃物。",
    WaterLabel.ILLEGAL_OCCUPATION: "水域、滩涂或岸线存在疑似违规占用。",
    WaterLabel.FLOATING_DEBRIS: "水面存在漂浮垃圾、枯枝或其他漂浮物。",
    WaterLabel.NORMAL: "未识别到上述明显异常。",
}

