from __future__ import annotations

# Unified Chinese plate charset for civilian + common special plates.
# The final "-" is reserved as the CTC blank token.

CCPD_PROVINCES = [
    "京", "津", "沪", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
    "新",
]

CCPD_ALPHABETS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "O",
]

CCPD_ADS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "O",
]

SPECIAL_CHARS = [
    "港", "澳", "学", "警", "挂", "使", "领", "民", "航", "临", "字",
    "试", "超", "练",
]

CHARS = (
    CCPD_PROVINCES
    + ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    + [
        "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
        "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
        "W", "X", "Y", "Z",
    ]
    + SPECIAL_CHARS
    + ["I", "O", "-"]
)

CHARS_DICT = {ch: idx for idx, ch in enumerate(CHARS)}
BLANK_IDX = len(CHARS) - 1
MAX_PLATE_LEN = 8
