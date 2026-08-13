#!/usr/bin/env python3
"""带中文注释的 Phase 1 模板：用 1D CNN 做主体独立的 IMU 三分类。

任务
----
把每个完整 trial 分为：

    ADL / Near_Fall / Fall

默认使用腰部 IMU 的 6 个通道：三轴加速度 + 三轴角速度。

这份版本专门解决“我的文件结构与示例不一样”的问题：

1. 不再假定数据必须是 ``sub1/ADLs/*.xlsx``；
2. 会递归寻找数据根目录下的全部 Excel 文件；
3. 会从路径的任意一级目录或文件名中识别 subject 和 class；
4. 先运行 ``--inspect-only``，可以看到每个文件被识别成了什么；
5. 真正可能需要修改的地方集中在下方“用户配置区”。

建议执行顺序（Windows CMD）
----------------------------
第一步：只检查路径、标签和表头，不加载全部数据、不训练：

    python phase1_cnn_loso_commented.py 
      --data-root "D:\00-FLagship Project\\SFU-IMU Dataset\\IMU Dataset" 
      --output-dir "results\\inspect" 
      --inspect-only

第二步：加载并重采样全部 trial，但不训练：

    python phase1_cnn_loso_commented.py 
      --data-root "D:\00-FLagship Project\\SFU-IMU Dataset\\IMU Dataset" 
      --output-dir "results\\audit" 
      --audit-only

第三步：只试跑一个测试受试者：

    python phase1_cnn_loso_commented.py 
      --data-root  "D:\00-FLagship Project\\SFU-IMU Dataset\\IMU Dataset" 
      --output-dir "results\\debug_sub1" 
      --test-subject sub1 
      --epochs 5

第四步：完整 LOSO：

    python phase1_cnn_loso_commented.py 
      --data-root "D:\00-FLagship Project\\SFU-IMU Dataset\\IMU Dataset"
      --output-dir "results\\phase1_1dCNN_waist6" 
      --sequence-length 1024 
      --epochs 100

更换随机种子验证  
    python phase1_cnn_loso_commented.py 
      --data-root "D:\00-FLagship Project\\SFU-IMU Dataset\\IMU Dataset"
      --output-dir "results\\phase1_1dCNN(seed=123)_waist6" 
      --sequence-length 1024 
      --epochs 100    
      --seed 123      

依赖
----
    python -m pip install tensorflow pandas openpyxl scikit-learn matplotlib

研究边界
--------
这是“完整 trial 级别”的分类基线，不是实时滑窗检测，也不是提前预测。
外层测试采用 Leave-One-Subject-Out；验证集也按完整受试者留出，避免把同一个人
的数据同时放入训练与验证/测试。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# 减少 TensorFlow 启动时的信息输出；报错仍然会显示。
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# 某些受限环境的用户配置目录不可写；把 Matplotlib 缓存放到系统临时目录。
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

# 服务器或无图形界面的环境也可以保存图片。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight

# --inspect-only 和 --audit-only 不需要 TensorFlow，因此允许延迟报错。
try:
    import tensorflow as tf
except ImportError:
    tf = None


# =============================================================================
# 用户配置区：如果文件结构或表头不同，通常只需要修改本区
# =============================================================================

# 【通常只改这里 1：文件类型】
# Omar 数据集是 .xlsx。如果你转换成 CSV，可改成 {".xlsx", ".csv"}。
ALLOWED_FILE_SUFFIXES = {".xlsx"}

# Excel 数据所在的工作表。0 表示第一个 sheet；如果数据在名为 Data 的 sheet，
# 改成 EXCEL_SHEET_NAME = "Data"。
EXCEL_SHEET_NAME: int | str = 0

# 【通常只改这里 2：受试者命名】
# 默认能识别 sub1、sub01、subject_1、Subject-01 等。
# 正则表达式中的第一对括号必须捕获“受试者编号”。
SUBJECT_PATTERN = re.compile(
    r"(?:sub(?:ject)?)[ _-]*0*(\d+)",
    flags=re.IGNORECASE,
)

# 如果你的目录叫 P01、P02，可把上面的正则改成：
# SUBJECT_PATTERN = re.compile(r"(?:sub(?:ject)?|p)[ _-]*0*(\d+)", re.IGNORECASE)
#
# 如果你的目录直接叫 01、02，不建议仅靠数字自动识别（trial1 也有数字）。
# 此时请直接修改下方 extract_subject()，代码中有示例。

# 内部统一把受试者命名成 sub1、sub2……，这样 LOSO 的排序不会混乱。
SUBJECT_CANONICAL_PREFIX = "sub"

# 【通常只改这里 3：类别目录/文件名】
# 左边是程序内部统一使用的类别；右边是你本地可能出现的写法。
# 识别不区分大小写、空格、连字符和下划线。
LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "ADL": (
        "ADL",
        "ADLs",
        "Activity of Daily Living",
        "Activities of Daily Living",
    ),
    "Near_Fall": (
        "Near_Fall",
        "Near_Falls",
        "Near Fall",
        "Near Falls",
        "NearFall",
    ),
    "Fall": (
        "Fall",
        "Falls",
    ),
}

# 类别顺序同时决定神经网络标签编号：ADL=0、Near_Fall=1、Fall=2。
CLASS_NAMES = tuple(LABEL_ALIASES)
LABEL_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}

# 【通常只改这里 4：传感器表头】
# 每个逻辑通道后面可以写多个候选表头。程序会忽略大小写、标点、空格和括号中的
# 单位，因此 "waist Acceleration X (m/s^2)" 与 "Waist Acceleration X" 可匹配。
CHANNEL_ALIASES: dict[str, tuple[str, ...]] = {
    "waist_acc_x": ("waist Acceleration X (m/s^2)", "waist acc x"),
    "waist_acc_y": ("waist Acceleration Y (m/s^2)", "waist acc y"),
    "waist_acc_z": ("waist Acceleration Z (m/s^2)", "waist acc z"),
    "waist_gyro_x": ("waist Angular Velocity X (rad/s)", "waist gyro x"),
    "waist_gyro_y": ("waist Angular Velocity Y (rad/s)", "waist gyro y"),
    "waist_gyro_z": ("waist Angular Velocity Z (rad/s)", "waist gyro z"),
}

# 模型实际使用哪些逻辑通道。以后若只做 Acc 消融，可只保留前三项。
MODEL_CHANNEL_KEYS = (
    "waist_acc_x",
    "waist_acc_y",
    "waist_acc_z",
    "waist_gyro_x",
    "waist_gyro_y",
    "waist_gyro_z",
)

# 时间列也允许多个名字；没有时间列时，程序会自动改用样本序号插值。
TIME_COLUMN_ALIASES = ("Time", "Timestamp", "time stamp")

# Omar 的 Time 单位是微秒，因此相邻时间差乘 1e-6 后得到秒。
# 若你的时间戳单位已经是秒，改成 1.0；若是毫秒，改成 1e-3。
TIME_UNIT_TO_SECONDS = 1e-6

# 用于审计的 Omar 理论数量。若你使用的不是完整 Omar 数据，可设为 None，
# 这样就不会因数量不同而发出警告。
EXPECTED_TRIALS_PER_SUBJECT: dict[str, int] | None = {
    "ADL": 24,
    "Near_Fall": 15,
    "Fall": 21,
}


# =============================================================================
# 数据结构与命令行参数
# =============================================================================


@dataclass(frozen=True)
class TrialRecord:
    """一个数据文件对应一个完整 trial。"""

    subject: str
    label: str
    scenario: str
    repetition: int | None
    path: Path


@dataclass(frozen=True)
class DiscoveryIssue:
    """记录为什么某个数据文件没被程序采用。"""

    path: Path
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="带注释的 LOSO 1D-CNN：ADL / Near-Fall / Fall 三分类"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="数据总目录。下面可以有任意层级，程序会递归寻找数据文件。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("phase1_cnn_results"),
        help="保存扫描报告、指标、预测和图片的目录。",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=1024,
        help="每个变长 trial 被插值到的固定采样点数。",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test-subject",
        type=str,
        default=None,
        help="只测试一个受试者，例如 sub1；不写则运行全部 LOSO 折。",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="只检查文件识别和少量表头，不加载全部 trial，也不训练。",
    )
    parser.add_argument(
        "--inspect-limit",
        type=int,
        default=5,
        help="--inspect-only 时最多检查多少个已识别文件的表头。",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="加载、清洗、重采样并审计全部文件，但不训练。",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="保存每一折的 Keras 模型；默认关闭以节省空间。",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="关闭根据训练折计算的类别权重。",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="TensorFlow 训练日志详细程度。",
    )
    args = parser.parse_args()

    if args.sequence_length < 32:
        parser.error("--sequence-length 至少应为 32。")
    if args.inspect_limit < 1:
        parser.error("--inspect-limit 必须为正整数。")
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("--epochs、--batch-size、--patience 必须为正数。")
    if args.inspect_only and args.audit_only:
        parser.error("--inspect-only 和 --audit-only 不能同时使用。")
    return args


# =============================================================================
# 路径扫描：文件结构不同时，重点看这里
# =============================================================================


def natural_key(text: str) -> list[int | str]:
    """自然排序：sub2 排在 sub10 前面。"""
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", text)
    ]


def normalize_name(value: object) -> str:
    """用于宽松比较路径名/表头：忽略单位、空格、标点和大小写。"""
    text = str(value).strip().casefold().replace("µ", "u").replace("μ", "u")
    # 表头的单位往往不同，例如 (m/s^2) 与 (g)；先去掉括号内容。
    text = re.sub(r"\([^)]*\)", "", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def extract_subject(relative_path: Path) -> str | None:
    """从任意一级目录或文件名中寻找 subject 编号。

    例如以下路径都会得到 sub1：

        sub1/ADLs/a.xlsx
        ADLs/Subject_01/a.xlsx
        ADLs/sub01_NW_trial1.xlsx

    如果你的文件夹只叫 01、02，可把函数主体改成类似：

        first_folder = relative_path.parts[0]
        if first_folder.isdigit():
            return f"sub{int(first_folder)}"

    但一定要根据你的真实层级指定“哪一级是受试者”，不要在整条路径里随便抓数字。
    """
    # 优先逐级检查目录名；最后才检查文件名。
    candidates = [*relative_path.parts[:-1], relative_path.stem]
    for candidate in candidates:
        match = SUBJECT_PATTERN.search(candidate)
        if match:
            subject_number = int(match.group(1))
            return f"{SUBJECT_CANONICAL_PREFIX}{subject_number}"
    return None


def build_label_lookup() -> list[tuple[str, str]]:
    """生成 (规范化别名, 统一标签)，长别名优先，避免 Near_Fall 被识别为 Fall。"""
    pairs = [
        (normalize_name(alias), label)
        for label, aliases in LABEL_ALIASES.items()
        for alias in aliases
    ]
    return sorted(pairs, key=lambda item: len(item[0]), reverse=True)


LABEL_LOOKUP = build_label_lookup()


def extract_label(relative_path: Path) -> str | None:
    """从路径的任意一级目录或文件名中识别 ADL/Near_Fall/Fall。"""
    components = [*relative_path.parts[:-1], relative_path.stem]

    # 第一轮只接受整个目录名/文件名完全等于别名，最稳妥。
    for component in components:
        normalized = normalize_name(component)
        for alias, label in LABEL_LOOKUP:
            if normalized == alias:
                return label

    # 第二轮允许类别嵌在文件名中，例如 sub1_NearFall_trip_trial1.xlsx。
    # 长别名先判断，因此 nearfall 不会先被短的 fall 截获。
    normalized_stem = normalize_name(relative_path.stem)
    for alias, label in LABEL_LOOKUP:
        if alias and alias in normalized_stem:
            return label
    return None


def parse_scenario_and_repetition(path: Path) -> tuple[str, int | None]:
    """从 Omar 文件名中解析动作代码和 trial 编号。

    AXR_AS_trial1.xlsx -> scenario=AS, repetition=1

    这两项只用于后期失败分析，不参与三分类训练。因此你的命名不同，即使得到
    unknown/None，也不会阻止模型运行；需要时再按真实文件名修改本函数。
    """
    tokens = [token for token in re.split(r"[_\-\s]+", path.stem) if token]
    for index, token in enumerate(tokens):
        match = re.fullmatch(r"trial[_-]?(\d+)", token, flags=re.IGNORECASE)
        if match:
            scenario = tokens[index - 1] if index > 0 else "unknown"
            return scenario, int(match.group(1))

    # 兼容 filename_trial_1 这种 trial 和数字分开的写法。
    for index, token in enumerate(tokens[:-1]):
        if token.casefold() == "trial" and tokens[index + 1].isdigit():
            scenario = tokens[index - 1] if index > 0 else "unknown"
            return scenario, int(tokens[index + 1])
    return "unknown", None


def discover_trials(data_root: Path) -> tuple[list[TrialRecord], list[DiscoveryIssue]]:
    """递归扫描所有候选文件，再分别推断 subject 与 label。"""
    if not data_root.is_dir():
        raise FileNotFoundError(f"数据根目录不存在：{data_root}")

    suffixes = {suffix.casefold() for suffix in ALLOWED_FILE_SUFFIXES}
    candidate_files = sorted(
        (
            path
            for path in data_root.rglob("*")
            if path.is_file()
            and path.suffix.casefold() in suffixes
            and not path.name.startswith("~$")  # 忽略 Excel 打开时产生的临时锁文件
            and not any(part.startswith(".") for part in path.relative_to(data_root).parts)
        ),
        key=lambda path: natural_key(str(path.relative_to(data_root))),
    )
    if not candidate_files:
        raise FileNotFoundError(
            f"在 {data_root} 下没有找到扩展名为 {sorted(suffixes)} 的文件。"
        )

    records: list[TrialRecord] = []
    issues: list[DiscoveryIssue] = []
    for file_path in candidate_files:
        relative_path = file_path.relative_to(data_root)
        subject = extract_subject(relative_path)
        label = extract_label(relative_path)

        reasons: list[str] = []
        if subject is None:
            reasons.append("无法从路径识别受试者")
        if label is None:
            reasons.append("无法从路径识别 ADL/Near_Fall/Fall")
        if reasons:
            issues.append(DiscoveryIssue(file_path, "；".join(reasons)))
            continue

        scenario, repetition = parse_scenario_and_repetition(file_path)
        records.append(
            TrialRecord(
                subject=subject,
                label=label,
                scenario=scenario,
                repetition=repetition,
                path=file_path,
            )
        )

    if not records:
        examples = "\n".join(
            f"  - {issue.path.relative_to(data_root)}: {issue.reason}"
            for issue in issues[:10]
        )
        raise FileNotFoundError(
            "找到了数据文件，但没有任何文件能同时识别 subject 和 label。\n"
            "请修改 SUBJECT_PATTERN 或 LABEL_ALIASES。\n"
            f"前几个问题文件：\n{examples}"
        )
    return records, issues


def discovery_frame(records: list[TrialRecord], data_root: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "relative_path": str(record.path.relative_to(data_root)),
                "subject": record.subject,
                "label": record.label,
                "scenario": record.scenario,
                "repetition": record.repetition,
            }
            for record in records
        ]
    )


def issue_frame(issues: list[DiscoveryIssue], data_root: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "relative_path": str(issue.path.relative_to(data_root)),
                "reason": issue.reason,
            }
            for issue in issues
        ],
        columns=["relative_path", "reason"],
    )


# =============================================================================
# Excel/CSV 表头解析与数据读取
# =============================================================================


def read_table(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """集中处理文件读取；换格式、sheet 或表头行时优先改这里。"""
    suffix = path.suffix.casefold()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=EXCEL_SHEET_NAME, nrows=nrows)
    if suffix == ".csv":
        return pd.read_csv(path, nrows=nrows)
    raise ValueError(f"暂不支持的文件类型：{suffix}")


def make_header_lookup(columns: Iterable[object]) -> dict[str, str]:
    """把宽松化后的表头映射回 Excel 中的真实表头。"""
    lookup: dict[str, str] = {}
    for column in columns:
        key = normalize_name(column)
        if key in lookup and lookup[key] != str(column):
            warnings.warn(
                f"表头 {column!r} 与 {lookup[key]!r} 规范化后相同；将采用前者。"
            )
            continue
        lookup[key] = str(column)
    return lookup


def resolve_one_column(
    lookup: dict[str, str],
    aliases: Iterable[str],
) -> str | None:
    for alias in aliases:
        actual = lookup.get(normalize_name(alias))
        if actual is not None:
            return actual
    return None


def resolve_signal_columns(frame: pd.DataFrame) -> list[str]:
    """按 MODEL_CHANNEL_KEYS 的顺序，找到 Excel 中对应的 6 个真实表头。"""
    lookup = make_header_lookup(frame.columns)
    resolved: list[str] = []
    missing: list[str] = []

    for logical_key in MODEL_CHANNEL_KEYS:
        if logical_key not in CHANNEL_ALIASES:
            raise KeyError(
                f"MODEL_CHANNEL_KEYS 中的 {logical_key!r} 没有写入 CHANNEL_ALIASES。"
            )
        actual = resolve_one_column(lookup, CHANNEL_ALIASES[logical_key])
        if actual is None:
            missing.append(logical_key)
        else:
            resolved.append(actual)

    if missing:
        available = "\n".join(f"    {index:>2}: {column}" for index, column in enumerate(frame.columns))
        raise KeyError(
            "缺少以下逻辑通道："
            + ", ".join(missing)
            + "\nExcel 中实际存在的表头是：\n"
            + available
            + "\n请在 CHANNEL_ALIASES 中给这些逻辑通道添加你的真实表头。"
        )
    return resolved


def resolve_time_column(frame: pd.DataFrame) -> str | None:
    lookup = make_header_lookup(frame.columns)
    return resolve_one_column(lookup, TIME_COLUMN_ALIASES)


def inspect_files(
    records: list[TrialRecord],
    issues: list[DiscoveryIssue],
    data_root: Path,
    inspect_limit: int,
) -> None:
    """打印路径识别结果，并抽查若干文件的表头。"""
    print("\n=== 路径识别概览 ===")
    print(f"成功识别：{len(records)} 个文件")
    print(f"跳过：    {len(issues)} 个文件")

    recognized = discovery_frame(records, data_root)
    counts = (
        recognized.groupby(["subject", "label"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CLASS_NAMES, fill_value=0)
    )
    print("\n当前识别出的 subject × class 数量：")
    print(counts.to_string())

    print("\n前几个成功识别的文件：")
    for record in records[:10]:
        relative = record.path.relative_to(data_root)
        print(
            f"  [OK] {relative} -> subject={record.subject}, "
            f"label={record.label}, scenario={record.scenario}, "
            f"trial={record.repetition}"
        )

    if issues:
        print("\n前几个被跳过的文件：")
        for issue in issues[:10]:
            print(f"  [SKIP] {issue.path.relative_to(data_root)} -> {issue.reason}")

    print("\n=== 表头抽查 ===")
    for record in records[:inspect_limit]:
        relative = record.path.relative_to(data_root)
        print(f"\n文件：{relative}")
        try:
            frame = read_table(record.path, nrows=5)
            print(f"  读取成功；前 5 行形状：{frame.shape}")
            for index, column in enumerate(frame.columns):
                print(f"  column[{index:>2}] = {column!r}")
            resolved = resolve_signal_columns(frame)
            time_column = resolve_time_column(frame)
            print(f"  模型通道匹配成功：{resolved}")
            print(f"  时间列：{time_column!r}（None 表示将使用样本序号）")
        except Exception as exc:
            print(f"  [ERROR] {type(exc).__name__}: {exc}")


def read_and_resample_trial(
    path: Path,
    sequence_length: int,
) -> tuple[np.ndarray, int, float | None]:
    """读取一个 trial，清洗后沿时间轴线性插值到固定长度。"""
    frame = read_table(path)
    if len(frame) < 2:
        raise ValueError("数据少于两行。")

    # 先解析真实表头，再按固定逻辑顺序抽出 6 个通道。
    channel_columns = resolve_signal_columns(frame)
    signals = frame[channel_columns].apply(pd.to_numeric, errors="coerce")
    signals = signals.replace([np.inf, -np.inf], np.nan)

    actual_time_column = resolve_time_column(frame)
    if actual_time_column is not None:
        timestamps = pd.to_numeric(
            frame[actual_time_column], errors="coerce"
        ).to_numpy(dtype=np.float64)

        # 时间无效的行不能参与插值；信号也删除相同行。
        valid_time = np.isfinite(timestamps)
        timestamps = timestamps[valid_time]
        signals = signals.loc[valid_time].reset_index(drop=True)
        if len(timestamps) < 2:
            raise ValueError("有效时间戳少于两个。")

        # 若时间戳不是严格有序，先稳定排序；重复时间戳只保留第一个。
        order = np.argsort(timestamps, kind="stable")
        timestamps = timestamps[order]
        signals = signals.iloc[order].reset_index(drop=True)
        unique_time = np.concatenate(([True], np.diff(timestamps) > 0))
        timestamps = timestamps[unique_time]
        signals = signals.loc[unique_time].reset_index(drop=True)
        if len(timestamps) < 2:
            raise ValueError("时间戳没有至少两个不同取值。")

        # Unix 微秒时间戳非常大。先减首项可避免浮点插值精度损失。
        x_old = timestamps - timestamps[0]
        duration_seconds: float | None = float(
            x_old[-1] * TIME_UNIT_TO_SECONDS
        )
    else:
        warnings.warn(f"{path.name}: 没找到时间列，改用样本序号进行插值。")
        x_old = np.arange(len(signals), dtype=np.float64)
        duration_seconds = None

    # 对偶发缺失值做线性插值；首尾缺失则用最近有效值补齐。
    signals = signals.interpolate(method="linear", limit_direction="both")
    if signals.isna().any().any():
        bad_columns = signals.columns[signals.isna().any()].tolist()
        raise ValueError(f"以下列仍存在无法修复的缺失值：{bad_columns}")

    values = signals.to_numpy(dtype=np.float32)
    original_samples = len(values)
    x_new = np.linspace(float(x_old[0]), float(x_old[-1]), sequence_length)
    resampled = np.column_stack(
        [
            np.interp(x_new, x_old, values[:, channel])
            for channel in range(values.shape[1])
        ]
    ).astype(np.float32)
    print(path.name)
    print("原始六通道:", values.shape)
    print("重采样后:", resampled.shape)
    print("前两行原始值:\n", values[:2])
    print("前两行重采样值:\n", resampled[:2])
    return resampled, original_samples, duration_seconds


def load_dataset(
    records: list[TrialRecord],
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    metadata_rows: list[dict[str, Any]] = []

    for number, record in enumerate(records, start=1):
        if number == 1 or number % 50 == 0 or number == len(records):
            print(f"读取 trial {number:>3}/{len(records)}：{record.path.name}")
        try:
            sequence, original_samples, duration_seconds = read_and_resample_trial(
                record.path, sequence_length
            )
        except Exception as exc:
            raise RuntimeError(
                f"读取失败：{record.path}\n原始错误：{type(exc).__name__}: {exc}"
            ) from exc

        sequences.append(sequence)
        labels.append(LABEL_TO_ID[record.label])
        metadata_rows.append(
            {
                "subject": record.subject,
                "label": record.label,
                "label_id": LABEL_TO_ID[record.label],
                "scenario": record.scenario,
                "repetition": record.repetition,
                "original_samples": original_samples,
                "duration_seconds": duration_seconds,
                "path": str(record.path.resolve()),
            }
        )

    return (
        np.stack(sequences),
        np.asarray(labels, dtype=np.int64),
        pd.DataFrame(metadata_rows),
    )


def audit_dataset(metadata: pd.DataFrame) -> dict[str, Any]:
    """检查每名受试者/类别数量以及 trial 长度和时长。"""
    subjects = sorted(metadata["subject"].unique(), key=natural_key)
    counts = (
        metadata.groupby(["subject", "label"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=subjects, columns=CLASS_NAMES, fill_value=0)
    )

    print("\n每名受试者的 trial 数量：")
    print(counts.to_string())
    print(f"\n受试者：{len(subjects)} | Trials：{len(metadata)}")
    print(
        "每个 trial 原始采样点数："
        f"min={metadata['original_samples'].min()}, "
        f"median={metadata['original_samples'].median():.1f}, "
        f"max={metadata['original_samples'].max()}"
    )

    durations = metadata["duration_seconds"].dropna()
    if not durations.empty:
        print(
            "Trial 时长（秒）："
            f"min={durations.min():.2f}, "
            f"median={durations.median():.2f}, "
            f"max={durations.max():.2f}"
        )

    if EXPECTED_TRIALS_PER_SUBJECT is not None:
        for subject in subjects:
            for label, expected in EXPECTED_TRIALS_PER_SUBJECT.items():
                actual = int(counts.loc[subject, label])
                if actual != expected:
                    warnings.warn(
                        f"{subject}/{label}: 找到 {actual} 个，Omar 完整数据应为 {expected} 个。"
                    )

    return {
        "subjects": subjects,
        "number_of_subjects": len(subjects),
        "number_of_trials": int(len(metadata)),
        "counts": {
            subject: {
                label: int(counts.loc[subject, label]) for label in CLASS_NAMES
            }
            for subject in subjects
        },
        "original_samples": {
            "min": int(metadata["original_samples"].min()),
            "median": float(metadata["original_samples"].median()),
            "max": int(metadata["original_samples"].max()),
        },
        "duration_seconds": (
            {
                "min": float(durations.min()),
                "median": float(durations.median()),
                "max": float(durations.max()),
            }
            if not durations.empty
            else None
        ),
    }


# =============================================================================
# LOSO 划分、标准化与 1D CNN
# =============================================================================


def choose_validation_subject(subjects: list[str], test_subject: str) -> str:
    """每折把测试受试者的下一个人固定作为验证集。"""
    if len(subjects) < 3:
        raise ValueError("至少需要 3 名受试者才能划分训练/验证/测试。")
    test_index = subjects.index(test_subject)
    return subjects[(test_index + 1) % len(subjects)]


def standardize_from_training(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """只用训练受试者计算每个通道的均值与标准差，防止数据泄漏。"""
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (
        ((x_train - mean) / std).astype(np.float32),
        ((x_validation - mean) / std).astype(np.float32),
        ((x_test - mean) / std).astype(np.float32),
        mean.reshape(-1),
        std.reshape(-1),
    )


def require_tensorflow() -> None:
    if tf is None:
        raise RuntimeError(
            "没有安装 TensorFlow。请运行：python -m pip install tensorflow"
        )


def set_all_seeds(seed: int) -> None:
    """尽量固定随机性，便于复现实验。"""
    random.seed(seed)
    np.random.seed(seed)
    require_tensorflow()
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        # 某些 TensorFlow/硬件组合不支持确定性算子，忽略即可。
        pass


def build_model(
    sequence_length: int,
    number_of_channels: int,
    number_of_classes: int,
    learning_rate: float,
) -> Any:
    """一个克制的小型 1D CNN baseline，输入形状为 [时间点, 通道]。"""
    require_tensorflow()

    inputs = tf.keras.Input(
        shape=(sequence_length, number_of_channels), name="waist_imu"
    )

    # 第一层用较大的 kernel=7 捕捉局部运动变化，再把时间长度减半。
    x = tf.keras.layers.Conv1D(32, 7, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # 第二层增加特征通道，继续提取更高层的局部时间模式。
    x = tf.keras.layers.Conv1D(64, 5, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # 第三层后用全局平均池化，避免直接 Flatten 带来过多参数。
    x = tf.keras.layers.Conv1D(128, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Dropout 用于缓解这个小数据集上的过拟合。
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="phase1_waist_1dcnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


# =============================================================================
# 指标、图片和输出文件
# =============================================================================


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    per_class_recall = recall_score(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
        average=None,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=np.arange(len(CLASS_NAMES)),
                average="macro",
                zero_division=0,
            )
        ),
        # 三类样本数量不平衡，因此 balanced accuracy 比普通 accuracy 更重要。
        "balanced_accuracy": float(np.mean(per_class_recall)),
    }
    for label, recall in zip(CLASS_NAMES, per_class_recall, strict=True):
        metrics[f"recall_{label}"] = float(recall)
    return metrics


def save_confusion_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    normalize: str | None = None,
) -> None:
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
        normalize=normalize,
    )
    display = ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES)
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    display.plot(
        ax=axis,
        cmap="Blues",
        colorbar=False,
        values_format=".2f" if normalize else "d",
    )
    axis.set_title("LOSO confusion matrix" + (" (row-normalized)" if normalize else ""))
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def json_ready(value: Any) -> Any:
    """把 NumPy/Path 类型转成 json.dump 能写入的普通 Python 类型。"""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        converted = float(value)
        return converted if np.isfinite(converted) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, ensure_ascii=False)


# =============================================================================
# 单折训练与完整主流程
# =============================================================================


def run_fold(
    x: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    subjects: list[str],
    test_subject: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame]:
    validation_subject = choose_validation_subject(subjects, test_subject)
    subject_column = metadata["subject"].to_numpy()

    # 三个 mask 都按完整 subject 划分，不会把同一个人的 trial 拆开。
    test_mask = subject_column == test_subject
    validation_mask = subject_column == validation_subject
    train_mask = ~(test_mask | validation_mask)

    train_subjects = sorted(
        metadata.loc[train_mask, "subject"].unique(), key=natural_key
    )
    print("\n" + "=" * 72)
    print(
        f"测试：{test_subject} | 验证：{validation_subject} | "
        f"训练：{', '.join(train_subjects)}"
    )

    x_train, y_train = x[train_mask], y[train_mask]
    x_validation, y_validation = x[validation_mask], y[validation_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    missing_classes = set(range(len(CLASS_NAMES))) - set(np.unique(y_train))
    if missing_classes:
        names = [CLASS_NAMES[index] for index in sorted(missing_classes)]
        raise ValueError(f"训练折缺少类别：{names}")

    x_train, x_validation, x_test, mean, std = standardize_from_training(
        x_train, x_validation, x_test
    )

    fold_dir = args.output_dir / f"test_{test_subject}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        fold_dir / "training_scaler.npz",
        channels=np.asarray(MODEL_CHANNEL_KEYS),
        mean=mean,
        std=std,
    )

    # 每一折使用略微不同但可重复的随机种子。
    fold_number = subjects.index(test_subject)
    tf.keras.backend.clear_session()
    set_all_seeds(args.seed + fold_number)
    model = build_model(
        sequence_length=x.shape[1],
        number_of_channels=x.shape[2],
        number_of_classes=len(CLASS_NAMES),
        learning_rate=args.learning_rate,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(3, args.patience // 3),
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    class_weights: dict[int, float] | None = None
    if not args.no_class_weights:
        class_ids = np.arange(len(CLASS_NAMES))
        weights = compute_class_weight(
            class_weight="balanced", classes=class_ids, y=y_train
        )
        class_weights = {
            int(class_id): float(weight)
            for class_id, weight in zip(class_ids, weights, strict=True)
        }

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        shuffle=True,
        verbose=args.verbose,
    )
    pd.DataFrame(history.history).to_csv(fold_dir / "history.csv", index=False)

    probabilities = model.predict(x_test, batch_size=args.batch_size, verbose=0)
    predictions = probabilities.argmax(axis=1)
    fold_metrics: dict[str, Any] = {
        "test_subject": test_subject,
        "validation_subject": validation_subject,
        "train_subjects": train_subjects,
        "train_trials": int(train_mask.sum()),
        "validation_trials": int(validation_mask.sum()),
        "test_trials": int(test_mask.sum()),
        "epochs_ran": int(len(history.history["loss"])),
        **calculate_metrics(y_test, predictions),
    }
    print(
        f"{test_subject}: Macro-F1={fold_metrics['macro_f1']:.3f}, "
        f"Balanced Accuracy={fold_metrics['balanced_accuracy']:.3f}"
    )

    prediction_frame = metadata.loc[test_mask].copy().reset_index(drop=True)
    prediction_frame["fold_test_subject"] = test_subject
    prediction_frame["validation_subject"] = validation_subject
    prediction_frame["true_label"] = [CLASS_NAMES[index] for index in y_test]
    prediction_frame["predicted_label"] = [
        CLASS_NAMES[index] for index in predictions
    ]
    for class_id, class_name in enumerate(CLASS_NAMES):
        prediction_frame[f"probability_{class_name}"] = probabilities[:, class_id]

    prediction_frame.to_csv(fold_dir / "predictions.csv", index=False)
    write_json(fold_dir / "metrics.json", fold_metrics)
    save_confusion_figure(y_test, predictions, fold_dir / "confusion_matrix.png")
    if args.save_models:
        model.save(fold_dir / "model.keras")

    return fold_metrics, prediction_frame


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 第一步永远只扫描路径；即使没装 TensorFlow 也可以执行。
    records, issues = discover_trials(data_root)
    discovery_frame(records, data_root).to_csv(
        args.output_dir / "discovered_files.csv", index=False
    )
    issue_frame(issues, data_root).to_csv(
        args.output_dir / "skipped_files.csv", index=False
    )

    if args.inspect_only:
        inspect_files(records, issues, data_root, args.inspect_limit)
        print(f"\n检查报告已保存到：{args.output_dir}")
        return 0

    # 第二步才真正读取所有数值并重采样。
    x, y, metadata = load_dataset(records, args.sequence_length)
    metadata.to_csv(args.output_dir / "metadata.csv", index=False)
    audit = audit_dataset(metadata)
    write_json(args.output_dir / "data_audit.json", audit)

    if args.audit_only:
        print(f"\n数据审计完成。结果：{args.output_dir}")
        return 0

    # 只有真正训练时才要求安装 TensorFlow。
    require_tensorflow()
    subjects = sorted(metadata["subject"].unique(), key=natural_key)

    if args.test_subject is not None:
        requested_subject = args.test_subject.casefold()
        if requested_subject not in subjects:
            raise ValueError(
                f"未知 --test-subject {args.test_subject!r}；可选值为 {subjects}"
            )
        test_subjects = [requested_subject]
    else:
        test_subjects = subjects

    print(f"\nTensorFlow：{tf.__version__}")
    print(f"输入张量：{x.shape} | 标签：{y.shape}")
    print(f"外层测试折：{', '.join(test_subjects)}")

    fold_metrics: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for test_subject in test_subjects:
        metrics, predictions = run_fold(
            x=x,
            y=y,
            metadata=metadata,
            subjects=subjects,
            test_subject=test_subject,
            args=args,
        )
        fold_metrics.append(metrics)
        prediction_frames.append(predictions)

    metrics_frame = pd.DataFrame(fold_metrics)
    metrics_frame.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_predictions.to_csv(args.output_dir / "loso_predictions.csv", index=False)

    y_true = all_predictions["true_label"].map(LABEL_TO_ID).to_numpy()
    y_pred = all_predictions["predicted_label"].map(LABEL_TO_ID).to_numpy()
    pooled_metrics = calculate_metrics(y_true, y_pred)
    numeric_columns = [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        *(f"recall_{label}" for label in CLASS_NAMES),
    ]
    summary = {
        "task": "whole-trial ADL / Near_Fall / Fall classification",
        "channels": MODEL_CHANNEL_KEYS,
        "sequence_length": args.sequence_length,
        "folds_completed": test_subjects,
        "pooled_held_out_predictions": pooled_metrics,
        "mean_across_subject_folds": metrics_frame[numeric_columns].mean().to_dict(),
        "standard_deviation_across_subject_folds": metrics_frame[numeric_columns]
        .std(ddof=1)
        .to_dict(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(CLASS_NAMES)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    save_confusion_figure(
        y_true, y_pred, args.output_dir / "loso_confusion_matrix.png"
    )
    save_confusion_figure(
        y_true,
        y_pred,
        args.output_dir / "loso_confusion_matrix_normalized.png",
        normalize="true",
    )

    print("\n最终 pooled held-out 指标：")
    for name, value in pooled_metrics.items():
        print(f"  {name:>22}: {value:.4f}")
    print(f"\n结果已保存到：{args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
