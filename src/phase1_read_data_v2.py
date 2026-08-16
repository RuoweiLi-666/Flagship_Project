from dataclasses import dataclass
from pathlib import Path
import re
import csv
from io import StringIO


@dataclass(frozen=True)
class TrialRecord:
    """一个数据文件对应一个完整 trial。"""

    subject: str
    label: str
    scenario: str
    repetition: int | None
    path: Path


def _sort_xlsx_files(file_path):
    """用于排序 xlsx 文件的辅助函数。按 subject 数字顺序、label、filename、repetition 排序。"""
    parts = file_path.parts
    subject = parts[-3] if len(parts) >= 3 else ""
    label = parts[-2] if len(parts) >= 2 else ""
    file_stem = file_path.stem
    
    # 从 subject 中提取数字（例如 "sub1" -> 1）
    subject_num = int(re.search(r"\d+", subject).group()) if re.search(r"\d+", subject) else 0
    
    # 从文件名中提取 repetition（例如 "AXR_AS_trial1" -> 1）
    repetition = int(re.search(r"trial(\d+)", file_stem).group(1)) if re.search(r"trial(\d+)", file_stem) else 0
    
    return (subject_num, label, file_stem, repetition)


def analyze_record_from_filename(file_path: str | Path) -> TrialRecord:
    """【迷你版本】从给定的绝对路径直接解析 trial 信息，只做路径解析，不做文件搜索。

    例子：
        file_path = "D:/project/SFU-IMU Dataset/IMU Dataset/sub1/ADLs/AXR_AS_trial1.xlsx"
        analyze_record_from_filename(file_path)
        -> TrialRecord(subject='sub1', label='ADLs', scenario='AS', repetition=1, path=...)

    说明：
    - 这是一个解析函数，不会搜索文件，直接从传入的绝对路径中解析信息。
    - 路径必须符合 SFU-IMU 数据集的格式：.../{subject}/{label}/{scenario}_{trial}.xlsx
    """
    file_path = Path(file_path).resolve()

    # 1) 验证文件是否存在
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    # 2) 解析路径的最后 3 个部分：sub1 / ADLs / AXR_AS_trial1.xlsx
    parts = file_path.parts
    if len(parts) < 7:
        raise ValueError(f"路径不符合 SFU-IMU 格式：{file_path}")

    subject = parts[-3]  # 例如 "sub1"
    label = parts[-2]    # 例如 "ADLs"
    file_stem = file_path.stem  # 例如 "AXR_AS_trial1"

    # 3) 解析文件名中的 scenario 和 repetition
    #    例如：AXR_AS_trial1      -> scenario='AS', repetition=1
    #          JXL_DSL_trial2      -> scenario='DSL', repetition=2
    repetition_match = re.search(r"_trial(\d+)$", file_stem)
    if repetition_match:
        repetition = int(repetition_match.group(1))
        base_name = file_stem[: repetition_match.start()]  # 例如 "AXR_AS"
    else:
        repetition = None
        base_name = file_stem

    # 4) 从 base_name 提取 scenario
    #    例如 "AXR_AS" -> "AS"，"JXL_DSL" -> "DSL"
    if "_" in base_name:
        scenario = base_name.rsplit("_", 1)[-1]
    else:
        scenario = base_name

    return TrialRecord(
        subject=subject,
        label=label,
        scenario=scenario,
        repetition=repetition,
        path=file_path,
    )


def analyze_all_records() -> list[TrialRecord]:
    """遍历 SFU-IMU Dataset 数据集中的所有 xlsx 文件，返回所有 trial 记录的列表。

    说明：
    - 数据集路径：SFU-IMU Dataset/IMU Dataset/
    - 里面有 sub1 到 sub10，每个 subject 有 60 个文件
    - 总共 600 个 .xlsx 文件
    - 函数会遍历所有文件并调用 analyze_record_from_filename() 进行解析

    返回：
    - 一个包含所有 TrialRecord 的列表
    """
    dataset_root = (Path(__file__).resolve().parent / "SFU-IMU Dataset" / "IMU Dataset").resolve()

    # 验证数据集路径是否存在
    if not dataset_root.exists():
        raise FileNotFoundError(f"数据集路径不存在：{dataset_root}")

    records = []  # 用来存放所有的 TrialRecord

    # 遍历数据集中的所有 .xlsx 文件，使用自定义排序
    for xlsx_file in sorted(dataset_root.rglob("*.xlsx"), key=_sort_xlsx_files):
        try:
            # 对每个文件调用 analyze_record_from_filename() 进行解析
            record = analyze_record_from_filename(xlsx_file)
            records.append(record)
        except (ValueError, FileNotFoundError) as e:
            # 如果某个文件不符合格式，可以选择跳过或打印警告
            print(f"警告：无法解析文件 {xlsx_file}：{e}")
            continue

    # 写出 CSV 文件
    fieldnames = ["subject", "label", "scenario", "repetition", "path"]
    output_path = Path(__file__).resolve().parent / "all_records.csv"

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    "subject": record.subject,
                    "label": record.label,
                    "scenario": record.scenario,
                    "repetition": record.repetition if record.repetition is not None else "",
                    "path": str(record.path),
                }
            )

    # 也打印到控制台，方便调试
    print(f"CSV 文件已保存到：{output_path}")
    return records


if __name__ == "__main__":
    # 测试 1：测试单个文件解析
    print("=" * 60)
    print("测试 1：解析单个文件")
    print("=" * 60)
    dataset_root = Path(__file__).resolve().parent / "SFU-IMU Dataset" / "IMU Dataset"
    test_file = dataset_root / "sub1" / "ADLs" / "AXR_AS_trial1.xlsx"

    if test_file.exists():
        record = analyze_record_from_filename(test_file)
        print(f"Subject:    {record.subject}")
        print(f"Label:      {record.label}")
        print(f"Scenario:   {record.scenario}")
        print(f"Repetition: {record.repetition}")
        print(f"Path:       {record.path}")
        print()
    else:
        print(f"文件不存在：{test_file}")
        print()

    # 测试 2：加载所有记录
    print("=" * 60)
    print("测试 2：加载所有记录")
    print("=" * 60)
    all_records = analyze_all_records()
    print(f"\n前 5 条记录：")
    for i, record in enumerate(all_records[:5], 1):
        print(f"{i}. {record.subject} | {record.label} | {record.scenario} | trial{record.repetition}")
