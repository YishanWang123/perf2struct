import json
from pathlib import Path
from statistics import mean, pstdev

INPUT_JSONL = "/root/memsdata/train/jsonl/labels_n1.jsonl"
OUTPUT_JSONL = "/root/newmemsdata/structured_output1.jsonl"
OUTPUT_STATS = "/root/newmemsdata/stats1.json"

PARASITIC_CAP = 25000.0

NONLINEARITY_ORDER = {
    "very_low": 0,
    "low": 1,
    "relatively_low": 2,
    "moderate": 3,
    "relatively_high": 4,
    "high": 5,
    "very_high": 6,
}

TYPE_TO_ID = {
    "lt": 0,
    "between": 1,
    "gt": 2,
}


def safe_float(x: str) -> float:
    s = x.strip().lower()
    s = s.replace("%", "")
    s = s.replace("hz", "")
    s = s.replace("n/m", "")
    return float(s)


def parse_range_field(value: str, unit: str, cap: float):
    """
    解析类似：
    less_than_5000Hz
    between_5000_and_10000Hz
    berween_5000_and_10000Hz   # typo 兼容
    over_20000Hz
    """
    s = value.strip().lower()
    s = s.replace("berween", "between")  # 兼容原数据中的 typo

    if s.startswith("less_than_"):
        num = s[len("less_than_"):]
        num = num.replace(unit, "")
        high = safe_float(num)
        low = 0.0
        rtype = "lt"

    elif s.startswith("between_"):
        body = s[len("between_"):]
        parts = body.split("_and_")
        if len(parts) != 2:
            raise ValueError(f"Cannot parse between-range: {value}")
        low = safe_float(parts[0].replace(unit, ""))
        high = safe_float(parts[1].replace(unit, ""))
        rtype = "between"

    elif s.startswith("over_"):
        num = s[len("over_"):]
        num = num.replace(unit, "")
        low = safe_float(num)
        high = cap
        rtype = "gt"

    else:
        raise ValueError(f"Unknown range format: {value}")

    center = (low + high) / 2.0
    width = high - low

    return {
        "type": rtype,
        "type_id": TYPE_TO_ID[rtype],
        "low": low,
        "high": high,
        "center": center,
        "width": width,
    }


def parse_text_context(text: str):
    """
    解析:
    drive_freq:39800Hz,split:less_than_0.2%,parasitic:berween_5000_and_10000Hz,x_stiffness:7100N/m,nonlinearity:moderate

    注意：
    这里仍然允许原始 text_context 里存在 split，
    但不会再把 split 提取到 features 中。
    """
    items = {}
    for chunk in text.split(","):
        if ":" not in chunk:
            continue
        key, value = chunk.split(":", 1)
        items[key.strip()] = value.strip()

    required_keys = ["drive_freq", "parasitic", "x_stiffness", "nonlinearity"]
    for k in required_keys:
        if k not in items:
            raise ValueError(f"Missing key `{k}` in text: {text}")

    drive_freq = safe_float(items["drive_freq"].replace("Hz", ""))
    x_stiffness = safe_float(items["x_stiffness"].replace("N/m", ""))

    parasitic_info = parse_range_field(items["parasitic"], unit="Hz", cap=PARASITIC_CAP)

    nl = items["nonlinearity"].strip().lower()
    if nl not in NONLINEARITY_ORDER:
        raise ValueError(f"Unknown nonlinearity label: {nl}")

    nl_id = NONLINEARITY_ORDER[nl]
    nl_ord = nl_id / (len(NONLINEARITY_ORDER) - 1)

    return {
        "drive_freq": drive_freq,
        "x_stiffness": x_stiffness,

        "parasitic_type": parasitic_info["type"],
        "parasitic_type_id": parasitic_info["type_id"],
        "parasitic_low": parasitic_info["low"],
        "parasitic_high": parasitic_info["high"],
        "parasitic_center": parasitic_info["center"],
        "parasitic_width": parasitic_info["width"],

        "nonlinearity": nl,
        "nonlinearity_id": nl_id,
        "nonlinearity_ord": nl_ord,
    }


def compute_stats(records, continuous_keys):
    stats = {}
    for key in continuous_keys:
        vals = [r["features"][key] for r in records]
        mu = mean(vals)
        sigma = pstdev(vals)
        if sigma == 0:
            sigma = 1.0
        vmin = min(vals)
        vmax = max(vals)
        stats[key] = {
            "mean": mu,
            "std": sigma,
            "min": vmin,
            "max": vmax,
        }
    return stats


def add_normalized_features(records, stats, continuous_keys):
    for r in records:
        for key in continuous_keys:
            x = r["features"][key]
            mu = stats[key]["mean"]
            sigma = stats[key]["std"]
            vmin = stats[key]["min"]
            vmax = stats[key]["max"]

            z = (x - mu) / sigma
            if vmax > vmin:
                mm = (x - vmin) / (vmax - vmin)
            else:
                mm = 0.0

            r["features"][f"{key}_z"] = z
            r["features"][f"{key}_minmax"] = mm


def main():
    input_path = Path(INPUT_JSONL)
    output_path = Path(OUTPUT_JSONL)
    stats_path = Path(OUTPUT_STATS)

    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            text = obj["text_context"]
            parsed = parse_text_context(text)

            record = {
                "image_file_name": obj["image_file_name"],
                "raw_text": text,
                "features": parsed,
            }
            records.append(record)

    continuous_keys = [
        "drive_freq",
        "x_stiffness",
        "parasitic_low",
        "parasitic_high",
        "parasitic_center",
        "parasitic_width",
    ]

    stats = compute_stats(records, continuous_keys)
    add_normalized_features(records, stats, continuous_keys)

    with output_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Done. Wrote {len(records)} records to {output_path}")
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()