import re
import json
from statistics import mean

# =========================
# 1. real data
# =========================
gt_text = r"""
paste gt_text here
"""

# =========================
# 2. predicted data
# =========================
pred_text = r"""
paste pred_text here
"""

# =========================================================
# utility functions
# =========================================================

def parse_attr_text(text_content: str):
    parts = text_content.split(",")
    data = {}
    for p in parts:
        key, value = p.split(":", 1)
        data[key.strip()] = value.strip()

    def parse_number(s: str):
        m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
        return float(m.group()) if m else None

    return {
        "drive_freq": parse_number(data["drive_freq"]),
        "split": data["split"],
        "parasitic": data["parasitic"],
        "x_stiffness": parse_number(data["x_stiffness"]),
        "nonlinearity": data["nonlinearity"],
    }


def parse_any_text_block(raw_text: str):
    """
    同时兼容:
    - text_context
    - text_content
    """
    result = {}

    pattern = re.compile(
        r'\{"image_file_name":"([^"]+)","text_(?:context|content)":"([^"]+)"\}'
    )

    for m in pattern.finditer(raw_text):
        image_file = m.group(1)
        text_content = m.group(2)

        idx_match = re.search(r"1_(\d+)\.png", image_file)
        if not idx_match:
            continue

        idx = int(idx_match.group(1))

        result[idx] = {
            "image_file_name": image_file,
            **parse_attr_text(text_content),
        }

    return result


def safe_relative_error(pred, gt):
    if gt == 0:
        return None
    return abs(pred - gt) / abs(gt)


def exact_match(a, b):
    return 1 if a == b else 0


parasitic_rank = {
    "less_than_5000Hz": 0,
    "berween_5000_and_10000Hz": 1,
    "between_5000_and_10000Hz": 1,
    "berween_10000_and_15000Hz": 2,
    "between_10000_and_15000Hz": 2,
    "berween_15000_and_20000Hz": 3,
    "between_15000_and_20000Hz": 3,
}

nonlinearity_rank = {
    "very_low": 0,
    "low": 1,
    "relatively_low": 2,
    "moderate": 3,
    "relatively_high": 4,
    "high": 5,
    "very_high": 6,
}

def evaluate(gt_map, pred_map):
    rows = []


    for pred_idx, pred in pred_map.items():
        gt_idx = pred_idx + 1

        if gt_idx not in gt_map:
            continue

        gt = gt_map[gt_idx]

        row = {
            "pred_idx": pred_idx,
            "gt_idx": gt_idx,
            "pred_file": pred["image_file_name"],
            "gt_file": gt["image_file_name"],

            "gt_drive_freq": gt["drive_freq"],
            "pred_drive_freq": pred["drive_freq"],
            "drive_freq_ae": abs(pred["drive_freq"] - gt["drive_freq"]),
            "drive_freq_re": safe_relative_error(pred["drive_freq"], gt["drive_freq"]),

            "gt_x_stiffness": gt["x_stiffness"],
            "pred_x_stiffness": pred["x_stiffness"],
            "x_stiffness_ae": abs(pred["x_stiffness"] - gt["x_stiffness"]),
            "x_stiffness_re": safe_relative_error(pred["x_stiffness"], gt["x_stiffness"]),

            "gt_parasitic": gt["parasitic"],
            "pred_parasitic": pred["parasitic"],
            "parasitic_exact": exact_match(pred["parasitic"], gt["parasitic"]),

            "gt_nonlinearity": gt["nonlinearity"],
            "pred_nonlinearity": pred["nonlinearity"],
            "nonlinearity_exact": exact_match(pred["nonlinearity"], gt["nonlinearity"]),
        }

        if pred["parasitic"] in parasitic_rank and gt["parasitic"] in parasitic_rank:
            row["parasitic_distance"] = abs(
                parasitic_rank[pred["parasitic"]] - parasitic_rank[gt["parasitic"]]
            )
        else:
            row["parasitic_distance"] = None

        if pred["nonlinearity"] in nonlinearity_rank and gt["nonlinearity"] in nonlinearity_rank:
            row["nonlinearity_distance"] = abs(
                nonlinearity_rank[pred["nonlinearity"]] - nonlinearity_rank[gt["nonlinearity"]]
            )
        else:
            row["nonlinearity_distance"] = None

        rows.append(row)

    return rows


def summarize(rows):
    def avg(vals):
        vals = [v for v in vals if v is not None]
        return mean(vals) if vals else None

    return {
        "n": len(rows),
        "drive_freq_mae": avg([r["drive_freq_ae"] for r in rows]),
        "drive_freq_mre": avg([r["drive_freq_re"] for r in rows]),
        "x_stiffness_mae": avg([r["x_stiffness_ae"] for r in rows]),
        "x_stiffness_mre": avg([r["x_stiffness_re"] for r in rows]),
        "parasitic_accuracy": avg([r["parasitic_exact"] for r in rows]),
        "parasitic_mean_distance": avg([r["parasitic_distance"] for r in rows]),
        "nonlinearity_accuracy": avg([r["nonlinearity_exact"] for r in rows]),
        "nonlinearity_mean_distance": avg([r["nonlinearity_distance"] for r in rows]),
    }


def pretty_print_summary(name, s):
    print(f"\n===== {name} =====")
    print(f"n = {s['n']}")
    print(f"drive_freq MAE = {s['drive_freq_mae']:.4f} Hz" if s["drive_freq_mae"] is not None else "drive_freq MAE = None")
    print(f"drive_freq MRE = {s['drive_freq_mre'] * 100:.4f}%" if s["drive_freq_mre"] is not None else "drive_freq MRE = None")
    print(f"x_stiffness MAE = {s['x_stiffness_mae']:.4f} N/m" if s["x_stiffness_mae"] is not None else "x_stiffness MAE = None")
    print(f"x_stiffness MRE = {s['x_stiffness_mre'] * 100:.4f}%" if s["x_stiffness_mre"] is not None else "x_stiffness MRE = None")
    print(f"parasitic exact accuracy = {s['parasitic_accuracy'] * 100:.4f}%" if s["parasitic_accuracy"] is not None else "parasitic exact accuracy = None")
    print(f"parasitic mean interval distance = {s['parasitic_mean_distance']:.4f}" if s["parasitic_mean_distance"] is not None else "parasitic mean interval distance = None")
    print(f"nonlinearity exact accuracy = {s['nonlinearity_accuracy'] * 100:.4f}%" if s["nonlinearity_accuracy"] is not None else "nonlinearity exact accuracy = None")
    print(f"nonlinearity mean ordinal distance = {s['nonlinearity_mean_distance']:.4f}" if s["nonlinearity_mean_distance"] is not None else "nonlinearity mean ordinal distance = None")


if __name__ == "__main__":
    gt_map = parse_any_text_block(gt_text)
    pred_map = parse_any_text_block(pred_text)

    print(f"GT count   : {len(gt_map)}")
    print(f"PRED count : {len(pred_map)}")

    rows = evaluate(gt_map, pred_map)

    print(f"Matched    : {len(rows)}")

    total_summary = summarize(rows)
    pretty_print_summary("Overall", total_summary)

    print("\nFirst 5 matched rows:")
    for r in rows[:5]:
        print(json.dumps(r, ensure_ascii=False, indent=2))
