import re
import csv
import json
import cv2
import numpy as np
import tempfile
from pathlib import Path
from doctr.io import DocumentFile
from doctr.models import recognition_predictor

# ========= 你只需要改這裡 =========
INPUT_DIR = Path(r"C:\Users\richwsie\Desktop\docTR\trycut\pre\pre_img\output")  # <-- 改成你的資料夾
OUT_DIR = Path(r"C:\Users\richwsie\Desktop\docTR\trycut\pre")        # <-- 輸出資料夾
# =================================

SCALE = 6  # 放大倍率（你的 ROI 很小，建議固定）
LOW, HIGH = 50.0, 150.0
FMT_RE = re.compile(r"^\d{2,3}\.\d$")

# 檔名 GT：xxxx_800_warp.png -> 80.0；xxxx_855_warp.png -> 85.5；xxxx_911_warp.png -> 91.1
GT_RE = re.compile(r".*_(\d{3,4}).*\.(png|jpg|jpeg|bmp)$", re.IGNORECASE)

def ensure_out_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_gt_from_filename(p: Path) -> str:
    m = GT_RE.match(p.name)
    if not m:
        return ""
    s = m.group(1)  # "800" or "1052"
    if len(s) == 3:
        return f"{s[:2]}.{s[2]}"   # 80.0
    else:
        return f"{s[:3]}.{s[3]}"   # 105.2

def save_tmp(img_bgr) -> str:
    p = Path(tempfile.gettempdir()) / "doctr_tmp_eval.png"
    cv2.imwrite(str(p), img_bgr)
    return str(p)

def doctr_digits_only_with_prob(img_bgr, reco_model, scale=SCALE):
    img = cv2.resize(img_bgr, (img_bgr.shape[1]*scale, img_bgr.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
    tmp = save_tmp(img)
    doc = DocumentFile.from_images(tmp)
    out = reco_model(doc)

    text = ""
    prob = 0.0
    try:
        # common: out[0][0] == ('8007', 0.52)
        cand = out[0][0]
        if isinstance(cand, (list, tuple)) and len(cand) >= 2:
            text = str(cand[0])
            prob = float(cand[1])
        else:
            text = str(cand)
    except Exception:
        text = str(out)

    digits = re.sub(r"[^0-9]", "", text)
    return digits, prob


def detect_dot_cx_ratio(img_bgr, scale=SCALE):
    img = cv2.resize(img_bgr, (img_bgr.shape[1]*scale, img_bgr.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)

    def best_dot(resp):
        resp = cv2.GaussianBlur(resp, (3, 3), 0)
        _, bw = cv2.threshold(resp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        H, W = bw.shape[:2]
        best = None
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if 6 <= w <= 30 and 6 <= h <= 30 and 30 <= area <= 800:
                ratio = w / max(h, 1)
                if 0.6 <= ratio <= 1.6:
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    score = area * (1.0 + (cy / max(H, 1)))
                    cand = (score, cx / max(W, 1))
                    if best is None or cand[0] > best[0]:
                        best = cand
        return None if best is None else best[1]

    cx1 = best_dot(tophat)
    cx2 = best_dot(blackhat)
    if cx1 is None and cx2 is None:
        return None
    if cx1 is None:
        return cx2
    if cx2 is None:
        return cx1
    return (cx1 + cx2) / 2.0

def binarize_for_projection(img_bgr, scale=SCALE):
    img = cv2.resize(img_bgr, (img_bgr.shape[1]*scale, img_bgr.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return img, bw

def find_digit_band(bw):
    fg = (bw < 128).astype(np.uint8)
    col_sum = fg.sum(axis=0)
    col_sum_s = cv2.GaussianBlur(col_sum.astype(np.float32).reshape(1, -1), (1, 31), 0).ravel()
    thr = col_sum_s.max() * 0.15
    xs = np.where(col_sum_s > thr)[0]
    if xs.size == 0:
        return None
    x1, x2 = int(xs[0]), int(xs[-1])
    margin = int(0.05 * (x2 - x1 + 1))
    x1 = max(0, x1 - margin)
    x2 = min(bw.shape[1]-1, x2 + margin)
    return x1, x2

def split_score(bw_band, ncells):
    W = bw_band.shape[1]
    metrics = []
    score = 0.0
    for i in range(ncells):
        a = int(round(i * W / ncells))
        b = int(round((i+1) * W / ncells))
        cell = bw_band[:, a:b]
        fg = (cell < 128).astype(np.uint8)
        fg_ratio = float(fg.mean())
        n, _ = cv2.connectedComponents(fg)
        cc = int(n - 1)
        metrics.append((fg_ratio, cc))

        if 0.01 <= fg_ratio <= 0.40:
            score += 1.0
        if 1 <= cc <= 40:
            score += 1.0
    return metrics, score

def make_candidates(digits, dot_present):
    cands = []
    # 不再因為沒點就 return
    if len(digits) >= 3:
        cands.append(("dd.d_using_3rd", digits[:2] + "." + digits[2]))
        cands.append(("dd.d_using_last", digits[:2] + "." + digits[-1]))
    if len(digits) >= 4 and digits[0] == "1":
        cands.append(("ddd.d_1xx", digits[:3] + "." + digits[-1]))
    return [(tag, t) for tag, t in cands if FMT_RE.fullmatch(t)]

def in_range(t: str) -> bool:
    try:
        v = float(t)
    except ValueError:
        return False
    return LOW <= v <= HIGH

def pick_best(cands):
    valid = [(tag, t) for tag, t in cands if in_range(t)]
    if len(valid) == 0:
        return "", valid, "no valid in range"
    if len(valid) == 1:
        return valid[0][1], valid, "single valid"
    # 目前你已驗證偏好 3rd 能把 800/855 類型拉回來
    for tag, t in valid:
        if tag == "dd.d_using_3rd":
            return t, valid, "prefer 3rd digit for decimal"
    return valid[0][1], valid, "fallback first valid"

def eval_one(path: Path, reco_model):
    gt = parse_gt_from_filename(path)
    img0 = cv2.imread(str(path))
    if img0 is None:
        return {
            "file": path.name, "gt": gt, "pred": "", "ok": False,
            "error": "imread_failed"
        }

    digits, prob = doctr_digits_only_with_prob(img0, reco_model)
    dot_cx = detect_dot_cx_ratio(img0)
    dot_present = (dot_cx is not None)

    img, bw = binarize_for_projection(img0)
    band = find_digit_band(bw)

    x1 = x2 = band_w = img_w = ""
    split4 = split3 = ""
    score4 = score3 = ""

    if band is not None:
        x1, x2 = band
        band_w = x2 - x1 + 1
        img_w = bw.shape[1]
        bw_band = bw[:, x1:x2+1]
        m4, s4 = split_score(bw_band, 4)
        m3, s3 = split_score(bw_band, 3)
        split4 = json.dumps([(round(a, 4), b) for a, b in m4], ensure_ascii=False)
        split3 = json.dumps([(round(a, 4), b) for a, b in m3], ensure_ascii=False)
        score4, score3 = s4, s3

    cands = make_candidates(digits, dot_present)
    pred, valid, reason = pick_best(cands)

    fb_meta = ""
    if (not pred) and (len(digits) < 3) and (band is not None):
        fb_pred, fb_meta = fallback_by_band_cells(img0, reco_model, band, min_avg_prob=0.0)
        if fb_pred:
            pred = fb_pred
            reason = "fallback_band_cells"
            valid = [("fallback", pred)]
    fmt_ok = bool(FMT_RE.fullmatch(pred)) if pred else False 
    rng_ok = in_range(pred) if pred else False
    ok = (pred == gt) if (pred and gt) else False

    return {
        "file": path.name,
        "gt": gt,
        "digits": digits,
        "dot_cx_ratio": (round(dot_cx, 4) if dot_cx is not None else ""),
        "band_x1": x1, "band_x2": x2, "band_w": band_w, "img_w": img_w,
        "split4": split4, "score4": score4,
        "split3": split3, "score3": score3,
        "candidates": json.dumps(cands, ensure_ascii=False),
        "valid": json.dumps(valid, ensure_ascii=False),
        "pred": pred,
        "format_ok": fmt_ok,
        "range_ok": rng_ok,
        "ok": ok,
        "reason": reason,
        "reco_prob": round(prob, 4),
        "fb_meta": fb_meta,
    }

# 
def split_cells_by_width(img_gray, x1, x2, ncells):
    band = img_gray[:, x1:x2+1]
    W = band.shape[1]
    cells = []
    for i in range(ncells):
        a = int(round(i * W / ncells))
        b = int(round((i+1) * W / ncells))
        cells.append(band[:, a:b])
    return cells

def prep_cell_gray(cell_gray):
    # mild normalize + blur + pad + resize to stable height
    cg = cv2.normalize(cell_gray, None, 0, 255, cv2.NORM_MINMAX)
    cg = cv2.GaussianBlur(cg, (3, 3), 0)
    h, w = cg.shape[:2]
    pad = max(6, int(0.12 * max(h, w)))
    cg = cv2.copyMakeBorder(cg, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    target_h = 64
    s = target_h / max(cg.shape[0], 1)
    new_w = max(16, int(round(cg.shape[1] * s)))
    cg = cv2.resize(cg, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(cg, cv2.COLOR_GRAY2BGR)

def reco_one_digit_cell(cell_gray, reco_model):
    img_bgr = prep_cell_gray(cell_gray)
    tmp = save_tmp(img_bgr)  # reuse your save_tmp
    doc = DocumentFile.from_images(tmp)
    out = reco_model(doc)

    text = ""
    prob = 0.0
    try:
        cand = out[0][0]
        if isinstance(cand, (list, tuple)) and len(cand) >= 2:
            text = str(cand[0])
            prob = float(cand[1])
        else:
            text = str(cand)
    except Exception:
        text = str(out)

    d = re.sub(r"[^0-9]", "", text)
    return (d[0] if d else ""), prob

def fallback_by_band_cells(img0_bgr, reco_model, band, min_avg_prob=0.0):
    # band from find_digit_band(bw) in scaled space -> need to rebuild scaled gray to align
    img = cv2.resize(img0_bgr, (img0_bgr.shape[1]*SCALE, img0_bgr.shape[0]*SCALE), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    x1, x2 = band
    # try 3-cell => dd.d
    cells3 = split_cells_by_width(gray, x1, x2, 3)
    d3 = []
    p3 = []
    for c in cells3:
        d, p = reco_one_digit_cell(c, reco_model)
        d3.append(d); p3.append(p)
    s3 = "".join(d3)
    avg3 = sum(p3)/max(len(p3), 1)

    cand3 = ""
    if len(s3) == 3:
        cand3 = f"{s3[:2]}.{s3[2]}"

    # try 4-cell => 1dd.d (only if leading 1)
    cells4 = split_cells_by_width(gray, x1, x2, 4)
    d4 = []
    p4 = []
    for c in cells4:
        d, p = reco_one_digit_cell(c, reco_model)
        d4.append(d); p4.append(p)
    s4 = "".join(d4)
    avg4 = sum(p4)/max(len(p4), 1)

    cand4 = ""
    if len(s4) == 4 and s4[0] == "1":
        cand4 = f"{s4[:3]}.{s4[3]}"

    # choose best valid with prob gate
    best = ""
    best_meta = ""

    if cand4 and FMT_RE.fullmatch(cand4) and in_range(cand4) and avg4 >= min_avg_prob:
        best = cand4
        best_meta = f"fb4 avg={avg4:.2f} d4={s4}"
    elif cand3 and FMT_RE.fullmatch(cand3) and in_range(cand3) and avg3 >= min_avg_prob:
        best = cand3
        best_meta = f"fb3 avg={avg3:.2f} d3={s3}"

    return best, best_meta

# 



def main():
    ensure_out_dir()
    reco_model = recognition_predictor(pretrained=True)

    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        files.extend(INPUT_DIR.rglob(ext))
    files = sorted(files)

    if not files:
        print("No images found in:", INPUT_DIR)
        return

    rows = []
    for p in files:
        rows.append(eval_one(p, reco_model))

    # 寫 CSV
    csv_path = OUT_DIR / "results.csv"
    fieldnames = [
        "file", "gt", "pred", "ok",
        "digits", "dot_cx_ratio",
        "band_x1", "band_x2", "band_w", "img_w",
        "split4", "score4", "split3", "score3",
        "candidates", "valid",
        "format_ok", "range_ok",
        "reason",
        "reco_prob", "fb_meta",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # Summary
    total = len(rows)
    gt_ok = sum(1 for r in rows if r.get("gt"))
    pred_ok = sum(1 for r in rows if r.get("pred"))
    fmt_ok = sum(1 for r in rows if r.get("format_ok"))
    rng_ok = sum(1 for r in rows if r.get("range_ok"))
    ok = sum(1 for r in rows if r.get("ok"))

    # 失敗原因統計（簡化）
    fail_no_pred = sum(1 for r in rows if r.get("gt") and not r.get("pred"))
    fail_wrong = sum(1 for r in rows if r.get("gt") and r.get("pred") and not r.get("ok"))

    #
    fb_tried = sum(1 for r in rows if r.get("gt") and r.get("pred")=="" and len(r.get("digits","")) < 3 and r.get("band_x1") != "")
    fb_used  = sum(1 for r in rows if r.get("reason") == "fallback_band_cells")
    fb_meta_nonempty = sum(1 for r in rows if r.get("fb_meta"))

    summary = []
    summary.append(f"INPUT_DIR: {INPUT_DIR}")
    summary.append(f"TOTAL FILES: {total}")
    summary.append(f"WITH GT PARSED: {gt_ok}")
    summary.append(f"WITH PRED: {pred_ok}")
    summary.append(f"FORMAT_OK: {fmt_ok}")
    summary.append(f"RANGE_OK: {rng_ok}")
    summary.append(f"CORRECT: {ok}")
    summary.append(f"ACCURACY (correct / with_gt): {ok}/{gt_ok} = {(ok/gt_ok*100 if gt_ok else 0):.2f}%")
    summary.append(f"FAIL: no_pred={fail_no_pred}, wrong_pred={fail_wrong}")
    summary.append(f"FALLBACK_TRIED: {fb_tried}")
    summary.append(f"FALLBACK_USED: {fb_used}")
    summary.append(f"FALLBACK_META_NONEMPTY: {fb_meta_nonempty}")

    summary_path = OUT_DIR / "summary.txt"
    summary_path.write_text("\n".join(summary), encoding="utf-8")

    print("\n".join(summary))
    print("\nWrote:", csv_path)
    print("Wrote:", summary_path)

if __name__ == "__main__":
    main()
