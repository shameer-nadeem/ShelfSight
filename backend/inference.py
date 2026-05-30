from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = "../model/saad_best.pt"
OUR_BRAND  = "coca-cola"
COMPETITORS = ["fanta", "sprite"]
CONF       = 0.5
GRID_ROWS  = 2
GRID_COLS  = 5

PLANOGRAM = [
    ["coca-cola", "coca-cola", "coca-cola", "fanta",   "fanta"  ],
    ["coca-cola", "coca-cola", "sprite",    "sprite",  "sprite" ],
]

model = YOLO(MODEL_PATH)

def run_inference(image_path: str) -> dict:
    results = model.predict(source=image_path, conf=CONF, verbose=False)
    result  = results[0]
    img_w   = result.orig_shape[1]
    img_h   = result.orig_shape[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls)
        name   = model.names[cls_id]
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        detections.append({
            "class":    name,
            "conf":     float(box.conf),
            "center_x": (x1 + x2) / 2,
            "center_y": (y1 + y2) / 2,
        })

    # ── Share of Shelf ────────────────────────────────────────
    det_counts = {}
    for d in detections:
        det_counts[d["class"]] = det_counts.get(d["class"], 0) + 1

    brand_count = det_counts.get(OUR_BRAND, 0)
    total_count = sum(det_counts.values())
    sos = (brand_count / total_count * 100) if total_count > 0 else 0.0

    # ── Assign to grid ────────────────────────────────────────
    cell_w = img_w / GRID_COLS
    cell_h = img_h / GRID_ROWS

    cell_products = {(r, c): [] for r in range(GRID_ROWS) for c in range(GRID_COLS)}
    for d in detections:
        col = min(int(d["center_x"] / cell_w), GRID_COLS - 1)
        row = min(int(d["center_y"] / cell_h), GRID_ROWS - 1)
        cell_products[(row, col)].append(d["class"])

    actual_grid = []
    for r in range(GRID_ROWS):
        row_data = []
        for c in range(GRID_COLS):
            products = cell_products[(r, c)]
            row_data.append(max(set(products), key=products.count) if products else None)
        actual_grid.append(row_data)

    # ── Planogram compliance ──────────────────────────────────
    total_cells    = GRID_ROWS * GRID_COLS
    matching_cells = 0
    mismatches     = []

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            expected = PLANOGRAM[r][c]
            actual   = actual_grid[r][c]
            if expected == actual:
                matching_cells += 1
            else:
                mismatches.append({
                    "pos":      f"Row {r+1} Col {c+1}",
                    "expected": expected or "empty",
                    "actual":   actual   or "not detected",
                })

    compliance = (matching_cells / total_cells * 100) if total_cells > 0 else 0.0

    # ── Recommendations ───────────────────────────────────────
    recommendations = []
    missing_brand = [m for m in mismatches if m["expected"] == OUR_BRAND]
    wrong_product = [m for m in mismatches if m["expected"] != OUR_BRAND
                     and "not detected" not in m["actual"]]
    empty_slots   = [m for m in mismatches if "not detected" in m["actual"]]

    if missing_brand:
        positions = ", ".join([m["pos"] for m in missing_brand])
        gain = len(missing_brand) / total_cells * 100
        recommendations.append({
            "text": f"Add {OUR_BRAND} at: {positions}",
            "gain": gain,
        })

    for m in wrong_product:
        recommendations.append({
            "text": f"Replace '{m['actual']}' with '{m['expected']}' at {m['pos']}",
            "gain": 1 / total_cells * 100,
        })

    if empty_slots:
        positions = ", ".join([m["pos"] for m in empty_slots])
        recommendations.append({
            "text": f"Fill empty slots at: {positions}",
            "gain": len(empty_slots) / total_cells * 100,
        })

    return {
        "sos":             sos,
        "detections":      det_counts,
        "compliance":      compliance,
        "mismatches":      mismatches,
        "recommendations": recommendations,
        "actual_grid":     actual_grid,
    }
