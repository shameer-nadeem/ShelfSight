import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

MODEL_PATH  = "model/saad_best.pt"
OUR_BRAND   = "coca-cola"
COMPETITORS = ["fanta", "sprite"]
CONF        = 0.5
GRID_ROWS   = 2
GRID_COLS   = 5

PLANOGRAM = [
    ["coca-cola", "coca-cola", "coca-cola", "fanta",   "fanta"  ],
    ["coca-cola", "coca-cola", "sprite",    "sprite",  "sprite" ],
]

COLORS = {
    "coca-cola": (232, 89, 60),
    "fanta":     (240, 160, 48),
    "sprite":    (0,   200, 150),
}

model = YOLO(MODEL_PATH)

def analyse(image):
    if image is None:
        return None, "No image uploaded.", "No image uploaded.", "No image uploaded."

    img_pil = Image.fromarray(image)
    img_w, img_h = img_pil.size

    results    = model.predict(source=image, conf=CONF, verbose=False)
    result     = results[0]
    detections = []

    draw = ImageDraw.Draw(img_pil)
    for box in result.boxes:
        cls_id = int(box.cls)
        name   = model.names[cls_id]
        conf_  = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color  = COLORS.get(name, (150, 150, 150))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.rectangle([x1, y1-20, x1+len(name)*8+8, y1], fill=color)
        draw.text((x1+4, y1-18), f"{name} {conf_*100:.0f}%", fill=(255,255,255))
        detections.append({
            "class":    name,
            "conf":     conf_,
            "center_x": (x1+x2)/2,
            "center_y": (y1+y2)/2,
        })

    det_counts  = {}
    for d in detections:
        det_counts[d["class"]] = det_counts.get(d["class"], 0) + 1

    brand_count = det_counts.get(OUR_BRAND, 0)
    total_count = sum(det_counts.values())
    sos = (brand_count / total_count * 100) if total_count > 0 else 0.0

    sos_text = f"SHARE OF SHELF\n{'='*35}\n"
    for name, count in det_counts.items():
        pct = count / total_count * 100 if total_count > 0 else 0
        bar = "█" * int(pct / 5)
        sos_text += f"{name:<15}: {count:>3} facings  {pct:>5.1f}%  {bar}\n"
    sos_text += f"\n{'='*35}\nCoca-Cola SOS: {sos:.1f}%\n"
    if sos >= 50:   sos_text += "Status: GOOD"
    elif sos >= 30: sos_text += "Status: FAIR"
    else:           sos_text += "Status: POOR"

    cell_w = img_w / GRID_COLS
    cell_h = img_h / GRID_ROWS
    cell_products = {(r,c): [] for r in range(GRID_ROWS) for c in range(GRID_COLS)}
    for d in detections:
        col = min(int(d["center_x"] / cell_w), GRID_COLS-1)
        row = min(int(d["center_y"] / cell_h), GRID_ROWS-1)
        cell_products[(row, col)].append(d["class"])

    actual_grid = []
    for r in range(GRID_ROWS):
        row_data = []
        for c in range(GRID_COLS):
            products = cell_products[(r, c)]
            row_data.append(max(set(products), key=products.count) if products else None)
        actual_grid.append(row_data)

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
    if   compliance >= 80: status = "EXCELLENT"
    elif compliance >= 60: status = "GOOD"
    elif compliance >= 40: status = "FAIR"
    else:                  status = "POOR"

    comp_text  = f"PLANOGRAM COMPLIANCE\n{'='*35}\n"
    comp_text += f"Matching cells : {matching_cells}/{total_cells}\n"
    comp_text += f"Compliance     : {compliance:.1f}%\n"
    comp_text += f"Status         : {status}\n\n"
    comp_text += "Reference planogram:\n"
    for r in range(GRID_ROWS):
        row_str = " | ".join([(PLANOGRAM[r][c] or "empty")[:10].center(10) for c in range(GRID_COLS)])
        comp_text += f"  [{row_str}]\n"
    comp_text += "\nActual shelf:\n"
    for r in range(GRID_ROWS):
        row_str = " | ".join([(actual_grid[r][c] or "empty")[:10].center(10) for c in range(GRID_COLS)])
        comp_text += f"  [{row_str}]\n"
    if mismatches:
        comp_text += "\nMismatches:\n"
        for m in mismatches:
            comp_text += f"  {m['pos']}: expected '{m['expected']}' found '{m['actual']}'\n"

    reco_text = f"RECOMMENDATIONS\n{'='*35}\n"
    missing_brand = [m for m in mismatches if m["expected"] == OUR_BRAND]
    wrong_product = [m for m in mismatches if m["expected"] != OUR_BRAND and "not detected" not in m["actual"]]
    empty_slots   = [m for m in mismatches if "not detected" in m["actual"]]
    rec = 1
    if missing_brand:
        positions = ", ".join([m["pos"] for m in missing_brand])
        gain = len(missing_brand) / total_cells * 100
        reco_text += f"{rec}. Add coca-cola at: {positions}\n   Gain: +{gain:.1f}%\n\n"
        rec += 1
    for m in wrong_product:
        reco_text += f"{rec}. Replace '{m['actual']}' with '{m['expected']}' at {m['pos']}\n\n"
        rec += 1
    if empty_slots:
        positions = ", ".join([m["pos"] for m in empty_slots])
        reco_text += f"{rec}. Fill empty slots at: {positions}\n\n"
    if not mismatches:
        reco_text += "Perfect compliance - no changes needed."
    potential = (matching_cells + len(mismatches)) / total_cells * 100
    reco_text += f"\nCurrent : {compliance:.1f}%\nAfter fix: {potential:.1f}%\nGain: +{potential-compliance:.1f}%"

    return img_pil, sos_text, comp_text, reco_text


with gr.Blocks(title="ShelfSight", theme=gr.themes.Base()) as demo:
    gr.Markdown("# ShelfSight — AI Shelf Intelligence\n**CSC-233 AI Lab | BNU Spring 2026 | YOLOv8n+Aug | mAP50: 95.06%**\n\nUpload a shelf photo to detect products and calculate Share of Shelf and Planogram Compliance.")
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload shelf photo", type="numpy")
            analyse_btn = gr.Button("Analyse Shelf", variant="primary")
        with gr.Column(scale=1):
            image_output = gr.Image(label="Detected products")
    with gr.Row():
        sos_output  = gr.Textbox(label="Share of Shelf", lines=10)
        comp_output = gr.Textbox(label="Planogram Compliance", lines=10)
        reco_output = gr.Textbox(label="Recommendations", lines=10)
    analyse_btn.click(fn=analyse, inputs=image_input, outputs=[image_output, sos_output, comp_output, reco_output])

demo.launch()
