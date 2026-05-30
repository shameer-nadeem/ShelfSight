import gradio as gr
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import uvicorn

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

def run_inference(image_array):
    img_pil = Image.fromarray(image_array)
    img_w, img_h = img_pil.size

    results    = model.predict(source=image_array, conf=CONF, verbose=False)
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

    recommendations = []
    missing_brand = [m for m in mismatches if m["expected"] == OUR_BRAND]
    wrong_product = [m for m in mismatches if m["expected"] != OUR_BRAND and "not detected" not in m["actual"]]
    empty_slots   = [m for m in mismatches if "not detected" in m["actual"]]

    if missing_brand:
        positions = ", ".join([m["pos"] for m in missing_brand])
        gain = len(missing_brand) / total_cells * 100
        recommendations.append({"text": f"Add coca-cola at: {positions}", "gain": gain})
    for m in wrong_product:
        recommendations.append({"text": f"Replace '{m['actual']}' with '{m['expected']}' at {m['pos']}", "gain": 1/total_cells*100})
    if empty_slots:
        positions = ", ".join([m["pos"] for m in empty_slots])
        recommendations.append({"text": f"Fill empty slots at: {positions}", "gain": len(empty_slots)/total_cells*100})

    return {
        "sos":             sos,
        "detections":      det_counts,
        "compliance":      compliance,
        "status":          status,
        "mismatches":      mismatches,
        "recommendations": recommendations,
        "actual_grid":     actual_grid,
    }


# ── FastAPI app ───────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("frontend/index.html", "r") as f:
        return HTMLResponse(content=f.read())



# Override with proper file upload
from fastapi import File, UploadFile
import io

@app.post("/analyse")
async def analyse_image(file: UploadFile = File(...)):
    contents = await file.read()
    image    = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
    result   = run_inference(image)
    return JSONResponse(content=result)


# ── Gradio fallback UI ────────────────────────────────────
def gradio_analyse(image):
    if image is None:
        return None, "No image", "No image", "No image"
    result = run_inference(image)

    sos_text  = f"Coca-Cola SOS: {result['sos']:.1f}%\n\n"
    for name, count in result["detections"].items():
        total = sum(result["detections"].values())
        pct   = count/total*100 if total > 0 else 0
        sos_text += f"{name}: {count} facings ({pct:.1f}%)\n"

    comp_text  = f"Compliance: {result['compliance']:.1f}% — {result['status']}\n\n"
    for m in result["mismatches"]:
        comp_text += f"{m['pos']}: expected {m['expected']}, found {m['actual']}\n"

    reco_text = ""
    for i, r in enumerate(result["recommendations"], 1):
        reco_text += f"{i}. {r['text']}\n"
        if r.get("gain"):
            reco_text += f"   Gain: +{r['gain']:.1f}%\n"
    if not reco_text:
        reco_text = "Perfect compliance!"

    return image, sos_text, comp_text, reco_text


gradio_app = gr.Interface(
    fn=gradio_analyse,
    inputs=gr.Image(type="numpy", label="Upload shelf photo"),
    outputs=[
        gr.Image(label="Detections"),
        gr.Textbox(label="Share of Shelf", lines=8),
        gr.Textbox(label="Planogram Compliance", lines=8),
        gr.Textbox(label="Recommendations", lines=8),
    ],
    title="ShelfSight — AI Shelf Intelligence",
    description="CSC-233 AI Lab | BNU Spring 2026 | YOLOv8n+Aug | mAP50: 95.06%",
)

app = gr.mount_gradio_app(app, gradio_app, path="/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
