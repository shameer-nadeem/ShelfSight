from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import numpy as np
import gradio as gr
import io

MODEL_PATH  = "model/saad_best.pt"
OUR_BRANDS  = ["coca-cola-500ml", "coca-cola-can"]
COMPETITOR  = "competitor"
CONF        = 0.5
GRID_ROWS   = 2
GRID_COLS   = 5

PLANOGRAM = [
    ["coca-cola-500ml", "coca-cola-500ml", "coca-cola-can", "coca-cola-can", "competitor"],
    ["coca-cola-500ml", "coca-cola-can",   "coca-cola-can", "competitor",   "competitor"],
]

COLORS = {
    "coca-cola-500ml": (232, 89,  60),
    "coca-cola-can":   (192, 57,  43),
    "competitor":      (29,  158, 117),
}

model = YOLO(MODEL_PATH)

def run_inference(image_array):
    img_pil = Image.fromarray(image_array)
    img_w, img_h = img_pil.size

    results    = model.predict(source=image_array, conf=CONF, verbose=False)
    result     = results[0]
    detections = []

    for box in result.boxes:
        cls_id = int(box.cls)
        name   = model.names[cls_id]
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        detections.append({
            "class":    name,
            "conf":     float(box.conf),
            "center_x": (x1+x2)/2,
            "center_y": (y1+y2)/2,
        })

    det_counts  = {}
    for d in detections:
        det_counts[d["class"]] = det_counts.get(d["class"], 0) + 1

    brand_count = sum(det_counts.get(b, 0) for b in OUR_BRANDS)
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
                    "pos":      "Row " + str(r+1) + " Col " + str(c+1),
                    "expected": expected or "empty",
                    "actual":   actual   or "not detected",
                })

    compliance = (matching_cells / total_cells * 100) if total_cells > 0 else 0.0

    recommendations = []
    missing_brand = [m for m in mismatches if m["expected"] in OUR_BRANDS]
    wrong_product = [m for m in mismatches if m["expected"] not in OUR_BRANDS and "not detected" not in m["actual"]]
    empty_slots   = [m for m in mismatches if "not detected" in m["actual"]]

    if missing_brand:
        positions = ", ".join([m["pos"] for m in missing_brand])
        gain = len(missing_brand) / total_cells * 100
        recommendations.append({"text": "Add Coca-Cola at: " + positions, "gain": gain})
    for m in wrong_product:
        recommendations.append({"text": "Replace '" + m["actual"] + "' with '" + m["expected"] + "' at " + m["pos"], "gain": 1/total_cells*100})
    if empty_slots:
        positions = ", ".join([m["pos"] for m in empty_slots])
        recommendations.append({"text": "Fill empty slots at: " + positions, "gain": len(empty_slots)/total_cells*100})

    return {
        "sos":             sos,
        "detections":      det_counts,
        "compliance":      compliance,
        "mismatches":      mismatches,
        "recommendations": recommendations,
        "actual_grid":     actual_grid,
    }


fastapi_app = FastAPI()

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@fastapi_app.get("/", response_class=HTMLResponse)
async def root():
    with open("frontend/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@fastapi_app.post("/analyse")
async def analyse(file: UploadFile = File(...)):
    contents = await file.read()
    image    = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
    result   = run_inference(image)
    return JSONResponse(content=result)


def gradio_analyse(image):
    if image is None:
        return "No image uploaded.", "No image uploaded.", "No image uploaded."
    result = run_inference(image)
    sos_text  = "Coca-Cola SOS: " + str(round(result["sos"], 1)) + "%\n\n"
    total = sum(result["detections"].values())
    for name, count in result["detections"].items():
        pct = count/total*100 if total > 0 else 0
        sos_text += name + ": " + str(count) + " facings (" + str(round(pct,1)) + "%)\n"
    comp_text = "Compliance: " + str(round(result["compliance"],1)) + "%\n\n"
    for m in result["mismatches"]:
        comp_text += m["pos"] + ": expected " + m["expected"] + ", found " + m["actual"] + "\n"
    reco_text = ""
    for i, r in enumerate(result["recommendations"], 1):
        reco_text += str(i) + ". " + r["text"] + "\n"
        if r.get("gain"):
            reco_text += "   Gain: +" + str(round(r["gain"],1)) + "%\n"
    if not reco_text:
        reco_text = "Perfect compliance!"
    return sos_text, comp_text, reco_text


gradio_demo = gr.Interface(
    fn=gradio_analyse,
    inputs=gr.Image(type="numpy", label="Upload shelf photo"),
    outputs=[
        gr.Textbox(label="Share of Shelf", lines=8),
        gr.Textbox(label="Planogram Compliance", lines=8),
        gr.Textbox(label="Recommendations", lines=8),
    ],
    title="ShelfSight — Gradio Interface",
    description="Fallback interface — main UI available at /",
)

app = gr.mount_gradio_app(fastapi_app, gradio_demo, path="/gradio")
