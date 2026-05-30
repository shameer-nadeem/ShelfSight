# ShelfSight — AI-Based Shelf Intelligence System

> CSC-233 Artificial Intelligence Lab Project 

---

## Live Application

| Link | Description |
|------|-------------|
| [**Live App**](https://shameer-nadeem-shelfsight.hf.space) | Hosted on Hugging Face Spaces |
| [**Frontend UI**](https://shameer-nadeem.github.io/ShelfSight) | Hosted on GitHub Pages |

---

## Project Overview

ShelfSight is an AI-powered retail shelf analysis system that uses object detection to:

- Detect products on retail shelves in real time
- Calculate **Share of Shelf (SOS)** — percentage of shelf space occupied by Coca-Cola vs competitors
- Evaluate **Planogram Compliance** — how closely the real shelf matches the reference layout
- Generate **Recommendations** to improve compliance

---

## Team

| Name | Roll Number | Role |
|------|-------------|------|
| Shameer Nadeem (Lead) | F2024-0427 | Project Lead, Model 1 (YOLOv8n), Streamlit UI |
| Ibrahim Zahid | F2024-0550 | Image Collection, Model 2 (YOLOv8s) |
| Saad Riaz | F2024- | Dataset Search, Model 3 (YOLOv8n+Aug) — Winner |
| Samia Rehan | — | Dataset Search, Model 4 (MobileNet+SSD) |
| Ayesha Gohar | — | Model 5 (ResNet50), Poster & Standee |

---

## Repository Structure

```
ShelfSight/
├── frontend/
│   └── index.html          # Custom HTML/CSS/JS frontend
├── backend/
│   ├── app.py              # FastAPI backend server
│   ├── inference.py        # YOLO model inference + shelf metrics
│   └── requirements.txt    # Backend dependencies
├── model/
│   └── saad_best.pt        # Trained YOLOv8n+Aug model (winner)
├── training/
│   ├── Shameer_YOLOv8n.ipynb
│   ├── Ibrahim_YOLOv8s.ipynb
│   ├── Saad_YOLOv8n_Augmented_Model.ipynb
│   ├── Samia_MobileNetV2+SSD_Model4.ipynb
│   └── Ayesha_ResNet50.ipynb
├── app.py                  # Hugging Face Spaces deployment (Gradio)
├── requirements.txt        # App dependencies
└── README.md
```

---

## Model Comparison

| Model | Member | mAP50 | Precision | Recall |
|-------|--------|-------|-----------|--------|
| YOLOv8n | Shameer | 94.61% | 93.79% | 92.68% |
| YOLOv8s | Ibrahim | 94.61% | 93.79% | 92.68% |
| **YOLOv8n + Augmentation** | **Saad** | **95.06%** | **94.09%** | **92.91%** |
| MobileNet+SSD | Samia | — | — | — |
| ResNet50 | Ayesha | 51.20% | 49.20% | 48.60% |

**Winner: Saad's YOLOv8n + Augmentation — 95.06% mAP50**

---

## How It Works

1. User uploads a shelf photo
2. YOLOv8 model detects all products and draws bounding boxes
3. System counts facings per brand (coca-cola, fanta, sprite)
4. Share of Shelf is calculated: `brand_facings / total_facings x 100`
5. Shelf is divided into a grid and compared against the reference planogram
6. Compliance score and improvement recommendations are generated

---

## Dataset

- **Source:** Roboflow Universe — soda-bottles-haga dataset
- **Dataset Link:** https://universe.roboflow.com/rf20-vl/soda-bottles-haga
- **Size:** 2,249 labeled images
- **Classes:** coca-cola, fanta, sprite
- **Format:** YOLOv8 (bounding boxes)
- **Split:** 70% train / 20% validation / 10% test

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| AI Model | YOLOv8 (Ultralytics) |
| Backend | FastAPI + Python |
| Frontend | HTML / CSS / JavaScript |
| Live Deployment | Hugging Face Spaces (Gradio) |
| Dataset Management | Roboflow |
| Training | Google Colab (T4 GPU) |
| Version Control | GitHub |



