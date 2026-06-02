# ShelfSight — AI-Based Shelf Intelligence System

> CSC-233 Artificial Intelligence Lab Project — FAST NUCES

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
- Generate **Recommendations** to improve shelf compliance

---

## Team

| Name | Roll Number | Role |
|------|-------------|------|
| Shameer Nadeem (Lead) | F2024-0427 | Project Lead, Model 1 (YOLOv8n), Streamlit UI |
| Ibrahim Zahid | F2024-0550 | Image Collection, Model 2 (YOLOv8s) |
| Saad Riaz | F2024-0543 | Dataset Search, Model 3 (YOLOv8n+Aug) — Winner |
| Samia Rehan | F2024-0488| Dataset Search, Model 4 (MobileNet+SSD) |
| Ayesha Gohar | F2024-0135 | Model 5 (ResNet50), Poster & Standee |

---

## Repository Structure

```
ShelfSight/
├── backend/
│   ├── app.py                          # FastAPI backend server
│   ├── inference.py                    # YOLO model inference + shelf metrics
│   ├── instruction.txt                 # Setup and usage instructions
│   └── requirements.txt               # Backend dependencies
├── frontend/
│   └── index.html                      # Custom HTML/CSS/JS frontend
├── model/
│   └── saad_best.pt                    # Trained YOLOv8n+Aug model (winner)
├── Notebooks/
│   ├── MobileNetV2+SSD_Model4.ipynb   # Samia — Model 4
│   ├── ResNet50.ipynb                  # Ayesha — Model 5
│   ├── YOLOv8n_Augmented_Model.ipynb  # Saad — Model 3 (Winner)
│   ├── YOLOv8n.ipynb                   # Shameer — Model 1
│   └── YOLOv8s.ipynb                   # Ibrahim — Model 2
├── Results/
│   ├── Comparisons/                    # Side-by-side model comparison charts
│   ├── YOLOv8n_Augmented_Results/     # Saad's training results
│   ├── YOLOv8n_Results/               # Shameer's training results
│   └── YOLOv8s_Results/               # Ibrahim's training results
├── app.py                              # Hugging Face Spaces deployment (Gradio)
├── index.html                          # Root frontend entry point
├── poster.pdf                          # Project poster (Ayesha)
├── requirements.txt                    # App dependencies
└── README.md
```

---

## Model Weights

All trained model weights are available for download from the shared Google Drive folder:

| Model | Member | Download |
|-------|--------|----------|
| YOLOv8n | Shameer | [Download](https://drive.google.com/drive/folders/1tnHkBKC1cDyQXYLj_5WucJlghT_xBdfz?usp=sharing) |
| YOLOv8s | Ibrahim | [Download](https://drive.google.com/drive/folders/1tnHkBKC1cDyQXYLj_5WucJlghT_xBdfz?usp=sharing) |
| **YOLOv8n + Augmentation (Winner)** | **Saad** | [**Download**](https://drive.google.com/drive/folders/1tnHkBKC1cDyQXYLj_5WucJlghT_xBdfz?usp=sharing) |
| MobileNet+SSD | Samia | [Download](https://drive.google.com/drive/folders/1tnHkBKC1cDyQXYLj_5WucJlghT_xBdfz?usp=sharing) |
| ResNet50 | Ayesha | [Download](https://drive.google.com/drive/folders/1tnHkBKC1cDyQXYLj_5WucJlghT_xBdfz?usp=sharing) |

> All weights are stored in a shared Google Drive folder.

---

## Model Comparison

| Model | Member | mAP50 | Precision | Recall |
|-------|--------|-------|-----------|--------|
| YOLOv8n | Shameer | 94.61% | 93.79% | 92.68% |
| YOLOv8s | Ibrahim | 94.61% | 93.79% | 92.68% |
| **YOLOv8n + Augmentation** | **Saad** | **95.06%** | **94.09%** | **92.91%** |
| MobileNet+SSD | Samia | 47.80% | 44.00% | 43.60% |
| ResNet50 | Ayesha | 51.20% | 49.20% | 48.60% |

**Winner: Saad's YOLOv8n + Augmentation — 95.06% mAP50**

---

## How It Works

1. User uploads a shelf photo
2. YOLOv8 model detects all products and draws bounding boxes
3. System counts facings per brand (coca-cola, fanta, sprite)
4. Share of Shelf is calculated: `brand_facings / total_facings × 100`
5. Shelf is divided into a grid and compared against the reference planogram
6. Compliance score and improvement recommendations are generated

---

## Dataset

- **Source:** Roboflow Universe — soda-bottles-haga dataset
- **Dataset Link:** https://universe.roboflow.com/rf20-vl/soda-bottles-haga
https://universe.roboflow.com/larc2022-4tijc/coke-c5aa1/images/DrQmTLBvFTiq0MWrR9Gy?queryText=&pageSize=50&startingIndex=0&browseQuery=true
https://universe.roboflow.com/ashis-jenisharajendran/sprite-aqswq
https://universe.roboflow.com/smart-vm/coca-cola-16990/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true
https://app.roboflow.com/saad-riaz-fzzs8/fanta-lweui-8vnsg/
https://universe.roboflow.com/training-5hycg/coca-cola-obpxr/browse

All the datasets were mergered into one 

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
