# 🗞️ Samachar Vani — Hindi Newspaper → Podcast

> Automatically convert scanned Hindi newspaper images into narrated audio podcasts using computer vision, OCR, LLM refinement, and text-to-speech.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20%7C%20CUDA%20%7C%20CPU-orange?logo=pytorch)
![YOLOv8](https://img.shields.io/badge/YOLOv8-DocLayNet-green)
![EasyOCR](https://img.shields.io/badge/OCR-Hindi%20%2B%20English-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

# 📖 What It Does(soon demo will be available)

Feed it a photo of a Hindi newspaper. It produces a fully narrated audio podcast of the **top 3 most important stories — automatically.**

```
📸 Newspaper Image  →  🎙️ Audio Podcast (.mp3)  +  📝 Script (.txt)
```

Tested on **Dainik Jagran** front pages. Works on any Hindi newspaper with a standard multi-column layout.

---
<img width="1431" height="697" alt="Screenshot 2026-03-08 at 7 39 14 PM" src="https://github.com/user-attachments/assets/c539782a-ced4-42db-8c57-2e9a7d3a7ec7" />

# 🌐 Web Application — Samachar Vani

This project includes a **modern web application** called **Samachar Vani**.

Users can:

- Upload Hindi newspaper **PDF or images**
- Automatically generate **AI podcasts**
- Listen directly in the browser
- Track previous podcast jobs
- Login / signup to manage personal podcasts

The UI is designed around a simple idea:

> **Sip chai ☕ and listen to the news.**

### Frontend Features

- Mobile-friendly UI  
- Futuristic gradient design  
- Hindi typography hero section  
- Dark / Light theme  
- AI voice waveform animation  
- User dashboard with previous podcasts  

---

# 🔐 Authentication

The web application supports **user authentication**.

Users can:

- Create an account
- Login securely
- Generate podcasts linked to their account

Passwords are stored using **bcrypt hashing** via **Passlib**.

---

# ⚙️ FastAPI Backend

The project includes a **FastAPI backend** that powers the web application.

### API Endpoints

| Endpoint | Purpose |
|---|---|
| `/upload` | Upload newspaper image |
| `/job/{id}` | Check processing status |
| `/audio/{id}` | Download generated podcast |
| `/script/{id}` | Download generated script |
| `/signup` | Create user account |
| `/login` | Authenticate user |

The backend runs the **complete AI pipeline asynchronously** and stores results in the database.

---

# 🗄️ Database & Job Tracking

The backend uses **SQLite + SQLModel**.

It tracks:

- users
- uploaded newspapers
- job processing status
- generated podcasts
- scripts

Example schema:

```
User
 ├─ id
 ├─ username
 ├─ email
 └─ password_hash

NewspaperJob
 ├─ id
 ├─ image_path
 ├─ script_path
 ├─ audio_path
 ├─ status
 └─ created_at
```

This allows users to see **previous podcast generations**.

---

# 🧠 LLM Refinement (Ollama)

To improve the quality of generated summaries, the project integrates **local LLMs via Ollama**.

The LLM refines extracted article summaries into **clean radio-style Hindi narration**.

Example models:

```
phi3
llama3:8b
```

Prompt constraints ensure:

- pure Hindi output  
- no English words  
- concise 3–4 sentence summaries  
- no explanations or instructions  

---

# 🔁 Pipeline

```
newspaper image
    │
    ▼
Step 1  ── Preprocessing         Deskew, denoise, dual-resolution strategy
    │                            (full-res for OCR, 1280px for layout detection)
    ▼
Step 2  ── Layout Detection      YOLOv8x fine-tuned on DocLayNet
    │                            Detects: title, text, section_header, table, figure
    ▼
Step 3  ── Region Filtering      NMS deduplication of overlapping boxes
    │                            Removes: page headers, footers, advertisements
    ▼
Step 4  ── Crop Regions          Groups title + body into article blocks
    ▼
Step 5  ── OCR                   EasyOCR — Hindi + English
    ▼
Step 6  ── Headline Detection    Extract best headline
    ▼
Step 7  ── Importance Scoring    position · font size · keywords
    ▼
Step 8  ── Article Ranking       Select top stories
    ▼
Step 9  ── Text Reconstruction   Clean OCR noise
    ▼
Step 10 ── LLM Refinement        Ollama improves summary quality
    ▼
Step 11 ── Podcast Script        Hindi radio narration
    ▼
Step 12 ── Text-to-Speech        Edge TTS / gTTS
    ▼
🎙️ Audio Podcast (.mp3)
```

---

# ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Backend | FastAPI |
| Frontend | HTML + CSS + JavaScript |
| Layout Detection | YOLOv8x + DocLayNet |
| OCR | EasyOCR |
| LLM | Ollama |
| TTS | Edge-TTS / gTTS |
| Database | SQLite + SQLModel |
| Audio | pydub + ffmpeg |
| Deployment | Vercel + ngrok |

---

# 🚀 Setup

### Prerequisites

- Python 3.10+
- Apple M1/M2/M3 Mac, CUDA GPU, or CPU
- ffmpeg

Install ffmpeg:

```bash
brew install ffmpeg
```

---

### Install Dependencies

```bash
git clone https://github.com/anushka7220/samachar-vani.git
cd samachar-vani

pip install -r requirements.txt
```

---

# ▶️ Running the System

Start Ollama:

```bash
ollama serve
```

Download a model:

```bash
ollama run phi3
```

Run backend:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open browser:

```
http://localhost:8000
```

---

# 🌍 Deployment Architecture

To keep the project **completely free**, Ollama runs locally.

```
Frontend (Vercel)
        │
        ▼
FastAPI Backend (Local Machine)
        │
        ▼
YOLO + OCR Pipeline
        │
        ▼
Ollama LLM
        │
        ▼
Edge TTS
        │
        ▼
Podcast Output
```

Public API access is provided using a tunnel:

```
ngrok
```

Example:

```
localhost:8000 → https://abc123.ngrok.app
```

---

# 📂 Project Structure

```
samachar-vani/
│
├── api.py
├── main.py
├── requirements.txt
│
├── app/
│   ├── database.py
│   ├── models.py
│   ├── crud.py
│   └── auth.py
│
├── pipeline/
│   ├── orchestrator.py
│   ├── step1_preprocess.py
│   ├── step2_layout_detection.py
│   ├── step3_4_filter_crop.py
│   ├── step5_ocr.py
│   ├── step6_7_8_score_rank.py
│   ├── step9_10_11_script.py
│   ├── step12_tts.py
│   └── local_summary_refiner.py
│
├── frontend/
│   └── index.html
│
├── uploads/
└── results/
```

---

# 📊 Performance

Tested on **Apple M1 MacBook Air (8GB RAM)**

| Step | Time |
|---|---|
| Preprocessing | ~3s |
| Layout Detection | ~2.5s |
| OCR | ~15s |
| LLM Refinement | ~4s |
| TTS | ~7s |
| **Total** | **~30 seconds per newspaper** |

---

# ⚠️ Known Limitations

- OCR accuracy decreases for very low-quality scans
- Advertisements occasionally detected as articles
- Mixed Hindi-English headlines may split across OCR lines

---

# 🛣️ Roadmap

- Fine-tune YOLOv8 on Hindi newspaper dataset
- Support Marathi / Bengali / Gujarati newspapers
- WhatsApp news podcast bot
- Daily automated podcast generation
- Personalized news feed

---

# 📄 License

MIT License — see LICENSE.

---

# 🙏 Acknowledgements

- Ultralytics YOLOv8  
- DocLayNet dataset  
- EasyOCR  
- Ollama  
- Edge-TTS  
