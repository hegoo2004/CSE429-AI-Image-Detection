# CSE429 — AI-Generated Image Detection
**E-JUST | Spring 2026 | Dr. Ahmed Gomaa**

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/YOUR_USERNAME/ai-image-detector)

Re-implementation of **UniversalFakeDetect** (Ojha et al., CVPR 2023).  
Detects AI-generated images using a frozen CLIP-ViT-L/14 backbone + 769-parameter linear classifier.

---

## 🌐 Live Demo
👉 **[Try it on Hugging Face Spaces](https://huggingface.co/spaces/YOUR_USERNAME/ai-image-detector)**

Upload any image and the model will tell you if it's real or AI-generated.

---

## 👥 Team

| Name | ID | Role |
|------|-----|------|
| Mohamed Ahmed Mohamed Elhageen | 120220207 | Model Architecture & Implementation |
| Youssef Ahmed Abo Wali | 120220204 | Dataset Collection & Preprocessing |
| Belal Amr Abdelkarim | 120220168 | Training Pipeline & Optimization |
| Mohamed Mahmoud Kotb | 120220042 | Evaluation & Results Analysis |
| Seif Eldin Ebeid | 120220032 | Research Extension & Generalization |
| Omar Ibrahim Abdelhamid | 120220132 | System Deployment & Web Application |

---

## 📁 Project Structure

```
CSE429_FinalProject/
├── app.py                  ← Gradio web app (Hugging Face Spaces)
├── model.py                ← LinearProbingDetector + NearestNeighbourDetector
├── dataset.py              ← Download + preprocess datasets
├── dataloader.py           ← PyTorch DataLoader
├── train.py                ← Training loop + baselines
├── predict.py              ← Test any image from command line
├── visualise_results.py    ← Generate plots (loss, confusion matrix, ROC)
├── extension.py            ← Zero-shot eval on unseen generators
├── requirements.txt
├── tests/
│   └── test_all.py         ← 21 unit tests
└── checkpoints/
    └── best.pt             ← Trained model checkpoint
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/hegoo2004/CSE429-AI-Image-Detection.git
cd CSE429-AI-Image-Detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download and preprocess datasets
python dataset.py --datasets cifake fakeface --max_per_class 5000

# 4. Train
python train.py --epochs 20 --batch_size 32 --num_workers 2

# 5. Test an image
python predict.py --image path/to/image.jpg

# 6. Run all tests
python tests/test_all.py

# 7. Run web app locally
python app.py
```

---

## 📊 Results

| Model | AUC | Accuracy | F1 | AP |
|-------|-----|----------|----|-----|
| Random Classifier | 0.502| 0.502 | 0.4997 | 0.4984 |
| Simple CNN (baseline) |0.9732 | 0.9011 | 0.9038 | 0.9732 |
| **LinearProbingDetector (ours)** | **0.9247** | **0.8349** | **0.8456** | **0.9283** |

---

## 📄 Paper
Ojha, U., Li, Y., & Lee, Y.J. (2023). *Towards Universal Fake Image Detectors that Generalize Across Generative Models*. CVPR 2023.

---

## 🏫 Course
CSE 429 — Computer Vision and Pattern Recognition  
Egypt-Japan University of Science and Technology (E-JUST)  
Spring 2026 | Dr. Ahmed Gomaa
