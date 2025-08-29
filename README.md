# Vertex AI Customer Churn Prediction

🚀 An end-to-end **ML pipeline on Google Cloud Vertex AI**, built to learn the basics of **data engineering + data science workflows in GCP**.  
This project trains a simple model to predict whether a customer/person falls into a **"high income" category (>100K)** (proxy for churn/no churn).  

## 📌 Objectives
- Learn the basics of **Google Cloud Vertex AI** (pipelines, model training, deployment).  
- Practice **Python + ML integration with GCP services**.  
- Showcase an **end-to-end project** for interviews and portfolio.  
- Use **only free resources** within Google Cloud’s free tier.  

---

## 🗂️ Project Structure
```bash
vertex-ai-churn-prediction/
│── notebooks/              # Jupyter notebooks for exploration
│── pipeline/               # Pipeline components (Python scripts)
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── pipeline.py
│── deployment/             # Deployment scripts
│   ├── deploy_model.py
│   ├── predict.py
│── data/                   # (Optional small sample dataset for local testing)
│── README.md               # Project guide
│── requirements.txt        # Dependencies
```
---

## 🛠️ Tech Stack
- **Google Cloud Platform**  
  - Vertex AI  
  - BigQuery (public datasets)  
  - Cloud Storage  
- **Python**  
  - pandas, scikit-learn, joblib  
  - google-cloud-sdk  

---

## 🚀 Steps to Run

### 1️⃣ Setup GCP
1. Create a new GCP project (free tier).  
2. Enable APIs: Vertex AI, BigQuery, Cloud Storage.  
3. Authenticate:
   ```bash
   gcloud init
   gcloud auth login
