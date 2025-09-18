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
├── Dockerfile
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
├── data/                  # runtime
├── models/                # runtime
├── pipeline/
│   ├── data_prep.py
│   ├── train.py
│   └── register_model_vertex_ai.py
└── deployment/
    ├── deploy_cloud_run.sh
    └── app/
        └── main.py
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



Perfect 🎉 — having a clear **README.md** will make your GitHub repo shine in interviews. Here’s a draft you can copy into your repo’s `README.md` (or merge into your existing one). It explains the **project goal**, **steps**, and how to reproduce everything on GCP.

---

# 📊 Vertex AI Churn Prediction (Census Income Dataset)

This project is an **end-to-end ML pipeline on Google Cloud** using **free-tier resources**.
It shows how to:

* Prepare data from BigQuery public datasets
* Train a scikit-learn pipeline locally
* Package the model in a FastAPI service
* Deploy it as a **serverless API on Cloud Run**

👉 Great for learning **Data Scientist / Data Engineer workflows** in GCP.

---

## 🚀 Tech Stack

* **Python 3.11**
* **scikit-learn** (Logistic Regression, preprocessing pipeline)
* **FastAPI** + **Uvicorn** (serving)
* **Docker** (containerization, built via Cloud Build)
* **Google Cloud**: BigQuery, Cloud Storage (optional), Artifact Registry, Cloud Run

---

## 📂 Project Structure

```
vertex-ai-churn-prediction/
├── Dockerfile                 # Container config (root level for Cloud Build)
├── requirements.txt           # Python dependencies
├── pipeline/                  # Data prep + training scripts
│   ├── data_prep.py
│   ├── train.py
│   └── register_model_vertex_ai.py  # (optional)
├── models/                    # Saved model + metrics (generated at runtime)
├── data/                      # Train/test CSVs (generated at runtime)
├── deployment/
│   ├── app/                   # FastAPI serving app
│   │   └── main.py
│   └── deploy_cloud_run.sh    # Deployment script
└── README.md
```

---

## 📝 Setup Instructions

### 1. Clone repo

```bash
git clone https://github.com/YOUR_USERNAME/vertex-ai-churn-prediction.git
cd vertex-ai-churn-prediction
```

### 2. Authenticate GCP

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### 3. Enable required APIs

```bash
./scripts/enable_apis.sh
```

### 4. Prepare data

```bash
python pipeline/data_prep.py --project_id YOUR_PROJECT_ID
```

This pulls the **Census Income dataset** from BigQuery and creates:

* `data/train.csv`
* `data/test.csv`

### 5. Train model

```bash
python pipeline/train.py
```

Outputs:

* `models/model.joblib` → trained pipeline
* `models/metrics.json` → evaluation metrics
* `models/schema.json` → feature schema

### 6. Deploy model to Cloud Run

```bash
./deployment/deploy_cloud_run.sh
```

This will:

* Build & push Docker image to **Artifact Registry**
* Deploy the container to **Cloud Run**
* Print a public URL for the service

---

## 🧪 Test the API

```bash
curl -X POST https://YOUR_CLOUD_RUN_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"records":[{"features":{"age":39,"workclass":"Private","education":"Bachelors","marital_status":"Never-married","occupation":"Adm-clerical","relationship":"Not-in-family","race":"White","sex":"Male","capital_gain":2174,"capital_loss":0,"hours_per_week":40,"native_country":"United-States"}}]}'
```

Example response:

```json
{"predictions": [0], "probabilities": [0.23]}
```

---

## 🎯 Why this project?

* Shows **full lifecycle**: data → training → deployment
* Uses **only free-tier GCP services**
* Mirrors **real-world MLops workflows**:

  * Reproducible pipelines with sklearn
  * Containerized deployment
  * Serverless inference API

---

## 🔮 Possible Extensions

* Swap Logistic Regression for XGBoost or TensorFlow
* Register the model in **Vertex AI Model Registry**
* Automate retraining with **Vertex AI Pipelines**
* Add CI/CD with **Cloud Build triggers**
