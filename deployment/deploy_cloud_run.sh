#!/bin/bash
set -e

# Variables
PROJECT_ID=$(gcloud config get-value project)
REGION="europe-north1"   # or your chosen region
REPO_NAME="my-docker-repo"
SERVICE_NAME="churn-service"

# Enable Artifact Registry if not already
gcloud services enable artifactregistry.googleapis.com

# Create Artifact Registry repo if not exists
if ! gcloud artifacts repositories describe $REPO_NAME --location=$REGION >/dev/null 2>&1; then
  gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION \
    --description="Docker repo for ML models"
fi

# Build and push image
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$SERVICE_NAME:latest .

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$SERVICE_NAME:latest \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated
