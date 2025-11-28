# 🚀 Deploying Manga Translator to RunPod

This guide explains how to deploy the Manga Translator application to RunPod using Docker.

## Prerequisites

1.  **Docker Desktop** installed on your local machine.
2.  A **Docker Hub** account (to host your image).
3.  A **RunPod** account.

## Step 1: Build and Push the Docker Image

First, you need to build the Docker image locally and push it to a public registry like Docker Hub so RunPod can access it.

1.  **Login to Docker Hub** (in your terminal):
    ```bash
    docker login
    ```

2.  **Build the image** (replace `yourusername` with your Docker Hub username):
    ```bash
    docker build -t hugobart/mangatranslator:latest .
    ```
    *Note: This might take a few minutes as it installs PyTorch and other dependencies.*

3.  **Push the image**:
    ```bash
    docker push hugobart/mangatranslator:latest
    ```

## Step 2: Deploy on RunPod

1.  **Go to RunPod Console**: Log in to [runpod.io](https://www.runpod.io/console/pods).
2.  **Deploy a Pod**:
    *   Click **"Deploy"** on a GPU instance (e.g., RTX 3090 or RTX 4090 are good choices for OCR).
    *   Click **"Customize Deployment"** (or "Edit Template").
3.  **Configure the Container**:
    *   **Container Image**: Enter your image name: `hugobart/mangatranslator:latest`
    *   **Container Disk**: Increase if needed (default 10GB is usually fine, but 20GB is safer).
    *   **Volume Disk**: Optional, for persistent storage.
    *   **Expose Ports**:
        *   Add Port: `8501` (This is the Streamlit port).
    *   **Environment Variables** (Optional):
        *   You can add API keys here if you don't want to enter them in the UI every time (e.g., `OPENAI_API_KEY`).
4.  **Start the Pod**: Click **"Continue"** and then **"Deploy"**.

## Step 3: Access the App

1.  Once the Pod is **Running**, click on the **"Connect"** button.
2.  Look for the **"HTTP Service"** or **"TCP Port Mappings"**.
3.  If RunPod provides a direct HTTP link for port 8501, click it.
4.  If not, you might need to use the public IP and the mapped port (e.g., `http://123.45.67.89:12345` -> maps to container 8501).

## Troubleshooting

*   **GPU Usage**: The Dockerfile uses a standard Python image. PyTorch usually installs the CUDA version by default on Linux. To verify GPU access, check the logs in the Streamlit app or RunPod console.
*   **Storage**: If you process many files, ensure you have enough disk space.
