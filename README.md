# 🚀 Network Intrusion Detection System (NIDS) - Machine Learning Project

## 📌 Overview
This project implements a **Random Forest-based Network Intrusion Detection System (NIDS)** that detects cyber attacks using network traffic data. The model is trained on a dataset of various network attacks and deployed via **FastAPI**.

## 📂 Project Structure
```
📂 nids-ml-project
│── 📂 data                   # Raw dataset files (not included in repo)
│── 📂 models                 # Trained model & preprocessing tools
│── 📂 src                    # Training, preprocessing & prediction scripts
│   ├── preprocess.py         # Data preprocessing script
│   ├── train.py              # Model training script
│   ├── predict.py            # Inference script
│   ├── app.py                # FastAPI deployment
│── 📂 notebooks              # Jupyter notebooks for exploratory analysis
│   ├── exploratory_analysis.ipynb
│── README.md                 # Project documentation
│── requirements.txt          # Dependencies
│── .gitignore                # Ignore unnecessary files
```

## 💾 Dataset Information
The dataset consists of various network traffic records, categorized into **benign traffic** and multiple **attack types**. The features include:
- Packet transmission rates
- Protocol types
- Flow durations
- IP addresses and ports
- Attack labels (Benign, DoS, DDoS, etc.)

## 🛠️ Installation & Setup
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/YOUR_GITHUB_USERNAME/nids-ml-project.git
cd nids-ml-project
```
### **2️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

## 🎯 Running the Project
### **1️⃣ Preprocess the Data**
```sh
python src/preprocess.py
```
✅ **This script loads, cleans, encodes, and normalizes the dataset.**

### **2️⃣ Train the Model**
```sh
python src/train.py
```
✅ **Trains the model and saves it in `models/`.**

### **3️⃣ Run Predictions**
```sh
python src/predict.py
```
✅ **Loads the trained model to classify new samples.**

### **4️⃣ Run FastAPI for Real-Time Predictions**
```sh
uvicorn src.app:app --reload
```
✅ **Access the API at** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## ⚡ Future Improvements
- 🔥 **Improve model performance** with **XGBoost**.
- 🌐 **Deploy API on AWS/GCP**.
- 📊 **Add a web dashboard** for real-time monitoring.

## 👨‍💻 Contributions
Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to improve.

---
📢 **If you like this project, give it a ⭐ on GitHub!** 🚀🔥



