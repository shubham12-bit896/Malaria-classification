
```md
# 🦠 Malaria Cell Classification using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black)
![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-purple)

A deep learning–based web application that classifies microscopic blood smear images as **Infected (Parasitized)** or **Uninfected** using a trained CNN model.  
It includes a modern **Flask Web Interface**, allowing users to upload blood cell images and receive predictions instantly.

---

## 🚀 Features

- 🧬 Detects **Malaria-infected vs Healthy cells**
- 🧠 Built using TensorFlow/Keras CNN architecture
- 🖼 Real-time image upload and prediction
- 📊 Based on NIH Microscopic Image Dataset
- 💻 User-friendly web application using Flask

---

## 📥 Download Trained Model

GitHub does not allow large binary files, so the trained model is stored externally.

👉 **Download Model Weights (.h5):**  
🔗 https://drive.google.com/file/d/1HUdTj4PLBDuKOpPBNAhDDF_Mq49UgtPc/view?usp=drive_link

After downloading, create a folder named `model` and place the file inside:

```

model/malaria_model_fixed.h5

````

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository

```sh
git clone https://github.com/shubham12-bit896/Malaria-classification.git
cd Malaria-classification
````

### 2️⃣ Install Dependencies

```sh
pip install -r requirements.txt
```

### 3️⃣ Add Model File

Place the downloaded `.h5` model in:

```
model/malaria_model_fixed.h5
```

### 4️⃣ Run the Application

```sh
python app.py
```

Then open in browser:

```
http://127.0.0.1:5000/
```

Upload a microscopy image → Model predicts infection status.

---

## 📊 Dataset Used

* NIH Malaria Dataset (27,558 cell images)
* Two classes:

  * 🔴 Parasitized
  * 🟢 Uninfected

---

## 📂 Project Structure

```
├── model/
│   └── malaria_model_fixed.h5  (download manually)
├── static/
│   ├── uploads/
│   └── samples/
├── templates/
│   └── index.html
├── app.py
├── requirements.txt
└── README.md
```

---

## 💡 Future Improvements

* Deploy using Render / HuggingFace / Streamlit
* Add Grad-CAM heatmap for explainability
* Improve accuracy using VGG16 / MobileNet
* Add REST API or mobile app support

---

## 🤝 Contributing

Contributions and suggestions are welcome.
Feel free to open an issue or create a pull request.

---

## 👤 Author

**Shubham Katore**
GitHub: `shubham12-bit896`

---

⭐ If you found this project useful, please **Star this repository!**


