
# 🦠 Malaria Cell Classification using Deep Learning  

A deep-learning powered web application that classifies microscopic blood smear images as **Parasitized (Infected)** or **Uninfected** using a trained Convolutional Neural Network (CNN).  
The project includes an easy-to-use **Flask web interface** for real-time predictions.

---

### 📌 Tech Stack

| Category | Tools |
|---------|-------|
| Language | Python |
| Framework | Flask |
| Deep Learning | TensorFlow / Keras |
| Frontend | HTML, CSS |
| Deployment | Localhost (Future: Cloud) |

---

### 🔥 Features

✔ Real-time malaria cell detection  
✔ Upload-based prediction system  
✔ Trained on NIH malaria cell dataset  
✔ Clean and interactive UI  
✔ Lightweight + reproducible setup  

---

### 📥 Model Download (Required)

Due to file size limitations, the trained `.h5` model is hosted externally.

👉 Download model file:  
🔗 **https://drive.google.com/file/d/1HUdTj4PLBDuKOpPBNAhDDF_Mq49UgtPc/view?usp=drive_link**

Place it inside:

```

model/malaria_model_fixed.h5

````

(If `model/` folder does not exist, create it.)

---

### ⚙️ Installation & Running the App

```sh
# Clone the project
git clone https://github.com/shubham12-bit896/Malaria-classification.git
cd Malaria-classification

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
````

Now open your browser and go to:
👉 `http://127.0.0.1:5000/`

Upload an image → Get prediction 🎯

---

### 📂 Project Structure

```
│── app.py
│── requirements.txt
│── README.md
│
├── model/
│   └── malaria_model_fixed.h5   (download manually)
│
├── static/
│   ├── uploads/
│   └── samples/
│
└── templates/
    └── index.html
```

---

### 📊 Dataset

Dataset used: **NIH Malaria Cell Dataset**

* 27,558 total microscopic images
* Two categories:

  * 🦠 Parasitized (Infected)
  * 🧪 Uninfected

---

### 🚀 Future Enhancements

🔹 Deploy app using **Render / HuggingFace Spaces / Streamlit**
🔹 Add **Grad-CAM explainability visualization**
🔹 Improve model accuracy using **VGG16 / ResNet / MobileNet**
🔹 Add API support for integration with clinical software

---

### 🧑‍💻 Author

**Shubham Katore**
📍 Health Informatics & AI Projects
🔗 GitHub: `shubham12-bit896`

---

⭐ If you found this project helpful, please consider **starring the repo** — it motivates further improvements!


