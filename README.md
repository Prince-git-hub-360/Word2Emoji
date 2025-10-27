# Mood2Emoji
Kid-safe Text Mood Detector built using Streamlit, TextBlob, and Transformers.

# Mood2Emoji — Kid-Safe Text Mood Detector

### 👨‍💻 Author
**Prince Kumar**  
Email: [kumariafprince@gmail.com](mailto:kumariafprince@gmail.com)

---

## 🧠 Project Overview
**Mood2Emoji** is a fun, interactive **AI-powered web app** made for students aged **12–16**.  
It helps kids learn how computers can detect emotions from text safely and ethically.

The app reads a sentence, analyzes the **tone and emotion**, and shows a **kid-friendly emoji** 😀 😐 😞 with a short explanation like “Sounds happy!” or “Seems sad.”  
It also includes a **Teacher Mode** that explains how the app works in simple terms.

---

## 🎯 Features
- 📝 Input box to type any short sentence  
- 😀 😐 😞 Emoji output with emotion explanation  
- 🚫 Filters inappropriate or harmful words  
- 🧩 “Teacher Mode” to explain logic visually  
- ⚙️ Neutral fallback for gibberish or unclear text  

---

## 🧩 Tech Stack
- **Python 3.9+**  
- **Streamlit** — interactive web app  
- **TextBlob** — sentiment analysis (polarity scoring)  
- **Rule-based filters** — ensure kid-safe output  

---

## 📂 Project Structure
repo/
├── app.py # Streamlit app script
├── requirements.txt # Required dependencies
├── README.md # Project documentation
└── lesson_plan.pdf # Lesson plan for ages 12–16




---
## ⚙️ Setup & Run Instructions

Follow these steps to set up and run the project locally or deploy it on Streamlit Cloud.

---

### 🧩 1️⃣ Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone https://github.com/Prince-git-hub-360/Word2Emoji.git
cd Word2Emoji

pip install -r requirements.txt

streamlit run app.py

http://localhost:8501


