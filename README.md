# Karnataka Household Electricity Consumption Predictor 🔦🏠

This project is a **Machine Learning-powered web application** that predicts the **average monthly household electricity consumption** for districts in Karnataka, India. The predictions are based on government data collected since 1990, considering factors like:

- **Future Year** (up to 2035)
- **Household Size**
- **District**
- **Area of the house (in sq. ft)**

---

## 📊 About the Project

Karnataka's residential electricity consumption patterns have been steadily changing over the years. This ML model helps estimate future consumption trends, enabling:
- Better resource planning for households
- Insights for energy consultants and government agencies
- Awareness about energy demands based on demographics and home size

---

## 💻 Tech Stack

- **Python**
- **Flask (Backend API)**
- **HTML, CSS, JavaScript (Frontend)**
- **scikit-learn / Pandas / Numpy (ML & Data Handling)**
- **Render (Deployment)**

---

## 📁 Project Structure

```
📦karnataka-electricity-consumption-predictor
 ├📂__pycache__              # Python cache files (auto-generated)
 ├📂static
 ┃ ┗📂images                 # Images and media used in frontend
 ├📂templates                # HTML templates for the Flask frontend
   ├📄cover.html
   ├📄data.html
   ├📄home.html
   ├📄main.html
   ├📄index.html
   ├📄model.html
   ├📄result.html
 ├📄.gitignore               # Git ignored files
 ├📄Procfile                 # Render deployment config (gunicorn app:app)
 ├📄app.py                   # Main Flask web server script
 ├📄model.py                 # Trained ML model for consumption prediction
 ├📄requirements.txt         # Python dependencies list
 └📄README.md                # Project documentation
```

---

## 🚀 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/karnataka-electricity-consumption-predictor.git
   cd karnataka-electricity-consumption-predictor
   ```

2. **Create virtual environment & activate**
   ```bash
   python -m venv venv
   source venv/bin/activate  # on Mac/Linux
   venv\Scripts\activate     # on Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
   ```
   http://127.0.0.1:5000/
   ```

---

## 🌐 Live Deployment  
https://electricityconsumptionpredictor.onrender.com

---

## 📢 Future Improvements

- Add data visualization for predicted trends
- Enable CSV report download for the predictions
- Integrate energy-saving recommendations

---

## 📝 Author

**Shreya Pandit**  
*AI & ML Developer | Full-Stack Enthusiast*

---

## 📌 Notes

This project uses real **Karnataka Government household electricity consumption data** from **1990 onwards** for training. The model is capable of projecting future consumption trends up to **2035**.

---

