# karnataka-electricity-consumption-predictor

A machine learning-powered web application that predicts **average monthly household electricity consumption** for districts in **Karnataka, India**, based on real government data and user inputs.

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue.svg)](https://www.python.org/)
[![Powered by Flask](https://img.shields.io/badge/Backend-Flask-orange.svg)](https://flask.palletsprojects.com/)
[![Deployed on Render](https://img.shields.io/badge/Live%20App-Render-green.svg)](https://electricityconsumptionpredictor.onrender.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)



## About the Project

Karnataka's electricity consumption has evolved over the years due to urbanization, lifestyle changes, and home expansions. This app predicts future household electricity needs using key variables like:

- **Future Year** (up to 2035)
- **Household Size**
- **District**
- **Area of the house (in sq. ft)**

### Goals:
- Help households and consultants **plan energy usage**
- Provide **data-driven insights** to policymakers
- Raise awareness about **growing power demands**



## Tech Stack

| Layer       | Tech                     |
|-------------|--------------------------|
| ML Model    | `scikit-learn`, `pandas`, `numpy` |
| Backend     | `Flask`, `Python`        |
| Frontend    | `HTML`, `CSS`, `JavaScript` |
| Deployment  | `Render` (Cloud Hosting) |



## ML Model Details

The model was trained on historical data from 1990 onwards using:
- Feature scaling for numeric inputs
- One-hot encoding for categorical districts
- Linear regression / Random Forest for consumption prediction
- Saved using `joblib` for real-time prediction in Flask



## Project Structure

```bash
karnataka-electricity-consumption-predictor/
├── static/               # Static assets (images, CSS)
├── templates/            # HTML templates for web pages
│   ├── index.html
│   ├── model.html
│   ├── result.html
│   └── ...
├── app.py                # Flask server logic
├── model.py              # ML model pipeline
├── requirements.txt      # Python dependencies
├── Procfile              # Render deployment file
├── README.md             # Project overview
└── LICENSE               # MIT License
```
## How to Run Locally

1. **Clone the repo**

```bash
git clone https://github.com/yourusername/karnataka-electricity-consumption-predictor.git
cd karnataka-electricity-consumption-predictor
```

2. **Create and activate virtual environment**

```bash

python -m venv venv

# Activate:
source venv/bin/activate      # On Mac/Linux
venv\Scripts\activate         # On Windows
```
3. **Install dependencies**

```bash

pip install -r requirements.txt
```
4. **Run the app**

```bash

python app.py

```
5. **Visit in browser**

```text
http://127.0.0.1:5000/
```

## 🌐 Live Demo

Try it here:  
**[electricityconsumptionpredictor.onrender.com](https://electricityconsumptionpredictor.onrender.com)**  
No sign-in needed — just input your details and get your monthly prediction!



## Future Improvements

- Data visualization for predicted trends (graphs, heatmaps)
- Export to CSV or PDF for reports
- Personalized energy-saving recommendations
  


## License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and build upon it.



## ✨ Acknowledgments

- Government of Karnataka energy data sources  
- scikit-learn & Flask open-source community  
- [Render.com](https://render.com/) for free deployment support


## Author

**Disha S Basti**
