# 🚗 Car Price Prediction Web App

This project is a **Machine Learning-powered web application** that predicts the resale price of a used car based on various input parameters such as present price, kilometers driven, fuel type, number of owners, seller type, and transmission type. It uses a **Random Forest Regression model** trained on a dataset of car listings.

---

## 🧠 Project Workflow

1. **Data Preprocessing** (`model.py`)
   - Reads `carData.csv`, performs feature engineering and one-hot encoding.
   - Trains a `RandomForestRegressor` using `RandomizedSearchCV` for hyperparameter tuning.
   - Outputs performance metrics: **RMSE** and **R² score**.

2. **Model Deployment** (`app.py`)
   - Flask-based web application.
   - Loads the trained model using `pickle`.
   - Provides a user interface (`prediction.html`) to enter inputs.
   - Displays the predicted car resale price or a message if the car can't be sold.

---

## 🗂 Project Structure

```bash
├── app.py                   # Flask app and route logic
├── model.py                 # ML model training and evaluation
├── carData.csv              # Dataset used for training
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── templates/
│   └── prediction.html      # Web form UI (not uploaded but expected)
├── static/                  # (Optional) For any styling/images
└── rfr_model.pkl            # Trained model (not uploaded but required)
```

---

## ▶️ How to Run This Project

1. **Clone the repository and navigate to the folder:**
   ```bash
   git clone <your-repo-url>
   cd Car-Price-Prediction
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv env
   # Windows
   env\Scripts\activate
   # Linux/macOS
   source env/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model (Optional if `rfr_model.pkl` is already present):**
   ```bash
   python model.py
   ```

5. **Run the web server:**
   ```bash
   python app.py
   ```

6. **Open the application in your browser:**
   Visit the `localhost` link provided in the terminal.

---

## 📊 Sample Input Parameters

- **Year:** Year the car was purchased
- **Present Price:** Current market price in lakhs
- **Kms Driven:** Distance driven in kilometers
- **Owner Count:** Number of previous owners
- **Fuel Type:** Petrol/Diesel
- **Seller Type:** Dealer/Individual
- **Transmission:** Manual/Automatic

---

## 📌 Notes

- Ensure the `rfr_model.pkl` file is in the same directory as `app.py` for prediction to work.
- The web UI (`prediction.html`) should be placed inside the `templates/` directory.
