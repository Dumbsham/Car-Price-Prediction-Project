# 🚗 Ford Car Price Prediction

A machine learning web application that predicts Ford car prices based on various features including model, year, transmission, mileage, fuel type, and more. Built with Streamlit for an interactive user experience.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.31.0-red.svg)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Overview

This project uses machine learning to predict the selling price of used Ford cars. The model is trained on historical Ford car data and provides instant price predictions through an interactive web interface.

## ✨ Features

- **🎯 Accurate Predictions**: Machine learning model trained on real Ford car data
- **🖥️ Interactive Web Interface**: User-friendly Streamlit application
- **📊 Multiple Insights**: Price estimates, depreciation analysis, and value metrics
- **💾 Export Results**: Download predictions as CSV reports
- **⚡ Real-time Processing**: Instant predictions with optimized model loading

## 🛠️ Tech Stack

- **Python 3.8+**: Core programming language
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning library
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Pickle**: Model serialization

## 📁 Project Structure

```
CAR_PRICE_PREDICTION/
│
├── car_app.py                    # Enhanced Streamlit web application ⭐
├── car_app_simple.py             # Simple Streamlit web application
├── Car_Price_Prediction.ipynb    # Jupyter notebook with model training
├── car_model.pkl                 # Trained machine learning model
├── car_scaler.pkl               # Feature scaler (StandardScaler)
├── label_encoders.pkl           # Label encoders for categorical features
├── ford.csv                     # Ford car dataset
├── requirements.txt             # Project dependencies
├── .gitignore                   # Git ignore file
|__ README.md                    # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ford-car-price-prediction.git
   cd ford-car-price-prediction
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

**Option 1: Enhanced Version (Recommended)**
```bash
streamlit run car_app.py
```


```

The app will automatically open in your browser at `http://localhost:8501`

## 💻 Usage

### Making a Prediction

1. **Select Car Model**: Choose from available Ford models
2. **Enter Year**: Year of manufacture (1990-2025)
3. **Choose Transmission**: Manual, Automatic, or Semi-Auto
4. **Input Mileage**: Total miles driven
5. **Select Fuel Type**: Petrol, Diesel, Hybrid, or Electric
6. **Enter Road Tax**: Annual road tax in £
7. **Input MPG**: Fuel efficiency in miles per gallon
8. **Enter Engine Size**: Engine displacement in liters
9. **Click Predict**: Get instant price estimate

### Example Prediction

```
Car Model: Fiesta
Year: 2018
Transmission: Manual
Mileage: 15,000 miles
Fuel Type: Petrol
Tax: £150
MPG: 55.0
Engine Size: 1.5L

Predicted Price: £12,450.00
```

## 📊 Model Information

### Features Used

The model uses 8 key features for prediction:

| Feature | Type | Description |
|---------|------|-------------|
| Model | Categorical | Ford car model (e.g., Fiesta, Focus, Kuga) |
| Year | Numerical | Year of manufacture |
| Transmission | Categorical | Type of transmission |
| Mileage | Numerical | Total miles driven |
| Fuel Type | Categorical | Type of fuel used |
| Tax | Numerical | Annual road tax (£) |
| MPG | Numerical | Fuel efficiency |
| Engine Size | Numerical | Engine displacement (L) |

### Model Pipeline

1. **Label Encoding**: Categorical features → numerical values
2. **Scaling**: StandardScaler for feature normalization
3. **Prediction**: Trained regression model
4. **Output**: Predicted price in GBP (£)

## 📈 Model Training

The model was trained using the Jupyter notebook `Car_Price_Prediction.ipynb`. The notebook includes:

- **Data Exploration**: Analysis of Ford car features
- **Data Preprocessing**: Cleaning and handling missing values
- **Feature Engineering**: Creating and selecting relevant features
- **Model Training**: Training regression model
- **Model Evaluation**: Performance metrics and validation
- **Model Serialization**: Saving models as .pkl files

### Retraining the Model

To retrain the model with new data:

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook Car_Price_Prediction.ipynb
   ```

2. Update the dataset (`ford.csv`) with new data
3. Run all cells to retrain the model
4. New `.pkl` files will be generated automatically

## 🎨 App Versions

This project includes two versions of the web application:

### Enhanced Version (car_app.py)
- Professional UI with custom CSS
- Sidebar with information and tips
- Additional metrics and insights
- CSV download functionality
- Form-based input with validation

### Simple Version (car_app_simple.py)
- Clean, minimalist interface
- Faster loading time
- Essential features only
- Perfect for quick testing

See `APP_VERSIONS.md` for detailed comparison and customization guide.

## 📦 Model Files

| File | Purpose |
|------|---------|
| `car_model.pkl` | Trained regression model for price prediction |
| `car_scaler.pkl` | StandardScaler for feature normalization |
| `label_encoders.pkl` | Dictionary of LabelEncoders for categorical variables |

## 🔧 Customization

### Changing the Color Scheme

Edit the CSS in `car_app.py`:

```python
# Find this section
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

# Replace with your colors
background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
```

### Adding New Metrics

Add custom metrics in the prediction results section:

```python
with col4:
    monthly_payment = prediction / 36  # 3-year loan
    st.metric(
        "Est. Monthly Payment",
        f"£{monthly_payment:,.0f}",
        help="Based on 3-year loan"
    )
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

Please read `CONTRIBUTING.md` for detailed contribution guidelines.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Troubleshooting

### Common Issues

**Issue**: Model files not found
```
Solution: Ensure car_model.pkl, car_scaler.pkl, and label_encoders.pkl are in the same directory as car_app.py
```

**Issue**: Import errors
```
Solution: Install all requirements using: pip install -r requirements.txt
```

**Issue**: Streamlit not found
```
Solution: Activate your virtual environment and reinstall: pip install streamlit
```

**Issue**: Prediction errors
```
Solution: Check that all input values are within valid ranges
```

## 📧 Contact & Support

- **Author**: Saksham Sharma
- **Email**: sakshamnoida37@Gmail.com
- **GitHub**: [Saksham Sharma](https://github.com/Dumbsham)
- **LinkedIn**: [Saksham Sharma](https://linkedin.com/in/saksham14sharma)

## 🙏 Acknowledgments

- Dataset: Ford car sales data
- Inspired by various car price prediction projects
- Built with ❤️ using Python, Streamlit, and scikit-learn
- Special thanks to the open-source & ML community

## 📊 Project Status

- ✅ Model Training Complete
- ✅ Web Application Deployed
- ✅ Documentation Complete

## 🔮 Future Enhancements

- [ ] Add support for more car manufacturers
- [ ] Implement model comparison (Random Forest, XGBoost, etc.)
- [ ] Add data visualization dashboard
- [ ] Deploy to cloud platform (Heroku, Streamlit Cloud)
- [ ] Add user authentication
- [ ] Implement feedback mechanism for prediction accuracy

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Scikit-learn Documentation](https://scikit-learn.org)
- [Pandas Documentation](https://pandas.pydata.org)

---

**⭐ If you find this project helpful, please give it a star!**

---

<div align="center">
Made with ❤️ by Saksham
</div>
