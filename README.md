# 🏠 Mumbai House Price Prediction

A comprehensive machine learning project for predicting residential property prices in Mumbai using Ridge Regression with an interactive Streamlit web application.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [Visualizations](#visualizations)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)

## 🎯 Overview

This project predicts house prices in Mumbai using machine learning algorithms and provides an interactive interface for real-time price estimation. The system analyzes various property features such as location, size, furnishing status, and amenities to deliver accurate price predictions.

**Key Highlights:**
- 🎯 **High Accuracy**: R² score > 0.70 with robust Ridge Regression
- 🖥️ **Interactive UI**: Real-time predictions via Streamlit web app
- 📊 **Comprehensive Analysis**: Detailed EDA with 17,000+ property listings
- 🏙️ **Mumbai-Specific**: Covers 20+ prime locations across Mumbai
- 📈 **Business Intelligence**: Power BI dashboards for insights

## ✨ Features

- **Real-Time Price Prediction**: Get instant property valuations based on key features
- **User-Friendly Interface**: Interactive Streamlit application with intuitive controls
- **Location Intelligence**: Covers major Mumbai neighborhoods with location-specific pricing
- **Furnishing Options**: Supports Furnished, Semi-Furnished, and Unfurnished properties
- **Robust Data Processing**: Advanced outlier removal and data cleaning pipeline
- **Visual Analytics**: Comprehensive charts showing price trends and correlations
- **Model Persistence**: Trained model saved for quick deployment
- **Flexible Input**: Accepts various property configurations (1-10 BHK, multiple parking spots)

## 📊 Dataset

### Overview
- **Source**: Self-compiled dataset from magicbricks.com
- **Size**: ~17,000 house listings in Mumbai
- **Format**: CSV files with structured property data
- **Coverage**: Multiple neighborhoods across Mumbai

### Features

| Feature | Description | Type |
|---------|-------------|------|
| **Location** | Neighborhood/area in Mumbai | Categorical (21 locations) |
| **Size (BHK)** | Number of bedrooms, hall, kitchen | Numeric (1-10) |
| **Carpet Area** | Property area in square feet | Numeric (100-10,000 sqft) |
| **Furnishing** | Furnished/Semi-Furnished/Unfurnished | Categorical |
| **Bathrooms** | Number of bathrooms | Numeric (1-10) |
| **Parking** | Available parking spots | Numeric (0-5) |
| **Price** | Property price (Target variable) | Numeric (INR) |

### Covered Locations
- **Western Suburbs**: Andheri East, Andheri West, Borivali, Dahisar, Goregaon East, Goregaon West, Kandivali West, Kandivali East, Malad, Juhu, Khar
- **Central Mumbai**: Dadar, Wadala
- **Eastern Suburbs**: Bhandup, Chembur, Ghatkopar, Kurla, Vikhroli, Powai
- **Airport Area**: Santacruz East, Santacruz West

## 🎬 Demo

### Web Application Interface
The Streamlit application provides an intuitive interface for price prediction:

1. **Location Selection**: Dropdown with 21 Mumbai locations
2. **Property Details**: Input fields for BHK, sqft, bathrooms, parking
3. **Furnishing Status**: Radio buttons for furnishing options
4. **Instant Prediction**: Real-time price estimation with visual feedback

### Sample Predictions
```
Property: 2 BHK in Andheri East, 1000 sqft, Furnished, 2 Bath, 1 Parking
Predicted Price: ₹95,00,000 - ₹1,05,00,000

Property: 3 BHK in Powai, 1500 sqft, Semi-Furnished, 2 Bath, 2 Parking
Predicted Price: ₹1,40,00,000 - ₹1,60,00,000
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning)

### Step-by-Step Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/supritR21/House-Prediction-ML-Simple-.git
cd House-Prediction-ML-Simple-
```

#### 2. Create Virtual Environment (Recommended)
```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
```
scikit-learn==1.3.0      # Machine learning algorithms
pandas>=1.5.0            # Data manipulation
numpy>=1.24.0            # Numerical operations
matplotlib>=3.7.0        # Plotting and visualization
seaborn>=0.12.0          # Statistical visualizations
joblib>=1.2.0            # Model serialization
streamlit>=1.25.0        # Web application framework
```

## 💻 Usage

### 1. Train the Model

First, train the model with the provided dataset:

```bash
python train_model_fixed.py
```

**What this does:**
- Loads raw data from `mumbaiproject.csv`
- Performs data cleaning and preprocessing
- Removes outliers using IQR method
- Trains Ridge Regression model
- Evaluates performance metrics
- Saves model as `work_new.joblib`
- Generates `finaldata_processed.csv`

**Expected Output:**
```
Step 1: Loading and preprocessing data...
Step 2: Data validation and cleaning...
Step 3: Removing outliers...
Step 4: Data statistics after preprocessing:
Step 5: Training the model...
Step 6: Model Evaluation
  R² Score: 0.7xxx
  MAPE: 0.xxxx
Step 7: Saving the best model...
✓ Model saved as 'work_new.joblib'
```

### 2. Run Data Analysis (Optional)

Generate visualizations and perform exploratory analysis:

```bash
python analysis.py
```

**Generated Outputs:**
- Correlation heatmaps
- Price distribution plots
- Location-wise analysis
- Feature relationship visualizations

### 3. Launch Web Application

Start the interactive Streamlit interface:

```bash
streamlit run frontend.py
```

The app will open automatically at `http://localhost:8501`

### 4. Using the Web App

1. **Select Location**: Choose from 21 Mumbai neighborhoods
2. **Choose Furnishing**: Furnished/Semi-Furnished/Unfurnished
3. **Enter Property Details**:
   - Carpet area (sqft)
   - Number of BHK
   - Number of bathrooms
   - Parking spots
4. **Get Prediction**: Click "Predict Price" button
5. **View Results**: See estimated price with property summary

## 🤖 Model Details

### Algorithm: Ridge Regression

Ridge Regression was chosen for its robustness and ability to handle multicollinearity while preventing extreme predictions.

**Why Ridge over Linear Regression?**
- Regularization prevents overfitting
- Handles correlated features better
- Produces more stable predictions
- No negative price predictions

### Machine Learning Pipeline

```python
Pipeline:
  1. ColumnTransformer
     ├── OneHotEncoder (location, furnishing)
     └── Passthrough (sqft, size, bath, parking)
  2. StandardScaler (feature normalization)
  3. Ridge Regression (alpha=1.0)
```

### Data Preprocessing Steps

#### 1. Data Cleaning
- Extract BHK size from location strings
- Convert price formats (Cr/Lac → absolute values)
- Parse and normalize square footage
- Handle missing values with statistical imputation
- Remove invalid entries

#### 2. Feature Engineering
- Location extraction and normalization
- Parking and bathroom count conversion
- Price standardization to INR
- Categorical encoding preparation

#### 3. Outlier Removal
- IQR (Interquartile Range) method
- 5th and 95th percentile bounds
- Applied to: price, sqft, size, bath, parking
- Preserves data distribution integrity

#### 4. Encoding & Scaling
- OneHotEncoding for categorical features
- StandardScaler for numerical features
- Unified preprocessing pipeline

### Model Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **R² Score** | 0.70+ | Explains 70%+ variance in prices |
| **MAPE** | <0.15 | Mean absolute percentage error |
| **Train-Test Split** | 80/20 | Stratified random sampling |
| **Cross-validation** | ✓ | Validated for consistency |
| **Negative Predictions** | 0 | No unrealistic outputs |

### Model Validation

```python
# Sample test predictions validated
Location: Andheri East, 2 BHK, 1000 sqft, Furnished
Predicted: ₹98,50,000 ✓

Location: Borivali, 1 BHK, 800 sqft, Semi-Furnished
Predicted: ₹65,20,000 ✓

Location: Kandivali West, 3 BHK, 1200 sqft, Unfurnished
Predicted: ₹1,15,75,000 ✓
```

## 📁 Project Structure

```
House-Price-Prediction/
│
├── 📊 Data Files
│   ├── mumbaiproject.csv           # Original raw dataset (17K+ listings)
│   ├── finaldata.csv               # Intermediate processed data
│   └── finaldata_processed.csv     # Final cleaned dataset
│
├── 🤖 Model Files
│   ├── work_new.joblib            # Trained Ridge Regression model
│   └── work.joblib                # Backup/previous model
│
├── 🐍 Python Scripts
│   ├── train_model_fixed.py       # Model training & preprocessing
│   ├── analysis.py                # EDA and visualizations
│   └── frontend.py                # Streamlit web application
│
├── 📋 Configuration
│   ├── requirements.txt           # Python dependencies
│   └── README.md                  # Project documentation
│
└── 📊 Visualizations
    └── Plots/                     # Generated charts & Power BI dashboards
        ├── img.png               # Correlation heatmap
        ├── img_1.png             # Location-wise pricing
        ├── img_3.png             # Multi-dimensional analysis
        ├── 11.jpg                # Power BI Dashboard - Page 1
        └── 12.jpg                # Power BI Dashboard - Page 2
```

## 📊 Visualizations

### 1. Feature Correlation Analysis
![Correlation Matrix](Plots/img.png)

**Insights:**
- Shows relationships between numerical features
- Identifies multicollinearity issues
- Guides feature selection and engineering
- Highlights strongest price predictors

### 2. Location-Based Price Analysis
![Average Prices by Location](Plots/img_1.png)

**Key Findings:**
- Premium locations: Juhu, Khar, Powai
- Affordable areas: Dahisar, Bhandup
- Central Mumbai commands premium pricing
- Western suburbs show varied pricing

### 3. Multi-Dimensional Analysis
![Price vs Location vs Size](Plots/img_3.png)

**Observations:**
- Size impact varies by location
- Non-linear relationship between features
- Location premium independent of size
- Interaction effects captured

### 4. Power BI Dashboards

#### Dashboard Page 1
![Power BI Dashboard - Overview](Plots/11.jpg)

**Features:**
- Overall market statistics
- Price distribution charts
- Location-wise comparisons
- Interactive filters

#### Dashboard Page 2
![Power BI Dashboard - Trends](Plots/12.jpg)

**Features:**
- Temporal trends (if available)
- Detailed breakdowns by property type
- Comparative analytics
- Market insights

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.8+**: Primary programming language
- **Streamlit**: Interactive web application framework
- **scikit-learn**: Machine learning models and preprocessing

### Data Science Stack
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations and arrays
- **matplotlib**: Data visualization and plotting
- **seaborn**: Statistical data visualization

### Machine Learning
- **Ridge Regression**: Primary prediction model
- **OneHotEncoder**: Categorical feature encoding
- **StandardScaler**: Feature normalization
- **ColumnTransformer**: Preprocessing pipeline

### Additional Tools
- **joblib**: Model serialization and persistence
- **Power BI**: Business intelligence dashboards
- **Git**: Version control
- **VS Code**: Development environment

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute
1. 🐛 **Report Bugs**: Open an issue with bug details
2. 💡 **Suggest Features**: Propose new features or improvements
3. 📝 **Improve Documentation**: Enhance README or code comments
4. 🔧 **Submit Pull Requests**: Fix bugs or add features

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/your-username/House-Prediction-ML-Simple-.git
cd House-Prediction-ML-Simple-

# Create a new branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "Add your descriptive commit message"

# Push and create PR
git push origin feature/your-feature-name
```

### Contribution Ideas
- 🌍 Add support for other cities
- 🤖 Implement additional ML algorithms (XGBoost, Random Forest)
- 📱 Create mobile-responsive design
- 🧪 Add unit tests and CI/CD
- 📊 Enhance visualizations with interactive plots
- 🌐 Add API endpoints for predictions
- 🔍 Implement feature importance analysis

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

```
MIT License

Copyright (c) 2025 supritR21

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## 👥 Authors

- **Suprit Raj** - [@supritR21](https://github.com/supritR21)

## 🙏 Acknowledgments

- **Data Source**: magicbricks.com for Mumbai property listings
- **Inspiration**: Real-world property valuation challenges in Indian real estate market
- **Community**: Open-source contributors and scikit-learn community
- **Tools**: Streamlit for rapid web app development

## 📧 Contact & Support

- **GitHub**: [@supritR21](https://github.com/supritR21)
- **Issues**: [Report issues](https://github.com/supritR21/House-Prediction-ML-Simple-/issues)
- **Discussions**: [Join discussions](https://github.com/supritR21/House-Prediction-ML-Simple-/discussions)

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Ridge Regression Explained](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)
- [pandas Documentation](https://pandas.pydata.org/docs/)

## ⚠️ Disclaimer

This model is for **educational and informational purposes only**. Actual property prices may vary significantly based on:
- Current market conditions
- Property condition and age
- Exact location and proximity to amenities
- Legal and documentation status
- Negotiation factors
- Market timing and demand

**Always consult with real estate professionals for accurate property valuations.**

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

**Made with ❤️ for the Mumbai Real Estate Community**

[Report Bug](https://github.com/supritR21/House-Prediction-ML-Simple-/issues) · [Request Feature](https://github.com/supritR21/House-Prediction-ML-Simple-/issues) · [Documentation](https://github.com/supritR21/House-Prediction-ML-Simple-/wiki)

</div>
