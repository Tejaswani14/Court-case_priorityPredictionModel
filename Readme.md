# ⚖️ Court Case Priority Prediction System

An AI-powered web application that predicts priority levels for court cases using machine learning. Built for the IBM Z Hackathon with Streamlit and Python.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![IBM Z](https://img.shields.io/badge/IBM_Z-000000?style=for-the-badge&logo=ibm&logoColor=white)

## 🏆 IBM Z Hackathon Project

This project was developed for the **IBM Z Hackathon** - a system designed to optimize court case management using AI and machine learning.

## 📁 Project Structure

```
IBM-Z-HACKATHON-MAIN/
├── 📊 Data Files
│   ├── court_cases.csv              # Sample court cases dataset
│   └── case_predictions.csv         # Example predictions output
├── 🔧 Core Application
│   ├── predict.py                   # Main Streamlit web application
│   └── model.py                     # Model training and development
├── 🤖 Model Artifacts
│   ├── best_priority_model.pkl      # Trained ML model
│   ├── label_encoders.pkl           # Categorical variable encoders
│   ├── feature_names.pkl            # Feature names for model
│   ├── feature_scaler.pkl           # Feature scaling parameters
│   └── model_metadata.pkl           # Model performance metadata
├── 📚 Documentation
│   ├── Readme.md                    # This file
│   └── abstract.md                  # Project abstract
└── ⚙️ Configuration
    └── .gitignore                   # Git ignore rules
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation & Setup

1. **Clone and navigate to the project**
   ```bash
   git clone <repository-url>
   cd IBM-Z-HACKATHON-MAIN
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit pandas numpy scikit-learn joblib plotly
   ```

3. **Launch the application**
   ```bash
   streamlit run predict.py
   ```

4. **Access the dashboard**
   - Open your browser to `http://localhost:8501`
   - Upload `court_cases.csv` to get predictions

## 🎯 Features

### Core Functionality
- **AI-Powered Priority Prediction**: Classifies cases into 5 priority levels
- **Batch CSV Processing**: Upload and analyze multiple cases simultaneously
- **Interactive Visualizations**: Pie charts and histograms for result analysis
- **Export Capabilities**: Download predictions as CSV files
- **Real-time Processing**: Instant predictions with confidence scores

### Priority Classification
| Level | Priority | Description |
|-------|----------|-------------|
| 0 | Very Low | Routine cases with minimal urgency |
| 1 | Low | Standard priority cases |
| 2 | Medium | Cases requiring moderate attention |
| 3 | High | Important cases needing prompt action |
| 4 | Critical | Urgent cases requiring immediate resolution |

## 📊 Model Details

### Training & Development
- **Training Script**: `model.py` - Contains model development code
- **Model Type**: Classification (as per metadata)
- **Features**: Legal, temporal, and complexity metrics
- **Performance**: Optimized for judicial case prioritization

### Pre-trained Artifacts
All necessary model files are included:
- `best_priority_model.pkl` - Main prediction model
- `label_encoders.pkl` - Handles categorical encoding
- `feature_scaler.pkl` - Standardizes numerical features
- `feature_names.pkl` - Ensures feature consistency
- `model_metadata.pkl` - Contains model performance metrics

## 🛠️ Usage Guide

### For Prediction (End Users)
1. Run `streamlit run predict.py`
2. Upload `court_cases.csv` via the sidebar
3. Click "Predict Priorities"
4. View results and download predictions

### For Development (Data Scientists)
1. Use `model.py` for model retraining/development
2. Modify `predict.py` for application enhancements

### Input Data Format
Use `court_cases.csv` as a template. Expected columns include:
- Case identifiers: `case_id`, `cnr_number`, `fir_number`
- Dates: `filed_date`, `last_hearing_date`
- Metrics: `case_age_days`, `adjournments_count`, `evidence_complexity_score`
- Party information: `number_of_petitioners`, `number_of_respondents`

## 🔄 Workflow

1. **Data Preparation** → Prepare your case data in CSV format
2. **Upload** → Use the Streamlit interface to upload data
3. **Prediction** → AI model processes and classifies cases
4. **Visualization** → Interactive charts show priority distribution
5. **Export** → Download results for further analysis

## 📈 Results & Output

The system generates:
- **Priority predictions** for each case
- **Confidence scores** for model reliability
- **Visual analytics** including:
  - Priority distribution pie chart
  - Confidence level histogram
  - Interactive data tables

## 🚀 Deployment

### Local Deployment
```bash
streamlit run predict.py
```

### Cloud Deployment Options
- Streamlit Cloud
- Heroku
- AWS EC2
- IBM Cloud

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📚 Documentation Files

- **`abstract.md`**: Project abstract and problem statement
- **`Readme.md`**: This comprehensive guide

## ⚠️ Important Notes

- **Git Ignore**: `.gitignore` configured for Python/ML projects
- **Sample Data**: `court_cases.csv` provides input format reference
- **Output Example**: `case_predictions.csv` shows expected results format

## 🆘 Troubleshooting

### Common Issues

1. **Model loading errors**
   - Ensure all `.pkl` files are in the same directory
   - Check file permissions

2. **CSV upload problems**
   - Verify CSV matches `court_cases.csv` format
   - Check for encoding issues

3. **Dependency errors**
   ```bash
   pip install --upgrade streamlit pandas scikit-learn joblib plotly
   ```

### Getting Help
- Check `model_metadata.pkl` for model specifications
- Use provided sample files for format reference

## 📞 Support

For IBM Z Hackathon-related queries:
- Refer to project documentation in `abstract.md`
- Use sample files for testing and validation

---

**Developed for IBM Z Hackathon with ❤️ – Revolutionizing Court Case Management with AI**

*JusticeAI – Court Case Prioritization System*
```

