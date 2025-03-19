# Job Role Prediction

This project is a machine learning-based job role prediction system that analyzes resumes and predicts the most suitable job role.

## Features
- Uses NLP techniques to process resume data
- Trained machine learning model for job role classification
- Deployable as a Streamlit web application

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/Job_Role_Prediction.git
   cd Job_Role_Prediction
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
Run the Streamlit app:
```sh
streamlit run app.py
```

## File Structure
```
Job_Role_Prediction/
│── app.py                # Streamlit application
│── resume_classifier.pkl  # Trained ML model
│── requirements.txt       # Dependencies
│── data/                 # Dataset files
│── models/               # Trained models
│── notebooks/            # Jupyter notebooks for experiments
```

## Troubleshooting
If you encounter a `KeyError` when loading `resume_classifier.pkl`, ensure:
- The file is correctly placed in the project directory.
- Your Python and library versions match the environment where the model was trained.
- To use the app click here https://job-role-prediction.streamlit.app/

  ```
## Author
Developed by: **CH TEJA YADAV**  
📧 Email: tejayadavch@gmail.com  
💻 GitHub: https://github.com/chtejayadav  

\



