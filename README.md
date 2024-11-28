## Programming For AI
#### Instructor: Miss Mehak Mahzar

## Group Memebers 
#### 23K-0030 Abser Mansoor
#### 23K-0032 Saif Ur Rehaman
#### 23K-0052 Muhammad Ali

## Libraries and Tools
- Flask
- Pandas
- Numpy
- Seaborn and Matplot
- HTML
  
## Prerequisites
- Python 3.8+
- pip (Python package manager)

## Dataset
- Source: [Kaggle Energy Efficiency Dataset](https://www.kaggle.com/datasets/elikplim/eergy-efficiency-dataset)
- Filename: `ENB2012_data.csv`

## Installation Steps

### 1. Clone the Repository
```bash
git clone [text](https://github.com/Muhammadali0981/paiProjectFall24)
cd energy-efficiency-analysis
```

### 2. Create Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Dataset
- Download the `ENB2012_data.csv` from the Kaggle link
- Place the file in the project root directory

### 5. Run the Application
```bash
python app.py
```

## Project Structure
```
energy-efficiency-analysis/
│
├── app.py                  # Main Flask application
│
├── templates/              # HTML templates
│   ├── index.html
│   ├── graphs.html
│   ├── regression.html
│   └── statistics.html
│
├── analysis/               # Analysis modules
│   ├── Graphs.py
│   ├── KNeighboursRegressor.py
│   └── statistical_analysis.py
│
├── static/                 # Generated visualization outputs
│   └── (auto-generated image files)
│
├── requirements.txt        # Project dependencies
└── ENB2012_data.csv        # Dataset file
```

## Features
- Interactive web interface
- Multiple data visualization techniques
- K-Nearest Neighbors Regression
- Comprehensive statistical analysis

## Accessing the Application
- Open a web browser
- Navigate to `http://127.0.0.1:5000/`
- Explore different sections:
  * Graphs
  * Regression Analysis
  * Statistical Analysis

## Troubleshooting
- Ensure all dependencies are installed
- Verify Python version compatibility
- Check that `ENB2012_data.csv` is in the correct location

