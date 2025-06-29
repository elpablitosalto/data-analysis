# Data Analysis Project

This project demonstrates data analysis and visualization using Python, Pandas, NumPy, and Matplotlib.

## Features

- Data loading and cleaning
- Statistical analysis
- Data visualization
- Report generation

## Setup

1. Install Python 3.8 or higher
2. Create a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   Ensure that seaborn is installed:
   ```
   pip install seaborn
   ```

## Project Structure

- `data/` - Directory for CSV data files
- `src/` - Source code
  - `data_loader.py` - Data loading and cleaning functions
  - `analysis.py` - Statistical analysis functions
  - `visualization.py` - Data visualization functions
- `reports/` - Generated reports and visualizations

## Usage

1. Place your CSV data files in the `data/` directory
2. Run the analysis:
   ```
   python src/main.py
   ```
3. Check the `reports/` directory for generated visualizations and analysis results 