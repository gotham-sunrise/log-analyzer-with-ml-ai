# Log Analyzer Project

## Overview
The **Log Analyzer Project** is a Python-based tool designed for **log classification and analysis**. It supports both **DeepSeek AI** and **OpenAI (GPT-4)** for AI-powered insights and leverages **machine learning** for log classification.

## Features
- **Log classification using `log_classifier.pkl`** (ML model trained with Naïve Bayes).
- **AI-powered log insights** using either **DeepSeek AI** or **OpenAI (GPT-4)**.
- **Pattern extraction** for logs (INFO, WARN, ERROR).
- **Automated report generation** in CSV format.
- **Customizable API selection** (`deepseek` or `openai`).

## Project Structure
```
log_analyzer_project/
│── deepseek-log-analyzer.py  # Log analyzer using DeepSeek AI
│── openai-log-analyzer.py    # Log analyzer using OpenAI GPT-4
│── models/
│   ├── log_classifier.pkl    # Pre-trained ML model for log classification
│── reports/
│   ├── log_report.csv        # Generated log analysis report
│── README.md                 # Main documentation
│── README-DeepSeek.md        # DeepSeek-specific documentation
│── README-OpenAI.md          # OpenAI-specific documentation
│── requirements.txt          # Dependencies
```

## Installation
Ensure you have Python 3.8+ installed, then install dependencies:
```sh
pip install -r requirements.txt
```

## Usage
### Log Analysis
#### Using DeepSeek AI
```sh
python deepseek-log-analyzer.py <logfile> <apikey> --report output.csv
```
#### Using OpenAI GPT-4
```sh
python openai-log-analyzer.py <logfile> <apikey> --report output.csv
```
- `<logfile>` - Path to the log file.
- `<apikey>` - Your API key for DeepSeek/OpenAI.
- `--report` - (Optional) Output file (default: `log_report.csv`).

## AI & Machine Learning Integration
1. **`log_classifier.pkl`** - Classifies logs into **INFO, WARNING, ERROR** using ML.
2. **DeepSeek/OpenAI API** - Provides insights into logs.

## Notes
- **DeepSeek and OpenAI APIs are interchangeable**.
- The log classification ML model can be **retrained** if needed.
- AI-based modifications should be **manually reviewed**.

## License
MIT License

## Contributions
Feel free to fork and improve the project!

## Author
Ming Li

