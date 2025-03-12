# Log Analyzer Project

## Overview
The **Log Analyzer Project** is a Python-based tool designed for **log classification and analysis**. It combines **machine learning (Naïve Bayes classifier)** and **AI-powered insights (DeepSeek/OpenAI)** to efficiently process logs.

## Features
- **Log classification using `log_classifier.pkl`** (ML model trained with Naïve Bayes).
- **AI-based log insights** using either **DeepSeek AI** or **OpenAI (GPT-4)**.
- **Pattern extraction** for logs (INFO, WARN, ERROR).
- **Automated report generation** in CSV format.
- **Customizable API selection** (DeepSeek or OpenAI).

## Project Structure
```
log-analyzer-with-ml-ai/
│── log_analysis.py         # Main script for log classification & AI analysis
│── generate_java_tests.py  # Script to generate Java unit tests using DeepSeek AI
│── code_review.py          # Java code review automation using DeepSeek AI
│── models/
│   ├── log_classifier.pkl  # Pre-trained ML model for log classification
│   ├── style_detector.pkl  # Style analysis model
│── reports/
│   ├── log_report.csv      # Generated log analysis report
│   ├── code_review.csv     # Code review report
│── README.md               # Main documentation
│── requirements.txt        # Dependencies
```

## Installation
Ensure you have Python 3.8+ installed, then install dependencies:
```sh
pip install -r requirements.txt
```

## Usage
### Log Analysis
```sh
python log_analysis.py <logfile> <apikey> --provider deepseek --report output.csv
```
- `<logfile>` - Path to the log file.
- `<apikey>` - Your DeepSeek/OpenAI API key.
- `--provider` - Choose `deepseek` or `openai`.
- `--report` - (Optional) Output file (default: `log_report.csv`).

### Java Code Review
```sh
python code_review.py <project_dir> <apikey> --report review.csv
```
- Uses **DeepSeek AI** to analyze Java code.
- Suggests **improvements** and **modifies** the code automatically.

### Java Unit Test Generation
```sh
python generate_java_tests.py <project_dir> <apikey>
```
- Automatically **creates unit tests** for all Java classes in `src/main/java/`.
- Uses **JUnit & Mockito** when necessary.
- Saves test files in `src/test/java/`.

## AI & Machine Learning Integration
1. **`log_classifier.pkl`** - Classifies logs into **INFO, WARNING, ERROR** using ML.
2. **DeepSeek/OpenAI API** - Provides insights into logs.
3. **`style_detector.pkl`** - Detects style inconsistencies in Java projects.

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

