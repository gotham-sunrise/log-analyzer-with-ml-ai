# Log Analysis Tool

## Overview
The **Log Analysis Tool** is a Python-based utility designed to analyze log files efficiently. It leverages:
- **Regular expressions** for pattern extraction.
- **OpenAI's ChatGPT API** for intelligent log analysis.
- **Machine Learning (Naïve Bayes)** for log classification into categories like `ERROR`, `WARN`, and `INFO`.
- **CSV and text reports** for insights and visualization.

## Features
- Extracts and counts occurrences of log levels (INFO, WARN, ERROR).
- Utilizes OpenAI’s GPT-4 for summarizing key log issues.
- Employs machine learning to classify logs.
- Generates structured reports in CSV and TXT formats.

## Installation
Ensure you have Python 3.8+ installed and run:

```sh
pip install -r requirements.txt
```

## Dependencies
The tool requires the following Python libraries:
```sh
pandas
openai
scikit-learn
joblib
```
Install them using:
```sh
pip install pandas openai scikit-learn joblib
```

## Usage
Run the tool with:
```sh
python log_analysis.py <logfile> <openai_api_key> --report <output_file>
```

### Arguments:
- `<logfile>` - Path to the log file to be analyzed.
- `<openai_api_key>` - Your OpenAI API key for GPT-4 analysis.
- `--report <output_file>` - (Optional) Path to save the report (default: `log_report.csv`).

### Example:
```sh
python log_analysis.py server.log sk-123456 --report analysis.csv
```

## Output
The tool generates:
1. **CSV Report (`log_report.csv`)** - Count of log levels (INFO, WARN, ERROR).
2. **ChatGPT Analysis (`log_report.txt`)** - AI-driven summary of key log issues.
3. **Classified Logs (`log_report_classified.csv`)** - Categorized logs using Machine Learning.

## Training Machine Learning Model
A Naïve Bayes classifier is trained on sample logs. To retrain:
```sh
rm log_classifier.pkl
python log_analysis.py server.log sk-123456
```
This will regenerate `log_classifier.pkl`.

## Notes
- Ensure your OpenAI API key is valid.
- Large log files may take longer to process.
- The machine learning model can be improved with additional training data.

## License
MIT License

## Contributions
Feel free to fork and contribute to improve the tool!

## Author
Ming Li

