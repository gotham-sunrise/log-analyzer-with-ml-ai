# Log Analysis Tool (DeepSeek AI)

## Overview
The **Log Analysis Tool** is a Python-based utility designed to analyze log files using:
- **Regular expressions** for pattern extraction.
- **DeepSeek AI API** for intelligent log analysis.
- **Machine Learning (Naïve Bayes)** for log classification into categories like `ERROR`, `WARN`, and `INFO`.
- **CSV and text reports** for insights and visualization.

## Features
- Extracts and counts occurrences of log levels (INFO, WARN, ERROR).
- Utilizes DeepSeek AI for summarizing key log issues.
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
requests
scikit-learn
joblib
```
Install them using:
```sh
pip install pandas requests scikit-learn joblib
```

## Usage
Run the tool with:
```sh
python log_analysis_deepseek.py <logfile> <deepseek_api_key> --report <output_file>
```

### Arguments:
- `<logfile>` - Path to the log file to be analyzed.
- `<deepseek_api_key>` - Your DeepSeek AI API key for log analysis.
- `--report <output_file>` - (Optional) Path to save the report (default: `log_report.csv`).

### Example:
```sh
python log_analysis_deepseek.py server.log sk-123456 --report analysis.csv
```

## Output
The tool generates:
1. **CSV Report (`log_report.csv`)** - Count of log levels (INFO, WARN, ERROR).
2. **DeepSeek AI Analysis (`log_report.txt`)** - AI-driven summary of key log issues.
3. **Classified Logs (`log_report_classified.csv`)** - Categorized logs using Machine Learning.

## Training Machine Learning Model
A Naïve Bayes classifier is trained on sample logs. To retrain:
```sh
rm log_classifier.pkl
python log_analysis_deepseek.py server.log sk-123456
```
This will regenerate `log_classifier.pkl`.

## Notes
- Ensure your DeepSeek API key is valid.
- Large log files may take longer to process.
- The machine learning model can be improved with additional training data.

## License
MIT License

## Contributions
Feel free to fork and contribute to improve the tool!

## Author
Ming Li

