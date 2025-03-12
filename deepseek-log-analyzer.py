import os
import re
import argparse
import pandas as pd
import joblib
import requests
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

class LogAnalyzer:
    def __init__(self, log_file, api_key, model_file="log_classifier.pkl"):
        self.log_file = log_file
        self.logs = []
        self.api_key = api_key
        self.model_file = model_file
        self.model = self.load_model()

    def load_logs(self):
        """Load logs from the file."""
        if not os.path.exists(self.log_file):
            print(f"File {self.log_file} not found.")
            return

        with open(self.log_file, 'r', encoding='utf-8') as file:
            self.logs = file.readlines()

    def extract_pattern(self, pattern):
        """Extract occurrences of a pattern from the logs."""
        matches = []
        regex = re.compile(pattern)
        
        for line in self.logs:
            match = regex.search(line)
            if match:
                matches.append(match.group())
        
        return matches

    def count_occurrences(self, pattern):
        """Count occurrences of a pattern in the logs."""
        matches = self.extract_pattern(pattern)
        return Counter(matches)

    def analyze_with_deepseek(self, log_sample):
        """Use DeepSeek API to analyze log data."""
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are an AI log analysis assistant."},
                {"role": "user", "content": f"Analyze the following log data and summarize key issues:\n{log_sample}"}
            ]
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return "DeepSeek API error: Unable to analyze logs."

    def load_model(self):
        """Load or train a machine learning model for log classification."""
        if os.path.exists(self.model_file):
            return joblib.load(self.model_file)
        else:
            return self.train_model()

    def train_model(self):
        """Train a Naive Bayes classifier for log categorization."""
        sample_logs = [
            "ERROR: Disk space exceeded",
            "WARN: High memory usage detected",
            "INFO: Server started successfully",
            "ERROR: Connection timeout occurred",
            "WARN: CPU usage is high"
        ]
        labels = ["error", "warning", "info", "error", "warning"]
        
        model = make_pipeline(CountVectorizer(), MultinomialNB())
        model.fit(sample_logs, labels)
        joblib.dump(model, self.model_file)
        return model

    def classify_logs(self):
        """Classify logs into categories using the trained model."""
        if not self.logs:
            print("No logs loaded to classify.")
            return []
        
        return self.model.predict(self.logs)

    def generate_report(self, output_file):
        """Generate a CSV report of log patterns, DeepSeek AI analysis, and ML classification."""
        error_counts = self.count_occurrences(r'ERROR|WARN|INFO')
        df = pd.DataFrame(error_counts.items(), columns=['Log Level', 'Count'])
        
        # Sample logs to analyze with DeepSeek
        log_sample = "\n".join(self.logs[:10])  # First 10 lines
        deepseek_analysis = self.analyze_with_deepseek(log_sample)
        
        # Classify logs with ML model
        log_categories = self.classify_logs()
        df_ml = pd.DataFrame({"Log Message": self.logs, "Category": log_categories})
        df_ml.to_csv(output_file.replace('.csv', '_classified.csv'), index=False)
        
        with open(output_file.replace('.csv', '.txt'), 'w', encoding='utf-8') as f:
            f.write("DeepSeek AI Log Analysis:\n")
            f.write(deepseek_analysis)
        
        df.to_csv(output_file, index=False)
        print(f"Report saved to {output_file}")
        print(f"DeepSeek AI analysis saved to {output_file.replace('.csv', '.txt')}")
        print(f"Classified logs saved to {output_file.replace('.csv', '_classified.csv')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log Analysis Tool using DeepSeek AI")
    parser.add_argument("logfile", help="Path to the log file")
    parser.add_argument("apikey", help="DeepSeek API Key")
    parser.add_argument("--report", help="Output CSV file for report", default="log_report.csv")
    args = parser.parse_args()
    
    analyzer = LogAnalyzer(args.logfile, args.apikey)
    analyzer.load_logs()
    analyzer.generate_report(args.report)
