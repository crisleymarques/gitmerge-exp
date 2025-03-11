# 🚀 Conflict-Fix-LLM

## 🤖 Automated Git Merge Conflict Resolution with LLMs

## Overview 📋

Conflict-Fix-LLM is a tool that leverages Large Language Models (LLMs) to analyze and resolve merge conflicts in Git repositories automatically. It integrates multiple LLM providers, including Google Gemini, Groq, and Maritaca, to generate optimal resolutions for conflicting code blocks.


## 🛠 Technologies

- Python 3.8+
- pandas (>=2.0.0) – Data processing and manipulation
- litellm (>=1.63.0) – Unified interface for interacting with multiple LLMs


## 💻 How to Run

### Prerequisites

Ensure you have the following installed on your system: Git, Python 3.8 or later

### Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/crisleymarques/gitmerge-exp.git
   cd gitmerge-exp
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   
5. Edit the `.env` file with your API keys:
   ```
   GOOGLE_API_KEY=your-google-api-key
   GROQ_API_KEY=your-groq-api-key
   MARITACA_API_KEY=your-maritaca-api-key
   
   DEFAULT_LLM_PROVIDER=google
   DEFAULT_LLM_MODEL=gemini-2.0-flash
   ```

### Running the Project

To run the project with default settings:

```bash
python -m src.experiment.main
```

To specify a different LLM provider and model:

```bash
python -m src.experiment.main --provider groq --model llama3-8b-8192
```

Available options:
- `--provider`: LLM provider (google, groq, maritaca)
- `--model`: Specific provider model

### Results

Results are saved in the `data/output/` directory in JSON format, including:
- Processed conflicts
- LLM-generated resolutions
- Time and performance metrics