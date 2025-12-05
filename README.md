# ClaimCheck

## Overview
ClaimCheck is a fact-checking system that processes claims and verifies their veracity using various modules and models.

## Pre-requisites
1. Create a new Programmable Search Engine in Google:
   - Go to the [Programmable Search Engine](https://cse.google.com/cse/) and create a new search engine.
   - Note down the CSE ID.
   - Enable the Custom Search JSON API in the [Google Cloud Console](https://console.cloud.google.com/).
   - Note down the API key.

2. Get your API key from [SerpAPI](https://serper.dev/).

3. Create a `.env` file in the project root directory and fill in the following environment variables:

-`SERPER_API_KEY`: API key from Serper
- `FACTCHECKER_MODEL_NAME`: The Ollama model name used for fact-checking (default: "qwen2.5:0.5b")
- `FACTCHECKER_MAX_ACTIONS`: Maximum number of actions to run (default: 2)

Example `.env` file content:

```
FACTCHECKER_MODEL_NAME=qwen2.5:0.5b
FACTCHECKER_MAX_ACTIONS=2
```

**Note**: Ensure the `.env` file is not committed to Git (add it to `.gitignore`).

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/idirlab/ClaimCheck.git
    cd ClaimCheck
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    playwright install chromium
    ```


## Usage
To run the fact-checking system from the command line, use the `fact-check.py` script. It takes three arguments: the path to the JSON file containing the claims, the number of records to process and model name.

### Command Line Arguments:
- `json_path`: Path to the AVeriTeC JSON file.
- `num_records`: Number of claims to run.
- `model_name`: Name of the model.

### Example:
```bash
python fact-check.py /path/to/json/file.json 5 qwen2.5:0.5b
```

Replace `/path/to/json/file.json` with the actual path to your AVeriTeC JSON file and `5` with the number of records you want to process. You can find AVeriTeC JSON files [here](https://fever.ai/dataset/averitec.html).


## Streamlit UI (Tiếng Việt)

Bạn có thể chạy giao diện đơn giản để nhập claim, chọn ngày cắt (cut-off) và xem quá trình suy luận, bằng chứng, kết luận.
You can run simple UI to input claim, choose cut-off date then have verdict and evidence.

```bash
streamlit run app.py
```
3) Mở trình duyệt theo URL mà Streamlit hiển thị (thường là http://localhost:8501). Nhập claim, chọn ngày, và nhấn "Chạy kiểm chứng".
4) Open streamlit UI (often at http://localhost:8501)
App will save report at `reports/<timestamp>/ including `report.md`, `evidence.md`, `report.json`. 