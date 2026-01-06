using a virtual python env to test headless=false first and then we will dockerize and create a requirements.txt
py -3.10 0m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -U pip
pip install pytest-playwright
python -m playwright install
