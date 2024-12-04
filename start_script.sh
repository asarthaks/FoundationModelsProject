python3 -m venv .env
echo "Virtual environment '.env' created."
source .env/bin/activate
echo "Virtual environment activated."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "Dependencies installed."
fi
python main.py
echo "Script executed."
deactivate 