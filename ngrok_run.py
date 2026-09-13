"""
ngrok_run.py
Run this file once. It will:
1. Set your ngrok authtoken
2. Kill any old ngrok tunnels
3. Open a new tunnel on port 8501
4. Start your Streamlit app (app.py)

Usage:
    python ngrok_run.py
"""

import subprocess
from pyngrok import ngrok, conf
from config import NGROK_AUTH_TOKEN, PORT

# ---- Step 1: Set authtoken ----
conf.get_default().auth_token = NGROK_AUTH_TOKEN

# ---- Step 2: Kill old tunnels ----
ngrok.kill()

# ---- Step 3: Open new tunnel ----
public_url = ngrok.connect(PORT).public_url
print(f"Streamlit App is live at: {public_url}")

# ---- Step 4: Start Streamlit app ----
try:
    subprocess.run(
        ["streamlit", "run", "app.py", "--server.port", str(PORT)],
        check=True
    )
except KeyboardInterrupt:
    print("\nStopping app and closing tunnel...")
finally:
    ngrok.kill()
