# Streamlit Community Cloud Entrypoint
# Delegates execution to app.py so both app.py and streamlit_app.py work seamlessly.

import sys
from pathlib import Path
import runpy

# Ensure repository root is on Python path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

app_path = REPO_ROOT / "app.py"
runpy.run_path(str(app_path), run_name="__main__")
