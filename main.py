"""
FinSight AI — Main Application Entrypoint
==========================================
Run this file to launch the complete FinSight AI web application:
    python main.py
Or run:
    python app/main.py
"""

import sys
import os

# Ensure project root is in sys.path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.main import demo

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    print(f"Launching FinSight AI on port {port}...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )
