"""
streamlit_app.py – Streamlit Cloud / Hugging Face Spaces entrypoint.
Just imports app.py so the platform's auto-detection works.
"""
import runpy
runpy.run_path("app.py", run_name="__main__")
