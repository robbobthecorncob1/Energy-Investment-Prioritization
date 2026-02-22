import subprocess
import sys

def run_script(script_name):
    """
    Runs the given script. Creates a subprocess for the script and handles errors.

    Args:
        script_name (string): The path to the script
    """
    print(f"Running {script_name}...")
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    if result.returncode != 0:
        print(f"Error in {script_name}. Pipeline halted.")
        sys.exit(1)

if __name__ == "__main__":
    scripts = [
        "01_data_prep.py",
        "02_model_training.py",
        "03_generate_perf_signals.py"
    ]
    
    for script in scripts:
        run_script(script)
    
    print("Pipeline complete! Run 'streamlit run app.py' to view results.")