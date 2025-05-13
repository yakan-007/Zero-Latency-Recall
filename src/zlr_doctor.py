import sys
import subprocess
import shutil
import platform
from pathlib import Path
import importlib
import re

# --- Configuration ---
MIN_PYTHON_VERSION = (3, 9)
REQUIRED_LIBS = [
    "pdfplumber",
    "watchdog",
    "yaml",  # PyYAML is imported as yaml
    "dotenv", # python-dotenv is imported as dotenv
]
OPTIONAL_LIBS = {
    "Advanced Extractor / OCR": [
        "pytesseract",
        "PIL",       # Pillow is imported as PIL
        "pdf2image",
        "fitz",      # PyMuPDF is imported as fitz
        "cv2",       # opencv-python is imported as cv2
        "numpy",
    ]
}

# --- Helper Functions ---

def print_status(check_name: str, success: bool, message: str = "", fix_hint: str = ""):
    status = "[OK]" if success else "[FAIL]"
    print(f"{status} {check_name}")
    if message:
        print(f"     {message}")
    if not success and fix_hint:
        print(f"     To fix: {fix_hint}")

def check_python_version():
    current_version = sys.version_info
    success = current_version >= MIN_PYTHON_VERSION
    message = f"Python version: {platform.python_version()} (recommend >={MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]})"
    print_status("Python Version Check", success, message)
    return success

def check_library(lib_name: str, import_as: str | None = None) -> bool:
    actual_import_name = import_as if import_as else lib_name
    try:
        importlib.import_module(actual_import_name)
        print_status(f"Library: {lib_name}", True)
        return True
    except ImportError:
        print_status(f"Library: {lib_name}", False, fix_hint=f"pip install {lib_name}")
        return False

# --- Helper Functions for External Tools ---

def find_executable(name: str) -> str | None:
    """Checks if an executable exists in PATH."""
    return shutil.which(name)

def get_tesseract_version(tesseract_cmd: str) -> str | None:
    """Gets Tesseract version."""
    try:
        result = subprocess.run([tesseract_cmd, "--version"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            # Output is typically "tesseract X.Y.Z ..."
            match = re.search(r"tesseract\\s+(\\S+)", result.stdout)
            if match:
                return match.group(1)
    except FileNotFoundError:
        pass # Handled by find_executable
    except Exception as e:
        print(f"     Error checking Tesseract version: {e}")
    return None

def get_tesseract_installed_langs(tesseract_cmd: str) -> list[str]:
    """Gets list of installed Tesseract languages."""
    langs = []
    try:
        result = subprocess.run([tesseract_cmd, "--list-langs"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            # Output after "List of available languages..."
            lines = result.stdout.splitlines()
            start_index = -1
            for i, line in enumerate(lines):
                if "List of available languages" in line: # Actual string varies
                    start_index = i + 1
                    break
            if start_index != -1:
                langs = [lang.strip() for lang in lines[start_index:] if lang.strip()]
    except FileNotFoundError:
        pass # Handled by find_executable
    except Exception as e:
        print(f"     Error listing Tesseract languages: {e}")
    return langs

def get_poppler_version(pdftoppm_cmd: str) -> str | None:
    """Gets Poppler (via pdftoppm) version."""
    try:
        # pdftoppm -v output goes to stderr
        result = subprocess.run([pdftoppm_cmd, "-v"], capture_output=True, text=True, check=False)
        if result.returncode == 0 or result.returncode == 1 : # Poppler often returns 1 for -v
             #stderr output example: "pdftoppm version 23.08.0"
            match = re.search(r"pdftoppm version\\s+(\\S+)", result.stderr)
            if match:
                return match.group(1)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"     Error checking Poppler (pdftoppm) version: {e}")
    return None

# --- Main Check Functions ---

def run_checks():
    print("ZLR System Health Check:")
    print("------------------------")

    all_critical_ok = True

    # Python version
    if not check_python_version():
        all_critical_ok = False

    # Required libraries
    print("\nRequired Libraries:")
    for lib in REQUIRED_LIBS:
        # Handle special import names
        import_as_name = None
        if lib == "yaml": import_as_name = "yaml" # PyYAML
        if lib == "dotenv": import_as_name = "dotenv" # python-dotenv

        if not check_library(lib, import_as=import_as_name):
            all_critical_ok = False
            # For core libs, we can consider them critical.

    # Optional libraries
    print("\nOptional Libraries:")
    for category, libs in OPTIONAL_LIBS.items():
        print(f"  {category}:")
        category_all_ok = True
        for lib in libs:
            import_as_name = None
            if lib == "PIL": import_as_name = "PIL"
            if lib == "fitz": import_as_name = "fitz"
            if lib == "cv2": import_as_name = "cv2"

            if not check_library(lib, import_as=import_as_name):
                category_all_ok = False
        # Optional libs don't make all_critical_ok False

    # --- Placeholder for other checks ---
    print("\nExternal Tools & Configuration (TODO):")

    # Tesseract OCR Check
    tesseract_exe = find_executable("tesseract")
    tesseract_ok = bool(tesseract_exe)
    tesseract_version = None
    jpn_lang_ok = False
    jpn_vert_lang_ok = False
    tesseract_fix_hint = "Install Tesseract OCR and add it to your PATH. See README for instructions."

    if tesseract_ok and tesseract_exe:
        tesseract_version = get_tesseract_version(tesseract_exe)
        if tesseract_version:
            message = f"Found at: {tesseract_exe}, Version: {tesseract_version}"
        else:
            message = f"Found at: {tesseract_exe} (Version could not be determined)"
        
        installed_langs = get_tesseract_installed_langs(tesseract_exe)
        if "jpn" in installed_langs:
            jpn_lang_ok = True
        if "jpn_vert" in installed_langs:
            jpn_vert_lang_ok = True
        
        lang_message = f"Installed languages: {', '.join(installed_langs) if installed_langs else 'None found'}"
        print_status("Tesseract OCR", True, message)
        print_status("  - Japanese language data (jpn)", jpn_lang_ok, fix_hint="Install 'jpn.traineddata' for Tesseract.")
        print_status("  - Vertical Japanese language data (jpn_vert)", jpn_vert_lang_ok, fix_hint="Install 'jpn_vert.traineddata' for Tesseract.")
        if not (jpn_lang_ok and jpn_vert_lang_ok): # If any lang is missing, overall tesseract is not "fully" ok for ZLR
             tesseract_ok = False # Downgrade status for summary
    else:
        print_status("Tesseract OCR", False, "Tesseract OCR not found in PATH.", fix_hint=tesseract_fix_hint)


    # Poppler (pdf2image dependency) Check
    # We check for pdftoppm, which is a core Poppler utility pdf2image often uses
    poppler_exe_name = "pdftoppm"
    poppler_exe = find_executable(poppler_exe_name)
    poppler_ok = bool(poppler_exe)
    poppler_version = None
    poppler_fix_hint = "Install Poppler and add its bin directory to your PATH. See README for instructions."

    if poppler_ok and poppler_exe:
        poppler_version = get_poppler_version(poppler_exe)
        message = f"Found {poppler_exe_name} at: {poppler_exe}"
        if poppler_version:
            message += f", Version: {poppler_version}"
        else:
            message += " (Version could not be determined)"
        print_status("Poppler Utilities (for pdf2image)", True, message)
    else:
        print_status("Poppler Utilities (for pdf2image)", False, f"{poppler_exe_name} not found in PATH.", fix_hint=poppler_fix_hint)

    # Configuration File Checks
    print("\nConfiguration Checks:")
    project_root = Path(__file__).resolve().parent.parent # Assuming zlr_doctor.py is in project root -> src, so parent.parent is project root

    # .env check
    env_path = project_root / ".env"
    env_exists = env_path.exists()
    obsidian_vault_set = False
    if env_exists:
        print_status(".env file", True, f"Found at: {env_path}")
        # Try to load .env to check OBSIDIAN_VAULT without importing full config
        try:
            from dotenv import dotenv_values
            env_vars = dotenv_values(env_path)
            if "OBSIDIAN_VAULT" in env_vars and env_vars["OBSIDIAN_VAULT"]:
                obsidian_vault_set = True
                print_status("  - OBSIDIAN_VAULT in .env", True, f"Set to: {env_vars['OBSIDIAN_VAULT']}")
            else:
                print_status("  - OBSIDIAN_VAULT in .env", False, "Not set or empty.", fix_hint="Add OBSIDIAN_VAULT=/path/to/your/vault to your .env file for Obsidian integration.")
        except Exception as e:
            print_status("  - OBSIDIAN_VAULT in .env", False, f"Could not parse .env to check OBSIDIAN_VAULT: {e}")
    else:
        print_status(".env file", False, f"Not found at: {env_path}", fix_hint="Create a .env file in the project root for environment-specific settings (e.g., OBSIDIAN_VAULT).")

    # extract/config.py WATCH_FOLDER_PATH check
    try:
        # Temporarily add extract to sys.path to import config if zlr_doctor is at root
        # This is a bit of a hack for a standalone script. Proper CLI integration would be cleaner.
        # original_sys_path = list(sys.path) # 削除
        # if str(project_root.parent) not in sys.path: # project_root が src を指していた時の名残 -> project_root.parent は src の親、つまりプロジェクトルートを指す
        #      sys.path.insert(0, str(project_root.parent)) # 削除
        
        from extract.config import WATCH_FOLDER_PATH, OBSIDIAN_INBOX_PATH # Also check OBSIDIAN_INBOX_PATH processing
        
        # sys.path = original_sys_path # Restore sys.path # 削除

        if WATCH_FOLDER_PATH:
            watch_folder_valid = Path(WATCH_FOLDER_PATH).is_dir()
            msg = f"extract.config.WATCH_FOLDER_PATH: {WATCH_FOLDER_PATH}"
            hint = "Ensure the path in extract.config.py for WATCH_FOLDER_PATH points to a valid directory."
            print_status("WATCH_FOLDER_PATH in config", watch_folder_valid, msg, fix_hint=hint if not watch_folder_valid else "")
        else:
            print_status("WATCH_FOLDER_PATH in config", False, "Not set in extract.config.py", fix_hint="Define WATCH_FOLDER_PATH in extract.config.py for the watch feature.")

        # OBSIDIAN_INBOX_PATH (derived from OBSIDIAN_VAULT) check from config perspective
        if obsidian_vault_set: # Only relevant if .env had the base path
            if OBSIDIAN_INBOX_PATH and OBSIDIAN_INBOX_PATH.exists():
                print_status("Derived OBSIDIAN_INBOX_PATH", True, f"Usable and points to: {OBSIDIAN_INBOX_PATH}")
            elif OBSIDIAN_INBOX_PATH: # Set but does not exist
                print_status("Derived OBSIDIAN_INBOX_PATH", False, f"Set to {OBSIDIAN_INBOX_PATH}, but directory does not exist.", fix_hint="Ensure the .../zlr-inbox directory exists in your Obsidian vault or will be auto-created.")
            # If OBSIDIAN_INBOX_PATH is None, it means OBSIDIAN_VAULT was likely not set, already covered by .env check.

    except ImportError as e:
        print_status("extract.config.py loading", False, f"Could not import from extract.config: {e}", fix_hint="Ensure zlr_doctor.py is in the project root or adjust Python's import path.")
    except AttributeError as e: # If WATCH_FOLDER_PATH is not defined in config
        print_status("WATCH_FOLDER_PATH in config", False, f"Attribute error: {e}. Likely not defined in extract.config.py.", fix_hint="Define WATCH_FOLDER_PATH in extract.config.py.")
    except Exception as e:
        print_status("extract.config.py WATCH_FOLDER_PATH", False, f"Error checking WATCH_FOLDER_PATH: {e}")

    print("------------------------")
    if all_critical_ok:
        print("System status: All critical dependencies seem to be met.")
        print("Further checks for external tools and configurations are pending.")
    else:
        print("System status: Critical dependencies are MISSING. Please address the [FAIL] items above.")


if __name__ == "__main__":
    run_checks() 