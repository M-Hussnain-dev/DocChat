#!/usr/bin/env python3
"""
One-command setup for DocChat.

    python setup.py            install everything
    python setup.py --run      same, then start the app
    python setup.py --test     same, then run the automated tests
    python setup.py --key      save your Gemini API key on this computer (never upload that file)
"""
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def step(msg):
    print(f"\n==> {msg}")


def ask(q):
    try:
        return input(f"{q} [y/N] ").strip().lower() == "y"
    except EOFError:
        return False


def save_key():
    import getpass
    key = getpass.getpass("Paste your Gemini API key (hidden while typing): ").strip()
    if not key:
        print("No key entered.")
        return
    folder = ROOT / ".streamlit"
    folder.mkdir(exist_ok=True)
    f = folder / "secrets.toml"
    lines = [l for l in f.read_text().splitlines() if not l.startswith("GEMINI_API_KEY")] if f.exists() else []
    f.write_text("\n".join(lines + [f'GEMINI_API_KEY = "{key}"']) + "\n")
    print(f"Saved to {f}\nKeep this file on your computer only: never upload the .streamlit folder to GitHub.")


def main():
    if "--key" in sys.argv:
        save_key()
        return
    if sys.version_info < (3, 9):
        sys.exit("Python 3.9 or newer is required.")
    step("Installing Python libraries")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])

    step("Checking Tesseract OCR (needed only for scanned PDFs and images)")
    win = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if shutil.which("tesseract") or win.exists():
        print("  found")
    else:
        system = platform.system()
        if system == "Windows":
            print("  Not installed. Optional: https://github.com/UB-Mannheim/tesseract/wiki (tick Urdu), then run setup again.")
        elif system == "Linux" and shutil.which("apt-get") and ask("  Install Tesseract with apt now?"):
            subprocess.call(["sudo", "apt-get", "install", "-y", "tesseract-ocr", "tesseract-ocr-eng", "tesseract-ocr-urd"])
        elif system == "Darwin" and shutil.which("brew") and ask("  Install Tesseract with brew now?"):
            subprocess.call(["brew", "install", "tesseract", "tesseract-lang"])
        else:
            print("  Not installed. Text PDFs, Word, slides, sheets and websites still work without it.")

    print("\nSetup finished. Start the app with:\n    streamlit run app.py\n")
    if "--test" in sys.argv:
        subprocess.call([sys.executable, "test_system.py"], cwd=ROOT)
    if "--run" in sys.argv:
        subprocess.call([sys.executable, "-m", "streamlit", "run", "app.py"], cwd=ROOT)


if __name__ == "__main__":
    main()
