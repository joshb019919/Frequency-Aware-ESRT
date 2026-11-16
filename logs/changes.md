## train.py
    Removed architecture from import line `from model import architecture, esrt`
    Put main training pipe in a `main()` function and called from `if __name__ == "__main__"` guard
        Otherwise, it'll try to call more subprocesses before the first one is done setting up, 
        forcing `ConnectionResetError: [Errno 104] Connection reset by peer`
    Put `environ.get("HF_TOKEN")` on main.py:66 and appropriate import stmt.