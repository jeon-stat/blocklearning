"""`python -m dlstudio [--port N] [--no-browser]`"""
import sys

from dlstudio.cli import main

sys.exit(main(sys.argv[1:]))
