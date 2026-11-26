"""`python -m blocklearning [--port N] [--no-browser]`"""
import sys

from blocklearning.cli import main

sys.exit(main(sys.argv[1:]))
