# ece461-project
Project for ECE46100 (30861) Purdue Fall 2025

# team members
- Josh LeBlanc

- Dylan Brannick

- Herbert Alexander de Bruyn

- James Neff

# Repository Structure
```text
ece461-project/
├── run                      # root executable: ./run install | test | <URL_FILE>
├── requirements.txt         # deps (pytest, pytest-cov, etc.)
├── README.md
├── config/                  # config files
├── docs/                    # design notes, diagrams, writeups
├── src/
│   └── ece461/              # your Python package (imported as 'ece461')
│       ├── __init__.py
│       ├── main.py          # URL mode entrypoint: python3 -m ece461.main <URL_FILE>
│       └── metrics/         # metric implementations
└── tests/
```
