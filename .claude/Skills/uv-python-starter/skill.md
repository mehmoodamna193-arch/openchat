---
name: uv-python-starter
description: Scaffold a new Python backend project with uv (init, run, sync) and a Hello World entry point. Use when the user asks to start, bootstrap or scaffold a Python/uv project or backend.
---

# uv Python Starter

Creates a minimal Python project called backend using [uv](https://docs.astral.sh/uv/), drops in a Hello World main.py, runs it, and syncs the environment.

## Prerequisites

Check that uv is installed:

bash
uv --version


If it is missing, install it:

- macOS / Linux: curl -LsSf https://astral.sh/uv/install.sh | sh
- Windows (PowerShell): powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
- Or: pip install uv

## Steps

1. *Create the project*

   bash
   uv init backend
   cd backend…