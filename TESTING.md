# Running Tests - Important Notes

## ⚠️ IMPORTANT: Use the Virtual Environment

You **must** use the virtual environment's pytest, not the system/global pytest.

### ❌ Wrong (uses system Python without dependencies):
```bash
pytest -v
```

### ✅ Correct (uses virtual environment with all dependencies):
```bash
# Option 1: Activate virtual environment first
source .venv/bin/activate
pytest -v

# Option 2: Use absolute path
.venv/bin/pytest -v
```

## Why This Matters

- **System pytest**: Uses Homebrew Python 3.11 without project dependencies (numpy, dotenv, vosk, etc.)
- **Virtual env pytest**: Uses Python 3.14 with all dependencies installed

## Running Tests

### Basic Commands
```bash
# Activate virtual environment (recommended)
source .venv/bin/activate

# Run all tests
pytest -v

# Run specific test file
pytest tests/test_config.py -v

# Run specific module
pytest tests/utils/ -v

# Run with coverage
pytest --cov=rdoai --cov-report=term-missing

# Run with HTML coverage report
pytest --cov=rdoai --cov-report=html
open htmlcov/index.html
```

### Without Activating Virtual Environment
```bash
# Run all tests
.venv/bin/pytest -v

# Run with coverage
.venv/bin/pytest --cov=rdoai --cov-report=term-missing
```

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'dotenv'"
**Cause**: Using system pytest instead of virtual environment pytest  
**Solution**: Use `.venv/bin/pytest` or activate the virtual environment first

### Error: "unrecognized arguments: --cov"
**Cause**: pytest-cov not installed in the environment you're using  
**Solution**: 
1. Install dev dependencies: `.venv/bin/pip install -r requirements-dev.txt`
2. Or run without coverage: `.venv/bin/pytest -v`

### Verify Your Setup
```bash
# Check which pytest you're using
which pytest              # Should show /opt/homebrew/bin/pytest (WRONG)
.venv/bin/pytest --version   # Should show pytest 9.0.2 (CORRECT)

# Check Python version
.venv/bin/python --version   # Should be Python 3.14.x
```
