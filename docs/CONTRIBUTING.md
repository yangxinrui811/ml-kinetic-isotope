# Contributing Guide

## How to Contribute

We welcome contributions to the Tunneling Phase Diagram project! Here's how you can help:

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists in the [issue tracker](https://github.com/yourusername/ml-kinetic-isotope/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - System information (OS, Python version, package versions)

### Submitting Code

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/ml-kinetic-isotope.git
   cd ml-kinetic-isotope
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add docstrings to new functions
   - Update documentation if needed

4. **Test your changes**
   ```bash
   python -m pytest tests/
   ```

5. **Commit and push**
   ```bash
   git add .
   git commit -m "Add: your feature description"
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Describe what your PR does
   - Reference any related issues
   - Ensure all tests pass

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add comments for complex logic
- Keep functions focused and modular

### Documentation

- Update README.md if adding new features
- Add docstrings to all public functions
- Update docs/ files as needed

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ml-kinetic-isotope.git
cd ml-kinetic-isotope

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Run tests
pytest
```

## Questions?

Contact us:
- Xinrui Yang: [EMAIL]
- Zhigang Wang: [EMAIL]

Thank you for contributing!
