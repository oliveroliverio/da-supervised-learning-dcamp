#!/bin/bash

echo "🧹 Cleaning up environment..."

# Remove Poetry files
if [ -f poetry.lock ]; then
    echo "Removing poetry.lock"
    rm poetry.lock
fi
if [ -f pyproject.toml ]; then
    echo "Removing pyproject.toml"
    rm pyproject.toml
fi

# Remove Pipenv files
if [ -f Pipfile ]; then
    echo "Removing Pipfile"
    rm Pipfile
fi
if [ -f Pipfile.lock ]; then
    echo "Removing Pipfile.lock"
    rm Pipfile.lock
fi

# Remove virtual environments
for venv in .venv venv; do
    if [ -d "$venv" ]; then
        echo "Removing $venv directory"
        rm -rf "$venv"
    fi
done

# Remove Python version file
if [ -f .python-version ]; then
    echo "Removing .python-version"
    rm -f .python-version
fi

# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete

echo "✨ Creating new virtual environment..."
python3 -m venv venv

echo "🔌 Activating virtual environment..."
source venv/bin/activate

echo "📦 Installing packages from requirements.txt..."
if [ -f requirements.txt ]; then
    echo "Upgrading pip..."
    pip install --upgrade pip
    echo "Installing packages..."
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt not found!"
fi

echo "🎉 Environment reset complete!"
echo "💡 To activate the environment, run: source venv/bin/activate"
