#!/bin/bash
# Setup script for VeriMedia project

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file template..."
    cat > .env << EOL
# OpenAI API key (required)
OPENAI_API_KEY=your_openai_api_key_here

# Flask secret key (required)
SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(16))')
EOL
    echo ".env file created with a randomly generated SECRET_KEY."
    echo "IMPORTANT: Edit the .env file and add your OpenAI API key before running the app."
fi

echo "Setup complete! Activate the virtual environment with:"
echo "source venv/bin/activate"
echo "Then run the app with:"
echo "python3 app.py" 