# VeriMedia

VeriMedia is an AI-powered tool designed to help journalists, content creators, and the general public analyze media content for ethical and accurate reporting.

## Features

- **Text Analysis**: Analyze text documents for problematic language, misinformation, and harmful content
- **Audio Analysis**: Transcribe and analyze audio recordings for ethical considerations
- **Video Analysis**: Extract audio from videos, transcribe, and analyze for content improvement
- **Language Detection**: Identify potentially problematic words with word cloud visualization
- **Comprehensive Reports**: Get detailed analysis reports with toxicity levels and improvement suggestions
- **PDF Export**: Download analysis results as PDF reports

## Getting Started

### Prerequisites

- Python 3.9+
- FFmpeg (for audio/video processing)

### Installation

#### Option 1: Automated Setup (recommended)

1. Clone the repository:
   ```
   git clone https://github.com/FlashCarrot/VeriMedia.git
   cd VeriMedia
   ```

2. Run the setup script:
   ```
   ./setup.sh
   ```

3. Edit the `.env` file to add your OpenAI API key

4. Run the application:
   ```
   python3 app.py
   ```

#### Option 2: Manual Setup

1. Clone the repository:
   ```
   git clone https://github.com/FlashCarrot/VeriMedia.git
   cd VeriMedia
   ```

2. Create a virtual environment and activate it:
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   SECRET_KEY=your_secret_key_for_sessions
   ```

5. Run the application:
   ```
   python3 app.py
   ```

6. Open your browser and navigate to:
   ```
   http://127.0.0.1:5004/
   ```

## Usage

1. Select the type of media you wish to analyze (text, audio, or video)
2. Upload your file (supported formats: txt, pdf, docx, mp3, wav, ogg, mp4, etc.)
3. View the analysis results, which include:
   - Toxicity level assessment
   - Specific improvement suggestions
   - Full transcription (for audio/video)
   - Word cloud of problematic language (for text)
   - Comprehensive analysis report
4. Download the report as a PDF for your records

## Security

- **API Keys**: Never commit your API keys to the repository. The `.env` file is included in `.gitignore` to prevent accidental commits.
- **Environment Variables**: Always use environment variables for sensitive information.
- **Secret Management**: For production deployment, use a proper secret management solution.

## Team

The VeriMedia tool is developed by a team of specialists:
- Fengyu Yang - Text Content Specialist
- Qianwen Zhu - Audio Content Specialist
- Qi Sheng - Video Content Specialist
- Somya Panchal - Integration and System Specialist

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Powered by OpenAI's GPT-4o and Whisper API for content analysis and transcription
- Developed as part of the UNICC (United Nations International Computing Centre) project 