# VeriMedia

VeriMedia is an AI-driven media analysis tool designed to enhance the capacity of media outlets to report ethically and accurately on topics related to refugees, migrants, and other forcibly displaced populations. This tool supports the detection and prevention of xenophobic language, misinformation, and harmful content in media environments, ultimately fostering more informed and empathetic public discourse.

## Features

- **Multi-format Analysis**: Upload and analyze text, audio, and video content
- **Toxicity Detection**: Identify potentially harmful or xenophobic language in content
- **Improvement Suggestions**: Receive actionable recommendations to enhance content inclusivity
- **Comprehensive Reports**: Get detailed analysis reports to understand areas for improvement
- **Multi-language Support**: Analyze content in multiple languages for global reporting

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/verimedia.git
   cd verimedia
   ```

2. Create a virtual environment and activate it:
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and add your configuration:
   ```
   SECRET_KEY=your-secret-key
   OPENAI_API_KEY=your-openai-api-key
   ```

## Usage

1. Start the Flask development server:
   ```
   python app.py
   ```

2. Open your web browser and navigate to `http://127.0.0.1:5000`

3. Upload your media content (text, audio, or video) for analysis

4. View the analysis results, including toxicity level, suggestions for improvement, and a comprehensive report

## Project Structure

```
verimedia/
├── app.py                  # Main Flask application
├── requirements.txt        # Project dependencies
├── .env                    # Environment variables (create this file)
├── static/                 # Static files
│   ├── css/                # CSS stylesheets
│   ├── js/                 # JavaScript files
│   └── images/             # Image files
├── templates/              # HTML templates
│   ├── base.html           # Base template
│   ├── index.html          # Home page
│   ├── results.html        # Results page
│   ├── about.html          # About page
│   └── contact.html        # Contact page
└── uploads/                # Uploaded files (created automatically)
```

## Technologies Used

- **Flask**: Web framework
- **OpenAI API**: For text analysis
- **HTML/CSS/JavaScript**: Frontend development

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Developed for UNICC (United Nations International Computing Centre)
- Special thanks to all contributors and supporters of this project 