from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, make_response, session
from werkzeug.utils import secure_filename
import os
import httpx
from openai import OpenAI
from dotenv import load_dotenv
import tempfile
import time
import subprocess
import matplotlib
# Set non-interactive backend before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64
from io import BytesIO
import re
import json
import platform

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-for-development')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload size for videos
app.config['MAX_VIDEO_SIZE'] = 500 * 1024 * 1024  # 500MB max video file size
app.config['MAX_AUDIO_SIZE'] = 25 * 1024 * 1024  # 25MB max audio size (OpenAI API limit)

# Update allowed extensions to match what we support
app.config['ALLOWED_EXTENSIONS'] = {
    'text': {'txt', 'pdf', 'doc', 'docx'},
    'audio': {'mp3', 'wav', 'ogg', 'm4a', 'flac', 'oga', 'mpga'},
    'video': {'mp4', 'webm', 'mpeg', 'mov', 'avi', 'mkv'}  # Added more video formats
}

# OpenAI Audio API supported formats (for reference and validation)
AUDIO_SUPPORTED_FORMATS = {'flac', 'm4a', 'mp3', 'mp4', 'mpeg', 'mpga', 'oga', 'ogg', 'wav', 'webm'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create a temporary directory for audio conversions
TEMP_AUDIO_DIR = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_audio')
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# OpenAI API key (will be set later)
openai_api_key = os.environ.get('OPENAI_API_KEY')
client = OpenAI(
    api_key=openai_api_key,
    http_client=httpx.Client()
)

def convert_video_to_audio(video_path):
    """Convert video to compressed audio format suitable for OpenAI API"""
    try:
        # Create a temporary audio file with a unique name
        temp_audio_file = tempfile.NamedTemporaryFile(suffix='.mp3', dir=TEMP_AUDIO_DIR, delete=False)
        temp_audio_path = temp_audio_file.name
        temp_audio_file.close()
        
        print(f"Converting video to audio: {video_path} -> {temp_audio_path}")
        
        # Try with moviepy first
        try:
            from moviepy.editor import VideoFileClip
            
            # Load video with optimized settings
            video_clip = VideoFileClip(
                video_path,
                audio_buffersize=20000,  # Increased buffer for larger files
                verbose=False,
                target_resolution=(360, 640)  # Lower resolution to save memory
            )
            
            if video_clip.audio:
                # Extract audio with aggressive compression
                video_clip.audio.write_audiofile(
                    temp_audio_path,
                    verbose=False,
                    logger=None,
                    bitrate='48k',  # Very low bitrate
                    ffmpeg_params=[
                        '-ac', '1',  # Mono
                        '-ar', '16000',  # 16kHz sample rate
                        '-q:a', '9'  # Lowest quality
                    ]
                )
                video_clip.close()
                print("Audio extraction successful with moviepy")
            else:
                video_clip.close()
                raise Exception("No audio track found in video")
                
        except Exception as moviepy_error:
            print(f"MoviePy extraction failed: {str(moviepy_error)}")
            
            # Try with ffmpeg directly
            try:
                print("Attempting ffmpeg extraction...")
                from imageio_ffmpeg import get_ffmpeg_exe
                ffmpeg_exe = get_ffmpeg_exe()
                
                # Use ffmpeg with aggressive compression
                cmd = [
                    ffmpeg_exe, '-y',  # Overwrite output files
                    '-i', video_path,  # Input file
                    '-vn',  # No video
                    '-acodec', 'libmp3lame',  # MP3 codec
                    '-ac', '1',  # Mono
                    '-ar', '16000',  # 16kHz sample rate
                    '-b:a', '48k',  # 48kbps bitrate
                    '-filter:a', 'volume=1.5',  # Slightly boost volume
                    '-filter:a', 'loudnorm',  # Normalize audio
                    '-q:a', '9',  # Lowest quality
                    temp_audio_path
                ]
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    print(f"FFmpeg error: {stderr.decode('utf-8', errors='ignore')}")
                    raise Exception("FFmpeg conversion failed")
                    
                print("Audio extraction successful with ffmpeg")
                
            except Exception as ffmpeg_error:
                print(f"FFmpeg extraction failed: {str(ffmpeg_error)}")
                raise Exception(f"Could not extract audio: {str(ffmpeg_error)}")
        
        # Verify the audio file
        if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
            raise Exception("Audio extraction failed - output file is empty")
        
        # Check if the extracted audio is still too large
        audio_size = os.path.getsize(temp_audio_path)
        if audio_size > app.config['MAX_AUDIO_SIZE']:
            print(f"Audio file too large ({audio_size/1024/1024:.1f}MB), attempting additional compression...")
            
            try:
                from pydub import AudioSegment
                
                # Load and compress audio
                audio = AudioSegment.from_file(temp_audio_path)
                
                # Apply aggressive compression
                audio = audio.set_channels(1)  # Mono
                audio = audio.set_frame_rate(16000)  # 16kHz
                
                # Export with minimal quality
                audio.export(
                    temp_audio_path,
                    format="mp3",
                    bitrate="32k",  # Even lower bitrate
                    parameters=[
                        "-q:a", "9",  # Lowest quality
                        "-ac", "1",  # Force mono again
                        "-ar", "16000"  # Force sample rate again
                    ]
                )
                
                # Check final size
                audio_size = os.path.getsize(temp_audio_path)
                if audio_size > app.config['MAX_AUDIO_SIZE']:
                    raise Exception(f"Audio still too large after compression: {audio_size/1024/1024:.1f}MB")
                    
            except Exception as comp_error:
                print(f"Audio compression failed: {str(comp_error)}")
                raise Exception(f"Audio compression failed: {str(comp_error)}")
        
        return temp_audio_path
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except:
                pass
        raise e

def cleanup_temp_files():
    """Clean up old temporary files"""
    try:
        # Clean files older than 1 hour
        threshold = time.time() - 3600
        
        for root, dirs, files in os.walk(TEMP_AUDIO_DIR):
            for file in files:
                path = os.path.join(root, file)
                if os.path.getctime(path) < threshold:
                    try:
                        os.unlink(path)
                    except:
                        pass
    except:
        pass

# Register cleanup function to run periodically
@app.before_request
def before_request():
    cleanup_temp_files()

# Error handler for file too large
@app.errorhandler(413)
def request_entity_too_large(error):
    flash('The file is too large. Maximum allowed size is 500MB.', 'error')
    return redirect(url_for('index')), 413

def allowed_file(filename, file_type=None):
    """Check if the file extension is allowed"""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if file_type:
        return ext in app.config['ALLOWED_EXTENSIONS'].get(file_type, [])
    return any(ext in extensions for extensions in app.config['ALLOWED_EXTENSIONS'].values())

@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis"""
    file_path = None
    audio_path = None
    
    try:
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(url_for('index'))
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        # Check file size before processing
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # Reset file pointer
        
        file_type = request.form.get('file_type')
        
        # Different size limits for different file types
        if file_type == 'video' and file_size > app.config['MAX_VIDEO_SIZE']:
            flash(f'The video file is too large. Maximum allowed size is {app.config["MAX_VIDEO_SIZE"] / (1024 * 1024):.0f}MB.', 'error')
            return redirect(url_for('index'))
        elif file_type == 'audio' and file_size > app.config['MAX_AUDIO_SIZE']:
            flash(f'The audio file is too large. Maximum allowed size is {app.config["MAX_AUDIO_SIZE"] / (1024 * 1024):.0f}MB.', 'error')
            return redirect(url_for('index'))
        
        if file and allowed_file(file.filename, file_type):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                # Process the file based on type
                if file_type == 'video':
                    # Convert video to audio first
                    audio_path = convert_video_to_audio(file_path)
                    result = process_file(audio_path, 'audio')
                else:
                    result = process_file(file_path, file_type)
                
                # Store results in session for PDF generation
                session['suggestions'] = result.get('suggestions', [])
                session['report_content'] = result.get('report', 'No report available')
                session['xenophobic_words'] = result.get('xenophobic_words', [])
                session['wordcloud_image'] = result.get('wordcloud_image')
                session['transcription'] = result.get('transcription', None)
                
                # Debug session values
                print(f"Session report_content length: {len(session.get('report_content', ''))}")
                print(f"Result report length: {len(result.get('report', ''))}")
                
                # Return the analysis results with the UI template
                return render_template('results.html', result=result)
                    
            except Exception as e:
                error_msg = str(e)
                print(f"Error processing file: {error_msg}")
                
                if "Could not extract audio" in error_msg:
                    flash("Could not extract audio from the video. Please ensure the video contains an audio track.", 'error')
                elif "too large" in error_msg.lower():
                    flash("The extracted audio is too large for processing. Please use a shorter video.", 'error')
                else:
                    flash(f"Error processing file: {error_msg}", 'error')
                
                return redirect(url_for('index'))
            
            finally:
                # Clean up files in finally block
                try:
                    if file_path and os.path.exists(file_path):
                        os.unlink(file_path)
                except Exception as cleanup_error:
                    print(f"Error cleaning up file {file_path}: {str(cleanup_error)}")
                
                try:
                    if audio_path and os.path.exists(audio_path):
                        os.unlink(audio_path)
                except Exception as cleanup_error:
                    print(f"Error cleaning up audio file {audio_path}: {str(cleanup_error)}")
        
        flash('Invalid file type', 'error')
        return redirect(url_for('index'))
        
    except Exception as e:
        # Handle any other unexpected errors
        error_msg = str(e)
        print(f"Unexpected error in upload_file: {error_msg}")
        
        # Clean up any files that might have been created
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
        except:
            pass
            
        try:
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)
        except:
            pass
        
        flash(f"An unexpected error occurred: {error_msg}", 'error')
        return redirect(url_for('index'))

def process_file(file_path, file_type):
    """Process the uploaded file and return analysis results"""
    # This is a placeholder for the actual processing logic
    # In a real implementation, this would call different processing functions
    # based on the file type (text, audio, video)
    
    if file_type == 'text':
        return analyze_text(file_path)
    elif file_type == 'audio':
        return analyze_audio(file_path)
    elif file_type == 'video':
        return analyze_video(file_path)
    
    return {
        'toxicity_level': 'Unknown',
        'suggestions': ['Unable to process this file type'],
        'report': 'No analysis available for this file type'
    }

def analyze_text(file_path):
    """Analyze text content using OpenAI API"""
    try:
        # Check if OpenAI API key is set
        if not openai_api_key:
            return {
                'toxicity_level': 'Error',
                'suggestions': ['OpenAI API key is not set'],
                'report': 'Error: Please set your OpenAI API key in the .env file to analyze text content.'
            }
        
        file_extension = file_path.rsplit('.', 1)[1].lower()
        content = ""
        
        if file_extension == 'txt':
            # Handle plain text files
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        elif file_extension == 'docx':
            # Handle Word documents
            try:
                import docx
                doc = docx.Document(file_path)
                content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            except ImportError:
                return {
                    'toxicity_level': 'Error',
                    'suggestions': ['Python-docx library is not installed'],
                    'report': 'Error: To analyze Word documents, please install the python-docx library.'
                }
        elif file_extension == 'doc':
            # Old Word format is more complex to handle
            return {
                'toxicity_level': 'Error',
                'suggestions': ['DOC format not supported'],
                'report': 'Error: The old .doc format is not supported. Please convert to .docx or .txt.'
            }
        elif file_extension == 'pdf':
            # Handle PDF files
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    content = ""
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        content += page.extract_text() + "\n"
            except ImportError:
                return {
                    'toxicity_level': 'Error',
                    'suggestions': ['PyPDF2 library is not installed'],
                    'report': 'Error: To analyze PDF documents, please install the PyPDF2 library.'
                }
        else:
            return {
                'toxicity_level': 'Error',
                'suggestions': ['Unsupported file format'],
                'report': f'Error: The file format .{file_extension} is not supported for text analysis.'
            }
        
        # Check if content was successfully extracted
        if not content.strip():
            return {
                'toxicity_level': 'Error',
                'suggestions': ['No text content found'],
                'report': 'Error: No text content could be extracted from the file.'
            }
        
        # Limit content length for API call (OpenAI has token limits)
        max_content_length = 15000  # Approximately 4000 tokens
        if len(content) > max_content_length:
            analyzed_content = content[:max_content_length]
            content_truncated = True
        else:
            analyzed_content = content
            content_truncated = False
        
        # Analyze the text content using GPT
        analysis_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing media content for ethical reporting on topics related to refugees, migrants, and other forcibly displaced populations. Your task is to analyze the text content for xenophobic language, misinformation, and harmful content. Do not use bold formatting, markdown formatting, or any special text formatting in your response."},
                {"role": "user", "content": f"Analyze the following text content for xenophobic language, misinformation, and harmful content. Provide: 1) A toxicity level (Low, Medium, High, or Very High), 2) Specific suggestions for improvement, 3) A comprehensive analysis report, and 4) A list of potentially xenophobic or problematic words/phrases found in the text. Format the list of xenophobic words as JSON in this format: {{\"xenophobic_words\": [\"word1\", \"word2\", ...]}}. Do not use any bold text or markdown formatting in your response.\n\nContent: {analyzed_content}"}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        # Extract the analysis from the response
        analysis_text = analysis_response.choices[0].message.content
        
        # Parse the analysis to extract toxicity level, suggestions, and report
        toxicity_level = "Medium"  # Default
        suggestions = []
        report = ""
        
        # Simple parsing of the GPT response
        if "toxicity level" in analysis_text.lower():
            for level in ["low", "medium", "high", "very high"]:
                if level in analysis_text.lower():
                    toxicity_level = level.capitalize()
                    break
        
        # Extract suggestions (look for numbered or bulleted lists)
        suggestion_lines = []
        in_suggestions = False
        for line in analysis_text.split('\n'):
            line = line.strip()
            if "suggestions" in line.lower() or "improvements" in line.lower():
                in_suggestions = True
                continue
            if in_suggestions and line and (line[0].isdigit() or line[0] in ['•', '-', '*']):
                # Clean up the line (remove leading numbers, bullets, etc.)
                clean_line = line
                while clean_line and not clean_line[0].isalpha():
                    clean_line = clean_line[1:].strip()
                suggestion_lines.append(clean_line)
            elif in_suggestions and line and ("report" in line.lower() or "analysis" in line.lower() or "xenophobic" in line.lower() or "problematic" in line.lower()):
                in_suggestions = False
        
        # Filter out any remaining headings that might have been included
        filtered_suggestions = []
        for suggestion in suggestion_lines:
            if not any(heading in suggestion.lower() for heading in ["comprehensive analysis report", "list of potentially xenophobic", "problematic words"]):
                # Clean up any double asterisks in the suggestion
                suggestion = cleanup_suggestion_text(suggestion)
                filtered_suggestions.append(suggestion)
        
        if filtered_suggestions:
            suggestions = filtered_suggestions
        else:
            # Fallback: just extract some reasonable suggestions
            suggestions = [
                "Consider using more inclusive language",
                "Provide more context when discussing sensitive topics",
                "Avoid generalizations about groups of people"
            ]
        
        # Extract report (everything after "report" or "analysis")
        report_parts = []
        in_report = False
        for line in analysis_text.split('\n'):
            if "report" in line.lower() or "analysis" in line.lower():
                in_report = True
                continue
            if in_report and "xenophobic_words" not in line.lower():
                report_parts.append(line)
            elif "xenophobic_words" in line.lower():
                in_report = False
        
        if report_parts:
            report = "\n".join(report_parts)
        else:
            # Use the whole analysis as the report if we couldn't parse it
            report = analysis_text
        
        # Clean up the report content
        report = cleanup_report_content(report)
        
        # Extract xenophobic words from the JSON in the response
        xenophobic_words = []
        
        # Look for JSON pattern in the analysis_text
        json_match = re.search(r'\{[\s\S]*?"xenophobic_words"[\s\S]*?\}', analysis_text)
        if json_match:
            try:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                xenophobic_words = data.get("xenophobic_words", [])
            except json.JSONDecodeError:
                # If parsing fails, try to extract words from any section labeled as xenophobic words
                xenophobic_section = ""
                in_xenophobic = False
                for line in analysis_text.split('\n'):
                    if "xenophobic words" in line.lower() or "problematic words" in line.lower():
                        in_xenophobic = True
                        continue
                    if in_xenophobic and line.strip():
                        xenophobic_section += line + " "
                
                # Extract words from this section
                if xenophobic_section:
                    # Remove bullets, numbers, etc.
                    cleaned = re.sub(r'^\s*[\d\*\-•]+\s*', '', xenophobic_section)
                    # Split by common separators
                    words = re.split(r'[,;"\'\s]+', cleaned)
                    xenophobic_words = [w.strip() for w in words if w.strip()]
        
        # Generate word cloud if xenophobic words were found
        wordcloud_image = None
        if xenophobic_words:
            try:
                # Join words with space, but repeat problematic words according to their severity
                text = " ".join(xenophobic_words)
                
                # Generate the word cloud with multilingual support
                # Find system fonts that support multiple languages
                system = platform.system()
                
                # Default font path - will be overridden based on OS
                font_path = None
                
                # Set appropriate font based on operating system
                if system == 'Darwin':  # macOS
                    possible_fonts = [
                        '/System/Library/Fonts/ArialUnicodeMS.ttf',  # Best for multilingual support
                        '/System/Library/Fonts/STHeiti Light.ttc',  # Chinese
                        '/Library/Fonts/Arial Unicode.ttf',         # Alternative location
                        '/System/Library/Fonts/AppleGothic.ttf',    # Korean + some Chinese
                        '/System/Library/Fonts/Helvetica.ttc',      # Common fallback
                        # Arabic-specific fonts
                        '/Library/Fonts/Damascus.ttc',              # Good for Arabic
                        '/System/Library/Fonts/GeezaPro.ttc',       # macOS Arabic font
                    ]
                elif system == 'Windows':
                    possible_fonts = [
                        'C:\\Windows\\Fonts\\arialuni.ttf',         # Arial Unicode - best
                        'C:\\Windows\\Fonts\\arial.ttf',
                        'C:\\Windows\\Fonts\\seguisym.ttf',         # Segoe UI Symbol
                        'C:\\Windows\\Fonts\\micross.ttf',          # Microsoft Sans Serif
                        # Arabic-specific fonts
                        'C:\\Windows\\Fonts\\arabtype.ttf',         # Arabic Typesetting
                        'C:\\Windows\\Fonts\\meiryo.ttc',           # Good multilingual support
                        'C:\\Windows\\Fonts\\segoeui.ttf',          # Segoe UI
                    ]
                else:  # Linux and others
                    possible_fonts = [
                        '/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf',  # Arabic specific
                        '/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf', # Arabic specific
                        '/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf',
                        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                        '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
                    ]
                
                # Find the first available font
                for font in possible_fonts:
                    if os.path.exists(font):
                        font_path = font
                        print(f"Using font: {font}")
                        break
                
                # If no font found, try a more extensive search for fonts with Arabic support
                if not font_path:
                    print("No suitable font found in standard locations, searching for any Arabic-compatible font...")
                    try:
                        from matplotlib.font_manager import FontManager
                        
                        # Get all fonts matplotlib knows about
                        font_manager = FontManager()
                        font_list = font_manager.ttflist
                        
                        # Try to find any Arabic-compatible font
                        for font in font_list:
                            try:
                                # Check if font might support Arabic (this is a heuristic)
                                if any(name in font.name.lower() for name in ['arabic', 'naskh', 'kufi', 'unicode', 'noto']):
                                    font_path = font.fname
                                    print(f"Found Arabic-compatible font: {font.name} at {font_path}")
                                    break
                            except:
                                continue
                    except Exception as e:
                        print(f"Error during font search: {e}")
                
                # Create custom preprocessor for Arabic text
                def arabic_preprocessor(text):
                    # Join words with space
                    if isinstance(text, list):
                        text = " ".join(text)
                    return text
                
                # Improved regex pattern for Arabic word boundaries
                # This regex handles Arabic characters, Latin characters, and numbers
                word_pattern = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\u1EE00-\u1EEFF\w]+'
                
                # Configure WordCloud with better multilingual support
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    colormap='viridis',
                    contour_width=1, 
                    contour_color='steelblue',
                    font_path=font_path,  # Add font support for multiple languages
                    regexp=word_pattern,   # Improved regex for Arabic and other scripts
                    collocations=False,   # Avoid duplicate word pairs
                    normalize_plurals=False,  # Better for multilingual text
                    include_numbers=False,
                    min_word_length=1,   # Include single-character words
                    max_words=100,
                    prefer_horizontal=0.9,  # Allow some vertical words
                ).generate_from_text(arabic_preprocessor(text))
                
                # Convert the image to a base64 string to embed in HTML
                img_buffer = BytesIO()
                wordcloud.to_image().save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                wordcloud_image = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            except Exception as wc_error:
                print(f"Error generating word cloud: {str(wc_error)}")
        
        # Add a note if content was truncated
        if content_truncated:
            report += "\n\nNote: The analyzed content was truncated due to length limitations. The analysis is based on the first portion of the document."
        
        return {
            'toxicity_level': toxicity_level,
            'suggestions': suggestions,
            'report': report,
            'wordcloud_image': wordcloud_image,
            'xenophobic_words': xenophobic_words
        }
    except Exception as e:
        return {
            'toxicity_level': 'Error',
            'suggestions': ['An error occurred during analysis'],
            'report': f'Error: {str(e)}'
        }

def analyze_audio(file_path):
    """Analyze audio content using OpenAI Audio API for transcription and GPT for analysis"""
    try:
        # Check if OpenAI API key is set
        if not openai_api_key:
            return {
                'toxicity_level': 'Error',
                'suggestions': ['OpenAI API key is not set'],
                'report': 'Error: Please set your OpenAI API key in the .env file to analyze audio content.'
            }
        
        # Get file extension
        file_extension = file_path.rsplit('.', 1)[1].lower()
        
        # Log file information
        print(f"Processing audio file: {file_path} (Format: {file_extension})")
        
        # Check file size (OpenAI has a 25MB limit)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        if file_size > 25:
            return {
                'toxicity_level': 'Error',
                'suggestions': ['File size too large'],
                'report': f'Error: The audio file is {file_size:.1f}MB, which exceeds the 25MB limit. Please compress or shorten your audio file.'
            }
        
        # For M4A files, we might need to convert to a more universally supported format
        temp_file_path = None
        try_conversion = False
        transcription_error = None
        
        # First try direct transcription
        try:
            with open(file_path, "rb") as audio_file:
                transcript_response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            transcribed_text = transcript_response
        except Exception as direct_error:
            print(f"Direct transcription failed: {str(direct_error)}")
            # If direct transcription fails, try conversion
            try_conversion = True
            transcription_error = direct_error
        
        # If direct transcription failed and it's an M4A file, try conversion
        if try_conversion and file_extension == 'm4a':
            try:
                from moviepy.editor import AudioFileClip
                import tempfile
                
                print("Converting M4A to WAV for better compatibility...")
                # Create a temporary WAV file
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_file_path = temp_file.name
                temp_file.close()
                
                # Convert M4A to WAV
                audio_clip = AudioFileClip(file_path)
                audio_clip.write_audiofile(temp_file_path, verbose=False, logger=None)
                audio_clip.close()
                
                # Try transcription with the converted file
                with open(temp_file_path, "rb") as audio_file:
                    transcript_response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text"
                    )
                transcribed_text = transcript_response
                
                print("Conversion and transcription successful")
            except Exception as conversion_error:
                # If conversion also fails, return error
                if temp_file_path and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                return {
                    'toxicity_level': 'Error',
                    'suggestions': ['Audio processing failed'],
                    'report': f'Error: Could not process the M4A file. Original error: {str(transcription_error)}. Conversion error: {str(conversion_error)}'
                }
        elif try_conversion:
            # If it's not an M4A file and direct transcription failed
            return {
                'toxicity_level': 'Error',
                'suggestions': ['Audio transcription failed'],
                'report': f'Error: Could not transcribe the audio file. {str(transcription_error)}'
            }
        
        # Clean up temporary file if it was created
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        
        # If transcription is empty or failed
        if not transcribed_text:
            return {
                'toxicity_level': 'Error',
                'suggestions': ['No speech detected in the audio'],
                'report': 'Error: The audio file could not be transcribed. Please ensure it contains clear speech.'
            }
        
        # Store the full transcription
        full_transcription = transcribed_text
        
        # Analyze the transcribed text using GPT
        analysis_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing media content for ethical reporting on topics related to refugees, migrants, and other forcibly displaced populations. Your task is to analyze the transcribed audio content for xenophobic language, misinformation, and harmful content. Do not use bold formatting, markdown formatting, or any special text formatting in your response."},
                {"role": "user", "content": f"Analyze the following transcribed audio content for xenophobic language, misinformation, and harmful content. Provide: 1) A toxicity level (Low, Medium, High, or Very High), 2) Specific suggestions for improvement, and 3) A comprehensive analysis report. Do not use any bold text or markdown formatting in your response.\n\nTranscribed content: {transcribed_text}"}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        # Extract the analysis from the response
        analysis_text = analysis_response.choices[0].message.content
        
        # Parse the analysis to extract toxicity level, suggestions, and report
        toxicity_level = "Medium"  # Default
        suggestions = []
        report = ""
        
        # Simple parsing of the GPT response
        if "toxicity level" in analysis_text.lower():
            for level in ["low", "medium", "high", "very high"]:
                if level in analysis_text.lower():
                    toxicity_level = level.capitalize()
                    break
        
        # Extract suggestions (look for numbered or bulleted lists)
        suggestion_lines = []
        in_suggestions = False
        for line in analysis_text.split('\n'):
            line = line.strip()
            if "suggestions" in line.lower() or "improvements" in line.lower():
                in_suggestions = True
                continue
            if in_suggestions and line and (line[0].isdigit() or line[0] in ['•', '-', '*']):
                # Clean up the line (remove leading numbers, bullets, etc.)
                clean_line = line
                while clean_line and not clean_line[0].isalpha():
                    clean_line = clean_line[1:].strip()
                suggestion_lines.append(clean_line)
            elif in_suggestions and line and ("report" in line.lower() or "analysis" in line.lower() or "xenophobic" in line.lower() or "problematic" in line.lower()):
                in_suggestions = False
        
        # Filter out any remaining headings that might have been included
        filtered_suggestions = []
        for suggestion in suggestion_lines:
            if not any(heading in suggestion.lower() for heading in ["comprehensive analysis report", "list of potentially xenophobic", "problematic words"]):
                # Clean up any double asterisks in the suggestion
                suggestion = cleanup_suggestion_text(suggestion)
                filtered_suggestions.append(suggestion)
        
        if filtered_suggestions:
            suggestions = filtered_suggestions
        else:
            # Fallback: just extract some reasonable suggestions
            suggestions = [
                "Consider using more inclusive language",
                "Provide more context when discussing sensitive topics",
                "Avoid generalizations about groups of people"
            ]
        
        # Extract report (everything after "report" or "analysis")
        report_parts = []
        in_report = False
        for line in analysis_text.split('\n'):
            if "report" in line.lower() or "analysis" in line.lower():
                in_report = True
                continue
            if in_report:
                report_parts.append(line)
        
        if report_parts:
            report = "\n".join(report_parts)
        else:
            # Use the whole analysis as the report if we couldn't parse it
            report = analysis_text
        
        # Clean up the report content
        report = cleanup_report_content(report)
        
        return {
            'toxicity_level': toxicity_level,
            'suggestions': suggestions,
            'report': report,
            'transcription': full_transcription
        }
    
    except Exception as e:
        return {
            'toxicity_level': 'Error',
            'suggestions': ['An error occurred during audio analysis'],
            'report': f'Error: {str(e)}'
        }

def analyze_video(file_path):
    """Analyze video content by extracting audio and sending to OpenAI Audio API"""
    try:
        # Check if OpenAI API key is set
        if not openai_api_key:
            return {
                'toxicity_level': 'Error',
                'suggestions': ['OpenAI API key is not set'],
                'report': 'Error: Please set your OpenAI API key in the .env file to analyze video content.'
            }
        
        # Get file extension and log information
        file_extension = file_path.rsplit('.', 1)[1].lower()
        print(f"Processing video file: {file_path} (Format: {file_extension})")
        
        # Check file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        audio_api_limit_mb = 25
        
        if file_size > 100:  # General app limit
            return {
                'toxicity_level': 'Error',
                'suggestions': ['File size too large'],
                'report': f'Error: The video file is {file_size:.1f}MB, which exceeds our 100MB limit. Please compress or shorten your video.'
            }
        
        # Always extract audio first
        print("Extracting audio from video...")
        temp_audio_path = None  # Initialize here so it's defined for all exception handlers
        try:
            # Create a temporary audio file
            import tempfile
            temp_audio_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            temp_audio_path = temp_audio_file.name
            temp_audio_file.close()
            
            # Try with moviepy first
            audio_extracted = False
            try:
                print("Attempting audio extraction with moviepy...")
                from moviepy.editor import VideoFileClip
                
                # Print debugging info
                print(f"File path: {file_path}")
                print(f"File exists: {os.path.exists(file_path)}")
                
                # Use conservative settings for larger files
                video_clip = VideoFileClip(
                    file_path, 
                    audio_buffersize=10000,  # Smaller buffer
                    verbose=False,
                    target_resolution=(360, 640)  # Lower resolution
                )
                
                if video_clip.audio:
                    print(f"Video loaded successfully. Duration: {video_clip.duration}s")
                    print("Extracting audio track...")
                    
                    # Use optimized settings for audio extraction
                    video_clip.audio.write_audiofile(
                        temp_audio_path,
                        verbose=False,
                        logger=None,
                        bitrate='64k',  # Lower bitrate
                        ffmpeg_params=[
                            '-ac', '1',  # Mono
                            '-ar', '16000',  # 16kHz sample rate
                            '-q:a', '9'  # Lowest quality
                        ]
                    )
                    
                    video_clip.close()
                    audio_extracted = True
                    print("Audio extraction successful with moviepy")
                else:
                    video_clip.close()
                    print("No audio track found in video")
                    return {
                        'toxicity_level': 'Error',
                        'suggestions': ['No audio track found in video'],
                        'report': 'Error: No audio track could be found in the video file. Please ensure your video contains speech content.'
                    }
            except Exception as moviepy_error:
                print(f"MoviePy audio extraction error: {str(moviepy_error)}")
                
                # If moviepy failed, try with imageio/ffmpeg
                if not audio_extracted:
                    try:
                        print("Trying alternative method with imageio/ffmpeg...")
                        import imageio.v3 as iio
                        from imageio_ffmpeg import get_ffmpeg_exe
                        
                        ffmpeg_exe = get_ffmpeg_exe()
                        print(f"Found bundled FFmpeg: {ffmpeg_exe}")
                        
                        # Extract with optimized ffmpeg settings
                        cmd = [
                            ffmpeg_exe, '-y',
                            '-i', file_path,
                            '-vn',  # No video
                            '-acodec', 'libmp3lame',
                            '-ac', '1',  # Mono
                            '-ar', '16000',  # 16kHz sample rate
                            '-ab', '64k',  # 64kbps bitrate
                            '-q:a', '9',  # Lowest quality
                            temp_audio_path
                        ]
                        
                        import subprocess
                        process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                        stdout, stderr = process.communicate()
                        
                        if process.returncode == 0 and os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                            audio_extracted = True
                            print("Audio extraction successful with ffmpeg")
                        else:
                            print(f"FFmpeg error: {stderr.decode('utf-8', errors='ignore')}")
                    except Exception as ffmpeg_error:
                        print(f"FFmpeg extraction failed: {str(ffmpeg_error)}")
            
            # Verify audio extraction
            if not audio_extracted or not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
                raise Exception("Could not extract audio from video using any available method")
            
            # Check audio file size
            audio_size = os.path.getsize(temp_audio_path) / (1024 * 1024)  # Size in MB
            print(f"Extracted audio size: {audio_size:.1f}MB")
            
            # Compress if still too large
            if audio_size > audio_api_limit_mb:
                print("Extracted audio is too large, attempting additional compression...")
                try:
                    from pydub import AudioSegment
                    
                    # Load the audio
                    audio = AudioSegment.from_file(temp_audio_path)
                    
                    # Create compressed file path
                    compressed_audio_path = temp_audio_path + '.compressed.mp3'
                    
                    # Apply aggressive compression
                    audio = audio.set_channels(1)  # Mono
                    audio = audio.set_frame_rate(16000)  # 16kHz
                    
                    # Export with minimal quality
                    audio.export(
                        compressed_audio_path,
                        format="mp3",
                        bitrate="48k",  # Even lower bitrate
                        parameters=["-q:a", "9"]  # Lowest quality
                    )
                    
                    # Replace original with compressed
                    os.unlink(temp_audio_path)
                    os.rename(compressed_audio_path, temp_audio_path)
                    
                    # Check new size
                    audio_size = os.path.getsize(temp_audio_path) / (1024 * 1024)
                    print(f"Compressed audio size: {audio_size:.1f}MB")
                except Exception as comp_error:
                    print(f"Compression error: {str(comp_error)}")
            
            # Final size check
            if audio_size > audio_api_limit_mb:
                os.unlink(temp_audio_path)
                return {
                    'toxicity_level': 'Error',
                    'suggestions': [
                        'Extracted audio still exceeds API limit',
                        'Please use a shorter video or compress it further',
                        f'Extracted audio was {audio_size:.1f}MB, limit is 25MB'
                    ],
                    'report': f'Error: Even after extracting and compressing the audio, the file is {audio_size:.1f}MB, which exceeds the 25MB API limit. Please use a shorter video or further compress your content.'
                }
            
            # Transcribe the audio
            print("Transcribing extracted audio...")
            try:
                with open(temp_audio_path, "rb") as audio_file:
                    transcript_response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text"
                    )
                transcribed_text = transcript_response
                print("Transcription successful")
            except Exception as transcribe_error:
                # Clean up and return error
                if temp_audio_path and os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                raise Exception(f"Transcription failed: {str(transcribe_error)}")
            
            # Clean up temp file
            try:
                if temp_audio_path and os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
            except Exception as cleanup_error:
                print(f"Warning: Could not delete temporary file: {str(cleanup_error)}")
            
            # Check transcription result
            if not transcribed_text:
                return {
                    'toxicity_level': 'Error',
                    'suggestions': ['No speech detected in the video'],
                    'report': 'Error: The video file could not be transcribed. Please ensure it contains clear speech.'
                }
            
            # Store the full transcription
            full_transcription = transcribed_text
            
            # Analyze transcribed text
            analysis_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert in analyzing media content for ethical reporting on topics related to refugees, migrants, and other forcibly displaced populations. Your task is to analyze the transcribed video content for xenophobic language, misinformation, and harmful content. Do not use bold formatting, markdown formatting, or any special text formatting in your response."},
                    {"role": "user", "content": f"Analyze the following transcribed video content for xenophobic language, misinformation, and harmful content. Provide: 1) A toxicity level (Low, Medium, High, or Very High), 2) Specific suggestions for improvement, and 3) A comprehensive analysis report. Do not use any bold text or markdown formatting in your response.\n\nTranscribed content: {transcribed_text}"}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Extract analysis
            analysis_text = analysis_response.choices[0].message.content
            print("Content analysis complete")
            
            # Parse analysis
            toxicity_level = "Medium"  # Default
            suggestions = []
            report = ""
            
            # Extract toxicity level
            if "toxicity level" in analysis_text.lower():
                for level in ["low", "medium", "high", "very high"]:
                    if level in analysis_text.lower():
                        toxicity_level = level.capitalize()
                        break
            
            # Extract suggestions
            suggestion_lines = []
            in_suggestions = False
            for line in analysis_text.split('\n'):
                line = line.strip()
                if "suggestions" in line.lower() or "improvements" in line.lower():
                    in_suggestions = True
                    continue
                if in_suggestions and line and (line[0].isdigit() or line[0] in ['•', '-', '*']):
                    clean_line = line
                    while clean_line and not clean_line[0].isalpha():
                        clean_line = clean_line[1:].strip()
                    suggestion_lines.append(clean_line)
                elif in_suggestions and line and ("report" in line.lower() or "analysis" in line.lower() or "xenophobic" in line.lower() or "problematic" in line.lower()):
                    in_suggestions = False
            
            # Filter out any remaining headings that might have been included
            filtered_suggestions = []
            for suggestion in suggestion_lines:
                if not any(heading in suggestion.lower() for heading in ["comprehensive analysis report", "list of potentially xenophobic", "problematic words"]):
                    # Clean up any double asterisks in the suggestion
                    suggestion = cleanup_suggestion_text(suggestion)
                    filtered_suggestions.append(suggestion)
            
            if filtered_suggestions:
                suggestions = filtered_suggestions
            else:
                suggestions = [
                    "Ensure diverse representation in visual content",
                    "Be mindful of stereotypical portrayals",
                    "Consider the tone used when discussing sensitive topics"
                ]
            
            # Extract report
            report_parts = []
            in_report = False
            for line in analysis_text.split('\n'):
                if "report" in line.lower() or "analysis" in line.lower():
                    in_report = True
                    continue
                if in_report:
                    report_parts.append(line)
            
            if report_parts:
                report = "\n".join(report_parts)
            else:
                report = analysis_text
            
            # Build final report with note about visual elements
            report += "\n\nNote: This analysis is based on the audio content of the video. A full analysis would also include evaluation of visual elements, which requires human review."
            
            # Clean up the report content
            report = cleanup_report_content(report)
            
            return {
                'toxicity_level': toxicity_level,
                'suggestions': suggestions,
                'report': report,
                'transcription': full_transcription
            }
            
        except Exception as processing_error:
            error_message = str(processing_error)
            print(f"Video processing error: {error_message}")
            
            suggestions = [
                'Video processing failed',
                'Please try a different video format or compress your video',
                f'Your file is {file_size:.1f}MB'
            ]
            
            if "Could not extract audio" in error_message:
                suggestions.append("Try converting your video to MP4 format with a tool like HandBrake")
            elif "not installed" in error_message.lower() or "missing" in error_message.lower():
                suggestions.append("Server is missing required libraries - contact administrator")
            elif "memory" in error_message.lower():
                suggestions.append("Video is too complex - try a shorter or simpler video file")
            
            # Clean up temp file if it exists
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except Exception:
                    pass
            
            return {
                'toxicity_level': 'Error',
                'suggestions': suggestions,
                'report': f'Error: Could not process your {file_size:.1f}MB video. Please try using a different video format (MP4 is recommended) or compress your video to a smaller size.\n\nTechnical details: {error_message}'
            }
    
    except Exception as outer_error:
        print(f"Unexpected error in analyze_video: {str(outer_error)}")
        return {
            'toxicity_level': 'Error',
            'suggestions': ['An error occurred during video analysis'],
            'report': f'Error: {str(outer_error)}'
        }

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

@app.route('/download_report_pdf', methods=['GET', 'POST'])
def download_report_pdf():
    """Generate and download a PDF report"""
    try:
        # Check if this is a POST request with form data
        if request.method == 'POST':
            # Get data from form
            toxicity_level = request.form.get('toxicity_level', 'Unknown')
            report_content = request.form.get('report_content', '')
            
            # Still use session for other data
            suggestions = session.get('suggestions', [])
            xenophobic_words = session.get('xenophobic_words', [])
            wordcloud_image = session.get('wordcloud_image')
        else:
            # Fallback to session data for GET requests
            toxicity_level = request.args.get('toxicity_level', 'Unknown')
            suggestions = session.get('suggestions', [])
            report_content = session.get('report_content', 'No report available')
            xenophobic_words = session.get('xenophobic_words', [])
            wordcloud_image = session.get('wordcloud_image')
        
        transcription = session.get('transcription', None)
        
        # Debug info
        print("=== PDF Generation Debug ===")
        print(f"Request method: {request.method}")
        print(f"Toxicity level: {toxicity_level}")
        print(f"Suggestions: {suggestions}")
        print(f"Report content length: {len(report_content)}")
        print(f"Report content starts with: {report_content[:100]}...")
        print(f"Number of xenophobic words: {len(xenophobic_words) if xenophobic_words else 0}")
        print(f"Has wordcloud: {wordcloud_image is not None}")
        print(f"Session keys: {list(session.keys())}")
        print("===========================")
        
        # Provide default suggestions if none are available
        if not suggestions:
            suggestions = [
                "Consider using more inclusive language",
                "Provide more context when discussing sensitive topics",
                "Avoid generalizations about groups of people"
            ]
        
        # Provide default report content if none is available or contains error message
        if not report_content or report_content == 'No report available' or report_content.startswith('Error:'):
            report_content = "The analysis could not generate a detailed report for this content. Please review the provided suggestions and consider running the analysis again with different content."
        else:
            # Clean up the report content one more time
            report_content = cleanup_report_content(report_content)
        
        # Import reportlab components
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
        from io import BytesIO
        
        # Create a BytesIO buffer to receive PDF data
        buffer = BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Create custom styles - check if they exist first
        if 'CustomTitle' not in styles:
            styles.add(ParagraphStyle(
                name='CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                alignment=TA_CENTER,
                spaceAfter=20
            ))
        
        if 'CustomSubtitle' not in styles:
            styles.add(ParagraphStyle(
                name='CustomSubtitle',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=12
            ))
        
        if 'Normal_Justify' not in styles:
            styles.add(ParagraphStyle(
                name='Normal_Justify',
                parent=styles['Normal'],
                alignment=TA_JUSTIFY
            ))
        
        # Build the PDF content
        content = []
        
        # Title
        content.append(Paragraph("VERIMEDIA ANALYSIS REPORT", styles['CustomTitle']))
        content.append(Spacer(1, 12))
        
        # Toxicity Level
        content.append(Paragraph(f"TOXICITY LEVEL: {toxicity_level}", styles['CustomSubtitle']))
        content.append(Spacer(1, 12))
        
        # Improvement Suggestions
        content.append(Paragraph("IMPROVEMENT SUGGESTIONS:", styles['CustomSubtitle']))
        
        if suggestions:
            suggestion_items = []
            for suggestion in suggestions:
                suggestion_items.append(ListItem(Paragraph(suggestion, styles['Normal'])))
            
            suggestion_list = ListFlowable(
                suggestion_items,
                bulletType='bullet',
                leftIndent=20,
                bulletFontSize=8,
                bulletOffsetY=2
            )
            content.append(suggestion_list)
        else:
            content.append(Paragraph("No suggestions available.", styles['Normal']))
        
        content.append(Spacer(1, 12))
        
        # Word Cloud of Xenophobic Words
        if wordcloud_image:
            content.append(Paragraph("XENOPHOBIC WORDS VISUALIZATION:", styles['CustomSubtitle']))
            
            # Decode the base64 image
            img_data = base64.b64decode(wordcloud_image)
            img_bio = BytesIO(img_data)
            
            # Add the image to the PDF
            img = Image(img_bio, width=450, height=225)
            content.append(img)
            
            # If we have xenophobic words, list them
            if xenophobic_words:
                content.append(Spacer(1, 12))
                content.append(Paragraph("Identified Xenophobic Words:", styles['Normal']))
                
                # Create a comma-separated list
                words_text = ", ".join(xenophobic_words)
                content.append(Paragraph(words_text, styles['Normal']))
            
            content.append(Spacer(1, 12))
        
        # Comprehensive Report (removed transcription section)
        content.append(Paragraph("COMPREHENSIVE REPORT:", styles['CustomSubtitle']))
        content.append(Paragraph(report_content, styles['Normal_Justify']))
        
        content.append(Spacer(1, 20))
        
        # Footer
        import datetime
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content.append(Paragraph(f"Generated by VeriMedia - UNICC", styles['Normal']))
        content.append(Paragraph(f"Date: {current_date}", styles['Normal']))
        
        # Build the PDF
        doc.build(content)
        
        # Get the value from the BytesIO buffer
        pdf_data = buffer.getvalue()
        buffer.close()
        
        # Create response
        response = make_response(pdf_data)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=VeriMedia_Analysis_Report.pdf'
        
        return response
    
    except Exception as e:
        flash(f"Error generating PDF: {str(e)}", "error")
        return redirect(url_for('index'))

def convert_video_format(input_path, target_format='mp4'):
    """
    Convert a video file to the target format for compatibility with Whisper API.
    Returns the path to the converted file.
    """
    print(f"Converting video to {target_format}: {input_path}")
    
    # Create output path with the target extension
    output_path = os.path.splitext(input_path)[0] + f'_converted.{target_format}'
    
    try:
        # Try using moviepy for conversion
        try:
            from moviepy.editor import VideoFileClip
            import tempfile
            
            print("Using MoviePy for conversion")
            
            # Create a temporary file for the output
            temp_dir = tempfile.mkdtemp()
            temp_output_path = os.path.join(temp_dir, f'converted_video.{target_format}')
            
            # Load the video file
            video = VideoFileClip(input_path)
            
            # Write to target format with specific settings for compatibility
            video.write_videofile(
                temp_output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=os.path.join(temp_dir, 'temp-audio.m4a'),
                remove_temp=True,
                fps=24,  # Standard frame rate
                preset='ultrafast',  # Faster encoding
                threads=4,  # Use multiple threads
                logger=None  # Suppress output
            )
            video.close()
            
            # Copy the converted file to the output path
            import shutil
            shutil.copy2(temp_output_path, output_path)
            
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
            except:
                pass  # Ignore cleanup errors
            
            print("MoviePy conversion successful")
            return output_path
        except Exception as e:
            print(f"MoviePy conversion error: {str(e)}")
            
        # If MoviePy fails, try extracting just the audio as a fallback
        try:
            print("Attempting to extract audio only as fallback")
            from moviepy.editor import VideoFileClip
            
            # Create an audio output path
            audio_output_path = os.path.splitext(input_path)[0] + '_audio.mp3'
            
            # Extract audio only
            video = VideoFileClip(input_path)
            if video.audio:
                video.audio.write_audiofile(audio_output_path, logger=None)
                video.close()
                print("Audio extraction successful")
                return audio_output_path
            else:
                video.close()
                print("No audio track found in video")
                raise Exception("No audio track found in video")
        except Exception as e:
            print(f"Audio extraction error: {str(e)}")
        
        # If all methods fail
        raise Exception("All conversion methods failed")
        
    except Exception as e:
        print(f"Conversion error: {str(e)}")
        raise

def cleanup_report_content(report_text):
    """Remove unwanted sections from report content such as JSON blocks and certain headers"""
    # Remove any mention of "List of Potentially Xenophobic or Problematic Words/Phrases"
    if report_text:
        # Remove the xenophobic words list header and any JSON block that follows
        patterns = [
            # Match numbered xenophobic word lists with JSON codeblocks
            r'\d+\)\s*List of potentially xenophobic or problematic words/phrases found in the text:.*?```json.*',
            r'\d+\)\s*List of potentially xenophobic or problematic words/phrases.*?```json.*',
            r'\d+\)\s*List of [Xx]enophobic [Ww]ords:.*?```json.*',
            
            # Match non-numbered variants
            r'List of potentially xenophobic or problematic words/phrases found in the text:.*?```json.*',
            r'List of potentially xenophobic or problematic words/phrases:.*?```json.*',
            r'List of [Xx]enophobic [Ww]ords:.*?```json.*',
            
            # Match partial JSON blocks
            r'\d+\)\s*List of potentially xenophobic or problematic words/phrases.*?{.*',
            r'\d+\)\s*List of [Xx]enophobic [Ww]ords.*?{.*',
            
            # Match with asterisks
            r'\d+\)\s*\*\*List of potentially xenophobic or problematic words/phrases\*\*:.*?```json.*',
            r'\*\*List of potentially xenophobic or problematic words/phrases\*\*:.*?```json.*',
            r'\*\*List of [Xx]enophobic [Ww]ords\*\*:.*?```json.*',
            
            # Cleanup comprehensive report headers
            r'\d+\)\s*\*\*Comprehensive Analysis Report\*\*:.*?$',
            r'\*\*Comprehensive Analysis Report\*\*:.*?$'
        ]
        
        # Apply each pattern with DOTALL to match across lines
        for pattern in patterns:
            report_text = re.sub(pattern, '', report_text, flags=re.DOTALL | re.IGNORECASE)
        
        # Additional cleanup: remove trailing colons and metadata markers
        report_text = re.sub(r'\n\d+\)\s*\*\*.*?\*\*:.*?$', '', report_text, flags=re.MULTILINE)
        
        # Remove standalone numbered markers (like "4)") at the end of lines or the entire text
        report_text = re.sub(r'\n\s*\d+\)\s*$', '', report_text)
        report_text = re.sub(r'\n\s*\d+\)\s*[^\n]*?xenophobic.*?$', '', report_text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Also remove any standalone number marker at the very end of the text
        report_text = re.sub(r'\s*\d+\)\s*$', '', report_text)
        
        # Clean up multiple newlines
        report_text = re.sub(r'\n{3,}', '\n\n', report_text)
        
        # Remove any trailing JSON opening braces
        report_text = re.sub(r'\s*```json\s*\{\s*$', '', report_text)
        report_text = re.sub(r'\s*\{\s*$', '', report_text)
        
    return report_text.strip()

def cleanup_suggestion_text(suggestion):
    """Fix formatting in suggestions by removing asterisks and fixing punctuation"""
    if suggestion:
        # Remove asterisks entirely
        suggestion = suggestion.replace('**', '')
        suggestion = suggestion.replace('*', '')
        
        # Fix common punctuation issues after removing asterisks
        
        # Find the first colon if it exists
        colon_index = suggestion.find(':')
        if colon_index > 0:
            # Get the parts before and after the colon
            heading = suggestion[:colon_index].strip()
            content = suggestion[colon_index+1:].strip()
            
            # Clean up the heading (remove numbers, etc.)
            if heading and heading[0].isdigit():
                # Remove any leading numbers or punctuation from heading
                heading = ''.join(c for c in heading if not (c.isdigit() or c in '.)- '))
                heading = heading.strip()
            
            # Reconstruct with proper formatting
            if heading and content:
                suggestion = f"{heading}: {content}"
        
        # Fix multiple spaces
        suggestion = ' '.join(suggestion.split())
    
    return suggestion.strip()

if __name__ == '__main__':
    app.run(debug=True, port=5004) 