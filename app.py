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
from wordcloud import WordCloud, STOPWORDS
import base64
from io import BytesIO
import re
import json
import matplotlib.font_manager as fm

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

# Fine-tuned model ID for toxicity classification
# This will be used if available, otherwise fall back to default models
try:
    with open("finetune_model_id.txt", "r") as f:
        FINE_TUNED_MODEL_ID = f.read().strip()
    print(f"Using fine-tuned model: {FINE_TUNED_MODEL_ID}")
except FileNotFoundError:
    FINE_TUNED_MODEL_ID = None
    print("No fine-tuned model found. Using default models.")

def determine_toxicity_level(text_content, content_type="text"):
    """
    Use the fine-tuned model to determine toxicity level of content
    
    Args:
        text_content (str): The content to analyze (could be text, audio transcript, or video transcript)
        content_type (str): Type of content - "text", "audio", or "video"
        
    Returns:
        str: The toxicity level - "None", "Mild", "High", or "Max"
    """
    try:
        # If no fine-tuned model is available, return None and let the regular analysis handle it
        if not FINE_TUNED_MODEL_ID:
            return None
        
        # Set the system message based on content type
        if content_type == "text":
            system_message = "You are an expert analyzing text content for ethical standards. Classify the toxicity level."
        elif content_type == "audio":
            system_message = "You are an expert analyzing audio transcripts for ethical standards. Classify the toxicity level."
        else:  # video
            system_message = "You are an expert analyzing video transcripts for ethical standards. Classify the toxicity level."
        
        # Create the user message
        user_message = f"Analyze the following {content_type} content for harmful language, misinformation, and problematic content. Content: {text_content}"
        
        # Use the fine-tuned model to classify toxicity
        response = client.chat.completions.create(
            model=FINE_TUNED_MODEL_ID,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,  # Use low temperature for more deterministic results
            max_tokens=10     # We only need a short response for the classification
        )
        
        # Extract the toxicity level from the response
        toxicity_level = response.choices[0].message.content.strip()
        
        # Extract just the toxicity level if it's in a longer response
        if "toxicity level" in toxicity_level.lower():
            for word in toxicity_level.split():
                if word.lower() in ["none", "mild", "high", "max"]:
                    toxicity_level = word
                    break
        
        # Ensure it's one of our expected values
        valid_levels = ["None", "Mild", "High", "Max"]
        if toxicity_level.capitalize() in valid_levels:
            return toxicity_level.capitalize()
        else:
            print(f"Unexpected toxicity level from fine-tuned model: {toxicity_level}")
            return None
            
    except Exception as e:
        print(f"Error using fine-tuned model: {str(e)}")
        return None

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
            
        # First, try using the fine-tuned model for toxicity level
        toxicity_level = determine_toxicity_level(analyzed_content, "text")
        
        # Analyze the text content using GPT
        analysis_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing media content for ethical reporting on topics related to refugees, migrants, and other forcibly displaced populations. Your task is to analyze the text content for xenophobic language, misinformation, and harmful content. Do not use bold formatting, markdown formatting, or any special text formatting in your response."},
                {"role": "user", "content": f"Analyze the following text content for xenophobic language, misinformation, and harmful content. Provide: 1) A toxicity level (None, Mild, High, or Max), 2) Specific suggestions for improvement, 3) A comprehensive analysis report, and 4) A list of potentially xenophobic or problematic words/phrases found in the text. Format the list of xenophobic words as JSON in this format: {{\"xenophobic_words\": [\"word1\", \"word2\", ...]}}. Do not use any bold text or markdown formatting in your response.\n\nContent: {analyzed_content}"}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        # Extract the analysis from the response
        analysis_text = analysis_response.choices[0].message.content
        
        # Parse the analysis to extract toxicity level, suggestions, and report
        if not toxicity_level:  # If fine-tuned model didn't provide a level, extract it from the analysis
            toxicity_level = "Medium"  # Default
            
            # Simple parsing of the GPT response
            if "toxicity level" in analysis_text.lower():
                for level in ["none", "mild", "high", "max"]:
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
            # Default suggestions if none were extracted
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
                
                # Get a suitable font for Chinese characters
                chinese_font_path = get_chinese_font()
                
                # Generate the word cloud with appropriate font
                wordcloud_params = {
                    'width': 800, 
                    'height': 400,
                    'background_color': 'white',
                    'colormap': 'viridis',
                    'contour_width': 1,
                    'contour_color': 'steelblue'
                }
                
                # Add font_path only if a suitable font was found
                if chinese_font_path:
                    wordcloud_params['font_path'] = chinese_font_path
                
                wordcloud = WordCloud(**wordcloud_params).generate(text)
                
                # Convert the image to a base64 string to embed in HTML
                img_buffer = BytesIO()
                # Save directly to buffer without using plt
                wordcloud.to_image().save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                wordcloud_image = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            except Exception as wc_error:
                print(f"Error generating word cloud: {str(wc_error)}")
        
        # Add a note if content was truncated
        if content_truncated:
            report += "\n\nNote: The analyzed content was truncated due to length limitations. The analysis is based on the first portion of the document."
        
        # If toxicity level is 'none', customize the report
        if toxicity_level.lower() == 'none':
            # Replace detailed report with a simpler positive message for 'none' toxicity
            positive_report = f"Your content has been analyzed and found to contain no problematic or xenophobic language. The content appears to be ethical, balanced, and respectful. It presents information in a fair and accurate manner that follows best practices for responsible reporting."
            
            # Keep a short summary if available in the report
            if "summary" in report.lower() or "content" in report.lower():
                # Try to extract a summary section if it exists
                summary_lines = []
                in_summary = False
                for line in report.split('\n'):
                    if "summary" in line.lower() or "content summary" in line.lower():
                        in_summary = True
                        continue
                    if in_summary and line.strip() and "toxicity" not in line.lower() and "xenophobic" not in line.lower():
                        summary_lines.append(line)
                    if in_summary and line.strip() and ("analysis" in line.lower() or "report" in line.lower()):
                        break
                
                if summary_lines:
                    summary_text = " ".join(summary_lines)
                    positive_report = f"Content Summary: {summary_text}\n\n{positive_report}"
            
            # Replace the detailed report with our positive message
            report = positive_report
            
            # Replace suggestions with positive phrases for 'none' toxicity
            suggestions = [
                "Continue using inclusive and balanced language",
                "Maintain your ethical reporting approach",
                "Keep providing accurate and fair content"
            ]
        
        # Enhance suggestions based on toxicity level
        elif toxicity_level.lower() == 'mild':
            # Append customized context for mild toxicity
            report += "\n\nMild Toxicity Context: Content with mild toxicity typically contains subtle issues that can be improved with minor adjustments. While generally acceptable, addressing these minor concerns will enhance the ethical quality of your content."
            
            # Ensure suggestions are appropriate for mild toxicity
            if not suggestions or len(suggestions) < 3:
                suggestions = [
                    "Consider replacing certain terms with more inclusive alternatives",
                    "Add more context when discussing potentially sensitive topics",
                    "Review specific sections that may benefit from more balanced perspectives",
                    "Use more precise language when referring to groups of people"
                ]
            
        elif toxicity_level.lower() == 'high':
            # Append customized context for high toxicity
            report += "\n\nHigh Toxicity Context: Content with high toxicity contains significant issues that require substantial revision. The problematic elements identified may perpetuate harmful stereotypes or contain misinformation that should be addressed promptly."
            
            # Ensure suggestions are appropriate for high toxicity
            if not suggestions or len(suggestions) < 3:
                suggestions = [
                    "Thoroughly revise sections containing problematic language or stereotypes",
                    "Replace generalizations about groups with specific, fact-based statements",
                    "Consider consulting cultural sensitivity experts for content revision",
                    "Add substantial context and multiple perspectives to balance the narrative",
                    "Remove or completely rewrite passages containing xenophobic language"
                ]
                
        elif toxicity_level.lower() == 'max':
            # Append customized context for max toxicity
            report += "\n\nMaximum Toxicity Alert: Content with maximum toxicity requires immediate and comprehensive revision. The analysis identifies serious ethical concerns that may cause harm if published or distributed in its current form."
            
            # Ensure suggestions are appropriate for max toxicity
            if not suggestions or len(suggestions) < 3:
                suggestions = [
                    "Consider completely rewriting the content with ethical reporting guidelines",
                    "Consult with subject matter experts on cultural sensitivity and accurate representation",
                    "Implement a comprehensive editorial review process before publication",
                    "Ensure claims are supported by credible, well-researched sources",
                    "Address the xenophobic elements identified in the analysis as a priority",
                    "Review your organization's ethical guidelines and ensure compliance"
                ]
        
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
        
        # First, try using the fine-tuned model for toxicity level
        toxicity_level = determine_toxicity_level(transcribed_text, "audio")
        
        # Analyze the transcribed text using GPT
        analysis_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing media content for ethical reporting on topics related to refugees, migrants, and other forcibly displaced populations. Your task is to analyze the transcribed audio content for xenophobic language, misinformation, and harmful content. Do not use bold formatting, markdown formatting, or any special text formatting in your response."},
                {"role": "user", "content": f"Analyze the following transcribed audio content for xenophobic language, misinformation, and harmful content. Provide: 1) A toxicity level (None, Mild, High, or Max), 2) Specific suggestions for improvement, and 3) A comprehensive analysis report. Do not use any bold text or markdown formatting in your response.\n\nTranscribed content: {transcribed_text}"}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        # Extract the analysis from the response
        analysis_text = analysis_response.choices[0].message.content
        
        # Parse the analysis to extract toxicity level, suggestions, and report
        if not toxicity_level:  # If fine-tuned model didn't provide a level, extract it from the analysis
            toxicity_level = "Medium"  # Default
            
            # Extract toxicity level
            if "toxicity level" in analysis_text.lower():
                for level in ["none", "mild", "high", "max"]:
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
        
        # If toxicity level is 'none', customize the report and suggestions
        if toxicity_level.lower() == 'none':
            # Replace detailed report with a simpler positive message for 'none' toxicity
            positive_report = f"Your audio content has been analyzed and found to contain no problematic or xenophobic language. The content appears to be ethical, balanced, and respectful. It presents information in a fair and accurate manner that follows best practices for responsible reporting."
            
            # Keep a short summary if available in the report
            if "summary" in analysis_text.lower() or "content" in analysis_text.lower():
                # Try to extract a summary section if it exists
                summary_lines = []
                in_summary = False
                for line in analysis_text.split('\n'):
                    if "summary" in line.lower() or "content summary" in line.lower():
                        in_summary = True
                        continue
                    if in_summary and line.strip() and "toxicity" not in line.lower() and "xenophobic" not in line.lower():
                        summary_lines.append(line)
                    if in_summary and line.strip() and ("analysis" in line.lower() or "report" in line.lower()):
                        break
                
                if summary_lines:
                    summary_text = " ".join(summary_lines)
                    positive_report = f"Content Summary: {summary_text}\n\n{positive_report}"
            
            # Replace the detailed report with our positive message
            report = positive_report
            
            # Replace suggestions with positive phrases for 'none' toxicity
            suggestions = [
                "Continue using inclusive and balanced language in audio content",
                "Maintain your ethical reporting approach in spoken media",
                "Keep providing accurate and fair audio content"
            ]
            
            return {
                'toxicity_level': toxicity_level,
                'suggestions': suggestions,
                'report': report,
                'transcription': full_transcription
            }
        
        # Enhance suggestions based on toxicity level
        elif toxicity_level.lower() == 'mild':
            # Append customized context for mild toxicity
            report += "\n\nMild Toxicity Context: Audio content with mild toxicity typically contains subtle issues that can be improved with minor adjustments. While generally acceptable for broadcasting, addressing these minor concerns will enhance the ethical quality of your audio content."
            
            # Ensure suggestions are appropriate for mild toxicity
            if not suggestion_lines or len(suggestion_lines) < 3:
                suggestions = [
                    "Review specific phrases that could be replaced with more inclusive terminology",
                    "Add brief contextual statements when discussing potentially sensitive topics",
                    "Consider the tone used when referring to different groups in your audio content",
                    "Ensure balanced representation of perspectives in interviews or discussions"
                ]
            else:
                suggestions = filtered_suggestions
            
        elif toxicity_level.lower() == 'high':
            # Append customized context for high toxicity
            report += "\n\nHigh Toxicity Context: Audio content with high toxicity contains significant issues that require substantial revision. The problematic elements identified may perpetuate harmful stereotypes or contain misinformation that should be addressed before broadcasting."
            
            # Ensure suggestions are appropriate for high toxicity
            if not suggestion_lines or len(suggestion_lines) < 3:
                suggestions = [
                    "Re-record segments containing problematic language or stereotypes",
                    "Include diverse voices and perspectives in your audio content",
                    "Develop a script review process focused on ethical language use",
                    "Provide necessary context and balanced viewpoints for controversial topics",
                    "Consider consulting with cultural sensitivity experts for future recordings"
                ]
            else:
                suggestions = filtered_suggestions
                
        elif toxicity_level.lower() == 'max':
            # Append customized context for max toxicity
            report += "\n\nMaximum Toxicity Alert: Audio content with maximum toxicity requires immediate and comprehensive revision. The analysis identifies serious ethical concerns that may cause harm if broadcast or distributed in its current form."
            
            # Ensure suggestions are appropriate for max toxicity
            if not suggestion_lines or len(suggestion_lines) < 3:
                suggestions = [
                    "Consider re-creating the audio content with ethical reporting guidelines",
                    "Implement mandatory review by diversity and sensitivity experts",
                    "Establish clear internal guidelines for language use in audio content",
                    "Provide training for content creators on ethical reporting standards",
                    "Develop a comprehensive review process for all future audio content",
                    "Consider issuing a statement addressing the problematic content if already published"
                ]
            else:
                suggestions = filtered_suggestions
        
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
                
                try:
                    # Try to load with a larger buffer first
                    video_clip = VideoFileClip(
                        file_path, 
                        audio_buffersize=20000,  # Larger buffer
                        verbose=True,           # Enable verbose to debug
                        target_resolution=(360, 640)  # Lower resolution
                    )
                except Exception as buffer_error:
                    print(f"Failed with large buffer: {str(buffer_error)}")
                    # Try with an even smaller buffer
                    video_clip = VideoFileClip(
                        file_path, 
                        audio_buffersize=5000,  # Much smaller buffer
                        verbose=True,          # Enable verbose to debug
                        target_resolution=(240, 320)  # Even lower resolution
                    )
                
                print("VideoFileClip loaded successfully")
                
                # Check for audio track, but don't immediately fail if None
                # Sometimes audio is present but not detected properly
                has_audio = video_clip.audio is not None
                print(f"Video has detected audio track: {has_audio}")
                
                if has_audio:
                    print(f"Video loaded successfully. Duration: {video_clip.duration}s")
                    print("Extracting audio track...")
                    
                    # Use optimized settings for audio extraction
                    video_clip.audio.write_audiofile(
                        temp_audio_path,
                        verbose=True,    # Enable verbose for debugging
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
                    print("No audio track detected by MoviePy, will try alternative method")
                    # Don't fail yet, try ffmpeg
            except Exception as moviepy_error:
                print(f"MoviePy audio extraction error: {str(moviepy_error)}")
                
            # If moviepy failed, try with imageio/ffmpeg
            if not audio_extracted:
                try:
                    print("Trying alternative method with imageio/ffmpeg...")
                    try:
                        from imageio_ffmpeg import get_ffmpeg_exe
                        ffmpeg_exe = get_ffmpeg_exe()
                        print(f"Found bundled FFmpeg: {ffmpeg_exe}")
                    except ImportError:
                        # Try using system ffmpeg as fallback
                        ffmpeg_exe = "ffmpeg"
                        print("Using system ffmpeg")
                    
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
                    
                    # Print ffmpeg output for debugging
                    print(f"FFmpeg stdout: {stdout.decode('utf-8', errors='ignore')}")
                    print(f"FFmpeg stderr: {stderr.decode('utf-8', errors='ignore')}")
                    
                    if process.returncode == 0 and os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                        audio_extracted = True
                        print("Audio extraction successful with ffmpeg")
                    else:
                        print(f"FFmpeg error (code {process.returncode}): {stderr.decode('utf-8', errors='ignore')}")
                        
                        # Try one more time with simplified parameters
                        print("Trying simplified ffmpeg command...")
                        simple_cmd = [
                            ffmpeg_exe, '-y',
                            '-i', file_path,
                            '-vn',  # No video
                            temp_audio_path
                        ]
                        
                        simple_process = subprocess.Popen(
                            simple_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                        simple_stdout, simple_stderr = simple_process.communicate()
                        
                        if simple_process.returncode == 0 and os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                            audio_extracted = True
                            print("Audio extraction successful with simplified ffmpeg command")
                        else:
                            print(f"Simplified FFmpeg error: {simple_stderr.decode('utf-8', errors='ignore')}")
                            
                except Exception as ffmpeg_error:
                    print(f"FFmpeg extraction failed: {str(ffmpeg_error)}")
            
            # Verify audio extraction
            if not audio_extracted:
                print("All audio extraction methods failed")
                
                # If all methods failed, create a small placeholder audio file
                # This will allow us to at least try to transcribe, or return a more specific error
                try:
                    print("Creating placeholder silent audio file...")
                    with open(temp_audio_path, 'wb') as f:
                        # A minimal valid MP3 file (essentially silence)
                        from pydub import AudioSegment
                        silent = AudioSegment.silent(duration=1000)  # 1 second of silence
                        silent.export(temp_audio_path, format="mp3")
                        
                    if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                        print("Created placeholder audio file")
                        audio_extracted = True
                    else:
                        raise Exception("Could not create placeholder file")
                except Exception as placeholder_error:
                    print(f"Failed to create placeholder: {str(placeholder_error)}")
                    raise Exception("Could not extract audio from video using any available method")
                    
            # Check audio file exists
            if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
                raise Exception("Audio extraction failed - output file is empty")
            
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
                
                error_msg = str(transcribe_error)
                print(f"Transcription failed: {error_msg}")
                
                if "duration" in error_msg.lower() or "empty" in error_msg.lower() or "no speech" in error_msg.lower():
                    return {
                        'toxicity_level': 'Error',
                        'suggestions': [
                            'No speech detected in the video', 
                            'Try a video with clearer audio',
                            'Ensure the video has an audio track'
                        ],
                        'report': 'Error: The video could not be transcribed. No speech was detected in the audio track or the audio quality is too poor for transcription.'
                    }
                else:
                    raise Exception(f"Transcription failed: {error_msg}")
            
            # Clean up temp file
            try:
                if temp_audio_path and os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
            except Exception as cleanup_error:
                print(f"Warning: Could not delete temporary file: {str(cleanup_error)}")
            
            # Check transcription result
            if not transcribed_text or transcribed_text.strip() == "":
                return {
                    'toxicity_level': 'Error',
                    'suggestions': ['No speech detected in the video'],
                    'report': 'Error: The video file could not be transcribed. Please ensure it contains clear speech.'
                }
            
            # Store the full transcription
            full_transcription = transcribed_text
            
            # First, try using the fine-tuned model for toxicity level
            toxicity_level = determine_toxicity_level(transcribed_text, "video")
            
            # Analyze transcribed text
            analysis_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert in analyzing media content for ethical reporting on topics related to refugees, migrants, and other forcibly displaced populations. Your task is to analyze the transcribed video content for xenophobic language, misinformation, and harmful content. Do not use bold formatting, markdown formatting, or any special text formatting in your response."},
                    {"role": "user", "content": f"Analyze the following transcribed video content for xenophobic language, misinformation, and harmful content. Provide: 1) A toxicity level (None, Mild, High, or Max), 2) Specific suggestions for improvement, and 3) A comprehensive analysis report. Do not use any bold text or markdown formatting in your response.\n\nTranscribed content: {transcribed_text}"}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            # Extract analysis
            analysis_text = analysis_response.choices[0].message.content
            print("Content analysis complete")
            
            # Parse analysis
            if not toxicity_level:  # If fine-tuned model didn't provide a level, extract it from the analysis
                toxicity_level = "Medium"  # Default
                
                # Extract toxicity level
                if "toxicity level" in analysis_text.lower():
                    for level in ["none", "mild", "high", "max"]:
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
            
            # If toxicity level is 'none', customize the report and suggestions
            if toxicity_level.lower() == 'none':
                # Replace detailed report with a simpler positive message for 'none' toxicity
                positive_report = f"Your video content has been analyzed and found to contain no problematic or xenophobic language. The audio content appears to be ethical, balanced, and respectful. It presents information in a fair and accurate manner that follows best practices for responsible reporting."
                
                # Keep a short summary if available in the report
                if "summary" in analysis_text.lower() or "content" in analysis_text.lower():
                    # Try to extract a summary section if it exists
                    summary_lines = []
                    in_summary = False
                    for line in analysis_text.split('\n'):
                        if "summary" in line.lower() or "content summary" in line.lower():
                            in_summary = True
                            continue
                        if in_summary and line.strip() and "toxicity" not in line.lower() and "xenophobic" not in line.lower():
                            summary_lines.append(line)
                        if in_summary and line.strip() and ("analysis" in line.lower() or "report" in line.lower()):
                            break
                    
                    if summary_lines:
                        summary_text = " ".join(summary_lines)
                        positive_report = f"Content Summary: {summary_text}\n\n{positive_report}"
                
                # Note about visual analysis
                positive_report += "\n\nNote: This analysis is based primarily on the audio content of the video. Visual elements should also be reviewed for ethical representation."
                
                # Replace the detailed report with our positive message
                report = positive_report
                
                # Replace suggestions with positive phrases for 'none' toxicity
                suggestions = [
                    "Continue using inclusive and balanced language in video content",
                    "Maintain your ethical reporting approach in visual and audio media",
                    "Keep providing accurate and fair video content"
                ]
                
                return {
                    'toxicity_level': toxicity_level,
                    'suggestions': suggestions,
                    'report': report,
                    'transcription': full_transcription
                }
            
            # Enhance suggestions based on toxicity level
            elif toxicity_level.lower() == 'mild':
                # Append customized context for mild toxicity
                report += "\n\nMild Toxicity Context: Video content with mild toxicity typically contains subtle issues that can be improved with minor adjustments. While generally acceptable for viewing, addressing these minor concerns will enhance the ethical quality of your video content."
                
                # Note about visual analysis
                report += "\n\nNote: This analysis is based primarily on the audio content of the video. Visual elements should also be reviewed for ethical representation."
                
                # Ensure suggestions are appropriate for mild toxicity
                if not suggestion_lines or len(suggestion_lines) < 3:
                    suggestions = [
                        "Review visual and verbal elements for more inclusive representation",
                        "Consider adding context through captions or commentary when discussing sensitive topics",
                        "Ensure visual elements don't inadvertently perpetuate stereotypes",
                        "Review the framing of interviews or discussions for balanced perspectives"
                    ]
                else:
                    suggestions = filtered_suggestions
                
            elif toxicity_level.lower() == 'high':
                # Append customized context for high toxicity
                report += "\n\nHigh Toxicity Context: Video content with high toxicity contains significant issues that require substantial revision. Both audio and potentially visual elements may perpetuate harmful stereotypes or contain misinformation."
                
                # Note about visual analysis
                report += "\n\nNote: This analysis is based primarily on the audio content of the video. Visual elements should be carefully reviewed as they may contain additional problematic content."
                
                # Ensure suggestions are appropriate for high toxicity
                if not suggestion_lines or len(suggestion_lines) < 3:
                    suggestions = [
                        "Re-edit segments containing problematic language or visual stereotypes",
                        "Ensure diverse representation both in verbal content and visual elements",
                        "Implement a comprehensive review process for both audio and visual content",
                        "Consider adding contextual information or disclaimers for historical footage",
                        "Review camera angles, framing, and visual metaphors for potential bias"
                    ]
                else:
                    suggestions = filtered_suggestions
                    
            elif toxicity_level.lower() == 'max':
                # Append customized context for max toxicity
                report += "\n\nMaximum Toxicity Alert: Video content with maximum toxicity requires immediate and comprehensive revision. The analysis identifies serious ethical concerns in the audio content that may cause harm if published. Visual elements should also be carefully reviewed."
                
                # Note about visual analysis
                report += "\n\nNote: This analysis focuses on the audio transcript. Visual elements may contain additional problematic content requiring separate evaluation and revision."
                
                # Ensure suggestions are appropriate for max toxicity
                if not suggestion_lines or len(suggestion_lines) < 3:
                    suggestions = [
                        "Consider completely re-producing the video with ethical guidelines in place",
                        "Implement mandatory review by diversity and ethics experts before release",
                        "Establish comprehensive guidelines for both verbal and visual content",
                        "Ensure balanced representation in all aspects of production",
                        "Consider issuing a statement addressing the problematic content if already published",
                        "Review your organization's ethical guidelines and ensure compliance"
                    ]
                else:
                    suggestions = filtered_suggestions
            
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
        
        # If toxicity level is 'none', use positive phrases instead of suggestions
        if toxicity_level.lower() == 'none':
            suggestions = [
                "Continue using inclusive and balanced language",
                "Maintain your ethical reporting approach",
                "Keep providing accurate and fair content",
                "Your content demonstrates responsible reporting practices",
                "The content is free from xenophobic or problematic language",
                "You've successfully created ethical and balanced content",
                "Your writing shows respect for diverse perspectives",
                "The material presents information in a fair and accurate manner"
            ]
        # Provide tailored suggestions for each toxicity level
        elif toxicity_level.lower() == 'mild':
            if not suggestions or len(suggestions) < 3:
                suggestions = [
                    "Consider alternative phrasing for potentially problematic terms",
                    "Add more context to sensitive topic discussions",
                    "Review specific highlighted sections for more balanced perspectives",
                    "Ensure diverse viewpoints are represented",
                    "Be more specific when referring to groups to avoid generalizations",
                    "Use more precise language in sections identified in the report"
                ]
        elif toxicity_level.lower() == 'high':
            if not suggestions or len(suggestions) < 3:
                suggestions = [
                    "Substantially revise sections containing problematic language",
                    "Replace generalizations with specific, fact-based statements",
                    "Add context, nuance, and multiple perspectives to balance the narrative",
                    "Consider consulting cultural sensitivity experts for revisions",
                    "Review and update terminology used throughout the content",
                    "Implement fact-checking for claims made in the content",
                    "Ensure ethical reporting standards are applied throughout"
                ]
        elif toxicity_level.lower() == 'max':
            if not suggestions or len(suggestions) < 3:
                suggestions = [
                    "Consider a comprehensive rewrite following ethical guidelines",
                    "Consult with subject matter experts on ethical representation",
                    "Implement a thorough editorial review process",
                    "Address all xenophobic elements identified in the analysis",
                    "Ensure all claims are supported by credible sources",
                    "Review organizational ethical guidelines and ensure compliance",
                    "Consider sensitivity training for content creators",
                    "Establish clear internal standards for future content"
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
        
        # Improvement Suggestions section title - change based on toxicity level
        if toxicity_level.lower() == 'none':
            content.append(Paragraph("POSITIVE CONTENT ATTRIBUTES:", styles['CustomSubtitle']))
        elif toxicity_level.lower() == 'mild':
            content.append(Paragraph("RECOMMENDED IMPROVEMENTS:", styles['CustomSubtitle']))
        elif toxicity_level.lower() == 'high':
            content.append(Paragraph("NECESSARY REVISIONS:", styles['CustomSubtitle']))
        elif toxicity_level.lower() == 'max':
            content.append(Paragraph("CRITICAL CHANGES REQUIRED:", styles['CustomSubtitle']))
        else:
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
        
        # Word Cloud of Xenophobic Words - only include if toxicity level is not 'none'
        if wordcloud_image and toxicity_level.lower() != 'none':
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
        
        # Comprehensive Report title changes based on toxicity level
        if toxicity_level.lower() == 'none':
            content.append(Paragraph("CONTENT ASSESSMENT:", styles['CustomSubtitle']))
        elif toxicity_level.lower() == 'mild':
            content.append(Paragraph("ANALYSIS REPORT:", styles['CustomSubtitle']))
        elif toxicity_level.lower() == 'high':
            content.append(Paragraph("TOXICITY ASSESSMENT REPORT:", styles['CustomSubtitle']))
        elif toxicity_level.lower() == 'max':
            content.append(Paragraph("CRITICAL ETHICS REPORT:", styles['CustomSubtitle']))
        else:
            content.append(Paragraph("COMPREHENSIVE REPORT:", styles['CustomSubtitle']))
        
        content.append(Paragraph(report_content, styles['Normal_Justify']))
        
        # Add toxicity level specific guidance at the end of high and max reports
        if toxicity_level.lower() == 'high':
            content.append(Spacer(1, 12))
            content.append(Paragraph("IMPORTANT NOTE:", ParagraphStyle(name='Warning', parent=styles['Normal'], textColor=colors.red, fontName='Helvetica-Bold')))
            content.append(Paragraph("Content with high toxicity levels may negatively impact your audience and reputation. We strongly recommend implementing the revisions outlined in this report before publication or distribution.", styles['Normal_Justify']))
        elif toxicity_level.lower() == 'max':
            content.append(Spacer(1, 12))
            content.append(Paragraph("URGENT ACTION REQUIRED:", ParagraphStyle(name='Critical', parent=styles['Normal'], textColor=colors.red, fontName='Helvetica-Bold')))
            content.append(Paragraph("Content with maximum toxicity levels is likely to cause harm if published or distributed in its current form. We strongly advise against using this content without comprehensive revision following the guidelines provided in this report.", styles['Normal_Justify']))
        
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

# Find a suitable font for Chinese characters
def get_chinese_font():
    """Find a font that supports Chinese characters"""
    # Common fonts that support Chinese characters
    chinese_fonts = [
        'Arial Unicode MS', 
        'Microsoft YaHei', 
        'WenQuanYi Micro Hei',
        'Noto Sans CJK SC', 
        'Noto Sans CJK TC',
        'Noto Sans CJK JP',
        'SimHei', 
        'SimSun', 
        'FangSong',
        'STHeiti'
    ]
    
    for font in chinese_fonts:
        try:
            # Check if the font is available
            font_path = fm.findfont(fm.FontProperties(family=font))
            if font_path and 'ttf' in font_path.lower():
                print(f"Using font for Chinese characters: {font}")
                return font_path
        except:
            continue
    
    # If no suitable font is found, use the default font
    # This won't display Chinese characters correctly, but will prevent crashes
    print("Warning: No suitable font for Chinese characters found. Using default font.")
    return None

if __name__ == '__main__':
    app.run(debug=True, port=5004) 