from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, make_response, session
from werkzeug.utils import secure_filename
import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-for-development')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size

# Update allowed extensions to match what Whisper API supports
app.config['ALLOWED_EXTENSIONS'] = {
    'text': {'txt', 'pdf', 'doc', 'docx'},
    'audio': {'mp3', 'wav', 'ogg', 'm4a', 'flac', 'oga', 'mpga'},
    'video': {'mp4', 'webm', 'mpeg', 'mov'}  # Added MOV to allowed video formats
}

# Whisper API supported formats (for reference and validation)
WHISPER_SUPPORTED_FORMATS = {'flac', 'm4a', 'mp3', 'mp4', 'mpeg', 'mpga', 'oga', 'ogg', 'wav', 'webm'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# OpenAI API key (will be set later)
openai.api_key = os.environ.get('OPENAI_API_KEY')

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
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    file_type = request.form.get('file_type')
    
    # Check if the file has an extension
    if '.' not in file.filename:
        flash('Invalid file (no extension)', 'error')
        return redirect(url_for('index'))
    
    # Get the file extension
    file_extension = file.filename.rsplit('.', 1)[1].lower()
    
    # Special handling for video files - check if it's a format supported by Whisper API
    if file_type == 'video' and file_extension not in WHISPER_SUPPORTED_FORMATS and file_extension != 'mov':
        supported_video_formats = [fmt for fmt in WHISPER_SUPPORTED_FORMATS if fmt in ['mp4', 'webm', 'mpeg']]
        supported_formats_str = ", ".join(supported_video_formats + ['mov'])
        flash(f'Unsupported video format: .{file_extension}. Only {supported_formats_str} are supported for video analysis.', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename, file_type):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Handle MOV files or other non-directly supported video formats by converting to MP4
        converted_file_path = None
        if file_type == 'video' and (file_extension == 'mov' or file_extension not in WHISPER_SUPPORTED_FORMATS):
            try:
                flash('Converting video file to a compatible format for processing...', 'info')
                converted_file_path = convert_video_format(file_path, 'mp4')
                file_path = converted_file_path  # Use the converted file for processing
            except Exception as e:
                flash(f'Error converting video file: {str(e)}', 'error')
                return redirect(url_for('index'))
        
        try:
            # Process the file based on its type
            result = process_file(file_path, file_type)
            
            # Clean up converted file if it exists
            if converted_file_path and os.path.exists(converted_file_path):
                try:
                    os.unlink(converted_file_path)
                except:
                    pass  # Ignore errors in cleanup
            
            # Store results in session for PDF generation
            session['suggestions'] = result.get('suggestions', [])
            session['report_content'] = result.get('report', 'No report available')
            
            # Return the analysis results
            return render_template('results.html', result=result)
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(url_for('index'))
        finally:
            # Clean up converted file if it exists and wasn't cleaned up
            if converted_file_path and os.path.exists(converted_file_path):
                try:
                    os.unlink(converted_file_path)
                except:
                    pass  # Ignore errors in cleanup
    
    flash('File type not allowed', 'error')
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
        if not openai.api_key:
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
        analysis_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing media content for ethical reporting on topics related to refugees, migrants, and other forcibly displaced populations. Your task is to analyze the text content for xenophobic language, misinformation, and harmful content."},
                {"role": "user", "content": f"Analyze the following text content for xenophobic language, misinformation, and harmful content. Provide: 1) A toxicity level (Low, Medium, High, or Very High), 2) Specific suggestions for improvement, and 3) A comprehensive analysis report.\n\nContent: {analyzed_content}"}
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
            elif in_suggestions and line and "report" in line.lower():
                in_suggestions = False
        
        if suggestion_lines:
            suggestions = suggestion_lines
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
        
        # Add a note if content was truncated
        if content_truncated:
            report += "\n\nNote: The analyzed content was truncated due to length limitations. The analysis is based on the first portion of the document."
        
        return {
            'toxicity_level': toxicity_level,
            'suggestions': suggestions,
            'report': report
        }
    except Exception as e:
        return {
            'toxicity_level': 'Error',
            'suggestions': ['An error occurred during analysis'],
            'report': f'Error: {str(e)}'
        }

def analyze_audio(file_path):
    """Analyze audio content using OpenAI Whisper API for transcription and GPT for analysis"""
    try:
        # Check if OpenAI API key is set
        if not openai.api_key:
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
        
        # First try direct transcription
        try:
            with open(file_path, "rb") as audio_file:
                transcript_response = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            transcribed_text = transcript_response.text
        except Exception as e:
            print(f"Direct transcription failed: {str(e)}")
            # If direct transcription fails, try conversion
            try_conversion = True
        
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
                    transcript_response = openai.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                transcribed_text = transcript_response.text
                
                print("Conversion and transcription successful")
            except Exception as conversion_error:
                # If conversion also fails, return error
                if temp_file_path and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                return {
                    'toxicity_level': 'Error',
                    'suggestions': ['Audio processing failed'],
                    'report': f'Error: Could not process the M4A file. Original error: {str(e)}. Conversion error: {str(conversion_error)}'
                }
        elif try_conversion:
            # If it's not an M4A file and direct transcription failed
            return {
                'toxicity_level': 'Error',
                'suggestions': ['Audio transcription failed'],
                'report': f'Error: Could not transcribe the audio file. {str(e)}'
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
        
        # Store the transcription for reference
        transcription_summary = f"Audio Transcription:\n\n{transcribed_text[:500]}..."
        if len(transcribed_text) <= 500:
            transcription_summary = f"Audio Transcription:\n\n{transcribed_text}"
        
        # Analyze the transcribed text using GPT
        analysis_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing media content for ethical reporting on topics related to refugees, migrants, and other forcibly displaced populations. Your task is to analyze the transcribed audio content for xenophobic language, misinformation, and harmful content."},
                {"role": "user", "content": f"Analyze the following transcribed audio content for xenophobic language, misinformation, and harmful content. Provide: 1) A toxicity level (Low, Medium, High, or Very High), 2) Specific suggestions for improvement, and 3) A comprehensive analysis report.\n\nTranscribed content: {transcribed_text}"}
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
            elif in_suggestions and line and "report" in line.lower():
                in_suggestions = False
        
        if suggestion_lines:
            suggestions = suggestion_lines
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
        
        # Add the transcription to the report
        full_report = f"{report}\n\n{transcription_summary}"
        
        return {
            'toxicity_level': toxicity_level,
            'suggestions': suggestions,
            'report': full_report
        }
    
    except Exception as e:
        return {
            'toxicity_level': 'Error',
            'suggestions': ['An error occurred during audio analysis'],
            'report': f'Error: {str(e)}'
        }

def analyze_video(file_path):
    """Analyze video content by sending directly to Whisper API or extracting audio with minimal dependencies"""
    try:
        # Check if OpenAI API key is set
        if not openai.api_key:
            return {
                'toxicity_level': 'Error',
                'suggestions': ['OpenAI API key is not set'],
                'report': 'Error: Please set your OpenAI API key in the .env file to analyze video content.'
            }
        
        # Get file extension and log information
        file_extension = file_path.rsplit('.', 1)[1].lower()
        print(f"Processing video file: {file_path} (Format: {file_extension})")
        
        # Check file size - OpenAI Whisper API has a 25MB limit
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        whisper_limit_mb = 25  # OpenAI's Whisper API has a 25MB limit
        
        if file_size > 100:  # General app limit
            return {
                'toxicity_level': 'Error',
                'suggestions': ['File size too large'],
                'report': f'Error: The video file is {file_size:.1f}MB, which exceeds our 100MB limit. Please compress or shorten your video.'
            }
        elif file_size > whisper_limit_mb:
            # File is under our app limit but over OpenAI's limit
            return {
                'toxicity_level': 'Error',
                'suggestions': [
                    'File exceeds OpenAI\'s 25MB limit',
                    'Please compress your video or extract the audio',
                    f'Your file is {file_size:.1f}MB, but the limit is 25MB'
                ],
                'report': f'Error: Your video file is {file_size:.1f}MB, which exceeds OpenAI\'s 25MB limit for audio processing. Please either:\n\n1. Compress your video to under 25MB\n2. Extract just the audio from your video and upload it as an audio file (MP3, WAV, M4A)\n3. Split your video into smaller segments and analyze each separately'
            }
        
        # For video files, we need to ensure they're in a compatible format
        # Let's try to use a more direct approach with the file
        try:
            print(f"Attempting to transcribe {file_extension} video directly with Whisper API...")
            
            # For debugging, print more information about the file
            print(f"File exists: {os.path.exists(file_path)}")
            print(f"File size: {file_size:.2f} MB")
            
            # Read the file in binary mode to ensure proper handling
            with open(file_path, "rb") as video_file:
                # Create a named temporary file that keeps the original extension
                import tempfile
                import shutil
                
                # Create a temporary file with the same extension
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}")
                temp_path = temp_file.name
                temp_file.close()
                
                # Copy the original file to the temporary file
                shutil.copy2(file_path, temp_path)
                
                # Open the temporary file for the API call
                with open(temp_path, "rb") as api_file:
                    transcript_response = openai.audio.transcriptions.create(
                        model="whisper-1",
                        file=api_file
                    )
                
                # Clean up the temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass  # Ignore cleanup errors
                
            transcribed_text = transcript_response.text
            print("Direct transcription successful")
            
            # If we get here, the direct transcription worked
            if not transcribed_text:
                return {
                    'toxicity_level': 'Error',
                    'suggestions': ['No speech detected in the video'],
                    'report': 'Error: The video file could not be transcribed. Please ensure it contains clear speech.'
                }
        except Exception as direct_error:
            print(f"Direct transcription failed with error: {str(direct_error)}")
            
            # Try to provide more specific error information
            error_message = str(direct_error)
            
            # Check for specific error types
            if "413" in error_message or "size limit" in error_message.lower():
                # This is a file size error
                return {
                    'toxicity_level': 'Error',
                    'suggestions': [
                        'File exceeds OpenAI\'s 25MB limit',
                        'Please compress your video or extract the audio',
                        f'Your file is {file_size:.1f}MB, but the limit is 25MB'
                    ],
                    'report': f'Error: Your video file is {file_size:.1f}MB, which exceeds OpenAI\'s 25MB limit for audio processing. Please either:\n\n1. Compress your video to under 25MB\n2. Extract just the audio from your video and upload it as an audio file (MP3, WAV, M4A)\n3. Split your video into smaller segments and analyze each separately'
                }
            elif "unsupported format" in error_message.lower() or "invalid file" in error_message.lower():
                # This is a format error - should not happen with our conversion, but just in case
                return {
                    'toxicity_level': 'Error',
                    'suggestions': [
                        'Video format issue',
                        'Please try converting to MP4 manually',
                        'Our automatic conversion failed'
                    ],
                    'report': f'Error: There was an issue with the video format. Our automatic conversion process was unable to create a compatible file. Please try converting your video to MP4 format manually using a tool like HandBrake or FFmpeg. Technical details: {error_message}'
                }
            elif "permission" in error_message.lower():
                suggestion = "There was a permission error accessing the file."
            else:
                suggestion = "There was an error processing the video file."
            
            # General error case
            return {
                'toxicity_level': 'Error',
                'suggestions': ['Video processing failed', suggestion],
                'report': f'Error: Could not process the video file. Technical details: {error_message}\n\nFor best results, please try converting your video to MP4 format manually or extract the audio to an MP3 file.'
            }
        
        # Store the transcription for reference
        transcription_summary = f"Video Transcription:\n\n{transcribed_text[:500]}..."
        if len(transcribed_text) <= 500:
            transcription_summary = f"Video Transcription:\n\n{transcribed_text}"
        
        # Analyze the transcribed text using GPT
        print("Analyzing transcribed content...")
        analysis_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing media content for ethical reporting on topics related to refugees, migrants, and other forcibly displaced populations. Your task is to analyze the transcribed video content for xenophobic language, misinformation, and harmful content."},
                {"role": "user", "content": f"Analyze the following transcribed video content for xenophobic language, misinformation, and harmful content. Provide: 1) A toxicity level (Low, Medium, High, or Very High), 2) Specific suggestions for improvement, and 3) A comprehensive analysis report.\n\nTranscribed content: {transcribed_text}"}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        # Extract the analysis from the response
        analysis_text = analysis_response.choices[0].message.content
        print("Analysis complete")
        
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
            elif in_suggestions and line and "report" in line.lower():
                in_suggestions = False
        
        if suggestion_lines:
            suggestions = suggestion_lines
        else:
            # Fallback: just extract some reasonable suggestions
            suggestions = [
                "Ensure diverse representation in visual content",
                "Be mindful of stereotypical portrayals",
                "Consider the tone used when discussing sensitive topics"
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
        
        # Add the transcription to the report
        full_report = f"{report}\n\n{transcription_summary}"
        
        # Add a note about visual content
        full_report += "\n\nNote: This analysis is based on the audio content of the video. A full analysis would also include evaluation of visual elements, which requires human review."
        
        # If this was a converted MOV file, add a note about that
        if '_converted.mp4' in file_path or file_extension == 'mov':
            full_report += "\n\nNote: Your MOV file was automatically converted to MP4 format for processing."
        
        return {
            'toxicity_level': toxicity_level,
            'suggestions': suggestions,
            'report': full_report
        }
    
    except Exception as e:
        print(f"Unexpected error in analyze_video: {str(e)}")
        return {
            'toxicity_level': 'Error',
            'suggestions': ['An error occurred during video analysis'],
            'report': f'Error: {str(e)}'
        }

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

@app.route('/contact')
def contact():
    """Render the contact page"""
    return render_template('contact.html')

@app.route('/download_report_pdf')
def download_report_pdf():
    """Generate and download a PDF report"""
    try:
        # Get report data from session
        toxicity_level = request.args.get('toxicity_level', 'Unknown')
        suggestions = session.get('suggestions', [])
        report_content = session.get('report_content', 'No report available')
        
        # Import reportlab components
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
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
        
        # Comprehensive Report
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

if __name__ == '__main__':
    app.run(debug=True, port=5004) 