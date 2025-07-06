from flask import Flask, request, jsonify, render_template, send_from_directory
from moviepy import VideoFileClip
from pydub import AudioSegment
from pydub.utils import make_chunks
import speech_recognition as sr
import os
from transformers import BartForConditionalGeneration, BartTokenizer

model_dir = "./model"

model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name, cache_dir=model_dir)
model = BartForConditionalGeneration.from_pretrained(model_name, cache_dir=model_dir)

def split_text_into_chunks(text, max_length):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(tokenizer.encode(current_chunk + sentence, max_length=512)) <= max_length:
            current_chunk += sentence + '. '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def generate_text(input_text, max_length, min_length):
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def nas(transcribed_text):
    def format_text_as_points(text):
        """Formats text by splitting sentences into points (one sentence per line)."""
        sentences = [sentence.strip() for sentence in text.split('.') if sentence.strip()]
        return "\n".join(f"- {sentence}." for sentence in sentences)

    chunks = split_text_into_chunks(transcribed_text, max_length=512)

    # Generate notes (more detailed text) with length ~2000 words
    notes = ''
    notes_target_word_count = 3000
    current_word_count = 0

    for chunk in chunks:
        chunk_notes = generate_text(chunk, max_length=300, min_length=200)
        chunk_word_count = len(chunk_notes.split())
        if current_word_count + chunk_word_count <= notes_target_word_count:
            notes += chunk_notes + "\n\n"
            current_word_count += chunk_word_count
        else:
            remaining_words = notes_target_word_count - current_word_count
            notes += " ".join(chunk_notes.split()[:remaining_words]) + "\n\n"
            break
    n2 = notes + ''
    

    chunks2 = split_text_into_chunks(n2, max_length=512)
    
    # Generate summary (concise version) with length ~500 words
    summary = ''
    summary_target_word_count = 500
    current_word_count = 0

    for chunk in chunks2:
        chunk_summary = generate_text(chunk, max_length=100, min_length=50)
        chunk_word_count = len(chunk_summary.split())
        if current_word_count + chunk_word_count <= summary_target_word_count:
            summary += chunk_summary + " "
            current_word_count += chunk_word_count
        else:
            remaining_words = summary_target_word_count - current_word_count
            summary += " ".join(chunk_summary.split()[:remaining_words]) + " "
            break
    
    
    # Format notes as points
    notes = format_text_as_points(notes)
    notes = "NOTES:\n\n" + notes
    # Save notes to a file
    with open("notes.txt", "w", encoding="utf-8") as notes_file:
        notes_file.write(notes)
        
    # Format summary as points
    summary = format_text_as_points(summary)
    summary = "SUMMARY:\n\n" + summary
    # Save summary to a file
    with open("summary.txt", "w", encoding="utf-8") as summary_file:
        summary_file.write(summary)
        
    notes_summary = notes + "\n\n\n\n\n" + summary

    print("Notes and summary generated and saved successfully!")
    return notes_summary



app = Flask(__name__)

# Path to store uploaded files temporarily
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template("index.html")

# Route for the about page
@app.route('/about')
def about():
    return render_template('about.html')

# Route for the FAQ page
@app.route('/faq')
def faq():
    return render_template('faq.html')

# Route for the contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/download/<filename>')
def download_file(filename):
    # 'static' is the folder where the file is stored
    return send_from_directory(directory='static', path=filename, as_attachment=True)

@app.route('/uploaded', methods=['POST'])
def transcribe():
    # return jsonify({"transcription": "uploaded"})
    try:
        # Step 1: Check if a file is uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected for uploading"}), 400

        # Step 2: Save the uploaded video file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Step 3: Extract audio from the uploaded video file
        temp_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_audio.wav")

        # Use a `with` block to ensure the file is properly closed
        with VideoFileClip(file_path) as video_clip:
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(temp_audio_path)
            audio_clip.close()

        # Step 4: Load audio and split into chunks
        audio = AudioSegment.from_file(temp_audio_path, format="wav")
        chunk_length = 30  # Length of each chunk in seconds
        chunks = make_chunks(audio, chunk_length * 1000)

        # Create a folder for temporary chunks
        chunk_folder = os.path.join(app.config['UPLOAD_FOLDER'], "audio_chunks")
        os.makedirs(chunk_folder, exist_ok=True)

        # Initialize recognizer
        r = sr.Recognizer()

        # Step 5: Transcribe each chunk
        transcription = ""
        for i, chunk in enumerate(chunks):
            chunk_path = os.path.join(chunk_folder, f"chunk{i}.wav")
            chunk.export(chunk_path, format="wav")  # Save chunk as a temporary file

            with sr.AudioFile(chunk_path) as source:
                audio_data = r.record(source)

            try:
                # Transcribe chunk
                text = r.recognize_google(audio_data)
                transcription += text + " "
                print(f"Chunk {i + 1}/{len(chunks)} transcribed.")
            except sr.UnknownValueError:
                transcription += "[Unintelligible audio] "
                print(f"Chunk {i + 1}/{len(chunks)}: Unable to understand audio.")
            except sr.RequestError as e:
                transcription += f"[Error: {e}] "
                print(f"Chunk {i + 1}/{len(chunks)}: API error - {e}.")
        with open("./static/transcribe.txt", "w") as file:
            file.write(transcription)
            
        notes_and_summary = nas(transcription)
        
        with open("./static/nas_file.txt", "w") as file:
            file.write(notes_and_summary)
        
        # Step 6: Return transcription to the frontend
        return jsonify({"transcription": notes_and_summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Cleanup temporary files
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if os.path.exists(chunk_folder):
            for file in os.listdir(chunk_folder):
                os.remove(os.path.join(chunk_folder, file))
            os.rmdir(chunk_folder)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)  # Retry file removal if it was locked
            except PermissionError:
                pass



if __name__ == '__main__':
    app.run(debug=True)

    
# with open("transcribe2.txt", "w") as file:
#     file.write(transcription)