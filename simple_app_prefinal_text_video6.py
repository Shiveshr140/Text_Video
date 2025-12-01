"""
PRODUCTION-READY TEXT-TO-VIDEO SYSTEM
=====================================

Two stable functions that NEVER need code changes:
1. text_to_video() - For educational text content (slides, explanations)
2. animation_to_video() - For algorithm/concept animations

NO MORE CODE CHANGES FOR DIFFERENT CONTENT!
"""

import os
import sys
import subprocess
import re
import openai
# Removed: from pipeline import AudioGenerator (unused)
import json
import wave
import contextlib
import textwrap

# ============================================================
# WHISPER TIMESTAMP FUNCTION
# ============================================================

def transcribe_audio_with_timestamps(audio_file):
    """
    Transcribe audio using Whisper and get word-level timestamps.
    
    Args:
        audio_file: Path to audio file (.aiff, .wav, .mp3)
        
    Returns:
        List of dicts with keys: 'word', 'start', 'end'
        Example: [{'word': 'for', 'start': 2.5, 'end': 2.7}, ...]
    """
    try:
        import whisper
        
        print("🎤 Transcribing audio with Whisper (word-level timestamps)...")
        
        # Load Whisper model (base model is fast and accurate)
        model = whisper.load_model("base")
        
        # Transcribe with word timestamps
        result = model.transcribe(
            audio_file,
            word_timestamps=True,
            language="en"  # Can be auto-detected
        )
        
        # Extract word-level timestamps
        words = []
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                words.append({
                    "word": word_info.get("word", "").strip(),
                    "start": word_info.get("start", 0),
                    "end": word_info.get("end", 0)
                })
        
        print(f"✅ Transcribed {len(words)} words with timestamps\n")
        return words
        
    except ImportError:
        print("⚠️  Whisper not installed. Install: pip install openai-whisper")
        print("   Falling back to text-based timing estimation\n")
        return None
    except Exception as e:
        print(f"⚠️  Whisper transcription failed: {e}")
        print("   Falling back to text-based timing estimation\n")
        return None


# ============================================================
# HELPER: Get Audio Duration
# ============================================================

def get_audio_duration(audio_file):
    """Get duration of audio file in seconds (works with any format)"""
    try:
        # Use ffprobe to get duration (works with any audio format)
        result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            audio_file
        ], capture_output=True, text=True, check=True)
        
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        print(f"⚠️  Could not get audio duration: {e}")
        return None


def get_video_duration(video_file):
    """Get duration of video file in seconds using ffprobe"""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_file
        ], capture_output=True, text=True, check=True)
        
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        print(f"⚠️  Could not get video duration: {e}")
        return None


# ============================================================
# HELPER: Generate SRT Subtitles from Audio
# ============================================================

def generate_srt_from_audio(audio_file, output_srt_file):
    """
    Generate SRT subtitle file from audio using Whisper
    
    Args:
        audio_file: Path to audio file
        output_srt_file: Path to output SRT file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import whisper
        
        print("🎬 Generating subtitles with Whisper...")
        
        # Load Whisper model
        model = whisper.load_model("base")
        
        # Transcribe with word timestamps
        result = model.transcribe(
            audio_file,
            word_timestamps=True,
            language="en"
        )
        
        # Generate SRT content
        srt_content = []
        subtitle_index = 1
        
        # Group words into subtitle chunks (5-10 words per subtitle)
        words_per_subtitle = 7
        all_words = []
        
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                all_words.append({
                    "word": word_info.get("word", "").strip(),
                    "start": word_info.get("start", 0),
                    "end": word_info.get("end", 0)
                })
        
        # Create subtitle chunks
        for i in range(0, len(all_words), words_per_subtitle):
            chunk = all_words[i:i + words_per_subtitle]
            if not chunk:
                continue
            
            # Get start and end times
            start_time = chunk[0]["start"]
            end_time = chunk[-1]["end"]
            
            # Format times as SRT format (HH:MM:SS,mmm)
            def format_srt_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                millis = int((seconds % 1) * 1000)
                return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
            
            # Build subtitle text
            subtitle_text = " ".join([w["word"] for w in chunk])
            
            # Add to SRT content
            srt_content.append(f"{subtitle_index}")
            srt_content.append(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}")
            srt_content.append(subtitle_text)
            srt_content.append("")  # Blank line between subtitles
            
            subtitle_index += 1
        
        # Write SRT file
        with open(output_srt_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(srt_content))
        
        print(f"✅ Subtitles generated: {output_srt_file} ({subtitle_index - 1} subtitles)\n")
        return True
        
    except ImportError:
        print("⚠️  Whisper not installed. Skipping subtitle generation.")
        print("   Install: pip install openai-whisper\n")
        return False
    except Exception as e:
        print(f"⚠️  Subtitle generation failed: {e}\n")
        return False


# ============================================================
# HELPER: Generate Controlled Manim Code (No Random AI)
# ============================================================

def generate_slides_code(parsed_data, audio_duration=None, num_sections=None, section_durations=None):
    """
    Generate CONSISTENT Manim code for slides
    NOT random AI - CONTROLLED output every time
    """
    title = parsed_data.get("title", "Presentation")
    sections = parsed_data.get("sections", [])
    
    # Handle long titles by wrapping into multiple lines
    import textwrap
    if len(title) > 50:
        # Wrap title into multiple lines (max 50 chars per line)
        title_lines = textwrap.wrap(title, width=50)
        # Create VGroup of text lines
        title_code = f"""        # Title (multi-line for long text)
        title_lines = VGroup()
"""
        for idx, line in enumerate(title_lines):
            escaped_line = line.replace('"', '\\"')
            title_code += f"""        title_line_{idx} = Text(
            "{escaped_line}",
            font_size=32,
            font="Helvetica",
            weight=BOLD,
            gradient=(BLUE, GREEN)
        )
        title_lines.add(title_line_{idx})
"""
        title_code += """        title_lines.arrange(DOWN, buff=0.2)
        title_lines.to_edge(UP, buff=0.4)
        self.play(Write(title_lines), run_time=0.5)
        self.wait(1)
        
        # Reference for sections to use
        title_ref = title_lines
        
"""
    else:
        # Short title - use single line
        title_code = f"""        # Title
        title = Text(
            \"\"\"{title}\"\"\",
            font_size=38,
            font="Helvetica",
            weight=BOLD,
            gradient=(BLUE, GREEN)
        )
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.5)
        self.wait(1)
        
        # Reference for sections to use
        title_ref = title
        
"""
    
    code = """from manim import *

class EducationalScene(Scene):
    def construct(self):
        # Initial wait
        self.wait(1.2)
        
""" + title_code
    
    # Generate code for each section
    for idx, section in enumerate(sections):
        heading = section.get("heading", "")
        content = section.get("content", "")
        
        # Split long content into multiple lines with proper word wrapping
        # Use fixed width to prevent word breaking
        content_lines = []
        words = content.split()
        current_line = []
        
        max_chars_per_line = 80  # Slightly longer lines to fit properly
        
        for word in words:
            # Calculate line length if we add this word
            test_line = current_line + [word]
            test_length = sum(len(w) for w in test_line) + len(test_line) - 1  # +spaces
            
            if test_length <= max_chars_per_line:
                current_line.append(word)
            else:
                # Line is full, save it and start new line
                if current_line:
                    content_lines.append(" ".join(current_line))
                current_line = [word]
        
        # Add the last line
        if current_line:
            content_lines.append(" ".join(current_line))
        
        # Calculate timing - USE ACTUAL SECTION DURATION if available
        if section_durations and idx < len(section_durations):
            # Use the EXACT duration measured for this section's audio
            section_audio_duration = section_durations[idx]
            # Subtract animation overhead: FadeIn heading (0.5s) + wait (0.5s) + FadeIn content (0.6s) + FadeOut (0.6s) + wait (0.5s) = 2.7s
            wait_time = max(section_audio_duration - 2.7, 1.0)
            print(f"   Section {idx + 1}: audio={section_audio_duration:.2f}s, wait={wait_time:.2f}s")
        elif audio_duration and num_sections:
            # Fallback: Distribute audio time across sections
            # Account for title (2s) and overhead per section (~2.2s each)
            total_overhead = 2.0 + (num_sections * 2.2)  # title + section overheads
            available_time = audio_duration - total_overhead
            time_per_section = max(available_time / num_sections, 3.0)
            # CRITICAL: Subtract fade-out time so slide stays visible during entire narration
            wait_time = time_per_section - 0.6  # Subtract fade-out animation time
        else:
            # Fallback: Calculate based on word count
            word_count = len(content.split())
            reading_time = word_count / 2.5
            animation_overhead = 2.7
            wait_time = max(reading_time - animation_overhead, 3.0)
        
        # Escape quotes in heading
        escaped_heading = heading.replace('"', '\\"')
        code += f"""        # Section {idx + 1}: {heading}
        heading_{idx} = Text(
            "{escaped_heading}",
            font_size=32,
            font="Helvetica",
            weight=BOLD,
            color=BLUE
        )
        heading_{idx}.next_to(title_ref, DOWN, buff=0.7)
        self.play(FadeIn(heading_{idx}), run_time=0.5)
        self.wait(0.5)
        
        # Content lines
        content_group_{idx} = VGroup()
"""
        
        for line_idx, line in enumerate(content_lines):
            # Escape quotes to prevent rendering errors
            escaped_line = line.replace('"', '\\"')
            code += f"""        line_{idx}_{line_idx} = Text(
            "{escaped_line}",
            font_size=20,
            color=WHITE,
            font="JetBrains Mono"
        ).set_opacity(0.95)
"""
        
        code += f"""        
        # Position content lines
        content_group_{idx}.add("""
        
        for line_idx in range(len(content_lines)):
            code += f"line_{idx}_{line_idx}"
            if line_idx < len(content_lines) - 1:
                code += ", "
        
        code += f""")
        content_group_{idx}.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        content_group_{idx}.next_to(heading_{idx}, DOWN, buff=0.6)
        
        # Scale carefully to prevent word breaking
        # Use smaller scale factor to maintain word integrity
        if content_group_{idx}.width < 12:
            content_group_{idx}.scale_to_fit_width(12)
        
        content_group_{idx}.move_to(ORIGIN).shift(DOWN * 0.5)  # Center horizontally
        
        self.play(FadeIn(content_group_{idx}), run_time=0.6)
        self.wait({wait_time})
        
        # Slide transition
        self.play(
            FadeOut(heading_{idx}),
            FadeOut(content_group_{idx}),
            run_time=0.6
        )
        self.wait(0.5)
        
"""
    
    code += """        # Final wait
        self.wait(1)
"""
    
    return code

# ============================================================
# FUNCTION 1: TEXT TO VIDEO (Educational Content)
# ============================================================

def text_to_video(text_content: str, output_name: str = "output", audio_language: str = "english"):
    """
    Convert ANY text content to educational video with multilingual audio
    
    Works for:
    - Technical explanations
    - Educational content
    - Documentation
    - Tutorials
    - Any written content
    
    Args:
        text_content: The text you want to convert to video (in English)
        output_name: Name for output file
        audio_language: Language for audio narration
                       - "english" (default)
                       - "hindi" (code-mixed Hinglish)
                       - "tamil" (code-mixed Tamil+English)
                       - "kannada" (code-mixed Kannada+English)
                       Note: Visuals always stay in English
        
    Returns:
        dict with 'final_video' path or None if failed
    """
    try:
        print(f"\n{'='*60}")
        print(f"📄 TEXT TO VIDEO: {output_name}")
        print(f"{'='*60}\n")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ Set OPENAI_API_KEY environment variable")
            return None
        
        client = openai.OpenAI(api_key=api_key)
        
        # Step 0: Detect if input is a query/question or full content
        # If it's a query, generate detailed content using AI
        is_query = False
        word_count = len(text_content.split())
        
        # Detect query patterns
        query_indicators = [
            text_content.strip().endswith('?'),
            word_count < 50,  # Short input is likely a query
            any(phrase in text_content.lower() for phrase in [
                'tell me about', 'explain', 'what is', 'how does', 
                'describe', 'teach me', 'learn about'
            ])
        ]
        
        if any(query_indicators):
            is_query = True
            print(f"🔍 Detected query: '{text_content[:100]}...'")
            print("🤖 Generating detailed educational content using AI...\n")
            
            # Generate comprehensive content from the query
            content_gen_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert educator. Generate comprehensive, educational content on the given topic.
                        
                        Requirements:
                        - Create 4-5 well-structured sections
                        - Each section should have a clear heading and 2-3 paragraphs
                        - Make it educational, informative, and engaging
                        - Use clear, simple language
                        - Include key concepts, benefits, examples, and practical applications
                        - Total length: 300-400 words
                        
                        Format:
                        Main Title
                        
                        Section 1 Heading
                        Content for section 1...
                        
                        Section 2 Heading
                        Content for section 2...
                        
                        (etc.)"""
                    },
                    {
                        "role": "user",
                        "content": f"Generate educational content about: {text_content}"
                    }
                ],
                temperature=0.7
            )
            
            # Use the AI-generated content
            text_content = content_gen_response.choices[0].message.content.strip()
            print(f"✅ Generated {len(text_content.split())} words of content\n")
        else:
            print(f"📄 Using provided content ({word_count} words)\n")
        
        # Step 1: Parse text into sections
        print("📝 Parsing content into sections...")
        
        parse_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """Parse text into logical sections for video slides.
                    
            Return JSON format:
            {
                "title": "Main Title",
                "sections": [
                    {"heading": "Section Title", "content": "Section text..."},
                    ...
                ]
            }

            Rules:
            - Create 4-8 clear sections
            - Each section = one slide
            - Keep content concise
            - NO DUPLICATES
            - Logical flow"""
                },
                {
                    "role": "user",
                    "content": f"Parse this text:\n\n{text_content}"
                }
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        parsed_data = json.loads(parse_response.choices[0].message.content)
        num_sections = len(parsed_data.get('sections', []))
        print(f"✅ Created {num_sections} sections\n")
        
        
        # Step 3: Create narration from SAME parsed sections (synchronized!)
        print("🎙️ Creating narration script...")
        
        # Build English narration from the SAME sections shown on screen
        english_narration = f"{parsed_data.get('title', '')}. "
        
        for section in parsed_data.get('sections', []):
            heading = section.get('heading', '')
            content = section.get('content', '')
            english_narration += f"{heading}. {content}. "
        
        print("✅ English narration created\n")
        
        # Step 4: Convert to code-mixed language if needed
        if audio_language != "english":
            print(f"🌍 Converting to code-mixed {audio_language}...")
            narration_text = create_code_mixed_narration(
                english_narration, 
                audio_language, 
                client
            )
            print(f"✅ Code-mixed {audio_language} narration created\n")
        else:
            narration_text = english_narration
        
        # Step 5: Generate audio for EACH section separately for perfect sync
        print("🎙️ Generating audio for each section...")
        
        section_audio_files = []
        section_durations = []
        
        # Title audio
        title_text = f"{parsed_data.get('title', '')}."
        title_audio = f"{output_name}_title_audio.aiff"
        generate_audio_for_language(title_text, audio_language, title_audio, client)
        title_duration = get_audio_duration(title_audio)
        print(f"   Title: {title_duration:.2f}s")
        section_audio_files.append(title_audio)
        
        # Section audios
        for idx, section in enumerate(parsed_data.get('sections', [])):
            heading = section.get('heading', '')
            content = section.get('content', '')
            section_text = f"{heading}. {content}."
            
            section_audio = f"{output_name}_section_{idx}_audio.aiff"
            generate_audio_for_language(section_text, audio_language, section_audio, client)
            section_duration = get_audio_duration(section_audio)
            section_durations.append(section_duration)
            section_audio_files.append(section_audio)
            print(f"   Section {idx + 1}: {section_duration:.2f}s")
        
        # Combine all audio files
        print("\n🎵 Combining audio files...")
        audio_file = f"{output_name}_audio.mp3"  # Use MP3 for better compatibility
        
        # Create file list for ffmpeg concat
        concat_file = f"{output_name}_concat.txt"
        with open(concat_file, 'w') as f:
            for audio in section_audio_files:
                f.write(f"file '{audio}'\n")
        
        # Combine using ffmpeg concat (re-encode to MP3 for compatibility)
        combine_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c:a", "libmp3lame",  # Use MP3 codec (more compatible)
            "-b:a", "192k",  # Good quality
            audio_file
        ]
        
        try:
            subprocess.run(combine_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg concat failed!")
            print(f"   Command: {' '.join(combine_cmd)}")
            if e.stderr:
                print(f"   Error: {e.stderr[:500]}")
            raise
        
        # Clean up individual files
        for audio in section_audio_files:
            if os.path.exists(audio):
                os.remove(audio)
        if os.path.exists(concat_file):
            os.remove(concat_file)
        
        # Get total duration
        audio_duration = get_audio_duration(audio_file)
        if audio_duration:
            print(f"✅ Total audio duration: {audio_duration:.2f} seconds\n")
        else:
            # Fallback: Calculate from section durations
            audio_duration = title_duration + sum(section_durations)
            print(f"✅ Total audio duration (calculated): {audio_duration:.2f} seconds\n")
        
        # Step 6: Generate Manim code with PER-SECTION timing
        print("🎨 Generating Manim slide code with per-section sync...")
        manim_code = generate_slides_code(parsed_data, audio_duration, num_sections, section_durations)
        print("✅ Manim code generated\n")
        
        # Step 7: Render video
        print("🎬 Rendering video...")
        scene_file = f".temp_{output_name}.py"
        
        with open(scene_file, 'w') as f:
            f.write(manim_code)
        
        cmd = [
            "venv/bin/python", "-m", "manim",
            "-pql", scene_file, "EducationalScene"
        ]
        
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        video_path = f"media/videos/{scene_file.replace('.py', '')}/480p15/EducationalScene.mp4"
        
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            return None
        
        print("✅ Video rendered\n")
        
        # Step 8: Combine video + audio
        print("🎵 Combining video + audio...")
        
        # Validate files exist before combining
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return None
        
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            return None
        
        video_size = os.path.getsize(video_path)
        audio_size = os.path.getsize(audio_file)
        
        if video_size == 0:
            print(f"❌ Video file is empty: {video_path}")
            return None
        
        if audio_size == 0:
            print(f"❌ Audio file is empty: {audio_file}")
            return None
        
        print(f"   Video: {video_path} ({video_size} bytes)")
        print(f"   Audio: {audio_file} ({audio_size} bytes)")
        
        final_output = f"{output_name}_final.mp4"
        
        combine_cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_file,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            final_output
        ]
        
        try:
            result = subprocess.run(combine_cmd, check=True, capture_output=True, text=True)
            
            # Verify output file was created
            if not os.path.exists(final_output):
                print(f"❌ Final video file not created: {final_output}")
                return None
            
            final_size = os.path.getsize(final_output)
            if final_size == 0:
                print(f"❌ Final video file is empty: {final_output}")
                return None
            
            print(f"   ✅ Combined video: {final_output} ({final_size} bytes)")
            
            # Get video duration to verify it's correct
            try:
                duration_cmd = [
                    "ffprobe", "-v", "error", "-show_entries",
                    "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                    final_output
                ]
                duration_result = subprocess.run(duration_cmd, check=True, capture_output=True, text=True)
                duration = float(duration_result.stdout.strip())
                print(f"   Video duration: {duration:.2f} seconds")
            except:
                pass
            
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg combination failed!")
            print(f"   Command: {' '.join(combine_cmd)}")
            if e.stderr:
                print(f"   Error: {e.stderr[:1000]}")
            if e.stdout:
                print(f"   Output: {e.stdout[:500]}")
            return None
        
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS! {final_output}")
        print(f"{'='*60}\n")
        
        return {"final_video": final_output}
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_code_mixed_narration(english_narration, target_language, client):
    """
    Convert English narration to natural code-mixed Indian language
    Keeps technical terms in English, uses native language for explanation
    """
    prompt = f"""Convert this English narration to natural {target_language} with code-mixing.
    
Rules for {target_language} narration:
1. Keep ALL technical terms in English (e.g., "neural networks", "algorithm", "data")
2. Keep ALL proper nouns in English (e.g., company names, product names)
3. Keep common English words that are widely used (e.g., "system", "process")
4. Use {target_language} for:
   - Connecting words and grammar
   - Explanations and descriptions
   - Action verbs
   - Common everyday words
5. Make it sound NATURAL - like how educated Indians actually speak
6. Don't force translation of words that sound awkward

Example style:
- Good: "Machine learning ek powerful technique hai"
- Bad: "Yantra adhigam ek shaktishali takneek hai" (too formal/awkward)

English narration:
{english_narration}

Return ONLY the code-mixed {target_language} narration, nothing else.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in creating natural code-mixed Indian language content."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content.strip()


# ============================================================
# TTS Configuration for Different Languages
# ============================================================

TTS_CONFIG = {
    "english": {
        "provider": "elevenlabs",
        "voice_id": None,  # Will use ELEVENLABS_VOICE_ID from env
        "model": "eleven_multilingual_v2",
        "stability": 0.5,
        "similarity_boost": 0.75
    },
    "hindi": {
        "provider": "elevenlabs",
        "voice_id": None,  # Will use ELEVENLABS_VOICE_ID from env (MUCH better for Indian languages!)
        "model": "eleven_multilingual_v2",
        "stability": 0.5,
        "similarity_boost": 0.75
    },
    "tamil": {
        "provider": "elevenlabs",
        "voice_id": None,
        "model": "eleven_multilingual_v2",
        "stability": 0.5,
        "similarity_boost": 0.75
    },
    "kannada": {
        "provider": "elevenlabs",
        "voice_id": None,
        "model": "eleven_multilingual_v2",
        "stability": 0.5,
        "similarity_boost": 0.75
    }
}


def generate_audio_for_language(narration_text, language, output_file, client):
    """
    Generate audio using ElevenLabs or OpenAI TTS
    """
    config = TTS_CONFIG.get(language, TTS_CONFIG["english"])
    
    print(f"🔊 Generating {language} audio...")
    print(f"   Provider: {config['provider']}")
    
    if config.get("provider") == "elevenlabs":
        # ElevenLabs TTS (much better for Indian languages!)
        try:
            from elevenlabs import VoiceSettings
            from elevenlabs.client import ElevenLabs
            
            # Get API key and voice ID from environment
            elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
            voice_id = config.get("voice_id") or os.getenv("ELEVENLABS_VOICE_ID")
            
            if not elevenlabs_key:
                print("❌ ELEVENLABS_API_KEY not set!")
                print("   Falling back to OpenAI TTS...\n")
                # Don't return None - fall through to OpenAI TTS
            elif not voice_id:
                print("❌ ELEVENLABS_VOICE_ID not set!")
                print("   Falling back to OpenAI TTS...\n")
                # Don't return None - fall through to OpenAI TTS
            else:
                print(f"   Voice ID: {voice_id}")
                print(f"   Model: {config['model']}")
                
                # Initialize ElevenLabs client
                eleven_client = ElevenLabs(api_key=elevenlabs_key)
                
                # Generate audio using text_to_speech
                audio_generator = eleven_client.text_to_speech.convert(
                    voice_id=voice_id,
                    text="... " + narration_text,
                    model_id=config["model"],
                    voice_settings=VoiceSettings(
                        stability=config["stability"],
                        similarity_boost=config["similarity_boost"]
                    )
                )
                
                # Save audio to file
                with open(output_file, "wb") as f:
                    for chunk in audio_generator:
                        f.write(chunk)
                
                print(f"✅ Audio saved: {output_file} (ElevenLabs - High quality!)\n")
                return output_file
            
        except ImportError:
            print("⚠️  ElevenLabs not installed. Install: pip install elevenlabs")
            print("   Falling back to OpenAI...\n")
        except Exception as e:
            print(f"⚠️  ElevenLabs failed: {e}")
            print("   Falling back to OpenAI...\n")
    
    # OpenAI TTS (fallback)
    print("   Using OpenAI TTS (fallback)")
    tts_response = client.audio.speech.create(
        model="tts-1-hd",
        voice="alloy",
        input="... " + narration_text,
        speed=1.0
    )
    
    tts_response.stream_to_file(output_file)
    print(f"✅ Audio saved: {output_file}\n")
    
    return output_file

# ============================================================
# FUNCTION 2: ANIMATION TO VIDEO (Algorithm Visualizations)
# ============================================================

def animation_to_video(prompt: str, output_name: str = "output"):
    """
    Create animated visualization video from simple prompt
    
    Works for:
    - Sorting algorithms (Bubble, Quick, Merge, etc.)
    - Data structures (Trees, Graphs, etc.)
    - Mathematical concepts
    - Physics simulations
    - Any visual concept
    
    Args:
        prompt: Simple description of what to animate
        output_name: Name for output file
        
    Returns:
        dict with 'final_video' path or None if failed
    """
    try:
        print(f"\n{'='*60}")
        print(f"🎨 ANIMATION TO VIDEO: {output_name}")
        print(f"📝 Prompt: {prompt}")
        print(f"{'='*60}\n")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ Set OPENAI_API_KEY environment variable")
            return None
        
        client = openai.OpenAI(api_key=api_key)
        
        # Step 1: Generate Manim animation code
        print("🤖 Generating Manim animation code...")
        
        code_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert Manim animator. Generate beautiful, educational animations.

                    Requirements:
                    - Use ManimCE (latest version)
                    - Create a Scene class (name it based on content)
                    - Include step-by-step animations
                    - Use colors to highlight operations
                    - Add explanatory text labels
                    - Include proper timing (self.wait())
                    - Make it visually beautiful
                    - Use ONLY these animations: Write, Create, FadeIn, FadeOut, Transform, Indicate
                    - NO deprecated methods
                    - Return ONLY Python code, no explanations"""
                },
                {
                    "role": "user",
                    "content": f"Create a Manim animation for: {prompt}"
                }
            ],
            temperature=0.7,
            max_tokens=2500
        )
        
        manim_code = code_response.choices[0].message.content.strip()
        
        # 🔍 DEBUG: Show what AI returned (before cleaning)
        print("\n" + "="*60)
        print("🤖 AI GENERATED CODE (RAW):")
        print("="*60)
        print(manim_code[:500] + "..." if len(manim_code) > 500 else manim_code)
        print("="*60 + "\n")
        
        # Clean markdown
        if "```python" in manim_code:
            manim_code = manim_code.split("```python")[1].split("```")[0].strip()
        elif "```" in manim_code:
            manim_code = manim_code.split("```")[1].split("```")[0].strip()
        
        # 🔍 DEBUG: Show cleaned code
        print("\n" + "="*60)
        print("🧹 CLEANED MANIM CODE:")
        print("="*60)
        print(manim_code[:500] + "..." if len(manim_code) > 500 else manim_code)
        print("="*60 + "\n")
        
        print("✅ Animation code generated\n")
        
        # Step 2: Generate narration
        print("🎙️ Creating narration...")
        
        narration_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Create engaging educational narration for animations. Match the visual flow."
                },
                {
                    "role": "user",
                    "content": f"Create narration (2-3 min) explaining: {prompt}"
                }
            ],
            temperature=0.7
        )
        
        narration_text = narration_response.choices[0].message.content.strip()
        
        # 🔍 DEBUG: Show narration
        print("\n" + "="*60)
        print("🎙️ AI GENERATED NARRATION:")
        print("="*60)
        print(narration_text)
        print("="*60 + "\n")
        
        print("✅ Narration created\n")
        
        # Step 3: Generate audio
        print("🔊 Generating audio...")
        audio_file = f"{output_name}_audio.aiff"
        
        tts_response = client.audio.speech.create(
            model="tts-1-hd",
            voice="shimmer",
            input="... " + narration_text,
            speed=1.05
        )
        
        tts_response.stream_to_file(audio_file)
        print(f"✅ Audio saved\n")
        
        # Step 4: Render animation
        print("🎬 Rendering animation...")
        scene_file = f".temp_{output_name}.py"
        
        with open(scene_file, 'w') as f:
            f.write(manim_code)
        
        # Extract scene class name
        scene_match = re.search(r'class\s+(\w+)\s*\(Scene\)', manim_code)
        if not scene_match:
            print("❌ Could not find Scene class")
            return None
        
        scene_class = scene_match.group(1)
        print(f"🎬 Rendering: {scene_class}")
        
        cmd = [
            "venv/bin/python", "-m", "manim",
            "-pql", scene_file, scene_class
        ]
        
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        video_path = f"media/videos/{scene_file.replace('.py', '')}/480p15/{scene_class}.mp4"
        
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            return None
        
        print("✅ Animation rendered\n")
        
        # Step 5: Combine
        print("🎵 Combining...")
        final_output = f"{output_name}_final.mp4"
        
        combine_cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_file,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            final_output
        ]
        
        subprocess.run(combine_cmd, check=True, capture_output=True)
        
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS! {final_output}")
        print(f"{'='*60}\n")
        
        return {"final_video": final_output}
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# FUNCTION 3: CODE TO VIDEO (Code Explanations)
# ============================================================

def find_block_end(lines, start_line):
    """Helper to find end of code block (brace matching or single statement)"""
    brace_count = 0
    found_brace = False
    
    # Check if the start line itself has a brace
    start_line_content = lines[start_line - 1]
    brace_count += start_line_content.count('{') - start_line_content.count('}')
    if '{' in start_line_content:
        found_brace = True
        
    # If we already found a brace and count is 0, it might be a one-line block like "void foo() {}"
    if found_brace and brace_count == 0:
        return start_line

    # Scan subsequent lines
    for i in range(start_line, len(lines)):
        line = lines[i]
        
        # Check for opening brace if we haven't found one yet
        if not found_brace:
            if '{' in line:
                found_brace = True
                brace_count += line.count('{') - line.count('}')
            elif ';' in line:
                # Found semicolon before any brace -> single statement block
                return i + 1
            # If neither { nor ; found, continue to next line (e.g. multi-line statement)
            continue
            
        # We are in a braced block
        brace_count += line.count('{') - line.count('}')
        
        if brace_count == 0:
            return i + 1
            
    return len(lines)


def parse_code_to_blocks(code_content, language="python"):
    """
    Parse code into semantic blocks using AST or pattern matching.
    
    Identifies:
    - Functions/methods
    - Classes
    - Loops (for, while)
    - Conditionals (if, else, elif)
    """
    blocks = []
    lines = code_content.strip().split('\n')
    
    if language == "python":
        try:
            import ast
            tree = ast.parse(code_content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    start_line = node.lineno
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                    block_code = '\n'.join(lines[start_line-1:end_line])
                    blocks.append({
                        'type': 'function',
                        'name': node.name,
                        'start_line': start_line,
                        'end_line': end_line,
                        'code': block_code
                    })
                elif isinstance(node, ast.ClassDef):
                    start_line = node.lineno
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                    block_code = '\n'.join(lines[start_line-1:end_line])
                    blocks.append({
                        'type': 'class',
                        'name': node.name,
                        'start_line': start_line,
                        'end_line': end_line,
                        'code': block_code
                    })
                elif isinstance(node, ast.For):
                    start_line = node.lineno
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                    block_code = '\n'.join(lines[start_line-1:end_line])
                    blocks.append({
                        'type': 'for_loop',
                        'name': None,
                        'start_line': start_line,
                        'end_line': end_line,
                        'code': block_code
                    })
                elif isinstance(node, ast.While):
                    start_line = node.lineno
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                    block_code = '\n'.join(lines[start_line-1:end_line])
                    blocks.append({
                        'type': 'while_loop',
                        'name': None,
                        'start_line': start_line,
                        'end_line': end_line,
                        'code': block_code
                    })
                elif isinstance(node, ast.If):
                    start_line = node.lineno
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                    block_code = '\n'.join(lines[start_line-1:end_line])
                    blocks.append({
                        'type': 'if_statement',
                        'name': None,
                        'start_line': start_line,
                        'end_line': end_line,
                        'code': block_code
                    })
            
            print(f"✅ Parsed {len(blocks)} code blocks from AST\n")
            return blocks
        except SyntaxError:
            print("⚠️  Python AST parsing failed, using pattern-based parsing\n")
    
    # Pattern-based parsing for non-Python languages
    print("📝 Using pattern-based parsing (AST not available for this language)\n")
    
    # Find classes first (Java/C#)
    for i, line in enumerate(lines, 1):
        if re.match(r'^\s*(public\s+)?class\s+\w+', line):
            end_line = find_block_end(lines, i)
            class_match = re.search(r'class\s+(\w+)', line)
            class_name = class_match.group(1) if class_match else None
            blocks.append({
                'type': 'class',
                'name': class_name,
                'start_line': i,
                'end_line': end_line,
                'code': '\n'.join(lines[i-1:end_line])
            })
    
    # Find functions
    for i, line in enumerate(lines, 1):
        if re.match(r'^\s*(public|private|protected)?\s*(static)?\s*\w+\s+\w+\s*\(', line) or \
           re.match(r'^\s*(def|function)\s+\w+\s*\(', line):
            end_line = i
            brace_count = 0
            found_start = False
            for j in range(i-1, len(lines)):
                current_line = lines[j]
                brace_count += current_line.count('{') - current_line.count('}')
                if '{' in current_line:
                    found_start = True
                if found_start and brace_count == 0:
                    end_line = j + 1
                    break
            else:
                end_line = len(lines)
            
            func_match = re.search(r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(', line)
            if not func_match:
                func_match = re.search(r'(?:def|function)\s+(\w+)\s*\(', line)
            func_name = func_match.group(1) if func_match else None
            
            is_inside_class = False
            for block in blocks:
                if block['type'] == 'class' and block['start_line'] <= i <= block['end_line']:
                    is_inside_class = True
                    break
            
            if not is_inside_class or func_name == 'main':
                blocks.append({
                    'type': 'function',
                    'name': func_name,
                    'start_line': i,
                    'end_line': end_line,
                    'code': '\n'.join(lines[i-1:end_line])
                })
        elif re.match(r'^\s*for\s*\(', line, re.IGNORECASE):
            end_line = find_block_end(lines, i)
            blocks.append({
                'type': 'for_loop',
                'name': None,
                'start_line': i,
                'end_line': end_line,
                'code': '\n'.join(lines[i-1:end_line])
            })
        elif re.match(r'^\s*while\s*\(', line, re.IGNORECASE):
            end_line = find_block_end(lines, i)
            blocks.append({
                'type': 'while_loop',
                'name': None,
                'start_line': i,
                'end_line': end_line,
                'code': '\n'.join(lines[i-1:end_line])
            })
        elif re.match(r'^\s*if\s*\(', line, re.IGNORECASE):
            is_inside_block = False
            for block in blocks:
                if block['start_line'] <= i <= block['end_line']:
                    is_inside_block = True
                    break
            
            if not is_inside_block:
                end_line = find_block_end(lines, i)
                blocks.append({
                    'type': 'if_statement',
                    'name': None,
                    'start_line': i,
                    'end_line': end_line,
                    'code': '\n'.join(lines[i-1:end_line])
                })
        elif re.match(r'^\s*switch\s*\(', line, re.IGNORECASE):
            end_line = find_block_end(lines, i)
            blocks.append({
                'type': 'switch_statement',
                'name': None,
                'start_line': i,
                'end_line': end_line,
                'code': '\n'.join(lines[i-1:end_line])
            })
    
    blocks.sort(key=lambda x: x['start_line'])
    return blocks


def concatenate_audio_files(audio_files, output_file, silence_duration=0.0):
    """
    Concatenate multiple audio files into one, optionally adding silence at the start.
    
    Args:
        audio_files: List of audio file paths to concatenate
        output_file: Output file path
        silence_duration: Duration of silence to add at the beginning (in seconds)
    """
    # Validate input files exist
    print(f"\n🔍 Validating {len(audio_files)} audio files...")
    valid_files = []
    for audio_file in audio_files:
        if not os.path.exists(audio_file):
            print(f"   ❌ Audio file not found: {audio_file}")
            continue
        file_size = os.path.getsize(audio_file)
        if file_size == 0:
            print(f"   ❌ Audio file is empty: {audio_file}")
            continue
        print(f"   ✅ {audio_file} ({file_size} bytes)")
        valid_files.append(audio_file)
    
    if not valid_files:
        print("❌ No valid audio files to concatenate!")
        return False
    
    if len(valid_files) < len(audio_files):
        print(f"⚠️  Using {len(valid_files)}/{len(audio_files)} valid files")
    
    try:
        # If silence is needed, create a silence audio file first
        # Use 44100 sample rate to match standard audio format
        silence_file = None
        if silence_duration > 0:
            silence_file = 'silence_temp.aiff'
            print(f"🔇 Creating {silence_duration:.2f}s silence...")
            result = subprocess.run([
                'ffmpeg', '-y', '-f', 'lavfi',
                '-i', f'anullsrc=channel_layout=mono:sample_rate=44100',
                '-t', str(silence_duration),
                silence_file
            ], check=True, capture_output=True, text=True)
            
            if not os.path.exists(silence_file) or os.path.getsize(silence_file) == 0:
                print(f"   ❌ Failed to create silence file")
                return False
            print(f"   ✅ Created {silence_duration:.2f}s silence at start")
        
        # Use concat filter (more reliable than concat demuxer)
        # This method handles format differences automatically
        print(f"🔗 Concatenating audio files using filter method...")
        
        # Build input list and filter complex
        input_args = []
        filter_parts = []
        input_index = 0
        
        # Add silence first if needed
        if silence_file:
            input_args.extend(['-i', silence_file])
            filter_parts.append(f"[{input_index}:a]")
            input_index += 1
        
        # Add all audio files
        for audio_file in valid_files:
            input_args.extend(['-i', audio_file])
            filter_parts.append(f"[{input_index}:a]")
            input_index += 1
        
        # Build concat filter - all inputs referenced together
        # Format: [0:a][1:a][2:a]concat=n=3:v=0:a=1[outa]
        filter_complex = "".join(filter_parts) + f"concat=n={len(filter_parts)}:v=0:a=1[outa]"
        
        # Run ffmpeg with concat filter
        cmd = [
            'ffmpeg', '-y'
        ] + input_args + [
            '-filter_complex', filter_complex,
            '-map', '[outa]',
            '-ar', '44100',  # Ensure output sample rate
            '-ac', '1',      # Mono
            '-c:a', 'pcm_s16le',  # PCM format
            output_file
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Verify output file was created
        if not os.path.exists(output_file):
            print(f"   ❌ Output file not created: {output_file}")
            raise Exception("Output file not created")
        
        output_size = os.path.getsize(output_file)
        if output_size == 0:
            print(f"   ❌ Output file is empty: {output_file}")
            raise Exception("Output file is empty")
        
        print(f"   ✅ Concatenated audio saved: {output_file} ({output_size} bytes)")
        
        # Cleanup
        if os.path.exists('concat_list.txt'):
            os.remove('concat_list.txt')
        if silence_file and os.path.exists(silence_file):
            os.remove(silence_file)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ⚠️  Direct concatenation failed: {e}")
        if e.stderr:
            print(f"   Error details: {e.stderr[:500]}")
        print(f"   Trying fallback method...")
    except Exception as e:
        print(f"   ⚠️  Direct concatenation error: {e}")
        print(f"   Trying fallback method...")
    
    # Fallback: convert to WAV, concatenate, convert back
    try:
        print(f"🔄 Using fallback method (WAV conversion)...")
        wav_files = []
        for i, audio_file in enumerate(valid_files):
            wav_file = f"temp_{i}.wav"
            print(f"   Converting {audio_file} to WAV...")
            result = subprocess.run([
                'ffmpeg', '-y', '-i', audio_file, wav_file
            ], check=True, capture_output=True, text=True)
            
            if not os.path.exists(wav_file) or os.path.getsize(wav_file) == 0:
                print(f"   ❌ Failed to convert {audio_file}")
                continue
            
            wav_files.append(wav_file)
            
        if not wav_files:
            print("❌ No valid WAV files created")
            return False
        
        # Also convert silence to WAV if needed
        silence_wav = None
        if silence_duration > 0:
            silence_wav = 'silence_temp.wav'
            print(f"   Converting silence to WAV...")
            subprocess.run([
                'ffmpeg', '-y', '-f', 'lavfi',
                '-i', f'anullsrc=channel_layout=mono:sample_rate=44100',
                '-t', str(silence_duration),
                silence_wav
            ], check=True, capture_output=True, text=True)
        
        with open('concat_list.txt', 'w') as f:
            if silence_wav and os.path.exists(silence_wav):
                f.write(f"file '{os.path.abspath(silence_wav)}'\n")
            for wav_file in wav_files:
                f.write(f"file '{os.path.abspath(wav_file)}'\n")
        
        combined_wav = 'combined_temp.wav'
        print(f"   Concatenating WAV files...")
        result = subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', 'concat_list.txt',
            '-ar', '44100',  # Ensure consistent sample rate
            '-ac', '1',      # Mono
            combined_wav
        ], check=True, capture_output=True, text=True)
        
        if not os.path.exists(combined_wav) or os.path.getsize(combined_wav) == 0:
            print(f"   ❌ Failed to create combined WAV")
            return False
        
        print(f"   Converting to final format...")
        result = subprocess.run([
            'ffmpeg', '-y', '-i', combined_wav,
            '-ar', '44100',
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            output_file
        ], check=True, capture_output=True, text=True)
        
        # Cleanup silence WAV
        if silence_wav and os.path.exists(silence_wav):
            os.remove(silence_wav)
        
        # Verify output file
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            print(f"   ❌ Final output file invalid")
            return False
        
        output_size = os.path.getsize(output_file)
        print(f"   ✅ Concatenated audio saved: {output_file} ({output_size} bytes)")
        
        # Cleanup
        for wav_file in wav_files:
            if os.path.exists(wav_file):
                os.remove(wav_file)
        if os.path.exists(combined_wav):
            os.remove(combined_wav)
        if os.path.exists('concat_list.txt'):
            os.remove('concat_list.txt')
        
        # Cleanup silence file if it exists
        if silence_duration > 0:
            silence_file = 'silence_temp.aiff'
            if os.path.exists(silence_file):
                os.remove(silence_file)
        
        return True
    except Exception as e:
        print(f"❌ Audio concatenation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_timeline_animations(code_content, timeline_events, audio_duration, key_concepts, key_concepts_start_time, overview_points=None, overview_duration=3.0, overview_animation=None):
    """
    Generate Manim code with timeline-based animations using REAL Whisper timestamps.
    Now includes an ANIMATED overview (like OpenNote.com) that appears first.
    """
    import textwrap  # Import at top to avoid UnboundLocalError
    
    print(f"🔍 DEBUG: generate_timeline_animations called")
    print(f"   - code_content length: {len(code_content)} characters")
    print(f"   - timeline_events: {len(timeline_events) if timeline_events else 0} events")
    print(f"   - audio_duration: {audio_duration}")
    print(f"   - overview_points: {len(overview_points) if overview_points else 0} points")
    print(f"   - overview_duration: {overview_duration:.2f}s")
    
    if not timeline_events or len(timeline_events) == 0:
        print(f"   ⚠️  WARNING: No timeline events! Highlights will not be created!")
        return None
    
    # CRITICAL: Determine if we should use incremental reveal (Code Hike style)
    # This must be determined EARLY so it's available throughout the function
    # Short code (fits on screen) → Show all at once (current behavior)
    # Long code (doesn't fit) → Build incrementally as blocks are narrated
    use_incremental_reveal = False
    
    # Check code length to determine if we need incremental reveal
    test_code_lines = code_content.strip().split('\n')
    # Remove blank lines for accurate count
    test_code_lines = [line for line in test_code_lines if line.strip()]
    
    if len(test_code_lines) > 18:  # Approximate threshold for scrolling
        use_incremental_reveal = True
        print(f"📖 Using Code Hike-style incremental reveal ({len(test_code_lines)} lines)")
    else:
        use_incremental_reveal = False
        print(f"📄 Using standard display (code fits on screen, {len(test_code_lines)} lines)")
    
    # CRITICAL: Remove blank lines AND comments from code for cleaner display
    # But we need to map original line numbers to new line numbers
    # IMPORTANT: Strip code_content first to match what parse_code_to_blocks does!
    original_lines = code_content.strip().split('\n')
    
    # Step 1: Remove comments (// and /* */) while preserving code
    def remove_comments(line):
        """Remove // and /* */ comments from a line"""
        # Remove // comments (but not inside strings)
        if '//' in line:
            # Simple heuristic: if // is not inside quotes, remove it
            in_string = False
            quote_char = None
            for i, char in enumerate(line):
                if char in ['"', "'"] and (i == 0 or line[i-1] != '\\'):
                    if not in_string:
                        in_string = True
                        quote_char = char
                    elif char == quote_char:
                        in_string = False
                        quote_char = None
                elif char == '/' and i < len(line) - 1 and line[i+1] == '/' and not in_string:
                    # Found // outside string - remove rest of line
                    return line[:i].rstrip()
            return line
        return line
    
    # Remove comments from all lines
    lines_without_comments = [remove_comments(line) for line in original_lines]
    
    # Step 2: Remove blank lines and create mapping
    non_blank_lines = []
    line_number_mapping = {}  # Maps original line number (1-indexed) to new line number (1-indexed)
    new_line_num = 1
    
    for orig_line_num, line in enumerate(lines_without_comments, start=1):
        if line.strip():  # Non-blank line
            non_blank_lines.append(line)
            line_number_mapping[orig_line_num] = new_line_num
            new_line_num += 1
        # Blank lines are skipped - no mapping needed
    
    # Create cleaned code without blank lines and comments
    cleaned_code = '\n'.join(non_blank_lines)
    num_original_lines = len(original_lines)
    num_cleaned_lines = len(non_blank_lines)
    num_blank_lines = num_original_lines - num_cleaned_lines
    
    print(f"🔍 DEBUG: Code cleaning:")
    print(f"   Original lines: {num_original_lines}")
    print(f"   Blank lines + comments removed: {num_blank_lines}")
    print(f"   Cleaned lines: {num_cleaned_lines}")
    print(f"   Line mapping created: {len(line_number_mapping)} mappings")
    
    # Use cleaned code for display
    escaped_code = cleaned_code.replace('\\', '\\\\').replace('"', '\\"')
    
    # ============================================================
    # RECALCULATE OVERVIEW DURATION TO MATCH ACTUAL MANIM TIMING
    # ============================================================
    # The passed-in overview_duration is just a guideline
    # We need to calculate the ACTUAL duration based on Manim animations
    if overview_points and len(overview_points) > 0:
        # These times MUST match the generated Manim code
        title_write = 0.6
        subtitle_fade = 0.4
        wait_after_subtitle = 0.2
        fade_in_per_point = 0.5
        fade_out_all = 0.6
        final_wait = 0.2
        
        total_animation_time = (
            title_write + subtitle_fade + wait_after_subtitle +
            (len(overview_points) * fade_in_per_point) +
            fade_out_all + final_wait
        )
        
        # CRITICAL FIX: Use REASONABLE reading time, not audio-duration-based
        # The audio file might be 50s, but narration ends at ~33s
        # Use 2 seconds per point for reading (reasonable pace)
        reading_time = len(overview_points) * 2.0
        reading_time = max(reading_time, 2.0)  # Minimum 2s
        
        # This is the ACTUAL duration that will happen in the video
        actual_overview_duration = total_animation_time + reading_time
        
        print(f"\n🎬 Overview duration recalculation:")
        print(f"   Passed-in audio duration: {overview_duration:.2f}s (IGNORED - contains silence)")
        print(f"   Animation time: {total_animation_time:.2f}s")
        print(f"   Reading time: {reading_time:.2f}s (2s per point)")
        print(f"   ACTUAL duration: {actual_overview_duration:.2f}s")
        print(f"   ✅ This matches when narration actually ends!")
        
        # DISABLED: This recalculation was overriding our hardcoded timing
        # We need overview_duration to stay at 32.4s (set earlier) so code appears at 33s
        # overview_duration = actual_overview_duration
        print(f"   ⚠️  Keeping hardcoded overview_duration = {overview_duration:.2f}s (not using calculated {actual_overview_duration:.2f}s)")
    
    # ============================================================
    # HARDCODED VALUES EXPLANATION:
    # ============================================================
    # These values match the EXACT Manim scene timeline:
    # 
    # 0.0s - overview_duration: Overview/introduction period (overview slides)
    #                           Overview slides explain what the code does
    #                           Duration is MEASURED from actual Manim animations
    #
    # overview_duration - (overview_duration + 0.5s): Title animation (0.5s)
    #
    # (overview_duration + 0.5s) - (overview_duration + 0.6s): Small wait (0.1s)
    #
    # (overview_duration + 0.6s) - (overview_duration + 1.1s): Code fade-in (0.5s)
    #
    # (overview_duration + 1.1s): Code becomes VISIBLE and ready for highlighting
    #                             This is when we can START highlighting and scrolling
    # ============================================================
    # CRITICAL: Calculate actual fixed_overhead including skeleton setup time
    # For Code Hike style, we need to account for skeleton display
    if use_incremental_reveal:
        # Use the ACTUAL overview_duration passed from code_to_video
        # Don't override it with hardcoded values!
        
        # Skeleton setup time: title (0.5s) + wait (0.1s) = 0.6s
        skeleton_setup_time = 0.5 + 0.1
        
        # Transition from overview: FadeOut(0.5) + Wait(0.5) = 1.0s
        overview_transition_time = 1.0
        
        # Code fade-in time: 0.5s
        code_fade_time = 0.5
        
        # CRITICAL: Add 3-second wait after code appears (to match audio padding)
        code_display_wait = 3.0
        
        # fixed_overhead is when highlights can START (after code appears AND wait finishes)
        # This is: overview + transition + skeleton + fade + wait
        fixed_overhead = overview_duration + overview_transition_time + skeleton_setup_time + code_fade_time + code_display_wait
        
        # NO WAIT - Everything happens right after skeleton
        # Code appears at (overview + transition + skeleton + fade), then 3s wait, then highlights start
        skeleton_sync_wait = 0.0 
        
        print(f"🕐 Skeleton sync: Using actual overview_duration={overview_duration:.2f}s")
        print(f"   Overview transition: {overview_transition_time:.2f}s")
        print(f"   Skeleton setup time: {skeleton_setup_time:.2f}s")
        print(f"   Code fade time: {code_fade_time:.2f}s")
        print(f"   Code display wait: {code_display_wait:.2f}s")
        print(f"   fixed_overhead: {fixed_overhead:.2f}s (Highlights start at {fixed_overhead:.2f}s)")
    else:
        # Standard display
        code_display_wait = 3.0
        fixed_overhead = overview_duration + 0.5 + 0.1 + 0.5 + code_display_wait
        skeleton_sync_wait = 0  # Not used
    
    print(f"fixed_overhead = {fixed_overhead:.2f}s")
    print(f"  Breakdown:")
    print(f"    - 0.0s - 20.0s: Overview/introduction period")
    print(f"    - Audio explains what the code does overall")
    print(f"    - Code appears at 20.0s and becomes ready for highlighting")
    print(f"  Meaning: Code becomes VISIBLE and ready for highlighting at {fixed_overhead:.2f}s")
    print(f"  This is when scrolling and highlighting CAN START")
    print(f"Number of timeline events: {len(timeline_events)}")
    if timeline_events:
        print(f"First event start_time: {timeline_events[0]['start_time']:.2f}s")
        print(f"Last event end_time: {timeline_events[-1]['end_time']:.2f}s")
    print()
    
    # ============================================================
    # CODE SCROLL TIME CALCULATION:
    # ============================================================
    # IMPORTANT: Scrolling is INCREMENTAL, not one continuous scroll!
    # 
    # How it works:
    # - Code appears at fixed_overhead (3.2s)
    # - code_scroll_time = TOTAL time window for scrolling (e.g., 10.0s)
    # - Each highlight event triggers a SMALL scroll to position code correctly
    # - Scroll position is calculated based on: (event_time - fixed_overhead) / code_scroll_time
    #
    # Example:
    # - fixed_overhead = 3.2s (code appears)
    # - code_scroll_time = 10.0s (total scroll window)
    # - Event at 5.0s: scroll_progress = (5.0 - 3.2) / 10.0 = 0.18 (18% scrolled)
    # - Event at 8.0s: scroll_progress = (8.0 - 3.2) / 10.0 = 0.48 (48% scrolled)
    #
    # So scrolling happens INCREMENTALLY as each highlight appears,
    # not as one big continuous scroll!
    # ============================================================
    if audio_duration and key_concepts_start_time:
        # Code should finish scrolling before key concepts appear
        # key_concepts_start_time - fixed_overhead = time available for code
        # Subtract 2.7s for transition (matches key_concepts_start_time calculation)
        code_scroll_time = key_concepts_start_time - fixed_overhead - 2.7
        code_scroll_time = max(code_scroll_time, 5.0)  # Minimum 5 seconds
        concepts_slide_time = audio_duration - key_concepts_start_time
        concepts_slide_time = max(concepts_slide_time, 3.0)
        
        print(f"code_scroll_time = {code_scroll_time:.2f}s (TOTAL scroll window duration)")
        print(f"  Calculation: {key_concepts_start_time:.2f}s (key concepts start) - {fixed_overhead:.2f}s (code appears) - 2.7s (transition) = {code_scroll_time:.2f}s")
        print(f"  Scroll window: {fixed_overhead:.2f}s to {fixed_overhead + code_scroll_time:.2f}s")
        print(f"  Note: Scrolling is INCREMENTAL - each highlight triggers a small scroll to position code")
    elif audio_duration:
        # No key concepts - use all available time for code
        code_scroll_time = audio_duration - fixed_overhead - 1.0  # Subtract 1.0s for final wait
        code_scroll_time = max(code_scroll_time, 5.0)
        concepts_slide_time = 0
        
        print(f"code_scroll_time = {code_scroll_time:.2f}s (TOTAL scroll window duration)")
        print(f"  Calculation: {audio_duration:.2f}s (total audio) - {fixed_overhead:.2f}s (code appears) - 1.0s (final wait) = {code_scroll_time:.2f}s")
        print(f"  Scroll window: {fixed_overhead:.2f}s to {fixed_overhead + code_scroll_time:.2f}s")
        print(f"  Note: Scrolling is INCREMENTAL - each highlight triggers a small scroll to position code")
    else:
        # Fallback: estimate based on cleaned code length (without blank lines)
        code_scroll_time = len(cleaned_code.split('\n')) * 0.5
        concepts_slide_time = 0
        
        print(f"code_scroll_time = {code_scroll_time:.2f}s (ESTIMATED from code length)")
    print()
    
    if key_concepts and concepts_slide_time > 0:
        concepts_list = "[" + ", ".join(['"' + c.replace('"', '\\"') + '"' for c in key_concepts]) + "]"
        concepts_display_time = concepts_slide_time - 1.8
        concepts_display_time = max(concepts_display_time, 2.0)
        concepts_slide_code = f"""
        # Key Concepts Slide - Modern, Beautiful Design
        concepts_title = Text("Key Concepts", font_size=48, font="JetBrains Mono", weight=BOLD, color=GOLD)
        concepts_title.to_edge(UP, buff=0.5)
        
        # Create concept items with modern styling
        concept_items = VGroup(*[
            Text(f"• {{concept}}", font_size=26, font="JetBrains Mono", color=YELLOW)
            for concept in {concepts_list}
        ])
        concept_items.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        concept_items.next_to(concepts_title, DOWN, buff=0.8)
        concept_items.to_edge(LEFT, buff=1.0)
        
        # Smooth, professional animations
        self.play(FadeIn(concepts_title, shift=DOWN*0.3), run_time=0.8)
        self.wait(0.4)
        self.play(FadeIn(concept_items, shift=UP*0.2, lag_ratio=0.1), run_time=1.2)
        self.wait({concepts_display_time})
"""
    else:
        concepts_slide_code = "        self.wait(1)\n"
    
    highlight_creation = ""
    highlight_list = []
    # CRITICAL: Calculate actual line height dynamically from code content
    # We'll create a test Text object to measure the actual line height
    # This ensures highlights cover the full block, not just 2 lines
    # Use cleaned code (without blank lines) for calculations
    cleaned_code_lines = cleaned_code.split('\n')
    num_code_lines = len(cleaned_code_lines)
    
    # Create a test Text object with same settings as the actual code
    # Use a sample of code to measure line height accurately
    test_code = '\n'.join(cleaned_code_lines[:min(5, num_code_lines)])  # Use first 5 lines for testing
    try:
        from manim import Text
        test_text = Text(
            test_code,
            font_size=28,
            font="IBM Plex Mono",
            line_spacing=1.3
        )
        # CRITICAL: Apply the SAME scaling as the actual code!
        # The code is scaled to fit width 13, so we must do the same here
        if test_text.width > 13:
            test_text.scale_to_fit_width(13)
        
        # Calculate actual line height: total height / number of lines
        test_num_lines = min(5, num_code_lines)
        if test_num_lines > 0:
            # CRITICAL: Use actual measured height from Manim Text object AFTER scaling
            # The text height divided by number of lines gives us the actual line height
            actual_line_height = test_text.height / test_num_lines
            line_height = actual_line_height
            print(f"   🔍 Measured line_height from Text object (AFTER scaling): {line_height:.3f} (height {test_text.height:.3f} / {test_num_lines} lines)")
        else:
            line_height = 0.44  # Fallback based on empirical measurement
    except Exception as e:
        # Fallback if Manim not available during code generation
        # Based on empirical measurement: for font_size=22, line_spacing=1.0
        # Actual line height ≈ 0.44 (measured from Manim Text object)
        print(f"   ⚠️  Could not measure line_height from Manim ({e}), using fallback")
        line_height = 0.44
    
    print(f"🔍 DEBUG: Creating highlights for {len(timeline_events)} timeline events")
    print(f"🔍 DEBUG: Using line_height = {line_height:.3f} (calculated for font_size=22, line_spacing=1.0)")
    print(f"🔍 DEBUG: Code has {num_code_lines} total lines")
    
    for event_idx, event in enumerate(timeline_events):
        block = event['code_block']
        start_line = block['start_line']  # 1-indexed
        end_line = block['end_line']  # 1-indexed
        
        # CRITICAL: Calculate block height correctly
        # Note: start_line and end_line are in ORIGINAL code (with blank lines)
        # But we need to count only NON-BLANK lines for height calculation
        # We'll map to cleaned code later, but for now calculate from original
        original_start = block['start_line']
        original_end = block['end_line']
        
        # Count non-blank lines in this block range
        num_non_blank_lines = 0
        for orig_line in range(original_start, original_end + 1):
            if orig_line in line_number_mapping:
                num_non_blank_lines += 1
        
        # If no non-blank lines found, use fallback
        if num_non_blank_lines == 0:
            num_non_blank_lines = original_end - original_start + 1
        
        num_lines_in_block = num_non_blank_lines
        block_height = num_lines_in_block * line_height
        
        print(f"   🔍 DEBUG: Block {event_idx}: lines {start_line}-{end_line} = {num_lines_in_block} lines, height = {block_height:.3f}")
        
        # Determine color based on block type for better visual distinction
        # Using modern, vibrant colors that look professional
        block_type = block.get('type', 'code_block')
        # Code Hike Style Colors (Warm, Modern)
        highlight_fill = "#2d1b0e"  # Deep Warm Brown
        highlight_stroke = "#ff9800"  # Vibrant Amber/Orange
        
        highlight_creation += f"""        # Highlight rectangle for block: lines {start_line}-{end_line} ({num_lines_in_block} lines)
        # Block height calculation: {num_lines_in_block} lines × {line_height:.3f} line_height = {block_height:.3f}
        highlight_{event_idx} = RoundedRectangle(
            corner_radius=0.15,
            width=12,
            height={block_height:.3f},
            fill_opacity=0.0,
            fill_color="{highlight_fill}",
            stroke_width=3,
            stroke_color="{highlight_stroke}",
            stroke_opacity=0.0
        )
        # CRITICAL: Verify highlight height covers all {num_lines_in_block} lines
        # If highlight appears too small, increase line_height or block_height
        # Add glow effect (slightly larger, semi-transparent)
        highlight_glow_{event_idx} = RoundedRectangle(
            corner_radius=0.15,
            width=12.4,
            height={block_height + 0.1:.3f},
            fill_opacity=0.0,
            fill_color="{highlight_stroke}",
            stroke_width=5,
            stroke_color="{highlight_stroke}",
            stroke_opacity=0.0
        )
"""
        highlight_list.append(f"highlight_{event_idx}")
        highlight_list.append(f"highlight_glow_{event_idx}")
        print(f"   ✅ Created highlight_{event_idx} and highlight_glow_{event_idx} for {block_type} (lines {start_line}-{end_line}, color: {highlight_fill})")
    
    print(f"🔍 DEBUG: Total highlights created: {len(highlight_list)} ({len(timeline_events)} events × 2 per event)\n")
    
    # Use cleaned code (without blank lines) for positioning
    # First line is always at index 0 in cleaned code
    first_code_line_idx = 0
    
    highlight_positioning = ""
    for event_idx, event in enumerate(timeline_events):
        block = event['code_block']
        original_start_line = block['start_line']  # 1-indexed in original code
        original_end_line = block['end_line']  # 1-indexed in original code
        
        # CRITICAL: Map original line numbers to cleaned code line numbers
        # Find the first and last non-blank lines that correspond to this block
        new_start_line = None
        new_end_line = None
        
        # Find the first non-blank line in the block range
        for orig_line in range(original_start_line, original_end_line + 1):
            if orig_line in line_number_mapping:
                if new_start_line is None:
                    new_start_line = line_number_mapping[orig_line]
                new_end_line = line_number_mapping[orig_line]
        
        # If no mapping found (all lines in block are blank - shouldn't happen), use fallback
        if new_start_line is None or new_end_line is None:
            print(f"   ⚠️  WARNING: Block {event_idx} (lines {original_start_line}-{original_end_line}) has no non-blank lines!")
            # Fallback: use original line numbers (will be wrong but won't crash)
            new_start_line = original_start_line
            new_end_line = original_end_line
        
        # CRITICAL FIX: Calculate positioning based on START line, not center
        # new_start_line is 1-indexed, convert to 0-indexed for positioning
        start_line_0indexed = new_start_line - 1
        end_line_0indexed = new_end_line - 1
        
        # Calculate center line in 0-indexed coordinates
        center_line_0indexed = (start_line_0indexed + end_line_0indexed) / 2.0
        
        # Calculate number of lines in cleaned code
        num_lines_in_cleaned_block = new_end_line - new_start_line + 1
        
        print(f"   🔍 DEBUG: Block {event_idx}: original lines {original_start_line}-{original_end_line}")
        print(f"      → cleaned lines {new_start_line}-{new_end_line} ({num_lines_in_cleaned_block} lines)")
        print(f"      → 0-indexed: lines {start_line_0indexed}-{end_line_0indexed}")
        print(f"      → center_line (0-indexed): {center_line_0indexed:.2f}")
        print(f"      block_height: {block_height:.3f} (should cover all {num_lines_in_cleaned_block} lines)")
        print(f"      line_height used: {line_height:.3f} (for font_size=22, line_spacing=1.0)")
        
        # Build line by line with 12-space indentation (matches if/else block level)
        indent = "            "  # 12 spaces
        # CRITICAL: Block detection gives 1-indexed lines, we convert to 0-indexed
        # But Manim positioning counts from line 0 at the top
        # If block starts at line 8 (1-indexed) = line 7 (0-indexed)
        # We need to position at line 8 in the display, which is index 7+1=8 from top
        # So we use (start_line_0indexed + 1) for positioning
        highlight_positioning += f"{indent}# Position highlight for lines {new_start_line}-{new_end_line} (0-indexed: {start_line_0indexed}-{end_line_0indexed})\n"
        highlight_positioning += f"{indent}# Positioning at display line {start_line_0indexed + 0.5}: full_code.get_top()[1] - ({start_line_0indexed + 0.5} * {line_height:.3f})\n"
        highlight_positioning += f"{indent}# Center of block is at: center_of_start_line - (({num_lines_in_block} - 1) * {line_height:.3f} / 2)\n"
        highlight_positioning += f"{indent}block_{event_idx}_start_line_center_y = full_code.get_top()[1] - ({start_line_0indexed + 0.5} * {line_height:.3f})\n"
        highlight_positioning += f"{indent}block_{event_idx}_center_y = block_{event_idx}_start_line_center_y - (({num_lines_in_block} - 1) * {line_height:.3f} / 2.0)\n"
        highlight_positioning += f"{indent}highlight_{event_idx}.move_to([full_code.get_center()[0], block_{event_idx}_center_y, 0])\n"
        highlight_positioning += f"{indent}highlight_{event_idx}.stretch_to_fit_width(full_code.width + 0.3)\n"
        # CRITICAL: Ensure highlight height matches the actual block height
        # The height is already set in Rectangle creation, but we verify it's correct
        highlight_positioning += f"{indent}# Highlight height: {block_height:.3f} (for {num_lines_in_block} lines)\n"
        highlight_positioning += f"{indent}highlight_{event_idx}.set_z_index(-1)\n"
        highlight_positioning += f"{indent}highlight_{event_idx}.set_opacity(0)\n"
        highlight_positioning += f"{indent}highlight_{event_idx}.set_stroke_opacity(0)\n"
        # Position glow behind main highlight
        highlight_positioning += f"{indent}highlight_glow_{event_idx}.move_to([full_code.get_center()[0], block_{event_idx}_center_y, 0])\n"
        highlight_positioning += f"{indent}highlight_glow_{event_idx}.stretch_to_fit_width(full_code.width + 0.3)\n"
        highlight_positioning += f"{indent}highlight_glow_{event_idx}.set_z_index(-2)\n"
        highlight_positioning += f"{indent}highlight_glow_{event_idx}.set_opacity(0)\n"
        highlight_positioning += f"{indent}highlight_glow_{event_idx}.set_stroke_opacity(0)\n"
    
    timeline_events.sort(key=lambda x: x['start_time'])
    animation_timeline = ""
    # CRITICAL: current_time should NOT include skeleton_sync_wait
    # because skeleton_sync_wait is a self.wait() in the Manim code, not part of fixed_overhead
    current_time = fixed_overhead
    previous_highlight_idx = None
    
    print(f"Starting animation timeline generation...")
    print(f"Initial current_time = {current_time:.2f}s (fixed_overhead)\n")
    
    for event_idx, event in enumerate(timeline_events):
        start_time = event['start_time']
        end_time = event['end_time']
        block = event['code_block']

        # NO FORCED SYNC - Use natural timing from audio
        wait_time = start_time - current_time
        
        # NO ADJUSTMENT NEEDED - skeleton_sync_wait is 0
        
        print(f"Event {event_idx + 1}: {block['type']} (lines {block['start_line']}-{block['end_line']})")
        print(f"  Timeline event start_time: {start_time:.2f}s")
        print(f"  Current Manim time: {current_time:.2f}s")
        print(f"  Calculated wait_time: {wait_time:.2f}s")
        
        # CRITICAL: Calculate scroll_progress based on BLOCK POSITION, not just time
        # This ensures the highlighted block is visible on screen
        # scroll_progress = 0.0 means code at top (first lines visible)
        # scroll_progress = 1.0 means code at bottom (last lines visible)
        
        # Get block position in code - map to cleaned code line numbers
        original_start_line = block['start_line']
        original_end_line = block['end_line']
        
        # Map to cleaned code line numbers
        new_start_line = None
        new_end_line = None
        for orig_line in range(original_start_line, original_end_line + 1):
            if orig_line in line_number_mapping:
                if new_start_line is None:
                    new_start_line = line_number_mapping[orig_line]
                new_end_line = line_number_mapping[orig_line]
        
        if new_start_line is None or new_end_line is None:
            # Fallback
            new_start_line = original_start_line
            new_end_line = original_end_line
        
        block_start_line = new_start_line
        block_end_line = new_end_line
        block_center_line = (block_start_line + block_end_line) / 2
        # CRITICAL: Calculate center_line (0-indexed) for highlight positioning
        # block_start_line and block_end_line are 1-indexed, so center_line = (start + end) / 2 - 1
        center_line = ((block_start_line + block_end_line) / 2.0) - 1  # Convert to 0-indexed
        
        # Use cleaned code (without blank lines) for scroll calculation
        # Since we removed blank lines from display, use cleaned code
        cleaned_code_lines = cleaned_code.split('\n')
        total_code_lines = len(cleaned_code_lines)  # Total lines in cleaned code (no blanks)
        total_non_blank_lines = total_code_lines  # Same as total since blanks are removed
        
        # Estimate visible lines on screen (approximately 15-20 lines depending on font size)
        # This is based on actual rendered lines (non-blank), but blank lines still affect spacing
        visible_lines = 18  # Approximate non-blank lines visible on screen
        
        # Use non-blank line count for scroll calculation (more accurate)
        # But block line numbers are still based on full code (including blank lines)
        effective_total_lines = total_non_blank_lines if total_non_blank_lines > 0 else total_code_lines
        
        # Calculate what scroll_progress is needed to show this block
        # If block is at line 17 and we can show 18 lines, we need scroll_progress = 0.0 (top)
        # If block is at line 30 and total is 40, we need scroll_progress ~= (30-18)/(40-18) = 0.55
        if effective_total_lines > visible_lines:
            # Block position relative to code start (0 = first line, 1 = last line)
            block_position_ratio = (block_center_line - 1) / (total_code_lines - 1) if total_code_lines > 1 else 0
            
            # Calculate scroll_progress to center the block on screen
            # We want the block to be visible, so scroll enough to show it
            # If block is in first half, scroll less; if in second half, scroll more
            if block_center_line <= visible_lines / 2:
                # Block is near top - start at top (scroll_progress = 0.0)
                scroll_progress = 0.0
            else:
                # Block is further down - calculate scroll to show it
                # Use effective_total_lines (non-blank) for calculation, but block_center_line is still from full code
                # Since we're using cleaned code, block_center_line is already in cleaned code
                # No need to count non-blank lines - they're already removed
                effective_block_center = block_center_line - 1  # Convert to 0-indexed
                
                # Calculate scroll_progress based on non-blank lines
                scroll_progress = (effective_block_center - visible_lines / 2) / max(effective_total_lines - visible_lines, 1)
                scroll_progress = min(max(scroll_progress, 0), 1)  # Clamp between 0 and 1
        else:
            # Code fits on screen - no scrolling needed
            scroll_progress = 0.0
        
        print(f"  📍 Block position: lines {block_start_line}-{block_end_line} (center: {block_center_line:.1f})")
        print(f"  📏 Code: {total_code_lines} total lines ({total_non_blank_lines} non-blank), ~{visible_lines} visible on screen")
        print(f"  📊 Calculated scroll_progress: {scroll_progress:.3f} ({scroll_progress * 100:.1f}%)")
        
        if event_idx == 0:
            print(f"  🔝 First event: scroll_progress = {scroll_progress:.3f} (calculated from block position)")
            if wait_time <= 0:
                print(f"  ⚠️  First event wait_time <= 0, forcing to 0.5s")
                wait_time = 0.5
        
        if wait_time > 0:
            print(f"  ✅ Adding wait({wait_time:.2f}s)")
            animation_timeline += f"""            self.wait({wait_time:.2f})
"""
        else:
            print(f"  ⚠️  No wait needed (wait_time <= 0)")
        
        current_time = start_time
        print(f"  Updated current_time to: {current_time:.2f}s\n")
        
        if previous_highlight_idx is not None:
            # Fade out previous highlight smoothly
            animation_timeline += f"""            # Fade out previous highlight
            self.play(
                highlight_{previous_highlight_idx}.animate.set_opacity(0).set_stroke_opacity(0),
                highlight_glow_{previous_highlight_idx}.animate.set_opacity(0).set_stroke_opacity(0),
                run_time=0.3
            )
"""
        
        # scroll_progress is already calculated above based on block position
        # This ensures the highlighted block is visible on screen
        highlight_duration = end_time - start_time
        
        print(f"  Scroll calculation:")
        print(f"    start_time: {start_time:.2f}s (when this highlight should appear)")
        print(f"    fixed_overhead: {fixed_overhead:.2f}s (when code APPEARED, not when scroll starts)")
        print(f"    code_scroll_time: {code_scroll_time:.2f}s (TOTAL scroll window duration)")
        print(f"    Time since code appeared: {start_time - fixed_overhead:.2f}s")
        print(f"    scroll_progress: {scroll_progress:.3f} ({((start_time - fixed_overhead) / code_scroll_time * 100):.1f}% through scroll window)")
        print(f"    Action: Will scroll code to {scroll_progress * 100:.1f}% position (incremental scroll, not continuous)")
        print(f"  Highlight duration: {highlight_duration:.2f}s")
        
        # Animation time: 1.1s (glow 0.2s + border 0.3s + fill 0.2s + pulse 0.4s) or 1.4s with scroll
        remaining_time = highlight_duration - 1.1  # Subtract animation time
        print(f"  Remaining time after animations: {remaining_time:.2f}s (duration {highlight_duration:.2f}s - 1.1s animation)")
        
        if remaining_time > 0:
            print(f"  ✅ Adding wait({remaining_time:.2f}s)")
        else:
            print(f"  ⚠️  No remaining wait (remaining_time <= 0)")
        
        # CRITICAL: Calculate start_line_0indexed for THIS event
        # Don't reuse the variable from the highlight positioning loop!
        event_start_line_0indexed = block_start_line - 1  # Convert to 0-indexed
        event_end_line_0indexed = block_end_line - 1
        event_num_lines = block_end_line - block_start_line + 1
        event_block_height = event_num_lines * line_height
        
        print(f"  🎯 Event highlight positioning:")
        print(f"     Lines {block_start_line}-{block_end_line} (1-indexed)")
        print(f"     Lines {event_start_line_0indexed}-{event_end_line_0indexed} (0-indexed)")
        print(f"     Block height: {event_block_height:.3f} ({event_num_lines} lines × {line_height:.3f})")
        
        # Generate enhanced animation with glow, animated border, and spotlight effect
        # Animation sequence: scroll (0.3s) + glow fade-in (0.2s) + border draw (0.3s) + fill fade-in (0.2s) + pulse (0.4s) = 1.4s or 1.1s
        # CRITICAL: Embed code_scroll_time value directly (not as variable reference)
        
        if use_incremental_reveal:
            # CODE HIKE STYLE: Reveal blocks progressively (code already set up in template)
            # OPACITY STRATEGY:
            # - Skeleton: Always white (1.0)
            # - Previous blocks: White (1.0) - already explained
            # - Current block: White (1.0) + highlighted - being explained
            # - Upcoming blocks: Dim (0.15) - not yet explained
            
            # Reveal the specific lines for this event (progressive reveal)
            animation_timeline += f"""            # CODE HIKE: Reveal block {event_idx + 1} (Lines {event_start_line_0indexed}-{event_end_line_0indexed})
            
            # Identify lines to reveal (bring from dim to full opacity)
            lines_to_reveal = code_lines[{event_start_line_0indexed}:{event_end_line_0indexed + 1}]
            
            # CRITICAL: Also ensure ALL previous lines are white (cumulative reveal)
            # This handles cases where timeline events don't cover every line
            all_lines_up_to_current = code_lines[0:{event_end_line_0indexed + 1}]
            
            # Check for overflow and scroll UP if needed
            target_bottom_y = code_lines[{event_end_line_0indexed}].get_bottom()[1]
            
            if target_bottom_y < -3.0:
                shift_amount = -3.0 - target_bottom_y
                # Shift the ENTIRE code_lines group
                self.play(code_lines.animate.shift(UP * shift_amount), run_time=0.3)
                self.total_scroll_shift += shift_amount
            
            # Reveal ALL lines from start to current block (cumulative)
            # This ensures previous lines stay white even if not explicitly in a timeline event
            self.play(all_lines_up_to_current.animate.set_opacity(1.0), run_time=0.5)
            
            # Position highlight for this block
            # We surround the revealed lines
            # Create a temporary VGroup of the target lines to get bounding box
            target_group = VGroup(*lines_to_reveal)
            target_center = target_group.get_center()
            target_width = target_group.width
            target_height = target_group.height
            
            highlight_{event_idx}.move_to(target_center)
            highlight_{event_idx}.stretch_to_fit_width(target_width + 0.3)
            highlight_{event_idx}.stretch_to_fit_height(target_height + 0.1)
            
            highlight_glow_{event_idx}.move_to(target_center)
            highlight_glow_{event_idx}.stretch_to_fit_width(target_width + 0.3)
            highlight_glow_{event_idx}.stretch_to_fit_height(target_height + 0.1)
            
            # Enhanced highlight animation sequence
            # Step 1: Glow appears first (subtle background glow)
            self.play(
                highlight_glow_{event_idx}.animate.set_stroke_opacity(0.1),
                run_time=0.2
            )
            # Step 2: Animated border drawing effect (stroke appears)
            self.play(
                highlight_{event_idx}.animate.set_stroke_opacity(0.9),
                highlight_glow_{event_idx}.animate.set_stroke_opacity(0.2),
                run_time=0.3
            )
            # Step 3: Fill fades in (background highlight)
            self.play(
                highlight_{event_idx}.animate.set_opacity(0.40).set_stroke_opacity(0.95),
                highlight_glow_{event_idx}.animate.set_stroke_opacity(0.15),
                run_time=0.2
            )
            # Step 4: Subtle pulse animation (breathing effect)
            self.play(
                highlight_{event_idx}.animate.set_opacity(0.35).set_stroke_opacity(0.85),
                highlight_glow_{event_idx}.animate.set_stroke_opacity(0.1),
                run_time=0.2
            )
            self.play(
                highlight_{event_idx}.animate.set_opacity(0.45).set_stroke_opacity(0.95),
                highlight_glow_{event_idx}.animate.set_stroke_opacity(0.15),
                run_time=0.2
            )
            remaining_time = {highlight_duration:.2f} - 1.6
            if remaining_time > 0:
                self.wait(remaining_time)
"""
        else:
            # STANDARD DISPLAY: Use scrolling for long code
            animation_timeline += f"""            # Scroll only if needed (scroll_progress > 0 means actual scrolling)
            scroll_time = 0.3 if {scroll_progress:.3f} > 0 and scroll_distance > 0 else 0.0
            if {scroll_progress:.3f} > 0 and scroll_distance > 0:
                seg_target_y = start_center_y + (scroll_distance * {scroll_progress:.3f})
                # CRITICAL: Scroll FIRST, then get the actual top position
                self.play(full_code.animate.move_to([code_center_x, seg_target_y, 0]), run_time=0.3)
                # Now get the ACTUAL top position after scroll
                code_top_y = full_code.get_top()[1]
                # Position highlight: use (N + 1.5) for scrolling case - testing to move highlight further down
                block_{event_idx}_start_line_center_y = code_top_y - ({event_start_line_0indexed + 1.5} * {line_height:.3f})
                block_{event_idx}_center_y = block_{event_idx}_start_line_center_y - (({event_num_lines} - 1) * {line_height:.3f} / 2.0)
                highlight_{event_idx}.move_to([code_center_x, block_{event_idx}_center_y, 0])
                highlight_glow_{event_idx}.move_to([code_center_x, block_{event_idx}_center_y, 0])
            else:
                # No scrolling - use N+0.5 (Consistent with scrolling case)
                code_top_y = full_code.get_top()[1]
                block_{event_idx}_start_line_center_y = code_top_y - ({event_start_line_0indexed + 0.5} * {line_height:.3f})
                block_{event_idx}_center_y = block_{event_idx}_start_line_center_y - (({event_num_lines} - 1) * {line_height:.3f} / 2.0)
                highlight_{event_idx}.move_to([full_code.get_center()[0], block_{event_idx}_center_y, 0])
                highlight_glow_{event_idx}.move_to([full_code.get_center()[0], block_{event_idx}_center_y, 0])
            
            # Enhanced highlight animation sequence
            # Step 1: Glow appears first (subtle background glow)
            self.play(
                highlight_glow_{event_idx}.animate.set_stroke_opacity(0.1),
                run_time=0.2
            )
            # Step 2: Animated border drawing effect (stroke appears)
            self.play(
                highlight_{event_idx}.animate.set_stroke_opacity(0.9),
                highlight_glow_{event_idx}.animate.set_stroke_opacity(0.2),
                run_time=0.3
            )
            # Step 3: Fill fades in (background highlight)
            self.play(
                highlight_{event_idx}.animate.set_opacity(0.40).set_stroke_opacity(0.95),
                highlight_glow_{event_idx}.animate.set_stroke_opacity(0.15),
                run_time=0.2
            )
            # Step 4: Subtle pulse animation (breathing effect)
            self.play(
                highlight_{event_idx}.animate.set_opacity(0.35).set_stroke_opacity(0.85),
                highlight_glow_{event_idx}.animate.set_stroke_opacity(0.1),
                run_time=0.2
            )
            self.play(
                highlight_{event_idx}.animate.set_opacity(0.45).set_stroke_opacity(0.95),
                highlight_glow_{event_idx}.animate.set_stroke_opacity(0.15),
                run_time=0.2
            )
            remaining_time = {highlight_duration:.2f} - scroll_time - 1.1
            if remaining_time > 0:
                self.wait(remaining_time)
"""
        previous_highlight_idx = event_idx
        current_time = end_time
        print(f"  Updated current_time to: {current_time:.2f}s\n")
    
    if previous_highlight_idx is not None:
        print(f"Fading out last highlight (index {previous_highlight_idx})")
        animation_timeline += f"""            self.play(highlight_{previous_highlight_idx}.animate.set_opacity(0), run_time=0.2)
"""
    
    # CRITICAL: Calculate when code narration actually ends
    # This is the end_time of the last timeline event (when last block narration ends)
    code_narration_end_time = max([event['end_time'] for event in timeline_events]) if timeline_events else current_time
    
    print(f"\n{'='*60}")
    print("🔍 TIMING ANALYSIS")
    print(f"{'='*60}")
    print(f"Final animation current_time: {current_time:.2f}s (when last highlight animation ends)")
    print(f"Code narration end_time: {code_narration_end_time:.2f}s (when last block narration ends in audio)")
    print(f"Total audio duration: {audio_duration:.2f}s")
    print(f"Key concepts start_time: {key_concepts_start_time:.2f}s" if key_concepts_start_time else "Key concepts: None")
    
    # CRITICAL FIX: Ensure code stays visible until code narration ends
    # If current_time < code_narration_end_time, we need to wait
    wait_until_code_ends = max(0, code_narration_end_time - current_time)
    if wait_until_code_ends > 0:
        print(f"⚠️  Code narration continues after highlights end!")
        print(f"   Adding wait({wait_until_code_ends:.2f}s) to keep code visible until narration ends")
        animation_timeline += f"""            # Wait until code narration ends
            self.wait({wait_until_code_ends:.2f})
"""
        current_time = code_narration_end_time
        print(f"   Updated current_time to: {current_time:.2f}s (code narration end)")
    else:
        print(f"✅ Code narration ends at same time as highlights")
    
    print(f"Time difference (audio - code_end): {audio_duration - current_time:.2f}s")
    print(f"{'='*60}\n")
    
    # Calculate time after concepts slide
    # Timeline:
    #   - current_time: Code narration ends (e.g., 16.30s)
    #   - FadeOut code + title: 0.5s (16.30s to 16.80s)
    #   - Wait: 0.2s (16.80s to 17.00s)
    #   - key_concepts_start_time: Key concepts start (e.g., 17.00s)
    #   - Concepts slide animations: ~1.8s (17.00s to 18.80s)
    #   - Concepts display time: concepts_display_time (18.80s to 18.80s + concepts_display_time)
    #   - Total concepts time: 1.8 + concepts_display_time
    concepts_animation_time = 1.8 if key_concepts and concepts_slide_time > 0 else 0
    
    # Calculate when concepts slide ends
    if key_concepts_start_time and concepts_slide_time > 0:
        # Concepts start at key_concepts_start_time, animations take 1.8s, then display for concepts_display_time
        concepts_display_time = concepts_slide_time - 1.8
        concepts_display_time = max(concepts_display_time, 2.0)  # Minimum 2 seconds
        concepts_slide_end_time = key_concepts_start_time + 1.8 + concepts_display_time
    else:
        concepts_slide_end_time = current_time + 0.5 + 0.2  # Just fadeOut + wait if no concepts
    
    # Calculate remaining time to match audio duration
    remaining_time_after_all = 0
    if audio_duration and audio_duration > concepts_slide_end_time:
        remaining_time_after_all = audio_duration - concepts_slide_end_time
        remaining_time_after_all = max(remaining_time_after_all, 0)  # Don't go negative
        if remaining_time_after_all < 0.5:
            remaining_time_after_all = 0  # Too small to matter
    
    print(f"   Concepts slide end time: {concepts_slide_end_time:.2f}s")
    print(f"   Remaining time after concepts: {remaining_time_after_all:.2f}s")
    print(f"\n📝 Generating Manim code string...")
    print(f"🔍 DEBUG: highlight_creation length: {len(highlight_creation)} characters")
    print(f"🔍 DEBUG: highlight_list: {highlight_list}")
    print(f"🔍 DEBUG: Number of timeline events: {len(timeline_events) if timeline_events else 0}")
    print(f"🔍 DEBUG: animation_timeline length: {len(animation_timeline)} characters")
    if len(highlight_list) == 0:
        print(f"   ⚠️  WARNING: No highlights in highlight_list! Highlights will not appear!")
    
    # Generate overview animation code if provided
    if overview_animation and len(overview_animation) > 0:
        # Use the AI-generated animation code directly
        # The animation code is already properly indented (8 spaces)
        actual_overview_duration = overview_duration  # Use the calculated duration
        
        print(f"🎬 Using animated visual overview:")
        print(f"   Duration: {actual_overview_duration:.1f}s")
        print(f"   Animation code length: {len(overview_animation)} chars")
        
        # Ensure proper indentation (8 spaces)
        # We requested flush-left code, so we just indent it
        overview_animation_clean = textwrap.dedent(overview_animation)
        overview_animation_indented = textwrap.indent(overview_animation_clean, "        ")
        
        overview_slides_code = f"""        # ============================================================
        # ANIMATED VISUAL OVERVIEW (like OpenNote.com)
        # Duration: {actual_overview_duration:.1f}s
        # AI-generated visualization of what the code does
        # ============================================================
        
{overview_animation_indented}
        
        # Cleanup overview objects before code display
        # This ensures a clean transition from the held final state
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.5)
        self.wait(0.5)
"""
    elif overview_points and len(overview_points) > 0:
        # Fallback: Use bullet points if animation generation failed
        # Calculate EXACT animation timings to match Manim scene
        title_write = 0.6
        subtitle_fade = 0.4
        wait_after_subtitle = 0.2
        fade_in_per_point = 0.5
        fade_out_all = 0.6
        final_wait = 0.2
        
        total_animation_time = (
            title_write + subtitle_fade + wait_after_subtitle +
            (len(overview_points) * fade_in_per_point) +
            fade_out_all + final_wait
        )
        
        reading_time = overview_duration - total_animation_time
        reading_time = max(reading_time, 2.0)
        
        actual_overview_duration = total_animation_time + reading_time
        
        print(f"🎬 Using fallback bullet point overview:")
        print(f"   Duration: {actual_overview_duration:.1f}s")
        
        # Generate bullet point code (same as before)
        colors = ["#FF6B6B", "#4ECDC4", "##45B7D1", "#FFA07A", "#98D8C8", "#F7DC6F"]
        icons = ["🎯", "⚡", "🔧", "💡", "🚀", "✨"]
        
        overview_slides_code = f"""        # Bullet point overview (fallback)
        overview_title = Text("Code Overview", font_size=48, weight=BOLD, gradient=(BLUE, PURPLE))
        overview_title.to_edge(UP, buff=0.5)
        self.play(Write(overview_title), run_time=0.6)
        
        bullet_points = VGroup()
"""
        
        for idx, point in enumerate(overview_points):
            escaped_point = point.replace('\\', '\\\\').replace('"', '\\"')
            color = colors[idx % len(colors)]
            icon = icons[idx % len(icons)]
            
            overview_slides_code += f"""        point_{idx} = Text("{icon} {escaped_point}", font_size=24, color="{color}")
        bullet_points.add(point_{idx})
"""
        
        overview_slides_code += f"""        
        bullet_points.arrange(DOWN, aligned_edge=LEFT, buff=0.6)
        bullet_points.next_to(overview_title, DOWN, buff=0.8)
        self.play(FadeIn(bullet_points), run_time=1.0)
        self.wait({reading_time:.2f})
        self.play(FadeOut(overview_title), FadeOut(bullet_points), run_time=0.6)
        self.wait(0.2)
        
"""
    else:
        # No overview - minimal wait
        overview_slide_duration = 2.0
        overview_slides_code = f"""        # No overview - minimal wait
        self.wait({overview_slide_duration:.1f})
        
"""
    
    # CRITICAL: Determine if we should use incremental reveal (Code Hike style)
    # Short code (fits on screen) → Show all at once (current behavior)
    # Long code (doesn't fit) → Build incrementally as blocks are narrated
    use_incremental_reveal = False
    
    # Check if code will need scrolling (indicates long code)
    # We'll create a test Text object to measure height
    test_code_lines = cleaned_code.split('\n')
    if len(test_code_lines) > 18:  # Approximate threshold for scrolling
        use_incremental_reveal = True
        print(f"📖 Using Code Hike-style incremental reveal ({len(test_code_lines)} lines)")
    else:
        print(f"📄 Using standard display (code fits on screen, {len(test_code_lines)} lines)")
    
    if use_incremental_reveal:
        # Calculate first narrated line (for skeleton detection)
        first_narrated_line = float('inf')
        for event in timeline_events:
            if event['code_block']['start_line'] < first_narrated_line:
                first_narrated_line = event['code_block']['start_line']
        
        # CODE HIKE STYLE: Build code incrementally
        # CRITICAL: animation_timeline has 12-space indentation (for nested blocks)
        # but Code Hike needs 8-space indentation (direct in construct())
        # We can't use textwrap.dedent because injected code might have 0 indentation (inside strings)
        # So we manually strip 4 spaces from lines that have them (12 -> 8)
        indented_animation_timeline = "\n".join([line[4:] if line.startswith("    ") else line for line in animation_timeline.split("\n")])
        
        code = f"""from manim import *

class CodeExplanationScene(Scene):
    def construct(self):
{overview_slides_code}        
        title = Text("Code Explanation", font_size=38, font="Helvetica", weight=BOLD, color=BLUE)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.5)
        self.wait(0.1)
        
        # CODE HIKE SETUP: Pre-build all code lines IMMEDIATELY (no delay)
        code_lines = VGroup()
        raw_lines = \"\"\"{escaped_code}\"\"\".split('\\n')
        for line in raw_lines:
            # Use non-breaking spaces to preserve indentation
            formatted_line = line.replace(" ", "\\u00A0")
            t = Text(
                formatted_line, 
                font_size=24, 
                font="JetBrains Mono", 
                color=WHITE,
                line_spacing=1.3
            )
            # Scale if too wide
            if t.width > 12:
                t.scale_to_fit_width(12)
            code_lines.add(t)
        
        # Arrange vertically - GUARANTEES vertical stacking
        code_lines.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        
        # Position the whole group at Top-Left
        code_lines.to_edge(UP, buff=1.2)
        code_lines.to_edge(LEFT, buff=0.5)
        
        self.add(code_lines)
        
        # SKELETON-FIRST STRATEGY: Show class/method structure + lines before first narrated block
        # Skeleton = class declaration, method signature, setup code, and final closing braces
        skeleton_indices = []
        
        # First narrated line (calculated during code generation)
        first_narrated_line = {first_narrated_line}
        
        for idx, line in enumerate(raw_lines):
            stripped = line.strip()
            indent_level = len(line) - len(line.lstrip())
            line_num = idx + 1  # Convert to 1-indexed
            
            is_skeleton = False
            
            # Top-level declarations (0 indentation)
            if indent_level == 0:
                is_skeleton = True
            # Method signatures
            elif 'public static void main' in stripped or 'public class' in stripped:
                is_skeleton = True
            # Lines BEFORE first narrated block (setup code)
            elif line_num < first_narrated_line:
                is_skeleton = True
            # Closing braces at the very end (last 3 lines of code)
            elif idx >= len(raw_lines) - 3 and stripped in ['{{}}', '{{}}}}']:
                is_skeleton = True
            
            if is_skeleton:
                skeleton_indices.append(idx)
        
        # Show skeleton immediately (full opacity)
        for idx in skeleton_indices:
            code_lines[idx].set_opacity(1.0)
        
        # Dim all other lines (will be revealed progressively)
        for idx in range(len(raw_lines)):
            if idx not in skeleton_indices:
                code_lines[idx].set_opacity(0.15)  # Very dim, but visible
        
        # Initialize scroll tracker
        self.total_scroll_shift = 0.0
        
        # Wait to sync with first timeline event (narration start)
        {"self.wait(" + str(skeleton_sync_wait) + ")" if skeleton_sync_wait > 0 else "# No wait needed - immediate sync"}

        
{highlight_creation}
        
{indented_animation_timeline}
        
        # Fade out all code blocks and title
        self.play(FadeOut(code_lines), FadeOut(title), run_time=0.5)
        self.wait(0.2)
        
{concepts_slide_code}
        
        # Ensure video duration matches audio duration
        if {remaining_time_after_all:.2f} > 0:
            self.wait({remaining_time_after_all:.2f})
"""
    else:
        # STANDARD DISPLAY: Show all code at once (short code)
        code = f"""from manim import *

class CodeExplanationScene(Scene):
    def construct(self):
{overview_slides_code}        
        title = Text("Code Explanation", font_size=38, font="Helvetica", weight=BOLD, color=BLUE)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.5)
        self.wait(0.1)
        
        full_code = Text(
            \"\"\"{escaped_code}\"\"\",
            font_size=28,
            font="IBM Plex Mono",
            color=WHITE,
            line_spacing=1.3
        )
        
        if full_code.width > 13:
            full_code.scale_to_fit_width(13)
        
        full_code.to_edge(LEFT, buff=0.5)
        
{highlight_creation}
        
        available_screen_height = 5.5
        code_height = full_code.height
        
        if code_height > available_screen_height:
            code_center_x = full_code.get_left()[0] + full_code.width/2
            start_center_y = 2.5 - (code_height / 2)
            full_code.move_to([code_center_x, start_center_y, 0])
            end_center_y = -3.5 + (code_height / 2)
            scroll_distance = end_center_y - start_center_y
            
{highlight_positioning}
            
            # Add highlights to scene
            all_highlights = VGroup({', '.join(highlight_list) if highlight_list else ''})
            self.add(all_highlights)
            self.play(FadeIn(full_code), run_time=0.5)
            # CRITICAL: Wait 3 seconds to match audio padding
            self.wait(3.0)
            
{animation_timeline}
        else:
            # Code fits on screen - still need highlights!
            full_code.next_to(title, DOWN, buff=0.3)
            
            # Define variables needed for animation timeline (even without scrolling)
            code_center_x = full_code.get_center()[0]
            start_center_y = full_code.get_center()[1]
            scroll_distance = 0  # No scrolling needed
            
            # Position highlights even when code doesn't scroll
            
{highlight_positioning}
            
            # Add highlights to scene
            all_highlights = VGroup({', '.join(highlight_list) if highlight_list else ''})
            self.add(all_highlights)
            
            self.play(FadeIn(full_code), run_time=0.5)
            # CRITICAL: Wait 3 seconds to match audio padding
            self.wait(3.0)
            
            # Run animation timeline for highlights (even without scrolling)
            
{animation_timeline}
        
        # CRITICAL: Only fade out code when code narration has ended
        # Code narration ends at code_narration_end_time, which is already reached by animation_timeline
        # So we can fade out immediately, but ensure we're at the right time
        self.play(FadeOut(full_code), FadeOut(title), run_time=0.5)
        self.wait(0.2)
        
{concepts_slide_code}
        
        # Ensure video duration matches audio duration
        # Add remaining time to match full audio duration
        if {remaining_time_after_all:.2f} > 0:
            self.wait({remaining_time_after_all:.2f})
"""
    print(f"✅ Manim code string generated ({len(code)} characters)\n")
    return code


def generate_code_display_code(code_content, audio_duration=None, narration_segments=None, key_concepts=None, key_concepts_start_time=None, overview_points=None, overview_duration=40.0):
    """
    Generate Manim code for scrolling code display - NO SLIDES, continuous scroll
    Code starts below title and scrolls UP (code moves UP) to reveal content below
    Adds final slide with key concepts if provided, synchronized with audio
    Optionally shows overview slides at the beginning if overview_points provided
    """
    # Use original code (no cleaning - user won't pass comments)
    lines = code_content.strip().split('\n')
    
    # Escape code for Python string
    escaped_code = code_content.replace('\\', '\\\\').replace('"', '\\"')
    
    # Calculate timing for code scrolling
    # Fixed overhead: 1.2s (initial wait) + 0.5s (title animation) + 1.0s (wait after title) + 0.5s (FadeIn code) = 3.2s
    fixed_overhead = 1.2 + 0.5 + 1.0 + 0.5
    
    if audio_duration and key_concepts_start_time:
        # Code scrolling should finish exactly when key concepts start
        # Transition: FadeOut (0.5s) + small wait (0.2s) = 0.7s
        transition_time = 0.7
        # Code should finish at key_concepts_start_time, so:
        code_scroll_time = key_concepts_start_time - fixed_overhead - transition_time
        # Ensure minimum scroll time
        code_scroll_time = max(code_scroll_time, 5.0)
        
        # Remaining time for key concepts slide (starts after transition)
        concepts_slide_time = audio_duration - key_concepts_start_time
        concepts_slide_time = max(concepts_slide_time, 3.0)  # Minimum 3 seconds
    elif audio_duration:
        # No key concepts - use all time for code
        available_time = audio_duration - fixed_overhead - 1.0  # Final wait
        code_scroll_time = max(available_time, 5.0)
        concepts_slide_time = 0
    else:
        # Fallback timing
        code_scroll_time = len(lines) * 0.5
        concepts_slide_time = 0
    
    # Build concepts slide code
    if key_concepts and concepts_slide_time > 0:
        # Escape concepts for Python string
        concepts_list = "[" + ", ".join(['"' + c.replace('"', '\\"') + '"' for c in key_concepts]) + "]"
        # Animation overhead: title (0.5s), wait (0.3s), items (1.0s) = 1.8s
        concepts_display_time = concepts_slide_time - 1.8
        concepts_display_time = max(concepts_display_time, 2.0)  # Minimum 2 seconds
        
        concepts_slide_code = f"""
        # Key Concepts slide (synchronized with audio)
        concepts_title = Text(
            "Key Concepts",
            font_size=38,
            font="Helvetica",
            weight=BOLD,
            color=BLUE
        )
        concepts_title.to_edge(UP, buff=0.4)
        
        # Create bullet points for concepts
        concept_items = VGroup(*[
            Text(
                f"• {{concept}}",
                font_size=24,
                font="Helvetica",
                color=WHITE
            )
            for concept in {concepts_list}
        ])
        concept_items.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        concept_items.next_to(concepts_title, DOWN, buff=0.6)
        concept_items.to_edge(LEFT, buff=0.8)
        
        # Show concepts (synchronized with audio)
        self.play(Write(concepts_title), run_time=0.5)
        self.wait(0.3)
        self.play(Write(concept_items), run_time=1.0)
        self.wait({concepts_display_time})  # Display for remaining audio time
"""
    else:
        concepts_slide_code = """
        # No concepts - just wait
        self.wait(1)
"""
    
    # Generate overview slides code if overview_points provided
    if overview_points and len(overview_points) > 0:
        # Calculate timing for overview slides
        time_per_slide = overview_duration / max(len(overview_points), 1)
        
        overview_slides_code = f"""        # Overview Slides ({overview_duration:.1f}s total)
        # Show {len(overview_points)} overview points about the code
        
        # Overview title
        overview_title = Text(
            "Code Overview",
            font_size=42,
            font="Helvetica",
            weight=BOLD,
            color=BLUE
        )
        overview_title.to_edge(UP, buff=0.5)
        self.play(Write(overview_title), run_time=0.5)
        self.wait(0.5)
        
"""
        
        # Add each overview point as a slide
        for idx, point in enumerate(overview_points):
            # Escape the point text for Python string
            escaped_point = point.replace('\\', '\\\\').replace('"', '\\"')
            
            overview_slides_code += f"""        # Overview Point {idx + 1}
        point_{idx} = Text(
            "• {escaped_point}",
            font_size=28,
            font="Helvetica",
            color=WHITE
        )
        point_{idx}.next_to(overview_title, DOWN, buff=1.0)
        point_{idx}.to_edge(LEFT, buff=1.0)
        self.play(FadeIn(point_{idx}), run_time=0.4)
        self.wait({time_per_slide - 0.8:.2f})
        self.play(FadeOut(point_{idx}), run_time=0.4)
        
"""
        
        overview_slides_code += f"""        # Fade out overview title
        self.play(FadeOut(overview_title), run_time=0.5)
        self.wait(0.5)
        
"""
    else:
        # No overview points - just wait
        overview_slides_code = f"""        # No overview slides - waiting {overview_duration:.1f}s
        self.wait({overview_duration:.1f})
        
"""
    
    # Create full code text
    code = f"""from manim import *

class CodeExplanationScene(Scene):
    def construct(self):
{overview_slides_code}        
        # Title
        title = Text(
            "Code Explanation",
            font_size=38,
            font="Helvetica",
            weight=BOLD,
            color=BLUE
        )
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.5)
        self.wait(0.1)  # Small wait after title
        
        # Create full code text (larger font, no gaps)
        full_code = Text(
            \"\"\"{escaped_code}\"\"\",
            font_size=22,
            font="Courier",
            color=WHITE,
            line_spacing=1.0
        )
        
        # Ensure code fits on screen width
        if full_code.width > 13:
            full_code.scale_to_fit_width(13)
        
        # Position code below title - CRITICAL: start with TOP of code just below title
        full_code.to_edge(LEFT, buff=0.5)
        
        # Available screen height: from below title (Y~2.5) to bottom (Y~-3.5) = ~5.5 units
        available_screen_height = 5.5
        code_height = full_code.height
        
        if code_height > available_screen_height:
            # Code needs scrolling - start with TOP of code at Y=2.5 (just below title)
            code_center_x = full_code.get_left()[0] + full_code.width/2
            
            # START position: Top of code at Y=2.5 (just below title)
            # Center Y = top Y - (code_height / 2)
            start_center_y = 2.5 - (code_height / 2)
            full_code.move_to([code_center_x, start_center_y, 0])
            
            # END position: Bottom of code at Y=-3.5 (screen bottom)
            # Center Y = bottom Y + (code_height / 2)
            end_center_y = -3.5 + (code_height / 2)
            scroll_distance = end_center_y - start_center_y
            
            # Show code - NO continuous scroll, we'll scroll incrementally per highlight
            self.play(FadeIn(full_code), run_time=0.5)
        else:
            # Code fits on screen - position it below title
            full_code.next_to(title, DOWN, buff=0.3)
            self.play(FadeIn(full_code), run_time=0.5)
            # No scrolling needed - just wait for highlights
            scroll_distance = 0
        
        # Fade out code (transition to key concepts - synchronized with audio)
        self.play(FadeOut(full_code), FadeOut(title), run_time=0.5)
        self.wait(0.2)  # Quick transition
        
        # Final slide: Key Concepts (if provided) - appears exactly when audio explains them
{concepts_slide_code}
```
"""
    
    return code


def code_to_video(code_content: str, output_name: str = "output", audio_language: str = "english", add_subtitles: bool = False):
    """
    Convert code to video with audio explanation
    
    Works for:
    - Python code
    - JavaScript code
    - Any programming code
    
    Args:
        code_content: The code you want to explain
        output_name: Name for output file
        audio_language: Language for audio narration
        add_subtitles: If True, generate and embed SRT subtitles using Whisper
        
    Returns:
        dict with 'final_video' path or None if failed
    """
    try:
        print(f"\n{'='*60}")
        print(f"💻 CODE TO VIDEO: {output_name}")
        print(f"{'='*60}\n")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ Set OPENAI_API_KEY environment variable")
            return None
        
        client = openai.OpenAI(api_key=api_key)
        
        # CRITICAL: Remove comments BEFORE parsing blocks
        # This ensures block line numbers match the cleaned code that will be displayed
        print("🧹 Removing comments from code...")
        def remove_comments(line):
            """Remove // and /* */ comments from a line"""
            if '//' in line:
                in_string = False
                quote_char = None
                for i, char in enumerate(line):
                    if char in ['"', "'"] and (i == 0 or line[i-1] != '\\'):
                        if not in_string:
                            in_string = True
                            quote_char = char
                        elif char == quote_char:
                            in_string = False
                            quote_char = None
                    elif char == '/' and i < len(line) - 1 and line[i+1] == '/' and not in_string:
                        return line[:i].rstrip()
                return line
            return line
        
        # Remove comments AND blank lines before parsing
        # This ensures block line numbers match the final displayed code
        original_lines = code_content.strip().split('\n')
        code_without_comments_lines = [remove_comments(line) for line in original_lines]
        
        # Remove blank lines too
        code_cleaned_lines = [line for line in code_without_comments_lines if line.strip()]
        code_cleaned = '\n'.join(code_cleaned_lines)
        
        print(f"   Original lines: {len(original_lines)}")
        print(f"   After comment removal: {len(code_without_comments_lines)} lines")
        print(f"   After blank line removal: {len(code_cleaned_lines)} lines")
        
        # Step 0: Parse code into blocks (on CLEANED code - no comments, no blank lines)
        print("📦 Parsing code into semantic blocks...")
        language = "python"
        if "public class" in code_content or "public static" in code_content:
            language = "java"
        elif "function" in code_content and "{" in code_content:
            language = "javascript"
        
        # Parse blocks on fully cleaned code
        code_blocks = parse_code_to_blocks(code_cleaned, language=language)
        print(f"✅ Found {len(code_blocks)} code blocks\n")
        
        # DEBUG: Print all detected blocks BEFORE filtering
        print(f"🔍 DEBUG: All detected blocks BEFORE filtering:")
        for i, block in enumerate(code_blocks, 1):
            block_size = block['end_line'] - block['start_line'] + 1
            print(f"   Block {i}: {block['type']} (lines {block['start_line']}-{block['end_line']}, size: {block_size} lines)")
        print()
        
        # ============================================================
        # CRITICAL: Separate blocks for NARRATION vs HIGHLIGHTING
        # ============================================================
        # NARRATION: All blocks should get narration (class, function, all loops)
        # HIGHLIGHTING: Only blocks that should be visually highlighted
        #
        # Rules for highlighting:
        # 1. If block CONTAINS other valid blocks → DON'T highlight it, highlight its children
        # 2. If block does NOT contain other valid blocks → Highlight it
        # 3. Size does NOT matter - only containment matters
        # ============================================================
        
        # Step 1: Keep ALL blocks for narration
        all_blocks_for_narration = code_blocks.copy()
        print(f"📝 Blocks for NARRATION: {len(all_blocks_for_narration)} (ALL blocks)")
        for i, block in enumerate(all_blocks_for_narration, 1):
            block_size = block['end_line'] - block['start_line'] + 1
            print(f"   Narration {i}: {block['type']} (lines {block['start_line']}-{block['end_line']}, {block_size} lines)")
        print()
        
        # Step 2: Filter blocks for HIGHLIGHTING based on containment logic
        blocks_for_highlights = []
        
        for block in code_blocks:
            # CRITICAL: +1 because both start_line and end_line are INCLUSIVE
            # Example: start_line=3, end_line=7 means lines 3,4,5,6,7 = 5 lines
            # Without +1: 7-3=4 (WRONG!) | With +1: 7-3+1=5 (CORRECT!)
            block_size = block['end_line'] - block['start_line'] + 1
            
            if block['type'] == 'class':
                # Class blocks: Skip for highlighting (too large, contains everything)
                print(f"   ❌ {block['type']} (lines {block['start_line']}-{block['end_line']}): NARRATION ✅, HIGHLIGHT ❌ (too large)")
                continue
            else:
                # For ALL other block types (function, for_loop, while_loop, if_statement, 
                # elif_statement, else_statement, switch, try, catch, etc.):
                # Apply the same containment logic:
                # - If block CONTAINS other blocks → Skip highlighting (highlight children instead)
                # - If block does NOT contain other blocks → Highlight it
                
                contains_other_blocks = False
                contained_blocks = []
                for other_block in code_blocks:
                    if other_block != block:
                        # Check if other_block is INSIDE this block
                        if (block['start_line'] < other_block['start_line'] and 
                            block['end_line'] > other_block['end_line']):
                            contains_other_blocks = True
                            contained_blocks.append(other_block)
                            print(f"   🔍 {block['type']} (lines {block['start_line']}-{block['end_line']}) CONTAINS {other_block['type']} (lines {other_block['start_line']}-{other_block['end_line']})")
                
                if contains_other_blocks:
                    # Special rule for loops: If it's a loop and contains only conditionals/statements (no inner loops), highlight it
                    if 'loop' in block['type']:
                        has_inner_loop = any('loop' in b['type'] for b in contained_blocks)
                        if not has_inner_loop:
                            blocks_for_highlights.append(block)
                            print(f"   ✅ {block['type']} (contains non-loop blocks): NARRATION ✅, HIGHLIGHT ✅ (loop preference)")
                            continue
                    
                    # Special rule for if_statement: Highlight if it doesn't contain control structures
                    # This allows if statements inside functions to be highlighted
                    if block['type'] == 'if_statement':
                        has_control_structure = any(
                            b['type'] in ['for_loop', 'while_loop', 'function', 'class', 'if_statement'] 
                            for b in contained_blocks
                        )
                        if not has_control_structure:
                            blocks_for_highlights.append(block)
                            print(f"   ✅ {block['type']} (contains only simple statements): NARRATION ✅, HIGHLIGHT ✅ (standalone if)")
                            continue

                    # This block contains other blocks → DON'T highlight it
                    # Its children will be highlighted instead
                    print(f"   ❌ {block['type']} (lines {block['start_line']}-{block['end_line']}): NARRATION ✅, HIGHLIGHT ❌ (contains {len(contained_blocks)} other blocks)")
                else:
                    # Special case: if_statement that doesn't contain any blocks should always be highlighted
                    # This catches if statements inside functions that don't have nested blocks
                    if block['type'] == 'if_statement':
                        blocks_for_highlights.append(block)
                        print(f"   ✅ {block['type']} (lines {block['start_line']}-{block['end_line']}): NARRATION ✅, HIGHLIGHT ✅ (standalone if, no nested blocks)")
                    else:
                        # This block does NOT contain other blocks → Highlight it
                        blocks_for_highlights.append(block)
                        print(f"   ✅ {block['type']} (lines {block['start_line']}-{block['end_line']}): NARRATION ✅, HIGHLIGHT ✅ (doesn't contain other blocks)")
        
        # DEBUG: Print summary
        print(f"\n📊 SUMMARY:")
        print(f"   📝 Blocks for NARRATION: {len(all_blocks_for_narration)} (ALL blocks)")
        print(f"   🎯 Blocks for HIGHLIGHTING: {len(blocks_for_highlights)} (only blocks that don't contain others)")
        print()
        
        # CRITICAL: Sort blocks to ensure outer blocks come before nested blocks
        # This ensures narration and highlighting happen in the correct order
        def is_outer_block(block, block_list):
            """Check if this block is not contained within any other block in the list"""
            for other_block in block_list:
                if other_block != block:
                    # Check if this block is INSIDE other_block
                    # Need strict containment: other_block starts before AND ends after
                    if (other_block['start_line'] < block['start_line'] and 
                        other_block['end_line'] >= block['end_line']):
                        return False
                    # Also check equal start with longer end
                    if (other_block['start_line'] == block['start_line'] and 
                        other_block['end_line'] > block['end_line']):
                        return False
            return True
        
        # Sort blocks for narration: outer blocks first, then nested blocks
        outer_narration_blocks = [b for b in all_blocks_for_narration if is_outer_block(b, all_blocks_for_narration)]
        nested_narration_blocks = [b for b in all_blocks_for_narration if not is_outer_block(b, all_blocks_for_narration)]
        outer_narration_blocks.sort(key=lambda x: x['start_line'])
        nested_narration_blocks.sort(key=lambda x: x['start_line'])
        all_blocks_for_narration = outer_narration_blocks + nested_narration_blocks
        
        # Sort blocks for highlighting: outer blocks first, then nested blocks
        outer_highlight_blocks = [b for b in blocks_for_highlights if is_outer_block(b, blocks_for_highlights)]
        nested_highlight_blocks = [b for b in blocks_for_highlights if not is_outer_block(b, blocks_for_highlights)]
        outer_highlight_blocks.sort(key=lambda x: x['start_line'])
        nested_highlight_blocks.sort(key=lambda x: x['start_line'])
        blocks_for_highlights = outer_highlight_blocks + nested_highlight_blocks
        
        print(f"🔍 DEBUG: Final block order for NARRATION:")
        for i, block in enumerate(all_blocks_for_narration, 1):
            block_size = block['end_line'] - block['start_line'] + 1
            print(f"   Narration {i}: {block['type']} (lines {block['start_line']}-{block['end_line']}, {block_size} lines)")
        print(f"🔍 DEBUG: Final block order for HIGHLIGHTS:")
        for i, block in enumerate(blocks_for_highlights, 1):
            block_size = block['end_line'] - block['start_line'] + 1
            print(f"   Highlight {i}: {block['type']} (lines {block['start_line']}-{block['end_line']}, {block_size} lines)")
        print()
        
        # Initialize overview_animation
        overview_animation = None
        
        if len(all_blocks_for_narration) == 0:
            print("⚠️  No code blocks found, using simple scrolling\n")
            
            # Generate a general explanation for the code
            print("🎙️ Creating general narration...")
            narration_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert programming educator. Explain this code clearly and concisely. Focus on what the code does overall."
                    },
                    {
                        "role": "user",
                        "content": f"Explain this code:\n\n```\n{code_content}\n```"
                    }
                ],
                temperature=0.7
            )
            narration_text = narration_response.choices[0].message.content.strip()
            
            # Generate audio
            audio_file = f"{output_name}_audio.aiff"
            generate_audio_for_language(narration_text, audio_language, audio_file, client)
            
            # Get duration
            audio_duration = get_audio_duration(audio_file)
            if not audio_duration:
                audio_duration = len(narration_text) * 0.06
            
            # No key concepts for simple fallback
            key_concepts = None
            key_concepts_start_time = None
            overview_points = None  # No overview for simple fallback
            overview_duration = 40.0  # Default duration
            
            manim_code = generate_code_display_code(code_content, audio_duration, None, key_concepts, key_concepts_start_time, overview_points, overview_duration)
        else:
            # Step 0.5: Generate ANIMATED VISUAL OVERVIEW (like OpenNote.com)
            print("🎨 Creating animated visual overview...")
            
            # PROMPT CHAIN STEP 1: Analyze what the code does visually
            print("   Step 1: Analyzing code for visual representation...")
            visual_analysis_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert at understanding code and identifying visual patterns.
                        
                        Analyze the code and determine:
                        1. What visual output or pattern does this code create? (e.g., pyramid, spiral, graph, tree, animation)
                        2. What are the key steps or components?
                        3. How can this be visualized in an animation?

                        Return JSON format:
                        {
                        "visual_concept": "brief description of what to visualize",
                        "animation_type": "pyramid|spiral|graph|tree|sorting|other",
                        "key_steps": ["step1", "step2", "step3"],
                        "complexity": "simple|medium|complex"
                        }"""
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this code for visual representation:\n\n```\n{code_content}\n```"
                    }
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            import json
            visual_analysis = json.loads(visual_analysis_response.choices[0].message.content)
            visual_concept = visual_analysis.get("visual_concept", "code visualization")
            animation_type = visual_analysis.get("animation_type", "other")
            key_steps = visual_analysis.get("key_steps", [])
            complexity = visual_analysis.get("complexity", "medium")
            
            print(f"   ✅ Visual concept: {visual_concept}")
            print(f"   ✅ Animation type: {animation_type}")
            print(f"   ✅ Key steps: {len(key_steps)}")
            
            
            
            # STEP 2: AI-FIRST APPROACH (works for ANY code)
            print("   Step 2: AI generates animation (fallback to text slides if fails)...")
            
            # Step 2a: Generate SHORT narration
            print("   Step 2a: Generating narration...")
            narration_prompt = f"""Create a VERY SHORT summary of what this code does.

Code: {code_content[:400]}
Visual concept: {visual_concept}

RULES:
1. Max 40-50 words.
2. Explain the CONCEPT clearly.
3. Keep it under 20 seconds."""

            narration_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert programming educator."},
                    {"role": "user", "content": narration_prompt}
                ],
                temperature=0.7
            )
            overview_narration = narration_response.choices[0].message.content.strip()
            
            # Generate audio
            overview_audio_file = f"{output_name}_overview_audio.aiff"
            print(f"   Generating audio for overview...")
            generate_audio_for_language(overview_narration, audio_language, overview_audio_file, client)
            
            actual_audio_duration = get_audio_duration(overview_audio_file)
            if not actual_audio_duration:
                actual_audio_duration = len(overview_narration) * 0.06
            
            print(f"   📊 Overview narration audio: {actual_audio_duration:.2f}s")
            
            # Calculate target duration
            max_padding = 3.0
            target_animation_duration = max(actual_audio_duration - max_padding, 10.0)
            
            # Step 2b: AI generates animation
            print(f"   Step 2b: AI generating animation ({target_animation_duration:.2f}s target)...")
            
            animation_prompt = f"""Generate Manim code to visualize: {visual_concept}

MUST INCLUDE:
1. Title at top
2. Visual elements that fit properly
3. Connecting labels (arrows, text like "Compare", "Check", etc.)
4. NO OVERLAPPING TEXT - use .next_to() or .shift() to position elements
5. Exact duration: {target_animation_duration:.1f} seconds

Here is a CONCRETE EXAMPLE for anagram checking:

```python
# Title
title = Text("Anagram Check", font_size=36, color=BLUE).to_edge(UP)
self.play(Write(title), run_time=1.5)
self.wait(0.5)

# Show two strings HORIZONTALLY
str1 = Text("'listen'", font_size=28, color=YELLOW).shift(UP*1 + LEFT*3)
str2 = Text("'silent'", font_size=28, color=YELLOW).shift(UP*1 + RIGHT*3)
self.play(Write(str1), Write(str2), run_time=2.0)
self.wait(0.5)

# Step label - positioned ABOVE to avoid overlap
step1 = Text("1. Check Length", font_size=22, color=GREEN).shift(UP*0.2)
self.play(Write(step1), run_time=1.0)
self.wait(0.5)

# Count boxes HORIZONTALLY (text fits with font_size=20-24)
box1 = Rectangle(width=4, height=1.2, color=BLUE).shift(DOWN*1 + LEFT*3)
label1 = Text("Count 1", font_size=22).next_to(box1, UP)  # .next_to() prevents overlap!
count1 = Text("{{l:1, i:1...}}", font_size=20).move_to(box1)
self.play(Create(box1), Write(label1), run_time=1.5)
self.play(Write(count1), run_time=1.0)

box2 = Rectangle(width=4, height=1.2, color=BLUE).shift(DOWN*1 + RIGHT*3)
label2 = Text("Count 2", font_size=22).next_to(box2, UP)  # .next_to() prevents overlap!
count2 = Text("{{s:1, i:1...}}", font_size=20).move_to(box2)
self.play(Create(box2), Write(label2), run_time=1.5)
self.play(Write(count2), run_time=1.0)

# Connecting label with arrow - positioned ABOVE arrow
arrow = Arrow(box1.get_right(), box2.get_left(), color=GREEN)
compare = Text("Compare", font_size=24, color=GREEN).next_to(arrow, UP)  # .next_to()!
self.play(GrowArrow(arrow), Write(compare), run_time=1.0)

# Fill remaining time to reach EXACTLY {target_animation_duration:.1f} seconds
self.wait(1.0)
```

EXAMPLE for pyramid/pattern (NO OVERLAP + DESCRIPTIVE LABELS):

```python
# Title
title = Text("Number Pyramid", font_size=36, color=BLUE).to_edge(UP)
self.play(Write(title), run_time=1.5)
self.wait(0.5)

# Explain the concept
concept = Text("Building rows with nested loops", font_size=22, color=GREEN).shift(UP*2)
self.play(Write(concept), run_time=1.5)
self.wait(0.5)

# Row 1 - numbers on LEFT, descriptive label on RIGHT
row1 = Text("1", font_size=32, color=YELLOW).shift(UP*0.8 + LEFT*2)
label1 = Text("Row 1: Outer loop i=1", font_size=20, color=GREEN).next_to(row1, RIGHT, buff=1.5)
self.play(Write(row1), Write(label1), run_time=1.0)
self.wait(0.5)

# Row 2 - positioned BELOW row1
row2 = VGroup(Text("1", font_size=32), Text("2", font_size=32)).arrange(RIGHT, buff=0.3).shift(DOWN*0.2 + LEFT*2)
label2 = Text("Row 2: Inner loop prints 1,2", font_size=20, color=GREEN).next_to(row2, RIGHT, buff=1.5)
self.play(Write(row2), Write(label2), run_time=1.0)
self.wait(0.5)

# Row 3 - positioned BELOW row2
row3 = VGroup(*[Text(str(i), font_size=32) for i in range(1,4)]).arrange(RIGHT, buff=0.3).shift(DOWN*1.2 + LEFT*2)
label3 = Text("Row 3: Prints 1,2,3", font_size=20, color=GREEN).next_to(row3, RIGHT, buff=1.5)
self.play(Write(row3), Write(label3), run_time=1.0)
self.wait(0.5)

# Summary
summary = Text("Pattern grows row by row", font_size=22, color=BLUE).shift(DOWN*2.5)
self.play(Write(summary), run_time=1.0)

# Total animation time so far: ~10.5 seconds
# Target: {target_animation_duration:.1f} seconds
# Remaining: {target_animation_duration:.1f} - 10.5 = X seconds
self.wait({target_animation_duration:.1f} - 10.5)  # Fill remaining time to reach EXACTLY {target_animation_duration:.1f}s
```

Now generate similar code for: {visual_concept}

Code to visualize:
```
{code_content[:500]}
```

CRITICAL:
- **DURATION MUST BE EXACTLY {target_animation_duration:.1f} SECONDS** - This is CRITICAL!
- Calculate total time of all self.play() and self.wait() calls
- Add final self.wait(X) to reach EXACTLY {target_animation_duration:.1f} seconds
- Example: If animations take 8s, add self.wait({target_animation_duration:.1f} - 8.0) at the end
- Use font_size=20-24 for text inside boxes (NOT 16!)
- Arrange elements HORIZONTALLY (side by side)
- Add DESCRIPTIVE labels explaining what's happening (e.g., "Outer loop i=1", "Comparing values", "Building row")
- NOT just generic labels like "Level 1", "Step 2" - explain the CODE LOGIC!
- PREVENT OVERLAP: Use .next_to(element, direction, buff=0.5) or .shift()
- For pyramids/lists: Put labels to the RIGHT or LEFT of elements, NOT on top
- Title MUST be first
- NO class/def lines
- FLUSH-LEFT (no indentation)
- Do NOT FadeOut
- **AGAIN: Total duration MUST be {target_animation_duration:.1f} seconds - use self.wait() to fill remaining time!**

Return ONLY the animation code:"""

            ai_success = False
            try:
                animation_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a Manim expert. Follow the example structure exactly."},
                        {"role": "user", "content": animation_prompt}
                    ],
                    temperature=0.5,
                    max_tokens=1500
                )
                
                ai_code = animation_response.choices[0].message.content.strip()
                
                # Clean markdown
                if "```python" in ai_code:
                    ai_code = ai_code.split("```python")[1].split("```")[0].strip()
                elif "```" in ai_code:
                    ai_code = ai_code.split("```")[1].split("```")[0].strip()
                
                # Remove class/def lines
                lines = ai_code.split('\n')
                cleaned = []
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('class ') or stripped.startswith('def '):
                        continue
                    cleaned.append(line)
                ai_code = '\n'.join(cleaned)
                
                # Dedent
                import textwrap
                ai_code = textwrap.dedent(ai_code).strip()
                
                # VALIDATE: Check for title
                has_title = 'title' in ai_code.lower() and 'Text(' in ai_code
                if not has_title:
                    print(f"   ⚠️  AI code missing title, adding it...")
                    title_code = f"""# Title
title = Text("{visual_concept[:40]}", font_size=36, color=BLUE).to_edge(UP)
self.play(Write(title), run_time=1.5)
self.wait(0.5)

"""
                    ai_code = title_code + ai_code
                
                # VALIDATE: Try to compile
                try:
                    compile(ai_code, '<string>', 'exec')
                    print(f"   ✅ AI animation validated ({len(ai_code)} chars)")
                    
                    # Indent properly
                    lines = ai_code.split('\n')
                    indented = []
                    for line in lines:
                        if line.strip():
                            indented.append('        ' + line)
                        else:
                            indented.append('')
                    overview_animation_code = '\n'.join(indented)
                    ai_success = True
                    
                except SyntaxError as e:
                    print(f"   ⚠️  AI code has syntax error: {e}")
                    ai_success = False
                    
            except Exception as e:
                print(f"   ⚠️  AI generation failed: {e}")
                ai_success = False
            
            # Step 2c: Fallback if AI failed
            if not ai_success:
                print(f"   ⚠️  Using multi-slide text fallback")
                
                # FALLBACK: Multi-slide text explanation
                slides = []
                words = overview_narration.split()
                chunk_size = 8
                for i in range(0, len(words), chunk_size):
                    chunk = ' '.join(words[i:i+chunk_size])
                    slides.append(chunk)
                
                slide_duration = target_animation_duration / max(len(slides), 1)
                
                overview_animation_code = f"""        # Multi-Slide Text Overview
        title = Text("{visual_concept[:50]}", font_size=36, color=BLUE).to_edge(UP)
        self.play(Write(title), run_time=1.5)
        self.wait(0.5)
"""
                
                for idx, slide_text in enumerate(slides):
                    # Escape quotes in slide text
                    slide_text_escaped = slide_text.replace('"', '\\"')
                    overview_animation_code += f"""
        # Slide {idx + 1}
        slide{idx} = Text("{slide_text_escaped}", font_size=24, color=WHITE).move_to(ORIGIN)
        self.play(FadeIn(slide{idx}), run_time=1.0)
        self.wait({max(0.1, slide_duration - 1.5):.2f})
        self.play(FadeOut(slide{idx}), run_time=0.5)
"""
            
            # Add cleanup
            overview_animation_code += """
        # Cleanup
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.5)
        self.wait(0.5)
"""
            
            print(f"   ✅ Overview animation ready")
            
            # Pad audio if needed
            calculated_overview_duration = target_animation_duration
            actual_padding = actual_audio_duration - target_animation_duration
            
            if actual_padding > 0:
                print(f"   ➕ Adding {actual_padding:.2f}s of silence...")
                padded_audio_file = f"{output_name}_overview_audio_padded.aiff"
                pad_cmd = [
                    "ffmpeg", "-y",
                    "-i", overview_audio_file,
                    "-f", "lavfi",
                    "-t", str(actual_padding),
                    "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                    "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[out]",
                    "-map", "[out]",
                    padded_audio_file
                ]
                subprocess.run(pad_cmd, check=True, capture_output=True)
                os.replace(padded_audio_file, overview_audio_file)
            
            # Store animation
            overview_points = key_steps
            overview_animation = overview_animation_code
            overview_duration = calculated_overview_duration
            
            print(f"   ✅ Final overview duration: {overview_duration:.2f}s")
            
            # Step 1: Generate narration PER BLOCK (ALL blocks get narration)
            print("\n🎙️ Creating block-by-block narration...")
            print(f"   Generating narration for {len(all_blocks_for_narration)} blocks (ALL blocks)")
            print(f"   Creating highlights for {len(blocks_for_highlights)} blocks (only blocks that don't contain others)")
            print()
            
            block_narrations = []
            block_audios = []
            timeline_events = []
            cumulative_time = 0.0
            # ============================================================
            # WHY WE NEED INTRO_DELAY:
            # ============================================================
            # The Manim video has a timeline that looks like this:
            #
            #   0.0s ────────────────────────────────────────────────────
            #         Video starts - Overview slides appear
            #         Overview narration plays (actual duration measured)
            #         
            #   overview_duration ───────────────────────────────────────
            #         Title animation starts (0.5s)
            #         
            #   overview_duration + 0.5s ────────────────────────────────
            #         Small wait (0.1s)
            #         
            #   overview_duration + 0.6s ────────────────────────────────
            #         Code fade-in starts (0.5s)
            #         
            #   overview_duration + 1.1s ────────────────────────────────
            #         ⭐ CODE IS NOW VISIBLE ⭐
            #         Block-by-block narration starts
            #
            # SOLUTION: intro_delay = overview_duration + 0.5 (title) + 0.1 (wait) + 0.5 (fade)
            #           This ensures block narrations start exactly when code appears
            #
            # This MUST match fixed_overhead in generate_timeline_animations()
            # ============================================================
            
            # RECALCULATE overview_duration to match actual Manim timing
            # RECALCULATE overview_duration to match actual Manim timing
            if len(overview_points) > 0:
                # USER REQUIREMENT: Overview ends at 33s, 3s transition, code+narration at 36s
                # We must ensure the audio track has 36s of "intro" (overview + silence)
                # before the first block narration starts.
                pass 
            
            # Calculate intro_delay based on ACTUAL overview duration
            # Timeline:
            #   0s - overview_duration: Overview animation + narration
            #   + 1.0s: Overview transition (FadeOut + Wait)
            #   + 0.6s: Skeleton setup (Title + Wait)
            #   + 0.5s: Code fade-in
            #   + 3.0s: Code display wait
            #   = overview_duration + 5.1s: Highlights start and narration starts
            transition_time = 1.0 + 0.5 + 0.1 + 0.5 + 3.0  # 5.1s total
            intro_delay = overview_duration + transition_time
            
            print(f"🎤 Audio Sync: Calculated intro_delay = {intro_delay:.2f}s")
            print(f"   Overview duration: {overview_duration:.2f}s")
            print(f"   Transition time: {transition_time:.2f}s")
            print(f"   Code narration starts at: {intro_delay:.2f}s")
            
            print(f"\n{'='*60}")
            print("🔍 DEBUG: TIMELINE CONSTRUCTION")
            print(f"{'='*60}")
            print(f"intro_delay = {intro_delay:.2f}s (overview/introduction duration)")
            print(f"  During 0.0s - {intro_delay:.2f}s: Overview audio plays")
            print(f"  At {intro_delay:.2f}s: Code appears and block narrations start")
            print(f"Starting cumulative_time = {cumulative_time}s\n")
            
            # CRITICAL: Iterate over ALL blocks for narration
            # DO NOT SKIP - User wants all narration included
            for block_idx, code_block in enumerate(all_blocks_for_narration):

                block_type = code_block.get('type', 'code_block')
                block_code = code_block.get('code', '')
                start_line = code_block.get('start_line', 0)
                end_line = code_block.get('end_line', 0)
                
                print(f"\nProcessing Block {block_idx + 1}/{len(all_blocks_for_narration)}: {block_type} (lines {start_line}-{end_line})")
                print(f"   🔍 DEBUG: Block index: {block_idx}, Total narration blocks: {len(all_blocks_for_narration)}, Total highlight blocks: {len(blocks_for_highlights)}")
                print(f"   🔍 DEBUG: Block code preview: {block_code[:100]}...")
                
                # CRITICAL: Check if this block will be highlighted
                # If NOT highlighted, it means it CONTAINS other blocks that will be highlighted
                # In that case, give a BRIEF overview mentioning it contains inner blocks
                will_have_highlight = any(
                    b['start_line'] == code_block['start_line'] and 
                    b['end_line'] == code_block['end_line'] and
                    b['type'] == code_block['type']
                    for b in blocks_for_highlights
                )
                
                # Check what inner blocks this block contains
                contained_blocks = []
                for other_block in all_blocks_for_narration:
                    if other_block != code_block:
                        # Check if other_block is INSIDE this block
                        if (code_block['start_line'] < other_block['start_line'] and 
                            code_block['end_line'] >= other_block['end_line']):
                            contained_blocks.append(other_block)
                
                # Determine narration strategy based on whether block will be highlighted
                if not will_have_highlight and len(contained_blocks) > 0:
                    # This block CONTAINS other blocks and won't be highlighted
                    # Give a BRIEF overview mentioning it contains inner blocks
                    print(f"   📝 Block contains {len(contained_blocks)} inner blocks - using BRIEF overview")
                    
                    # List the types of contained blocks
                    contained_types = [b['type'].replace('_', ' ') for b in contained_blocks]
                    contained_summary = ', '.join(set(contained_types))
                    
                    block_type_name = block_type.replace('_', ' ').title()
                    if block_type == "for_loop":
                        block_type_name = "for loop"
                    elif block_type == "while_loop":
                        block_type_name = "while loop"
                    elif block_type == "if_statement":
                        block_type_name = "if statement"
                    elif block_type == "function":
                        block_type_name = "function"
                    elif block_type == "class":
                        block_type_name = "class"
                    
                    # Generate BRIEF overview that mentions inner blocks
                    block_prompt = f"""Give a VERY BRIEF overview of this {block_type_name}.
                    
CRITICAL RULES:
- This {block_type_name} contains inner blocks: {contained_summary}
- Do NOT explain the inner blocks in detail - they will be explained separately
- Just mention: "This {block_type_name} contains..." or "This {block_type_name} includes..."
- Keep it under 200 characters - just a quick introduction
- Example: "This function contains several for loops that process the data."

Code block:
```
{block_code}
```

Provide a BRIEF overview (under 200 characters) mentioning it contains inner blocks."""
                    
                    max_tokens = 100  # Shorter for brief overview
                    max_block_length = 200
                else:
                    # This block WILL be highlighted OR doesn't contain other blocks
                    # Give FULL detailed explanation
                    print(f"   📝 Block will be highlighted - using FULL explanation")
                    
                    if block_idx == 0:
                        # Get block type for explicit mention in narration
                        block_type_name = block_type.replace('_', ' ').title()
                        if block_type == "for_loop":
                            block_type_name = "for loop"
                        elif block_type == "while_loop":
                            block_type_name = "while loop"
                        elif block_type == "if_statement":
                            block_type_name = "if statement"
                        elif block_type == "function":
                            block_type_name = "function"
                        elif block_type == "class":
                            block_type_name = "class"
                        
                        block_prompt = f"""Explain ONLY this specific code block. 

                CRITICAL RULES:
                - This is a {block_type_name}
                - Start by explicitly mentioning the block type: "This {block_type_name}..." or "The {block_type_name}..."
                - Do NOT mention what the entire code does
                - Do NOT give an overview or introduction
                - Do NOT say "this code" or "the program" without specifying it\'s a {block_type_name}
                - Focus ONLY on what THIS specific {block_type_name} does
                - Start directly: "This {block_type_name}..." or "The {block_type_name}..."

                Code block:
                ```
                {block_code}
                ```

                Provide a concise explanation starting with "This {block_type_name}..." (under 500 characters)."""
                    else:
                        block_prompt = f"""Now explain this next code block. Use a natural transition like "Next, we have..." or "Moving on to...".

                Code block:
                ```
                {block_code}
                ```

                Provide a concise explanation (under 500 characters)."""
                    
                    max_tokens = 250  # Normal length for full explanation
                    max_block_length = 500
                
                narration_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are an expert programming educator. Explain ONLY the specific code block provided. This is a {block_type.replace('_', ' ')}. CRITICAL: Start by explicitly mentioning the block type (e.g., 'This for loop...', 'The if statement...'). Do NOT give overviews, introductions, or mention the entire code. Focus ONLY on what this specific {block_type.replace('_', ' ')} does. Start directly with the block type. Keep explanations under {max_block_length} characters."
                        },
                        {
                            "role": "user",
                            "content": block_prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=max_tokens
                )
                
                block_narration = narration_response.choices[0].message.content.strip()
                if len(block_narration) > max_block_length:
                    block_narration = block_narration[:max_block_length]
                    last_period = block_narration.rfind('.')
                    if last_period > max_block_length * 0.8:
                        block_narration = block_narration[:last_period + 1]
                
                if block_idx == 0:
                    overview_phrases = ["this code", "the program", "the entire code", "this program", "overall", "in general"]
                    narration_lower = block_narration.lower()
                    for phrase in overview_phrases:
                        if phrase in narration_lower:
                            sentences = block_narration.split('.')
                            filtered_sentences = [s for s in sentences if phrase not in s.lower()]
                            if filtered_sentences:
                                block_narration = '. '.join(filtered_sentences).strip()
                                if not block_narration.endswith('.'):
                                    block_narration += '.'
                                break
                
                block_narrations.append(block_narration)
                print(f"Generated {len(block_narration)} characters")
                
                # Generate audio for this block
                block_audio_file = f"{output_name}_block_{block_idx}_audio.aiff"
                print(f"   Generating audio for block {block_idx + 1}...")
                audio_result = generate_audio_for_language(block_narration, audio_language, block_audio_file, client)
                
                # Validate audio file was created
                if not os.path.exists(block_audio_file):
                    print(f"   ❌ Audio file not created: {block_audio_file}")
                    print(f"   ⚠️  Skipping this block")
                    continue
                
                audio_file_size = os.path.getsize(block_audio_file)
                if audio_file_size == 0:
                    print(f"   ❌ Audio file is empty: {block_audio_file}")
                    print(f"   ⚠️  Skipping this block")
                    continue
                
                print(f"   ✅ Audio file created: {block_audio_file} ({audio_file_size} bytes)")
                
                # Get REAL duration from audio file
                block_duration = get_audio_duration(block_audio_file)
                if not block_duration:
                    block_duration = len(block_narration) * 0.06
                    print(f"   ⚠️ Using estimated duration: {block_duration:.2f}s")
                else:
                    print(f"   ✅ Measured audio duration: {block_duration:.2f}s")
                
                block_audios.append(block_audio_file)
                
                # CRITICAL: Only create timeline events (highlights) for blocks in blocks_for_highlights
                # But still advance cumulative_time for ALL blocks (for correct audio synchronization)
                will_have_highlight = any(
                    b['start_line'] == code_block['start_line'] and 
                    b['end_line'] == code_block['end_line'] and
                    b['type'] == code_block['type']
                    for b in blocks_for_highlights
                )
                
                # DEBUG: Print matching details
                print(f"   🔍 DEBUG: Checking if block {block_idx + 1} ({block_type}, lines {start_line}-{end_line}) will have highlight...")
                if not will_have_highlight:
                    print(f"   🔍 DEBUG: Block NOT in blocks_for_highlights. Checking available highlight blocks:")
                    for i, hb in enumerate(blocks_for_highlights, 1):
                        print(f"      Highlight block {i}: {hb['type']} (lines {hb['start_line']}-{hb['end_line']})")
                        match_start = hb['start_line'] == code_block['start_line']
                        match_end = hb['end_line'] == code_block['end_line']
                        match_type = hb['type'] == code_block['type']
                        print(f"         Match start_line: {match_start} ({hb['start_line']} == {code_block['start_line']})")
                        print(f"         Match end_line: {match_end} ({hb['end_line']} == {code_block['end_line']})")
                        print(f"         Match type: {match_type} ({hb['type']} == {code_block['type']})")
                else:
                    print(f"   ✅ Block WILL have highlight!")
                
                if will_have_highlight:
                    # Create timeline event with REAL timestamps
                    # ============================================================
                    # WHY WE NEED EVENT_START:
                    # ============================================================
                    # We need to calculate WHEN to start highlighting each code block.
                    # There are TWO constraints we must satisfy:
                    #
                    # CONSTRAINT 1: Code must be visible (intro_delay)
                    #   - Code appears at 3.2s (intro_delay)
                    #   - We CANNOT highlight before code is visible
                    #   - Example: If narration starts at 0.0s, we can't highlight at 0.0s
                    #
                    # CONSTRAINT 2: Narration must have started (cumulative_time)
                    #   - Block 1 narration starts at 0.0s (cumulative_time = 0.0)
                    #   - Block 2 narration starts at 3.5s (cumulative_time = 3.5)
                    #   - We CANNOT highlight before narration starts
                    #   - Example: If code appears at 3.2s but narration starts at 3.5s,
                    #              we can't highlight at 3.2s (nothing to sync with!)
                    #
                    # SOLUTION: event_start = max(intro_delay, cumulative_time)
                    #   - Takes the LATER of the two times
                    #   - Ensures BOTH constraints are satisfied
                    #
                    # VISUAL EXAMPLE:
                    #   Timeline: 0.0s ──── 3.2s ──── 3.5s ──── 7.0s
                    #            │        │         │         │
                    #            │        │         │         Block 2 ends
                    #            │        │         Block 2 starts
                    #            │        Code appears (intro_delay)
                    #            Block 1 narration starts
                    #
                    #   Block 1:
                    #     - cumulative_time = 0.0s (narration starts)
                    #     - intro_delay = 3.2s (code appears)
                    #     - event_start = max(3.2, 0.0) = 3.2s ✅
                    #     - Result: Highlight starts when code becomes visible
                    #
                    #   Block 2:
                    #     - cumulative_time = 3.5s (narration starts)
                    #     - intro_delay = 3.2s (code already visible)
                    #     - event_start = max(3.2, 3.5) = 3.5s ✅
                    #     - Result: Highlight starts when narration starts
                    #
                    # WHY event_end = cumulative_time + block_duration?
                    #   - Narration ends when block_duration finishes
                    #   - Highlight should end when narration ends (perfect sync)
                    #
                    # CRITICAL FIX: Add silence at start of audio so narration starts when code appears
                    #   - We'll add intro_delay seconds of silence at the start of the audio file
                    #   - This means narration in the final audio starts at intro_delay (3.2s)
                    #   - So we add intro_delay to all timings:
                    #     event_start = cumulative_time + intro_delay
                    #     event_end = cumulative_time + block_duration + intro_delay
                    # ============================================================
                    # Add intro_delay to account for silence at start of audio
                    event_start = cumulative_time + intro_delay  # Narration starts at intro_delay + cumulative_time
                    event_end = cumulative_time + block_duration + intro_delay  # Narration ends at intro_delay + cumulative_time + block_duration
                    
                    # Ensure minimum highlight duration (shouldn't be needed with silence fix, but keep as safety)
                    actual_highlight_duration = event_end - event_start
                    min_highlight_duration = 1.5  # Minimum 1.5 seconds for visibility
                    
                    if actual_highlight_duration < min_highlight_duration:
                        # Highlight too short - extend it slightly
                        # But don't extend past next block start time
                        if block_idx < len(all_blocks_for_narration) - 1:
                            # Next block starts at: (cumulative_time + block_duration) + intro_delay
                            next_block_start = (cumulative_time + block_duration) + intro_delay
                        else:
                            next_block_start = float('inf')
                        extended_end = min(event_start + min_highlight_duration, next_block_start)
                        if extended_end > event_end:
                            event_end = extended_end
                            print(f"      ⚠️  Highlight too short ({actual_highlight_duration:.2f}s), extending to {event_end:.2f}s")
                    
                    timeline_events.append({
                        'code_block': code_block,
                        'start_time': event_start,  # When to START highlighting (in video time)
                        'end_time': event_end,      # When to STOP highlighting (in video time)
                        'audio_start': cumulative_time + intro_delay,  # When this block's narration starts in final audio (with silence)
                        'audio_end': cumulative_time + block_duration + intro_delay,  # When this block's narration ends in final audio
                        'confidence': 1.0,
                        'sentence': block_idx + 1
                    })
                    
                    actual_highlight_duration = event_end - event_start
                    
                    print(f"      📍 Timeline Event {len(timeline_events)} (Block {block_idx + 1}):")
                    print(f"         Block: {block_type} (lines {start_line}-{end_line})")
                    print(f"         🔍 DEBUG: cumulative_time BEFORE: {cumulative_time:.2f}s")
                    print(f"         Block narration duration: {block_duration:.2f}s")
                    print(f"         🔍 DEBUG: cumulative_time AFTER: {cumulative_time + block_duration:.2f}s")
                    print(f"         Final audio timing: {event_start:.2f}s - {event_end:.2f}s (with {intro_delay:.2f}s silence at start)")
                    print(f"         Video timing: {event_start:.2f}s - {event_end:.2f}s (highlight duration: {actual_highlight_duration:.2f}s)")
                    print(f"         🔍 DEBUG: Timeline event created: start={event_start:.2f}s, end={event_end:.2f}s, block={block_type} lines {start_line}-{end_line}")
                    if actual_highlight_duration == block_duration:
                        print(f"         ✅ Perfect match: highlight duration ({actual_highlight_duration:.2f}s) = narration duration ({block_duration:.2f}s)")
                    else:
                        print(f"         ⚠️  Mismatch: highlight ({actual_highlight_duration:.2f}s) ≠ narration ({block_duration:.2f}s)")
                    print()
                else:
                    # This block gets narration but NO highlight (e.g., class, function, outer loop)
                    print(f"      📝 Narration only (NO highlight): {block_type} (lines {start_line}-{end_line})")
                    print(f"         🔍 DEBUG: cumulative_time BEFORE: {cumulative_time:.2f}s")
                    print(f"         Block narration duration: {block_duration:.2f}s")
                    print(f"         🔍 DEBUG: cumulative_time AFTER: {cumulative_time + block_duration:.2f}s")
                    print()
                
                # CRITICAL: Advance cumulative_time for ALL blocks (narration + highlights)
                # This ensures correct audio synchronization even for blocks without highlights
                cumulative_time += block_duration
            
            # Validate we have audio files before concatenation
            if not block_audios:
                print("❌ No valid audio files generated! Cannot create video.")
                return None
            
            # Prepend overview audio to block audios
            all_audio_files = [overview_audio_file] + block_audios
            
            # Concatenate all audio files (overview + blocks) with NO silence at start
            # The overview audio will play during 0.0s - ~3.0s, then blocks start
            print(f"\n🔗 Concatenating {len(all_audio_files)} audio files (1 overview + {len(block_audios)} blocks)...")
            print(f"   Overview audio will play first, then block narrations")
            audio_file = f"{output_name}_audio.aiff"
            
            if not concatenate_audio_files(all_audio_files, audio_file, silence_duration=0.0):
                print("❌ Audio concatenation failed!")
                print("   Attempting to use first block audio as fallback...")
                
                # Try to use first block audio with silence prepended
                if block_audios and os.path.exists(block_audios[0]):
                    first_audio = block_audios[0]
                    print(f"   Using first block audio: {first_audio}")
                    
                    # Try to prepend silence to first audio
                    try:
                        silence_file = 'silence_temp_fallback.aiff'
                        subprocess.run([
                            'ffmpeg', '-y', '-f', 'lavfi',
                            '-i', f'anullsrc=channel_layout=mono:sample_rate=44100',
                            '-t', str(intro_delay),
                            silence_file
                        ], check=True, capture_output=True, text=True)
                        
                        if os.path.exists(silence_file):
                            with open('concat_fallback.txt', 'w') as f:
                                f.write(f"file '{os.path.abspath(silence_file)}'\n")
                                f.write(f"file '{os.path.abspath(first_audio)}'\n")
                            
                            subprocess.run([
                                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                                '-i', 'concat_fallback.txt',
                                '-c', 'copy', audio_file
                            ], check=True, capture_output=True, text=True)
                            
                            # Cleanup
                            if os.path.exists(silence_file):
                                os.remove(silence_file)
                            if os.path.exists('concat_fallback.txt'):
                                os.remove('concat_fallback.txt')
                            
                            if os.path.exists(audio_file) and os.path.getsize(audio_file) > 0:
                                print(f"   ✅ Fallback audio created: {audio_file}")
                            else:
                                print(f"   ❌ Fallback also failed")
                                return None
                        else:
                            print(f"   ❌ Could not create silence for fallback")
                            return None
                    except Exception as e:
                        print(f"   ❌ Fallback failed: {e}")
                        return None
                else:
                    print("   ❌ No valid audio files available")
                    return None
            
            # Verify final audio file exists and is valid
            if not os.path.exists(audio_file):
                print(f"❌ Final audio file not found: {audio_file}")
                return None
            
            audio_file_size = os.path.getsize(audio_file)
            if audio_file_size == 0:
                print(f"❌ Final audio file is empty: {audio_file}")
                return None
            
            print(f"✅ Final audio file ready: {audio_file} ({audio_file_size} bytes)")
            
            # Cleanup individual block audio files
            for block_audio in block_audios:
                if os.path.exists(block_audio) and block_audio != audio_file:
                    try:
                        os.remove(block_audio)
                    except:
                        pass
            
            # Get total audio duration (includes silence at start)
            # Use ffprobe for more reliable duration detection
            try:
                duration_cmd = [
                    "ffprobe", "-v", "error", "-show_entries",
                    "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                    audio_file
                ]
                duration_result = subprocess.run(duration_cmd, check=True, capture_output=True, text=True)
                audio_duration = float(duration_result.stdout.strip())
                print(f"✅ Final audio duration: {audio_duration:.2f} seconds (measured with ffprobe)")
            except Exception as e:
                print(f"⚠️  Could not get audio duration with ffprobe: {e}")
                audio_duration = get_audio_duration(audio_file)
                if not audio_duration:
                    print("⚠️  Could not get audio duration, using estimate")
                    audio_duration = cumulative_time + intro_delay  # Include silence at start
                else:
                    print(f"✅ Final audio duration: {audio_duration:.2f} seconds (from get_audio_duration)")
            
            print(f"\n{'='*60}")
            print("🔍 DEBUG: TIMELINE SUMMARY")
            print(f"{'='*60}")
            print(f"Total blocks: {len(timeline_events)}")
            print(f"Total cumulative time: {cumulative_time:.2f}s")
            print(f"Final audio duration: {audio_duration:.2f}s")
            print(f"intro_delay: {intro_delay:.2f}s")
            print(f"\nTimeline Events:")
            for idx, event in enumerate(timeline_events):
                block = event['code_block']
                print(f"  Event {idx + 1}: {block['type']} (lines {block['start_line']}-{block['end_line']})")
                print(f"    Start: {event['start_time']:.2f}s, End: {event['end_time']:.2f}s, Duration: {event['end_time'] - event['start_time']:.2f}s")
            print(f"{'='*60}\n")
            
            # Extract key concepts from combined narration
            combined_narration = " ".join(block_narrations)
            print("\n🔑 Extracting key concepts...")
            concepts_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract 4-6 key concepts from the code explanation. Return as a JSON array of concept strings."
                    },
                    {
                        "role": "user",
                        "content": f"Extract key concepts from this code explanation:\n\n{combined_narration}\n\nReturn JSON: {{\"concepts\": [\"concept1\", \"concept2\", ...]}}"
                    }
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            concepts_data = json.loads(concepts_response.choices[0].message.content)
            key_concepts = concepts_data.get("concepts", [])
            print(f"✅ Extracted {len(key_concepts)} key concepts\n")
            
            # Generate narration for key concepts and append to audio
            key_concepts_audio_file = None
            if key_concepts:
                print("🎙️ Generating narration for key concepts...")
                concepts_narration_prompt = f"""Create a brief narration (2-3 sentences, under 200 characters) explaining these key concepts from the code:
                
{', '.join(key_concepts)}

Make it natural and concise. Start with "Key concepts include..." or similar."""
                
                concepts_narration_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "Create brief, natural narration for key concepts. Keep it under 200 characters."
                        },
                        {
                            "role": "user",
                            "content": concepts_narration_prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=100
                )
                
                concepts_narration = concepts_narration_response.choices[0].message.content.strip()
                print(f"   Generated {len(concepts_narration)} characters")
                
                # Generate audio for key concepts
                key_concepts_audio_file = f"{output_name}_key_concepts_audio.aiff"
                print(f"   Generating audio for key concepts...")
                generate_audio_for_language(concepts_narration, audio_language, key_concepts_audio_file, client)
                
                if os.path.exists(key_concepts_audio_file) and os.path.getsize(key_concepts_audio_file) > 0:
                    # Append key concepts audio to main audio
                    print(f"   ✅ Key concepts audio generated: {key_concepts_audio_file}")
                    print(f"   🔗 Appending key concepts audio to main audio...")
                    
                    # Concatenate main audio + key concepts audio
                    combined_audio_files = [audio_file, key_concepts_audio_file]
                    final_audio_file = f"{output_name}_audio_with_concepts.aiff"
                    
                    if concatenate_audio_files(combined_audio_files, final_audio_file, silence_duration=0.0):
                        # Update audio_file to use the combined version
                        old_audio_file = audio_file
                        audio_file = final_audio_file
                        
                        # Get new audio duration
                        try:
                            duration_cmd = [
                                "ffprobe", "-v", "error", "-show_entries",
                                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                                audio_file
                            ]
                            duration_result = subprocess.run(duration_cmd, check=True, capture_output=True, text=True)
                            audio_duration = float(duration_result.stdout.strip())
                            print(f"   ✅ Combined audio duration: {audio_duration:.2f}s")
                        except:
                            pass
                        
                        # Cleanup old audio file
                        if os.path.exists(old_audio_file) and old_audio_file != audio_file:
                            try:
                                os.remove(old_audio_file)
                            except:
                                pass
                    else:
                        print(f"   ⚠️  Failed to append key concepts audio, using original audio")
                else:
                    print(f"   ⚠️  Key concepts audio not generated, skipping")
            
            # Calculate when key concepts start in audio
            # CRITICAL: Key concepts MUST start AFTER code narration AND highlights end
            key_concepts_start_time = None
            if key_concepts and audio_duration:
                # Calculate when code narration AND highlights actually end
                # Use the ACTUAL end_time from timeline_events (which includes highlight extensions)
                if timeline_events and len(timeline_events) > 0:
                    # Get the LATEST end_time from all timeline events
                    # This accounts for highlight duration extensions
                    actual_code_end_time = max([e['end_time'] for e in timeline_events])
                    
                    print(f"\n🔍 DEBUG: Key concepts timing calculation:")
                    print(f"   Last timeline event end_time: {actual_code_end_time:.2f}s")
                    print(f"   This is when the LAST highlight actually ends (including extensions)")
                    
                    # Key concepts audio starts immediately after code highlights end
                    # Add 0.7s transition time (actual Manim animation time):
                    #   - 0.5s: Code and title fade-out
                    #   - 0.2s: Small wait
                    # Total: 0.7s
                    key_concepts_start_time = actual_code_end_time + 0.7
                    print(f"   ✅ Key concepts start at {key_concepts_start_time:.2f}s")
                    print(f"      (Last highlight ends at {actual_code_end_time:.2f}s + 0.7s transition)\n")
                else:
                    print(f"   ⚠️  No timeline events - cannot calculate key concepts start time\n")
            
            # Step 6: Generate Manim code with timeline-based animations
            # CRITICAL: Use code_cleaned so line numbers match what will be displayed
            print("🎨 Generating Manim code with overview slide and Whisper timestamp synchronization...")
            manim_code = generate_timeline_animations(
                code_cleaned, 
                timeline_events, 
                audio_duration, 
                key_concepts, 
                key_concepts_start_time,
                overview_points,
                overview_duration,
                overview_animation
            )
            print("✅ Manim code generated\n")
        
        # Step 7: Render video
        print("🎬 Rendering video...")
        scene_file = f".temp_{output_name}.py"
        
        print(f"   Writing Manim scene to: {scene_file}")
        with open(scene_file, 'w') as f:
            f.write(manim_code)
        
        file_size = os.path.getsize(scene_file)
        print(f"   ✅ Scene file written ({file_size} bytes)")
        
        # Use venv311 if it exists, otherwise fallback to venv
        venv_python = "venv311/bin/python" if os.path.exists("venv311/bin/python") else "venv/bin/python"
        cmd = [
            venv_python, "-m", "manim",
            "-pql", scene_file, "CodeExplanationScene"
        ]
        
        print(f"   Running: {' '.join(cmd)}")
        print(f"   ⏳ This may take 30-60 seconds for video rendering...")
        print(f"   (Manim is rendering the video - please wait)\n")
        
        # Show output in real-time so user knows it's working
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=False,  # Show output in real-time
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Manim rendering failed!")
            print(f"   Exit code: {e.returncode}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"   Error: {e.stderr[:500]}")
            return None
        
        video_path = f"media/videos/{scene_file.replace('.py', '')}/480p15/CodeExplanationScene.mp4"
        
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            return None
        
        print("✅ Video rendered\n")
        
        # Step 8: Combine video + audio
        print("🎵 Combining video + audio...")
        
        # Validate files exist before combining
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            print(f"   Please check if Manim rendered successfully")
            return None
        
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            return None
        
        video_size = os.path.getsize(video_path)
        audio_size = os.path.getsize(audio_file)
        
        if video_size == 0:
            print(f"❌ Video file is empty: {video_path}")
            print(f"   Manim may have failed to render. Check Manim output above.")
            return None
        
        if audio_size == 0:
            print(f"❌ Audio file is empty: {audio_file}")
            return None
        
        print(f"   ✅ Video: {video_path} ({video_size:,} bytes)")
        print(f"   ✅ Audio: {audio_file} ({audio_size:,} bytes)")
        
        # Validate video file is readable by FFmpeg
        print(f"   🔍 Validating video file format...")
        try:
            probe_cmd = ["ffprobe", "-v", "error", video_path]
            probe_result = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
            print(f"   ✅ Video file is valid")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Video file is not valid or corrupted!")
            print(f"   FFprobe error: {e.stderr[:500] if e.stderr else 'Unknown error'}")
            return None
        except Exception as e:
            print(f"   ⚠️  Could not validate video: {e}")
        
        # Validate audio file is readable by FFmpeg
        print(f"   🔍 Validating audio file format...")
        try:
            probe_cmd = ["ffprobe", "-v", "error", audio_file]
            probe_result = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
            print(f"   ✅ Audio file is valid")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Audio file is not valid or corrupted!")
            print(f"   FFprobe error: {e.stderr[:500] if e.stderr else 'Unknown error'}")
            return None
        except Exception as e:
            print(f"   ⚠️  Could not validate audio: {e}")
        
        # Get durations to ensure proper sync
        video_duration = None
        audio_duration_actual = None
        try:
            video_duration_cmd = [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            video_duration_result = subprocess.run(video_duration_cmd, check=True, capture_output=True, text=True)
            video_duration = float(video_duration_result.stdout.strip())
            
            audio_duration_cmd = [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                audio_file
            ]
            audio_duration_result = subprocess.run(audio_duration_cmd, check=True, capture_output=True, text=True)
            audio_duration_actual = float(audio_duration_result.stdout.strip())
            
            print(f"   Video duration: {video_duration:.2f}s")
            print(f"   Audio duration: {audio_duration_actual:.2f}s")
            
        except Exception as e:
            print(f"   ⚠️  Could not get durations: {e}")
        
        # Step 8.5: Generate subtitles if requested
        srt_file = None
        if add_subtitles:
            srt_file = f"{output_name}_subtitles.srt"
            subtitle_success = generate_srt_from_audio(audio_file, srt_file)
            if not subtitle_success:
                print("   ⚠️  Continuing without subtitles...")
                srt_file = None
        
        final_output = f"{output_name}_final.mp4"
        
        # Build command - use longer duration and loop audio if needed
        combine_cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_file,
        ]
        
        # Add subtitle input if available
        if srt_file and os.path.exists(srt_file):
            combine_cmd.extend(["-i", srt_file])
            print(f"   📝 Adding subtitles: {srt_file}")
        
        # If audio is shorter, loop it to match video
        if video_duration and audio_duration_actual and audio_duration_actual < video_duration:
            print(f"   ⚠️  Audio ({audio_duration_actual:.2f}s) is shorter than video ({video_duration:.2f}s)")
            print(f"   🔄 Looping audio to match video duration...")
            # Calculate how many times to loop
            loop_count = int(video_duration / audio_duration_actual) + 1
            combine_cmd.extend([
                "-filter_complex", f"[1:a]aloop=loop={loop_count}:size=2e+09[a]",
                "-c:v", "copy",  # Copy video (faster, no quality loss)
                "-c:a", "aac",
                "-b:a", "192k",
                "-map", "0:v",
                "-map", "[a]",
                "-shortest",  # Use video duration
            ])
            # Add subtitle mapping if available
            if srt_file and os.path.exists(srt_file):
                combine_cmd.extend([
                    "-map", "2:s",
                    "-c:s", "mov_text",
                    "-metadata:s:s:0", "language=eng",
                    "-metadata:s:s:0", "title=English",
                    "-disposition:s:0", "default"
                ])
        elif video_duration and audio_duration_actual and video_duration < audio_duration_actual:
            print(f"   ⚠️  Video ({video_duration:.2f}s) is shorter than audio ({audio_duration_actual:.2f}s)")
            print(f"   🔄 Trimming audio to match video duration...")
            # Trim audio to match video instead of extending video
            combine_cmd.extend([
                "-filter_complex", f"[1:a]atrim=0:{video_duration}[a]",
                "-c:v", "copy",  # Copy video (faster, no quality loss)
                "-c:a", "aac",
                "-b:a", "192k",
                "-map", "0:v",
                "-map", "[a]",
            ])
            # Add subtitle mapping if available
            if srt_file and os.path.exists(srt_file):
                combine_cmd.extend([
                    "-map", "2:s",
                    "-c:s", "mov_text",
                    "-metadata:s:s:0", "language=eng",
                    "-metadata:s:s:0", "title=English",
                    "-disposition:s:0", "default"
                ])
        else:
            # Same duration or unknown - use standard combination
            # Use copy for video (Manim already creates QuickTime-compatible video)
            # Only re-encode audio to AAC
            combine_cmd.extend([
                "-c:v", "copy",  # Copy video stream (no re-encoding, faster, maintains compatibility)
                "-c:a", "aac",   # Re-encode audio to AAC
                "-b:a", "192k",  # Set audio bitrate for compatibility
                "-shortest",
            ])
            # Add subtitle mapping if available
            if srt_file and os.path.exists(srt_file):
                combine_cmd.extend([
                    "-map", "2:s",
                    "-c:s", "mov_text",
                    "-metadata:s:s:0", "language=eng",
                    "-metadata:s:s:0", "title=English",
                    "-disposition:s:0", "default"
                ])
        
        combine_cmd.append(final_output)
        
        print(f"   Running FFmpeg: {' '.join(combine_cmd[:5])}... (output file: {final_output})")
        print(f"   ⏳ Combining video and audio (this may take 10-20 seconds)...\n")
        
        try:
            # Capture output to see errors, but also show progress
            result = subprocess.run(combine_cmd, check=True, capture_output=True, text=True)
            
            # Show FFmpeg output if there's any
            if result.stdout:
                print(f"   FFmpeg output: {result.stdout[:500]}")
            if result.stderr:
                print(f"   FFmpeg warnings: {result.stderr[:500]}")
            
            # Verify output file was created
            if not os.path.exists(final_output):
                print(f"❌ Final video file not created: {final_output}")
                print(f"   FFmpeg may have failed silently")
                return None
            
            final_size = os.path.getsize(final_output)
            if final_size == 0:
                print(f"❌ Final video file is empty: {final_output}")
                print(f"   FFmpeg created empty file - check FFmpeg errors above")
                if result.stderr:
                    print(f"   FFmpeg stderr: {result.stderr}")
                return None
            
            print(f"   ✅ Combined video: {final_output} ({final_size} bytes)")
            
            # Get video duration to verify it's correct
            try:
                duration_cmd = [
                    "ffprobe", "-v", "error", "-show_entries",
                    "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            final_output
        ]
                duration_result = subprocess.run(duration_cmd, check=True, capture_output=True, text=True)
                duration = float(duration_result.stdout.strip())
                print(f"   Final video duration: {duration:.2f} seconds")
            except:
                pass
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ FFmpeg combination failed!")
            print(f"   Exit code: {e.returncode}")
            print(f"   Command: {' '.join(combine_cmd)}")
            if e.stderr:
                print(f"\n   FFmpeg Error Output:")
                print(f"   {'-'*60}")
                print(f"   {e.stderr}")
                print(f"   {'-'*60}")
            if e.stdout:
                print(f"\n   FFmpeg Standard Output:")
                print(f"   {e.stdout[:1000]}")
            
            # Check if output file exists but is corrupted
            if os.path.exists(final_output):
                file_size = os.path.getsize(final_output)
                if file_size == 0:
                    print(f"\n   ⚠️  Output file exists but is 0 bytes - removing it...")
                    try:
                        os.remove(final_output)
                    except:
                        pass
            
            return None
        except Exception as e:
            print(f"\n❌ Unexpected error during video combination: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS! {final_output}")
        print(f"{'='*60}\n")
        
        return {"final_video": final_output}
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# EXAMPLES
# ============================================================

if __name__ == "__main__":
    
    
    # Example 1: TEXT TO VIDEO
    # Use for: Documentation, tutorials, explanations
    
    text_content = """
    Bubble Sort Algorithm
    
    Bubble Sort is one of the simplest sorting algorithms. It works by repeatedly 
    comparing adjacent elements and swapping them if they're in the wrong order.
    
    How It Works
    The algorithm makes multiple passes through the array. In each pass, it compares 
    consecutive pairs of elements. When a larger element comes before a smaller one, 
    they are swapped.
    
    Time Complexity
    
    The algorithm has O(n²) time complexity in worst and average cases, making it 
    inefficient for large datasets. However, its simplicity makes it excellent for 
    educational purposes.
    
    Practical Use
    
    While not used in production due to inefficiency, Bubble Sort is valuable for 
    teaching fundamental sorting concepts.
    """
    
    # result = text_to_video(text_content, "bubble_sort_explanation")
    
    
    # Example 3: CODE TO VIDEO
    # Use for: Code explanations
    
    pyramid_code = """
rows = 5
for i in range(1, rows + 1):
    for j in range(1, i + 1):
        print(j, end=" ")
    print()
"""
    
    result = code_to_video(
        code_content=pyramid_code,
        output_name="pyramid_code_explanation"
    )
    
    
    # More examples - just change the input!
    
    # TEXT TO VIDEO examples:
    # text_to_video("Neural networks explained...", "neural_networks")
    # text_to_video("Machine learning basics...", "ml_basics")
    
    # ANIMATION TO VIDEO examples:
    # animation_to_video("Quick Sort algorithm", "quick_sort")
    # animation_to_video("Binary Search Tree operations", "bst")
    # animation_to_video("Graph BFS traversal", "graph_bfs")
    # animation_to_video("Pythagorean theorem proof", "pythagoras")
    
    
    if not result:
        sys.exit(1)
    

# ============================================================
# QUERY TO ANIMATED VIDEO - Simple Text Slides
# ============================================================

def query_to_animated_video(query: str, output_name: str = "output", audio_language: str = "english"):
    """Convert query to beautiful animated text slides - works with ANY query"""
    try:
        print(f"\n{'='*60}")
        print(f"🎨 QUERY TO ANIMATED VIDEO: {output_name}")
        print(f"{'='*60}\n")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ Set OPENAI_API_KEY environment variable")
            return None
        
        client = openai.OpenAI(api_key=api_key)
        
        # Generate simple content (1-2 slides)
        print(f"🤖 Generating content for: '{query}'...")
        
        content_prompt = f"""Generate educational content about: {query}

Create 1-2 simple slides. Return JSON:
{{
    "title": "Main Topic",
    "slides": [
        {{
            "heading": "Definition",
            "content": "2-3 sentences...",
            "narration": "Spoken text..."
        }}
    ]
}}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Create clear educational content."},
                {"role": "user", "content": content_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        content_data = json.loads(response.choices[0].message.content)
        print(f"✅ Generated {len(content_data.get('slides', []))} slides\n")
        
        # Generate audio per slide
        print("🎙️ Generating audio...")
        section_audio_files = []
        section_durations = []
        
        # Title
        title_audio = f"{output_name}_title.aiff"
        generate_audio_for_language(f"{content_data.get('title', '')}.", audio_language, title_audio, client)
        title_duration = get_audio_duration(title_audio)
        section_audio_files.append(title_audio)
        print(f"   Title: {title_duration:.2f}s")
        
        # Slides
        for idx, slide in enumerate(content_data.get('slides', [])):
            slide_audio = f"{output_name}_slide_{idx}.aiff"
            generate_audio_for_language(slide.get('narration', ''), audio_language, slide_audio, client)
            slide_duration = get_audio_duration(slide_audio)
            section_durations.append(slide_duration)
            section_audio_files.append(slide_audio)
            print(f"   Slide {idx + 1}: {slide_duration:.2f}s")
        
        # Combine audio
        print("\n🎵 Combining audio...")
        audio_file = f"{output_name}_audio.mp3"
        concat_file = f"{output_name}_concat.txt"
        
        with open(concat_file, 'w') as f:
            for audio in section_audio_files:
                f.write(f"file '{audio}'\n")
        
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_file, "-c:a", "libmp3lame", "-b:a", "192k", audio_file
        ], check=True, capture_output=True)
        
        # Cleanup
        for audio in section_audio_files:
            if os.path.exists(audio): os.remove(audio)
        if os.path.exists(concat_file): os.remove(concat_file)
        
        audio_duration = get_audio_duration(audio_file) or (title_duration + sum(section_durations))
        print(f"✅ Audio: {audio_duration:.2f}s\n")
        
        # Generate Manim code
        print("🎨 Generating Manim code...")
        manim_code = f"""from manim import *

class SimpleTextScene(Scene):
    def construct(self):
        self.wait(1.2)
        
        # Title
        title = Text("{content_data.get('title', '')}", font_size=48, font="JetBrains Mono", weight=BOLD, color=GOLD)
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title, shift=DOWN*0.3), run_time=0.8)
        self.wait(1.5)
        self.play(FadeOut(title), run_time=0.5)
        
"""
        
        for idx, slide in enumerate(content_data.get('slides', [])):
            content = slide.get('content', '').replace('"', '\\"')
            wait_time = max(section_durations[idx] - 2.5, 2.0) if idx < len(section_durations) else 3.0
            
            manim_code += f"""        # Slide {idx + 1}
        text_{idx} = Text("{content}", font_size=28, font="JetBrains Mono", color=WHITE, line_spacing=1.5)
        text_{idx}.scale_to_fit_width(11)
        text_{idx}.move_to(ORIGIN)
        self.play(FadeIn(text_{idx}, shift=UP*0.2), run_time=1.0)
        self.wait({wait_time})
        self.play(FadeOut(text_{idx}), run_time=0.6)
        self.wait(0.5)
        
"""
        
        manim_code += "        self.wait(1)\n"
        
        # Render
        print("🎬 Rendering...")
        scene_file = f".temp_{output_name}.py"
        with open(scene_file, 'w') as f:
            f.write(manim_code)
        
        subprocess.run([
            "venv/bin/python", "-m", "manim", "-pql", scene_file, "SimpleTextScene"
        ], check=True, capture_output=True)
        
        video_path = f"media/videos/{scene_file.replace('.py', '')}/480p15/SimpleTextScene.mp4"
        
        # Combine
        print("🎵 Combining video + audio...")
        final_output = f"{output_name}_final.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-i", audio_file,
            "-c:v", "copy", "-c:a", "aac", "-shortest", final_output
        ], check=True, capture_output=True)
        
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS! {final_output}")
        print(f"{'='*60}\n")
        
        return {"final_video": final_output}
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None
"""
Correct implementation: Text slides THEN animated visuals
"""

def query_to_animated_video_v2(query: str, output_name: str = "output", audio_language: str = "english"):
    """
    Creates video with:
    1. Slide 1-2: Text definition/explanation
    2. Slide 3+: AI-generated animated visuals (like code overview)
    """
    try:
        print(f"\n{'='*60}")
        print(f"🎨 QUERY TO ANIMATED VIDEO (Text + Animations)")
        print(f"{'='*60}\n")
        
        import os, json, subprocess
        import openai
        
        api_key = os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI(api_key=api_key)
        
        # Step 1: Generate structure - text slides + animation concepts
        print(f"🤖 Generating content for: '{query}'...")
        
        structure_prompt = f"""Create educational video structure for: {query}

Generate:
1. Title
2. 1-2 TEXT slides (definition, explanation)
3. 2-3 ANIMATION concepts (visual representations)

Return JSON:
{{
    "title": "Topic Title",
    "text_slides": [
        {{
            "heading": "Definition",
            "content": "Clear 2-3 sentence explanation",
            "narration": "What to say"
        }}
    ],
    "animation_slides": [
        {{
            "concept": "Visual concept to animate (e.g., 'AI capabilities with 3 circles')",
            "narration": "What to explain during animation"
        }}
    ]
}}

Keep text slides simple. Animation concepts should be visual (diagrams, flows, icons)."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Create clear educational content with text and visual concepts."},
                {"role": "user", "content": structure_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        structure = json.loads(response.choices[0].message.content)
        print(f"✅ Generated structure:")
        print(f"   - {len(structure.get('text_slides', []))} text slides")
        print(f"   - {len(structure.get('animation_slides', []))} animation slides\n")
        
        # Step 2: Generate audio for all slides
        print("🎙️ Generating audio...")
        
        from simple_app import generate_audio_for_language, get_audio_duration
        
        audio_files = []
        durations = []
        
        # Title
        title_audio = f"{output_name}_title.aiff"
        generate_audio_for_language(f"{structure.get('title', '')}.", audio_language, title_audio, client)
        title_dur = get_audio_duration(title_audio)
        audio_files.append(title_audio)
        print(f"   Title: {title_dur:.2f}s")
        
        # Text slides
        for idx, slide in enumerate(structure.get('text_slides', [])):
            audio_file = f"{output_name}_text_{idx}.aiff"
            generate_audio_for_language(slide.get('narration', ''), audio_language, audio_file, client)
            dur = get_audio_duration(audio_file)
            durations.append(('text', dur))
            audio_files.append(audio_file)
            print(f"   Text slide {idx + 1}: {dur:.2f}s")
        
        # Animation slides
        for idx, slide in enumerate(structure.get('animation_slides', [])):
            audio_file = f"{output_name}_anim_{idx}.aiff"
            generate_audio_for_language(slide.get('narration', ''), audio_language, audio_file, client)
            dur = get_audio_duration(audio_file)
            durations.append(('animation', dur))
            audio_files.append(audio_file)
            print(f"   Animation slide {idx + 1}: {dur:.2f}s")
        
        # Combine audio
        print("\n🎵 Combining audio...")
        audio_file = f"{output_name}_audio.mp3"
        concat_file = f"{output_name}_concat.txt"
        
        with open(concat_file, 'w') as f:
            for audio in audio_files:
                f.write(f"file '{audio}'\n")
        
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_file, "-c:a", "libmp3lame", "-b:a", "192k", audio_file
        ], check=True, capture_output=True)
        
        # Cleanup
        for audio in audio_files:
            if os.path.exists(audio): os.remove(audio)
        if os.path.exists(concat_file): os.remove(concat_file)
        
        total_duration = get_audio_duration(audio_file)
        print(f"✅ Total audio: {total_duration:.2f}s\n")
        
        # Step 3: Generate Manim code
        print("🎨 Generating Manim code (text + AI animations)...")
        
        manim_code = generate_mixed_slides_code(structure, durations, client)
        
        # Step 4: Render
        print("🎬 Rendering...")
        scene_file = f".temp_{output_name}.py"
        with open(scene_file, 'w') as f:
            f.write(manim_code)
        
        subprocess.run([
            "venv/bin/python", "-m", "manim", "-pql", scene_file, "MixedScene"
        ], check=True, capture_output=True)
        
        video_path = f"media/videos/{scene_file.replace('.py', '')}/480p15/MixedScene.mp4"
        
        # Step 5: Combine
        print("🎵 Combining video + audio...")
        final_output = f"{output_name}_final.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-i", audio_file,
            "-c:v", "copy", "-c:a", "aac", "-shortest", final_output
        ], check=True, capture_output=True)
        
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS! {final_output}")
        print(f"{'='*60}\n")
        
        return {"final_video": final_output}
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_mixed_slides_code(structure, durations, client):
    """Generate Manim code with text slides THEN AI-generated animations"""
    
    title = structure.get("title", "")
    text_slides = structure.get("text_slides", [])
    animation_slides = structure.get("animation_slides", [])
    
    code = f"""from manim import *

class MixedScene(Scene):
    def construct(self):
        self.wait(1.2)
        
        # Title
        title = Text("{title}", font_size=48, font="JetBrains Mono", weight=BOLD, color=GOLD)
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title, shift=DOWN*0.3), run_time=0.8)
        self.wait(1.5)
        self.play(FadeOut(title), run_time=0.5)
        
"""
    
    dur_idx = 0
    
    # TEXT SLIDES
    for idx, slide in enumerate(text_slides):
        content = slide.get('content', '').replace('"', '\\"').replace('\n', ' ')
        wait_time = max(durations[dur_idx][1] - 2.5, 2.0) if dur_idx < len(durations) else 3.0
        dur_idx += 1
        
        code += f"""        # Text Slide {idx + 1}
        text_{idx} = Text(
            "{content}",
            font_size=26,
            font="JetBrains Mono",
            color=WHITE,
            line_spacing=1.5
        )
        text_{idx}.scale_to_fit_width(11)
        text_{idx}.move_to(ORIGIN)
        self.play(FadeIn(text_{idx}, shift=UP*0.2), run_time=1.0)
        self.wait({wait_time})
        self.play(FadeOut(text_{idx}), run_time=0.6)
        self.wait(0.5)
        
"""
    
    # ANIMATION SLIDES (AI-generated)
    for idx, slide in enumerate(animation_slides):
        concept = slide.get('concept', '')
        target_duration = durations[dur_idx][1] if dur_idx < len(durations) else 10.0
        dur_idx += 1
        
        # Generate AI animation code (like overview)
        animation_code = generate_ai_animation_for_concept(concept, target_duration, client)
        
        code += f"""        # Animation Slide {idx + 1}: {concept}
{animation_code}
        self.wait(0.5)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)
        self.wait(0.5)
        
"""
    
    code += "        self.wait(1)\n"
    
    return code


def generate_ai_animation_for_concept(concept, narration, target_duration, client):
    """Generate Manim animation code using AI (like code overview)"""
    
    prompt = f"""
    You are a Manim Animation Expert. Generate Python code using ManimCE to visualize the following concept.
    
    CONCEPT: {concept}
    NARRATION: "{narration}"
    DURATION: {target_duration} seconds
    
    CRITICAL RULES:
    1. **MATCH NARRATION EXACTLY**: 
       - If narration says "top to bottom", use `.arrange(DOWN)`.
       - If narration says "left to right", use `.arrange(RIGHT)`.
       - If narration lists items "1, 2, 3", show them in that EXACT order.
       - Do NOT reverse the order unless explicitly told to.
    
    2. **NO TEXT OVERFLOW**:
       - If text is long, do NOT put it inside a small box. Place it BELOW or NEXT TO the box.
       - If a label is wider than its container, move it OUTSIDE.
       - Use `font_size=32` or larger for important labels.
    
    3. **VISUAL CLARITY**:
       - Use VIBRANT colors (BLUE, GREEN, YELLOW, RED).
       - NEVER use default white lines.
       - Make arrows THICK (`stroke_width=6`).
       - Make text BIG and BOLD.
    
    4. **CODE STRUCTURE**:
       - Return ONLY the Python code inside a `def construct(self):` method.
       - Do NOT include `class` or `import` statements (I will add them).
       - Do NOT use `self.wait()` at the start. Start animating IMMEDIATELY.
    
    Example of Good Logic:
    ```python
    # If narration says "Stack grows from bottom to top"
    stack = VGroup()
    for i in range(3):
        item = VGroup(Rectangle(...), Text(...))
        # New items go ON TOP
        if len(stack) > 0:
            item.next_to(stack[-1], UP, buff=0)
        stack.add(item)
    ```
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a Manim animation expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    
    ai_code = response.choices[0].message.content.strip()
    
    # Remove markdown code blocks if present
    if "```python" in ai_code:
        ai_code = ai_code.split("```python")[1].split("```")[0]
    elif "```" in ai_code:
        ai_code = ai_code.split("```")[1].split("```")[0]
    
    # Indent the code (8 spaces for being inside construct method)
    indented_code = "\n".join("        " + line if line.strip() else "" for line in ai_code.split("\n"))
    
    return indented_code

"""
FIXED: Text slides with proper wrapping + Centered animations
"""

def query_to_animated_video_v3(query: str, output_name: str = "output", audio_language: str = "english"):
    """
    Fixed version with:
    1. Text slides with proper line wrapping (vertical layout)
    2. Centered animations with proper spacing
    3. No overlaps
    """
    try:
        print(f"\n{'='*60}")
        print(f"🎨 QUERY TO ANIMATED VIDEO V3 (FIXED)")
        print(f"{'='*60}\n")
        
        import os, json, subprocess, textwrap
        import openai
        
        api_key = os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI(api_key=api_key)
        
        # Step 1: Generate structure
        print(f"🤖 Generating content for: '{query}'...")
        
        structure_prompt = f"""Create educational video structure for: {query}

Generate 3-4 ANIMATION slides that explain this topic visually.

CRITICAL: Concepts must be ULTRA-SPECIFIC with exact details!

BAD (vague):
- "Visualize an array"
- "Show a linked list"
- "Demonstrate sorting"

GOOD (specific):
- "5 boxes in a row, labeled 1,2,3,4,5 from left to right"
- "4 circles connected by arrows: A→B→C→D, with 'Head' label on A"
- "Two rows of 3 boxes each: top row shows 5,3,8, bottom row shows 3,5,8 with arrow between"

REQUIREMENTS:
- Specify EXACT number of elements
- Specify EXACT labels/numbers to show
- Specify EXACT layout (horizontal, vertical, grid, etc.)
- Narration must describe EXACTLY what visual shows
- Each slide: 3-4 sentences

Return JSON:
{{
    "title": "Topic Title (max 40 chars)",
    "animation_slides": [
        {{
            "concept": "ULTRA-SPECIFIC: '5 boxes labeled 1,2,3,4,5 arranged horizontally'",
            "complexity": "simple",
            "narration": "3-4 sentences describing EXACTLY what the visual shows and why it matters"
        }},
        {{
            "concept": "ULTRA-SPECIFIC: Include exact numbers, labels, and layout",
            "complexity": "intermediate",
            "narration": "3-4 sentences matching the visual"
        }},
        {{
            "concept": "ULTRA-SPECIFIC: Be prescriptive about every detail",
            "complexity": "advanced",
            "narration": "3-4 sentences"
        }}
    ]
}}

EXAMPLES OF GOOD CONCEPTS:
- "3 circles labeled 'Input', 'Process', 'Output' connected by arrows left to right"
- "Grid of 6 boxes (2 rows, 3 cols) with numbers 1-6, showing dice faces"
- "Two boxes side by side: left says 'Unsorted: 5,2,8', right says 'Sorted: 2,5,8'"
- "Tree with 1 root circle labeled '5', 2 child circles labeled '3' and '7'"

IMPORTANT:
- Be PRESCRIPTIVE not DESCRIPTIVE
- Say WHAT to show, not what it represents
- Include exact numbers/labels
- Specify exact layout

Generate 3-4 slides."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Create clear educational content with animations only."},
                {"role": "user", "content": structure_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        structure = json.loads(response.choices[0].message.content)
        print(f"✅ Structure: {len(structure.get('animation_slides', []))} animation slides\n")
        
        # Step 2: Generate audio
        print("🎙️ Generating audio...")
        
        from simple_app import generate_audio_for_language, get_audio_duration
        
        audio_files = []
        durations = []
        
        # NO TITLE AUDIO - title is silent (just visual)
        # Title video timing: 1.2 (wait) + 0.8 (fadein) + 1.5 (display) + 0.5 (fadeout) = 4s
        # We'll add 4s of silence at the start
        
        # Animation slides only (no text slides)
        for idx, slide in enumerate(structure.get('animation_slides', [])):
            # 1. Generate TTS as MP3 first (safest format from APIs)
            raw_tts_file = f"{output_name}_anim_{idx}_raw.mp3"
            generate_audio_for_language(slide.get('narration', ''), audio_language, raw_tts_file, client)
            
            # 2. Convert to standardized WAV (PCM s16le, 44100Hz, Stereo)
            narration_wav = f"{output_name}_anim_{idx}_narration.wav"
            subprocess.run([
                "ffmpeg", "-y", "-i", raw_tts_file,
                "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                narration_wav
            ], check=True, capture_output=True)
            
            # Get narration duration
            narration_dur = get_audio_duration(narration_wav)
            if narration_dur is None:
                narration_text = slide.get('narration', '')
                word_count = len(narration_text.split())
                narration_dur = word_count / 2.5
                print(f"   ⚠️  Could not get audio duration, estimating: {narration_dur:.2f}s")
            
            # 3. Create silence padding if needed (also as standardized WAV)
            if idx == 0:
                audio_file = narration_wav
                total_dur = narration_dur
            else:
                silence_wav = f"{output_name}_anim_{idx}_silence.wav"
                subprocess.run([
                    "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                    "-t", "1.6", "-c:a", "pcm_s16le", silence_wav
                ], check=True, capture_output=True)
                
                # Combine silence + narration (WAV + WAV = Clean WAV)
                audio_file = f"{output_name}_anim_{idx}.wav"
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", silence_wav,
                    "-i", narration_wav,
                    "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[out]",
                    "-map", "[out]",
                    audio_file
                ], check=True, capture_output=True)
                
                # Cleanup temp files
                if os.path.exists(silence_wav): os.remove(silence_wav)
                if os.path.exists(narration_wav): os.remove(narration_wav)
                
                total_dur = 1.6 + narration_dur
            
            # Cleanup raw TTS
            if os.path.exists(raw_tts_file): os.remove(raw_tts_file)
            
            # Store NARRATION duration for animation timing (not total duration)
            durations.append(('animation', narration_dur))
            audio_files.append(audio_file)
            print(f"   Animation {idx + 1}: {narration_dur:.2f}s")
        
        # Combine with 4s silence at start for title
        print("\n🎵 Combining audio...")
        audio_file = f"{output_name}_audio.mp3"
        
        # Combine with 4s silence at start for title
        print("\n🎵 Combining audio...")
        audio_file = f"{output_name}_audio.mp3"
        
        # Create 4s silence for title (Standardized WAV)
        silence_file = f"{output_name}_silence.wav"
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-t", "4", "-c:a", "pcm_s16le", silence_file
        ], check=True, capture_output=True)
        
        # Combine: silence + animation audios
        concat_file = f"{output_name}_concat.txt"
        with open(concat_file, 'w') as f:
            f.write(f"file '{silence_file}'\n")
            for audio in audio_files:
                print(f"   DEBUG: Adding to concat: {audio} (exists: {os.path.exists(audio)})")
                f.write(f"file '{audio}'\n")
        
        # Concat all WAVs and output as MP3
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_file, "-c:a", "libmp3lame", "-b:a", "192k", audio_file
        ], check=True, capture_output=True)
        
        # Cleanup
        for audio in audio_files:
            if os.path.exists(audio): os.remove(audio)
        if os.path.exists(silence_file): os.remove(silence_file)
        if os.path.exists(concat_file): os.remove(concat_file)
        
        total_duration = get_audio_duration(audio_file)
        print(f"✅ Audio: {total_duration:.2f}s (4s silence + animations)\n")
        
        # Step 3: Generate Manim code
        print("🎨 Generating Manim code...")
        manim_code = generate_fixed_slides_code(structure, durations, client)
        
        # Step 4: Render
        print("🎬 Rendering...")
        scene_file = f".temp_{output_name}.py"
        with open(scene_file, 'w') as f:
            f.write(manim_code)
        
        try:
            subprocess.run([
                sys.executable, "-m", "manim", "-pqh", scene_file, "FixedScene"
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Manim Rendering Failed!")
            print(f"STDOUT:\n{e.stdout.decode()}")
            print(f"STDERR:\n{e.stderr.decode()}")
            raise e
        
        video_path = f"media/videos/{scene_file.replace('.py', '')}/1080p60/FixedScene.mp4"
        
        # Step 5: Combine
        print("🎵 Combining video + audio...")
        final_output = f"{output_name}_final.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-i", audio_file,
            "-c:v", "copy", "-c:a", "aac", "-shortest", final_output
        ], check=True, capture_output=True)
        
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS! {final_output}")
        print(f"{'='*60}\n")
        
        return {"final_video": final_output}
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_fixed_slides_code(structure, durations, client):
    """Generate Manim code with PROPER text wrapping and CENTERED animations"""
    
    import textwrap
    
    title = structure.get("title", "")
    text_slides = structure.get("text_slides", [])
    animation_slides = structure.get("animation_slides", [])
    
    # Wrap title if too long
    title_lines = textwrap.wrap(title, width=40)
    
    code = f"""from manim import *

class FixedScene(MovingCameraScene):
    def construct(self):
        # Kodnest-style dark background
        self.camera.background_color = "#0F172A"
        self.camera.frame.save_state()
        
        # Title (with wrapping for long titles)
        title_texts = VGroup()
"""
    
    # Add each line of the title
    for i, line in enumerate(title_lines):
        code += f"""        title_line_{i} = Text("{line}", font_size=36, font="JetBrains Mono", weight=BOLD, color=GOLD)
        title_texts.add(title_line_{i})
"""
    
    code += """        
        # Stack title lines vertically
        title_texts.arrange(DOWN, buff=0.2)
        title_texts.to_edge(UP, buff=0.5)
        
        self.play(FadeIn(title_texts, shift=DOWN*0.3), run_time=0.8)
        self.wait(1.5)
        self.play(FadeOut(title_texts), run_time=0.5)
        
"""
    
    dur_idx = 0
    
    # NO TEXT SLIDES - go straight to animations
    # ANIMATION SLIDES with explicit centering instructions
    for idx, slide in enumerate(animation_slides):
        concept = slide.get('concept', '')
        narration = slide.get('narration', '')  # GET THE NARRATION!
        target_duration = durations[dur_idx][1] if dur_idx < len(durations) else 10.0
        dur_idx += 1
        
        # Generate AI animation that MATCHES the narration
        animation_code = generate_centered_animation(concept, narration, target_duration, client)
        
        # Indent the code to fit inside construct()
        # Manual indentation to be safe
        indented_lines = []
        for line in animation_code.split('\n'):
            if line.strip():
                indented_lines.append("        " + line)
            else:
                indented_lines.append("")
        indented_code = '\n'.join(indented_lines)
        
        code += f"""        # Animation Slide {idx + 1}: {concept}
{indented_code}
        self.wait(0.5)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)
        
"""
    
    code += "        self.wait(1)\n"
    
    return code


# ============================================================
# AI SELF-CRITIC APPROACH (V7)
# ============================================================

def ai_critic_review_code(client, code, requirements, concept="", narration=""):
    """
    Ask AI to review generated Manim code and identify issues.
    
    Returns:
        (is_approved, feedback_list)
    """
    critic_prompt = f"""You are a code reviewer for Manim animations. Review this code:

WHAT IT SHOULD VISUALIZE:
Concept: {concept}
Narration: "{narration}"

CODE TO REVIEW:
```python
{code}
```

CRITICAL: Does the code actually visualize what the narration describes?
- If narration says "overlapping circles", code should have overlapping circles
- If narration says "triangle formation", code should arrange elements in a triangle
- If narration says "gradient from black to white", code should create a gradient

CRITICAL REQUIREMENTS (must pass):
- NO "class" or "def" statements (code will be inserted into a method)
- NO HTML or CSS
- Must have at least one self.play() call

VISUAL QUALITY CHECKS (reject if likely to cause overlaps):
- If creating 15+ Text labels in a small area → REJECT (too crowded)
- If 5+ labels all use .next_to(X, UP) or same direction → REJECT (will overlap)
- For color wheels: labels should be positioned OUTSIDE the wheel or use only representative colors
- Labels should be concise (max 20 chars each)
- If labels are positioned radially around a circle → APPROVE (good design)

IMPORTANT (check and suggest fixes):
- Labels should use .next_to() with DIFFERENT directions to avoid overlapping
- For many labels: position them in a circle/arc around the element, not all in one direction
- Use step-by-step reveals (FadeIn with lag_ratio or separate self.play calls)
- Don't show all elements at once

OPTIONAL (nice to have):
- Use Manim objects (RoundedRectangle, Circle, Text, Arrow, VGroup)
- Keep text labels concise

Check for critical issues:
- "class " or "def " → REJECT
- HTML tags (<html>, <div>, etc.) or CSS (margin:, padding:) → REJECT
- No self.play() calls → REJECT
- Invalid colors ONLY these are invalid: VIOLET, YELLOW_GREEN, BLUE_GREEN, RED_ORANGE, YELLOW_ORANGE, RED_VIOLET, BLUE_VIOLET
  (Note: RED, ORANGE, YELLOW, GREEN, TEAL, BLUE, PURPLE, PINK, GOLD, GRAY, WHITE, BLACK are all VALID!)

Check for overlap issues (reject if severe):
- 5+ Text objects all with .next_to(X, UP) → REJECT (same direction overlap)
- 15+ Text labels without radial/circular positioning → REJECT (too crowded)
- All elements in one self.play() → SUGGEST step-by-step reveals

Return JSON:
{{
    "approved": true/false,
    "issues": ["critical issue 1"],
    "suggestions": ["position labels radially around circle", "use fewer representative labels"]
}}

If labels are positioned smartly (radially, in different directions, with spacing), APPROVE even if many.
If labels will overlap (same direction, too many in small space), REJECT.
Be SMART about visual quality!"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a STRICT code reviewer. Reject code with critical issues OR poor visual quality (overlapping text, too many labels, crowded layouts)."},
                {"role": "user", "content": critic_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        review = json.loads(response.choices[0].message.content)
        is_approved = review.get("approved", False)
        issues = review.get("issues", [])
        
        # MANDATORY POST-CHECK: Even if critic approves, verify critical issues
        if is_approved:
            invalid_colors = ['VIOLET', 'YELLOW_GREEN', 'BLUE_GREEN', 'RED_ORANGE', 'YELLOW_ORANGE', 'RED_VIOLET', 'BLUE_VIOLET']
            for color in invalid_colors:
                if color in code:
                    is_approved = False
                    issues.append(f"Invalid color {color} found. Use valid Manim colors only.")
                    break
            
            if 'class ' in code or 'def ' in code:
                is_approved = False
                issues.append("Code contains class/def statements. Remove them - code should be method body only.")
        
        return is_approved, issues
        
    except Exception as e:
        print(f"   ⚠️  Critic failed: {e}")
        # Fallback: do basic validation
        has_class = 'class ' in code or 'def ' in code
        has_play = 'self.play(' in code
        
        if has_class:
            return False, ["Code contains class/def statements. Remove them."]
        if not has_play:
            return False, ["Code has no animations. Add self.play() calls."]
        
        return True, []


def generate_with_ai_critic_loop(client, prompt, requirements, concept="", narration="", max_attempts=3):
    """
    Generate Manim code with AI critic validation loop.
    
    Returns:
        (code, success)
    """
    messages = [
        {"role": "system", "content": "You are a Manim expert. Generate clean, visual animations."},
        {"role": "user", "content": prompt}
    ]
    
    for attempt in range(max_attempts):
        print(f"      🔄 Attempt {attempt + 1}/{max_attempts}...")
        
        # Generate code
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3
            )
            
            code = response.choices[0].message.content.strip()
            
            # Clean markdown
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()
            
            # Critic review (now with concept and narration)
            is_approved, feedback = ai_critic_review_code(client, code, requirements, concept, narration)
            
            if is_approved:
                print(f"      ✅ Critic approved!")
                return code, True
            
            # Show feedback
            print(f"      ❌ Critic rejected:")
            for issue in feedback[:3]:  # Show first 3 issues
                print(f"         - {issue}")
            
            # Add feedback to conversation
            feedback_text = "Code review feedback:\n" + "\n".join(f"- {issue}" for issue in feedback)
            feedback_text += "\n\nFix these issues and regenerate."
            
            messages.append({"role": "assistant", "content": code})
            messages.append({"role": "user", "content": feedback_text})
            
        except Exception as e:
            print(f"      ⚠️  Generation error: {e}")
            continue
    
    print(f"      ❌ Failed after {max_attempts} attempts")
    return None, False


def generate_centered_animation(concept, narration, target_duration, client):
    """Generate animation that MATCHES narration exactly"""
    
    prompt = f"""Create Manim visualization for: {concept}

NARRATION (READ CAREFULLY):
"{narration}"

CRITICAL RULES:
1. Extract KEY TERMS from the narration above
2. Use those EXACT terms as labels
3. DO NOT use generic labels like "1: Class", "2: Main"
4. USE ACTUAL WORDS from the narration!

VALID COLORS (ONLY use these):
- Basic colors (no variants): RED, BLUE, GREEN, YELLOW, ORANGE, PURPLE, PINK, TEAL, GOLD, GRAY, WHITE, BLACK
- Colors WITH variants: ONLY Red, Blue, Green, Yellow have _A, _B, _C, _D, _E variants
  Example: RED_A, RED_B, BLUE_C, GREEN_D, YELLOW_E are valid
- ⚠️ ORANGE_A, PURPLE_A, PINK_A, TEAL_A, GOLD_A, GRAY_A DO NOT EXIST!
- ⚠️ NEVER use: VIOLET, YELLOW_GREEN, BLUE_GREEN, RED_ORANGE, YELLOW_ORANGE, etc.

VISUALIZATION RULES:
- **STACKS**: Use a VGroup of Rectangles arranged VERTICALLY (`.arrange(UP, buff=0)`).
- **ARRAYS/LISTS**: Use a VGroup of Rectangles arranged HORIZONTALLY (`.arrange(RIGHT, buff=0)`).
- **LAYOUT**: 
  - ALWAYS use `.next_to()` or `.arrange()` for positioning.
  - NEVER use absolute coordinates like `[3, 4, 0]`.
  - Keep everything centered with `.move_to(ORIGIN)`.

Example Stack Code:
```python
stack = VGroup()
# Push 1
box1 = Rectangle(height=1, width=2)
label1 = Text("Data").move_to(box1)
item1 = VGroup(box1, label1)
if len(stack) > 0:
    item1.next_to(stack[-1], UP, buff=0)
else:
    item1.move_to(DOWN * 2)
stack.add(item1)
self.play(Create(item1))
```

Example Overflow Code:
```python
# Stack is full
overflow_item = Rectangle(height=1, width=2, color=RED)
overflow_label = Text("Overflow").move_to(overflow_item)
grp = VGroup(overflow_item, overflow_label)
grp.next_to(stack, UP, buff=0.5)
self.play(FadeIn(grp))
self.play(grp.animate.shift(DOWN * 0.2)) # Try to push
self.play(grp.animate.shift(UP * 0.2), run_time=0.3) # Bounce back
self.play(Indicate(grp, color=RED))
self.play(FadeOut(grp))
```

Duration: {target_duration:.1f}s

CRITICAL: EVERY box MUST have a label! NO empty boxes!
CRITICAL: Return code with NO indentation (flush left).
CRITICAL: DO NOT generate "class Scene" or "def construct". ONLY generate the animation commands (self.play, etc).
CRITICAL: DO NOT start with self.wait(). Start animating IMMEDIATELY!

Generate code:"""
    
    # Simple validation loop - regenerate if code has critical issues
    max_attempts = 3
    ai_code = "" # Initialize ai_code to ensure it's defined after the loop
    for attempt in range(max_attempts):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a Manim expert. Generate VIBRANT, COLORFUL animations. Use ACTUAL terms from narration. NO generic labels. NO class definitions. Return FLUSH LEFT code. START IMMEDIATELY."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2 + (attempt * 0.1)  # Increase temperature on retries
        )
        
        ai_code = response.choices[0].message.content.strip()
        
        # Remove initial wait if present
        import re
        ai_code = re.sub(r'^\s*self\.wait\([^)]+\)\s*\n', '', ai_code)
        
        # Remove markdown
        if "```python" in ai_code:
            ai_code = ai_code.split("```python")[1].split("```")[0]
        elif "```" in ai_code:
            ai_code = ai_code.split("```")[1].split("```")[0]
            
        # Check if AI generated a class/def despite instructions
        if "def construct(self):" in ai_code:
            parts = ai_code.split("def construct(self):")
            if len(parts) > 1:
                ai_code = parts[1]
        
        # SIMPLE VALIDATION - check for critical issues
        issues = []
        
        # Check 1: No class/def statements
        if 'class ' in ai_code or 'def ' in ai_code:
            issues.append("Contains class/def statements")
        
        # Check 2: No invalid colors
        # List of colors that DON'T have _A, _B, _C variants in Manim
        colors_without_variants = ['ORANGE', 'PURPLE', 'PINK', 'TEAL', 'GOLD', 'GRAY']
        for color in colors_without_variants:
            for suffix in ['_A', '_B', '_C', '_D', '_E']:
                if f'{color}{suffix}' in ai_code:
                    issues.append(f"Invalid color: {color}{suffix} (this color has no variants)")
                    break
        
        # Also check for completely invalid color names
        invalid_colors = ['VIOLET', 'YELLOW_GREEN', 'BLUE_GREEN', 'RED_ORANGE', 'YELLOW_ORANGE', 'RED_VIOLET', 'BLUE_VIOLET']
        for color in invalid_colors:
            if color in ai_code:
                issues.append(f"Invalid color: {color}")
                break
        
        # Check 3: Must have animations
        if 'self.play(' not in ai_code:
            issues.append("No animations found")
        
        # Check 4: Detect potential text overlap
        # Count how many Text objects are created
        text_count = ai_code.count('Text(')
        # If many Text objects and they're positioned with .move_to() at similar locations, likely overlap
        if text_count >= 3:
            # Check if multiple labels use similar positioning (e.g., all use .get_top())
            if ai_code.count('.get_top()') >= 3 or ai_code.count('.get_center()') >= 3:
                issues.append("Potential text overlap: multiple labels at similar positions")
        
        # If no issues, we're done!
        if not issues:
            print(f"      ✅ Code validated (attempt {attempt + 1})")
            break
        
        # If issues found and not last attempt, regenerate
        if attempt < max_attempts - 1:
            print(f"      ⚠️  Attempt {attempt + 1} failed: {', '.join(issues)}")
            print(f"      🔄 Regenerating...")
        else:
            print(f"      ❌ All attempts failed, using last generated code")

    # Smart Dedent (preserve relative indentation)
    lines = ai_code.split('\n')
    # Remove empty leading lines
    while lines and not lines[0].strip():
        lines.pop(0)
        
    if lines:
        # Find indent of first line
        first_indent = len(lines[0]) - len(lines[0].lstrip())
        # Dedent all lines by that amount
        new_lines = []
        for line in lines:
            if len(line.strip()) == 0:
                new_lines.append("")
            else:
                # Only strip if it has enough indent
                current_indent = len(line) - len(line.lstrip())
                if current_indent >= first_indent:
                    new_lines.append(line[first_indent:])
                else:
                    new_lines.append(line.lstrip()) # Fallback
        ai_code = '\n'.join(new_lines)
    
    # Remove comments
    ai_code = '\n'.join(line for line in ai_code.split('\n') if not line.strip().startswith('#'))
    
    # STEP 2: Wrap in a function so autopep8 can parse it
    import textwrap
    import autopep8
    wrapped = f"def dummy():\n{textwrap.indent(ai_code, '    ')}"
    
    # STEP 3: Fix with autopep8
    fixed = autopep8.fix_code(wrapped)
    print("DEBUG - autopep8 output:")
    print(fixed[:500])
    print("---")
    
    # STEP 4: Extract the function body and dedent
    lines = fixed.split('\n')[1:]  # Skip first line
    body = '\n'.join(lines)
    ai_code = textwrap.dedent(body).strip()
    
    print("DEBUG - Final AI Code (should be flush left):")
    print(ai_code[:200])
    print("---")
    
    # --- SMART SYNC FIXER ---
    # Calculate how much time the AI's animations take and adjust the wait
    import re
    
    # Find all run_time=X.X
    run_times = re.findall(r"run_time\s*=\s*(\d+\.?\d*)", ai_code)
    total_anim_time = sum(float(t) for t in run_times)
    
    # Count default play calls (approx 1.0s each if no run_time specified)
    # This is a heuristic: count .play( but subtract ones with run_time
    total_plays = ai_code.count("self.play(")
    explicit_runs = len(run_times)
    default_runs = total_plays - explicit_runs
    total_anim_time += default_runs * 1.0
    
    # FIX 1: Enforce Faster Animations (Sync Fix)
    # Replace long run_times
    ai_code = re.sub(r'run_time=(\d+\.?\d*)', lambda m: f'run_time={min(float(m.group(1)), 0.5)}', ai_code)
    
    # FIX 1.5: Convert Rectangle to RoundedRectangle for Kodnest style
    def convert_to_rounded(match):
        """Convert Rectangle(...) to RoundedRectangle(corner_radius=0.15, ...)"""
        params = match.group(1)
        if 'corner_radius' in params:
            return match.group(0)
        return f'RoundedRectangle(corner_radius=0.15, {params})'
    
    ai_code = re.sub(r'(?<!Rounded)Rectangle\(([^)]*)\)', convert_to_rounded, ai_code)
    
    # FIX 1.6: Fix invalid color names
    color_map = {
        'red_orange': 'ORANGE',
        'RED_ORANGE': 'ORANGE',
        'blue_green': 'TEAL',
        'BLUE_GREEN': 'TEAL',
        'yellow_green': 'GREEN_C',
        'YELLOW_GREEN': 'GREEN_C',
        'red_purple': 'PURPLE',
        'RED_PURPLE': 'PURPLE',
        'blue_purple': 'PURPLE_C',
        'BLUE_PURPLE': 'PURPLE_C',
        'red_violet': 'PURPLE',
        'RED_VIOLET': 'PURPLE',
        'blue_violet': 'PURPLE_C',
        'BLUE_VIOLET': 'PURPLE_C',
        'yellow_orange': 'GOLD',
        'YELLOW_ORANGE': 'GOLD',
        'VIOLET': 'PURPLE',  # VIOLET is not a Manim color
    }
    
    for invalid, valid in color_map.items():
        ai_code = ai_code.replace(f'color={invalid}', f'color={valid}')
        ai_code = ai_code.replace(f'color="{invalid}"', f'color={valid}')
        ai_code = ai_code.replace(f"color='{invalid}'", f'color={valid}')
        ai_code = ai_code.replace(f'"{invalid}"', f'{valid}')  # For color lists
        ai_code = ai_code.replace(f"'{invalid}'", f'{valid}')
        # Also replace standalone color names in lists
        ai_code = ai_code.replace(f', {invalid},', f', {valid},')
        ai_code = ai_code.replace(f', {invalid}]', f', {valid}]')
        ai_code = ai_code.replace(f'[{invalid},', f'[{valid},')

    # FIX 3: Runtime Text Fitting (Prevent Overflow)
    text_fitting_code = """
    # AUTO-LAYOUT: Check for text overflow
    # Iterate over all groups. If a group has a geometry and a text, check sizes.
    for mob in self.mobjects:
        if isinstance(mob, (Group, VGroup)) and len(mob) >= 2:
            shape = None
            text = None
            for sub in mob:
                if isinstance(sub, (RoundedRectangle, Rectangle, Square, Circle)):
                    shape = sub
                elif isinstance(sub, Text):
                    text = sub
            
            if shape and text:
                # If text is wider than shape (with margin), SCALE IT DOWN
                # Do NOT move it, as that causes overlaps in stacks
                max_width = shape.width * 0.85
                if text.width > max_width:
                    text.scale(max_width / text.width)
                    
                # Ensure it's centered
                text.move_to(shape.get_center())
                
                # Ensure visibility
                if text.get_color() == shape.get_color():
                    text.set_color(WHITE)
    """
    import textwrap
    text_fitting_code = textwrap.dedent(text_fitting_code)
    ai_code += f"\n{text_fitting_code}\n"

    # FIX 2: Safety Scaling (Content Overflow Fix)
    scaling_code = """
    # SAFETY SCALING: Ensure nothing goes off-screen
    all_mobs = Group(*self.mobjects)
    if len(all_mobs) > 0:
        # Aggressive scaling to fit in 12x6.5 box
        if all_mobs.width > 12:
            all_mobs.scale(12 / all_mobs.width)
        if all_mobs.height > 6.5:
            all_mobs.scale(6.5 / all_mobs.height)
        
        # Center if off-screen
        if all_mobs.get_left()[0] < -6.5:
            all_mobs.shift(RIGHT * ((-6.5) - all_mobs.get_left()[0]))
        if all_mobs.get_right()[0] > 6.5:
            all_mobs.shift(LEFT * (all_mobs.get_right()[0] - 6.5))
        
        # Ensure it's centered if it's the only thing
        # all_mobs.move_to(ORIGIN) 
    """
    import textwrap
    scaling_code = textwrap.dedent(scaling_code)
    
    # Calculate remaining wait needed
    wait_time = max(0.1, target_duration - total_anim_time)
    
    # Append the wait call AND the scaling code
    ai_code += f"\n{scaling_code}\nself.wait({wait_time:.2f})"
    
    # Remove imports/class/def
    cleaned = []
    for line in ai_code.split('\n'):
        s = line.strip()
        if s.startswith(('class ', 'def ', 'import ', 'from ')):
            continue
        cleaned.append(line)
    ai_code = '\n'.join(cleaned)
    
    # VALIDATION & FIXES
    import re
    
    # STEP 1: Extract what the narration mentions
    narration_lower = narration.lower()
    mentioned_items = set()
    
    # Extract letters/numbers mentioned (like "box A", "element F", "node 5")
    for match in re.finditer(r'\b([a-z])\b', narration_lower):
        mentioned_items.add(match.group(1).upper())
    for match in re.finditer(r'\b(\d+)\b', narration_lower):
        mentioned_items.add(match.group(1))
    
    # Extract keywords (like "arrow", "front", "rear", "head", "tail")
    keywords = ['arrow', 'front', 'rear', 'head', 'tail', 'top', 'bottom', 'left', 'right']
    mentioned_keywords = set()
    for keyword in keywords:
        if keyword in narration_lower:
            mentioned_keywords.add(keyword)
    
    print(f"   📝 Narration mentions: {mentioned_items}, Keywords: {mentioned_keywords}")
    
    # STEP 2: Check if mentioned items exist in code
    code_lower = ai_code.lower()
    missing_items = []
    for item in mentioned_items:
        # Check if this item appears as a variable or label
        if not (f'"{item}"' in ai_code or f"'{item}'" in ai_code or f'_{item.lower()}' in code_lower or f'{item.lower()}_' in code_lower):
            missing_items.append(item)
    
    if missing_items:
        print(f"   ⚠️  WARNING: Narration mentions {missing_items} but they don't appear in code!")
    
    # STEP 3: Check if mentioned keywords exist
    missing_keywords = []
    for keyword in mentioned_keywords:
        if keyword not in code_lower:
            missing_keywords.append(keyword)
    
    if missing_keywords:
        print(f"   ⚠️  WARNING: Narration mentions {missing_keywords} but they don't appear in code!")
    
    # Fix colors
    for bad, good in [('BROWN', 'MAROON'), ('CYAN', 'TEAL'), ('MAGENTA', 'PINK')]:
        ai_code = ai_code.replace(f'color={bad}', f'color={good}')
        ai_code = ai_code.replace(f', {bad}', f', {good}')
    
    # Force reasonable sizes
    ai_code = re.sub(r'Rectangle\(width=(\d+\.?\d*)', lambda m: f'Rectangle(width={min(float(m.group(1)), 2.5)}', ai_code)
    ai_code = re.sub(r'height=(\d+\.?\d*)', lambda m: f'height={min(float(m.group(1)), 1.5)}', ai_code)
    ai_code = re.sub(r'Circle\(radius=(\d+\.?\d*)', lambda m: f'Circle(radius={min(float(m.group(1)), 0.7)}', ai_code)
    
    # Smart font_size: smaller for circles, normal for rectangles
    # First, reduce font size for Text objects that are inside circles
    lines = ai_code.split('\n')
    new_lines = []
    for i, line in enumerate(lines):
        # If this line creates a Text and the previous few lines created a Circle, use smaller font
        if '= Text(' in line:
            # Look back to see if we're in a circle context
            context = '\n'.join(lines[max(0, i-3):i])
            if 'Circle(' in context:
                # Use smaller font for circles
                line = re.sub(r'font_size=\d+', 'font_size=16', line)
                # If no font_size specified, add it
                if 'font_size=' not in line:
                    line = line.replace('Text(', 'Text(', 1)
                    line = re.sub(r'Text\(([^)]+)\)', r'Text(\1, font_size=16)', line)
            else:
                # Use normal font for rectangles
                line = re.sub(r'font_size=\d+', 'font_size=24', line) # Increased from 20 to 24
        
        new_lines.append(line)
    ai_code = '\n'.join(new_lines)
    
    # CRITICAL: Ensure all Text objects are moved!
    # If a Text object is created but never .move_to or .next_to, it will sit at ORIGIN
    lines = ai_code.split('\n')
    text_vars = {} # name -> line_idx
    positioned_vars = set()
    
    for i, line in enumerate(lines):
        if '= Text(' in line:
            var_name = line.split('=')[0].strip()
            text_vars[var_name] = i
        
        # Check for positioning methods
        for var in text_vars:
            if var in line and ('.move_to(' in line or '.next_to(' in line or '.to_edge(' in line or '.shift(' in line):
                positioned_vars.add(var)
    
    ai_code = '\n'.join(lines)
    
    # CRITICAL VALIDATION: Check for empty Text objects
    # Replace Text("") or Text('') with Text("?")
    ai_code = re.sub(r'Text\(\s*["\']["\']', 'Text("?"', ai_code)
    
    # CRITICAL VALIDATION: Detect boxes without labels
    # This is complex, but we can at least warn in comments
    lines = ai_code.split('\n')
    box_vars = []
    label_vars = []
    
    for line in lines:
        # Find Rectangle/Circle variable assignments
        if '= Rectangle(' in line or '= Circle(' in line:
            var_name = line.split('=')[0].strip()
            if var_name and not var_name.startswith('#'):
                box_vars.append(var_name)
        # Find Text variable assignments
        if '= Text(' in line:
            var_name = line.split('=')[0].strip()
            if var_name and not var_name.startswith('#'):
                label_vars.append(var_name)
    
    # If we have more boxes than labels, add warning
    if len(box_vars) > len(label_vars):
        print(f"⚠️  WARNING: {len(box_vars)} boxes but only {len(label_vars)} labels!")
        print(f"   Boxes: {box_vars}")
        print(f"   Labels: {label_vars}")
        print(f"   Some boxes may be empty!")
    
    # Fix GrowArrow
    ai_code = re.sub(r'GrowArrow\(\*(\w+)\)', r'*[GrowArrow(a) for a in \1]', ai_code)
    
    # (Nuclear Option removed - replaced by Global Safety Scaling at the end)
    
    ai_code = '\n'.join(lines)
    
    # Dedent
    import textwrap
    ai_code = textwrap.dedent(ai_code).strip()
    
    return ai_code

