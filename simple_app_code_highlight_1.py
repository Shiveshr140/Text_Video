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
from pipeline import AudioGenerator
import json
import wave
import contextlib


# ============================================================
# HELPER: Get Audio Duration
# ============================================================

def concatenate_audio_files(audio_files, output_file):
    """Concatenate multiple audio files into one using FFmpeg"""
    if not audio_files:
        return None
    
    if len(audio_files) == 1:
        # Just copy the single file
        subprocess.run(['cp', audio_files[0], output_file], check=True)
        return output_file
    
    # Create FFmpeg concat file
    concat_file = output_file.replace('.aiff', '_concat.txt')
    with open(concat_file, 'w') as f:
        for audio_file in audio_files:
            f.write(f"file '{os.path.abspath(audio_file)}'\n")
    
    # Concatenate using FFmpeg
    try:
        subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',
            output_file
        ], capture_output=True, check=True)
        
        # Clean up
        os.remove(concat_file)
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"⚠️  FFmpeg concat failed, trying alternative method...")
        # Alternative: convert to wav, concat, convert back
        wav_files = []
        for i, audio_file in enumerate(audio_files):
            wav_file = f"{output_file}_temp_{i}.wav"
            subprocess.run([
                'ffmpeg', '-y', '-i', audio_file,
                '-acodec', 'pcm_s16le', '-ar', '44100',
                wav_file
            ], capture_output=True, check=True)
            wav_files.append(wav_file)
        
        # Create concat file for wavs
        concat_file_wav = output_file.replace('.aiff', '_concat_wav.txt')
        with open(concat_file_wav, 'w') as f:
            for wav_file in wav_files:
                f.write(f"file '{os.path.abspath(wav_file)}'\n")
        
        # Concatenate wavs
        output_wav = output_file.replace('.aiff', '_temp.wav')
        subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', concat_file_wav,
            '-acodec', 'pcm_s16le', '-ar', '44100',
            output_wav
        ], capture_output=True, check=True)
        
        # Convert back to aiff
        subprocess.run([
            'ffmpeg', '-y', '-i', output_wav,
            output_file
        ], capture_output=True, check=True)
        
        # Clean up
        os.remove(concat_file_wav)
        os.remove(output_wav)
        for wav_file in wav_files:
            if os.path.exists(wav_file):
                os.remove(wav_file)
        
        return output_file


def get_audio_duration(audio_file):
    """Get duration of audio file in seconds"""
    try:
        # Convert aiff to wav for easier processing
        wav_file = audio_file.replace('.aiff', '_temp.wav')
        subprocess.run([
            'ffmpeg', '-y', '-i', audio_file,
            '-acodec', 'pcm_s16le', '-ar', '44100',
            wav_file
        ], capture_output=True, check=True)
        
        with contextlib.closing(wave.open(wav_file, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        
        # Clean up temp file
        if os.path.exists(wav_file):
            os.remove(wav_file)
        
        return duration
    except Exception as e:
        print(f"⚠️  Could not get audio duration: {e}")
        return None


# ============================================================
# HELPER: Generate Controlled Manim Code (No Random AI)
# ============================================================

def generate_slides_code(parsed_data, audio_duration=None, num_sections=None):
    """
    Generate CONSISTENT Manim code for slides
    NOT random AI - CONTROLLED output every time
    """
    title = parsed_data.get("title", "Presentation")
    sections = parsed_data.get("sections", [])
    
    code = """from manim import *

class EducationalScene(Scene):
    def construct(self):
        # Initial wait
        self.wait(1.2)
        
        # Title
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
        
""".format(title=title)
    
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
        
        # Calculate timing based on actual audio duration if available
        if audio_duration and num_sections:
            # Distribute audio time across sections
            # Account for title (2s) and overhead per section (~2.7s each)
            total_overhead = 2.0 + (num_sections * 2.7)  # title + section overheads
            available_time = audio_duration - total_overhead
            time_per_section = max(available_time / num_sections, 3.0)
            wait_time = time_per_section
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
        heading_{idx}.next_to(title, DOWN, buff=0.7)
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
            font_size=14,
            color=WHITE,
            font="Helvetica"
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
            
            # Debug: Show sample of code-mixed narration
            print("📝 Sample narration:")
            print(narration_text[:200] + "..." if len(narration_text) > 200 else narration_text)
            print()
        else:
            narration_text = english_narration
        
        # Step 5: Generate audio in target language
        audio_file = f"{output_name}_audio.aiff"
        generate_audio_for_language(narration_text, audio_language, audio_file, client)
        
        # Step 5.5: Get audio duration for perfect sync
        print("⏱️  Measuring audio duration...")
        audio_duration = get_audio_duration(audio_file)
        if audio_duration:
            print(f"✅ Audio duration: {audio_duration:.2f} seconds\n")
        else:
            print("⚠️  Using fallback timing\n")
        
        # Step 6: Generate Manim code with audio-based timing
        print("🎨 Generating Manim slide code with audio sync...")
        manim_code = generate_slides_code(parsed_data, audio_duration, num_sections)
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
                return None
            
            if not voice_id:
                print("❌ ELEVENLABS_VOICE_ID not set!")
                return None
            
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
# CODE HIGHLIGHTING SYSTEM (Whisper + AST-based)
# ============================================================
# 
# This system provides precise code highlighting synchronized with narration
# using:
# 1. Whisper for word-level audio timestamps
# 2. AST parsing to identify code blocks
# 3. Semantic mapping of narration to code concepts
# 4. Timeline-based animation generation
#
# Architecture:
# - transcribe_audio_with_timestamps(): Gets word-level timestamps from audio
# - parse_code_to_blocks(): Parses code into AST nodes (functions, loops, etc.)
# - map_narration_to_code(): Maps narration words to code blocks
# - generate_timeline_animations(): Creates timeline-based Manim animations
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


def parse_code_to_blocks(code_content, language="python"):
    """
    Parse code into semantic blocks using AST.
    
    Identifies:
    - Functions/methods
    - Classes
    - Loops (for, while)
    - Conditionals (if, else, elif)
    - Variable declarations
    - Import statements
    
    Args:
        code_content: Source code as string
        language: Programming language ("python", "java", "javascript", etc.)
        
    Returns:
        List of dicts with keys: 'type', 'name', 'start_line', 'end_line', 'code'
        Example: [
            {'type': 'function', 'name': 'main', 'start_line': 3, 'end_line': 10, 'code': '...'},
            {'type': 'for_loop', 'name': None, 'start_line': 5, 'end_line': 8, 'code': '...'}
        ]
    """
    blocks = []
    lines = code_content.strip().split('\n')
    
    if language == "python":
        try:
            import ast
            
            tree = ast.parse(code_content)
            
            for node in ast.walk(tree):
                # Functions
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
                
                # Classes
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
                
                # For loops
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
                
                # While loops
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
                
                # If statements
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
            
        except SyntaxError as e:
            print(f"⚠️  Python AST parsing failed: {e}")
            print("   Falling back to simple line-based parsing\n")
    
    # Fallback: Simple pattern-based parsing for non-Python languages
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
    
    # Find functions (pattern: "def function_name" or "function function_name" or "public static void main")
    # For Java: "public static void main" or "public void methodName" etc.
    for i, line in enumerate(lines, 1):
        # Match Java/C# function patterns: public static void main, public void method, etc.
        if re.match(r'^\s*(public|private|protected)?\s*(static)?\s*\w+\s+\w+\s*\(', line) or \
           re.match(r'^\s*(def|function)\s+\w+\s*\(', line):
            # Find function end (next function/class or end of file)
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
            
            # Extract function name
            func_match = re.search(r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(', line)
            if not func_match:
                func_match = re.search(r'(?:def|function)\s+(\w+)\s*\(', line)
            func_name = func_match.group(1) if func_match else None
            
            # Only add if not already inside a class block
            is_inside_class = False
            for block in blocks:
                if block['type'] == 'class' and block['start_line'] <= i <= block['end_line']:
                    is_inside_class = True
                    break
            
            if not is_inside_class or func_name == 'main':  # Always include main method
                blocks.append({
                    'type': 'function',
                    'name': func_name,
                    'start_line': i,
                    'end_line': end_line,
                    'code': '\n'.join(lines[i-1:end_line])
                })
        
        # Find for loops (include nested loops inside functions)
        elif re.match(r'^\s*for\s*\(', line):
            end_line = find_block_end(lines, i)
            blocks.append({
                'type': 'for_loop',
                'name': None,
                'start_line': i,
                'end_line': end_line,
                'code': '\n'.join(lines[i-1:end_line])
            })
        
        # Find if statements (only if not already part of a function/class)
        elif re.match(r'^\s*if\s*\(', line):
            # Check if this if is already inside a function/class block
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
    
    # Sort blocks by start_line to maintain code order
    blocks.sort(key=lambda x: x['start_line'])
    
    return blocks


def find_block_end(lines, start_line):
    """Helper to find end of code block (simple brace matching)"""
    brace_count = 0
    for i in range(start_line - 1, len(lines)):
        line = lines[i]
        brace_count += line.count('{') - line.count('}')
        if brace_count == 0 and i > start_line - 1:
            return i + 1
    return len(lines)


def map_narration_to_code(narration_words, code_blocks, english_narration, client):
    """
    Map narration words to code blocks using semantic matching.
    
    Uses GPT-4 to understand context and map narration phrases to code concepts.
    
    Args:
        narration_words: List of word dicts from Whisper (with timestamps)
        code_blocks: List of code block dicts from parse_code_to_blocks()
        english_narration: Full narration text
        client: OpenAI client
        
    Returns:
        List of dicts: [{
            'code_block': {...},  # Code block to highlight
            'start_time': 2.5,    # When to start highlighting
            'end_time': 5.0,       # When to stop highlighting
            'confidence': 0.9      # Match confidence
        }, ...]
    """
    print("🔗 Mapping narration to code blocks...")
    
    # Build code blocks summary for AI
    blocks_summary = []
    for block in code_blocks:
        blocks_summary.append({
            'type': block['type'],
            'name': block.get('name', 'unnamed'),
            'lines': f"{block['start_line']}-{block['end_line']}",
            'preview': block['code'][:100] + "..." if len(block['code']) > 100 else block['code']
        })
    
    # Use GPT-4 to map narration to code blocks with better context
    # Number the narration sentences to help with mapping
    narration_sentences = english_narration.split('. ')
    numbered_narration = "\n".join([f"{i+1}. {s}" for i, s in enumerate(narration_sentences)])
    
    mapping_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are an expert at mapping code explanations to code blocks.

Given numbered narration sentences and code blocks, identify which sentences explain which code blocks.

Return JSON format:
{
    "mappings": [
        {
            "sentence_number": 3,  // Which sentence (1-indexed) explains this block
            "code_block_index": 2,  // Index in code_blocks array
            "phrase_in_sentence": "the outer for loop",  // Key phrase that identifies the block
            "confidence": 0.9  // How confident (0.0-1.0) that this mapping is correct
        },
        ...
    ]
}

CRITICAL RULES - BE VERY CONSERVATIVE:
1. Only map when narration EXPLICITLY and CLEARLY explains a SPECIFIC code block
2. Map in the ORDER narration appears (sentence 1, then 2, etc.)
3. Be precise - if explaining "main function", map to main function block
4. If explaining "first for loop", map to the FIRST for loop in code
5. If explaining "nested for loop", map to the NESTED for loop
6. DO NOT map:
   - Generic mentions like "code", "program", "this code"
   - System.out.print statements (unless sentence explicitly says "System.out.print")
   - Single line statements (unless explicitly explained)
   - Variable declarations (unless explicitly explained)
7. Only map major structures: functions, classes, loops, conditionals
8. Set confidence < 0.7 if unsure - we'll filter those out
9. If a sentence doesn't clearly explain a specific block, DON'T map it"""
            },
            {
                "role": "user",
                "content": f"""Numbered Narration:
{numbered_narration}

Code Blocks (in order they appear):
{json.dumps(blocks_summary, indent=2)}

Map each narration sentence to the code block it explains. Return JSON."""
            }
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    
    mapping_data = json.loads(mapping_response.choices[0].message.content)
    mappings = mapping_data.get("mappings", [])
    
    # Sort mappings by sentence number to maintain narration order
    mappings.sort(key=lambda x: x.get("sentence_number", 999))
    
    # Convert mappings to timeline events with timestamps
    timeline_events = []
    for mapping in mappings:
        block_idx = mapping.get("code_block_index")
        if block_idx >= len(code_blocks):
            continue
            
        code_block = code_blocks[block_idx]
        sentence_num = mapping.get("sentence_number", 1)
        phrase = mapping.get("phrase_in_sentence", "").lower()
        
        # Filter by confidence - only use high confidence mappings
        confidence = mapping.get("confidence", 0.8)
        if confidence < 0.7:
            continue
        
        # Filter out trivial code blocks unless explicitly mentioned
        code_preview = code_block.get('code', '').lower()
        # Skip if it's just System.out.print statements and not explicitly mentioned
        if 'system.out.print' in code_preview:
            # Only include if phrase explicitly mentions print/System.out
            if 'system.out' not in phrase and 'print' not in phrase and 'output' not in phrase:
                continue
        # Skip very small blocks (single lines) unless explicitly mentioned
        if code_block['end_line'] - code_block['start_line'] == 0 and len(phrase) < 15:
            continue
        
        # Find the sentence in narration
        if sentence_num <= len(narration_sentences):
            target_sentence = narration_sentences[sentence_num - 1].lower()
        else:
            continue
        
        # Find timestamps for this sentence using Whisper transcription
        # Build transcription text from Whisper words
        transcription_text = " ".join([w.get("word", "").strip() for w in narration_words]).lower()
        
        # Find where this sentence appears in transcription
        # Use key words from sentence (first 3-4 significant words)
        sentence_keywords = [w for w in target_sentence.split() if len(w) > 3][:4]
        if not sentence_keywords:
            continue
            
        # Find the position of these keywords in transcription
        keyword_positions = []
        transcription_lower = transcription_text.lower()
        for keyword in sentence_keywords:
            pos = transcription_lower.find(keyword.lower())
            if pos >= 0:
                keyword_positions.append(pos)
        
        if not keyword_positions:
            continue
        
        # Find the word index where sentence starts (first keyword position)
        first_keyword_pos = min(keyword_positions)
        words_before = len(transcription_text[:first_keyword_pos].split())
        
        # Get timestamps for words around this position
        start_time = None
        end_time = None
        words_checked = 0
        
        for word_info in narration_words:
            words_checked += 1
            # Check if we're in the range of the sentence (within 10 words of keyword)
            if abs(words_checked - words_before) <= 10:
                if start_time is None:
                    start_time = word_info.get("start", 0)
                end_time = word_info.get("end", 0)
        
        # If we found the sentence, create timeline event
        if start_time is not None:
            # Use actual sentence duration from Whisper, or estimate
            if end_time is None or end_time <= start_time:
                # Estimate: count words in sentence, assume 0.5s per word
                word_count = len(target_sentence.split())
                end_time = start_time + (word_count * 0.5)
            
            # Add minimum delay - don't highlight in first 5 seconds (title/intro)
            min_start_time = 5.0
            if start_time < min_start_time:
                start_time = min_start_time
            
            # Ensure minimum duration
            if end_time - start_time < 1.0:
                end_time = start_time + 2.0
            
            timeline_events.append({
                'code_block': code_block,
                'start_time': start_time,
                'end_time': end_time,
                'confidence': mapping.get("confidence", 0.8),
                'sentence': sentence_num
            })
    
    print(f"✅ Created {len(timeline_events)} timeline events\n")
    return timeline_events


def generate_timeline_animations(code_content, timeline_events, audio_duration, key_concepts, key_concepts_start_time):
    """
    Generate Manim code with timeline-based animations.
    
    Creates a timeline where each event happens at a specific time,
    rather than sequential waits.
    
    Args:
        code_content: Full source code
        timeline_events: List of events from map_narration_to_code()
        audio_duration: Total audio duration
        key_concepts: List of key concept strings
        key_concepts_start_time: When key concepts slide should appear
        
    Returns:
        Manim code as string
    """
    # Escape code
    escaped_code = code_content.replace('\\', '\\\\').replace('"', '\\"')
    
    # Calculate timing
    fixed_overhead = 1.2 + 1.5 + 0.5  # Initial wait + title + FadeIn
    
    if audio_duration and key_concepts_start_time:
        code_scroll_time = key_concepts_start_time - fixed_overhead - 0.7
        code_scroll_time = max(code_scroll_time, 5.0)
        concepts_slide_time = audio_duration - key_concepts_start_time
        concepts_slide_time = max(concepts_slide_time, 3.0)
    elif audio_duration:
        code_scroll_time = audio_duration - fixed_overhead - 1.0
        code_scroll_time = max(code_scroll_time, 5.0)
        concepts_slide_time = 0
    else:
        code_scroll_time = len(code_content.split('\n')) * 0.5
        concepts_slide_time = 0
    
    # Build concepts slide
    if key_concepts and concepts_slide_time > 0:
        concepts_list = "[" + ", ".join(['"' + c.replace('"', '\\"') + '"' for c in key_concepts]) + "]"
        concepts_display_time = concepts_slide_time - 1.8
        concepts_display_time = max(concepts_display_time, 2.0)
        
        concepts_slide_code = f"""
        # Key Concepts slide
        concepts_title = Text("Key Concepts", font_size=38, font="Helvetica", weight=BOLD, color=BLUE)
        concepts_title.to_edge(UP, buff=0.4)
        concept_items = VGroup(*[Text(f"• {{concept}}", font_size=24, font="Helvetica", color=WHITE) for concept in {concepts_list}])
        concept_items.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        concept_items.next_to(concepts_title, DOWN, buff=0.6)
        concept_items.to_edge(LEFT, buff=0.8)
        self.play(Write(concepts_title), run_time=0.5)
        self.wait(0.3)
        self.play(Write(concept_items), run_time=1.0)
        self.wait({concepts_display_time})
"""
    else:
        concepts_slide_code = "        self.wait(1)\n"
    
    # Build highlight rectangles for each code block
    highlight_creation = ""
    highlight_list = []
    line_height = 0.5
    
    for event_idx, event in enumerate(timeline_events):
        block = event['code_block']
        start_line = block['start_line']
        end_line = block['end_line']
        block_height = (end_line - start_line + 1) * line_height
        
        highlight_creation += f"""        # Highlight for {block['type']} {block.get('name', '')} (lines {start_line}-{end_line})
        highlight_{event_idx} = Rectangle(
            width=12,
            height={block_height:.3f},
            fill_opacity=0.0,  # Start invisible, will animate in
            fill_color=YELLOW,
            stroke_width=2,
            stroke_color=YELLOW,
            stroke_opacity=0.0  # Border will animate
        )
"""
        highlight_list.append(f"highlight_{event_idx}")
    
    # Build highlight positioning
    # First, we need to account for empty lines at the start of code
    code_lines = code_content.strip().split('\n')
    # Find first non-empty line to adjust line numbering
    first_code_line_idx = 0
    for i, line in enumerate(code_lines):
        if line.strip():
            first_code_line_idx = i
            break
    
    highlight_positioning = ""
    for event_idx, event in enumerate(timeline_events):
        block = event['code_block']
        start_line = block['start_line']
        end_line = block['end_line']
        # Adjust for 0-indexed vs 1-indexed and empty lines
        # start_line is 1-indexed from parser, but code display might have leading empty lines
        adjusted_start = start_line - 1 - first_code_line_idx
        adjusted_end = end_line - 1 - first_code_line_idx
        center_line = (adjusted_start + adjusted_end) / 2
        
        highlight_positioning += f"""            # Position highlight for {block['type']} (lines {start_line}-{end_line})
            block_{event_idx}_center_y = full_code.get_top()[1] - ({center_line} * {line_height:.3f})
            highlight_{event_idx}.move_to([full_code.get_center()[0], block_{event_idx}_center_y, 0])
            highlight_{event_idx}.stretch_to_fit_width(full_code.width + 0.2)
            highlight_{event_idx}.set_z_index(-1)
            highlight_{event_idx}.set_opacity(0)  # Start invisible
            highlight_{event_idx}.set_stroke_opacity(0)  # Border starts invisible
"""
    
    # Build timeline-based animations
    # Sort events by start_time
    timeline_events.sort(key=lambda x: x['start_time'])
    
    animation_timeline = ""
    # Start timing from when code appears (fixed_overhead)
    # First event should wait until its narration actually starts
    current_time = fixed_overhead
    previous_highlight_idx = None  # Track which highlight was active
    
    for event_idx, event in enumerate(timeline_events):
        start_time = event['start_time']
        end_time = event['end_time']
        block = event['code_block']
        
        # Wait until event start time
        wait_time = start_time - current_time
        # For first event, ensure we wait even if timing matches exactly
        if event_idx == 0 and wait_time <= 0:
            wait_time = 0.5  # Small delay to ensure audio has started
        
        if wait_time > 0:
            animation_timeline += f"""            # Wait until {block['type']} narration starts (at {start_time:.2f}s)
            self.wait({wait_time:.2f})
"""
        current_time = start_time
        
        # Fade out previous highlight BEFORE starting new one
        if previous_highlight_idx is not None:
            animation_timeline += f"""            # Fade out previous highlight
            self.play(highlight_{previous_highlight_idx}.animate.set_opacity(0), run_time=0.2)
"""
        
        # Calculate scroll progress
        scroll_progress = (start_time - fixed_overhead) / code_scroll_time if code_scroll_time > 0 else 0
        scroll_progress = min(max(scroll_progress, 0), 1)
        
        # Highlight duration
        highlight_duration = end_time - start_time
        
        # Scroll and highlight NEW block with ANIMATION
        # Use fade-in and pulse effect for better visual appeal
        animation_timeline += f"""            # Highlight {block['type']} {block.get('name', '')} (lines {block['start_line']}-{block['end_line']})
            seg_target_y = start_center_y + (scroll_distance * {scroll_progress:.3f})
            # Scroll to position
            self.play(
                full_code.animate.move_to([code_center_x, seg_target_y, 0]),
                run_time=0.3
            )
            # Animated highlight: fade in with border and pulse effect
            self.play(
                highlight_{event_idx}.animate.set_opacity(0.35).set_stroke_opacity(0.8),
                run_time=0.4
            )
            # Subtle pulse animation during highlight (breathing effect)
            self.play(
                highlight_{event_idx}.animate.set_opacity(0.25).set_stroke_opacity(0.6),
                run_time=0.3
            )
            self.play(
                highlight_{event_idx}.animate.set_opacity(0.35).set_stroke_opacity(0.8),
                run_time=0.3
            )
            # Wait for remaining narration time (subtract animation time: 0.4 + 0.3 + 0.3 = 1.0)
            remaining_time = {highlight_duration:.2f} - 1.0
            if remaining_time > 0:
                self.wait(remaining_time)
"""
        # Don't fade out here - fade out when next block starts (or at end)
        previous_highlight_idx = event_idx
        current_time = end_time
    
    # Fade out last highlight if there are more events or at end
    if previous_highlight_idx is not None:
        animation_timeline += f"""            # Fade out last highlight
            self.play(highlight_{previous_highlight_idx}.animate.set_opacity(0), run_time=0.2)
"""
    
    # Generate full Manim code
    code = f"""from manim import *

class CodeExplanationScene(Scene):
    def construct(self):
        # Initial wait
        self.wait(1.2)
        
        # Title
        title = Text("Code Explanation", font_size=38, font="Helvetica", weight=BOLD, color=BLUE)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.5)
        self.wait(1)
        
        # Create full code text
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
        
        full_code.to_edge(LEFT, buff=0.5)
        
        # Create highlight rectangles
{highlight_creation}
        
        # Available screen height
        available_screen_height = 5.5
        code_height = full_code.height
        
        if code_height > available_screen_height:
            code_center_x = full_code.get_left()[0] + full_code.width/2
            start_center_y = 2.5 - (code_height / 2)
            full_code.move_to([code_center_x, start_center_y, 0])
            end_center_y = -3.5 + (code_height / 2)
            scroll_distance = end_center_y - start_center_y
            
            # Position highlights
{highlight_positioning}
            
            # Add highlights to scene
            all_highlights = VGroup({', '.join(highlight_list) if highlight_list else ''})
            self.add(all_highlights)
            
            # Show code
            self.play(FadeIn(full_code), run_time=0.5)
            
            # Timeline-based animations
{animation_timeline}
        else:
            full_code.next_to(title, DOWN, buff=0.3)
            self.play(FadeIn(full_code), run_time=0.5)
            self.wait({code_scroll_time})
        
        # Fade out code
        self.play(FadeOut(full_code), FadeOut(title), run_time=0.5)
        self.wait(0.2)
        
        # Key Concepts slide
{concepts_slide_code}
"""
    
    return code


# ============================================================
# FUNCTION 3: CODE TO VIDEO (Code Explanations)
# ============================================================

def generate_code_display_code(code_content, audio_duration=None, narration_segments=None, key_concepts=None, key_concepts_start_time=None):
    """
    Generate Manim code for scrolling code display - NO SLIDES, continuous scroll
    Code starts below title and scrolls UP (code moves UP) to reveal content below
    Adds final slide with key concepts if provided, synchronized with audio
    """
    # Use original code (no cleaning - user won't pass comments)
    lines = code_content.strip().split('\n')
    
    # Escape code for Python string
    escaped_code = code_content.replace('\\', '\\\\').replace('"', '\\"')
    
    # Calculate timing for code scrolling
    # Fixed overhead: initial wait (1.2s), title (1.5s), FadeIn (0.5s)
    fixed_overhead = 1.2 + 1.5 + 0.5
    
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
    
    # Create full code text
    code = f"""from manim import *

class CodeExplanationScene(Scene):
    def construct(self):
        # Initial wait
        self.wait(1.2)
        
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
        self.wait(1)
        
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
            
            # Scroll duration matches code explanation audio (before key concepts)
            scroll_duration = {code_scroll_time}
            
            # Show code and scroll UP (code moves UP, revealing content below)
            self.play(FadeIn(full_code), run_time=0.5)
            # Scroll UP: code moves from start (top visible) to end (bottom visible)
            self.play(full_code.animate.move_to([code_center_x, end_center_y, 0]), run_time=scroll_duration)
        else:
            # Code fits on screen - position it below title
            full_code.next_to(title, DOWN, buff=0.3)
            self.play(FadeIn(full_code), run_time=0.5)
            self.wait({code_scroll_time})
        
        # Fade out code (transition to key concepts - synchronized with audio)
        self.play(FadeOut(full_code), FadeOut(title), run_time=0.5)
        self.wait(0.2)  # Quick transition
        
        # Final slide: Key Concepts (if provided) - appears exactly when audio explains them
{concepts_slide_code}
"""
    
    return code


def code_to_video(code_content: str, output_name: str = "output", audio_language: str = "english"):
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
        
        # Step 0: NEW APPROACH - Parse code into blocks FIRST
        print("📦 Parsing code into semantic blocks...")
        language = "python"  # Default
        if "public class" in code_content or "public static" in code_content:
            language = "java"
        elif "function" in code_content and "{" in code_content:
            language = "javascript"
        
        code_blocks = parse_code_to_blocks(code_content, language=language)
        print(f"✅ Found {len(code_blocks)} code blocks\n")
        
        # Filter out blocks that are too large (they cover too much)
        # Large blocks (class/main function/outer loops) are not useful for highlighting
        filtered_blocks = []
        for block in code_blocks:
            block_size = block['end_line'] - block['start_line']
            
            if block['type'] == 'class':
                # Skip class blocks - they're too large and not useful for highlighting
                # Only keep if it's very small (just declaration, <= 2 lines) AND it's the only block
                if block_size <= 2 and len(code_blocks) == 1:
                    filtered_blocks.append(block)
                # Otherwise skip - we'll highlight specific blocks instead
            elif block['type'] == 'function':
                # Skip function blocks - they're usually too large
                # Only keep if it's very small (<= 3 lines) AND it's the only block
                if block_size <= 3 and len(code_blocks) == 1:
                    filtered_blocks.append(block)
                # Otherwise skip - we'll highlight the loops/statements inside instead
            elif block['type'] in ['for_loop', 'while_loop']:
                # Skip large loops that contain other blocks
                # Check if this loop contains other blocks
                contains_other_blocks = False
                for other_block in code_blocks:
                    if other_block != block and other_block['type'] in ['for_loop', 'while_loop', 'if_statement']:
                        # Check if other block is inside this block
                        if (block['start_line'] < other_block['start_line'] and 
                            block['end_line'] > other_block['end_line']):
                            contains_other_blocks = True
                            break
                
                # Only keep if it's small (<= 3 lines) or doesn't contain other blocks
                if block_size <= 3 or not contains_other_blocks:
                    filtered_blocks.append(block)
                # Otherwise skip - we'll highlight the inner blocks instead
            else:
                # Keep all other blocks (if statements, etc.)
                filtered_blocks.append(block)
        
        code_blocks = filtered_blocks
        original_count = len([b for b in code_blocks if b['type'] in ['class', 'function']])
        if len(code_blocks) < len([b for b in parse_code_to_blocks(code_content, language=language) if b['type'] in ['class', 'function']]):
            print(f"📝 Filtered to {len(code_blocks)} blocks (removed large class/function blocks)\n")
        
        # Step 1: NEW APPROACH - Generate narration PER BLOCK
        print("🎙️ Creating block-by-block narration...")
        print("   This ensures perfect synchronization between narration and code highlighting")
        
        block_narrations = []
        block_audios = []
        timeline_events = []
        cumulative_time = 0.0
        
        # Add intro delay (title appears, then code)
        # Make sure first highlight doesn't appear until its narration actually starts
        intro_delay = 2.7  # 1.2s initial wait + 0.5s title + 1s wait
        
        for block_idx, code_block in enumerate(code_blocks):
            block_type = code_block.get('type', 'code_block')
            block_name = code_block.get('name', 'unnamed')
            block_code = code_block.get('code', '')
            start_line = code_block.get('start_line', 0)
            end_line = code_block.get('end_line', 0)
            
            print(f"   Block {block_idx + 1}/{len(code_blocks)}: {block_type} (lines {start_line}-{end_line})")
            
            # Generate narration for this specific block
            # CRITICAL: NO OVERVIEWS - Each block explains ONLY itself
            if block_idx == 0:
                # First block: explain ONLY this block, STRICTLY no overview
                block_prompt = f"""Explain ONLY this specific code block. 

CRITICAL RULES:
- Do NOT mention what the entire code does
- Do NOT give an overview or introduction
- Do NOT say "this code" or "the program"
- Focus ONLY on what THIS specific block does
- Start directly with what this block does

Code block:
```
{block_code}
```

Provide a concise explanation of ONLY this block (under 500 characters). Start directly, no introduction."""
            else:
                # Subsequent blocks: add transition
                block_prompt = f"""Now explain this next code block. Use a natural transition like "Next, we have..." or "Moving on to...".

Code block:
```
{block_code}
```

Provide a concise explanation (under 500 characters)."""
            
            narration_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert programming educator. Explain ONLY the specific code block provided. CRITICAL: Do NOT give overviews, introductions, or mention the entire code. Focus ONLY on what this specific block does. Start directly. Keep explanations under 500 characters."
                    },
                    {
                        "role": "user",
                        "content": block_prompt
                    }
                ],
                temperature=0.7,
                max_tokens=250
            )
            
            block_narration = narration_response.choices[0].message.content.strip()
            
            # Truncate if too long
            max_block_length = 500
            if len(block_narration) > max_block_length:
                block_narration = block_narration[:max_block_length]
                last_period = block_narration.rfind('.')
                if last_period > max_block_length * 0.8:
                    block_narration = block_narration[:last_period + 1]
            
            # Remove any overview-like phrases from first block
            if block_idx == 0:
                # Remove common overview phrases
                overview_phrases = [
                    "this code",
                    "the program",
                    "the entire code",
                    "this program",
                    "overall",
                    "in general",
                    "the code above",
                    "the code below"
                ]
                narration_lower = block_narration.lower()
                for phrase in overview_phrases:
                    if phrase in narration_lower:
                        # Try to remove sentences containing overview phrases
                        sentences = block_narration.split('.')
                        filtered_sentences = [s for s in sentences if phrase not in s.lower()]
                        if filtered_sentences:
                            block_narration = '. '.join(filtered_sentences).strip()
                            if not block_narration.endswith('.'):
                                block_narration += '.'
                            break
            
            block_narrations.append(block_narration)
            print(f"      Generated {len(block_narration)} characters")
            
            # Generate audio for this block
            block_audio_file = f"{output_name}_block_{block_idx}_audio.aiff"
            generate_audio_for_language(block_narration, audio_language, block_audio_file, client)
            
            # Get duration
            block_duration = get_audio_duration(block_audio_file)
            if not block_duration:
                block_duration = len(block_narration) * 0.06  # Estimate: ~60ms per character
            
            block_audios.append(block_audio_file)
            
            # Create timeline event - PERFECT SYNC!
            timeline_events.append({
                'code_block': code_block,
                'start_time': intro_delay + cumulative_time,
                'end_time': intro_delay + cumulative_time + block_duration,
                'confidence': 1.0,
                'sentence': block_idx + 1
            })
            
            cumulative_time += block_duration
            print(f"      Audio duration: {block_duration:.2f}s (starts at {intro_delay + cumulative_time - block_duration:.2f}s)\n")
        
        # Concatenate all block audios
        print("🔗 Concatenating block audio files...")
        audio_file = f"{output_name}_audio.aiff"
        concatenate_audio_files(block_audios, audio_file)
        print(f"✅ Combined audio: {audio_file}\n")
        
        # Get total audio duration
        audio_duration = get_audio_duration(audio_file)
        if not audio_duration:
            audio_duration = cumulative_time
        
        # Combine all narrations for key concepts extraction
        english_narration = " ".join(block_narrations)
        print(f"✅ Total narration: {len(english_narration)} characters, {audio_duration:.2f} seconds\n")
        
        # Step 1.5: Extract key concepts from the code
        print("🔑 Extracting key concepts...")
        concepts_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Extract 4-6 key concepts from the code explanation. Return as a JSON array of concept strings."
                },
                {
                    "role": "user",
                    "content": f"Extract key concepts from this code explanation:\n\n{english_narration}\n\nReturn JSON: {{\"concepts\": [\"concept1\", \"concept2\", ...]}}"
                }
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        concepts_data = json.loads(concepts_response.choices[0].message.content)
        key_concepts = concepts_data.get("concepts", [])
        print(f"✅ Extracted {len(key_concepts)} key concepts\n")
        
        # Step 1.6: Generate detailed key concepts explanation
        # Store main narration length BEFORE adding key concepts (for timing calculation)
        main_narration_length = len(english_narration)
        key_concepts_start_time = None
        
        if key_concepts:
            print("📝 Generating detailed key concepts explanation...")
            # Generate a detailed explanation of each concept (not just listing)
            concepts_prompt = f"""Based on this code explanation, provide a brief but detailed explanation of these key concepts: {', '.join(key_concepts)}.

For each concept, explain what it is and how it's used in this code. Keep it concise but educational (under 500 characters total). Format as a natural paragraph, not a list."""
            
            concepts_explanation_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert programming educator. Explain key concepts clearly and concisely."
                    },
                    {
                        "role": "user",
                        "content": f"{concepts_prompt}\n\nCode explanation context:\n{english_narration[:1000]}"
                    }
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            concepts_explanation = "\n\n" + concepts_explanation_response.choices[0].message.content.strip()
            
            # Check if adding concepts would exceed TTS limit
            max_tts_length = 4000
            if len(english_narration) + len(concepts_explanation) <= max_tts_length:
                english_narration += concepts_explanation
                print(f"✅ Key concepts explanation added ({len(english_narration)} total characters)\n")
            else:
                # Truncate main narration slightly to make room for concepts
                available_space = max_tts_length - len(concepts_explanation) - 50  # 50 char buffer
                if available_space > 2000:  # Only if we have reasonable space
                    english_narration = english_narration[:available_space]
                    last_period = english_narration.rfind('.')
                    if last_period > available_space * 0.8:
                        english_narration = english_narration[:last_period + 1]
                    english_narration += concepts_explanation
                    main_narration_length = len(english_narration) - len(concepts_explanation)
                    print(f"✅ Key concepts added (truncated main narration to fit, {len(english_narration)} total characters)\n")
                else:
                    print(f"⚠️  Not enough space for key concepts in audio (keeping original narration)\n")
                    key_concepts = None  # Don't show concepts if we can't fit them
        
        # Step 2: Convert to code-mixed language if needed
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
        
        # Step 3: Generate audio
        audio_file = f"{output_name}_audio.aiff"
        generate_audio_for_language(narration_text, audio_language, audio_file, client)
        
        # Step 4: Get audio duration
        print("⏱️  Measuring audio duration...")
        audio_duration = get_audio_duration(audio_file)
        if audio_duration:
            print(f"✅ Audio duration: {audio_duration:.2f} seconds\n")
        else:
            print("⚠️  Using fallback timing\n")
        
        # Step 4.5: Calculate when key concepts start in audio (for synchronization)
        if key_concepts and audio_duration and main_narration_length > 0:
            # Calculate proportion: main narration length / total narration length
            total_narration_length = len(english_narration)
            if total_narration_length > 0:
                # Estimate when key concepts start based on text proportion
                # No buffer - transition should happen exactly when concepts start
                concepts_proportion = main_narration_length / total_narration_length
                key_concepts_start_time = audio_duration * concepts_proportion
                print(f"⏱️  Key concepts start at ~{key_concepts_start_time:.2f} seconds\n")
            else:
                key_concepts_start_time = None
        else:
            key_concepts_start_time = None
        
        # Step 6: Generate Manim code with PERFECT SYNC highlighting
        # timeline_events already created with perfect synchronization!
        if timeline_events:
            print("🎨 Generating Manim code with block-by-block highlighting...")
            print(f"   {len(timeline_events)} blocks will be highlighted in perfect sync\n")
            manim_code = generate_timeline_animations(
                code_content, 
                timeline_events, 
                audio_duration, 
                key_concepts, 
                key_concepts_start_time
            )
        else:
            print("🎨 Generating Manim code (simple scroll, no highlighting)...")
            # Fallback to simple scrolling without highlighting
            manim_code = generate_code_display_code(
                code_content, 
                audio_duration, 
                None,  # No narration_segments
                key_concepts, 
                key_concepts_start_time
            )
        print("✅ Manim code generated\n")
        
        # Clean up temporary block audio files
        for block_audio in block_audios:
            if os.path.exists(block_audio):
                try:
                    os.remove(block_audio)
                except:
                    pass
        
        # Step 7: Render video
        print("🎬 Rendering video...")
        scene_file = f".temp_{output_name}.py"
        
        with open(scene_file, 'w') as f:
            f.write(manim_code)
        
        cmd = [
            "venv/bin/python", "-m", "manim",
            "-pql", scene_file, "CodeExplanationScene"
        ]
        
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        video_path = f"media/videos/{scene_file.replace('.py', '')}/480p15/CodeExplanationScene.mp4"
        
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            return None
        
        print("✅ Video rendered\n")
        
        # Step 8: Combine video + audio
        print("🎵 Combining video + audio...")
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
    
    
    # Example 2: ANIMATION TO VIDEO
    # Use for: Algorithms, visualizations, demonstrations
    
    result = animation_to_video(
        prompt="Bubble Sort algorithm - visualize array sorting with comparisons and swaps",
        output_name="bubble_sort_animation"
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
    
    print("\n🎉 Done! Check your video!")