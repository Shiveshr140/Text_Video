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
# FUNCTION 3: CODE TO VIDEO (Code Explanations)
# ============================================================

def find_block_end(lines, start_line):
    """Helper to find end of code block (simple brace matching)"""
    brace_count = 0
    for i in range(start_line - 1, len(lines)):
        line = lines[i]
        brace_count += line.count('{') - line.count('}')
        if brace_count == 0 and i > start_line - 1:
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
        elif re.match(r'^\s*for\s*\(', line):
            end_line = find_block_end(lines, i)
            blocks.append({
                'type': 'for_loop',
                'name': None,
                'start_line': i,
                'end_line': end_line,
                'code': '\n'.join(lines[i-1:end_line])
            })
        elif re.match(r'^\s*if\s*\(', line):
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


def generate_timeline_animations(code_content, timeline_events, audio_duration, key_concepts, key_concepts_start_time):
    """
    Generate Manim code with timeline-based animations using REAL Whisper timestamps.
    """
    print(f"🔍 DEBUG: generate_timeline_animations called")
    print(f"   - code_content length: {len(code_content)} characters")
    print(f"   - timeline_events: {len(timeline_events) if timeline_events else 0} events")
    print(f"   - audio_duration: {audio_duration}")
    
    if not timeline_events or len(timeline_events) == 0:
        print(f"   ⚠️  WARNING: No timeline events! Highlights will not be created!")
        return None
    
    escaped_code = code_content.replace('\\', '\\\\').replace('"', '\\"')
    
    # ============================================================
    # HARDCODED VALUES EXPLANATION:
    # ============================================================
    # These values match the EXACT Manim scene timeline:
    # 
    # Line 1186: self.wait(1.2)                    → 0.0s to 1.2s
    # Line 1189: self.play(Write(title), run_time=0.5)  → 1.2s to 1.7s
    # Line 1190: self.wait(1)                     → 1.7s to 2.7s
    # Line 1221: self.play(FadeIn(full_code), run_time=0.5)  → 2.7s to 3.2s
    #
    # So code becomes VISIBLE at 3.2s (when FadeIn completes)
    # This is when we can START highlighting and scrolling
    # ============================================================
    fixed_overhead = 1.2 + 0.5 + 1.0 + 0.5  # = 3.2s (when code appears and becomes interactive)
    
    print(f"\n{'='*60}")
    print("🔍 DEBUG: MANIM CODE GENERATION")
    print(f"{'='*60}")
    print(f"fixed_overhead = {fixed_overhead:.2f}s")
    print(f"  Breakdown:")
    print(f"    - 1.2s: Initial wait (self.wait(1.2))")
    print(f"    - 0.5s: Title animation (self.play(Write(title), run_time=0.5))")
    print(f"    - 1.0s: Wait after title (self.wait(1))")
    print(f"    - 0.5s: Code fade-in (self.play(FadeIn(full_code), run_time=0.5))")
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
        # Subtract 0.7s for transition (FadeOut code + small wait)
        code_scroll_time = key_concepts_start_time - fixed_overhead - 0.7
        code_scroll_time = max(code_scroll_time, 5.0)  # Minimum 5 seconds
        concepts_slide_time = audio_duration - key_concepts_start_time
        concepts_slide_time = max(concepts_slide_time, 3.0)
        
        print(f"code_scroll_time = {code_scroll_time:.2f}s (TOTAL scroll window duration)")
        print(f"  Calculation: {key_concepts_start_time:.2f}s (key concepts start) - {fixed_overhead:.2f}s (code appears) - 0.7s (transition) = {code_scroll_time:.2f}s")
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
        # Fallback: estimate based on code length
        code_scroll_time = len(code_content.split('\n')) * 0.5
        concepts_slide_time = 0
        
        print(f"code_scroll_time = {code_scroll_time:.2f}s (ESTIMATED from code length)")
    print()
    
    if key_concepts and concepts_slide_time > 0:
        concepts_list = "[" + ", ".join(['"' + c.replace('"', '\\"') + '"' for c in key_concepts]) + "]"
        concepts_display_time = concepts_slide_time - 1.8
        concepts_display_time = max(concepts_display_time, 2.0)
        concepts_slide_code = f"""
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
    
    highlight_creation = ""
    highlight_list = []
    line_height = 0.5
    
    print(f"🔍 DEBUG: Creating highlights for {len(timeline_events)} timeline events")
    
    for event_idx, event in enumerate(timeline_events):
        block = event['code_block']
        start_line = block['start_line']
        end_line = block['end_line']
        block_height = (end_line - start_line + 1) * line_height
        
        # Determine color based on block type for better visual distinction
        block_type = block.get('type', 'code_block')
        if 'loop' in block_type:
            highlight_color = "#4A90E2"  # Blue for loops
        elif 'if' in block_type or 'else' in block_type:
            highlight_color = "#50C878"  # Green for conditionals
        elif 'function' in block_type or 'method' in block_type:
            highlight_color = "#FF6B6B"  # Red for functions
        elif 'class' in block_type:
            highlight_color = "#9B59B6"  # Purple for classes
        else:
            highlight_color = "#FFD700"  # Gold for other blocks
        
        highlight_creation += f"""        # Highlight rectangle with gradient effect
        highlight_{event_idx} = Rectangle(
            width=12,
            height={block_height:.3f},
            fill_opacity=0.0,
            fill_color="{highlight_color}",
            stroke_width=3,
            stroke_color="{highlight_color}",
            stroke_opacity=0.0
        )
        # Add glow effect (slightly larger, semi-transparent)
        highlight_glow_{event_idx} = Rectangle(
            width=12.4,
            height={block_height + 0.1:.3f},
            fill_opacity=0.0,
            fill_color="{highlight_color}",
            stroke_width=5,
            stroke_color="{highlight_color}",
            stroke_opacity=0.0
        )
"""
        highlight_list.append(f"highlight_{event_idx}")
        highlight_list.append(f"highlight_glow_{event_idx}")
        print(f"   ✅ Created highlight_{event_idx} and highlight_glow_{event_idx} for {block_type} (lines {start_line}-{end_line}, color: {highlight_color})")
    
    print(f"🔍 DEBUG: Total highlights created: {len(highlight_list)} ({len(timeline_events)} events × 2 per event)\n")
    
    code_lines = code_content.strip().split('\n')
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
        adjusted_start = start_line - 1 - first_code_line_idx
        adjusted_end = end_line - 1 - first_code_line_idx
        center_line = (adjusted_start + adjusted_end) / 2
        
        # Build line by line with 12-space indentation (matches if/else block level)
        indent = "            "  # 12 spaces
        highlight_positioning += f"{indent}block_{event_idx}_center_y = full_code.get_top()[1] - ({center_line} * {line_height:.3f})\n"
        highlight_positioning += f"{indent}highlight_{event_idx}.move_to([full_code.get_center()[0], block_{event_idx}_center_y, 0])\n"
        highlight_positioning += f"{indent}highlight_{event_idx}.stretch_to_fit_width(full_code.width + 0.3)\n"
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
    current_time = fixed_overhead
    previous_highlight_idx = None
    
    print(f"Starting animation timeline generation...")
    print(f"Initial current_time = {current_time:.2f}s (fixed_overhead)\n")
    
    for event_idx, event in enumerate(timeline_events):
        start_time = event['start_time']
        end_time = event['end_time']
        block = event['code_block']
        
        wait_time = start_time - current_time
        
        print(f"Event {event_idx + 1}: {block['type']} (lines {block['start_line']}-{block['end_line']})")
        print(f"  Timeline event start_time: {start_time:.2f}s")
        print(f"  Current Manim time: {current_time:.2f}s")
        print(f"  Calculated wait_time: {wait_time:.2f}s")
        
        # CRITICAL: For first event, ensure code scrolls to show it if needed
        # Calculate scroll progress first to see if we need initial scroll
        scroll_progress = (start_time - fixed_overhead) / code_scroll_time if code_scroll_time > 0 else 0
        scroll_progress = min(max(scroll_progress, 0), 1)
        
        if event_idx == 0:
            # For first block, if scroll_progress > 0, we need to scroll to it FIRST
            # This ensures the block is visible before highlighting starts
            if scroll_progress > 0 and code_scroll_time > 0:
                # Block is not at top - scroll to it FIRST before highlighting
                animation_timeline += f"""            # Scroll to show first block before highlighting
            if scroll_distance > 0:
                first_seg_target_y = start_center_y + (scroll_distance * {scroll_progress:.3f})
                self.play(full_code.animate.move_to([code_center_x, first_seg_target_y, 0]), run_time=0.4)
                current_time_first = {fixed_overhead:.2f} + 0.4
"""
                print(f"  🔄 First block needs scroll - scrolling to position first (progress: {scroll_progress:.3f})")
            
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
        
        # Calculate scroll progress: how far through the scroll window we should be
        # scroll_progress = 0.0 means code at top (start position)
        # scroll_progress = 1.0 means code at bottom (end position)
        # Formula: (current_time - when_code_appeared) / total_scroll_window
        # This tells us WHERE the code should be positioned when this highlight appears
        # Note: scroll_progress already calculated above for first event
        if event_idx > 0:
            scroll_progress = (start_time - fixed_overhead) / code_scroll_time if code_scroll_time > 0 else 0
            scroll_progress = min(max(scroll_progress, 0), 1)  # Clamp between 0 and 1
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
        
        # Generate enhanced animation with glow, animated border, and spotlight effect
        # Animation sequence: scroll (0.3s) + glow fade-in (0.2s) + border draw (0.3s) + fill fade-in (0.2s) + pulse (0.4s) = 1.4s or 1.1s
        animation_timeline += f"""            # Scroll only if needed (scroll_distance > 0)
            scroll_time = 0.3 if scroll_distance > 0 else 0.0
            if scroll_distance > 0:
                seg_target_y = start_center_y + (scroll_distance * {scroll_progress:.3f})
                # CRITICAL: Update highlight positions relative to new code position after scroll
                code_top_y = seg_target_y + (full_code.height / 2)
                block_{event_idx}_center_y = code_top_y - ({center_line} * {line_height:.3f})
                highlight_{event_idx}.move_to([code_center_x, block_{event_idx}_center_y, 0])
                highlight_glow_{event_idx}.move_to([code_center_x, block_{event_idx}_center_y, 0])
                self.play(full_code.animate.move_to([code_center_x, seg_target_y, 0]), run_time=0.3)
            
            # Enhanced highlight animation sequence
            # Step 1: Glow appears first (subtle background glow)
            self.play(
                highlight_glow_{event_idx}.animate.set_stroke_opacity(0.3),
                run_time=0.2
            )
            # Step 2: Animated border drawing effect (stroke appears)
            self.play(
                highlight_{event_idx}.animate.set_stroke_opacity(0.9),
                highlight_glow_{event_idx}.animate.set_stroke_opacity(0.5),
                run_time=0.3
            )
            # Step 3: Fill fades in (background highlight)
            self.play(
                highlight_{event_idx}.animate.set_opacity(0.25).set_stroke_opacity(0.95),
                highlight_glow_{event_idx}.animate.set_stroke_opacity(0.4),
                run_time=0.2
            )
            # Step 4: Subtle pulse animation (breathing effect)
            self.play(
                highlight_{event_idx}.animate.set_opacity(0.20).set_stroke_opacity(0.85),
                highlight_glow_{event_idx}.animate.set_stroke_opacity(0.3),
                run_time=0.2
            )
            self.play(
                highlight_{event_idx}.animate.set_opacity(0.28).set_stroke_opacity(0.95),
                highlight_glow_{event_idx}.animate.set_stroke_opacity(0.45),
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
    
    code = f"""from manim import *

class CodeExplanationScene(Scene):
    def construct(self):
        self.wait(1.2)
        title = Text("Code Explanation", font_size=38, font="Helvetica", weight=BOLD, color=BLUE)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.5)
        self.wait(1)
        
        full_code = Text(
            \"\"\"{escaped_code}\"\"\",
            font_size=22,
            font="Courier",
            color=WHITE,
            line_spacing=1.0
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
        
        # Step 0: Parse code into blocks FIRST
        print("📦 Parsing code into semantic blocks...")
        language = "python"
        if "public class" in code_content or "public static" in code_content:
            language = "java"
        elif "function" in code_content and "{" in code_content:
            language = "javascript"
        
        code_blocks = parse_code_to_blocks(code_content, language=language)
        print(f"✅ Found {len(code_blocks)} code blocks\n")
        
        # DEBUG: Print all detected blocks BEFORE filtering
        print(f"🔍 DEBUG: All detected blocks BEFORE filtering:")
        for i, block in enumerate(code_blocks, 1):
            block_size = block['end_line'] - block['start_line'] + 1
            print(f"   Block {i}: {block['type']} (lines {block['start_line']}-{block['end_line']}, size: {block_size} lines)")
        print()
        
        # Filter out large blocks
        # WHY: Parser finds ALL blocks (including entire functions/classes)
        # But for highlighting, we want SMALL, SPECIFIC blocks (loops, if statements)
        # Large blocks are too generic and don't help with precise synchronization
        filtered_blocks = []
        for block in code_blocks:
            # CRITICAL: +1 because both start_line and end_line are INCLUSIVE
            # Example: start_line=3, end_line=7 means lines 3,4,5,6,7 = 5 lines
            # Without +1: 7-3=4 (WRONG!) | With +1: 7-3+1=5 (CORRECT!)
            block_size = block['end_line'] - block['start_line'] + 1
            
            if block['type'] == 'class':
                # Skip entire classes - they're too large for highlighting
                # Only keep if it's tiny (just declaration, 2 lines) AND it's the only block
                if block_size <= 2 and len(code_blocks) == 1:
                    filtered_blocks.append(block)
            elif block['type'] == 'function':
                # Skip entire functions - they're usually too large
                # Only keep if it's small (3 lines) AND it's the only block
                if block_size <= 3 and len(code_blocks) == 1:
                    filtered_blocks.append(block)
            elif block['type'] in ['for_loop', 'while_loop']:
                # CRITICAL FIX: Check if this is an OUTER LOOP (not contained in any other LOOP)
                # We check against other LOOPS only, not functions/classes
                is_outer_loop = True
                for other_block in code_blocks:
                    if (other_block != block and 
                        other_block['type'] in ['for_loop', 'while_loop']):
                        # Check if this loop is INSIDE another loop
                        if (other_block['start_line'] < block['start_line'] and 
                            other_block['end_line'] >= block['end_line']):
                            is_outer_loop = False
                            break
                
                # CRITICAL: ALWAYS keep outer loops - they must be explained FIRST
                if is_outer_loop:
                    filtered_blocks.append(block)
                    print(f"   ✅ KEEPING OUTER LOOP: {block['type']} (lines {block['start_line']}-{block['end_line']}, size: {block_size} lines)")
                elif block_size <= 3:
                    filtered_blocks.append(block)
                    print(f"   ✅ Keeping small inner loop: {block['type']} (lines {block['start_line']}-{block['end_line']}, size: {block_size} lines)")
                else:
                    print(f"   ❌ Skipping large inner loop: {block['type']} (lines {block['start_line']}-{block['end_line']}, size: {block_size} lines)")
            else:
                # Keep all other blocks (if_statement, etc.) - they're usually small and specific
                filtered_blocks.append(block)
        
        code_blocks = filtered_blocks
        
        # DEBUG: Print blocks AFTER filtering
        print(f"🔍 DEBUG: Blocks AFTER filtering ({len(code_blocks)} blocks):")
        for i, block in enumerate(code_blocks, 1):
            block_size = block['end_line'] - block['start_line'] + 1
            print(f"   Block {i}: {block['type']} (lines {block['start_line']}-{block['end_line']}, size: {block_size} lines)")
        print()
        
        # CRITICAL FIX: Sort blocks by start_line AND prioritize outer blocks
        # Outer blocks (that don't have other blocks containing them) should come first
        def is_outer_block(block):
            """Check if this block is not contained within any other block"""
            for other_block in code_blocks:
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
        
        # Separate outer blocks from nested blocks
        outer_blocks = [b for b in code_blocks if is_outer_block(b)]
        nested_blocks = [b for b in code_blocks if not is_outer_block(b)]
        
        # Sort each group by start_line (outermost/earliest first)
        outer_blocks.sort(key=lambda x: x['start_line'])
        nested_blocks.sort(key=lambda x: x['start_line'])
        
        # Outer blocks first, then nested blocks
        code_blocks = outer_blocks + nested_blocks
        
        print(f"📊 Block order: {len(outer_blocks)} outer blocks, {len(nested_blocks)} nested blocks")
        print(f"🔍 DEBUG: Outer blocks ({len(outer_blocks)}):")
        for i, block in enumerate(outer_blocks, 1):
            print(f"   Outer {i}: {block['type']} (lines {block['start_line']}-{block['end_line']})")
        print(f"🔍 DEBUG: Nested blocks ({len(nested_blocks)}):")
        for i, block in enumerate(nested_blocks, 1):
            print(f"   Nested {i}: {block['type']} (lines {block['start_line']}-{block['end_line']})")
        print(f"🔍 DEBUG: Final block order:")
        for i, block in enumerate(code_blocks, 1):
            print(f"   Final {i}: {block['type']} (lines {block['start_line']}-{block['end_line']})")
        print()
        
        if len(code_blocks) == 0:
            print("⚠️  No code blocks found, using simple scrolling\n")
            manim_code = generate_code_display_code(code_content, audio_duration, None, key_concepts, key_concepts_start_time)
        else:
            # Step 1: Generate narration PER BLOCK
            print("🎙️ Creating block-by-block narration...")
            print("   This ensures perfect synchronization between narration and code highlighting")
            
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
            #         Video starts (black screen)
            #         
            #   1.2s ────────────────────────────────────────────────────
            #         self.wait(1.2) ends
            #         
            #   1.7s ────────────────────────────────────────────────────
            #         self.play(Write(title), run_time=0.5) ends
            #         Title is now visible
            #         
            #   2.7s ────────────────────────────────────────────────────
            #         self.wait(1) ends (wait after title)
            #         
            #   3.2s ────────────────────────────────────────────────────
            #         self.play(FadeIn(full_code), run_time=0.5) ends
            #         ⭐ CODE IS NOW VISIBLE ⭐
            #         This is intro_delay = 3.2s
            #
            # PROBLEM: Audio narration starts at 0.0s, but code isn't visible until 3.2s!
            # SOLUTION: We CANNOT highlight code before it appears, so we use intro_delay
            #           to ensure highlights only happen when code is visible.
            #
            # This MUST match fixed_overhead in generate_timeline_animations()
            # ============================================================
            intro_delay = 3.2
            
            print(f"\n{'='*60}")
            print("🔍 DEBUG: TIMELINE CONSTRUCTION")
            print(f"{'='*60}")
            print(f"intro_delay = {intro_delay}s (when code appears on screen)")
            print(f"Starting cumulative_time = {cumulative_time}s\n")
            
            for block_idx, code_block in enumerate(code_blocks):
                block_type = code_block.get('type', 'code_block')
                block_code = code_block.get('code', '')
                start_line = code_block.get('start_line', 0)
                end_line = code_block.get('end_line', 0)
                
                print(f"Block {block_idx + 1}/{len(code_blocks)}: {block_type} (lines {start_line}-{end_line})")
                print(f"   🔍 DEBUG: Block index: {block_idx}, Total blocks: {len(code_blocks)}")
                print(f"   🔍 DEBUG: Block code preview: {block_code[:100]}...")
                
                if block_idx == 0:
                    # Get block type for explicit mention in narration
                    block_type_name = block_type.replace('_', ' ').title()  # "for_loop" -> "For Loop"
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
                - Do NOT say "this code" or "the program" without specifying it's a {block_type_name}
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
                
                narration_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are an expert programming educator. Explain ONLY the specific code block provided. This is a {block_type.replace('_', ' ')}. CRITICAL: Start by explicitly mentioning the block type (e.g., 'This for loop...', 'The if statement...'). Do NOT give overviews, introductions, or mention the entire code. Focus ONLY on what this specific {block_type.replace('_', ' ')} does. Start directly with the block type. Keep explanations under 500 characters."
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
                max_block_length = 500
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
                    if block_idx < len(code_blocks) - 1:
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
                
                print(f"      📍 Timeline Event {block_idx + 1}:")
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
                
                cumulative_time += block_duration
            
            # Validate we have audio files before concatenation
            if not block_audios:
                print("❌ No valid audio files generated! Cannot create video.")
                return None
            
            # Concatenate all block audio files with silence at start
            # CRITICAL: Add intro_delay seconds of silence so narration starts when code appears
            print(f"\n🔗 Concatenating {len(block_audios)} audio files with {intro_delay:.2f}s silence at start...")
            print(f"   This ensures narration starts exactly when code becomes visible (at {intro_delay:.2f}s)")
            audio_file = f"{output_name}_audio.aiff"
            
            if not concatenate_audio_files(block_audios, audio_file, silence_duration=intro_delay):
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
            # CRITICAL: Key concepts MUST start AFTER code narration ends
            key_concepts_start_time = None
            if key_concepts and audio_duration:
                # Calculate when code narration ends
                code_narration_end_time = max([e['end_time'] for e in timeline_events]) if timeline_events else None
                
                if code_narration_end_time:
                    # Key concepts audio starts immediately after code narration ends
                    # Add 0.7s transition time (FadeOut code + small wait)
                    key_concepts_start_time = code_narration_end_time + 0.7
                    print(f"✅ Key concepts start at {key_concepts_start_time:.2f}s")
                    print(f"   (Code narration ends at {code_narration_end_time:.2f}s + 0.7s transition)\n")
                else:
                    print(f"⚠️  Could not determine code narration end time\n")
            
            # Step 6: Generate Manim code with timeline-based animations
            print("🎨 Generating Manim code with Whisper timestamp synchronization...")
            manim_code = generate_timeline_animations(code_content, timeline_events, audio_duration, key_concepts, key_concepts_start_time)
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
        
        final_output = f"{output_name}_final.mp4"
        
        # Build command - use longer duration and loop audio if needed
        combine_cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_file,
        ]
        
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