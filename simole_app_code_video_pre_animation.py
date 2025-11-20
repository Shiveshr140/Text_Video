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
        concepts_list = "[" + ", ".join([f'"{c.replace('"', '\\"')}"' for c in key_concepts]) + "]"
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
        
        # Step 1: Generate narration explaining the code
        print("🎙️ Creating code explanation narration...")
        print("   Note: TTS limit is 4096 characters (we use 4000 for safety)")
        
        # Estimate code size to adjust prompt
        code_lines = len(code_content.split('\n'))
        if code_lines > 50:
            detail_level = "concisely but comprehensively"
        else:
            detail_level = "in detail"
        
        narration_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert programming educator. Explain code {detail_level}, covering all important parts. Keep explanation under 3500 characters to fit TTS limits."
                },
                {
                    "role": "user",
                    "content": f"Explain this code {detail_level}, covering what it does, how it works, and key concepts. Keep it concise but comprehensive (under 3500 characters):\n\n```\n{code_content}\n```"
                }
            ],
            temperature=0.7
        )
        
        english_narration = narration_response.choices[0].message.content.strip()
        
        # Truncate if too long (TTS limit: 4096 chars, we use 4000 for safety)
        max_tts_length = 4000
        original_length = len(english_narration)
        if original_length > max_tts_length:
            print(f"⚠️  Narration too long ({original_length} chars), truncating to {max_tts_length}...")
            english_narration = english_narration[:max_tts_length]
            last_period = english_narration.rfind('.')
            if last_period > max_tts_length * 0.8:
                english_narration = english_narration[:last_period + 1]
            print(f"   Final length: {len(english_narration)} characters")
        
        print(f"✅ Narration created ({len(english_narration)} characters)\n")
        
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
        
        # Step 5: Calculate number of sections
        lines = code_content.strip().split('\n')
        section_size = 18
        num_sections = (len(lines) + section_size - 1) // section_size  # Ceiling division
        
        # Step 6: Generate Manim code to display code
        print("🎨 Generating Manim code for code display...")
        manim_code = generate_code_display_code(code_content, audio_duration, num_sections, key_concepts, key_concepts_start_time)
        print("✅ Manim code generated\n")
        
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