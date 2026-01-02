

import os
import sys
import subprocess
import re
import openai
import anthropic  # For Claude API
# Removed: from pipeline import AudioGenerator (unused)
import json
import wave
import contextlib
import textwrap
import shutil  # For cleanup of test directories



# ============================================================
# WHISPER TIMESTAMP FUNCTION
# ============================================================

# def transcribe_audio_with_timestamps(audio_file):

#     try:
#         import whisper
        
#         print("🎤 Transcribing audio with Whisper (word-level timestamps)...")
        
#         # Load Whisper model (base model is fast and accurate)
#         model = whisper.load_model("base")
        
#         # Transcribe with word timestamps
#         result = model.transcribe(
#             audio_file,
#             word_timestamps=True,
#             language="en"  
#         )
        
#         # Extract word-level timestamps
#         words = []
#         for segment in result.get("segments", []):
#             for word_info in segment.get("words", []):
#                 words.append({
#                     "word": word_info.get("word", "").strip(),
#                     "start": word_info.get("start", 0),
#                     "end": word_info.get("end", 0)
#                 })
        
#         print(f"✅ Transcribed {len(words)} words with timestamps\n")
#         return words
        
#     except ImportError:
#         print("⚠️  Whisper not installed. Install: pip install openai-whisper")
#         print("   Falling back to text-based timing estimation\n")
#         return None
#     except Exception as e:
#         print(f"⚠️  Whisper transcription failed: {e}")
#         print("   Falling back to text-based timing estimation\n")
#         return None


# ============================================================
# HELPER: Get Audio Duration
# ============================================================

def get_audio_duration(audio_file):
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




def generate_srt_from_audio(audio_file, output_srt_file):

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
        # Handle long titles by wrapping into multiple lines
        title_lines = textwrap.wrap(title, width=50)
        # Create VGroup of text lines
        title_code = f"""        # Title (multi-line for long text - BEAUTIFIED)
        title_lines = VGroup()
"""
        for idx, line in enumerate(title_lines):
            escaped_line = line.replace('"', '\\"')
            title_code += f"""        title_line_{idx} = Text(
            "{escaped_line}",
            font_size=36,
            font="Inter",
            weight=SEMIBOLD,
            gradient=(BLUE, PURPLE)
        )
        # Scale down if too wide (max width: 13 units)
        if title_line_{idx}.width > 13:
            title_line_{idx}.scale_to_fit_width(13)
        title_lines.add(title_line_{idx})
"""
        title_code += """        title_lines.arrange(DOWN, buff=0.2)
        title_lines.to_edge(UP, buff=0.5)
        self.play(FadeIn(title_lines, shift=DOWN*0.3), run_time=0.8)
        self.wait(1.5)
        
        # Reference for sections to use
        title_ref = title_lines
        
"""
    else:
        # Short title - use single line (BEAUTIFIED - matches animated video)
        title_code = f"""        # Title
        title = Text(
            \"\"\"{title}\"\"\",
            font_size=48,
            font=\"Inter\",
            weight=SEMIBOLD,
            gradient=(BLUE, PURPLE)
        )
        # Scale down if too wide (max width: 13 units)
        if title.width > 13:
            title.scale_to_fit_width(13)
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title, shift=DOWN*0.3), run_time=0.8)
        self.wait(1.5)
        
        # Reference for sections to use
        title_ref = title
        
"""
    
    code = """from manim import *

class EducationalScene(Scene):
    def construct(self):
        # Beautiful dark background (matches animated video)
        self.camera.background_color = "#0F172A"
        
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
            font="Inter",
            weight=SEMIBOLD,
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
            font="Inter"
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
    
    try:
        print(f"\n{'='*60}")
        print(f"📄 TEXT TO VIDEO: {output_name}")
        print(f"{'='*60}\n")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ Set OPENAI_API_KEY environment variable")
            return None
        
        client = openai.OpenAI(api_key=api_key)
        

        is_query = False
        word_count = len(text_content.split())
        
        # Detect query patterns
        query_indicators = [
            text_content.strip().endswith('?'),
            word_count < 100,  
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
                        
                        CRITICAL RULES:
                        - Create 4-5 well-structured sections
                        - Each section should have a clear heading and 2-3 paragraphs
                        - Use SPECIFIC, CONCRETE examples (NOT generic "Component A, B, C")
                        - If comparing two things, explain REAL differences with ACTUAL details
                        - Example: Instead of "LRM has components A, B, C", say "LRM uses input features, weight parameters, and bias terms"
                        - Make it educational, informative, and engaging
                        - Use clear, simple language
                        - Include key concepts, benefits, examples, and practical applications
                        - Total length: 300-400 words
                        - AVOID PLACEHOLDERS: Never use "Component 1", "Component 2", "Part A", "Part B"
                        - BE SPECIFIC: Use actual technical terms and real-world examples
                        
                        Format:
                        Main Title
                        
                        Section 1 Heading
                        Content for section 1 with SPECIFIC details...
                        
                        Section 2 Heading
                        Content for section 2 with CONCRETE examples...
                        
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
            "-pqh", scene_file, "EducationalScene"  # High quality: 1080p60
        ]
        
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        video_path = f"media/videos/{scene_file.replace('.py', '')}/1080p60/EducationalScene.mp4"
        
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
        
        # Get video duration to ensure audio matches
        video_duration_cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        video_duration_result = subprocess.run(video_duration_cmd, capture_output=True, text=True)
        video_duration = float(video_duration_result.stdout.strip())
        
        audio_duration = get_audio_duration(audio_file)
        
        print(f"   Video duration: {video_duration:.2f}s")
        print(f"   Audio duration: {audio_duration:.2f}s")
        
        # If video is longer than audio, pad audio with silence
        if video_duration > audio_duration:
            silence_duration = video_duration - audio_duration
            print(f"   Padding audio with {silence_duration:.2f}s of silence...")
            
            padded_audio = f"{output_name}_padded_audio.mp3"
            pad_cmd = [
                "ffmpeg", "-y",
                "-i", audio_file,
                "-f", "lavfi",
                "-t", str(silence_duration),
                "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[out]",
                "-map", "[out]",
                padded_audio
            ]
            subprocess.run(pad_cmd, check=True, capture_output=True)
            audio_file = padded_audio
            print(f"   ✅ Audio padded to {video_duration:.2f}s")
        
        combine_cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_file,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",  # Now safe to use since audio matches video duration
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
            
            # Filter out nested if statements (if inside loops or other if statements)
            filtered_blocks = []
            for block in blocks:
                if block['type'] == 'if_statement':
                    # Check if this if statement is nested inside a loop or another if
                    is_nested = False
                    for other_block in blocks:
                        if other_block != block and other_block['type'] in ['for_loop', 'while_loop', 'if_statement']:
                            # Check if this if is inside the other block
                            if other_block['start_line'] <= block['start_line'] and block['end_line'] <= other_block['end_line']:
                                is_nested = True
                                break
                    if not is_nested:
                        filtered_blocks.append(block)
                else:
                    # Keep all non-if blocks
                    filtered_blocks.append(block)
            
            blocks = filtered_blocks
            print(f"✅ Parsed {len(blocks)} code blocks from AST (filtered nested if statements)\n")
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
            # Skip if statements that are nested inside other control structures
            # This prevents duplicate highlights for nested blocks
            is_nested = False
            for block in blocks:
                # Check if this if statement is inside a class, loop, or another if statement
                if block['type'] in ['class', 'for_loop', 'while_loop', 'if_statement'] and block['start_line'] <= i <= block['end_line']:
                    is_nested = True
                    break
            
            if not is_nested:
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
    
    if len(test_code_lines) > 8:  # Approximate threshold for scrolling
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
        concepts_title = Text("Key Concepts", font_size=48, font="Inter", weight=SEMIBOLD, color=GOLD)
        concepts_title.to_edge(UP, buff=0.5)
        
        # Create concept items with modern styling
        concept_items = VGroup(*[
            Text(f"• {{concept}}", font_size=26, font="Inter", color=YELLOW)
            for concept in {concepts_list}
        ])
        concept_items.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        concept_items.next_to(concepts_title, DOWN, buff=0.6)
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
        current_time += 0.2  # Time for fade out
    
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
        overview_title = Text("Code Overview", font_size=48, font="Inter", weight=SEMIBOLD, gradient=(BLUE, PURPLE))
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
        bullet_points.next_to(overview_title, DOWN, buff=0.6)
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
    
    # Use the decision made at the beginning of the function
    # use_incremental_reveal is already set based on code length > 8 lines
    
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
        # Beautiful dark background (matches animated video)
        self.camera.background_color = "#0F172A"
        
{overview_slides_code}        
        title = Text("Code Explanation", font_size=48, font="Inter", weight=SEMIBOLD, gradient=(BLUE, PURPLE))
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title, shift=DOWN*0.3), run_time=0.8)
        self.wait(0.3)
        
        # CODE HIKE SETUP: Pre-build all code lines IMMEDIATELY (no delay)
        code_lines = VGroup()
        raw_lines = \"\"\"{escaped_code}\"\"\".split('\\n')
        for line in raw_lines:
            # Use non-breaking spaces to preserve indentation
            formatted_line = line.replace(" ", "\\u00A0")
            t = Text(
                formatted_line, 
                font_size=24, 
                font="Inter", 
                color=WHITE,
                line_spacing=1.3
            )
            # Scale if too wide
            if t.width > 12:
                t.scale_to_fit_width(12)
            code_lines.add(t)
        
        # Arrange vertically with proper indentation
        # First, stack all lines vertically (aligned to left edge temporarily)
        code_lines.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        
        # Then, shift each line right based on its indentation level
        # Calculate indentation for each line and shift accordingly
        indent_shift_per_space = 0.06  # Horizontal shift per space (4 spaces = 0.24 units, clearly visible)
        for idx, line in enumerate(raw_lines):
            # Count leading spaces
            leading_spaces = len(line) - len(line.lstrip())
            if leading_spaces > 0:
                # Shift this line to the right
                code_lines[idx].shift(RIGHT * (leading_spaces * indent_shift_per_space))
        
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
        # Beautiful dark background (matches animated video)
        self.camera.background_color = "#0F172A"
        
{overview_slides_code}        
        title = Text("Code Explanation", font_size=48, font="Inter", weight=SEMIBOLD, gradient=(BLUE, PURPLE))
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title, shift=DOWN*0.3), run_time=0.8)
        self.wait(0.3)
        
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





def code_to_video(code_content: str, output_name: str = "output", audio_language: str = "english", add_subtitles: bool = False):
    
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
            
            block_size = block['end_line'] - block['start_line'] + 1
            
            if block['type'] == 'class':
                print(f"   ❌ {block['type']} (lines {block['start_line']}-{block['end_line']}): NARRATION ✅, HIGHLIGHT ❌ (too large)")
                continue
            else:
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
            "-pqh", scene_file, "CodeExplanationScene"  # High quality: 1080p60
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
        
        video_path = f"media/videos/{scene_file.replace('.py', '')}/1080p60/CodeExplanationScene.mp4"
        
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




def query_to_animated_video_v3(query: str, output_name: str = "output", audio_language: str = "english"):
   
    try:
        print(f"\n{'='*60}")
        print(f"🎨 QUERY TO ANIMATED VIDEO V3 (AI-CLASSIFIED)")
        print(f"{'='*60}\n")
        
        import os, json, subprocess, textwrap
        import openai
        import anthropic
        
        api_key = os.getenv("OPENAI_API_KEY")
        claude_key = os.getenv("ANTHROPIC_API_KEY")
        client = openai.OpenAI(api_key=api_key)
        claude_client = anthropic.Anthropic(api_key=claude_key)
        
        # ============================================================
        # STEP 0: AI-DRIVEN QUERY CLASSIFICATION
        # ============================================================
        # Let GPT-4 analyze the query and decide the best approach
        
        print(f"🤖 Analyzing query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        print(f"   📊 Classifying query type with AI...")
        
        classification_prompt = f"""Analyze this query and classify it for educational video generation.

QUERY: {query}

Classify into ONE of these categories:

1. **algorithm** - Explaining how an algorithm works (sorting, searching, graph traversal, etc.)
   - Needs: Dynamic slide count (as many as needed to show complete process)
   - Slides: CONNECTED (each builds on previous, showing state progression)
   - Examples: "Explain quicksort", "How does BFS work?", "Show merge sort step by step"

2. **process** - Sequential/step-by-step explanation where order matters
   - Needs: 4-6 slides depending on complexity
   - Slides: CONNECTED (Step 1 → Step 2 → Step 3, each depends on previous)
   - Examples: "Explain TCP handshake", "How does OAuth flow work?", "Authentication process"

3. **conceptual** - Independent concepts/aspects that don't depend on each other
   - Needs: 4 slides for short, 6 for long
   - Slides: INDEPENDENT (each covers different aspect: definition, types, uses, comparison)
   - Examples: "What is a database?", "Explain REST API", "What is machine learning?"

4. **coding** - Code explanation or programming concepts with code
   - Needs: 4-6 slides based on code complexity
   - Slides: CONNECTED (show code execution flow)
   - Examples: Code snippets, "Explain this function", programming tutorials

5. **long_content** - User provided long text/content to visualize
   - Needs: 6 slides
   - Slides: Can be CONNECTED or INDEPENDENT based on content nature
   - Examples: Long paragraphs of text, articles, detailed explanations

Return JSON:
{{
    "query_type": "algorithm" | "process" | "conceptual" | "coding" | "long_content",
    "slides_connected": true | false,
    "recommended_slides": 4 | 5 | 6,
    "reasoning": "Brief explanation of why this classification"
}}

IMPORTANT GUIDELINES:
- For algorithms: recommend enough slides to show COMPLETE process (usually 5-8)
- For process: slides must be CONNECTED (true) - each step builds on previous
- For conceptual: slides can be INDEPENDENT (false) - each covers different aspect
- Short queries (< 100 words) that are conceptual → 4 slides
- Long queries (> 150 words) → 6 slides minimum
- Algorithms should have as many slides as needed (5-8 typically)"""

        classification_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert at classifying educational content for video generation. Be precise and accurate."},
                {"role": "user", "content": classification_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        classification = json.loads(classification_response.choices[0].message.content)
        
        query_type = classification.get("query_type", "conceptual")
        slides_connected = classification.get("slides_connected", False)
        num_slides = classification.get("recommended_slides", 4)
        reasoning = classification.get("reasoning", "")
        
        # Ensure num_slides is within bounds
        num_slides = max(4, min(num_slides, 6))
        
        print(f"   ✅ Classification Result:")
        print(f"      📂 Type: {query_type.upper()}")
        print(f"      🔗 Slides Connected: {'Yes (sequential/process)' if slides_connected else 'No (independent aspects)'}")
        print(f"      📊 Recommended Slides: {num_slides}")
        print(f"      💡 Reasoning: {reasoning}\n")
        
        # ============================================================
        # STEP 1: BUILD SLIDE TEMPLATE BASED ON CLASSIFICATION
        # ============================================================
        
        slides_template = ""
        algorithm_instruction = ""
        
        # Get complexity labels for each slide position
        def get_complexity(slide_num, total_slides):
            if slide_num == 1:
                return "good"
            elif slide_num == 2:
                return "advanced"
            elif slide_num == total_slides - 1:
                return "extremely advanced"
            else:
                return "most creative and extremely advanced"
        
        if query_type == "algorithm":
            # ALGORITHM: Connected slides showing state progression
            print(f"   🎯 USING TEMPLATE: ALGORITHM (connected slides, state progression)")
            print(f"      → Slides will show: Initial State → Step by Step Changes → Final Result")
            
            algorithm_instruction = """
🚨 CRITICAL RULE - EXTRACT AND USE EXACT VALUES FROM THE QUERY! 🚨

**STEP 1**: Look at the query/code and find ALL variable values (arrays, numbers, etc.)
**STEP 2**: Use those EXACT SAME values in ALL slides - NEVER change or pick different values!

Example: If query shows `arr = [-4, -2, 1, -3]` and `k = 2`:
- ✅ CORRECT: Use arr = [-4, -2, 1, -3] and k = 2 in EVERY slide
- ❌ WRONG: Use arr = [2, 1, 5, 1, 3, 2] in later slides (DIFFERENT VALUES!)

**IF query has specific values → USE THEM EXACTLY**
**IF query has NO values → Pick values ONCE in slide 1, then use SAME values in all slides**

EVERY slide must show the FULL array state with EXACT values from query!
"""
            
            for i in range(1, num_slides + 1):
                complexity = get_complexity(i, num_slides)
                if i == 1:
                    step_desc = "Initial state - Use EXACT array values from query (e.g., if query shows arr=[-4,-2,1,-3], use those exact values)"
                elif i == num_slides:
                    step_desc = "Final result - show the result using SAME EXACT values from slide 1"
                else:
                    step_desc = f"Step {i} - show FULL array state with SAME EXACT values from slide 1, highlight operations"
                
                slides_template += f"""        {{
            "visualization_description": "{step_desc}. CRITICAL: Use EXACT values from query code, keep them consistent!",
            "complexity": "{complexity}",
            "step": {i},
            "continues_from_previous": {str(i > 1).lower()},
            "narration": "Explain what operation happens and how it changes the state (4-5 sentences)"
        }}"""
                if i < num_slides:
                    slides_template += ","
                slides_template += "\n"
                
        elif query_type == "process" or (query_type == "long_content" and slides_connected):
            # PROCESS: Connected slides showing sequential steps
            print(f"   🎯 USING TEMPLATE: PROCESS (connected slides, sequential steps)")
            print(f"      → Slides will show: Step 1 → Step 2 → Step 3 → ... (each builds on previous)")
            for i in range(1, num_slides + 1):
                complexity = get_complexity(i, num_slides)
                if i == 1:
                    step_desc = "First step of the process - introduce the starting point"
                elif i == num_slides:
                    step_desc = "Final step - show completion and result of the entire process"
                else:
                    step_desc = f"Step {i} - CONTINUES from previous step, show what happens next"
                
                slides_template += f"""        {{
            "visualization_description": "{step_desc}. Each slide MUST connect to the previous one.",
            "complexity": "{complexity}",
            "step": {i},
            "continues_from_previous": {str(i > 1).lower()},
            "narration": "Explain this step and how it connects to the previous step (4-5 sentences)"
        }}"""
                if i < num_slides:
                    slides_template += ","
                slides_template += "\n"
                
        elif query_type == "coding":
            # CODING: Connected slides showing code execution
            print(f"   🎯 USING TEMPLATE: CODING (connected slides, code execution flow)")
            print(f"      → Slides will show: Setup → First Op → Core Logic → ... → Final Result")
            
            algorithm_instruction = """
🚨 CRITICAL RULE - EXTRACT AND USE EXACT VALUES FROM THE QUERY CODE! 🚨

**STEP 1**: Look at the query/code and find ALL variable values (arrays, numbers, parameters, etc.)
**STEP 2**: Use those EXACT SAME values throughout ALL slides - NEVER change or pick different values!

Example: If code shows `arr = [-4, -2, 1, -3]` and `k = 2`:
- ✅ CORRECT: Use arr = [-4, -2, 1, -3] and k = 2 in EVERY slide showing execution
- ❌ WRONG: Use arr = [2, 1, 5, 1, 3, 2] in slide 2 (COMPLETELY DIFFERENT VALUES!)

**IF query has specific values → USE THEM EXACTLY in all slides**
**IF query has NO values → Pick values ONCE in slide 1, then use SAME values throughout**

Show how the code executes step-by-step with the EXACT input values from the query!
"""
            
            step_names = ["Setup/Initial State", "First Operation", "Core Logic", "Next Step", "Processing", "Final Result/Output"]
            
            for i in range(1, num_slides + 1):
                complexity = get_complexity(i, num_slides)
                step_name = step_names[i-1] if i <= len(step_names) else f"Step {i}"
                
                slides_template += f"""        {{
            "visualization_description": "Show {step_name.lower()} using EXACT values from query code (e.g., if code shows arr=[-4,-2,1,-3], use those values). Keep values consistent across all slides!",
            "complexity": "{complexity}",
            "step": {i},
            "continues_from_previous": {str(i > 1).lower()},
            "narration": "Explain what happens in this step of the code (4-5 sentences)"
        }}"""
                if i < num_slides:
                    slides_template += ","
                slides_template += "\n"
                
        elif query_type == "conceptual" or (query_type == "long_content" and not slides_connected):
            # CONCEPTUAL: Independent slides covering different aspects
            print(f"   🎯 USING TEMPLATE: CONCEPTUAL (independent slides, different aspects)")
            print(f"      → Slides will show: Definition | Types | Use Cases | Advanced (each independent)")
            aspect_hints = [
                "Core definition/introduction - what IS this concept",
                "Key components/types - break down the main parts",
                "How it works/use cases - practical applications",
                "Advanced aspects/comparisons - deeper insights",
                "Real-world examples - concrete implementations",
                "Summary/best practices - key takeaways"
            ]
            
            for i in range(1, num_slides + 1):
                complexity = get_complexity(i, num_slides)
                aspect = aspect_hints[i-1] if i <= len(aspect_hints) else f"Additional aspect {i}"
                
                slides_template += f"""        {{
            "visualization_description": "{aspect}. This slide covers a DIFFERENT aspect from other slides.",
            "complexity": "{complexity}",
            "aspect": {i},
            "independent": true,
            "narration": "Explain this specific aspect comprehensively (4-5 sentences)"
        }}"""
                if i < num_slides:
                    slides_template += ","
                slides_template += "\n"
        
        else:
            # DEFAULT: Long content visualization
            print(f"   🎯 USING TEMPLATE: DEFAULT/LONG_CONTENT")
            print(f"      → Slides will visualize the provided content")
            for i in range(1, num_slides + 1):
                complexity = get_complexity(i, num_slides)
                
                slides_template += f"""        {{
            "visualization_description": "Create {complexity} animation for aspect {i}. Use REAL terminology from the content.",
            "complexity": "{complexity}",
            "topic": "Topic {i} from the content",
            "narration": "Cover this specific aspect comprehensively (4-5 sentences)"
        }}"""
                if i < num_slides:
                    slides_template += ","
                slides_template += "\n"
        
        # Build the main structure prompt
        connected_instruction = ""
        if slides_connected:
            connected_instruction = """
**CONNECTED SLIDES REQUIRED**: Each slide MUST build on the previous one!
- Slide 1 establishes initial state
- Slide 2 shows what happens AFTER slide 1
- Slide 3 continues from slide 2's state
- And so on... Each slide references and builds upon the previous!
- The narration should use transitions like "Now...", "Next...", "After that...", "Building on this..."
"""
        else:
            connected_instruction = """
**INDEPENDENT SLIDES**: Each slide covers a DIFFERENT aspect of the topic.
- Slide 1: Definition/Introduction
- Slide 2: Types/Components  
- Slide 3: Use cases/Applications
- Slide 4: Advanced concepts/Comparisons
- Each slide is self-contained and covers a unique aspect!
"""
        
        structure_prompt = f"""
═══════════════════════════════════════════════════════════════════
🚨🚨🚨 CRITICAL RULE #1 - READ THIS FIRST! 🚨🚨🚨
═══════════════════════════════════════════════════════════════════
LOOK AT THE QUERY CODE AND EXTRACT ALL VARIABLE VALUES!
IF the query shows: arr = [-4, -2, 1, -3] and k = 2
THEN YOU MUST USE: arr = [-4, -2, 1, -3] and k = 2 IN EVERY SINGLE SLIDE!

❌ ABSOLUTELY FORBIDDEN: Using different array values like [5, -3, -2, 1, 7] in slide 2
✅ ABSOLUTELY REQUIRED: Use the EXACT SAME values [-4, -2, 1, -3] in ALL slides

THIS IS THE #1 MOST IMPORTANT RULE - NEVER VIOLATE IT!
═══════════════════════════════════════════════════════════════════

Create educational video structure for: {query}
{algorithm_instruction}
{connected_instruction}
Generate {num_slides} ANIMATION slides that explain this topic visually.

CRITICAL RULES:
1. Each slide must show a DIFFERENT aspect/stage of the topic (NO REPETITION!)
2. {"For CONNECTED slides: Show SEQUENTIAL progression - each slide builds on the previous state/step" if slides_connected else "For INDEPENDENT slides: Each slide covers a DIFFERENT aspect of the topic (definition, types, uses, advanced concepts)"}
3. **USE ACTUAL TERMINOLOGY from the query** - NOT generic "Node A, B, C" or "Component 1, 2, 3"
4. Concepts must be ULTRA-SPECIFIC with exact details!
5. Narration must explain EVERY element shown (no skipping connections/relationships!)
6. COMPARISONS: If comparing A vs B, keep it SIMPLE - max 3 bullet points per side, font_size >= 20, NO small boxes with lots of text inside!
7. Never Create animation that goes out of screen
8. **CHOOSE APPROPRIATE VISUALIZATION TYPE:**
   - For frameworks/systems (LangGraph, React, etc.): Use labeled diagrams with REAL component names
   - For algorithms (sorting, searching): Use animated arrays/data structures
   - For processes (recursion, loops): Use call stacks or flow diagrams
   - Be creative - choose what fits the concept best!

BAD (vague, generic):
- "Visualize an array"
- "Show a linked list"
- "Node A connects to Node B"
- "Component 1, Component 2, Component 3"

GOOD (specific, uses real terminology):
- "5 boxes in a row, labeled 1,2,3,4,5 from left to right"
- "4 circles connected by arrows: Head→Node(5)→Node(3)→Tail"
- "LangGraph StateGraph with nodes labeled: 'agent', 'tools', 'supervisor'"
- "React component tree: App → Header, Sidebar, Content"
- "Two columns comparing A vs B - max 3 short bullet points each, large readable text"

REQUIREMENTS:
- Specify EXACT number of elements
- **Use REAL names/labels from the topic** (not generic placeholders!)
- Specify EXACT layout (horizontal, vertical, grid, tree, etc.)
- Narration must describe EXACTLY what visual shows
- Each slide: 3-4 sentences
- **For technical topics: Use actual technical terms** (StateGraph, not "Graph"; Agent, not "Node A")

Return JSON:
{{
    "title": "Topic Title (max 40 chars)",
    "animation_slides": [
{slides_template}    ]
}}

**IMPORTANT**: 
- {"Generate EXACTLY " + str(num_slides) + " slides that are CONNECTED (each builds on previous)" if slides_connected else "Generate EXACTLY " + str(num_slides) + " INDEPENDENT slides (each covers different aspect)"}
- Each "visualization_description" should describe the ACTUAL visualization for THIS specific topic
- Use REAL names from the topic (not "Node A", "Component 1", etc.)
- Be specific about what to show (not generic instructions)
- ALL slides should have rich, dynamic animations (not just static boxes!)


EXAMPLES OF GOOD CONCEPTS (with real terminology):
- "3 circles labeled 'Input Layer', 'Hidden Layer', 'Output Layer' connected by arrows left to right"
- "LangGraph: StateGraph node at center, connected to 'agent', 'tools', 'supervisor' nodes"
- "React vs Remix: Left side shows 'Client Components', right side shows 'Server Components'"
- "Grid of 6 boxes (2 rows, 3 cols) with numbers 1-6, showing dice faces"
- "Two boxes side by side: left says 'Unsorted: 5,2,8', right says 'Sorted: 2,5,8'"
- "Recursion call stack: factorial(3) → factorial(2) → factorial(1) → returns"
- "Tree with 1 root circle labeled 'Binary Search Tree Root: 5', 2 child circles labeled 'Left: 3' and 'Right: 7'"

IMPORTANT:
- Be PRESCRIPTIVE not DESCRIPTIVE
- Say WHAT to show, not what it represents
- Include exact numbers/labels **using REAL terminology**
- Specify exact layout
- **NO GENERIC PLACEHOLDERS** - use actual concept names!

═══════════════════════════════════════════════════════════════════
🚨 FINAL REMINDER - DO NOT FORGET! 🚨
═══════════════════════════════════════════════════════════════════
IF the query code contains specific values (like arr = [-4, -2, 1, -3]):
→ Extract those values ONCE
→ Use the EXACT SAME values in EVERY slide
→ NEVER use different values in different slides
→ This is CRITICAL for algorithm/coding queries!

Example CORRECT behavior:
- Query: arr = [-4, -2, 1, -3], k = 2
- Slide 1: Show array [-4, -2, 1, -3] initially
- Slide 2: Show array [-4, -2, 1, -3] with operations
- Slide 3: Show array [-4, -2, 1, -3] in different state
- ALL SLIDES: Use [-4, -2, 1, -3] - NEVER change to [5, -3, -2, 1, 7]!
═══════════════════════════════════════════════════════════════════

Generate EXACTLY {num_slides} slides."""

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
        
        # VALIDATE STRUCTURE - Reject vague concepts!
        slides = structure.get('animation_slides', [])
        vague_concepts = []
        for idx, slide in enumerate(slides):
            concept = slide.get('visualization_description', slide.get('concept', ''))
            # Check if concept is specific enough
            has_numbers = any(char.isdigit() for char in concept)
            has_layout_word = any(word in concept.lower() for word in ['horizontal', 'vertical', 'grid', 'row', 'column', 'left', 'right', 'top', 'bottom', 'circle', 'box', 'square', 'rectangle'])
            is_short = len(concept) < 20  # Too short = vague
            
            if not has_numbers or not has_layout_word or is_short:
                vague_concepts.append(f"Slide {idx+1}: '{concept[:50]}...' - needs numbers, layout, and specific labels")
        
        if vague_concepts:
            print(f"⚠️  Structure too vague! Regenerating...")
            print(f"   Issues: {vague_concepts[0]}")
            # Regenerate with feedback
            feedback_prompt = f"""{structure_prompt}

YOUR PREVIOUS ATTEMPT WAS TOO VAGUE:
{chr(10).join(vague_concepts)}

FIX IT! Each concept MUST include:
1. EXACT numbers (e.g. "3 circles", "5 boxes")
2. EXACT labels (e.g. "labeled A, B, C" or "showing 1, 2, 3")
3. EXACT layout (e.g. "arranged horizontally", "in a 2x3 grid")

Example GOOD concept: "3 circles labeled 'Input', 'Process', 'Output' arranged horizontally with arrows between them"
Example BAD concept: "Complex Connections" (too vague!)

Regenerate with SPECIFIC concepts:"""
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Create clear educational content with animations only."},
                    {"role": "user", "content": feedback_prompt}
                ],
                temperature=0.5,  # Lower temperature for more focused output
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
        # NEW STRATEGY: Generate audio + code together, skip failures
        successful_slides = []  # Only slides that passed validation
        
        for idx, slide in enumerate(structure.get('animation_slides', [])):
            print(f"\n📍 Processing slide {idx + 1}...")
            
            # Initialize variables for cleanup
            raw_tts_file = f"{output_name}_anim_{idx}_raw.mp3"
            narration_wav = f"{output_name}_anim_{idx}_narration.wav"
            silence_wav = f"{output_name}_anim_{idx}_silence.wav"
            audio_file = None
            
            try:
                # Step 1: Generate TTS audio
                generate_audio_for_language(slide.get('narration', ''), audio_language, raw_tts_file, client)
                
                # Convert to standardized WAV
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
                
                # Step 2: Generate Manim code for this slide
                concept = slide.get('visualization_description', slide.get('concept', ''))
                narration = slide.get('narration', '')
                complexity = slide.get('complexity', 'simple')
                
                # Calculate target duration (subtract overhead)
                video_overhead = 1.1
                target_duration = max(narration_dur - video_overhead, 3.0)
                
                # Generate animation code
                animation_code = generate_centered_animation(
                    concept, narration, target_duration, 
                    claude_client, skip_global_sync=True, complexity=complexity
                )
                
                # Validate animation_code
                if not animation_code or not animation_code.strip():
                    raise ValueError("Generated animation code is empty")
                
                # CRITICAL: Apply auto-fixes BEFORE test render
                # This prevents runtime errors during test render
                import re
                
                # Fix 1: .arrange(rows=, cols=) → .arrange_in_grid(rows=, cols=)
                if '.arrange(rows=' in animation_code or '.arrange(cols=' in animation_code:
                    animation_code = re.sub(r'\.arrange\((rows=\d+,\s*cols=\d+[^)]*)\)', r'.arrange_in_grid(\1)', animation_code)
                    print(f"   🔧 AUTO-FIX: Changed .arrange(rows=, cols=) → .arrange_in_grid(rows=, cols=)")
                
                # Fix 2: self.wait(0) → self.wait(0.1)
                if 'self.wait(0' in animation_code:
                    animation_code = re.sub(r'self\.wait\(0(?:\.0)?\)', 'self.wait(0.1)', animation_code)
                    print(f"   🔧 AUTO-FIX: Changed self.wait(0) → self.wait(0.1)")
                
                # Step 2.5: TEST-RENDER this slide to catch runtime errors
                # This catches errors like .add_tip(), invalid methods, list index errors, etc.
                print(f"   🧪 Test-rendering slide {idx + 1} to check for runtime errors...")
                
                # Create a minimal test scene with this slide's code
                test_scene_code = f"""from manim import *

class TestSlide(Scene):
    def construct(self):
        self.camera.background_color = "#0F172A"
        
        # Slide code (indented)
{chr(10).join('        ' + line for line in animation_code.split(chr(10)))}
"""
                
                # Write test file
                test_file = f".temp_test_slide_{idx}.py"
                with open(test_file, 'w') as f:
                    f.write(test_scene_code)
                
                # Try to render (low quality, fast)
                try:
                    test_result = subprocess.run([
                        sys.executable, "-m", "manim", "-pql", "--disable_caching",
                        test_file, "TestSlide"
                    ], capture_output=True, timeout=90, check=True)
                    
                    print(f"   ✅ Test render passed - slide {idx + 1} has no runtime errors")
                    
                except subprocess.TimeoutExpired:
                    print(f"   ❌ Test render timeout - slide {idx + 1} is too slow or has infinite loop")
                    raise ValueError("Test render timeout")
                    
                except subprocess.CalledProcessError as e:
                    # Runtime error detected!
                    error_msg = e.stderr.decode() if e.stderr else str(e)
                    print(f"   ❌ Test render failed - slide {idx + 1} has runtime error:")
                    print(f"      {error_msg[:1500]}")  # Show more of the error
                    print(f"\n   🔍 FULL ERROR SAVED TO: .temp_slide_{idx+1}_error.txt")
                    # Save full error to file for debugging
                    with open(f".temp_slide_{idx+1}_error.txt", "w") as f:
                        f.write(error_msg)
                    raise ValueError(f"Runtime error in slide: {error_msg[:200]}")
                    
                finally:
                    # Cleanup test files
                    if os.path.exists(test_file):
                        os.remove(test_file)
                    # Cleanup test video directory
                    test_video_dir = f"media/videos/{test_file.replace('.py', '')}"
                    if os.path.exists(test_video_dir):
                        shutil.rmtree(test_video_dir)
                
                # Step 3: Create final audio file (with silence if needed)
                if len(successful_slides) == 0:
                    # First slide - no silence needed
                    audio_file = narration_wav
                    total_dur = narration_dur
                else:
                    # Add 1.6s silence before this slide
                    subprocess.run([
                        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                        "-t", "1.6", "-c:a", "pcm_s16le", silence_wav
                    ], check=True, capture_output=True)
                    
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
                
                # ✅ SUCCESS - Add to successful slides
                successful_slides.append({
                    'concept': concept,
                    'narration': narration,
                    'complexity': complexity,
                    'animation_code': animation_code,
                    'audio_file': audio_file,
                    'duration': narration_dur
                })
                
                print(f"   ✅ Slide {idx + 1}: {narration_dur:.2f}s (SUCCESS)")
                
            except Exception as e:
                # ❌ FAILED - Skip this slide entirely
                print(f"   ❌ Slide {idx + 1} FAILED: {str(e)[:100]}")
                print(f"   ⚠️  Skipping slide {idx + 1} (no audio, no video)")
                
                # Comprehensive cleanup of ALL possible temp files
                for temp_file in [raw_tts_file, narration_wav, silence_wav]:
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except:
                            pass  # Ignore cleanup errors
                
                # Cleanup audio_file if it was created
                if audio_file and os.path.exists(audio_file):
                    try:
                        os.remove(audio_file)
                    except:
                        pass
                
                continue  # Skip to next slide
        
        # Check if we have any successful slides
        if len(successful_slides) == 0:
            raise ValueError("All slides failed! Cannot create video.")
        
        print(f"\n✅ {len(successful_slides)}/{len(structure.get('animation_slides', []))} slides succeeded\n")
        
        # Extract data for next steps
        audio_files = [s['audio_file'] for s in successful_slides]
        durations = [('animation', s['duration']) for s in successful_slides]
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
        manim_code = generate_fixed_slides_code(structure, durations, claude_client, successful_slides)
        
        # Step 4: Render
        print("🎬 Rendering...")
        scene_file = f".temp_{output_name}.py"
        with open(scene_file, 'w') as f:
            f.write(manim_code)
        
        try:
            subprocess.run([
                sys.executable, "-m", "manim", "-pqh", scene_file, "FixedScene"
            ], check=True, capture_output=True)
            video_path = f"media/videos/{scene_file.replace('.py', '')}/1080p60/FixedScene.mp4"
        except subprocess.CalledProcessError as e:
            print(f"❌ Manim rendering failed (logic error in generated code)")
            print(f"   This is a runtime error, not a syntax error")
            print(f"   Error: {e.stderr.decode()[:500]}")
            print(f"   🔄 Generating simple fallback video...")
            
            # Generate simple fallback Manim code
            fallback_code = f'''from manim import *

class FixedScene(Scene):
    def construct(self):
        # Simple text-only fallback
        title = Text("{structure.get("title", "Video")[:40]}", font_size=48, color=WHITE)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=2.0)
        self.wait(2.0)
        
        # Show each slide as simple text
'''
            # Add each slide from successful_slides
            for idx, slide_data in enumerate(successful_slides):
                concept = slide_data.get('concept', f'Slide {idx+1}')[:50]
                slide_dur = slide_data.get('duration', 5.0)
                fallback_code += f'''
        # Slide {idx+1}
        slide_text = Text("{concept}", font_size=36, color=BLUE)
        slide_text.move_to(ORIGIN)
        self.play(FadeIn(slide_text), run_time=1.0)
        self.wait({slide_dur - 1.0})
        self.play(FadeOut(slide_text), run_time=0.5)
'''
            
            # Write fallback code
            fallback_file = f".temp_{output_name}_fallback.py"
            with open(fallback_file, 'w') as f:
                f.write(fallback_code)
            
            # Render fallback
            print(f"   🎬 Rendering fallback video...")
            subprocess.run([
                sys.executable, "-m", "manim", "-pqh", fallback_file, "FixedScene"
            ], check=True, capture_output=True)
            
            video_path = f"media/videos/{fallback_file.replace('.py', '')}/1080p60/FixedScene.mp4"
            print(f"   ✅ Fallback video generated successfully")
        
        
        # Step 5: Combine with audio padding
        print("🎵 Combining video + audio...")
        final_output = f"{output_name}_final.mp4"
        
        # Get video duration
        video_duration_cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        video_duration_result = subprocess.run(video_duration_cmd, capture_output=True, text=True)
        video_duration = float(video_duration_result.stdout.strip())
        
        audio_duration = get_audio_duration(audio_file)
        
        print(f"   Video duration: {video_duration:.2f}s")
        print(f"   Audio duration: {audio_duration:.2f}s")
        
        # Pad audio if needed
        if video_duration > audio_duration:
            silence_duration = video_duration - audio_duration
            print(f"   Padding audio with {silence_duration:.2f}s of silence...")
            
            padded_audio = f"{output_name}_padded_audio.mp3"
            pad_cmd = [
                "ffmpeg", "-y",
                "-i", audio_file,
                "-f", "lavfi",
                "-t", str(silence_duration),
                "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[out]",
                "-map", "[out]",
                padded_audio
            ]
            subprocess.run(pad_cmd, check=True, capture_output=True)
            audio_file = padded_audio
            print(f"   ✅ Audio padded to {video_duration:.2f}s")
        
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


def generate_fixed_slides_code(structure, durations, client, successful_slides):
    """Generate Manim code with PROPER text wrapping and CENTERED animations"""
    
    import textwrap
    
    title = structure.get("title", "")
    
    # Wrap title if too long
    title_lines = textwrap.wrap(title, width=40)
    
    code = f"""from manim import *

class FixedScene(MovingCameraScene):
    def construct(self):
        # Kodnest-style dark background
        self.camera.background_color = "#0F172A"
        self.camera.frame.save_state()
        
        # SYNC FIX: Audio has 4s silence at start
        # Title animations take 2.8s (0.8 fadein + 1.5 wait + 0.5 fadeout)
        # So we need 1.2s wait before title to fill the 4s silence
        self.wait(1.2)
        
        # Title (with wrapping for long titles)
        title_texts = VGroup()
"""
    
    # Add each line of the title
    for i, line in enumerate(title_lines):
        code += f"""        title_line_{i} = Text("{line}", font_size=36, font="Inter", weight=SEMIBOLD, color=GOLD)
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
    
    # Use pre-generated animation code from successful_slides
    for idx, slide in enumerate(successful_slides):
        concept = slide['concept']
        animation_code = slide['animation_code']  # Already generated!
        
        # Indent the code to fit inside construct()
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


def generate_centered_animation(concept, narration, target_duration, client, skip_global_sync=False, complexity="simple"):
    """Generate animation that MATCHES narration exactly AND follows complexity level"""
    
    # PROGRESSIVE COMPLEXITY - Let AI be creative!
    complexity_rules = {
        "simple": """
🎯 SIMPLE & CLEAR
Create a clean animation that introduces the concept.
Use whatever techniques work best - you have full creative freedom!
""",
        "good": """
🎯 GOOD QUALITY
Create a well-structured animation with clear visual elements.
Use appropriate Manim techniques!
""",
        "intermediate": """
🎯 MODERATE COMPLEXITY
Build with appropriate elements and connections for the topic.
Be creative - choose any Manim techniques that fit!
""",
        "advanced": """
🎯 ADVANCED & DYNAMIC
Create a sophisticated visualization with rich interactions.
Use advanced techniques - show your creativity!
""",
        "very advanced": """
🎯 VERY ADVANCED
Create a complex, dynamic visualization.
Use multiple animation techniques and show deep understanding!
""",
        "extremely advanced": """
🎯 EXTREMELY ADVANCED - FINALE! 🚀
Create an impressive, comprehensive visualization of the complete system.
Combine multiple techniques to create something memorable!
""",
        "most creative and extremely advanced": """
🎯 MOST CREATIVE & EXTREMELY ADVANCED - GRAND FINALE! 🎆
Create the most impressive, creative visualization possible!
Combine ALL techniques to create something truly memorable!
Use your full creative potential!
""",
        # Legacy mappings
        "complex": """
🎯 EXTREMELY ADVANCED - FINALE! 🚀
Create an impressive, comprehensive visualization of the complete system.
Combine multiple techniques to create something memorable!
"""
    }
    
    complexity_instruction = complexity_rules.get(complexity, complexity_rules["intermediate"])

    
    prompt = f"""
═══════════════════════════════════════════════════════════════════
🚨🚨🚨 RULE #1 - USE EXACT ARRAY VALUES FROM NARRATION! 🚨🚨🚨
═══════════════════════════════════════════════════════════════════
IF the narration mentions specific values (e.g., "array [-4, -2, 1, -3]"):
→ Extract those EXACT values
→ Use them in your code: values = [-4, -2, 1, -3]
→ NEVER make up different values like [5, -3, -2, 1, 7]

NARRATION: "...with array [-4, -2, 1, -3] and k equals 2..."
YOUR CODE MUST USE:
```python
values = [-4, -2, 1, -3]  # ← EXACT values from narration
k_value = 2               # ← EXACT value from narration
```

❌ WRONG: values = [2, 1, 5, 1, 3, 2]  # Different values!
✅ CORRECT: values = [-4, -2, 1, -3]   # Exact values from narration!
═══════════════════════════════════════════════════════════════════

🚨🚨🚨 CRITICAL - READ THIS! 🚨🚨🚨

🚫🚫🚫 NO 3D SCENES - WE DO NOT SUPPORT 3D! 🚫🚫🚫
**CRITICAL: You MUST use Scene (NOT ThreeDScene)!**
- ❌ FORBIDDEN: ThreeDScene, ThreeDAxes, Sphere(), Cube(), Cone()
- ❌ FORBIDDEN: set_camera_orientation(), move_camera(), .fix_in_frame()
- ✅ REQUIRED: Use Scene class ONLY
- ✅ CREATE 3D ILLUSIONS: Use 2D layering, shadows, gradients, isometric projections

⛔ ABSOLUTE REQUIREMENTS - CODE WILL BE REJECTED IF VIOLATED:
1. ❌ NO INCOMPLETE CODE - Every line must be complete and valid Python
2. ❌ NO MISSING OPENING/CLOSING PARENTHESES - Check every ( has matching )
3. ❌ NO MISMATCHED BRACKETS - Check every [ has matching ] (NOT ))
4. ❌ NO TRUNCATED STATEMENTS - Every statement must be complete
5. ❌ NO PLAIN ENGLISH TEXT - Only valid Python code (use # for comments)
6. ❌ ALL variables must be DEFINED before use
7. ❌ NO EMPTY ANIMATIONS - Check list comprehensions in self.play() are not empty
8. 🚨🚨🚨 NO 'def' STATEMENTS - NEVER define functions! Your code runs inside construct()
9. 🚨🚨🚨 NO 'class' STATEMENTS - NEVER define classes! Just write animation code directly

Example of MISMATCHED BRACKETS (FORBIDDEN):
```python
VGroup(*[Line(..., stroke_width=2))  # ← WRONG! Should be )] not ))
```

CORRECT:
```python
VGroup(*[Line(..., stroke_width=2)])  # ← Correct bracket matching
```

Example of INCOMPLETE CODE (FORBIDDEN):
```python
color=color, fill_opacity=0.3)  # ← MISSING Circle( at start!
```

CORRECT:
```python
circle = Circle(color=color, fill_opacity=0.3)  # ← Complete statement
```

FORBIDDEN PATTERNS - YOUR CODE WILL BE REJECTED IF YOU DO THIS:
❌ Boxes labeled "Knowledge", "Speed", "Interaction" with no meaning
❌ Empty shapes labeled "Input→Process→Output" 
❌ Abstract symbols (O(1), O(n), N^2) without showing what they mean
❌ Random shapes that don't match the narration

REQUIRED - YOUR CODE MUST DO THIS:
✅ Show ACTUAL operations from narration (arrays, loops, comparisons)
✅ Use REAL data and examples
✅ If narration says "comparing elements", show elements being compared
✅ Make the animation TEACH something, not just show labels

Create Manim visualization for: {concept}

NARRATION (READ CAREFULLY):
"{narration}"

{complexity_instruction}

🚨 MANDATORY PATTERN DETECTION 🚨
**IF concept contains "vs" OR nested relationships (AI/ML/DL, Parent/Child, etc.):**
→ YOU MUST use NESTED CIRCLES (concentric circles at ORIGIN)
→ DO NOT use scattered circles or random layouts
→ See "NESTED CIRCLES / VENN DIAGRAMS" section below for EXACT code

Example triggers: "AI vs ML vs DL", "Python vs Java", "Parent vs Child"
→ Use the nested circles pattern (largest → smallest, all at ORIGIN)

🎨 BE CREATIVE & ADVANCED - NOT SIMPLE!
- Use transformations, paths, curves, and motion
- Create visual stories, not just static shapes
- Use 6-12 objects for rich, professional animations
- Combine multiple animation techniques

ADVANCED TECHNIQUES TO USE:
1. **Transformations**: ReplacementTransform, Transform, MoveToTarget
2. **Paths & Curves**: ArcBetweenPoints, TracedPath (NO CurvedArrow - use Arrow instead!)
3. **Animations**: Indicate, Flash, Circumscribe, ShowPassingFlash, FadeIn, FadeOut
4. **Groups**: Use VGroup to organize and animate multiple objects together
5. **Effects**: FadeToColor, ScaleInPlace
6. **Colors**: Use gradients and color transitions
7. **Dynamic Creation**: Use loops to create patterns

CRITICAL RULES:

🚨 RULE 0 - MOST IMPORTANT - NO ABSTRACT DIAGRAMS! 🚨
❌ FORBIDDEN: Boxes labeled O(1), O(n), O(n^2) that look identical
❌ FORBIDDEN: Random "Input→Process→Output" with messy arrows  
❌ FORBIDDEN: Just writing N^2 without showing WHY
✅ REQUIRED: Show ACTUAL operations with REAL data (arrays, loops, comparisons)
✅ REQUIRED: If narration says "comparing elements", show elements being compared!

1. **TEXT STYLING - KEEP IT SIMPLE**:
   - **DO NOT use**: `font="Inter"`, `weight=MEDIUM`, `weight=SEMIBOLD` - these cause rendering issues!
   - **Color**: Use `color=WHITE` or `color="#E8E8E8"` on dark backgrounds
   - **Size**: font_size between 20 and 28 (headers: 28, labels: 24, content: 20) - MINIMUM 20!
   - Example: `Text("42", font_size=24, color=WHITE)`
   - Example: `Text("Sorted", font_size=20, color=GREEN_B)`
   - ❌ WRONG: `Text("Label", font="Inter", weight=MEDIUM, disable_ligatures=True)` - DON'T use these params!

🚨 RULE 0.5 - TEXT MUST NEVER OVERLAP! 🚨
❌ CRITICAL ERROR: Text labels overlapping each other
✅ REQUIRED: Every text label must be clearly readable with NO overlap

**How to prevent overlap:**
1. **Use .next_to() with different directions**:
   ```python
   # ❌ WRONG - all labels at UP will overlap!
   label1 = Text("React").next_to(box1, UP)
   label2 = Text("Ecosystem").next_to(box1, UP)  # Overlaps label1!
   
   # ✅ CORRECT - use different positions
   label1 = Text("React").move_to(box1.get_top() + UP*0.3)
   label2 = Text("Ecosystem").move_to(box1.get_bottom() + DOWN*0.3)
   ```

2. **Put labels INSIDE boxes when possible**:
   ```python
   # ✅ BEST - label inside the box
   box = Rectangle(width=3, height=1.5)
   label = Text("React Ecosystem", font_size=16).move_to(box)
   ```

3. **Use VGroup and arrange()**:
   ```python
   # ✅ CORRECT - arrange prevents overlap
   labels = VGroup(
       Text("Label 1"),
       Text("Label 2"),
       Text("Label 3")
   ).arrange(DOWN, buff=0.3)
   ```

4. **Spread labels around objects**:
   - If you have 4 boxes, put labels at: UP, DOWN, LEFT, RIGHT
   - NOT all at UP (they will overlap!)

🚨 CRITICAL - FONT SIZE IS A CODE PARAMETER, NOT TEXT CONTENT! 🚨
1. **TEXT STYLING - SIMPLE ONLY**: 
   - ✅ CORRECT: `Text("42", font_size=24, color=WHITE)`
   - ✅ CORRECT: `Text("Sorted", font_size=20, color=GREEN_B)`
   - ❌ WRONG: `Text("Label", font="Inter", weight=MEDIUM)` ← DON'T use font/weight params!
   - ❌ WRONG: `Text("Pythagorean Triplet (font_size=16)")` ← font_size is NOT text content!
   - ❌ WRONG: `Text("First Loop, font_size=16")` ← font_size is a PARAMETER, not text!
   - font_size is a CODE PARAMETER, NOT text to display on screen!
   - If you see "font_size=X" in examples, it's showing CODE syntax, not what to display!
2. **MATCH NARRATION - SHOW, DON'T JUST LABEL!**:
   - ❌ WRONG: Just showing circles labeled "1", "N", "N^2" (meaningless!)
   - ✅ CORRECT: Show ACTUAL examples from narration (arrays, loops, comparisons, operations)
   - **For Big O / Time Complexity**: Show actual operations (e.g., array with elements, nested loops comparing items)
   - **For Algorithms**: Show the actual steps (e.g., Binary Search: show array, highlight middle, move left/right)
   - **For Data Structures**: Show the structure in action (e.g., Stack: show push/pop with actual items)
   - **Use CONCRETE terms from narration** - If narration says "comparing each element", show elements being compared!
   - **NO abstract math symbols as main content** - "N^2" alone is meaningless, show WHY it's N^2!
2.5. **ALGORITHM ARRAYS - USE ONE CONSISTENT ARRAY!**:
   - ❌ WRONG: Creating new array for each step (wastes tokens, confusing!)
   - ✅ CORRECT: Create array ONCE, reuse it throughout the slide
   - **Match narration EXACTLY**: If narration says "midpoint", highlight midpoint. If narration says "target", show target.
   - **DON'T show elements not mentioned**: If narration talks about midpoint, don't add "Target" circle!
   - Example: `array = VGroup(...)` at start, then `array[mid].set_color(YELLOW)` for each step
   - This saves tokens and keeps animation consistent!
2.6. **ALGORITHMS - SHOW STEPS, NOT JUST RESULTS!**:
   - ❌ WRONG: Just showing final answer with fancy effects
   - ✅ CORRECT: Show initial state → process steps → final result
   - Show comparisons, decisions, swaps, movements - the ACTUAL algorithm logic
   - Use concrete values and labels, not placeholders
3. **SYNC DURATION**: Animation must match target_duration EXACTLY
4. **KEEP OBJECTS VISIBLE**: 
   - Create ALL objects FIRST, show together. NEVER FadeOut objects that should stay visible!
   - **SORTING**: ❌ NEVER FadeOut array elements! ✅ Only MOVE/SWAP. Array size must stay constant (6 elements → 6 elements)!
5. **NO UNNECESSARY ROTATIONS**: NEVER use .rotate() on arrows, circles, or diagrams unless narration specifically says "rotate". Rotation makes things confusing!
6. **NO GOING BACK**: Once an animation moves forward, don't animate backwards unless narration says so. Keep animations progressive!
6.5. **TEXT LABELS MUST STAY PUT**: Once you position a text label, NEVER animate it to move elsewhere! ❌ WRONG: `self.play(label.animate.move_to(other_object))` at end of slide. Labels should stay where they were created. If you need text at a new position, create a NEW Text object!
7. **STAY ON-SCREEN - CRITICAL! CODE WILL BE REJECTED IF VIOLATED**: 
   - Screen is 12 units wide × 6 units tall (SAFE ZONE)
   - **Circle radius**: MAX 1.5 (use 0.6-1.0 for most cases)
   - **Rectangle/Square**: MAX width=3.0, MAX height=3.0 (use 2.0 or less for safety)
   - **For 5+ objects horizontally**: Use radius=0.5, buff=0.5
   - **For nested circles**: radius=2.5 (outer), 1.5 (middle), 0.8 (inner)
   - **CRITICAL: Animations must stay on screen!**
     - When using `.shift(UP*X)` or `.shift(LEFT*X)`, keep X ≤ 3.5
     - Objects that move during animations can go off-screen - TEST POSITIONS!
     - Example: If object at `UP*2` shifts by `UP*3`, final position is `UP*5` (OFF-SCREEN!)
   - If objects go off-screen, they will be CUT OFF and invisible!
   - **Your code will be REJECTED if objects are too large or positioned off-screen**
7.5. **TITLES - FADE OUT AFTER SHOWING!**:
   - ❌ WRONG: Title stays on screen and overlaps with animations!
   - ✅ CORRECT: Show title for 2-3 seconds, then fade it out:
     ```python
     title = Text("Title", font_size=28).to_edge(UP)
     self.play(Write(title))
     self.wait(2)  # Show title for 2 seconds
     self.play(FadeOut(title))  # Fade out to make room for animations
     
     # Now create animations without title overlap
     box = Rectangle(...)
     self.play(Create(box))
     ```
   - **Rule**: ALWAYS fade out the title after 2-3 seconds to avoid overlapping with animations
8. **ONLY SHOW WHAT NARRATION MENTIONS**: Do NOT add extra arrows, feedback loops, or creative elements not in narration! If narration says "A connects to B", show ONE arrow from A to B. NO random arrows, NO loops back, NO extra connections!
9. **COMPARISONS**: Create column headers FIRST (at top), then boxes below. ❌ NEVER create boxes first → headers get hidden!
9.5. **LABEL POSITIONING - CRITICAL!**:
   - ❌ NEVER use `.move_to(box)` for labels → puts text ON TOP OF box!
   - ❌ NEVER use `.shift(UP*2)` if box is also at `UP*2` → text overlaps box!
   - ✅ ALWAYS use `.next_to(box, UP)` or `.next_to(box, LEFT)` for labels
   - ✅ For arrow labels: Use `.next_to(arrow, UP, buff=0.2)` to keep text ABOVE arrow
   - Example:
     ```python
     tcp_label = Text("TCP Packets").next_to(tcp_boxes, UP, buff=0.5)  # ✅ ABOVE boxes
     arrow_label = Text("Data").next_to(arrow, UP, buff=0.2)  # ✅ ABOVE arrow
     ```
10. **HORIZONTAL ARRANGEMENTS - BE CAREFUL!**: 
    - If arranging 5+ objects horizontally, use SMALL sizes (radius=0.5, buff=0.5)
    - Better: Arrange in 2 rows or use circular/arc layout
    - Example: 7 circles → 2 rows of 3-4 circles each, NOT one long row!
11. **ANIMATIONS vs OBJECTS - CRITICAL!**:
    - ❌ WRONG: VGroup(Circumscribe(obj), Indicate(obj))  # Animations cannot be in VGroup!
    - ❌ WRONG: pulse = ShowPassingFlash(...); pulses.add(pulse)  # Still adding animation to VGroup!
    - ✅ CORRECT: self.play(Circumscribe(obj), Indicate(obj))  # Use self.play() for animations
    - ✅ CORRECT: self.play(ShowPassingFlash(...))  # Animations go in self.play(), NOT VGroup!
    - **OBJECTS** go in VGroup: Circle, Square, Text, Arrow, Rectangle
    - **ANIMATIONS** go in self.play(): Create, FadeIn, Write, Circumscribe, Indicate, Transform, ShowPassingFlash
    - **NEVER store animations in variables to add to VGroup later!**
12. **ALGORITHM ACCURACY - CRITICAL FOR SORTING/SEARCHING!**:
    - For Binary Search: Midpoint = (left + right) // 2, highlight the CORRECT element
    - For Quick Sort: Pivot selection must be accurate
    - For Merge Sort: Show actual merge logic, not random combinations
    - For N-Queens/Backtracking: Use 0-indexed positions (0 to N-1). Show queen as ONE representation (Circle OR "Q", NOT both). Position labels must match actual queen positions!
    - **VERIFY YOUR LOGIC**: If narration mentions specific numbers/positions, use EXACTLY those numbers
    - Example: Array [1,3,4,6,7,8,9], midpoint is index 3 (element 6), NOT element 4!
13. **NO EXTERNAL IMAGES/SVG FILES - CRITICAL!**:
    - ❌ NEVER use SVGMobject(), ImageMobject(), or any external image files
    - ❌ NEVER reference .svg, .png, .jpg files (e.g., cloud.svg, tree.png)
    - ✅ ONLY use Manim's built-in shapes: Circle, Square, Rectangle, Triangle, Arrow, Line, Text
    - ✅ For clouds: Use Circle() or Ellipse() shapes
    - ✅ For trees: Use Rectangle() for trunk + Triangle() for leaves
    - **Reason**: External files don't exist on the server and will cause OSError!


❌ WRONG (Abstract symbols - meaningless!):
```python
# For "O(N^2) means quadratic time complexity"
circle1 = Circle().shift(LEFT*3)
label1 = Text("1", font_size=20, color=WHITE).move_to(circle1)

circle2 = Circle()
label2 = Text("N", font_size=20, color=WHITE).move_to(circle2)

circle3 = Circle().shift(RIGHT*3)
label3 = Text("N^2", font_size=20, color=WHITE).move_to(circle3)

# ← This shows NOTHING! User learns NOTHING!
```

✅ CORRECT (Show actual nested loop comparison):
```python
# For "O(N^2) means comparing each element with every other element"
# Show 4 elements in an array
elements = VGroup(*[
    Rectangle(width=1, height=1, color=BLUE).add(Text(str(i), font_size=16, color=WHITE))
    for i in [3, 7, 1, 9]
]).arrange(RIGHT, buff=0.2)

# Show nested loops by highlighting pairs
# First element (3) compares with all others
self.play(elements[0].animate.set_color(YELLOW))
for j in range(1, 4):
    self.play(elements[j].animate.set_color(RED))
    self.wait(0.3)
    self.play(elements[j].animate.set_color(BLUE))

# ← This SHOWS why it's N^2! User understands!
```

❌ WRONG (Storing animations in VGroup - CRASHES!):
```python
# For showing pulses on elements
pulses = VGroup()  # ← Creating VGroup for animations
for element in elements:
    pulse = ShowPassingFlash(element.copy())  # ← This is an ANIMATION!
    pulses.add(pulse)  # ← ERROR! Cannot add animations to VGroup!

# This will CRASH with: "Only values of type VMobject can be added"
```

✅ CORRECT (Play animations directly):
```python
# For showing pulses on elements
for element in elements:
    self.play(ShowPassingFlash(element.copy()), run_time=0.5)  # ← Direct play!

# OR play all at once:
self.play(*[ShowPassingFlash(elem.copy()) for elem in elements])
```

❌ WRONG (ShowPassingFlash with color parameter - CRASHES!):
```python
# ShowPassingFlash does NOT accept 'color' parameter!
self.play(ShowPassingFlash(arrow, color=YELLOW))  # ← ERROR!
self.play(ShowPassingFlash(line, color=RED, run_time=2.0))  # ← ERROR!
```

✅ CORRECT (ShowPassingFlash without color):
```python
# ShowPassingFlash only accepts: time_width, run_time
self.play(ShowPassingFlash(arrow))  # ← Correct!
self.play(ShowPassingFlash(line, time_width=0.5, run_time=1.5))  # ← Correct!

# To change color, modify the object BEFORE passing it:
arrow_copy = arrow.copy().set_color(YELLOW)
self.play(ShowPassingFlash(arrow_copy))  # ← Correct way to use color!
```


❌ WRONG (shows one circle at a time):
```python
circle1 = Circle(...)
self.play(Create(circle1))
self.play(FadeOut(circle1))  # ← Removes it! BAD!

circle2 = Circle(...)
self.play(Create(circle2))  # ← Only one visible at a time
```

✅ CORRECT (shows all circles together):
```python
# Create ALL objects first
circle1 = Circle(radius=3.0, ...)
circle2 = Circle(radius=2.0, ...)
circle3 = Circle(radius=1.0, ...)

# Position them
circle1.move_to(ORIGIN)
circle2.move_to(ORIGIN)
circle3.move_to(ORIGIN)

# Show ALL together (no FadeOut!)
self.play(Create(circle1), Create(circle2), Create(circle3))
# All 3 stay visible for rest of animation
```


🔴 LAYOUT CONSTRAINTS FOR COMPARISONS (A vs B):
**If concept involves comparing two things (Brain vs AI, Python vs Java, etc.):**
1. **UP-DOWN SPLIT**: Put "A" elements on TOP, "B" elements on BOTTOM (NOT left-right!)
2. **USE .shift()**: After creating all objects, use `.shift(UP * 2)` for A-side, `.shift(DOWN * 2)` for B-side
3. **ADD SECTION LABELS**: ALWAYS add Text labels "AI" and "Human" (or "A" and "B") to identify each section
4. **LIMIT OBJECTS**: Max 3-4 objects per side (total 6-8 objects) to fit on screen
5. **SMALL CIRCLES**: Use radius=0.6 for circles (NOT radius=1.0 or larger!)
6. **COMPACT LAYOUT**: Use `buff=1.0` for spacing (NOT buff=2.0 or larger!)

Example for "Brain vs AI":
```python
# TOP SECTION: Brain (3 nodes)
brain_nodes = VGroup(
    Circle(radius=0.6, color=BLUE),
    Circle(radius=0.6, color=BLUE),
    Circle(radius=0.6, color=BLUE)
)
brain_nodes.arrange(RIGHT, buff=1.0)
brain_nodes.shift(UP * 2)  # Move to top

# SECTION LABEL for Brain
brain_label = Text("Human", font_size=24, color=WHITE)
brain_label.next_to(brain_nodes, UP, buff=0.5)

# BOTTOM SECTION: AI (3 nodes)
ai_nodes = VGroup(
    Circle(radius=0.6, color=GREEN),
    Circle(radius=0.6, color=GREEN),
    Circle(radius=0.6, color=GREEN)
)
ai_nodes.arrange(RIGHT, buff=1.0)
ai_nodes.shift(DOWN * 2)  # Move to bottom

# SECTION LABEL for AI
ai_label = Text("AI", font_size=24, color=WHITE)
ai_label.next_to(ai_nodes, DOWN, buff=0.5)
```



🔴 CRITICAL: COMPARISON LAYOUTS (A vs B)
**When comparing two things (Traditional vs New, Before vs After, etc.):**
1. **MUST have EQUAL elements on BOTH sides**: If left has 3 boxes, right MUST have 3 boxes
2. **NEVER put two different comparison texts in the SAME box**: Each box has ONE text only!
3. **Use LEFT and RIGHT positioning**: Left side = first item, Right side = second item

❌ WRONG - TWO TEXTS IN SAME BOX (causes overlap!):
```python
box1 = RoundedRectangle(...)
text1 = Text("Fraud Prevention").move_to(box1)  # First text in box1
text2 = Text("Trust Continuity").move_to(box1)  # WRONG! Second text in SAME box!
```

✅ CORRECT - SEPARATE BOXES for each side:
```python
# LEFT SIDE (Option A)
left_box1 = RoundedRectangle(width=3, height=1, color=BLUE).shift(LEFT*3 + UP*1)
left_text1 = Text("Feature A", font_size=16).move_to(left_box1)

# RIGHT SIDE (Option B)  
right_box1 = RoundedRectangle(width=3, height=1, color=GREEN).shift(RIGHT*3 + UP*1)
right_text1 = Text("Feature B", font_size=16).move_to(right_box1)

# LEFT header
left_header = Text("Option A", font_size=20).next_to(left_box1, UP, buff=0.5)
# RIGHT header
right_header = Text("Option B", font_size=20).next_to(right_box1, UP, buff=0.5)
```

🔴 NESTED CIRCLES / VENN DIAGRAMS:
**For AI/ML/DL relationships or any nested concept:**
1. **CREATE ALL CIRCLES**: You MUST create ALL circles mentioned! If narration says "three nested circles", create 3 Circle() objects!
2. **CONCENTRIC CIRCLES**: Largest circle first, then medium, then smallest - ALL at ORIGIN
3. **LABEL POSITIONING**: Put labels OUTSIDE circles using .next_to(circle, direction, buff=0.5)
4. **NEVER put labels INSIDE overlapping circles** - they will be unreadable!
5. **Use different positions**: Outer circle label at TOP, middle at LEFT, inner at BOTTOM

Example for "AI ⊃ ML ⊃ DL" (MUST CREATE ALL 3 CIRCLES!):
```python
# Create circles from LARGEST to SMALLEST (ALL THREE!)
ai_circle = Circle(radius=3.0, color=BLUE, fill_opacity=0.1)
ml_circle = Circle(radius=2.0, color=GREEN, fill_opacity=0.15)
dl_circle = Circle(radius=1.0, color=RED, fill_opacity=0.2)

# Center all circles at same point
ai_circle.move_to(ORIGIN)
ml_circle.move_to(ORIGIN)
dl_circle.move_to(ORIGIN)

# Labels OUTSIDE circles (different directions to avoid overlap)
ai_label = Text("Artificial Intelligence", font_size=20, color=WHITE)
ai_label.next_to(ai_circle, UP, buff=0.3)  # TOP

ml_label = Text("Machine Learning", font_size=18, color=WHITE)
ml_label.next_to(ml_circle, LEFT, buff=0.3)  # LEFT

dl_label = Text("Deep Learning", font_size=16, color=WHITE)

dl_label.next_to(dl_circle, DOWN, buff=0.3)  # BOTTOM
```

VALID COLORS (ONLY use these):
- **FOR TEXT LABELS**: ONLY use WHITE (background is dark!)
- **FOR SHAPES**: RED, BLUE, GREEN, YELLOW, ORANGE, PURPLE, PINK, TEAL, GOLD, GRAY
- Colors WITH variants: ONLY Red, Blue, Green, Yellow have _A, _B, _C, _D, _E variants
  Example: RED_A, RED_B, BLUE_C, GREEN_D, YELLOW_E are valid
- ⚠️ ORANGE_A, PURPLE_A, PINK_A, TEAL_A, GOLD_A, GRAY_A DO NOT EXIST!
- ⚠️ Circle does NOT support gradient= parameter! Use fill_color= instead
  Example: Circle(fill_color=BLUE_C, fill_opacity=0.8) ✅
  Example: Rectangle(gradient=[BLUE, RED]) ✅ (Rectangle supports gradient)

🔴 CRITICAL POSITIONING RULES FOR ARRAYS/BOXES:
**NUMBERS MUST GO INSIDE BOXES, NOT BELOW THEM!**

CORRECT PATTERN for arrays/boxes with numbers:
```python
# Create boxes
boxes = VGroup(
    RoundedRectangle(...),
    RoundedRectangle(...),
    RoundedRectangle(...)
)

# Arrange boxes FIRST
boxes.arrange(RIGHT, buff=0.5)

# Put numbers INSIDE boxes using .move_to()
numbers = VGroup(
    Text("7", color=WHITE, font_size=16),
    Text("23", color=WHITE, font_size=16),
    Text("5", color=WHITE, font_size=16)
)

for num, box in zip(numbers, boxes):
    num.move_to(box.get_center())  # INSIDE the box!

# If you need to move boxes, group box+number together FIRST:
elements = VGroup(*[VGroup(box, num) for box, num in zip(boxes, numbers)])
# Now you can move elements[0], elements[1], etc. and numbers move with boxes!
```

❌ WRONG - DO NOT USE .next_to() for numbers in boxes:
```python
# This causes overlap when boxes move!
Text("7", color=WHITE).next_to(box, DOWN)  # ❌ WRONG!
```

❌ WRONG - DO NOT move boxes without moving numbers:
```python
# This separates numbers from boxes!
self.play(boxes[0].animate.shift(LEFT * 2))  # ❌ Numbers stay behind!
```

✅ CORRECT - Group box+number together before moving:
```python
elements = VGroup(*[VGroup(box, num) for box, num in zip(boxes, numbers)])
self.play(elements[0].animate.shift(LEFT * 2))  # ✅ Box and number move together!
```

🔴 CRITICAL: SAFE PATTERN FOR REARRANGING/PARTITIONING
When you need to rearrange elements (like sorting, partitioning), use this SAFE pattern:

```python
# Step 1: Show initial array
initial_elements = VGroup(*[VGroup(box, num) for box, num in zip(boxes, numbers)])
self.play(Create(initial_elements))

# Step 2: Fade out initial, create new arrangement
# DO NOT try to move elements - this causes overlaps!
self.play(FadeOut(initial_elements))

# Step 3: Create NEW elements in correct positions
left_group = VGroup(...)  # Elements less than pivot
pivot_element = VGroup(...)  # Pivot element  
right_group = VGroup(...)  # Elements greater than pivot

# Arrange them with proper spacing
all_groups = VGroup(left_group, pivot_element, right_group)
all_groups.arrange(RIGHT, buff=1.5)  # Use larger buff to avoid overlap!

# Step 4: Fade in new arrangement
self.play(FadeIn(all_groups))
```

❌ WRONG - Moving many elements causes overlaps:
```python
# This will cause overlaps and positioning errors!
self.play(
    elements[0].animate.shift(LEFT * 3),
    elements[1].animate.shift(LEFT * 1),
    elements[2].animate.shift(RIGHT * 2),
    # Too complex! Elements will overlap!
)
```

✅ For descriptive labels (like "Pivot", "Less", "Greater"):
- These can use `.next_to(box, DOWN)` 
- But ONLY after boxes are arranged!

🔴 CRITICAL RULES FOR NETWORK/GRAPH VISUALIZATIONS:
**For neural networks, flowcharts, graphs with nodes and connections:**

1. **NO ROTATIONS**: NEVER use `.rotate()` or `.rotate_about_origin()` - it makes everything confusing!
2. **STRAIGHT ARROWS ONLY**: Use `Arrow()` NOT `CurvedArrow()` - curved arrows create messy overlaps
3. **LARGE SPACING**: Use `buff=2.0` or more between layers to prevent overlap
4. **SLOW ANIMATIONS**: Use `run_time=2.0` minimum for creating connections - don't rush!
5. **LAYER BY LAYER**: Create one layer, then connections, then next layer - NOT all at once!
6. **TITLE FADE-OUT**: If you create a slide title with `title.to_edge(UP)`:
   - Display it for 3-5 seconds
   - Then fade it out: `self.play(FadeOut(title), run_time=0.5)`
   - This gives more vertical space for the main animation
   - Example: `self.wait(3.0)` then `self.play(FadeOut(title), run_time=0.5)`
7. **NO TEXT ANIMATIONS**: Once text labels are positioned, DO NOT animate them! Labels should stay static.


CORRECT PATTERN for neural network:
```python
# Layer 1: Input neurons
input_layer = VGroup(*[Circle(radius=0.3, color=BLUE) for _ in range(3)])
input_layer.arrange(DOWN, buff=1.0)  # Vertical spacing
input_layer.shift(LEFT * 4)  # Position on left

self.play(Create(input_layer), run_time=1.5)
        self.wait(0.5)

# Layer 2: Hidden neurons  
hidden_layer = VGroup(*[Circle(radius=0.3, color=GREEN) for _ in range(3)])
hidden_layer.arrange(DOWN, buff=1.0)
hidden_layer.shift(ORIGIN)  # Center

self.play(Create(hidden_layer), run_time=1.5)
        self.wait(0.5)
        
# Connections: Input to Hidden (SLOW!)
connections = VGroup()
for inp in input_layer:
    for hid in hidden_layer:
                arrow = Arrow(inp.get_right(), hid.get_left(), buff=0.1, stroke_width=2, color=GRAY)
        connections.add(arrow)

self.play(Create(connections), run_time=3.0)  # SLOW - 3 seconds!
self.wait(1.0)

# If you need to add more layers, FADE OUT old structure first!
self.play(FadeOut(VGroup(input_layer, hidden_layer, connections)), run_time=1.0)

# Then create new complete structure
# This prevents label overlap and confusion!
```

🔴 CRITICAL: If showing progressive network growth (adding layers):
- **ALWAYS FadeOut() the entire previous structure before showing new one**
- **DO NOT keep old circles and just add new ones** - this causes label overlap!
- **Create complete new network each time** with all layers and labels

❌ WRONG - DO NOT DO THIS:
```python
# This creates chaos!
network.rotate(PI/4)  # ❌ NO ROTATIONS!
CurvedArrow(start, end)  # ❌ Use Arrow() instead!
self.play(Create(all_connections), run_time=0.5)  # ❌ Too fast!

# Adding layers without removing old ones - causes overlap!
new_layer = VGroup(...)
self.play(Create(new_layer))  # ❌ Old labels still there!
```



CRITICAL SYNCHRONIZATION RULES:
⏱️ TARGET DURATION: {target_duration:.2f} seconds
- Calculate total animation time: sum of all run_time values + self.wait() calls
- Add final self.wait() to match target_duration EXACTLY
- Example calculation:
  ```
  total_time = 2.0 + 1.5 + 1.0 + 0.5  # Sum of all run_time values
  remaining = {target_duration:.2f} - total_time
  if remaining > 0:
      self.wait(remaining)
  ```
⚠️ IF YOU DON'T MATCH THE DURATION EXACTLY, THE NARRATION WILL BE CUT OFF!

CRITICAL: DO NOT define helper functions with 'def' - this causes indentation errors!
CRITICAL: DO NOT use 'return' statements - your code goes directly into construct()!
CRITICAL: DO NOT use these methods (they require helper functions):
  - UpdateFromFunc() ❌ (requires def, causes errors!)
  - UpdateFromAlphaFunc() ❌ (requires def, causes errors!)
  - always_redraw() ❌ (requires def, causes errors!)

🚨 CRITICAL - CREATE BEFORE USE (prevents UnboundLocalError):
❌ WRONG: `header.next_to(boxes, UP)` then `boxes = VGroup(...)` → ERROR! boxes not defined yet
✅ CORRECT: `boxes = VGroup(...)` then `header.next_to(boxes, UP)` → Works!

🚨 CRITICAL - BOX WITH TITLE + LIST (prevents overlap):
When showing a box with a title and list items:
❌ WRONG: Title inside box at top, list items with buff=0.3 → OVERLAP!
✅ CORRECT Option 1: Title ABOVE box using .next_to(box, UP)
✅ CORRECT Option 2: If title inside box, use buff >= 1.0 for first list item

🚨🚨🚨 CRITICAL - INVALID PARAMETERS WILL CRASH! 🚨🚨🚨
These parameters DO NOT EXIST in Manim. Using them will cause TypeError at runtime!

❌ Arc(start_color=...) - DOES NOT EXIST! Use Arc(color=...) instead
❌ Arc(end_color=...) - DOES NOT EXIST! Use Arc(color=...) instead  
❌ Circle(gradient=...) - DOES NOT EXIST! Use fill_color=... instead
❌ Text(str(x, font_size=...)) - WRONG! Use Text(str(x), font_size=...) instead

CRITICAL: Max radius/height/width values: radius <= 3, height <= 5, width <= 12
CRITICAL: ONLY use valid Manim methods! These methods DO NOT EXIST:
  - .arrange_in_circle() ❌ (use .arrange() or manual positioning)
  - .arrange_in_grid() ❌ (use .arrange())
  - .rotate_in_place() ❌ (use .rotate())
  - .scale_in_place() ❌ (use .scale())
  - SurroundingRoundedRectangle() ❌ (DOES NOT EXIST! Use SurroundingRectangle())
  - CurvedArrow() ❌ (DOES NOT EXIST! Use Arrow())
  - Arc(start_color=...) ❌ (Arc doesn't have start_color! Use color=... instead)

VISUALIZATION RULES:
- **LAYOUT**: 
  - ALWAYS use `.next_to()` or `.arrange()` for positioning.
  - NEVER use absolute coordinates like `[3, 4, 0]`.
  - Keep everything centered with `.move_to(ORIGIN)`.
- **⚠️ FORBIDDEN - NO EXTERNAL FILES**:
  - NEVER use SVGMobject() - external SVG files are not available!
  - NEVER use ImageMobject() - external images are not available!
  - ONLY use built-in Manim shapes: Circle, Square, Rectangle, Triangle, Text, Dot, Arrow, Line, etc.
  - If you need icons, use simple shapes (Circle for sun, Triangle for arrow, etc.).

🚨 PREVENT TEXT OVERLAPPING (CRITICAL):
- Use `buff=0.5+` in `.next_to()`, `buff=1.0+` in `.arrange()`
- For 5+ labels: font_size ≤ 24, spread in different directions (UP/DOWN/LEFT/RIGHT)
- For equations: Use multi-line Text() or shorter labels
- Example: `label.next_to(shape, DOWN, buff=0.5)` ✅
- DON'T: `label.next_to(shape, UP, buff=0.1)` ❌ (too small!)


🚨 NESTED/CONCENTRIC CIRCLES - CRITICAL RULE:
If you mention "nested circles", "concentric circles", "AI/ML/DL hierarchy", or "circles inside circles":
  ❌ WRONG: Creating ONE circle with multiple labels around it
  ✅ CORRECT: Create MULTIPLE Circle() objects with DIFFERENT radii, all at ORIGIN
  
Example for "AI contains ML contains DL" (3 nested circles):
```python
# Create 3 circles with DIFFERENT radii, all centered at ORIGIN
ai_circle = Circle(radius=2.5, color=RED, fill_opacity=0.1)
ml_circle = Circle(radius=1.7, color=BLUE, fill_opacity=0.1)
dl_circle = Circle(radius=1.0, color=GREEN, fill_opacity=0.1)

# All circles at same center (ORIGIN) - this creates nested effect
ai_circle.move_to(ORIGIN)
ml_circle.move_to(ORIGIN)
dl_circle.move_to(ORIGIN)

# Labels go OUTSIDE the circles, not inside
ai_label = Text("AI", font_size=20).next_to(ai_circle, UP, buff=0.3)
ml_label = Text("ML", font_size=20).next_to(ml_circle, RIGHT, buff=0.3)
dl_label = Text("DL", font_size=20).next_to(dl_circle, DOWN, buff=0.3)
```

CRITICAL: Use DIFFERENT radii (e.g., 2.5, 1.7, 1.0) - NOT the same radius!
CRITICAL: All circles must be at ORIGIN (or same center point)
CRITICAL: Labels go OUTSIDE circles using .next_to(), NOT inside


🧠 NEURAL NETWORK DIAGRAMS - ARROW LABEL POSITIONING:
- ❌ DON'T put all weight labels at arrow centers (they overlap!)\n- ✅ Position labels NEAR the SOURCE node: `label.next_to(arrow.get_start(), RIGHT, buff=0.2)`
- ✅ OR stagger vertically: top arrows get UP, bottom arrows get DOWN
- Example: For H1→O1 (top), use `.next_to(arrow, UP)`. For H3→O3 (bottom), use `.next_to(arrow, DOWN)`

📊 ARRAYS/VECTORS - NUMBERS MUST GO INSIDE BOXES:
- ❌ DON'T create boxes and numbers separately (numbers float around!)
- ✅ CREATE boxes WITH numbers inside using VGroup:
  ```python
  # CORRECT: Number INSIDE box
  box = Rectangle(width=0.5, height=0.8, color=BLUE)
  num = Text("0.4", font_size=16).move_to(box)  # ← INSIDE!
  cell = VGroup(box, num)
  ```
- ✅ For arrays: `VGroup(*[VGroup(Rectangle(...), Text(str(val), font_size=16).move_to(box)) for val in values])`
- This ensures numbers are ALWAYS inside their boxes, never floating!

🚨 MULTI-LINE TEXT INSIDE BOXES - CRITICAL:
If you need to put MULTIPLE lines of text inside a box (like equations, formulas, or multi-line labels):
  ❌ WRONG: Fixed box size with long text → TEXT OVERFLOWS AND OVERLAPS!
  ✅ CORRECT: Calculate box height based on text OR use smaller font

Example - WRONG (text overlaps):
```python
box = RoundedRectangle(width=4, height=2, color=YELLOW)  # ← Fixed height!
text1 = Text("Title")  # ← Default font too big!
text2 = Text("Line 1")
text3 = Text("Line 2")
content = VGroup(text1, text2, text3).arrange(DOWN, buff=0.1)
content.move_to(box)  # ← OVERLAPS! Text is taller than box!
```

Example - CORRECT (no overlap):
```python
# Option 1: Use smaller font (specify in code, not in text content!)
text1 = Text("Title", font_size=18)
text2 = Text("Line 1", font_size=18)
text3 = Text("Line 2", font_size=18)
content = VGroup(text1, text2, text3).arrange(DOWN, buff=0.3)  # ← More spacing!

# Option 2: Calculate box height from text
box_height = content.height + 0.8  # ← Add padding!
box = RoundedRectangle(width=4, height=box_height, color=YELLOW)
content.move_to(box)  # ← Perfect fit!
```

CRITICAL RULES for text in boxes:
1. Font size is a CODE parameter - NEVER write "font_size=X" in the actual text content!
2. If text has 2+ lines: Use smaller font in code (font_size=18 parameter)
3. ALWAYS add padding: box_height = text.height + 0.8
4. OR use .scale() to shrink text if it's too big

CORRECT LABEL EXAMPLE (Multiple circles with labels):
```python
# Circle 1 with label BELOW
circle1 = Circle(radius=1, color=BLUE)
label1 = Text("Input")
label1.next_to(circle1, DOWN)  # Label goes BELOW circle

# Circle 2 with label BELOW
circle2 = Circle(radius=1, color=GREEN)
label2 = Text("Process")
label2.next_to(circle2, DOWN)  # Label goes BELOW circle

# Circle 3 with label BELOW
circle3 = Circle(radius=1, color=RED)
label3 = Text("Output")
label3.next_to(circle3, DOWN)  # Label goes BELOW circle

# Arrange circles horizontally
circles = VGroup(circle1, circle2, circle3)
circles.arrange(RIGHT, buff=2)

# Arrange labels to match (they follow their circles)
labels = VGroup(label1, label2, label3)
```

WRONG LABEL EXAMPLE (DO NOT DO THIS):
```python
# ❌ BAD - All labels will overlap at center!
label1 = Text("Input").move_to(ORIGIN)
label2 = Text("Process").move_to(ORIGIN)
label3 = Text("Output").move_to(ORIGIN)
```

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

CRITICAL: **ALL Text() MUST have color=WHITE** - Background is DARK!
CRITICAL: EVERY Text object needs: Text("label", color=WHITE)
CRITICAL: NEVER use Text("label") without color - it defaults to BLACK (invisible!)
CRITICAL: NEVER use color=BLACK for text - it's invisible on dark background!

CORRECT TEXT EXAMPLES:
✅ label = Text("Input", color=WHITE)
✅ title = Text("Process Flow", color=WHITE).scale(0.8)
✅ note = Text("Step 1", color=WHITE).next_to(circle, DOWN)

WRONG TEXT EXAMPLES (DO NOT DO THIS):
❌ label = Text("Input")  # Missing color! Will be BLACK!
❌ label = Text("Input", color=BLACK)  # BLACK text on dark background!
❌ label = Text("Input").set_color(BLACK)  # Still BLACK!

CRITICAL: EVERY box MUST have a label! NO empty boxes!

🎨 CRITICAL: **TEXT STYLING (Inter Font)** 🎨
For EVERY Text() object, you MUST include these parameters:
- font="Inter" (ALWAYS - ensures consistent typography)
- weight=MEDIUM (for numbers and primary labels)
- weight=SEMIBOLD (ONLY for state labels like "Sorted", "Comparing", etc.)
- font_size=16-24 (based on text importance)
- color=WHITE or color="#E8E8E8" (off-white on dark background)

CORRECT EXAMPLES:
✅ Text("Top", font="Inter", weight=MEDIUM, font_size=20, color=WHITE)
✅ Text("Node 1", font="Inter", weight=MEDIUM, font_size=18, color="#E8E8E8")
✅ Text("Sorted", font="Inter", weight=SEMIBOLD, font_size=22, color=WHITE)

WRONG (don't use custom fonts - causes rendering issues):
❌ Text("Top", font="Inter", weight=MEDIUM, disable_ligatures=True)

CRITICAL: Return code with NO indentation (flush left).
CRITICAL: DO NOT generate "class Scene" or "def construct". ONLY generate the animation commands (self.play, etc).
CRITICAL: DO NOT start with self.wait(). Start animating IMMEDIATELY!
CRITICAL: KEEP ALL OBJECTS ON-SCREEN! Screen height is ~7 units, width is ~14 units.
CRITICAL: If stacking items vertically (UP), limit to MAX 5 items. Use .arrange() or .scale() to fit on screen.
CRITICAL: Max radius/height/width values: radius <= 3, height <= 5, width <= 12


Generate code:"""
    
    # Single attempt only - regeneration was found to make code WORSE
    max_attempts = 1
    ai_code = ""
    
    # COST OPTIMIZATION: Split the prompt into static (cacheable) and dynamic parts
    # The original 'prompt' variable (defined above) contains both static rules and dynamic vars
    # We split it to cache the static part (8500 tokens) and send only dynamic part (300 tokens)
    
    # The prompt has this structure:
    # [Static rules] + "Create Manim visualization for: {concept}\n\nNARRATION...\n{narration}\n\n{complexity_instruction}" + [More static rules]
    
    # Find where the dynamic part starts and ends
    dynamic_marker_start = f"Create Manim visualization for: {concept}"
    dynamic_marker_end = complexity_instruction
    
    # Split the prompt
    if dynamic_marker_start in prompt and dynamic_marker_end in prompt:
        # Extract static part before dynamic vars
        static_part1 = prompt.split(dynamic_marker_start)[0]
        
        # Extract static part after dynamic vars
        temp = prompt.split(dynamic_marker_end, 1)
        if len(temp) > 1:
            static_part2 = temp[1]
        else:
            static_part2 = ""
        
        # Combine static parts for caching
        static_system_prompt = static_part1.strip() + "\n\n" + static_part2.strip()
        
        # CRITICAL FIX: Remove target_duration from static prompt (it's dynamic!)
        # Replace ALL {target_duration} placeholders with a generic placeholder
        static_system_prompt = static_system_prompt.replace(f"{target_duration:.2f}", "TARGET_DURATION")
        static_system_prompt = static_system_prompt.replace(f"{target_duration:.1f}", "TARGET_DURATION")
        
        # Dynamic user message (not cached)
        dynamic_user_message = f"""Create Manim visualization for: {concept}

NARRATION (READ CAREFULLY):
\"{narration}\"

{complexity_instruction}

⏱️ TARGET DURATION: {target_duration:.2f} seconds
- Your animation must match this duration EXACTLY
- Calculate total time and add final self.wait() if needed"""
    else:
        # Fallback: use entire prompt as system (less efficient but works)
        static_system_prompt = prompt
        dynamic_user_message = ""
    
    # DEBUG: Save static prompt to file to check what's changing
    import hashlib
    prompt_hash = hashlib.md5(static_system_prompt.encode()).hexdigest()
    debug_file = f".debug_static_prompt_{concept[:20].replace(' ', '_')}_{prompt_hash[:8]}.txt"
    with open(debug_file, 'w') as f:
        f.write(f"=== STATIC PROMPT DEBUG ===\n")
        f.write(f"Concept: {concept}\n")
        f.write(f"Hash: {prompt_hash}\n")
        f.write(f"Length: {len(static_system_prompt)} chars\n")
        f.write(f"\n=== CONTENT ===\n")
        f.write(static_system_prompt)
    print(f"      🔍 DEBUG: Static prompt hash: {prompt_hash[:8]} (saved to {debug_file})")
    
    for attempt in range(max_attempts):
        # Use Claude for Manim code generation with PROMPT CACHING
        print(f"      💰 Calling Claude API (attempt {attempt + 1}/{max_attempts})...")
        
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=3000,
            temperature=0.2 + (attempt * 0.1),
            timeout=120.0,  # 2 minute timeout to prevent hanging
            system=[
                {
                    "type": "text",
                    "text": static_system_prompt,  # 8500 tokens - CACHED!
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": dynamic_user_message  # 300 tokens - NOT cached
                }
            ]
        )
        
        # COST TRACKING - Log token usage
        usage = response.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        # Claude Sonnet 4.5 pricing: $3/M input, $15/M output
        input_cost = (input_tokens / 1_000_000) * 3
        output_cost = (output_tokens / 1_000_000) * 15
        total_cost = input_cost + output_cost
        
        print(f"      📊 Tokens: {input_tokens} in + {output_tokens} out")
        print(f"      💵 Cost: ${total_cost:.4f} (${input_cost:.4f} + ${output_cost:.4f})")
        
        ai_code = response.content[0].text.strip()
        
        # Import re for regex operations
        import re
        
        # CRITICAL: Strip markdown code blocks (Claude sometimes wraps code in ```python```)
        if ai_code.startswith('```'):
            # Remove opening ```python or ```
            ai_code = re.sub(r'^```(?:python)?\s*\n', '', ai_code)
            # Remove closing ```
            ai_code = re.sub(r'\n```\s*$', '', ai_code)
            ai_code = ai_code.strip()
            print(f"      🔧 Stripped markdown code blocks")
        
        # DEBUGGING: Save first attempt to file for inspection
        if attempt == 0:
            # Use concept name (sanitized) for filename
            safe_concept = "".join(c if c.isalnum() else "_" for c in concept[:30])
            debug_file = f".temp_ai_first_attempt_{safe_concept}.py"
            with open(debug_file, 'w') as f:
                f.write(ai_code)
            print(f"      💾 Saved first AI attempt to: {debug_file}")
        
        # Remove initial wait if present
        ai_code = re.sub(r'^\s*self\.wait\([^)]+\)\s*\n', '', ai_code)
        
        # Check if AI generated a class/def despite instructions
        if "def construct(self):" in ai_code:
            parts = ai_code.split("def construct(self):")
            if len(parts) > 1:
                ai_code = parts[1]
        
        # SIMPLE VALIDATION - check for critical issues
        issues = []
        
        # CHECK 0: PYTHON SYNTAX VALIDATION (catches .set_color= and other syntax errors)
        # This is the ROOT FIX - catch syntax errors BEFORE Manim runs!
        try:
            import ast
            # Try to parse the code as Python
            ast.parse(ai_code)
        except SyntaxError as e:
            # Syntax error found - add to issues with helpful message
            error_line = e.lineno if e.lineno else "unknown"
            error_msg = e.msg if e.msg else "syntax error"
            issues.append(f"Python syntax error at line {error_line}: {error_msg}")
            
            # Try to give specific guidance for common errors
            if "cannot contain assignment" in error_msg or "expression cannot contain assignment" in error_msg:
                # This catches .set_color=RED, .scale=2, etc.
                issues.append("HINT: Use .set_color(RED) NOT .set_color=RED (parentheses required for method calls!)")
            elif "positional argument follows keyword argument" in error_msg:
                # This catches func(kwarg=val, positional_arg) - CRITICAL ERROR
                issues.append("CRITICAL: Positional arguments must come BEFORE keyword arguments!")
                issues.append("EXAMPLE: func(pos_arg, kwarg=val) NOT func(kwarg=val, pos_arg)")
            elif "invalid syntax" in error_msg:
                # Generic syntax error - check for common patterns
                if ".set_color=" in ai_code or ".set_fill=" in ai_code or ".set_stroke=" in ai_code:
                    issues.append("HINT: Method calls need parentheses: .set_color(RED) not .set_color=RED")
                elif ".scale=" in ai_code or ".shift=" in ai_code:
                    issues.append("HINT: Method calls need parentheses: .scale(2) not .scale=2")
            
                    print(f"      Returning None for fallback\n")
                    return None
            else:
                print(f"      ❌ No auto-fix available for this error")
                print(f"      Returning None for fallback\n")
                return None
        
        # Check 0.9: Empty list comprehensions in self.play() (causes "no animations" error)
        # Pattern: self.play(*[... for x in empty_list])
        if "self.play(" in ai_code and "for " in ai_code and " in " in ai_code:
            # Check for list comprehensions that might be empty
            lines = ai_code.split('\n')
            for line_num, line in enumerate(lines, 1):
                if "self.play(" in line and "*[" in line and "for " in line:
                    # This might be a list comprehension that could be empty
                    issues.append(f"⚠️  Line {line_num}: self.play() with list comprehension might be empty - add 'if len(list) > 0' check")
        
        # Check 0.95: Comparison queries should use nested circles
        # Pattern: "vs" in concept or narration
        if " vs " in concept.lower() or " vs " in narration.lower():
            # Check if code creates nested circles (concentric at ORIGIN)
            has_nested_circles = False
            if "Circle(" in ai_code and ".move_to(ORIGIN)" in ai_code:
                # Count how many circles are moved to ORIGIN
                origin_count = ai_code.count(".move_to(ORIGIN)")
                circle_count = ai_code.count("Circle(")
                if origin_count >= 2 and circle_count >= 2:
                    has_nested_circles = True
            
            if not has_nested_circles:
                issues.append("⚠️  COMPARISON QUERY: Should use NESTED CIRCLES (concentric at ORIGIN) for 'vs' comparisons!")
                issues.append("   Example: ai_circle.move_to(ORIGIN), ml_circle.move_to(ORIGIN), dl_circle.move_to(ORIGIN)")
        
        # Check 0.96: Text labels should use different directions to avoid overlap
        # Pattern: Multiple .next_to() calls with same direction
        if ".next_to(" in ai_code:
            lines = ai_code.split('\n')
            directions_used = []
            for line in lines:
                if ".next_to(" in line:
                    # Extract direction (UP, DOWN, LEFT, RIGHT, etc.)
                    for direction in ["UP", "DOWN", "LEFT", "RIGHT", "UL", "UR", "DL", "DR"]:
                        if direction in line:
                            directions_used.append(direction)
                            break
            
            # Check if same direction used multiple times
            # Increased threshold from 2 to 4 - small overlaps are usually fine
            if len(directions_used) > 1:
                from collections import Counter
                direction_counts = Counter(directions_used)
                for direction, count in direction_counts.items():
                    if count > 4:  # Same direction used more than 4 times - likely a real problem
                        issues.append(f"⚠️  TEXT OVERLAP: Direction '{direction}' used {count} times - spread labels using UP, DOWN, LEFT, RIGHT")
        
        # Check 1: No class/def statements
        if 'class ' in ai_code or 'def ' in ai_code:
            issues.append("CRITICAL: Contains class/def statements - code will break! Remove all helper functions.")
        
        # Check 1.5: No 3D objects (we only support 2D Scene)
        forbidden_3d_objects = ['Sphere(', 'Cube(', 'Cone(', 'Cylinder(', 'ThreeDScene', 'ThreeDAxes', 'Surface(', 'ParametricSurface(']
        for obj in forbidden_3d_objects:
            if obj in ai_code:
                issues.append(f"CRITICAL: 3D object '{obj}' detected! Use 2D alternatives: Circle for Sphere, Square for Cube, Triangle for Cone")
                break
        
        # Check 1.6: No updater functions (causes errors without proper setup)
        forbidden_updaters = ['add_updater(', 'remove_updater(', 'UpdateFromFunc(', 'UpdateFromAlphaFunc(', 'always_redraw(']
        for updater in forbidden_updaters:
            if updater in ai_code:
                issues.append(f"CRITICAL: Updater '{updater}' detected! Updaters require helper functions which are forbidden. Use simple animations instead.")
                break
        
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
        
        # Check 3: No invalid Manim methods
        # List of methods that DON'T exist in Manim but AI often tries to use
        
        # Check 3.4: GENERIC parameter validation - catch ANY invalid parameter
        # Define valid parameters for common Manim objects
        valid_params = {
            'Arc': ['radius', 'start_angle', 'angle', 'color', 'stroke_width', 'fill_opacity', 'fill_color'],
            'Circle': ['radius', 'color', 'fill_opacity', 'fill_color', 'stroke_width', 'stroke_color'],
            'Rectangle': ['width', 'height', 'color', 'fill_opacity', 'fill_color', 'stroke_width'],
            'Square': ['side_length', 'color', 'fill_opacity', 'fill_color', 'stroke_width'],
            'Text': ['text', 'font_size', 'color', 'font', 'weight', 'slant'],
            'Arrow': ['start', 'end', 'color', 'stroke_width', 'buff', 'max_tip_length_to_length_ratio'],
            'Line': ['start', 'end', 'color', 'stroke_width'],
        }
        
        # Common INVALID parameters that AI often tries to use
        invalid_params = {
            'Arc': ['start_color', 'end_color', 'gradient'],
            'Circle': ['gradient', 'border_color', 'border_width'],
            'Text': [],  # font_size should be OUTSIDE str()
        }
        
        import re
        for obj_type, invalid_list in invalid_params.items():
            for invalid_param in invalid_list:
                # Check if this invalid parameter is used
                pattern = rf'{obj_type}\([^)]*{invalid_param}='
                if re.search(pattern, ai_code):
                    print(f"\n      🔧 INVALID PARAMETER: {obj_type}({invalid_param}=...)")
                    print(f"      This parameter doesn't exist for {obj_type}!")
                    
                    # Try to suggest fix
                    if invalid_param in ['start_color', 'end_color', 'gradient']:
                        print(f"      Suggestion: Use 'color=' instead\n")
                        # Auto-fix
                        ai_code = re.sub(rf'{invalid_param}=', 'color=', ai_code)
                        print(f"      ✅ AUTO-FIXED: Replaced {invalid_param}= with color=\n")
                        
                        # CRITICAL: Validate the fix worked
                        try:
                            ast.parse(ai_code)
                            # Fix successful - don't add to issues
                            print(f"      ✅ Auto-fix validated - code is now syntactically correct\n")
                        except SyntaxError:
                            # Fix failed - add to issues
                            issues.append(f"CRITICAL: {obj_type}({invalid_param}=...) is invalid!")
                    else:
                        issues.append(f"CRITICAL: {obj_type}({invalid_param}=...) is invalid!")
        
        # Check 3.5: Arc(start_color=...) is INVALID - auto-fix to color=
        if "Arc(" in ai_code and "start_color=" in ai_code:
            print(f"\n      🔧 INVALID PARAMETER DETECTED: Arc(start_color=...)")
            print(f"      Arc() doesn't have 'start_color' parameter!")
            print(f"      Auto-fixing: start_color= → color=\n")
            
            # Auto-fix: Replace start_color= with color= in Arc() calls
            fixed_code = ai_code.replace("start_color=", "color=")
            
            # Validate fix
            try:
                ast.parse(fixed_code)
                print(f"      ✅ AUTO-FIX SUCCESSFUL - Replaced start_color= with color=\n")
                ai_code = fixed_code
            except SyntaxError:
                print(f"      ❌ AUTO-FIX FAILED - Syntax still invalid\n")
                issues.append("CRITICAL: Arc(start_color=...) is invalid! Use Arc(color=...) instead")
        
        # Check 3.6: Invalid Manim methods
        invalid_methods = [
            '.arrange_in_circle(',  # Doesn't exist - use .arrange() or manual positioning
            '.arrange_in_grid(',    # Doesn't exist in older Manim - use .arrange()
            '.rotate_in_place(',    # Doesn't exist - use .rotate()
            '.scale_in_place(',     # Doesn't exist - use .scale()
            'SurroundingRoundedRectangle(',  # Doesn't exist - use SurroundingRectangle() instead
        ]
        
        found_invalid_method = False
        for method in invalid_methods:
            if method in ai_code:
                issues.append(f"Invalid Manim method: {method}")
                found_invalid_method = True
                break
        
        # AUTO-FIX: Replace invalid methods if found
        if found_invalid_method:
            print(f"      🔧 Attempting auto-fix for invalid methods...")
            fixed_code = ai_code
            fixed_code = fixed_code.replace('.arrange_in_circle(', '.arrange(')
            fixed_code = fixed_code.replace('.arrange_in_grid(', '.arrange(')
            fixed_code = fixed_code.replace('.rotate_in_place(', '.rotate(')
            fixed_code = fixed_code.replace('.scale_in_place(', '.scale(')
            fixed_code = fixed_code.replace('SurroundingRoundedRectangle(', 'SurroundingRectangle(')
            
            # Re-check if methods are still present
            still_has_invalid = False
            for method in invalid_methods:
                if method in fixed_code:
                    still_has_invalid = True
                    break
            
            if not still_has_invalid:
                print(f"      ✅ AUTO-FIX SUCCESSFUL - Invalid methods replaced")
                ai_code = fixed_code
                # Remove the invalid method error from issues
                issues = [i for i in issues if "Invalid Manim method:" not in i]
            else:
                print(f"      ❌ AUTO-FIX FAILED - Invalid methods still present")
        
        
        # Check for common syntax errors
        # Pattern: str(label, font_size=...) - font_size should be in Text(), not str()
        if 'str(' in ai_code and 'font_size' in ai_code:
            # Check if font_size appears inside str()
            import re
            bad_str_pattern = re.search(r'str\([^)]*font_size', ai_code)
            if bad_str_pattern:
                issues.append("SYNTAX ERROR: font_size is inside str() - should be Text(str(label), font_size=16) NOT Text(str(label, font_size=16))")
        

        # Check 4: Must have animations
        if 'self.play(' not in ai_code:
            issues.append("No animations found")
        
        # Check 5: NO EMPTY BOXES - Every Rectangle/Square must have a Text label
        rectangle_count = ai_code.count('Rectangle(') + ai_code.count('Square(')
        text_count = ai_code.count('Text(')
        
        # If we have boxes but fewer labels, some boxes are empty
        if rectangle_count > 0 and text_count < rectangle_count:
            issues.append(f"CRITICAL: {rectangle_count} boxes but only {text_count} labels - EVERY box needs a label inside! NO empty boxes allowed")
        
        # Check 5: Text label overlap detection
        # Pattern 1: Text().move_to() - Check if it's positioning inside boxes (VALID) or free positioning (BAD)
        # VALID: Text(...).move_to(box) or Text(...).move_to(box.get_center()) - centering text inside a box
        # INVALID: Text(...).move_to(UP*2) when multiple labels use the SAME position
        text_move_to_pattern = re.findall(r'Text\([^)]+\)\.move_to\(([^)]+)\)', ai_code)
        
        # Filter out valid cases:
        # 1. Text inside boxes: .move_to(box), .move_to(box.get_center()), etc.
        # 2. Each label on a different box/position
        valid_box_patterns = ['get_center()', 'get_top()', 'get_bottom()', 'get_left()', 'get_right()']
        
        # Extract position targets (e.g., "box", "UP*2", "circle.get_center()")
        position_targets = []
        for pos in text_move_to_pattern:
            # Check if it's a valid box positioning
            is_valid_box = any(pattern in pos for pattern in valid_box_patterns)
            # Check if it's a simple variable name (likely a box: "box", "circle", "rect1")
            is_simple_var = re.match(r'^[a-zA-Z_]\w*$', pos.strip())
            
            if not (is_valid_box or is_simple_var):
                # It's a coordinate-based positioning (UP*2, LEFT*3, etc.)
                position_targets.append(pos.strip())
        
        # Only flag if multiple labels use the SAME coordinate position
        from collections import Counter
        position_counts = Counter(position_targets)
        duplicate_positions = [pos for pos, count in position_counts.items() if count >= 2]
        
        if duplicate_positions:
            issues.append(f"CRITICAL: Multiple Text labels using .move_to({duplicate_positions[0]}) - will overlap! Use different positions or .next_to() instead")

        
        # REMOVED: General move_to check - too many false positives
        
        # Pattern 3: Labels positioned BEFORE .arrange() - causes overlap!
        # Check if code has pattern: label.next_to(circle) ... VGroup(...).arrange()
        if '.arrange(' in ai_code and '.next_to(' in ai_code:
            lines = ai_code.split('\n')
            next_to_lines = [i for i, line in enumerate(lines) if '.next_to(' in line and 'Text(' in line]
            arrange_lines = [i for i, line in enumerate(lines) if '.arrange(' in line]
            
            # If labels are positioned before arrange, they will overlap
            if next_to_lines and arrange_lines:
                if any(next_to_line < arrange_line for next_to_line in next_to_lines for arrange_line in arrange_lines):
                    issues.append("CRITICAL: Labels positioned BEFORE .arrange() - will overlap! Position labels AFTER arranging objects")
        
        # Pattern 3b: AGGRESSIVE CHECK - Multiple labels using same direction
        # If 3+ labels all use .next_to(..., DOWN) or .next_to(..., UP), they'll likely overlap
        next_to_down_count = ai_code.count('.next_to(') and ai_code.count(', DOWN)')
        next_to_up_count = ai_code.count('.next_to(') and ai_code.count(', UP)')
        next_to_left_count = ai_code.count('.next_to(') and ai_code.count(', LEFT)')
        next_to_right_count = ai_code.count('.next_to(') and ai_code.count(', RIGHT)')
        
        text_count = ai_code.count('Text(')
        if text_count >= 3:
            if next_to_down_count >= 3 or next_to_up_count >= 3:
                issues.append("CRITICAL: Multiple labels using same direction (DOWN/UP) - will overlap! Use different positions or arrange objects first")
            if next_to_left_count >= 3 or next_to_right_count >= 3:
                issues.append("CRITICAL: Multiple labels using same direction (LEFT/RIGHT) - will overlap! Use different positions or arrange objects first")
        
        # Pattern 4: Check if labels are positioned at same coordinates
        text_positions = re.findall(r'Text\([^)]+\)\.move_to\(([^)]+)\)', ai_code)
        if len(text_positions) != len(set(text_positions)):
            issues.append("Potential text overlap: multiple labels at similar positions")
        
        # Check for unnecessary arcs/curves (unless narration mentions connections)
        narration_lower = narration.lower()
        has_arc_keywords = any(word in narration_lower for word in ['connection', 'flow', 'path', 'relationship', 'link', 'connect'])
        if not has_arc_keywords:
            if 'CurvedArrow' in ai_code or 'ArcBetweenPoints' in ai_code or 'Arc(' in ai_code:
                issues.append("⚠️ Unnecessary arcs/curves detected - narration doesn't mention connections or flows. Remove them!")
        
        # Check 5a: Text inside boxes - font size validation
        # Pattern: Text inside Rectangle/Square should use smaller font
        # RELAXED: Allow font_size up to 28 for numbers/short text in boxes
        text_in_box_pattern = re.findall(r'Text\([^)]*font_size\s*=\s*(\d+)[^)]*\)\.move_to\([^)]*get_center', ai_code)
        for font_size in text_in_box_pattern:
            if int(font_size) > 28:
                issues.append(f"Text inside box has font_size={font_size} (too large, will overflow). Use font_size=12-28 for text inside shapes")
            
        # Check 5b: CRITICAL - Animations in VGroup/Group
        # Pattern: VGroup(Circumscribe(...), ...) - WRONG! Animations cannot be in VGroup
        animation_names = ['Circumscribe', 'Indicate', 'FadeIn', 'FadeOut', 'Create', 'Write', 
                          'Transform', 'ReplacementTransform', 'ShowPassingFlash', 'GrowFromCenter']
        
        for anim_name in animation_names:
            # Check if animation is inside VGroup() or Group()
            if re.search(rf'VGroup\([^)]*{anim_name}\(', ai_code) or re.search(rf'Group\([^)]*{anim_name}\(', ai_code):
                issues.append(f"CRITICAL: {anim_name}() is an ANIMATION! Cannot add to VGroup/Group. Use self.play({anim_name}(...)) instead.")
                break
            
        # Check 6: Detect potential off-screen animations

        # Pattern: Many items stacked vertically (likely to go off-screen)
        vgroup_count = ai_code.count('VGroup')
        next_to_up_count = ai_code.count('.next_to(') and ai_code.count('UP')
        
        # Check 4: No external file dependencies (SVG, images, etc.)
        if 'SVGMobject' in ai_code:
            issues.append("Uses SVGMobject (external files not available)")

        if 'ImageMobject' in ai_code:
            issues.append("Uses ImageMobject (external files not available)")
        if '.svg' in ai_code or '.png' in ai_code or '.jpg' in ai_code:
            issues.append("References external image files")
        
        # Pattern: Text labels at same position (potential overlap)
        text_positions = re.findall(r'Text\([^)]+\)\.move_to\(([^)]+)\)', ai_code)
        if len(text_positions) != len(set(text_positions)):
            issues.append("Potential text overlap: multiple labels at similar positions")
        
        # Check 7: BLACK text on dark background (CRITICAL - invisible!)
        # Pattern: Text(..., color=BLACK) or Text(...).set_color(BLACK)
        if 'color=BLACK' in ai_code or 'color=Black' in ai_code or '.set_color(BLACK)' in ai_code:
            issues.append("CRITICAL: BLACK text on dark background - invisible! Use WHITE instead")
        
        # Check for any Text without explicit color (defaults to BLACK in Manim)
        # FIXED: Check each Text() call individually on the same line
        lines_with_text = [line for line in ai_code.split('\n') if 'Text(' in line]
        text_without_color_count = 0
        for line in lines_with_text:
            # Check if this line has Text() but no color= parameter
            if 'Text(' in line and 'color=' not in line:
                text_without_color_count += 1
        
        if text_without_color_count > 0:
            issues.append(f"Text objects missing color ({text_without_color_count} found) - will default to BLACK (invisible!). Add color=WHITE")
        
        # Check 8: NARRATION VALIDATION - Does animation match the narration?
        # Be LESS strict - allow creative freedom
        # Extract key terms from narration (nouns, verbs, concepts)
        narration_words = set(narration.lower().split())
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'as', 'it', 'its', 'their', 'them', 'they', 'we', 'you', 'your', 'our'}
        key_narration_terms = narration_words - common_words
        
        # Extract Text labels from code
        code_labels = re.findall(r'Text\(["\']([^"\']+)["\']', ai_code)
        code_labels_lower = set(label.lower() for label in code_labels)
        
        # RELAXED: Only check if animation is completely unrelated (< 5% match)
        # Allow AI creative freedom to visualize concepts
        if len(key_narration_terms) > 0:
            matching_terms = sum(1 for term in key_narration_terms if any(term in label for label in code_labels_lower))
            match_ratio = matching_terms / len(key_narration_terms)
            # Only reject if COMPLETELY unrelated
            if match_ratio < 0.05 and len(code_labels) > 0:
                issues.append(f"Animation seems unrelated to narration. Consider using some terms from: {', '.join(list(key_narration_terms)[:3])}")
        
        # Check 8b: SPECIFIC OBJECT COUNT VALIDATION
        # If narration says "three circles", code should have 3 Circle() objects
        narration_lower = narration.lower()
        
        # Check for number + object type patterns
        object_patterns = {
            'circle': r'Circle\(',
            'square': r'Square\(',
            'rectangle': r'Rectangle\(',
            'line': r'Line\(',
            'arrow': r'Arrow\(',
            'dot': r'Dot\(',
        }
        
        number_words = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10
        }
        
        for obj_name, obj_pattern in object_patterns.items():
            # Check if narration mentions a number + this object type
            for num_word, num_value in number_words.items():
                # Pattern: "three circles" or "3 circles"
                if f'{num_word} {obj_name}' in narration_lower:
                    # Count how many of this object are in the code
                    actual_count = len(re.findall(obj_pattern, ai_code))
                    if actual_count != num_value:
                        issues.append(f"CRITICAL: Narration says '{num_word} {obj_name}s' but code has {actual_count} {obj_name.capitalize()}() objects! Create exactly {num_value} {obj_name.capitalize()}() objects.")
                        break
        
        # Check 8b2: NESTED/CONCENTRIC CIRCLES VALIDATION
        # Special check for "nested circles", "concentric circles", "AI ML DL", etc.
        nested_keywords = ['nested', 'concentric', 'inside', 'contains', 'subset', 'superset']
        if any(keyword in narration_lower for keyword in nested_keywords) and 'circle' in narration_lower:
            # Count circles in code
            circle_count = len(re.findall(r'Circle\(', ai_code))
            if circle_count < 2:
                issues.append(f"CRITICAL: Narration mentions nested/concentric circles but code has only {circle_count} Circle()! Create at least 3 Circle() objects with different radii (e.g., radius=3.0, radius=2.0, radius=1.0) all at ORIGIN.")
            
            # CRITICAL: Check that circles have DIFFERENT radii!
            radii = re.findall(r'Circle\(radius=([\d.]+)', ai_code)
            if len(radii) >= 2:
                unique_radii = set(radii)
                if len(unique_radii) == 1:
                    issues.append(f"CRITICAL: All {len(radii)} circles have the SAME radius ({radii[0]})! For nested circles, use DIFFERENT radii: radius=3.0 (outer), radius=2.0 (middle), radius=1.0 (inner).")
        
        # Special check for "AI vs ML vs DL" or similar hierarchical concepts
        if ('ai' in narration_lower and 'ml' in narration_lower and 'dl' in narration_lower) or \
           ('artificial intelligence' in narration_lower and 'machine learning' in narration_lower):
            circle_count = len(re.findall(r'Circle\(', ai_code))
            if circle_count < 3:
                issues.append(f"CRITICAL: AI/ML/DL hierarchy requires 3 nested circles but code has only {circle_count}! Create: Circle(radius=3.0), Circle(radius=2.0), Circle(radius=1.0) all at ORIGIN with labels OUTSIDE.")
            
            # CRITICAL: Check that circles have DIFFERENT radii!
            radii = re.findall(r'Circle\(radius=([\d.]+)', ai_code)
            if len(radii) >= 3:
                unique_radii = set(radii)
                if len(unique_radii) == 1:
                    issues.append(f"CRITICAL: All 3 AI/ML/DL circles have the SAME radius ({radii[0]})! They will overlap completely! Use: Circle(radius=2.5), Circle(radius=1.5), Circle(radius=0.8) with DIFFERENT sizes.")
        

        # Check 8c: ORIENTATION VALIDATION
        # If narration says "vertical line", code should have vertical Line (not horizontal)
        # If narration says "horizontal line", code should have horizontal Line
        if 'vertical line' in narration_lower:
            # Check if there's a Line with vertical orientation (start/end have same x, different y)
            vertical_line_pattern = r'Line\([^)]*start\s*=\s*\[[^,]+,\s*[^,]+[^)]*end\s*=\s*\[[^,]+,\s*[^,]+[^)]*\)'
            # This is complex to validate perfectly, so just warn if we see "horizontal" patterns
            if 'start=[-' in ai_code and ', 0,' in ai_code and 'end=[' in ai_code:
                # Likely horizontal line pattern: start=[-7, 0, 0], end=[7, 0, 0]
                issues.append("CRITICAL: Narration says 'vertical line' but code appears to create horizontal Line! Use start=[0, -3, 0], end=[0, 3, 0] for vertical.")
        
        if 'horizontal line' in narration_lower:
            # Similar check for horizontal
            if 'start=[0,' in ai_code and 'end=[0,' in ai_code:
                issues.append("CRITICAL: Narration says 'horizontal line' but code appears to create vertical Line! Use start=[-7, 0, 0], end=[7, 0, 0] for horizontal.")
        
        # Check 9: PREMATURE FADEOUT - Objects removed before they should be visible
        # If narration mentions "3 circles", "nested", "together", etc., check for FadeOut
        multi_object_keywords = ['circles', 'nested', 'together', 'all', 'three', 'multiple', 'both']
        if any(keyword in narration.lower() for keyword in multi_object_keywords):
            # Count Create() and FadeOut() calls
            create_count = ai_code.count('Create(')
            fadeout_count = ai_code.count('FadeOut(')
            
            # If we're creating multiple objects but fading them out, that's wrong
            if create_count >= 2 and fadeout_count >= create_count - 1:
                issues.append("CRITICAL: FadeOut() used too early! Narration mentions multiple objects that should stay visible together. Create ALL objects first, then show them together WITHOUT FadeOut.")
        
        # Check 9b: GENERAL FADEOUT CHECK - Don't fade out objects unless narration says so
        # BUT ALLOW FadeOut(title) - titles SHOULD fade out after displaying
        # Count FadeOut calls EXCLUDING title fadeouts
        fadeout_title_count = len(re.findall(r'FadeOut\s*\(\s*title\s*\)', ai_code, re.IGNORECASE))
        fadeout_total = ai_code.count('FadeOut(')
        fadeout_non_title = fadeout_total - fadeout_title_count
        
        if fadeout_non_title > 0:
            # Check if narration mentions disappearing, fading, removing, etc.
            disappear_keywords = ['disappear', 'fade', 'remove', 'vanish', 'hide', 'goes away', 'is removed']
            if not any(keyword in narration.lower() for keyword in disappear_keywords):
                # FadeOut used on non-title objects but narration doesn't mention objects disappearing!
                # Just a WARNING, not CRITICAL - let the code render
                issues.append(f"⚠️ FadeOut() used {fadeout_non_title} time(s) on non-title objects - consider keeping objects visible")
        
        # Check 9c: SORTING ANIMATIONS - NEVER FadeOut array elements!
        sorting_keywords = ['sort', 'sorting', 'insertion', 'quick', 'merge', 'bubble', 'array', 'swap']
        if any(keyword in narration.lower() for keyword in sorting_keywords):
            # Check for FadeOut on boxes/squares/array elements
            if 'FadeOut(' in ai_code and ('box' in ai_code.lower() or 'square' in ai_code.lower() or 'array' in ai_code.lower()):
                issues.append("CRITICAL: SORTING ANIMATION ERROR! Never use FadeOut() on array elements during sorting! Array size must stay constant. Only MOVE/SWAP elements, don't remove them!")
        
        # Check 9: SIMPLICITY - Too many text objects = confusing
        if len(code_labels) > 8:
            issues.append(f"Too many text labels ({len(code_labels)}) - keep it simple! Max 8 labels per slide")
        
        # Check 10: CLARITY - Very long text labels are hard to read
        long_labels = [label for label in code_labels if len(label) > 40]
        if long_labels:
            issues.append(f"Text labels too long (max 40 chars): {long_labels[0][:30]}...")
        
        # Check 11: FONT SIZE - Text too large causes overlap
        font_sizes = re.findall(r'font_size\s*=\s*(\d+)', ai_code)
        if font_sizes:
            large_fonts = [int(size) for size in font_sizes if int(size) > 36]
            if large_fonts:
                issues.append(f"Font size too large ({max(large_fonts)}) - max 36! Use font_size=24-36")
        # If no font_size specified, warn (defaults to 48 which is huge!)
        elif text_count > 0:
            issues.append("No font_size specified - will default to 48 (too large!). Add font_size=24-36 to all Text()")
        
        # Pattern: Very large radius/height/width values - WARNING only
        import re
        large_values = re.findall(r'(?:radius|height|width)\s*=\s*(\d+\.?\d*)', ai_code)
        if large_values:
            max_value = max(float(val) for val in large_values)
            if max_value > 4.0:
                issues.append(f"Large objects detected ({max_value:.1f}) - will be auto-scaled to fit screen")
        
        # Check for objects positioned too far from center - WARNING only
        shift_values = re.findall(r'\.shift\([^)]*?([+-]?\d+\.?\d*)\s*\*\s*(?:UP|DOWN|LEFT|RIGHT)', ai_code)
        if shift_values:
            max_shift = max(float(val) for val in shift_values)
            if max_shift > 5.0:
                issues.append(f"Large shifts detected ({max_shift:.1f}) - objects may go off-screen during animation")
        
        # Check for Arrow endpoints going off-screen
        # Pattern: Arrow(start=[x, y, z], end=[x, y, z])
        # FIXED: Use simpler regex that matches ALL coordinate pairs
        arrow_coords = re.findall(r'(?:start|end)=\[([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)', ai_code)
        if arrow_coords:
            for x, y in arrow_coords:
                x_val, y_val = float(x), float(y)
                # Screen bounds: x: -6 to 6, y: -3 to 3
                if abs(x_val) > 6 or abs(y_val) > 3.5:
                    issues.append(f"CRITICAL: Arrow endpoint ({x_val}, {y_val}) goes OFF-SCREEN! Keep arrows within x=±6, y=±3")
        
        # Check for .move_to() with coordinates going off-screen
        # Pattern: .move_to([x, y, z]) or .move_to(np.array([x, y, z]))
        moveto_coords = re.findall(r'\.move_to\(\[([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)', ai_code)
        if moveto_coords:
            for x, y in moveto_coords:
                x_val, y_val = float(x), float(y)
                if abs(x_val) > 6 or abs(y_val) > 3.5:
                    issues.append(f"CRITICAL: .move_to([{x_val}, {y_val}]) goes OFF-SCREEN! Keep within x=±6, y=±3")
        
        # Check for text labels overlapping with boxes or behind arrows
        # Pattern 1: label = Text(...).move_to(box)  (chained)
        # Pattern 2: label.move_to(box)  (separate line)
        # This causes labels to appear ON TOP OF boxes or BEHIND arrows
        
        # Pattern 1: Detect chained .move_to() calls on Text objects
        # Example: tcp_label = Text("TCP").move_to(tcp_box)
        chained_pattern = r'(\w+)\s*=\s*Text\([^)]+\)\.move_to\((\w+)\)'
        chained_matches = re.findall(chained_pattern, ai_code)
        
        for text_obj, target in chained_matches:
            # Check if target is a Rectangle/Square/Circle (likely a box)
            if re.search(rf'{target}\s*=\s*(?:Rectangle|Square|Circle)\(', ai_code):
                issues.append(f"CRITICAL: Label '{text_obj}' uses .move_to({target}) which puts text ON TOP OF the box! Use .next_to({target}, UP) or .next_to({target}, LEFT) instead!")
        
        # Pattern 2: Detect separate .move_to() calls
        # Example: tcp_label.move_to(tcp_box)
        separate_pattern = r'(\w+)\.move_to\((\w+)\)'
        separate_matches = re.findall(separate_pattern, ai_code)
        
        for text_obj, target in separate_matches:
            # Check if text_obj is a Text object AND target is a box
            is_text = re.search(rf'{text_obj}\s*=\s*Text\(', ai_code)
            is_box = re.search(rf'{target}\s*=\s*(?:Rectangle|Square|Circle)\(', ai_code)
            
            if is_text and is_box:
                issues.append(f"CRITICAL: Label '{text_obj}' uses .move_to({target}) which puts text ON TOP OF the box! Use .next_to({target}, UP) or .next_to({target}, LEFT) instead!")
        
        
        # Check for invalid .arrange(rows=, cols=) syntax
        # This causes TypeError: Mobject.next_to() got unexpected keyword argument 'rows'
        # MUST be caught BEFORE test render to avoid wasting API calls
        if '.arrange(rows=' in ai_code or '.arrange(cols=' in ai_code:
            issues.append(f"CRITICAL: Invalid .arrange(rows=, cols=) syntax! Use .arrange_in_grid(rows=, cols=) instead!")
        
        # If no issues, we're done!
        if not issues:
            print(f"      ✅ Code validated (attempt {attempt + 1}/{max_attempts})")
            break
        
        # DEBUG: Print detailed breakdown of issues
        print(f"\n      {'='*60}")
        print(f"      🔍 VALIDATION FAILED - Attempt {attempt + 1}/{max_attempts}")
        print(f"      {'='*60}")
        print(f"      Total issues found: {len(issues)}")
        for i, issue in enumerate(issues, 1):
            print(f"      Issue {i}: {issue}")
        print(f"      {'='*60}\n")
        
        # Handle validation issues (max_attempts=1, so no regeneration)
        if issues:
            # CRITICAL: Check for syntax errors or invalid methods
            has_syntax_error = any("syntax error" in issue.lower() for issue in issues)
            has_invalid_method = any("invalid manim method" in issue.lower() for issue in issues)
            has_critical_error = any("critical:" in issue.lower() for issue in issues)
            
            # Check for specific critical issues that should ALWAYS reject
            has_3d_object = any("3d object" in issue.lower() for issue in issues)
            has_updater = any("updater" in issue.lower() for issue in issues)
            has_def_statement = any("class/def" in issue.lower() for issue in issues)
            
            if has_syntax_error or has_invalid_method:
                print(f"      ❌ CRITICAL: Code has SYNTAX ERRORS or INVALID METHODS!")
                print(f"      ❌ REJECTING CODE - Will use fallback animation\n")
                return None  # Reject broken code
            elif has_3d_object or has_updater:
                print(f"      ❌ CRITICAL: Code has 3D OBJECTS or UPDATERS!")
                print(f"      ❌ REJECTING CODE - These will crash at runtime\n")
                return None  # Reject broken code
            elif has_def_statement:
                print(f"      ⚠️  Code has DEF statements - will be fixed in post-processing")
                # Don't reject - post-processing will strip def statements
            elif has_critical_error:
                print(f"      ⚠️  Code has CRITICAL issues but may still render")
                print(f"      ⚠️  Proceeding with caution...\n")
            else:
                print(f"      ⚠️  Code has minor issues - proceeding\n")
        else:
            print(f"      ✅ Code validated")


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
    
    # STEP 4: Extract function body - BULLETPROOF
    # Manually find and skip the "def dummy():" line, then dedent everything
    lines = fixed.split('\n')
    
    # Find the line with "def dummy():"
    start_idx = 0
    for i, line in enumerate(lines):
        if 'def dummy():' in line:
            start_idx = i + 1  # Start from next line
            break
    
    # Get all lines after def
    body_lines = lines[start_idx:]
    
    # Join and use textwrap.dedent to remove common indentation
    body = '\n'.join(body_lines)
    ai_code = textwrap.dedent(body).strip()

    # FIX 0: Add missing imports if code uses them
    # Check if code uses random but doesn't import it
    if 'random.' in ai_code or 'random(' in ai_code:
        if 'import random' not in ai_code:
            ai_code = 'import random\n' + ai_code
    
    # Check if code uses numpy/np but doesn't import it
    if ('np.' in ai_code or 'numpy.' in ai_code) and 'import numpy' not in ai_code:
        ai_code = 'import numpy as np\n' + ai_code
    
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
    
    # CRITICAL SYNC FIX: Smart animation timing to match narration
    # BUT: Skip if this is a per-slide animation (already timed correctly)
    if not skip_global_sync:
        # Calculate TOTAL animation time FIRST, then adjust
        print(f"\n🕐 SYNC ANALYSIS:")
        print(f"   Target duration (narration): {target_duration:.2f}s")
        
        # Calculate current total time
        run_times = re.findall(r"run_time\s*=\s*(\d+\.?\d*)", ai_code)
        total_anim_time = sum(float(t) for t in run_times)
        
        wait_times = re.findall(r"self\.wait\((\d+\.?\d*)\)", ai_code)
        total_wait_time = sum(float(t) for t in wait_times)
        
        total_plays = ai_code.count("self.play(")
        explicit_runs = len(run_times)
        default_runs = total_plays - explicit_runs
        total_default_time = default_runs * 1.0
        
        total_video_time = total_anim_time + total_wait_time + total_default_time
        
        print(f"   Current video time: {total_video_time:.2f}s")
        print(f"   Animation run_times: {total_anim_time:.2f}s ({len(run_times)} animations)")
        print(f"   Wait times: {total_wait_time:.2f}s ({len(wait_times)} waits)")
        print(f"   Default plays: {total_default_time:.2f}s ({default_runs} plays)")
        
        # STRATEGY: Scale animations to match narration
        if total_video_time > target_duration + 2.0:  # Video too long (2s tolerance)
            # Need to SPEED UP animations
            scale_factor = (target_duration - total_wait_time - 1.0) / (total_anim_time + total_default_time)
            scale_factor = max(scale_factor, 0.5)  # Don't go below 0.5x (too fast)
            scale_factor = min(scale_factor, 1.0)  # Don't increase (we're already too long)
            
            print(f"   ⚠️  Video too long! Scaling animations by {scale_factor:.2f}x")
            
            # Scale all run_times
            def scale_runtime(match):
                original = float(match.group(1))
                scaled = original * scale_factor
                scaled = max(scaled, 0.3)  # Minimum 0.3s per animation
                return f'run_time={scaled:.2f}'
            
            ai_code = re.sub(r'run_time=(\d+\.?\d*)', scale_runtime, ai_code)
            
            # Recalculate
            run_times_after = re.findall(r"run_time\s*=\s*(\d+\.?\d*)", ai_code)
            total_anim_time_after = sum(float(t) for t in run_times_after)
            total_video_time = total_anim_time_after + total_wait_time + (default_runs * scale_factor)
            
            print(f"   ✅ Scaled video time: {total_video_time:.2f}s")
            
        elif total_video_time < target_duration - 2.0:  # Video too short (2s tolerance)
            # Need to ADD final wait
            final_wait_needed = target_duration - total_video_time
            print(f"   ⚠️  Video too short! Adding final wait: {final_wait_needed:.2f}s")
            
            # Detect indentation
            lines = ai_code.split('\n')
            last_non_empty_line = None
            for line in reversed(lines):
                if line.strip():
                    last_non_empty_line = line
                    break
            
            if last_non_empty_line:
                leading_spaces = len(last_non_empty_line) - len(last_non_empty_line.lstrip())
                indent = ' ' * leading_spaces
            else:
                indent = '        '
            
            ai_code += f"\n{indent}self.wait({final_wait_needed:.2f})  # SYNC FIX: Match narration"
            
            print(f"   ✅ Final video time: {target_duration:.2f}s")
        else:
            print(f"   ✅ Video time matches narration (within 2s tolerance)")
    else:
        print(f"\n🕐 SYNC: Skipping global sync (per-slide timing preserved)")
    

    # FIX 1.5: Convert Rectangle to RoundedRectangle for Kodnest style
    # DISABLED: This regex breaks multi-line Rectangle() calls
    # def convert_to_rounded(match):
    #     """Convert Rectangle(...) to RoundedRectangle(corner_radius=0.15, ...)"""
    #     params = match.group(1)
    #     if 'corner_radius' in params:
    #         return match.group(0)
    #     return f'RoundedRectangle(corner_radius=0.15, {params})'
    # 
    # ai_code = re.sub(r'(?<!Rounded)Rectangle\(([^)]*)\)', convert_to_rounded, ai_code)
    
    # FIX 1.6: Remove gradient from RoundedRectangle (not supported!)
    # RoundedRectangle doesn't support gradient= parameter, only fill_color=
    ai_code = re.sub(r'(RoundedRectangle\([^)]*),\s*gradient=\[[^\]]+\]', r'\1', ai_code)
    
    # FIX 1.7: Arc doesn't have start_color parameter - replace with color
    # Arc(start_color=RED) → Arc(color=RED)
    ai_code = re.sub(r'\bArc\(([^)]*)\bstart_color=', r'Arc(\1color=', ai_code)


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
        ai_code = ai_code.replace(f'[{invalid},', f'[{valid},')

    # FIX 2.7: Prevent multiple Text labels at same position (CRITICAL OVERLAP FIX!)
    # Pattern: Multiple Text().move_to(ORIGIN) or Text().move_to(circle.get_center())
    # This causes massive text overlap - we need to detect and fix it
    
    # Count how many Text objects use .move_to with same target
    move_to_origins = ai_code.count('Text(') and ai_code.count('.move_to(ORIGIN)')
    move_to_centers = ai_code.count('.move_to(') - move_to_origins
    
    # If we have 3+ Text objects all using .move_to(ORIGIN), this will cause overlap
    if move_to_origins >= 3:
        # Replace all but first with .next_to with different directions
        lines = ai_code.split('\n')
        origin_count = 0
        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'UP + LEFT', 'UP + RIGHT', 'DOWN + LEFT', 'DOWN + RIGHT']
        
        for i, line in enumerate(lines):
            if 'Text(' in line and '.move_to(ORIGIN)' in line:
                if origin_count > 0:  # Keep first one, fix the rest
                    # Replace .move_to(ORIGIN) with .shift(direction)
                    direction = directions[min(origin_count - 1, len(directions) - 1)]
                    lines[i] = line.replace('.move_to(ORIGIN)', f'.shift({direction} * 2)')
                origin_count += 1
        
        ai_code = '\n'.join(lines)
    
    # FIX 2.8: Remove rotations from network visualizations (causes confusion!)
    # DISABLED: .rotate() is valid - only remove if it causes actual problems
    # If we see errors related to rotations, we can enable this selectively
    # ai_code = re.sub(r'\.rotate\([^)]*\)', '', ai_code)
    # ai_code = re.sub(r'\.rotate_about_origin\([^)]*\)', '', ai_code)
    
    # FIX 2.9: Replace CurvedArrow with Arrow (prevents messy overlaps)
    # CurvedArrow doesn't exist in Manim - replace it
    ai_code = ai_code.replace('CurvedArrow(', 'Arrow(')
    # Keep ArcBetweenPoints - it's valid and creates nice curves
    
    # FIX 2.10: Remove invalid 'angle' parameter from Arrow() ONLY
    # Arrow() doesn't support angle parameter, but be specific to Arrow only
    ai_code = re.sub(r'Arrow\(([^)]*),\s*angle\s*=\s*[^,)]+', r'Arrow(\1', ai_code)
    ai_code = re.sub(r'Arrow\(angle\s*=\s*[^,)]+,\s*([^)]*)\)', r'Arrow(\1)', ai_code)
    
    # FIX 2.11: Fix method call syntax errors (.set_color= should be .set_color())
    # Pattern: .method_name=VALUE should be .method_name(VALUE)
    # Common mistakes: .set_color=RED, .set_fill=BLUE, .set_stroke=WHITE
    # SAFETY: These regex patterns can break code if they match incorrectly
    code_before_method_fixes = ai_code
    ai_code = re.sub(r'\.set_color\s*=\s*([A-Z_]+)', r'.set_color(\1)', ai_code)
    ai_code = re.sub(r'\.set_fill\s*=\s*([A-Z_]+)', r'.set_fill(\1)', ai_code)
    ai_code = re.sub(r'\.set_stroke\s*=\s*([A-Z_]+)', r'.set_stroke(\1)', ai_code)
    ai_code = re.sub(r'\.set_opacity\s*=\s*([0-9.]+)', r'.set_opacity(\1)', ai_code)
    ai_code = re.sub(r'\.scale\s*=\s*([0-9.]+)', r'.scale(\1)', ai_code)
    ai_code = re.sub(r'\.shift\s*=\s*([A-Z_*0-9. +\-]+)', r'.shift(\1)', ai_code)
    
    # FIX 2.12: Fix chained method calls with = instead of ()
    # Pattern: .copy().set_color=PINK should be .copy().set_color(PINK)
    ai_code = re.sub(r'\.copy\(\)\.set_color\s*=\s*([A-Z_]+)', r'.copy().set_color(\1)', ai_code)
    ai_code = re.sub(r'\.copy\(\)\.set_fill\s*=\s*([A-Z_]+)', r'.copy().set_fill(\1)', ai_code)
    
    # SAFETY CHECK: Validate syntax after method fixes
    try:
        import ast
        ast.parse(ai_code)
    except SyntaxError:
        print(f"   ⚠️  REVERTING: Method = fixes broke syntax, keeping original")
        ai_code = code_before_method_fixes
    
    # FIX 2.13: Replace invalid Manim methods that don't exist
    ai_code = ai_code.replace('.arrange_in_circle(', '.arrange(')
    ai_code = ai_code.replace('.arrange_in_grid(', '.arrange(')
    ai_code = ai_code.replace('.rotate_in_place(', '.rotate(')
    ai_code = ai_code.replace('.scale_in_place(', '.scale(')
    ai_code = ai_code.replace('SurroundingRoundedRectangle(', 'SurroundingRectangle(')
    
    # FIX 2.14: Replace invalid Manim color constants
    # Claude sometimes generates ORANGE_C, BLUE_C, etc. which don't exist
    # Replace with valid color constants
    ai_code = ai_code.replace('ORANGE_C', 'ORANGE')
    ai_code = ai_code.replace('BLUE_C', 'BLUE')
    ai_code = ai_code.replace('GREEN_C', 'GREEN')
    ai_code = ai_code.replace('RED_C', 'RED')
    ai_code = ai_code.replace('YELLOW_C', 'YELLOW')
    ai_code = ai_code.replace('PURPLE_C', 'PURPLE')
    ai_code = ai_code.replace('PINK_C', 'PINK')
    ai_code = ai_code.replace('TEAL_C', 'TEAL')
    ai_code = ai_code.replace('GOLD_C', 'GOLD')
    
    # FIX 2.5: Replace deprecated Manim methods

    # These methods were renamed in newer Manim versions
    ai_code = ai_code.replace('ShowCreation(', 'Create(')
    ai_code = ai_code.replace('GrowFromCenter(', 'Create(')
    ai_code = ai_code.replace('FadeInFrom(', 'FadeIn(')
    ai_code = ai_code.replace('FadeOutAndShift(', 'FadeOut(')
    ai_code = ai_code.replace('FadeInFromDown(', 'FadeIn(')
    ai_code = ai_code.replace('FadeInFromUp(', 'FadeIn(')
    ai_code = ai_code.replace('FadeInFromLeft(', 'FadeIn(')
    ai_code = ai_code.replace('FadeInFromRight(', 'FadeIn(')
    
    # FIX 2.6: Fix str(label, font_size=...) or str(label, weight=...) syntax error
    # Pattern: Text(str(num, font_size=16), ...) or Text(str(val, weight=MEDIUM), ...)
    # Should be: Text(str(num), font_size=16, ...) or Text(str(val), weight=MEDIUM, ...)
    import re
    # SAFETY: Save code before risky str() fixes
    code_before_str_fixes = ai_code
    # Find all str(..., font_size=...) and remove font_size from inside str()
    ai_code = re.sub(
        r'str\(([^,)]+),\s*font_size=\d+\)',  # Match str(something, font_size=N)
        r'str(\1)',  # Replace with str(something)
        ai_code
    )
    # FIX 2.6a: Also fix str(..., weight=...) - Claude puts weight inside str() too!
    # Pattern: Text(str(val, weight=MEDIUM), ...) → Text(str(val), weight=MEDIUM, ...)
    ai_code = re.sub(
        r'str\(([^,)]+),\s*weight=(\w+)\)',  # Match str(something, weight=X)
        r'str(\1)',  # Replace with str(something) - weight will be in Text() already
        ai_code
    )
    
    # FIX 2.6c: Also fix str(..., font="...") - Claude puts font inside str() too!
    # Pattern: Text(str(val, font="Inter"), ...) → Text(str(val), font="Inter", ...)
    ai_code = re.sub(
        r'str\(([^,)]+),\s*font="[^"]+"\)',  # Match str(something, font="X")
        r'str(\1)',  # Replace with str(something) - font will be in Text() already
        ai_code
    )
    
    # FIX 2.6b: Fix missing closing parenthesis in str() before font_size/font/color/weight
    # Pattern: Text(str(input_numbers[i * 4 + j] font_size=18, color=WHITE)
    # Should be: Text(str(input_numbers[i * 4 + j]), font_size=18, color=WHITE)
    # This happens when Claude forgets to close str() before adding other parameters
    # CRITICAL: Handle multi-line cases where there's a newline between ] and font_size
    ai_code = re.sub(
        r'str\(([^\)]+?)\s*\n?\s*(font_size|font|color|weight)=',  # Match str(something\n font=
        r'str(\1), \2=',  # Replace with str(something), font=
        ai_code,
        flags=re.MULTILINE | re.DOTALL
    )
    
    # SAFETY CHECK: Validate syntax after str() fixes
    try:
        import ast
        ast.parse(ai_code)
    except SyntaxError:
        print(f"   ⚠️  REVERTING: str() fixes broke syntax, keeping original")
        ai_code = code_before_str_fixes


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

    # FIX 2B: Final Safety Scaling (Content Overflow Fix - AFTER first animation)
    # Objects are now in self.mobjects, so we can scale them
    # IMPORTANT: Only scale if objects are SIGNIFICANTLY off-screen (not just slightly)
    final_scaling_code = """
# SAFETY SCALING: Only scale if objects are SIGNIFICANTLY off-screen
all_mobs = Group(*self.mobjects)

if len(all_mobs) > 0:
    # Calculate scale factor based on ACTUAL BOUNDS
    scale_factor = 1.0
    
    # Check if objects go SIGNIFICANTLY beyond screen bounds
    # Use stricter thresholds - only scale if really needed
    # Screen safe zone: x: [-5.5, 5.5], y: [-2.8, 2.8] (excluding title area)
    top = all_mobs.get_top()[1]
    bottom = all_mobs.get_bottom()[1]
    left = all_mobs.get_left()[0]
    right = all_mobs.get_right()[0]
    
    # Scale if objects go beyond visible screen bounds
    # More aggressive thresholds to catch more off-screen cases
    if top > 3.0:  # Above safe zone (title area is at ~3.5)
        scale_factor = min(scale_factor, (2.8 / top) * 0.95)
    if bottom < -3.0:  # Below safe zone
        scale_factor = min(scale_factor, (2.8 / abs(bottom)) * 0.95)
    if right > 6.0:  # Beyond right edge
        scale_factor = min(scale_factor, (5.5 / right) * 0.95)
    if left < -6.0:  # Beyond left edge
        scale_factor = min(scale_factor, (5.5 / abs(left)) * 0.95)
    
    # ALSO check total dimensions - more aggressive thresholds
    if all_mobs.width > 12:  # Screen width ~14, safe is 12
        scale_factor = min(scale_factor, 11 / all_mobs.width * 0.9)
    if all_mobs.height > 6:  # Screen height ~8, safe is 6
        scale_factor = min(scale_factor, 5.5 / all_mobs.height * 0.9)
    
    # Apply scaling ONLY if significantly needed (scale_factor < 0.95)
    # Scale about CENTER to prevent shifting
    # CRITICAL: Apply INSTANTLY (no animation) to prevent text from "flying"
    if scale_factor < 0.95:
        center = all_mobs.get_center()
        all_mobs.scale(scale_factor, about_point=center)  # Instant, no animation!
    
    # Center if still off-screen after scaling (minimal shifting)
    shift_needed = False
    shift_vector = np.array([0.0, 0.0, 0.0])  # Use floats to avoid dtype casting error
    
    # Recalculate bounds after scaling
    if scale_factor < 0.95:
        top = all_mobs.get_top()[1]
        bottom = all_mobs.get_bottom()[1]
        left = all_mobs.get_left()[0]
        right = all_mobs.get_right()[0]
    
    # Only shift if STILL significantly off-screen
    if top > 3.2:
        shift_vector += DOWN * (top - 3.0)
        shift_needed = True
    if bottom < -3.2:
        shift_vector += UP * ((-3.0) - bottom)
        shift_needed = True
    if left < -6.5:
        shift_vector += RIGHT * ((-6.0) - left)
        shift_needed = True
    if right > 6.5:
        shift_vector += LEFT * (right - 6.0)
        shift_needed = True
    
    if shift_needed:
        all_mobs.shift(shift_vector)  # Instant, no animation to prevent text flying!
    """
    import textwrap
    final_scaling_code = textwrap.dedent(final_scaling_code)
    
    # CORRECT APPROACH: Add scaling RIGHT AFTER the LAST animation
    # This way all objects are drawn, then we immediately scale to fit
    # MUST handle multi-line self.play() calls!
    
    lines = ai_code.split('\n')
    last_play_start = -1
    last_play_end = -1
    
    # Find the LAST self.play() call
    for i, line in enumerate(lines):
        if 'self.play(' in line.strip():
            last_play_start = i
    
    # If we found a self.play(), find where it ENDS (handle multi-line)
    if last_play_start != -1:
        paren_count = 0
        for i in range(last_play_start, len(lines)):
            line = lines[i]
            paren_count += line.count('(') - line.count(')')
            
            # When parentheses are balanced, this is the end of the statement
            if paren_count == 0:
                last_play_end = i
                break
    
    # FIX 2: ALWAYS ENABLE SCALE-FIT
    # The scale-fit code has its own safety checks (only scales if scale_factor < 0.95)
    # So it's safe to always run - it won't touch animations that fit fine
    # This catches ALL off-screen cases: .next_to() chains, accumulated positions, etc.
    
    enable_scale_fit = True  # ALWAYS enable - the code has internal safety checks
    print(f"   🎯 Scale-fit ENABLED (will only apply if content goes off-screen)")
    
    if enable_scale_fit and last_play_end != -1:
        # Insert after the complete self.play() statement
        insert_index = last_play_end + 1
        lines.insert(insert_index, '')
        lines.insert(insert_index, final_scaling_code)
        ai_code = '\n'.join(lines)
        print(f"   🎯 Inserted scaling AFTER last animation at line {insert_index}")
    

    # Calculate remaining wait needed (only if not skipping global sync)
    # FIX: Add 20% buffer to wait time to prevent early fade-out
    if not skip_global_sync:
        wait_time = max(0.5, (target_duration - total_anim_time) * 1.2)  # 20% buffer
        # Append the wait call AFTER scaling
        ai_code += f"\nself.wait({wait_time:.2f})"
    else:
        # Just add a small buffer wait to prevent premature fade
        ai_code += f"\nself.wait(0.5)"
    

    
    # Remove imports/class (but NOT def - we reject code with def statements earlier)
    # Note: def statements should already be rejected in validation
    # Removing def lines here breaks the code (leaves orphaned function bodies)
    cleaned = []
    for line in ai_code.split('\n'):
        s = line.strip()
        if s.startswith(('class ', 'import ', 'from ')):
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
    
    # Fix colors
    for bad, good in [('BROWN', 'MAROON'), ('CYAN', 'TEAL'), ('MAGENTA', 'PINK')]:
        ai_code = ai_code.replace(f'color={bad}', f'color={good}')
        ai_code = ai_code.replace(f', {bad}', f', {good}')
    
    # Force reasonable sizes
    ai_code = re.sub(r'Rectangle\(width=(\d+\.?\d*)', lambda m: f'Rectangle(width={min(float(m.group(1)), 2.5)}', ai_code)
    ai_code = re.sub(r'height=(\d+\.?\d*)', lambda m: f'height={min(float(m.group(1)), 1.5)}', ai_code)
    ai_code = re.sub(r'Circle\(radius=(\d+\.?\d*)', lambda m: f'Circle(radius={min(float(m.group(1)), 0.7)}', ai_code)
    
    
    # CONSISTENT FONT SIZE: Set all Text objects to font_size=16
    # This prevents inconsistent sizing that looks unprofessional
    lines = ai_code.split('\n')
    new_lines = []
    for i, line in enumerate(lines):
        if '= Text(' in line:
            # If font_size already specified, replace it with 16
            if 'font_size=' in line:
                line = re.sub(r'font_size=\d+', 'font_size=16', line)
            else:
                # If no font_size, add it
                # Find the closing parenthesis of Text()
                if 'Text(' in line:
                    # Add font_size before the closing )
                    line = re.sub(r'Text\(([^)]+)\)', r'Text(\1, font_size=16)', line)
        
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
    
    # Check for potential text overlap (multiple .next_to() with same direction)
    next_to_pattern = re.compile(r'(\w+)\.next_to\((\w+),\s*(\w+)')
    next_to_calls = {}
    for line in lines:
        match = next_to_pattern.search(line)
        if match:
            label_var, target_var, direction = match.groups()
            key = f"{target_var}_{direction}"
            if key not in next_to_calls:
                next_to_calls[key] = []
            next_to_calls[key].append(label_var)
    
    # DISABLED AUTO-FIX: Don't change label directions - it breaks layouts
    # Just warn about potential overlaps but let the original AI-generated layout stand
    for key, labels in next_to_calls.items():
        if len(labels) > 1:
            target, direction = key.rsplit('_', 1)
            print(f"   ⚠️  Note: Multiple labels at {direction} of {target} (may overlap, but keeping original layout)")
    
    # Fix GrowArrow
    ai_code = re.sub(r'GrowArrow\(\*(\w+)\)', r'*[GrowArrow(a) for a in \1]', ai_code)
    
    # AUTO-FIX: list.add() → list.append() (Claude often confuses list with VGroup)
    # Pattern: arrows = []; arrows.add(x) → arrows.append(x)
    lines = ai_code.split('\n')
    list_vars = set()
    
    # Find variables initialized as lists (not VGroups)
    for line in lines:
        # Pattern: var = [] or var = list()
        match = re.match(r'\s*(\w+)\s*=\s*\[\s*\]', line)
        if match:
            list_vars.add(match.group(1))
    
    # Fix .add() → .append() for list variables
    if list_vars:
        for i, line in enumerate(lines):
            for var in list_vars:
                if f'{var}.add(' in line:
                    lines[i] = line.replace(f'{var}.add(', f'{var}.append(')
                    print(f"   🔧 AUTO-FIX: Changed {var}.add() → {var}.append() (list not VGroup)")
        ai_code = '\n'.join(lines)
    
    # AUTO-FIX: Clamp shift values to prevent off-screen animations
    # This prevents objects from going off-screen DURING animations
    # Pattern: .shift(UP*5) → .shift(UP*3.5) if 5 > 3.5
    def clamp_shift(match):
        full_match = match.group(0)
        direction = match.group(1)  # UP, DOWN, LEFT, RIGHT
        value = float(match.group(2))  # The number
        
        # Clamp to safe limits
        max_shift = 3.5
        if abs(value) > max_shift:
            clamped_value = max_shift if value > 0 else -max_shift
            print(f"   🔧 AUTO-FIX: Clamped .shift({direction}*{value}) → .shift({direction}*{clamped_value})")
            return f".shift({direction}*{clamped_value})"
        return full_match
    
    # Apply clamping to all shift operations
    # Pattern: .shift(UP*5) or .shift(LEFT*-3)
    ai_code = re.sub(r'\.shift\(([A-Z]+)\s*\*\s*([+-]?\d+\.?\d*)\)', clamp_shift, ai_code)
    
    # AUTO-FIX: Replace .arrange(rows=, cols=) with .arrange_in_grid(rows=, cols=)
    # Manim's .arrange() doesn't accept rows/cols parameters - must use .arrange_in_grid()
    if '.arrange(rows=' in ai_code or '.arrange(cols=' in ai_code:
        ai_code = re.sub(r'\.arrange\((rows=\d+,\s*cols=\d+[^)]*)\)', r'.arrange_in_grid(\1)', ai_code)
        print(f"   🔧 AUTO-FIX: Changed .arrange(rows=, cols=) → .arrange_in_grid(rows=, cols=)")
    
    # AUTO-FIX: Remove stray markdown code fences
    # AI sometimes includes ``` in the Python code which causes syntax errors
    if '```' in ai_code:
        # Remove lines that are just ``` or ```python
        lines = ai_code.split('\n')
        cleaned_lines = [line for line in lines if not line.strip() in ['```', '```python']]
        ai_code = '\n'.join(cleaned_lines)
        print(f"   🔧 AUTO-FIX: Removed stray markdown code fences")
    
    # ============================================================
    # SAFE POST-PROCESSING HELPER
    # ============================================================
    # Validates syntax after each post-processing step
    # If syntax is broken, reverts to previous code
    def safe_postprocess(code_before, code_after, step_name):
        """Validate code after post-processing step. Revert if broken."""
        import ast
        try:
            ast.parse(code_after)
            return code_after  # Syntax OK, return modified code
        except SyntaxError as e:
            print(f"   ⚠️  REVERTING: {step_name} broke syntax at line {e.lineno}: {e.msg}")
            print(f"   ⚠️  Keeping original code for this step")
            return code_before  # Syntax broken, revert to original
    
    # ============================================================
    # POST-PROCESSING: FIX LABEL.move_to(BOX) → LABEL.next_to(BOX, UP)
    # ============================================================
    # Problem: AI places LABELS on top of boxes using .move_to(box)
    # Solution: Convert to .next_to(box, UP) - BUT ONLY FOR LABELS
    # SKIP: Content meant to be INSIDE boxes (content, text, value, etc.)
    # SKIP: VGroup patterns where text+box are intentionally grouped
    
    print("   🔧 POST-PROCESSING: Fixing Label.move_to(box) → .next_to(box, UP)...")
    
    # SAFETY: Save code before risky label fixes
    code_before_label_fixes = ai_code
    lines = ai_code.split('\n')
    label_fixes = 0
    
    # Find all boxes/shapes (Rectangle, Square, Circle, RoundedRectangle, Ellipse, Polygon)
    box_pattern = re.compile(r'(\w+)\s*=\s*(?:Rectangle|Square|Circle|RoundedRectangle|Ellipse|Polygon|RegularPolygon)\s*\(')
    boxes = set()
    for line in lines:
        match = box_pattern.search(line)
        if match:
            boxes.add(match.group(1))
    
    # Find VGroup pairs to skip
    vgroup_pairs = set()
    full_code = '\n'.join(lines)
    for box_name in boxes:
        vgroup_pattern = rf'VGroup\s*\(\s*(?:{re.escape(box_name)}\s*,\s*(\w+)|(\w+)\s*,\s*{re.escape(box_name)})\s*\)'
        vgroup_matches = re.findall(vgroup_pattern, full_code)
        for match in vgroup_matches:
            text_name = match[0] if match[0] else match[1]
            if text_name:
                vgroup_pairs.add((box_name, text_name))
    
    # Keywords that indicate a LABEL (should be ABOVE box)
    label_keywords = ['label', 'title', 'header', 'name', 'heading']
    # Keywords that indicate CONTENT (should be INSIDE box)
    content_keywords = ['content', 'text', 'value', 'data', 'info', 'description', 'body']
    
    if boxes:
        for i, line in enumerate(lines):
            for box_name in boxes:
                # Pattern: var = Text(...).move_to(box)
                chained_pattern = rf'(\w+)\s*=\s*(Text\([^)]+\))\.move_to\({re.escape(box_name)}(?:\.get_center\(\))?\)'
                chained_match = re.search(chained_pattern, line)
                if chained_match:
                    text_var = chained_match.group(1).lower()
                    
                    # Skip VGroup pairs
                    if (box_name, chained_match.group(1)) in vgroup_pairs:
                        continue
                    
                    # Check if it's CONTENT that should stay inside the box
                    is_content = any(kw in text_var for kw in content_keywords)
                    
                    # FIX ALL Text.move_to(box) EXCEPT content text
                    # This matches the validation logic - any Text on a box is CRITICAL
                    if not is_content:
                        new_line = re.sub(
                            rf'(Text\([^)]+\))\.move_to\({re.escape(box_name)}(?:\.get_center\(\))?\)',
                            rf'\1.next_to({box_name}, UP, buff=0.3)',
                            line
                        )
                        if new_line != line:
                            lines[i] = new_line
                            label_fixes += 1
                            print(f"      ✅ Fixed CHAINED: {chained_match.group(1)}.move_to({box_name}) → .next_to({box_name}, UP)")
    
    # ALSO handle SEPARATE LINE pattern: var.move_to(box) on its own line
    # This catches cases like:
    #   label_a = Text("Particle A", ...)
    #   label_a.move_to(particle_a)  # ← This line needs fixing!
    
    # First, find all Text variables
    text_vars = set()
    text_var_pattern = re.compile(r'(\w+)\s*=\s*Text\s*\(')
    for line in lines:
        match = text_var_pattern.search(line)
        if match:
            text_vars.add(match.group(1))
    
    if boxes and text_vars:
        for i, line in enumerate(lines):
            for box_name in boxes:
                # Pattern: text_var.move_to(box) on separate line
                for text_var in text_vars:
                    separate_pattern = rf'^(\s*)({re.escape(text_var)})\.move_to\({re.escape(box_name)}(?:\.get_center\(\))?\)\s*$'
                    separate_match = re.match(separate_pattern, line)
                    if separate_match:
                        indent = separate_match.group(1)
                        var_name = separate_match.group(2)
                        
                        # Skip VGroup pairs (text intentionally grouped with box)
                        if (box_name, var_name) in vgroup_pairs:
                            continue
                        
                        # Check if it's CONTENT that should stay inside
                        var_lower = var_name.lower()
                        is_content = any(kw in var_lower for kw in content_keywords)
                        
                        # FIX ALL Text.move_to(box) EXCEPT content text
                        # This matches the validation logic - any Text on a box is CRITICAL
                        if not is_content:
                            lines[i] = f"{indent}{var_name}.next_to({box_name}, UP, buff=0.3)"
                            label_fixes += 1
                            print(f"      ✅ Fixed SEPARATE LINE: {var_name}.move_to({box_name}) → .next_to({box_name}, UP)")
    
    if label_fixes > 0:
        ai_code = '\n'.join(lines)
        # SAFETY CHECK: Validate syntax after label fixes
        try:
            import ast
            ast.parse(ai_code)
            print(f"   ✅ Fixed {label_fixes} labels positioned on boxes")
        except SyntaxError:
            print(f"   ⚠️  REVERTING: Label fixes broke syntax, keeping original")
            ai_code = code_before_label_fixes
    else:
        print(f"   ✅ No label.move_to(box) issues detected")
    
    # ============================================================
    # POST-PROCESSING: FIX OVERLAPPING TEXT IN SAME BOX/POSITION
    # ============================================================
    # Problem: AI places multiple Text objects at same position:
    #   title = Text("Title").move_to(box)
    #   content = Text("Content").move_to(box)  ← OVERLAPS title!
    # Solution: Stack them vertically using .next_to(previous, DOWN)
    
    print("   🔧 POST-PROCESSING: Fixing overlapping text at same position...")
    
    # SAFETY: Save code before risky overlap fixes
    code_before_overlap_fixes = ai_code
    lines = ai_code.split('\n')
    
    # ============================================================
    # STEP 1: Find ALL Text objects and their targets
    # ============================================================
    # We need to catch BOTH patterns:
    #   CHAINED: title = Text("...").move_to(box)
    #   SEPARATE: title = Text("...") then title.move_to(box)
    
    text_positions = {}  # target -> [(line_idx, var_name, indent, is_chained), ...]
    
    # First, find all Text variable definitions
    text_vars = {}  # var_name -> definition_line_idx
    text_def_pattern = re.compile(r'(\s*)(\w+)\s*=\s*Text\s*\(')
    for i, line in enumerate(lines):
        match = text_def_pattern.match(line)
        if match:
            var_name = match.group(2)
            indent = match.group(1)
            text_vars[var_name] = (i, indent)
    
    # Pattern 1: CHAINED - var = Text(...).move_to(target)
    chained_pattern = re.compile(r'(\s*)(\w+)\s*=\s*Text\([^)]+\)\.move_to\((\w+)(?:\.get_center\(\))?\)')
    
    for i, line in enumerate(lines):
        match = chained_pattern.match(line)
        if match:
            indent = match.group(1)
            var_name = match.group(2)
            target = match.group(3)
            
            if target not in text_positions:
                text_positions[target] = []
            text_positions[target].append((i, var_name, indent, True))  # True = chained
    
    # Pattern 2: SEPARATE - var.move_to(target) on its own line
    separate_pattern = re.compile(r'^(\s*)(\w+)\.move_to\((\w+)(?:\.get_center\(\))?\)\s*$')
    
    for i, line in enumerate(lines):
        match = separate_pattern.match(line)
        if match:
            indent = match.group(1)
            var_name = match.group(2)
            target = match.group(3)
            
            # Only count if var_name is a known Text variable
            if var_name in text_vars:
                if target not in text_positions:
                    text_positions[target] = []
                # Check if not already added (avoid duplicates)
                if not any(v == var_name for _, v, _, _ in text_positions[target]):
                    text_positions[target].append((i, var_name, indent, False))  # False = separate
    
    # Pattern 3: MATH EXPRESSION - var.move_to(target.get_X() + DIRECTION * value)
    # Catches: label.move_to(box.get_left() + RIGHT * 0.8)
    math_pattern = re.compile(r'^(\s*)(\w+)\.move_to\((\w+)\.get_(?:left|right|top|bottom|center)\(\)\s*[+\-]')
    
    for i, line in enumerate(lines):
        match = math_pattern.match(line)
        if match:
            indent = match.group(1)
            var_name = match.group(2)
            target = match.group(3)
            
            if var_name in text_vars:
                if target not in text_positions:
                    text_positions[target] = []
                if not any(v == var_name for _, v, _, _ in text_positions[target]):
                    text_positions[target].append((i, var_name, indent, False))
    
    # ============================================================
    # STEP 2: Fix overlapping text - stack them vertically
    # ============================================================
    overlap_fixes = 0
    for target, texts in text_positions.items():
        if len(texts) > 1:
            print(f"      ⚠️  Found {len(texts)} Text objects at same position: {target}")
            
            for idx, (line_idx, var_name, indent, is_chained) in enumerate(texts):
                if idx == 0:
                    # First text - position at TOP of box, not center
                    old_line = lines[line_idx]
                    if is_chained:
                        new_line = re.sub(
                            r'\.move_to\(' + re.escape(target) + r'(?:\.get_center\(\))?\)',
                            f'.move_to({target}.get_top()).shift(DOWN*0.5)',
                            old_line
                        )
                    else:
                        new_line = f"{indent}{var_name}.move_to({target}.get_top()).shift(DOWN*0.5)"
                    lines[line_idx] = new_line
                    print(f"      ✅ Fixed FIRST: {var_name} → .move_to({target}.get_top()).shift(DOWN*0.5)")
                else:
                    # Second+ text - position below previous
                    prev_var_name = texts[idx - 1][1]
                    old_line = lines[line_idx]
                    if is_chained:
                        new_line = re.sub(
                            r'\.move_to\(' + re.escape(target) + r'(?:\.get_center\(\))?\)',
                            f'.next_to({prev_var_name}, DOWN, buff=0.3)',
                            old_line
                        )
                    else:
                        new_line = f"{indent}{var_name}.next_to({prev_var_name}, DOWN, buff=0.3)"
                    lines[line_idx] = new_line
                    print(f"      ✅ Fixed: {var_name} → .next_to({prev_var_name}, DOWN)")
                
                overlap_fixes += 1
    
    if overlap_fixes > 0:
        ai_code = '\n'.join(lines)
        # SAFETY CHECK
        try:
            import ast
            ast.parse(ai_code)
            print(f"   ✅ Fixed {overlap_fixes} overlapping text positions")
        except SyntaxError:
            print(f"   ⚠️  REVERTING: Overlap fixes broke syntax, keeping original")
            ai_code = code_before_overlap_fixes
    else:
        print(f"   ✅ No overlapping text detected")
    
    # ============================================================
    # POST-PROCESSING: FIX MULTIPLE TEXTS AT SAME POSITION
    # ============================================================
    # Problem: AI places LABELS on top of boxes using .move_to(box)
    # Solution: Convert to .next_to(box, UP) - BUT ONLY FOR LABELS
    # SKIP: Content meant to be INSIDE boxes (content, text, value, etc.)
    # SKIP: VGroup patterns where text+box are intentionally grouped
    
    print("   🔧 POST-PROCESSING: Fixing Label.move_to(box) → .next_to(box, UP)...")
    
    # SAFETY: Save code before risky label fixes
    code_before_label_fixes = ai_code
    lines = ai_code.split('\n')
    label_fixes = 0
    
    # Find all boxes/shapes (Rectangle, Square, Circle, RoundedRectangle, Ellipse, Polygon)
    box_pattern = re.compile(r'(\w+)\s*=\s*(?:Rectangle|Square|Circle|RoundedRectangle|Ellipse|Polygon|RegularPolygon)\s*\(')
    boxes = set()
    for line in lines:
        match = box_pattern.search(line)
        if match:
            boxes.add(match.group(1))
    
    # Find VGroup pairs to skip
    vgroup_pairs = set()
    full_code = '\n'.join(lines)
    for box_name in boxes:
        vgroup_pattern = rf'VGroup\s*\(\s*(?:{re.escape(box_name)}\s*,\s*(\w+)|(\w+)\s*,\s*{re.escape(box_name)})\s*\)'
        vgroup_matches = re.findall(vgroup_pattern, full_code)
        for match in vgroup_matches:
            text_name = match[0] if match[0] else match[1]
            if text_name:
                vgroup_pairs.add((box_name, text_name))
    
    # Keywords that indicate a LABEL (should be ABOVE box)
    label_keywords = ['label', 'title', 'header', 'name', 'heading']
    # Keywords that indicate CONTENT (should be INSIDE box)
    content_keywords = ['content', 'text', 'value', 'data', 'info', 'description', 'body']
    
    if boxes:
        for i, line in enumerate(lines):
            for box_name in boxes:
                # Pattern: var = Text(...).move_to(box)
                chained_pattern = rf'(\w+)\s*=\s*(Text\([^)]+\))\.move_to\({re.escape(box_name)}(?:\.get_center\(\))?\)'
                chained_match = re.search(chained_pattern, line)
                if chained_match:
                    text_var = chained_match.group(1).lower()
                    
                    # Skip VGroup pairs
                    if (box_name, chained_match.group(1)) in vgroup_pairs:
                        continue
                    
                    # Check if it's CONTENT that should stay inside the box
                    is_content = any(kw in text_var for kw in content_keywords)
                    
                    # FIX ALL Text.move_to(box) EXCEPT content text
                    # This matches the validation logic - any Text on a box is CRITICAL
                    if not is_content:
                        new_line = re.sub(
                            rf'(Text\([^)]+\))\.move_to\({re.escape(box_name)}(?:\.get_center\(\))?\)',
                            rf'\1.next_to({box_name}, UP, buff=0.3)',
                            line
                        )
                        if new_line != line:
                            lines[i] = new_line
                            label_fixes += 1
                            print(f"      ✅ Fixed CHAINED: {chained_match.group(1)}.move_to({box_name}) → .next_to({box_name}, UP)")
    
    # ALSO handle SEPARATE LINE pattern: var.move_to(box) on its own line
    # This catches cases like:
    #   label_a = Text("Particle A", ...)
    #   label_a.move_to(particle_a)  # ← This line needs fixing!
    
    # First, find all Text variables
    text_vars = set()
    text_var_pattern = re.compile(r'(\w+)\s*=\s*Text\s*\(')
    for line in lines:
        match = text_var_pattern.search(line)
        if match:
            text_vars.add(match.group(1))
    
    if boxes and text_vars:
        for i, line in enumerate(lines):
            for box_name in boxes:
                # Pattern: text_var.move_to(box) on separate line
                for text_var in text_vars:
                    separate_pattern = rf'^(\s*)({re.escape(text_var)})\.move_to\({re.escape(box_name)}(?:\.get_center\(\))?\)\s*$'
                    separate_match = re.match(separate_pattern, line)
                    if separate_match:
                        indent = separate_match.group(1)
                        var_name = separate_match.group(2)
                        
                        # Skip VGroup pairs (text intentionally grouped with box)
                        if (box_name, var_name) in vgroup_pairs:
                            continue
                        
                        # Check if it's CONTENT that should stay inside
                        var_lower = var_name.lower()
                        is_content = any(kw in var_lower for kw in content_keywords)
                        
                        # FIX ALL Text.move_to(box) EXCEPT content text
                        # This matches the validation logic - any Text on a box is CRITICAL
                        if not is_content:
                            lines[i] = f"{indent}{var_name}.next_to({box_name}, UP, buff=0.3)"
                            label_fixes += 1
                            print(f"      ✅ Fixed SEPARATE LINE: {var_name}.move_to({box_name}) → .next_to({box_name}, UP)")
    
    if label_fixes > 0:
        ai_code = '\n'.join(lines)
        # SAFETY CHECK: Validate syntax after label fixes
        try:
            import ast
            ast.parse(ai_code)
            print(f"   ✅ Fixed {label_fixes} labels positioned on boxes")
        except SyntaxError:
            print(f"   ⚠️  REVERTING: Label fixes broke syntax, keeping original")
            ai_code = code_before_label_fixes
    else:
        print(f"   ✅ No label.move_to(box) issues detected")
    
    # ============================================================
    # POST-PROCESSING: FIX LABELS POSITIONED BEFORE .arrange()
    # ============================================================
    # Problem: AI creates labels with .next_to(box, DOWN) BEFORE boxes are arranged
    #   label1.next_to(box1, DOWN)  # box1 is at ORIGIN
    #   label2.next_to(box2, DOWN)  # box2 is also at ORIGIN - OVERLAP!
    #   VGroup(...).arrange(RIGHT)  # boxes move, but labels DON'T
    # Solution: Add label repositioning AFTER the .arrange() call
    
    # ============================================================
    # POST-PROCESSING: FIX LABELS POSITIONED BEFORE .arrange()
    # ============================================================
    # Problem: When labels are positioned BEFORE .arrange(), they all overlap at ORIGIN
    # Only runs when we detect this specific pattern exists
    
    lines = ai_code.split('\n')
    
    # Step 1: Find labels that use .next_to(something, DOWN/UP) pattern
    label_positions = {}  # label_var -> (target, direction, buff, line_idx)
    label_pattern = re.compile(r'(\w+)\.next_to\((\w+),\s*(DOWN|UP)(?:,\s*buff\s*=\s*([\d.]+))?\)')
    
    for i, line in enumerate(lines):
        match = label_pattern.search(line)
        if match:
            label_var = match.group(1)
            target = match.group(2)
            direction = match.group(3)
            buff = match.group(4) or '0.6'
            # Only track label variables (contains 'label')
            if 'label' in label_var.lower():
                label_positions[label_var] = (target, direction, buff, i)
    
    # Step 2: Find .arrange() calls and check if any labels were created BEFORE them
    arrange_fixes_needed = []
    
    for i, line in enumerate(lines):
        if '.arrange(' in line and ('RIGHT' in line or 'LEFT' in line or 'DOWN' in line or 'UP' in line):
            # Find which labels were positioned before this .arrange() call
            labels_before_arrange = []
            for label_var, (target, direction, buff, label_line_idx) in label_positions.items():
                if label_line_idx < i:  # Label was positioned before arrange
                    labels_before_arrange.append((label_var, target, direction, buff))
            
            if labels_before_arrange:
                arrange_fixes_needed.append((i, labels_before_arrange, line))
    
    # Step 3: Only apply fix if problem actually exists
    if arrange_fixes_needed:
        print("   🔧 POST-PROCESSING: Fixing labels positioned before .arrange()...")
        
        # Apply fixes in reverse order (so line numbers don't shift)
        for arrange_line_idx, labels_to_fix, arrange_line in reversed(arrange_fixes_needed):
            # Get indentation of arrange line
            indent = arrange_line[:len(arrange_line) - len(arrange_line.lstrip())]
            
            # Insert repositioning lines ONE AT A TIME (after the arrange line)
            insert_position = arrange_line_idx + 1
            
            # First insert the comment
            lines.insert(insert_position, f"{indent}# Reposition labels after arrange")
            insert_position += 1
            
            # Then insert each label repositioning line
            for label_var, target, direction, buff in labels_to_fix:
                lines.insert(insert_position, f"{indent}{label_var}.next_to({target}, {direction}, buff={buff})")
                insert_position += 1
            
            # Add empty line after
            lines.insert(insert_position, "")
            
            print(f"      ✅ Added repositioning for {len(labels_to_fix)} labels after .arrange()")
        
        ai_code = '\n'.join(lines)
        print(f"   ✅ Fixed {len(arrange_fixes_needed)} label-arrange positioning issues")
    else:
        print("   🔧 POST-PROCESSING: Fixing labels positioned before .arrange()...")
        print("   ✅ No label-arrange positioning issues detected")
    
    # ============================================================
    # POST-PROCESSING: FIX OVERLAPPING LABELS (SAME TARGET + DIRECTION)
    # ============================================================
    # Problem: Multiple labels positioned .next_to(X, DIRECTION) end up overlapping
    #   label1.next_to(cloud, DOWN)  # All at same position!
    #   label2.next_to(cloud, DOWN)  # OVERLAP!
    #   label3.next_to(cloud, DOWN)  # OVERLAP!
    # Solution: Add offset to spread labels apart
    
    print("   🔧 POST-PROCESSING: Spreading overlapping labels (same target+direction)...")
    
    lines = ai_code.split('\n')
    
    # Track labels by (target, direction): {(target, direction): [(line_idx, var_name), ...]}
    labels_by_position = {}
    
    # Match: var.next_to(target, DIRECTION) - handles all 4 directions
    next_to_pattern = re.compile(r'(\w+)\.next_to\((\w+),\s*(UP|DOWN|LEFT|RIGHT)(?:,\s*buff\s*=\s*[\d.]+)?\)')
    
    for i, line in enumerate(lines):
        match = next_to_pattern.search(line)
        if match:
            var_name = match.group(1)
            target = match.group(2)
            direction = match.group(3)
            
            key = (target, direction)
            if key not in labels_by_position:
                labels_by_position[key] = []
            labels_by_position[key].append((i, var_name))
    
    spread_fixes = 0
    
    # Process each group with 2+ labels at same position
    for (target, direction), label_list in labels_by_position.items():
        if len(label_list) >= 2:
            print(f"      ⚠️  Found {len(label_list)} labels at .next_to({target}, {direction})")
            
            for idx, (line_idx, var_name) in enumerate(label_list):
                if idx == 0:
                    continue  # First label stays in place
                
                # Add offset to spread labels
                indent = lines[line_idx][:len(lines[line_idx]) - len(lines[line_idx].lstrip())]
                offset = idx * 0.4  # 0.4 units apart
                
                # Determine shift direction based on original direction
                if direction in ['UP', 'DOWN']:
                    # Vertical arrangement - spread further in same direction
                    shift_dir = direction
                else:
                    # Horizontal arrangement - spread vertically
                    shift_dir = 'DOWN'
                
                shift_line = f"{indent}{var_name}.shift({shift_dir} * {offset})"
                lines.insert(line_idx + 1 + spread_fixes, shift_line)  # Account for previous insertions
                
                spread_fixes += 1
                print(f"      ✅ Added: {var_name}.shift({shift_dir} * {offset})")
    
    if spread_fixes > 0:
        ai_code = '\n'.join(lines)
        print(f"   ✅ Spread {spread_fixes} labels to prevent overlap")
    else:
        print(f"   ✅ No label spreading needed")
    
    # ============================================================
    # POST-PROCESSING: CLEAN UP FONT PARAMETERS
    # ============================================================
    # Remove problematic font parameters but KEEP disable_ligatures
    # disable_ligatures=True helps fix letter spacing issues!
    
    print("   🔧 POST-PROCESSING: Cleaning up font parameters...")
    
    lines = ai_code.split('\n')
    typography_fixes = 0
    
    for i, line in enumerate(lines):
        original_line = line
        
        # Remove font="Inter" - this font has kerning issues
        if 'font="Inter"' in line or "font='Inter'" in line:
            line = re.sub(r',?\s*font=["\']Inter["\']', '', line)
            typography_fixes += 1
        
        # Remove weight parameter (MEDIUM, SEMIBOLD, etc.) - not needed
        if 'weight=' in line:
            line = re.sub(r',?\s*weight=\w+', '', line)
            typography_fixes += 1
        
        # KEEP disable_ligatures=True - it helps with letter spacing!
        # Don't remove it
        
        # Clean up any leftover issues
        line = re.sub(r',\s*,', ',', line)  # double commas
        line = re.sub(r'\(\s*,', '(', line)  # leading comma after (
        line = re.sub(r',\s*\)', ')', line)  # trailing comma before )
        
        # Fix lines that start with just whitespace + comma (from multi-line Text() cleanup)
        # e.g. "           , disable_ligatures=True)" → "           disable_ligatures=True)"
        line = re.sub(r'^(\s*),\s*', r'\1', line)
        
        if line != original_line:
            lines[i] = line
    
    if typography_fixes > 0:
        ai_code = '\n'.join(lines)
        ai_code = safe_postprocess(ai_code, ai_code, "font cleanup")
        print(f"   ✅ Cleaned up {typography_fixes} font parameters (removed Inter, kept disable_ligatures)")
    else:
        print(f"   ✅ No font parameters to clean up")
    
    # AUTO-FIX: Replace invalid self.wait(0.0) or self.wait(0) with minimum valid time
    # Manim requires wait time > 0
    # Match exactly: self.wait(0) or self.wait(0.0) but NOT self.wait(0.5) etc.
    if 'self.wait(0' in ai_code:
        # Replace self.wait(0) or self.wait(0.0) with self.wait(0.1)
        ai_code = re.sub(r'self\.wait\(0(?:\.0)?\)', 'self.wait(0.1)', ai_code)
        print(f"   🔧 AUTO-FIX: Changed self.wait(0) → self.wait(0.1) (Manim requires wait > 0)")
    
    # AUTO-FIX: Detect overlapping text labels (text placed on text)
    # Find all text objects and their positions
    text_positions = {}  # {text_name: (x, y)}
    text_pattern = r'(\w+)\s*=\s*Text\('
    text_names = re.findall(text_pattern, ai_code)
    
    for text_name in text_names:
        # Find .next_to() or .shift() positioning
        next_to_match = re.search(rf'{text_name}\.next_to\([^,]+,\s*DOWN[^)]*\)', ai_code)
        if next_to_match:
            # Track this text's approximate position (below something)
            if 'DOWN' in text_positions.values():
                # Another text already at DOWN position - shift this one
                old_line = next_to_match.group(0)
                new_line = old_line.replace('DOWN', 'DOWN').replace(')', ', buff=0.5)')
                if 'buff=' not in old_line:
                    ai_code = ai_code.replace(old_line, old_line.replace(')', ', buff=0.5)'))
                    print(f"   🔧 AUTO-FIX: Added spacing to {text_name} to prevent text overlap")
    
    # AUTO-FIX: Convert header.shift() to header.next_to(boxes, UP)
    # Problem: AI uses absolute positions like tcp_header.shift(LEFT*3.5 + UP*1.5)
    # Solution: Change to tcp_header.next_to(tcp_boxes, UP) so headers are ALWAYS above boxes
    lines = ai_code.split('\n')
    fixed_lines = []
    header_positions = {}  # Track headers that need to be repositioned
    
    for i, line in enumerate(lines):
        # Detect pattern: header = Text(...) followed by header.shift(...)
        # Example: tcp_header.shift(LEFT * 3.5 + UP * 1.5)
        if '_header' in line and '.shift(' in line:
            # Extract header name
            header_match = re.match(r'\s*(\w*header\w*)\s*\.shift\(', line)
            if header_match:
                header_name = header_match.group(1)
                # Find corresponding boxes (e.g., tcp_header → tcp_boxes)
                boxes_name = header_name.replace('_header', '_boxes').replace('header', 'boxes')
                
                # Check if boxes exist in the code
                if boxes_name in ai_code:
                    # Preserve original indentation
                    indent = line[:len(line) - len(line.lstrip())]
                    # Replace shift with next_to
                    fixed_line = f"{indent}{header_name}.next_to({boxes_name}, UP, buff=0.5)"
                    fixed_lines.append(fixed_line)
                    header_positions[header_name] = boxes_name
                    print(f"   🔧 AUTO-FIX: Changed {header_name}.shift() → {header_name}.next_to({boxes_name}, UP)")
                    continue
        
        fixed_lines.append(line)
    
    ai_code = '\n'.join(fixed_lines)

    # ============================================================
    # POST-PROCESSING: REMOVE DEF/CLASS STATEMENTS (CRITICAL!)
    # ============================================================
    # Problem: Claude sometimes generates helper functions with def/class
    # These will crash because code runs inside construct()
    # Solution: Strip out def statements and keep only the body code
    
    print("   🔧 POST-PROCESSING: Removing def/class statements...")
    
    if 'def ' in ai_code or 'class ' in ai_code:
        lines = ai_code.split('\n')
        cleaned_lines = []
        inside_def = False
        def_indent = 0
        
        for line in lines:
            stripped = line.lstrip()
            current_indent = len(line) - len(stripped)
            
            # Check if this is a def/class line
            if stripped.startswith('def ') or stripped.startswith('class '):
                inside_def = True
                def_indent = current_indent
                print(f"      🔧 Removing: {stripped[:50]}...")
                continue
            
            # If we're inside a def, check if we've exited
            if inside_def:
                if stripped and current_indent <= def_indent:
                    # We've exited the def block
                    inside_def = False
                elif stripped.startswith('return '):
                    # Skip return statements
                    continue
                elif stripped:
                    # This is code inside the def - dedent and keep it
                    # Remove one level of indentation
                    dedented = line[4:] if line.startswith('    ') else line
                    cleaned_lines.append(dedented)
                    continue
            
            if not inside_def:
                cleaned_lines.append(line)
        
        ai_code = '\n'.join(cleaned_lines)
        print(f"      ✅ Removed def/class statements, kept body code")
    else:
        print("      ✅ No def/class statements found")

    # ============================================================
    # POST-PROCESSING: ADD MISSING IMPORTS
    # ============================================================
    # Problem: Claude uses math.pi, math.cos, etc. but forgets to import math
    # Solution: Detect math usage and add import if missing
    
    print("   🔧 POST-PROCESSING: Adding missing imports...")
    
    # Check if code uses math functions
    uses_math = any(pattern in ai_code for pattern in ['math.pi', 'math.cos', 'math.sin', 'math.sqrt', 'math.tan'])
    has_math_import = 'import math' in ai_code
    
    if uses_math and not has_math_import:
        # Add import math at the TOP of the code (before any other lines)
        ai_code = 'import math\n' + ai_code
        print("      ✅ Added 'import math' at TOP (detected math.pi, math.cos, etc.)")
    else:
        if uses_math:
            print("      ✅ 'import math' already present")
        else:
            print("      ✅ No math functions detected")
    
    # Also check for numpy usage
    uses_numpy = 'np.' in ai_code or 'numpy.' in ai_code
    has_numpy_import = 'import numpy' in ai_code
    
    if uses_numpy and not has_numpy_import:
        ai_code = 'import numpy as np\n' + ai_code
        print("      ✅ Added 'import numpy as np' at TOP")

    # ============================================================
    # POST-PROCESSING: AUTO-SCALE TEXT TO FIT INSIDE BOXES
    # ============================================================
    # Problem: Text overflows outside Rectangle/Square boxes
    # Solution: Detect VGroup(box, text) patterns and insert auto-scaling
    
    print("   🔧 POST-PROCESSING: Auto-scaling text to fit inside boxes...")
    
    lines = ai_code.split('\n')
    processed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        processed_lines.append(line)
        
        # Detect pattern: VGroup(Rectangle(...), Text(...).move_to(...))
        # This is the pattern Claude uses for boxes with text inside
        if 'VGroup(' in line and 'Rectangle(' in line and 'Text(' in line:
            # Extract variable name (e.g., "cell = VGroup(...)")
            var_match = re.match(r'\s*(\w+)\s*=\s*VGroup\(', line)
            if var_match:
                var_name = var_match.group(1)
                indent = line[:len(line) - len(line.lstrip())]
                
                # Add auto-scaling code after this line
                # This ensures text never overflows the box
                # ONLY scales if text is ACTUALLY too big (doesn't touch good text)
                # Use 0.95 to be less aggressive - only scale if really overflowing
                scale_code = f"{indent}# Auto-scale text ONLY if significantly overflowing\n"
                scale_code += f"{indent}if {var_name}[1].width > {var_name}[0].width * 0.95:\n"
                scale_code += f"{indent}    {var_name}[1].scale_to_fit_width({var_name}[0].width * 0.9)\n"
                scale_code += f"{indent}if {var_name}[1].height > {var_name}[0].height * 0.95:\n"
                scale_code += f"{indent}    {var_name}[1].scale_to_fit_height({var_name}[0].height * 0.9)"
                
                processed_lines.append(scale_code)
                print(f"      ✅ Added auto-scaling for {var_name}")
        
        i += 1
    
    ai_code = '\n'.join(processed_lines)
    print("   ✅ Text-in-box auto-scaling complete")

    # ============================================================
    # POST-PROCESSING: AUTO-SCALE VGROUP TEXT INSIDE BOXES
    # ============================================================
    # Problem: Claude creates multi-line VGroups and moves them into boxes
    # Pattern: content = VGroup(text1, text2, text3); content.move_to(box)
    # Solution: Detect this and auto-scale the VGroup to fit
    
    print("   🔧 POST-PROCESSING: Auto-scaling VGroup text inside boxes...")
    
    lines = ai_code.split('\n')
    insertions = []  # Collect insertions first to avoid index shifting
    processed_vgroups = set()  # Prevent duplicate scaling
    
    # Find patterns: vgroup.move_to(box)
    for i in range(len(lines)):
        line = lines[i]
        
        match = re.match(r'(\s*)(\w+)\.move_to\((\w+)', line)
        if match:
            indent = match.group(1)
            vgroup_name = match.group(2)
            box_name = match.group(3)
            
            # Skip if already processed
            if vgroup_name in processed_vgroups:
                continue
            
            # Verify vgroup_name is a VGroup (look backwards)
            is_vgroup = False
            for j in range(i-1, max(0, i-20), -1):
                if f'{vgroup_name} = VGroup(' in lines[j]:
                    is_vgroup = True
                    break
            
            if not is_vgroup:
                continue
            
            # Verify box_name is actually a box (Rectangle/RoundedRectangle)
            is_box = False
            for k in range(max(0, i-30), i):
                if f'{box_name} = ' in lines[k] and ('Rectangle(' in lines[k] or 'RoundedRectangle(' in lines[k]):
                    is_box = True
                    break
            
            if not is_box:
                continue
            
            # Both checks passed - add scaling (less aggressive)
            scale_code = f"{indent}# Auto-scale VGroup ONLY if significantly overflowing\n"
            scale_code += f"{indent}if {vgroup_name}.width > {box_name}.width * 0.95:\n"
            scale_code += f"{indent}    {vgroup_name}.scale_to_fit_width({box_name}.width * 0.9)\n"
            scale_code += f"{indent}if {vgroup_name}.height > {box_name}.height * 0.95:\n"
            scale_code += f"{indent}    {vgroup_name}.scale_to_fit_height({box_name}.height * 0.9)"
            
            insertions.append((i, scale_code))
            processed_vgroups.add(vgroup_name)
            print(f"      ✅ Added VGroup auto-scaling for {vgroup_name} → {box_name}")
    
    # Apply insertions in REVERSE order to avoid index shifting
    for line_num, code in reversed(insertions):
        lines.insert(line_num, code)
    ai_code = '\n'.join(lines)
    print("   ✅ VGroup text-in-box auto-scaling complete")

    # ============================================================
    # POST-PROCESSING: FIX COLUMN HEADER OVERLAP WITH CONTENT (DISABLED)
    # ============================================================
    # DISABLED: This fix was incorrectly detecting headers and breaking layouts
    # Example: detected "right_header" as LEFT side because "LEFT" appeared elsewhere
    # Keeping the original AI-generated layout is safer than auto-fixing
    
    print("   🔧 POST-PROCESSING: Column header overlap fix (DISABLED - was causing layout issues)")
    print("      ✅ Skipped (keeping original AI-generated layout)")

    # ============================================================
    # POST-PROCESSING: FIX MULTIPLE TEXTS IN SAME BOX (CRITICAL!)
    # ============================================================
    # Problem: Claude puts TWO texts in the same box causing overlap
    # Multiple patterns to detect:
    #   Pattern 1: text1.move_to(box1); text2.move_to(box1)
    #   Pattern 2: VGroup(box, text1, text2) - multiple texts in same VGroup
    #   Pattern 3: text = Text("A\nB").move_to(box) where A and B are unrelated
    
    print("   🔧 POST-PROCESSING: Detecting multiple texts in same box...")
    
    lines = ai_code.split('\n')
    
    # Track which boxes/targets have text moved into them
    box_text_mapping = {}  # {box_name: [(line_idx, text_var), ...]}
    
    for i, line in enumerate(lines):
        # Pattern 1: text_var.move_to(box_var) or Text("...").move_to(box_var)
        match = re.search(r'(\w+)\.move_to\((\w+)', line)
        if match:
            text_var = match.group(1)
            target_var = match.group(2)
            
            # Check if target_var is a box
            is_box = False
            for j in range(max(0, i-50), i):
                if f'{target_var} = ' in lines[j] and ('Rectangle(' in lines[j] or 'RoundedRectangle(' in lines[j] or 'Circle(' in lines[j]):
                    is_box = True
                    break
                    
            # Check if text_var is Text (defined above or inline)
            is_text = ('Text(' in line) or any(f'{text_var} = Text(' in lines[j] for j in range(max(0, i-30), i))
            
            if is_box and is_text:
                if target_var not in box_text_mapping:
                    box_text_mapping[target_var] = []
                box_text_mapping[target_var].append((i, text_var))
    
    # Find boxes with multiple texts - this is the overlap problem!
    duplicate_fixes = 0
    for box_name, text_placements in box_text_mapping.items():
        if len(text_placements) > 1:
            print(f"      ⚠️  OVERLAP FOUND: {len(text_placements)} texts placed in '{box_name}'!")
            
            # Keep only the FIRST text, comment out the rest
            for idx, (line_idx, text_var) in enumerate(text_placements[1:], 1):
                original_line = lines[line_idx]
                
                # SAFETY CHECK: Don't comment out if this is the ONLY line inside a for/while loop
                # This would leave an empty loop body causing syntax error
                is_only_line_in_loop = False
                if line_idx > 0:
                    # Check if previous non-blank line is a for/while statement
                    for check_idx in range(line_idx - 1, max(0, line_idx - 5), -1):
                        prev_line = lines[check_idx].strip()
                        if prev_line:  # Found non-blank line
                            if prev_line.endswith(':') and (prev_line.startswith('for ') or prev_line.startswith('while ')):
                                # Check if next non-blank line is outside the loop (less indented)
                                current_indent = len(original_line) - len(original_line.lstrip())
                                for next_idx in range(line_idx + 1, min(len(lines), line_idx + 5)):
                                    next_line = lines[next_idx]
                                    if next_line.strip():  # Found non-blank line
                                        next_indent = len(next_line) - len(next_line.lstrip())
                                        if next_indent <= current_indent - 4 or next_indent == 0:
                                            # Next line is outside loop - this IS the only statement
                                            is_only_line_in_loop = True
                                        break
                            break
                
                if is_only_line_in_loop:
                    print(f"         ⏭️  Skipping {text_var} - it's the only line in a for/while loop")
                    continue
                
                # Comment out the duplicate placement
                lines[line_idx] = f"# REMOVED (duplicate text in {box_name}): {original_line.strip()}"
                print(f"         ✅ Removed duplicate: {text_var} from {box_name}")
                duplicate_fixes += 1
    
    ai_code = '\n'.join(lines)
    
    # Pattern 2: VGroups containing multiple Text objects with .move_to()
    # Find VGroups like: content = VGroup(text1, text2).move_to(box)
    lines = ai_code.split('\n')
    for i, line in enumerate(lines):
        # Pattern: VGroup(var1, var2, ...).move_to(box)
        vgroup_match = re.search(r'VGroup\(([^)]+)\)\.move_to\((\w+)\)', line)
        if vgroup_match:
            vgroup_contents = vgroup_match.group(1)
            target = vgroup_match.group(2)
            
            # Count how many text variables are in this VGroup
            text_vars_in_vgroup = []
            for var in re.findall(r'\b(\w+)\b', vgroup_contents):
                # Check if var is a Text object
                for j in range(max(0, i-30), i):
                    if f'{var} = Text(' in lines[j]:
                        text_vars_in_vgroup.append(var)
                    break
            
            if len(text_vars_in_vgroup) > 1:
                print(f"      ⚠️  OVERLAP FOUND: VGroup with {len(text_vars_in_vgroup)} texts moved to '{target}'!")
                # Keep only first text, remove others from VGroup
                for extra_text in text_vars_in_vgroup[1:]:
                    # Remove extra text from VGroup
                    old_vgroup = vgroup_match.group(0)
                    new_vgroup = re.sub(rf',?\s*{extra_text}\s*,?', ',', old_vgroup)
                    new_vgroup = re.sub(r',\s*,', ',', new_vgroup)  # Fix double commas
                    new_vgroup = re.sub(r'\(,', '(', new_vgroup)  # Fix leading comma
                    new_vgroup = re.sub(r',\)', ')', new_vgroup)  # Fix trailing comma
                    lines[i] = lines[i].replace(old_vgroup, new_vgroup)
                    print(f"         ✅ Removed {extra_text} from VGroup")
                    duplicate_fixes += 1
    
    ai_code = '\n'.join(lines)
    
    if duplicate_fixes > 0:
        print(f"   ✅ Fixed {duplicate_fixes} duplicate text placements")
    else:
        print("   ✅ No duplicate text placements detected")

    # ============================================================
    # POST-PROCESSING: REMOVE LITERAL "font_size=X" FROM TEXT STRINGS
    # ============================================================
    # Problem: Claude puts "font_size=16" INSIDE the text string
    # Example: Text("Hello, font_size=16") instead of Text("Hello", font_size=16)
    # Solution: Remove this pattern from all Text() and MathTex() calls
    
    print("   🔧 POST-PROCESSING: Removing literal font_size from text strings...")
    
    import re
    
    # Pattern: Text("..., font_size=16)") or Text("...,font_size=16)")
    # Remove ", font_size=X)" or ",font_size=X)" from inside strings
    fixed_count = 0
    lines = ai_code.split('\n')
    
    for i, line in enumerate(lines):
        if 'Text(' in line or 'MathTex(' in line:
            # Remove patterns like ", font_size=16)" or ",font_size=16)" from inside quotes
            original = line
            
            # Pattern 1: ", font_size=16)" inside string
            line = re.sub(r'(Text\(["\'])([^"\']*),\s*font_size=\d+\)', r'\1\2)', line)
            line = re.sub(r'(MathTex\(["\'])([^"\']*),\s*font_size=\d+\)', r'\1\2)', line)
            
            # Pattern 2: ",font_size=16)" inside string (no space)
            line = re.sub(r'(Text\(["\'])([^"\']*),font_size=\d+\)', r'\1\2)', line)
            line = re.sub(r'(MathTex\(["\'])([^"\']*),font_size=\d+\)', r'\1\2)', line)
            
            # Pattern 3: " font_size=16)" inside string (with space before)
            line = re.sub(r'(Text\(["\'])([^"\']*)(\s+)font_size=\d+\)', r'\1\2)', line)
            line = re.sub(r'(MathTex\(["\'])([^"\']*)(\s+)font_size=\d+\)', r'\1\2)', line)
            
            if line != original:
                lines[i] = line
                fixed_count += 1
                print(f"      ✅ Removed font_size from text string: line {i+1}")
    
    ai_code = '\n'.join(lines)
    
    if fixed_count > 0:
        print(f"   ✅ Fixed {fixed_count} text strings with literal font_size")
    else:
        print("   ✅ No literal font_size in text strings detected")

    # ============================================================
    # POST-PROCESSING: AUTO-SCALE SINGLE TEXT OBJECTS IN BOXES
    # ============================================================
    # Problem: Single Text objects (not VGroups) overflow boxes
    # Solution: Add scaling for text.move_to(box) patterns
    
    print("   🔧 POST-PROCESSING: Auto-scaling single Text objects in boxes...")
    
    lines = ai_code.split('\n')
    insertions = []
    processed_texts = set()
    
    for i in range(len(lines)):
        line = lines[i]
        
        # Pattern: text_var.move_to(box_var)
        match = re.match(r'(\s*)(\w+)\.move_to\((\w+)', line)
        if match:
            indent = match.group(1)
            text_name = match.group(2)
            box_name = match.group(3)
            
            if text_name in processed_texts:
                continue
            
            # Check if text_name is Text/MathTex
            is_text = False
            for j in range(i-1, max(0, i-20), -1):
                if f'{text_name} = Text(' in lines[j] or f'{text_name} = MathTex(' in lines[j]:
                    is_text = True
                    break
            
            if not is_text:
                continue
            
            # Check if box_name is a box
            is_box = False
            for k in range(max(0, i-30), i):
                if f'{box_name} = ' in lines[k] and ('Rectangle(' in lines[k] or 'RoundedRectangle(' in lines[k]):
                    is_box = True
                    break
            
            if not is_box:
                continue
            
            # CRITICAL: Skip small boxes (like checkboxes)
            # Check box dimensions - only scale for reasonably large boxes
            skip_small_box = False
            for k in range(max(0, i-30), i):
                if f'{box_name} = ' in lines[k]:
                    width_match = re.search(r'width\s*=\s*(\d+\.?\d*)', lines[k])
                    height_match = re.search(r'height\s*=\s*(\d+\.?\d*)', lines[k])
                    
                    if width_match and height_match:
                        width = float(width_match.group(1))
                        height = float(height_match.group(1))
                        
                        # Skip checkbox-sized boxes
                        if width < 2.0 or height < 1.0:
                            skip_small_box = True
                            print(f"      ⏭️  Skipping small box {box_name} (w={width}, h={height})")
                            break
            
            if skip_small_box:
                continue
            
            # Add scaling (less aggressive - only for significant overflow)
            scale_code = f"{indent}# Auto-scale Text ONLY if significantly overflowing\n"
            scale_code += f"{indent}if {text_name}.width > {box_name}.width * 0.95:\n"
            scale_code += f"{indent}    {text_name}.scale_to_fit_width({box_name}.width * 0.9)\n"
            scale_code += f"{indent}if {text_name}.height > {box_name}.height * 0.95:\n"
            scale_code += f"{indent}    {text_name}.scale_to_fit_height({box_name}.height * 0.9)"
            
            insertions.append((i, scale_code))
            processed_texts.add(text_name)
            print(f"      ✅ Will add Text auto-scaling for {text_name} → {box_name}")
    
    # Apply insertions in REVERSE order
    for line_num, code in reversed(insertions):
        lines.insert(line_num, code)
    
    ai_code = '\n'.join(lines)
    
    if insertions:
        print(f"   ✅ Added auto-scaling to {len(insertions)} Text objects")
    else:
        print("   ✅ No single Text objects in boxes detected")

    # ============================================================
    # POST-PROCESSING: REMOVE MULTI-LINE POSITION-CHANGING ANIMATIONS
    # ============================================================
    # Problem: Multi-line .animate.shift()/.animate.move_to() cause syntax errors
    # Solution: Remove ONLY multi-line position animations, keep single-line ones
    # This allows intentional design movements while preventing syntax errors
    
    print("   🔧 POST-PROCESSING: Removing multi-line position-changing animations...")
    
    lines = ai_code.split('\n')
    safe_lines = []
    removed_count = 0
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Only check self.play() calls
        if 'self.play(' in line:
            # Check if this has position animation AND is multi-line
            has_position_anim = '.animate.shift(' in line or '.animate.move_to(' in line
            is_multiline = line.count('(') > line.count(')')
            
            if has_position_anim and is_multiline:
                # This is a multi-line position animation - REMOVE IT
                removed_count += 1
                print(f"      ⚠️  Removed multi-line position animation: {line.strip()[:60]}...")
                
                # Skip all continuation lines until closing paren
                i += 1
                while i < len(lines):
                    continuation_line = lines[i]
                    i += 1
                    if ')' in continuation_line:
                        break
                continue
            # else: Single-line position animation - KEEP IT (intentional design)
        
        safe_lines.append(line)
        i += 1
    
    ai_code = '\n'.join(safe_lines)
    
    if removed_count > 0:
        print(f"   ✅ Removed {removed_count} multi-line position animations")
    else:
        print("   ✅ No multi-line position animations found")

    # ============================================================
    # POST-PROCESSING: OVERLAPPING POSITIONS (DISABLED)
    # ============================================================
    # DISABLED: Was generating invalid Manim syntax like .next_to(parent, DOWN + LEFT*2.5)
    # which is NOT valid - direction must be a single vector
    
    print("   🔧 POST-PROCESSING: Overlap detection (DISABLED - was causing syntax errors)")

    # ============================================================
    # POST-PROCESSING: INCREASE SMALL BUFF VALUES IN .next_to()
    # ============================================================
    # Problem: Claude uses buff=0.1 or buff=0.2, causing text to overlap
    # Solution: Increase small buff values to minimum 0.5
    
    print("   🔧 POST-PROCESSING: Fixing small buff values in .next_to()...")
    
    import re
    
    lines = ai_code.split('\n')
    buff_fixed = 0
    
    for i, line in enumerate(lines):
        if '.next_to(' in line and 'buff=' in line:
            # Extract buff value: .next_to(parent, DOWN, buff=0.1)
            match = re.search(r'buff\s*=\s*(\d+\.?\d*)', line)
            if match:
                buff_value = float(match.group(1))
                
                # If buff is too small, increase it moderately
                if buff_value < 0.6:
                    new_buff = 0.6
                    old_pattern = f'buff={buff_value}'
                    new_pattern = f'buff={new_buff}'
                    lines[i] = line.replace(old_pattern, new_pattern)
                    buff_fixed += 1
                    print(f"      ✅ Increased buff: {buff_value} → {new_buff}")
    
    # Also check .arrange() buff values
    for i, line in enumerate(lines):
        if '.arrange(' in line:
            if 'buff=' in line:
                match = re.search(r'buff\s*=\s*(\d+\.?\d*)', line)
                if match:
                    buff_value = float(match.group(1))
                    if buff_value < 0.6:
                        new_buff = 0.6
                        lines[i] = line.replace(f'buff={buff_value}', f'buff={new_buff}')
                        buff_fixed += 1
                        print(f"      ✅ Increased .arrange() buff: {buff_value} → {new_buff}")
            else:
                # No buff - add default
                old_line = line
                line = line.replace('.arrange(DOWN)', '.arrange(DOWN, buff=0.6)')
                line = line.replace('.arrange(UP)', '.arrange(UP, buff=0.6)')
                line = line.replace('.arrange(LEFT)', '.arrange(LEFT, buff=0.6)')
                line = line.replace('.arrange(RIGHT)', '.arrange(RIGHT, buff=0.6)')
                if line != old_line:
                    lines[i] = line
                    buff_fixed += 1
                    print(f"      ✅ Added default .arrange() buff=1.0")
    
    ai_code = '\n'.join(lines)
    
    if buff_fixed > 0:
        print(f"   ✅ Fixed {buff_fixed} small buff values")
    else:
        print("   ✅ No small buff values detected")

    # ============================================================
    # POST-PROCESSING: AUTO-SCALE ALL VGROUPS TO FIT ON SCREEN
    # ============================================================
    # Problem: Large diagrams/flowcharts get cut off at edges
    # Solution: Add auto-scaling to ALL VGroups (universal fix)
    
    print("   🔧 POST-PROCESSING: Auto-scaling VGroups to fit on screen...")
    
    lines = ai_code.split('\n')
    processed_lines = []
    scale_added = 0
    
    for i, line in enumerate(lines):
        processed_lines.append(line)
        
        # Detect ANY VGroup creation
        # Pattern: var_name = VGroup(...)
        if '= VGroup(' in line:
            # CRITICAL: Skip multi-line VGroups to avoid breaking syntax
            
            # Check 1: Line ends with [ or has unclosed parens
            if line.rstrip().endswith('[') or (line.count('(') > line.count(')')):
                # Multi-line VGroup, skip
                continue
            
            # Check 2: Next line is a continuation (more indented or starts with ) or ])
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if next_line.strip():  # Only check non-empty lines
                    current_indent = len(line) - len(line.lstrip())
                    next_indent = len(next_line) - len(next_line.lstrip())
                    # If next line is more indented OR starts with closing bracket/paren, it's a continuation
                    if next_indent > current_indent or next_line.lstrip().startswith((')', ']', ',')):
                        # Multi-line VGroup, skip
                        continue
            
            # Check 3: Make sure we're not inside a for/while/if block that continues
            # Look at the previous line to see if it's a for/while/if without a colon at the end
            if i > 0:
                prev_line = lines[i - 1].strip()
                if prev_line and not prev_line.endswith(':') and prev_line.endswith('\\'):
                    # Previous line has line continuation, skip
                    continue
            
            # Extract variable name
            import re
            var_match = re.match(r'\s*(\w+)\s*=\s*VGroup\(', line)
            if var_match:
                var_name = var_match.group(1)
                indent = line[:len(line) - len(line.lstrip())]
                
                # Add auto-scaling code after this line
                # Screen is 12 units wide x 6 units tall (safe zone: 11x5.5)
                scale_code = f"{indent}# Auto-scale to fit on screen\n"
                scale_code += f"{indent}if {var_name}.height > 5.5:\n"
                scale_code += f"{indent}    {var_name}.scale_to_fit_height(5.5)\n"
                scale_code += f"{indent}if {var_name}.width > 11:\n"
                scale_code += f"{indent}    {var_name}.scale_to_fit_width(11)"
                
                processed_lines.append(scale_code)
                scale_added += 1
    
    ai_code = '\n'.join(processed_lines)
    
    if scale_added > 0:
        print(f"   ✅ Added auto-scaling to {scale_added} VGroups")
    else:
        print("   ✅ No VGroups detected")

    # ============================================================
    # POST-PROCESSING: RUNTIME BOUNDS CHECK FOR VGROUPS
    # ============================================================
    # Problem: VGroups can be positioned off-screen AFTER creation
    # Solution: Add runtime bounds check AFTER each VGroup is positioned
    # IMPROVED: Only apply to VGroups that use .arrange() (structured content)
    
    print("   🔧 POST-PROCESSING: Adding runtime bounds checking...")
    
    lines = ai_code.split('\n')
    insertions = []  # (line_index, var_name, indent) - collect insertions
    
    # Find VGroups that are positioned and use .arrange()
    for i, line in enumerate(lines):
        # Match VGroup positioning: vgroup_name.shift/move_to/next_to
        match = re.match(r'(\s*)(\w+)\.(shift|move_to|next_to)\(', line)
        if match:
            indent = match.group(1)
            var_name = match.group(2)
            
            # Check if this variable was defined as a VGroup with .arrange() earlier
            is_arranged_vgroup = False
            for j in range(max(0, i-30), i):
                if f'{var_name} = VGroup(' in lines[j] or f'{var_name} = VGroup (' in lines[j]:
                    # Also check if it uses .arrange() (structured content more likely to overflow)
                    for k in range(j, min(j+5, i)):
                        if f'{var_name}.arrange(' in lines[k] or '.arrange(' in lines[j]:
                            is_arranged_vgroup = True
                            break
                    break
            
            if is_arranged_vgroup:
                insertions.append((i + 1, var_name, indent))
    
    # Apply insertions in REVERSE order
    bounds_added = 0
    for line_idx, var_name, indent in reversed(insertions):
        # Use moderate bounds - catch off-screen but allow edge positioning
        bounds_code = f"{indent}# Bounds check for {var_name}\n"
        bounds_code += f"{indent}if {var_name}.get_bottom()[1] < -3.5:\n"
        bounds_code += f"{indent}    {var_name}.shift(UP * (-3.2 - {var_name}.get_bottom()[1]))\n"
        bounds_code += f"{indent}if {var_name}.get_top()[1] > 3.8:\n"
        bounds_code += f"{indent}    {var_name}.shift(DOWN * ({var_name}.get_top()[1] - 3.5))\n"
        bounds_code += f"{indent}if {var_name}.get_right()[0] > 7:\n"
        bounds_code += f"{indent}    {var_name}.shift(LEFT * ({var_name}.get_right()[0] - 6.5))"
        
        lines.insert(line_idx, bounds_code)
        bounds_added += 1
    
    if bounds_added > 0:
        print(f"      ✅ Added bounds checking for {bounds_added} arranged VGroups")
    else:
        print("      ✅ No arranged VGroups needing bounds check")
    
    ai_code = '\n'.join(lines)

    # ============================================================
    # POST-PROCESSING: DETECT AND FIX OFF-SCREEN POSITIONING
    # ============================================================
    # Problem: Objects positioned with .shift(LEFT*15) go way off-screen
    # Solution: Clamp EXTREME shift values (allows moderate off-screen for animations)
    
    print("   🔧 POST-PROCESSING: Checking for extreme off-screen positions...")
    
    import re
    lines = ai_code.split('\n')
    fixed_lines = []
    clamp_count = 0
    
    for line in lines:
        # Detect .shift() with large values
        # Pattern: .shift(LEFT * 3.5) or .shift(UP * 4.0)
        shift_matches = re.findall(r'\.shift\((LEFT|RIGHT|UP|DOWN)\s*\*\s*([\d.]+)\)', line)
        
        if shift_matches:
            original_line = line
            for direction, value in shift_matches:
                value_float = float(value)
                
                # Only clamp EXTREME values (clearly mistakes)
                # Allow moderate off-screen for slide-in animations
                max_horizontal = 8.0  # Increased from 6.0
                max_vertical = 6.0    # Increased from 5.0
                
                if direction in ['LEFT', 'RIGHT'] and value_float > max_horizontal:
                    # Clamp to safe value
                    line = line.replace(f'{direction} * {value}', f'{direction} * {max_horizontal}')
                    clamp_count += 1
                    print(f"      ⚠️  Clamped .shift({direction}*{value}) → {direction}*{max_horizontal}")
                
                elif direction in ['UP', 'DOWN'] and value_float > max_vertical:
                    # Clamp to safe value
                    line = line.replace(f'{direction} * {value}', f'{direction} * {max_vertical}')
                    clamp_count += 1
                    print(f"      ⚠️  Clamped .shift({direction}*{value}) → {direction}*{max_vertical}")
        
        # Also check for absolute positions in .move_to([x, y, z])
        # Only check SIMPLE numeric values (not expressions)
        move_to_match = re.search(r'\.move_to\(\[(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\]', line)
        if move_to_match:
            try:
                x = float(move_to_match.group(1))
                y = float(move_to_match.group(2))
                z = float(move_to_match.group(3))
                
                # Only clamp EXTREME values (clearly mistakes, not intentional)
                # Screen is roughly -7 to 7 horizontal, -4 to 4 vertical
                max_x = 7.0  # More lenient
                max_y = 5.0  # More lenient
                
                clamped = False
                new_x = x
                new_y = y
                
                # Only clamp if REALLY extreme (beyond visible screen)
                if abs(x) > max_x:
                    new_x = max_x if x > 0 else -max_x
                    clamped = True
                
                if abs(y) > max_y:
                    new_y = max_y if y > 0 else -max_y
                    clamped = True
                
                if clamped:
                    old_coords = f"[{x}, {y}, {z}]"
                    new_coords = f"[{new_x}, {new_y}, {z}]"
                    line = line.replace(f".move_to({old_coords})", f".move_to({new_coords})")
                    clamp_count += 1
                    print(f"      ⚠️  Clamped .move_to({old_coords}) → {new_coords}")
            except:
                pass  # Skip if conversion fails
        
        fixed_lines.append(line)
    
    ai_code = '\n'.join(fixed_lines)
    
    if clamp_count > 0:
        print(f"   ✅ Clamped {clamp_count} extreme off-screen positions")
    else:
        print("   ✅ No extreme off-screen positions detected")

    # ============================================================
    # POST-PROCESSING: FIX TEXT MIRRORING DURING SWAP ANIMATIONS
    # ============================================================
    # Problem: When two elements swap positions using .animate.shift() in
    # opposite directions (LEFT and RIGHT), text can appear mirrored/flipped
    # This happens because Manim interpolates the transform including rotation
    # Solution: Add path_arc=0 to swap animations to prevent rotation
    
    sorting_keywords_check = ['sort', 'bubble', 'insertion', 'selection', 'quick', 'merge', 'swap']
    is_sorting_animation = any(kw in narration.lower() for kw in sorting_keywords_check)
    
    if is_sorting_animation:
        print("   🔧 POST-PROCESSING: Fixing potential text mirroring in swap animations...")
        
        lines = ai_code.split('\n')
        swap_fixes = 0
        
        for i, line in enumerate(lines):
            # Detect swap animation patterns:
            # Pattern 1: self.play(elem1.animate.shift(LEFT*X), elem2.animate.shift(RIGHT*Y))
            # Pattern 2: self.play(elem1.animate.shift(RIGHT*X), elem2.animate.shift(LEFT*Y))
            if 'self.play(' in line and '.animate.shift(' in line:
                # Check if this is a swap (has both LEFT and RIGHT in same line)
                has_left = 'LEFT' in line
                has_right = 'RIGHT' in line
                
                if has_left and has_right:
                    # This looks like a swap animation - add path_arc=0 to prevent rotation
                    # Check if path_arc is already specified
                    if 'path_arc' not in line:
                        # Add path_arc=0 before the closing parenthesis of self.play()
                        # Find the run_time parameter if it exists
                        if 'run_time=' in line:
                            # Insert path_arc=0 before run_time
                            line = re.sub(r'(run_time\s*=)', r'path_arc=0, \1', line)
                        else:
                            # Add both path_arc=0 and ensure proper closing
                            # Replace the last ) with , path_arc=0)
                            # But be careful not to mess up nested parentheses
                            if line.rstrip().endswith(')'):
                                line = line.rstrip()[:-1] + ', path_arc=0)'
                        
                        lines[i] = line
                        swap_fixes += 1
                        print(f"      ✅ Added path_arc=0 to swap animation at line {i+1}")
        
        if swap_fixes > 0:
            ai_code = '\n'.join(lines)
            # Validate syntax
            ai_code = safe_postprocess(ai_code, ai_code, "swap animation fix")
            print(f"   ✅ Fixed {swap_fixes} swap animations to prevent text mirroring")
        else:
            print("   ✅ No swap animations needing path_arc fix detected")
    
    # ============================================================
    # POST-PROCESSING: FIX WRONG INDEX HIGHLIGHTING IN BUBBLE SORT
    # ============================================================
    # Problem: After bubble sort pass, AI highlights wrong element as
    # "largest in correct position". It should highlight index (n-i-1)
    # where i is the pass number (0-indexed), but AI often uses wrong index
    # Solution: Detect bubble sort patterns and fix the highlight index
    
    if 'bubble' in narration.lower() and ('pass' in narration.lower() or 'largest' in narration.lower()):
        print("   🔧 POST-PROCESSING: Checking bubble sort index highlighting...")
        
        lines = ai_code.split('\n')
        bubble_fixes = 0
        
        # Find array length from code
        array_length = None
        for line in lines:
            # Pattern: values = [5, 3, 8, 4, 2] or arr = [6, 2, 8, 4, 1]
            array_match = re.search(r'(?:values|arr|array|numbers)\s*=\s*\[([^\]]+)\]', line)
            if array_match:
                elements = array_match.group(1).split(',')
                array_length = len(elements)
                break
        
        if array_length:
            print(f"      📊 Detected array length: {array_length}")
            
            # After Pass 1 (i=0), largest is at index n-1 (last element)
            # After Pass 2 (i=1), largest is at index n-2
            # etc.
            
            # Look for "After Pass X" patterns and corresponding highlights
            for i, line in enumerate(lines):
                # Detect pass number from comments or text
                pass_match = re.search(r'[Pp]ass\s*(\d+)|after.*pass.*(\d+)', line, re.IGNORECASE)
                if pass_match:
                    pass_num = int(pass_match.group(1) or pass_match.group(2))
                    # Correct index for "largest in position" after this pass
                    correct_index = array_length - pass_num
                    
                    # Look ahead for highlight/color changes on wrong index
                    for j in range(i+1, min(i+15, len(lines))):
                        check_line = lines[j]
                        
                        # Pattern: elements[X].animate.set_color(GREEN) or boxes[X].set_color(GREEN)
                        # where X is the highlighted index
                        highlight_match = re.search(r'(elements|boxes|array|squares)\[(\d+)\].*(?:set_color|animate)', check_line)
                        if highlight_match:
                            var_name = highlight_match.group(1)
                            current_index = int(highlight_match.group(2))
                            
                            # Check if this is highlighting wrong index
                            if current_index != correct_index and current_index < array_length:
                                # Also check if this is related to "largest" or "sorted" position
                                context_window = '\n'.join(lines[max(0,j-3):j+3]).lower()
                                if 'largest' in context_window or 'sorted' in context_window or 'correct position' in context_window:
                                    # Fix the index
                                    old_pattern = f'{var_name}[{current_index}]'
                                    new_pattern = f'{var_name}[{correct_index}]'
                                    lines[j] = lines[j].replace(old_pattern, new_pattern)
                                    bubble_fixes += 1
                                    print(f"      ✅ Fixed: {old_pattern} → {new_pattern} (Pass {pass_num}, correct largest at index {correct_index})")
        
        if bubble_fixes > 0:
            ai_code = '\n'.join(lines)
            # Validate syntax
            ai_code = safe_postprocess(ai_code, ai_code, "bubble sort index fix")
            print(f"   ✅ Fixed {bubble_fixes} bubble sort index highlighting issues")
        else:
            print("   ✅ Bubble sort index highlighting looks correct")

    # ============================================================
    # POST-PROCESSING: REMOVE TEXT LABEL ANIMATIONS (PREVENT FLYING)
    # ============================================================
    # Problem: AI animates text labels to move to different positions
    # causing "text flying" effect mid-slide. Labels should stay put!
    # Patterns to detect and remove:
    #   - label.animate.move_to(...)
    #   - label.animate.shift(...)
    #   - text.animate.move_to(...)
    # Solution: Comment out these animations (keep labels in original position)
    
    print("   🔧 POST-PROCESSING: Removing text label animations (prevent flying)...")
    
    lines = ai_code.split('\n')
    label_anim_fixes = 0
    
    # EXACT variable names that are definitely labels (SAFE to remove animations)
    # Using exact matches to avoid false positives like "label_box" or "text_container"
    exact_label_names = [
        'label', 'text', 'title', 'header', 'caption', 'subtitle',
        'label1', 'label2', 'label3', 'label4', 'label5',
        'text1', 'text2', 'text3', 'text4', 'text5',
        'title_text', 'header_text', 'caption_text',
        'step_label', 'state_label', 'info_label', 'status_label',
        'name_label', 'value_label', 'desc_label', 'description_label',
        'array_label', 'box_label', 'node_label', 'cell_label',
        'left_label', 'right_label', 'top_label', 'bottom_label',
        'a_label', 'b_label', 'c_label', 'x_label', 'y_label',
        'result_label', 'output_label', 'input_label',
        'sorted_label', 'unsorted_label', 'comparing_label',
        'interference_label', 'observer_label', 'source_label', 'detection_label',
    ]
    
    for i, line in enumerate(lines):
        # Skip lines that are already comments
        if line.strip().startswith('#'):
            continue
            
        # Pattern: self.play(label.animate.move_to(...)) or self.play(label.animate.shift(...))
        
        # Check if line has .animate.move_to() or .animate.shift()
        if '.animate.move_to(' in line or '.animate.shift(' in line:
            # Extract the variable name before .animate
            anim_match = re.search(r'(\w+)\.animate\.(?:move_to|shift)\(', line)
            if anim_match:
                var_name = anim_match.group(1).lower()
                
                # SAFE CHECK: Only remove if variable name is EXACTLY in our safe list
                is_safe_label = var_name in exact_label_names
                
                if is_safe_label:
                    # Check if this is a standalone self.play() with ONLY this animation
                    if line.strip().startswith('self.play(') and line.strip().endswith(')'):
                        # Count animations in this line
                        anim_count = line.count('.animate.')
                        if anim_count == 1:
                            # Only one animation - comment out entire line
                            indent = len(line) - len(line.lstrip())
                            lines[i] = ' ' * indent + '# REMOVED (label animation): ' + line.strip()
                            label_anim_fixes += 1
                            print(f"      ✅ Removed label animation: {var_name}.animate at line {i+1}")
                        else:
                            # Multiple animations - just warn, don't touch
                            print(f"      ⚠️  Multi-animation line with {var_name} movement at line {i+1} (skipped)")
    
    if label_anim_fixes > 0:
        ai_code = '\n'.join(lines)
        ai_code = safe_postprocess(ai_code, ai_code, "label animation removal")
        print(f"   ✅ Removed {label_anim_fixes} text label animations to prevent flying")
    else:
        print("   ✅ No problematic text label animations detected")

    # ============================================================
    # POST-PROCESSING: FIX SORTED ARRAY NARRATION MISMATCH
    # ============================================================
    # Problem: Narration says "sorted array" or "final result" but code shows unsorted
    # Solution: Detect sorting keywords in narration + find array literals + sort them
    
    narration_lower = narration.lower()
    sorted_keywords = ['sorted', 'final result', 'in order', 'ascending', 'completed sort', 
                       'sorting complete', 'after sorting', 'sorted array', 'result is']
    
    # Only apply if narration suggests array should be sorted
    if any(kw in narration_lower for kw in sorted_keywords):
        print("   🔧 POST-PROCESSING: Checking for sorted array narration mismatch...")
        
        # Find array literals in code: [5, 3, 8, 1, 2] or values = [...]
        # Pattern matches: [num, num, num, ...] where nums are integers
        array_pattern = r'\[(\s*\d+\s*(?:,\s*\d+\s*)+)\]'
        
        def sort_array_if_needed(match):
            array_str = match.group(1)
            # Parse the numbers
            try:
                nums = [int(x.strip()) for x in array_str.split(',')]
                sorted_nums = sorted(nums)
                
                # Only replace if actually unsorted
                if nums != sorted_nums:
                    print(f"      ✅ Sorting array: {nums} → {sorted_nums}")
                    return '[' + ', '.join(str(n) for n in sorted_nums) + ']'
            except:
                pass
            return match.group(0)
        
        code_before_sort_fix = ai_code
        ai_code = re.sub(array_pattern, sort_array_if_needed, ai_code)
        
        # Validate syntax after sort fix
        try:
            ast.parse(ai_code)
        except SyntaxError:
            print(f"   ⚠️  REVERTING: Sort array fix broke syntax")
            ai_code = code_before_sort_fix
    else:
        print("   🔧 POST-PROCESSING: No sorting keywords in narration - skipping array sort check")
    
    # Dedent
    import textwrap
    ai_code = textwrap.dedent(ai_code).strip()
    
    # FINAL VALIDATION: Check syntax AFTER all post-processing
    # This catches errors introduced by our regex fixes
    try:
        import ast
        ast.parse(ai_code)
        print("   ✅ Final syntax validation passed")
    except SyntaxError as e:
        print(f"   ❌ CRITICAL: Post-processing introduced syntax error!")
        print(f"      Error at line {e.lineno}: {e.msg}")
        print(f"      This should NOT happen - our fixes broke the code!")
        # Print the problematic line
        if e.lineno:
            lines = ai_code.split('\n')
            if 0 <= e.lineno - 1 < len(lines):
                print(f"      Problematic line: {lines[e.lineno - 1]}")
                # Print surrounding context
                start = max(0, e.lineno - 3)
                end = min(len(lines), e.lineno + 2)
                print(f"      Context:")
                for i in range(start, end):
                    marker = "  >>> " if i == e.lineno - 1 else "      "
                    print(f"{marker}{i+1}: {lines[i]}")
        
        # CRITICAL: Raise exception so caller can retry with Claude
        # Don't silently continue with broken code!
        raise ValueError(f"Post-processing broke the code: {e.msg} at line {e.lineno}")
    
    # Only reached if validation passed
    return ai_code
# REAL FILE CHECK - Dec 29 2024 9:30PM
