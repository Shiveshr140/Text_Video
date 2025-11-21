import os
import sys
import subprocess
import re
import openai
from pipeline import AudioGenerator
import json
import wave
import contextlib
def transcribe_audio_with_timestamps(audio_file):
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(
            audio_file,
            word_timestamps=True,
            language="en"
        )
        words = []
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                words.append({
                    "word": word_info.get("word", "").strip(),
                    "start": word_info.get("start", 0),
                    "end": word_info.get("end", 0)
                })
        return words
    except ImportError:
        return None
    except Exception as e:
        return None
def get_audio_duration(audio_file):
    try:
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
        if os.path.exists(wav_file):
            os.remove(wav_file)
        return duration
    except Exception as e:
        return None
def generate_slides_code(parsed_data, audio_duration=None, num_sections=None):
    title = parsed_data.get("title", "Presentation")
    sections = parsed_data.get("sections", [])
    for idx, section in enumerate(sections):
        heading = section.get("heading", "")
        content = section.get("content", "")
        content_lines = []
        words = content.split()
        current_line = []
        max_chars_per_line = 80
        for word in words:
            test_line = current_line + [word]
            test_length = sum(len(w) for w in test_line) + len(test_line) - 1
            if test_length <= max_chars_per_line:
                current_line.append(word)
            else:
                if current_line:
                    content_lines.append(" ".join(current_line))
                current_line = [word]
        if current_line:
            content_lines.append(" ".join(current_line))
        if audio_duration and num_sections:
            total_overhead = 2.0 + (num_sections * 2.7)
            available_time = audio_duration - total_overhead
            time_per_section = max(available_time / num_sections, 3.0)
            wait_time = time_per_section
        else:
            word_count = len(content.split())
            reading_time = word_count / 2.5
            animation_overhead = 2.7
            wait_time = max(reading_time - animation_overhead, 3.0)
        escaped_heading = heading.replace('"', '\\"')
        for line_idx, line in enumerate(content_lines):
            escaped_line = line.replace('"', '\\"')
        for line_idx in range(len(content_lines)):
            code += f"line_{idx}_{line_idx}"
            if line_idx < len(content_lines) - 1:
                code += ", "
    return code
def text_to_video(text_content: str, output_name: str = "output", audio_language: str = "english"):
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        client = openai.OpenAI(api_key=api_key)
        parse_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
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
        english_narration = f"{parsed_data.get('title', '')}. "
        for section in parsed_data.get('sections', []):
            heading = section.get('heading', '')
            content = section.get('content', '')
            english_narration += f"{heading}. {content}. "
        if audio_language != "english":
            narration_text = create_code_mixed_narration(
                english_narration, 
                audio_language, 
                client
            )
        else:
            narration_text = english_narration
        audio_file = f"{output_name}_audio.aiff"
        generate_audio_for_language(narration_text, audio_language, audio_file, client)
        audio_duration = get_audio_duration(audio_file)
        if audio_duration:
        else:
        manim_code = generate_slides_code(parsed_data, audio_duration, num_sections)
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
            return None
        if not os.path.exists(video_path):
            return None
        if not os.path.exists(audio_file):
            return None
        video_size = os.path.getsize(video_path)
        audio_size = os.path.getsize(audio_file)
        if video_size == 0:
            return None
        if audio_size == 0:
            return None
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
            if not os.path.exists(final_output):
                return None
            final_size = os.path.getsize(final_output)
            if final_size == 0:
                return None
            try:
                duration_cmd = [
                    "ffprobe", "-v", "error", "-show_entries",
                    "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                    final_output
                ]
                duration_result = subprocess.run(duration_cmd, check=True, capture_output=True, text=True)
                duration = float(duration_result.stdout.strip())
            except:
                pass
        except subprocess.CalledProcessError as e:
            if e.stderr:
            if e.stdout:
            return None
        return {"final_video": final_output}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None
def create_code_mixed_narration(english_narration, target_language, client):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in creating natural code-mixed Indian language content."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()
TTS_CONFIG = {
    "english": {
        "provider": "elevenlabs",
        "voice_id": None,
        "model": "eleven_multilingual_v2",
        "stability": 0.5,
        "similarity_boost": 0.75
    },
    "hindi": {
        "provider": "elevenlabs",
        "voice_id": None,
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
    config = TTS_CONFIG.get(language, TTS_CONFIG["english"])
    if config.get("provider") == "elevenlabs":
        try:
            from elevenlabs import VoiceSettings
            from elevenlabs.client import ElevenLabs
            elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
            voice_id = config.get("voice_id") or os.getenv("ELEVENLABS_VOICE_ID")
            if not elevenlabs_key:
                return None
            if not voice_id:
                return None
            eleven_client = ElevenLabs(api_key=elevenlabs_key)
            audio_generator = eleven_client.text_to_speech.convert(
                voice_id=voice_id,
                text="... " + narration_text,
                model_id=config["model"],
                voice_settings=VoiceSettings(
                    stability=config["stability"],
                    similarity_boost=config["similarity_boost"]
                )
            )
            with open(output_file, "wb") as f:
                for chunk in audio_generator:
                    f.write(chunk)
            return output_file
        except ImportError:
        except Exception as e:
    tts_response = client.audio.speech.create(
        model="tts-1-hd",
        voice="alloy",
        input="... " + narration_text,
        speed=1.0
    )
    tts_response.stream_to_file(output_file)
    return output_file
def animation_to_video(prompt: str, output_name: str = "output"):
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        client = openai.OpenAI(api_key=api_key)
        code_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
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
        if "```python" in manim_code:
            manim_code = manim_code.split("```python")[1].split("```")[0].strip()
        elif "```" in manim_code:
            manim_code = manim_code.split("```")[1].split("```")[0].strip()
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
        audio_file = f"{output_name}_audio.aiff"
        tts_response = client.audio.speech.create(
            model="tts-1-hd",
            voice="shimmer",
            input="... " + narration_text,
            speed=1.05
        )
        tts_response.stream_to_file(audio_file)
        scene_file = f".temp_{output_name}.py"
        with open(scene_file, 'w') as f:
            f.write(manim_code)
        scene_match = re.search(r'class\s+(\w+)\s*\(Scene\)', manim_code)
        if not scene_match:
            return None
        scene_class = scene_match.group(1)
        cmd = [
            "venv/bin/python", "-m", "manim",
            "-pql", scene_file, scene_class
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        video_path = f"media/videos/{scene_file.replace('.py', '')}/480p15/{scene_class}.mp4"
        if not os.path.exists(video_path):
            return None
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
        return {"final_video": final_output}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None
def find_block_end(lines, start_line):
    brace_count = 0
    for i in range(start_line - 1, len(lines)):
        line = lines[i]
        brace_count += line.count('{') - line.count('}')
        if brace_count == 0 and i > start_line - 1:
            return i + 1
    return len(lines)
def parse_code_to_blocks(code_content, language="python"):
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
            return blocks
        except SyntaxError:
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
    valid_files = []
    for audio_file in audio_files:
        if not os.path.exists(audio_file):
            continue
        file_size = os.path.getsize(audio_file)
        if file_size == 0:
            continue
        valid_files.append(audio_file)
    if not valid_files:
        return False
    if len(valid_files) < len(audio_files):
    try:
        silence_file = None
        if silence_duration > 0:
            silence_file = 'silence_temp.aiff'
            result = subprocess.run([
                'ffmpeg', '-y', '-f', 'lavfi',
                '-i', f'anullsrc=channel_layout=mono:sample_rate=44100',
                '-t', str(silence_duration),
                silence_file
            ], check=True, capture_output=True, text=True)
            if not os.path.exists(silence_file) or os.path.getsize(silence_file) == 0:
                return False
        input_args = []
        filter_parts = []
        input_index = 0
        if silence_file:
            input_args.extend(['-i', silence_file])
            filter_parts.append(f"[{input_index}:a]")
            input_index += 1
        for audio_file in valid_files:
            input_args.extend(['-i', audio_file])
            filter_parts.append(f"[{input_index}:a]")
            input_index += 1
        filter_complex = "".join(filter_parts) + f"concat=n={len(filter_parts)}:v=0:a=1[outa]"
        cmd = [
            'ffmpeg', '-y'
        ] + input_args + [
            '-filter_complex', filter_complex,
            '-map', '[outa]',
            '-ar', '44100',
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            output_file
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if not os.path.exists(output_file):
            raise Exception("Output file not created")
        output_size = os.path.getsize(output_file)
        if output_size == 0:
            raise Exception("Output file is empty")
        if os.path.exists('concat_list.txt'):
            os.remove('concat_list.txt')
        if silence_file and os.path.exists(silence_file):
            os.remove(silence_file)
        return True
    except subprocess.CalledProcessError as e:
        if e.stderr:
    except Exception as e:
    try:
        wav_files = []
        for i, audio_file in enumerate(valid_files):
            wav_file = f"temp_{i}.wav"
            result = subprocess.run([
                'ffmpeg', '-y', '-i', audio_file, wav_file
            ], check=True, capture_output=True, text=True)
            if not os.path.exists(wav_file) or os.path.getsize(wav_file) == 0:
                continue
            wav_files.append(wav_file)
        if not wav_files:
            return False
        silence_wav = None
        if silence_duration > 0:
            silence_wav = 'silence_temp.wav'
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
        result = subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', 'concat_list.txt',
            '-ar', '44100',
            '-ac', '1',
            combined_wav
        ], check=True, capture_output=True, text=True)
        if not os.path.exists(combined_wav) or os.path.getsize(combined_wav) == 0:
            return False
        result = subprocess.run([
            'ffmpeg', '-y', '-i', combined_wav,
            '-ar', '44100',
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            output_file
        ], check=True, capture_output=True, text=True)
        if silence_wav and os.path.exists(silence_wav):
            os.remove(silence_wav)
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            return False
        output_size = os.path.getsize(output_file)
        for wav_file in wav_files:
            if os.path.exists(wav_file):
                os.remove(wav_file)
        if os.path.exists(combined_wav):
            os.remove(combined_wav)
        if os.path.exists('concat_list.txt'):
            os.remove('concat_list.txt')
        if silence_duration > 0:
            silence_file = 'silence_temp.aiff'
            if os.path.exists(silence_file):
                os.remove(silence_file)
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False
def generate_timeline_animations(code_content, timeline_events, audio_duration, key_concepts, key_concepts_start_time):
    if not timeline_events or len(timeline_events) == 0:
        return None
    original_lines = code_content.split('\n')
    non_blank_lines = []
    line_number_mapping = {}
    new_line_num = 1
    for orig_line_num, line in enumerate(original_lines, start=1):
        if line.strip():
            non_blank_lines.append(line)
            line_number_mapping[orig_line_num] = new_line_num
            new_line_num += 1
    cleaned_code = '\n'.join(non_blank_lines)
    num_original_lines = len(original_lines)
    num_cleaned_lines = len(non_blank_lines)
    num_blank_lines = num_original_lines - num_cleaned_lines
    escaped_code = cleaned_code.replace('\\', '\\\\').replace('"', '\\"')
    fixed_overhead = 1.2 + 0.5 + 1.0 + 0.5
    if timeline_events:
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
        code_scroll_time = len(cleaned_code.split('\n')) * 0.5
        concepts_slide_time = 0
    if key_concepts and concepts_slide_time > 0:
        concepts_list = "[" + ", ".join(['"' + c.replace('"', '\\"') + '"' for c in key_concepts]) + "]"
        concepts_display_time = concepts_slide_time - 1.8
        concepts_display_time = max(concepts_display_time, 2.0)
    else:
        concepts_slide_code = "        self.wait(1)\n"
    highlight_creation = ""
    highlight_list = []
    cleaned_code_lines = cleaned_code.split('\n')
    num_code_lines = len(cleaned_code_lines)
    test_code = '\n'.join(cleaned_code_lines[:min(5, num_code_lines)])
    try:
        from manim import Text
        test_text = Text(
            test_code,
            font_size=22,
            font="Courier",
            line_spacing=1.0
        )
        if num_code_lines > 0:
            line_height = (22 / 32.0) * 1.0
        else:
            line_height = 0.7
    except:
        line_height = 0.7
    for event_idx, event in enumerate(timeline_events):
        block = event['code_block']
        start_line = block['start_line']
        end_line = block['end_line']
        original_start = block['start_line']
        original_end = block['end_line']
        num_non_blank_lines = 0
        for orig_line in range(original_start, original_end + 1):
            if orig_line in line_number_mapping:
                num_non_blank_lines += 1
        if num_non_blank_lines == 0:
            num_non_blank_lines = original_end - original_start + 1
        num_lines_in_block = num_non_blank_lines
        block_height = num_lines_in_block * line_height
        block_type = block.get('type', 'code_block')
        if 'loop' in block_type:
            highlight_color = "#4A90E2"
        elif 'if' in block_type or 'else' in block_type:
            highlight_color = "#50C878"
        elif 'function' in block_type or 'method' in block_type:
            highlight_color = "#FF6B6B"
        elif 'class' in block_type:
            highlight_color = "#9B59B6"
        else:
            highlight_color = "#FFD700"
        highlight_list.append(f"highlight_{event_idx}")
        highlight_list.append(f"highlight_glow_{event_idx}")
    first_code_line_idx = 0
    highlight_positioning = ""
    for event_idx, event in enumerate(timeline_events):
        block = event['code_block']
        original_start_line = block['start_line']
        original_end_line = block['end_line']
        new_start_line = None
        new_end_line = None
        for orig_line in range(original_start_line, original_end_line + 1):
            if orig_line in line_number_mapping:
                if new_start_line is None:
                    new_start_line = line_number_mapping[orig_line]
                new_end_line = line_number_mapping[orig_line]
        if new_start_line is None or new_end_line is None:
            new_start_line = original_start_line
            new_end_line = original_end_line
        center_line = ((new_start_line + new_end_line) / 2.0) - 1
        num_lines_in_cleaned_block = new_end_line - new_start_line + 1
        indent = "            "
        highlight_positioning += f"{indent}block_{event_idx}_center_y = full_code.get_top()[1] - ({center_line} * {line_height:.3f})\n"
        highlight_positioning += f"{indent}highlight_{event_idx}.move_to([full_code.get_center()[0], block_{event_idx}_center_y, 0])\n"
        highlight_positioning += f"{indent}highlight_{event_idx}.stretch_to_fit_width(full_code.width + 0.3)\n"
        highlight_positioning += f"{indent}# Highlight height: {block_height:.3f} (for {num_lines_in_block} lines)\n"
        highlight_positioning += f"{indent}highlight_{event_idx}.set_z_index(-1)\n"
        highlight_positioning += f"{indent}highlight_{event_idx}.set_opacity(0)\n"
        highlight_positioning += f"{indent}highlight_{event_idx}.set_stroke_opacity(0)\n"
        highlight_positioning += f"{indent}highlight_glow_{event_idx}.move_to([full_code.get_center()[0], block_{event_idx}_center_y, 0])\n"
        highlight_positioning += f"{indent}highlight_glow_{event_idx}.stretch_to_fit_width(full_code.width + 0.3)\n"
        highlight_positioning += f"{indent}highlight_glow_{event_idx}.set_z_index(-2)\n"
        highlight_positioning += f"{indent}highlight_glow_{event_idx}.set_opacity(0)\n"
        highlight_positioning += f"{indent}highlight_glow_{event_idx}.set_stroke_opacity(0)\n"
    timeline_events.sort(key=lambda x: x['start_time'])
    animation_timeline = ""
    current_time = fixed_overhead
    previous_highlight_idx = None
    for event_idx, event in enumerate(timeline_events):
        start_time = event['start_time']
        end_time = event['end_time']
        block = event['code_block']
        wait_time = start_time - current_time
        original_start_line = block['start_line']
        original_end_line = block['end_line']
        new_start_line = None
        new_end_line = None
        for orig_line in range(original_start_line, original_end_line + 1):
            if orig_line in line_number_mapping:
                if new_start_line is None:
                    new_start_line = line_number_mapping[orig_line]
                new_end_line = line_number_mapping[orig_line]
        if new_start_line is None or new_end_line is None:
            new_start_line = original_start_line
            new_end_line = original_end_line
        block_start_line = new_start_line
        block_end_line = new_end_line
        block_center_line = (block_start_line + block_end_line) / 2
        cleaned_code_lines = cleaned_code.split('\n')
        total_code_lines = len(cleaned_code_lines)
        total_non_blank_lines = total_code_lines
        visible_lines = 18
        effective_total_lines = total_non_blank_lines if total_non_blank_lines > 0 else total_code_lines
        if effective_total_lines > visible_lines:
            block_position_ratio = (block_center_line - 1) / (total_code_lines - 1) if total_code_lines > 1 else 0
            if block_center_line <= visible_lines / 2:
                scroll_progress = 0.0
            else:
                effective_block_center = block_center_line - 1
                scroll_progress = (effective_block_center - visible_lines / 2) / max(effective_total_lines - visible_lines, 1)
                scroll_progress = min(max(scroll_progress, 0), 1)
        else:
            scroll_progress = 0.0
        if event_idx == 0:
            if wait_time <= 0:
                wait_time = 0.5
        if wait_time > 0:
        else:
        current_time = start_time
        if previous_highlight_idx is not None:
        highlight_duration = end_time - start_time
        remaining_time = highlight_duration - 1.1
        if remaining_time > 0:
        else:
        previous_highlight_idx = event_idx
        current_time = end_time
    if previous_highlight_idx is not None:
    code_narration_end_time = max([event['end_time'] for event in timeline_events]) if timeline_events else current_time
    wait_until_code_ends = max(0, code_narration_end_time - current_time)
    if wait_until_code_ends > 0:
        current_time = code_narration_end_time
    else:
    concepts_animation_time = 1.8 if key_concepts and concepts_slide_time > 0 else 0
    if key_concepts_start_time and concepts_slide_time > 0:
        concepts_display_time = concepts_slide_time - 1.8
        concepts_display_time = max(concepts_display_time, 2.0)
        concepts_slide_end_time = key_concepts_start_time + 1.8 + concepts_display_time
    else:
        concepts_slide_end_time = current_time + 0.5 + 0.2
    remaining_time_after_all = 0
    if audio_duration and audio_duration > concepts_slide_end_time:
        remaining_time_after_all = audio_duration - concepts_slide_end_time
        remaining_time_after_all = max(remaining_time_after_all, 0)
        if remaining_time_after_all < 0.5:
            remaining_time_after_all = 0
    if len(highlight_list) == 0:
    return code
def generate_code_display_code(code_content, audio_duration=None, narration_segments=None, key_concepts=None, key_concepts_start_time=None):
    lines = code_content.strip().split('\n')
    escaped_code = code_content.replace('\\', '\\\\').replace('"', '\\"')
    fixed_overhead = 1.2 + 0.5 + 1.0 + 0.5
    if audio_duration and key_concepts_start_time:
        transition_time = 0.7
        code_scroll_time = key_concepts_start_time - fixed_overhead - transition_time
        code_scroll_time = max(code_scroll_time, 5.0)
        concepts_slide_time = audio_duration - key_concepts_start_time
        concepts_slide_time = max(concepts_slide_time, 3.0)
    elif audio_duration:
        available_time = audio_duration - fixed_overhead - 1.0
        code_scroll_time = max(available_time, 5.0)
        concepts_slide_time = 0
    else:
        code_scroll_time = len(lines) * 0.5
        concepts_slide_time = 0
    if key_concepts and concepts_slide_time > 0:
        concepts_list = "[" + ", ".join(['"' + c.replace('"', '\\"') + '"' for c in key_concepts]) + "]"
        concepts_display_time = concepts_slide_time - 1.8
        concepts_display_time = max(concepts_display_time, 2.0)
    else:
    return code
def code_to_video(code_content: str, output_name: str = "output", audio_language: str = "english"):
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        client = openai.OpenAI(api_key=api_key)
        language = "python"
        if "public class" in code_content or "public static" in code_content:
            language = "java"
        elif "function" in code_content and "{" in code_content:
            language = "javascript"
        code_blocks = parse_code_to_blocks(code_content, language=language)
        for i, block in enumerate(code_blocks, 1):
            block_size = block['end_line'] - block['start_line'] + 1
        all_blocks_for_narration = code_blocks.copy()
        for i, block in enumerate(all_blocks_for_narration, 1):
            block_size = block['end_line'] - block['start_line'] + 1
        blocks_for_highlights = []
        for block in code_blocks:
            block_size = block['end_line'] - block['start_line'] + 1
            if block['type'] == 'class':
                continue
            else:
                contains_other_blocks = False
                contained_blocks = []
                for other_block in code_blocks:
                    if other_block != block:
                        if (block['start_line'] < other_block['start_line'] and 
                            block['end_line'] > other_block['end_line']):
                            contains_other_blocks = True
                            contained_blocks.append(other_block)
                if contains_other_blocks:
                else:
                    blocks_for_highlights.append(block)
        def is_outer_block(block, block_list):
            for other_block in block_list:
                if other_block != block:
                    if (other_block['start_line'] < block['start_line'] and 
                        other_block['end_line'] >= block['end_line']):
                        return False
                    if (other_block['start_line'] == block['start_line'] and 
                        other_block['end_line'] > block['end_line']):
                        return False
            return True
        outer_narration_blocks = [b for b in all_blocks_for_narration if is_outer_block(b, all_blocks_for_narration)]
        nested_narration_blocks = [b for b in all_blocks_for_narration if not is_outer_block(b, all_blocks_for_narration)]
        outer_narration_blocks.sort(key=lambda x: x['start_line'])
        nested_narration_blocks.sort(key=lambda x: x['start_line'])
        all_blocks_for_narration = outer_narration_blocks + nested_narration_blocks
        outer_highlight_blocks = [b for b in blocks_for_highlights if is_outer_block(b, blocks_for_highlights)]
        nested_highlight_blocks = [b for b in blocks_for_highlights if not is_outer_block(b, blocks_for_highlights)]
        outer_highlight_blocks.sort(key=lambda x: x['start_line'])
        nested_highlight_blocks.sort(key=lambda x: x['start_line'])
        blocks_for_highlights = outer_highlight_blocks + nested_highlight_blocks
        for i, block in enumerate(all_blocks_for_narration, 1):
            block_size = block['end_line'] - block['start_line'] + 1
        for i, block in enumerate(blocks_for_highlights, 1):
            block_size = block['end_line'] - block['start_line'] + 1
        if len(all_blocks_for_narration) == 0:
            manim_code = generate_code_display_code(code_content, audio_duration, None, key_concepts, key_concepts_start_time)
        else:
            block_narrations = []
            block_audios = []
            timeline_events = []
            cumulative_time = 0.0
            intro_delay = 3.2
            for block_idx, code_block in enumerate(all_blocks_for_narration):
                block_type = code_block.get('type', 'code_block')
                block_code = code_block.get('code', '')
                start_line = code_block.get('start_line', 0)
                end_line = code_block.get('end_line', 0)
                if block_idx == 0:
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
                else:
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
                block_audio_file = f"{output_name}_block_{block_idx}_audio.aiff"
                audio_result = generate_audio_for_language(block_narration, audio_language, block_audio_file, client)
                if not os.path.exists(block_audio_file):
                    continue
                audio_file_size = os.path.getsize(block_audio_file)
                if audio_file_size == 0:
                    continue
                block_duration = get_audio_duration(block_audio_file)
                if not block_duration:
                    block_duration = len(block_narration) * 0.06
                else:
                block_audios.append(block_audio_file)
                will_have_highlight = any(
                    b['start_line'] == code_block['start_line'] and 
                    b['end_line'] == code_block['end_line'] and
                    b['type'] == code_block['type']
                    for b in blocks_for_highlights
                )
                if not will_have_highlight:
                    for i, hb in enumerate(blocks_for_highlights, 1):
                        match_start = hb['start_line'] == code_block['start_line']
                        match_end = hb['end_line'] == code_block['end_line']
                        match_type = hb['type'] == code_block['type']
                else:
                if will_have_highlight:
                    event_start = cumulative_time + intro_delay
                    event_end = cumulative_time + block_duration + intro_delay
                    actual_highlight_duration = event_end - event_start
                    min_highlight_duration = 1.5
                    if actual_highlight_duration < min_highlight_duration:
                        if block_idx < len(all_blocks_for_narration) - 1:
                            next_block_start = (cumulative_time + block_duration) + intro_delay
                        else:
                            next_block_start = float('inf')
                        extended_end = min(event_start + min_highlight_duration, next_block_start)
                        if extended_end > event_end:
                            event_end = extended_end
                    timeline_events.append({
                        'code_block': code_block,
                        'start_time': event_start,
                        'end_time': event_end,
                        'audio_start': cumulative_time + intro_delay,
                        'audio_end': cumulative_time + block_duration + intro_delay,
                        'confidence': 1.0,
                        'sentence': block_idx + 1
                    })
                    actual_highlight_duration = event_end - event_start
                    if actual_highlight_duration == block_duration:
                    else:
                else:
                cumulative_time += block_duration
            if not block_audios:
                return None
            audio_file = f"{output_name}_audio.aiff"
            if not concatenate_audio_files(block_audios, audio_file, silence_duration=intro_delay):
                if block_audios and os.path.exists(block_audios[0]):
                    first_audio = block_audios[0]
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
                            if os.path.exists(silence_file):
                                os.remove(silence_file)
                            if os.path.exists('concat_fallback.txt'):
                                os.remove('concat_fallback.txt')
                            if os.path.exists(audio_file) and os.path.getsize(audio_file) > 0:
                            else:
                                return None
                        else:
                            return None
                    except Exception as e:
                        return None
                else:
                    return None
            if not os.path.exists(audio_file):
                return None
            audio_file_size = os.path.getsize(audio_file)
            if audio_file_size == 0:
                return None
            for block_audio in block_audios:
                if os.path.exists(block_audio) and block_audio != audio_file:
                    try:
                        os.remove(block_audio)
                    except:
                        pass
            try:
                duration_cmd = [
                    "ffprobe", "-v", "error", "-show_entries",
                    "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                    audio_file
                ]
                duration_result = subprocess.run(duration_cmd, check=True, capture_output=True, text=True)
                audio_duration = float(duration_result.stdout.strip())
            except Exception as e:
                audio_duration = get_audio_duration(audio_file)
                if not audio_duration:
                    audio_duration = cumulative_time + intro_delay
                else:
            for idx, event in enumerate(timeline_events):
                block = event['code_block']
            combined_narration = " ".join(block_narrations)
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
            key_concepts_audio_file = None
            if key_concepts:
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
                key_concepts_audio_file = f"{output_name}_key_concepts_audio.aiff"
                generate_audio_for_language(concepts_narration, audio_language, key_concepts_audio_file, client)
                if os.path.exists(key_concepts_audio_file) and os.path.getsize(key_concepts_audio_file) > 0:
                    combined_audio_files = [audio_file, key_concepts_audio_file]
                    final_audio_file = f"{output_name}_audio_with_concepts.aiff"
                    if concatenate_audio_files(combined_audio_files, final_audio_file, silence_duration=0.0):
                        old_audio_file = audio_file
                        audio_file = final_audio_file
                        try:
                            duration_cmd = [
                                "ffprobe", "-v", "error", "-show_entries",
                                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                                audio_file
                            ]
                            duration_result = subprocess.run(duration_cmd, check=True, capture_output=True, text=True)
                            audio_duration = float(duration_result.stdout.strip())
                        except:
                            pass
                        if os.path.exists(old_audio_file) and old_audio_file != audio_file:
                            try:
                                os.remove(old_audio_file)
                            except:
                                pass
                    else:
                else:
            key_concepts_start_time = None
            if key_concepts and audio_duration:
                code_narration_end_time = max([e['end_time'] for e in timeline_events]) if timeline_events else None
                if code_narration_end_time:
                    key_concepts_start_time = code_narration_end_time + 0.7
                else:
            manim_code = generate_timeline_animations(code_content, timeline_events, audio_duration, key_concepts, key_concepts_start_time)
        scene_file = f".temp_{output_name}.py"
        with open(scene_file, 'w') as f:
            f.write(manim_code)
        file_size = os.path.getsize(scene_file)
        venv_python = "venv311/bin/python" if os.path.exists("venv311/bin/python") else "venv/bin/python"
        cmd = [
            venv_python, "-m", "manim",
            "-pql", scene_file, "CodeExplanationScene"
        ]
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=False,
                text=True
            )
        except subprocess.CalledProcessError as e:
            if hasattr(e, 'stderr') and e.stderr:
            return None
        video_path = f"media/videos/{scene_file.replace('.py', '')}/480p15/CodeExplanationScene.mp4"
        if not os.path.exists(video_path):
            return None
        if not os.path.exists(video_path):
            return None
        if not os.path.exists(audio_file):
            return None
        video_size = os.path.getsize(video_path)
        audio_size = os.path.getsize(audio_file)
        if video_size == 0:
            return None
        if audio_size == 0:
            return None
        try:
            probe_cmd = ["ffprobe", "-v", "error", video_path]
            probe_result = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            return None
        except Exception as e:
        try:
            probe_cmd = ["ffprobe", "-v", "error", audio_file]
            probe_result = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            return None
        except Exception as e:
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
        except Exception as e:
        final_output = f"{output_name}_final.mp4"
        combine_cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_file,
        ]
        if video_duration and audio_duration_actual and audio_duration_actual < video_duration:
            loop_count = int(video_duration / audio_duration_actual) + 1
            combine_cmd.extend([
                "-filter_complex", f"[1:a]aloop=loop={loop_count}:size=2e+09[a]",
                "-c:v", "copy",
            "-c:a", "aac",
                "-b:a", "192k",
                "-map", "0:v",
                "-map", "[a]",
                "-shortest",
            ])
        elif video_duration and audio_duration_actual and video_duration < audio_duration_actual:
            combine_cmd.extend([
                "-filter_complex", f"[1:a]atrim=0:{video_duration}[a]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-map", "0:v",
                "-map", "[a]",
            ])
        else:
            combine_cmd.extend([
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
            "-shortest",
            ])
        combine_cmd.append(final_output)
        try:
            result = subprocess.run(combine_cmd, check=True, capture_output=True, text=True)
            if result.stdout:
            if result.stderr:
            if not os.path.exists(final_output):
                return None
            final_size = os.path.getsize(final_output)
            if final_size == 0:
                if result.stderr:
                return None
            try:
                duration_cmd = [
                    "ffprobe", "-v", "error", "-show_entries",
                    "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            final_output
        ]
                duration_result = subprocess.run(duration_cmd, check=True, capture_output=True, text=True)
                duration = float(duration_result.stdout.strip())
            except:
                pass
        except subprocess.CalledProcessError as e:
            if e.stderr:
            if e.stdout:
            if os.path.exists(final_output):
                file_size = os.path.getsize(final_output)
                if file_size == 0:
                    try:
                        os.remove(final_output)
                    except:
                        pass
            return None
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None
        return {"final_video": final_output}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None
if __name__ == "__main__":
    result = animation_to_video(
        prompt="Bubble Sort algorithm - visualize array sorting with comparisons and swaps",
        output_name="bubble_sort_animation"
    )
    if not result:
        sys.exit(1)