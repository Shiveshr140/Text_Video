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


# ============================================================
# HELPER: Generate Controlled Manim Code (No Random AI)
# ============================================================

def generate_slides_code(parsed_data):
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
            font_size=42,
            gradient=(BLUE, GREEN)
        )
        title.to_edge(UP)
        self.play(Write(title), run_time=0.5)
        self.wait(1)
        
""".format(title=title)
    
    # Generate code for each section
    for idx, section in enumerate(sections):
        heading = section.get("heading", "")
        content = section.get("content", "")
        
        # Split long content into multiple lines (wider width)
        content_lines = []
        words = content.split()
        current_line = []
        char_count = 0
        
        for word in words:
            if char_count + len(word) + 1 > 65:  # Max 65 chars per line (wider)
                content_lines.append(" ".join(current_line))
                current_line = [word]
                char_count = len(word)
            else:
                current_line.append(word)
                char_count += len(word) + 1
        
        if current_line:
            content_lines.append(" ".join(current_line))
        
        # Calculate timing based on content length (more time per slide)
        # Assuming average reading speed: 3 words per second
        word_count = len(content.split())
        wait_time = max(word_count / 2.5, 5.0)  # At least 5 seconds per slide
        
        code += f"""        # Section {idx + 1}: {heading}
        heading_{idx} = Text(
            "{heading}",
            font_size=36,
            color=BLUE
        )
        heading_{idx}.next_to(title, DOWN, buff=0.8)
        self.play(FadeIn(heading_{idx}), run_time=0.5)
        self.wait(0.5)
        
        # Content lines
        content_group_{idx} = VGroup()
"""
        
        for line_idx, line in enumerate(content_lines):
            code += f"""        line_{idx}_{line_idx} = Text(
            "{line}",
            font_size=28,
            color=WHITE
        )
"""
        
        code += f"""        
        # Position content lines
        content_group_{idx}.add("""
        
        for line_idx in range(len(content_lines)):
            code += f"line_{idx}_{line_idx}"
            if line_idx < len(content_lines) - 1:
                code += ", "
        
        code += f""")
        content_group_{idx}.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        content_group_{idx}.next_to(heading_{idx}, DOWN, buff=0.6)
        content_group_{idx}.scale_to_fit_width(13)  # Maximum width possible
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

def text_to_video(text_content: str, output_name: str = "output"):
    """
    Convert ANY text content to educational video with audio
    
    Works for:
    - Technical explanations
    - Educational content
    - Documentation
    - Tutorials
    - Any written content
    
    Args:
        text_content: The text you want to convert to video
        output_name: Name for output file
        
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
        print(f"✅ Created {len(parsed_data.get('sections', []))} sections\n")
        
        # Step 2: Generate Manim code (CONTROLLED - Not random AI)
        print("🎨 Generating Manim slide code...")
        
        manim_code = generate_slides_code(parsed_data)
        
        print("✅ Manim code generated\n")
        
        # Step 3: Create narration from SAME parsed sections (synchronized!)
        print("🎙️ Creating narration script...")
        
        # Build narration from the SAME sections shown on screen
        narration_text = f"{parsed_data.get('title', '')}. "
        
        for section in parsed_data.get('sections', []):
            heading = section.get('heading', '')
            content = section.get('content', '')
            narration_text += f"{heading}. {content}. "
        
        print("✅ Narration script created (synchronized with slides)\n")
        
        # Step 4: Generate audio
        print("🔊 Generating audio...")
        audio_file = f"{output_name}_audio.aiff"
        
        tts_response = client.audio.speech.create(
            model="tts-1-hd",
            voice="shimmer",
            input="... " + narration_text,
            speed=0.95  # Slower to match longer slide times
        )
        
        tts_response.stream_to_file(audio_file)
        print(f"✅ Audio saved: {audio_file}\n")
        
        # Step 5: Render video
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
        
        # Step 6: Combine
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
# EXAMPLES
# ============================================================

if __name__ == "__main__":
    
    print("""
╔══════════════════════════════════════════════════════════╗
║         PRODUCTION TEXT-TO-VIDEO SYSTEM                  ║
║                                                          ║
║  Two functions. No code changes. Works for everything.   ║
╚══════════════════════════════════════════════════════════╝
    """)
    
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