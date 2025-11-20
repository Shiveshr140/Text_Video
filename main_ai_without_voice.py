# ============================================================================
# AI-BASED MANIM CODE GENERATOR
# ============================================================================

import openai
import os
import sys
from typing import Dict, List, Any, Optional
import ast
import re
import subprocess

# Import classes from manim.py
try:
    from manim import *
except ImportError:
    print("⚠️ Warning: manim module not found. Install with: pip install manim")
    sys.exit(1)

# Import custom classes from manim.py file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Import from pipeline.py
    import importlib.util
    spec = importlib.util.spec_from_file_location("pipeline_module", "pipeline.py")
    if spec and spec.loader:
        pipeline_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipeline_module)
        
        # Import required classes
        ParsedContent = pipeline_module.ParsedContent
        SemanticAnalyzer = pipeline_module.SemanticAnalyzer
        VisualStrategySelector = pipeline_module.VisualStrategySelector
        AudioGenerator = pipeline_module.AudioGenerator
        AdaptiveSceneFactory = pipeline_module.AdaptiveSceneFactory
except Exception as e:
    print(f"⚠️ Warning: Could not import from pipeline.py: {e}")
    sys.exit(1)

class OpenAICodeGenerator:
    """Generates Manim code using OpenAI GPT-4"""
    
    def __init__(self, api_key: str = None):
        """Initialize with OpenAI API key"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass it directly.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = "gpt-4"  # or "gpt-4-turbo" for faster responses
    
    def generate_manim_code(self, parsed_content: ParsedContent, visual_strategy: Dict[str, Any]) -> str:
        """Generate Manim scene code from parsed content"""
        
        # Build comprehensive prompt
        prompt = self._build_prompt(parsed_content, visual_strategy)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2500,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            generated_code = response.choices[0].message.content
            
            # Extract and clean the code
            cleaned_code = self._extract_code(generated_code)
            
            # Validate the code
            if self._validate_manim_code(cleaned_code):
                print("✅ AI generated valid Manim code")
                return cleaned_code
            else:
                print("⚠️ AI generated code needs fixing")
                return self._fix_common_issues(cleaned_code)
                
        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
            return self._fallback_code_generation(parsed_content)
    
    def _get_system_prompt(self) -> str:
        """System prompt that defines the AI's role"""
        return """You are an expert Manim animator. Generate WORKING Manim 0.19.0 code ONLY.

CRITICAL RULES - FOLLOW EXACTLY:
1. ONLY use: Text, Paragraph, Line, Arrow, Circle, Rectangle, Square, Polygon, VGroup
2. ONLY animations: Create, Write, FadeIn, FadeOut, Transform, Rotate, Scale
3. NEVER use: ShowCreation, WriteOn, IntegerLine, NumberPlane, SVGMobject, ImageMobject, Code
4. NEVER reference external files (SVG, images, etc.)
5. Keep it SIMPLE - text and basic shapes only
6. Start with: from manim import *
7. Class name: AIGeneratedScene(Scene)
8. Use self.play() and self.wait()

EXAMPLE WORKING CODE:
from manim import *

class AIGeneratedScene(Scene):
    def construct(self):
        title = Text("Title Here", font_size=48)
        title.to_edge(UP)
        self.play(FadeIn(title))
        
        content = Text("Content text here", font_size=36)
        content.move_to([0, 0, 0])
        self.play(Write(content))
        self.wait(2)
        
        self.play(FadeOut(title), FadeOut(content))

RETURN ONLY CODE - NO MARKDOWN, NO EXPLANATIONS"""
    
    def _build_prompt(self, parsed_content: ParsedContent, visual_strategy: Dict[str, Any]) -> str:
        """Build detailed prompt from parsed content"""
        
        prompt = f"""Generate a Manim scene for the following educational content:

MAIN CONCEPT: {parsed_content.main_concept}
CONTENT TYPE: {parsed_content.content_type}
COMPLEXITY: {parsed_content.complexity}
VISUAL STRATEGY: {visual_strategy.get('scene_types', [])}

CONTENT BLOCKS ({len(parsed_content.blocks)} blocks):
"""
        
        for i, block in enumerate(parsed_content.blocks, 1):
            prompt += f"""
Block {i}:
- Type: {block.content_type}
- Visual: {block.visual_type}
- Timing: {block.timing}s
- Text: {block.text[:200]}{"..." if len(block.text) > 200 else ""}
- Has Code: {block.metadata.get('has_code', False)}
- Keywords: {', '.join(block.metadata.get('keywords', [])[:5])}
"""
        
        prompt += f"""

VISUAL ELEMENTS TO USE: {', '.join(visual_strategy.get('visual_elements', []))}
TIMING ADJUSTMENTS: {visual_strategy.get('timing_adjustments', {})}

Generate a complete Manim Scene class that:
1. Creates an engaging visual presentation of this content
2. Uses appropriate animations for each content block
3. Includes proper timing and pacing
4. Handles code blocks with Code() objects if present
5. Uses colors and layout strategically
6. Adds transitions between sections
7. Total duration should be approximately {sum(b.timing for b in parsed_content.blocks):.1f} seconds

SCENE CLASS NAME: AIGeneratedScene
"""
        return prompt
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from AI response"""
        # Look for code blocks with python
        code_blocks = re.findall(r'```python\n?(.*?)```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # Look for generic code blocks
        code_blocks = re.findall(r'```\n?(.*?)```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # If no code blocks, find the actual code by looking for "from manim import"
        lines = response.split('\n')
        code_start = -1
        for i, line in enumerate(lines):
            if 'from manim import' in line:
                code_start = i
                break
        
        if code_start >= 0:
            # Extract from that line to the end (or until explanation starts)
            code_lines = []
            for line in lines[code_start:]:
                # Stop if we hit markdown or explanation
                if line.strip().startswith('#') and 'example' in line.lower():
                    break
                code_lines.append(line)
            return '\n'.join(code_lines).strip()
        
        # If still nothing, return the whole response
        return response.strip()
    
    def _validate_manim_code(self, code: str) -> bool:
        """Validate that generated code is syntactically correct"""
        try:
            # Check Python syntax
            ast.parse(code)
            
            # Check for required Manim elements
            required_elements = [
                "from manim import",
                "class",
                "Scene",
                "def construct",
                "self"
            ]
            
            for element in required_elements:
                if element not in code:
                    print(f"⚠️ Missing required element: {element}")
                    return False
            
            return True
            
        except SyntaxError as e:
            print(f"⚠️ Syntax error in generated code: {e}")
            print(f"Code preview (first 500 chars):\n{code[:500]}")
            return False
        except Exception as e:
            print(f"⚠️ Error validating code: {e}")
            return False
    
    def _fix_common_issues(self, code: str) -> str:
        """Fix common issues in AI-generated code"""
        print("🔧 Attempting to fix code issues...")
        
        # Fix deprecated/unsupported functions
        code = code.replace("ShowCreation", "Create")
        code = code.replace("WriteOn", "Write")
        code = code.replace("FadeInFrom", "FadeIn")
        code = code.replace("FadeOutAndShift", "FadeOut")
        
        # Remove unsupported objects - replace with Text
        code = re.sub(r'IntegerLine\([^)]*\)', 'Text("Integers")', code)
        code = re.sub(r'NumberPlane\([^)]*\)', 'Text("Plane")', code)
        code = re.sub(r'SVGMobject\("[^"]*"\)', 'Text("SVG")', code)
        code = re.sub(r'ImageMobject\("[^"]*"\)', 'Text("Image")', code)
        
        # Remove Code blocks - replace with Text
        code = re.sub(r'Code\([^)]*\)', 'Text("Code block")', code)
        
        # Ensure imports
        if "from manim import" not in code:
            code = "from manim import *\n\n" + code
        
        # Fix unclosed strings (common AI error)
        lines = code.split('\n')
        fixed_lines = []
        for i, line in enumerate(lines):
            # Check if line has unclosed string
            single_quotes = line.count("'") - line.count("\\'")
            double_quotes = line.count('"') - line.count('\\"')
            
            # If odd number of quotes (unclosed), try to fix
            if (single_quotes % 2 != 0 or double_quotes % 2 != 0) and '"' in line:
                # Try to close double quotes
                if double_quotes % 2 != 0:
                    if not line.rstrip().endswith('"') and not line.rstrip().endswith('\\'):
                        line = line.rstrip() + '"'
            
            fixed_lines.append(line)
        code = '\n'.join(fixed_lines)
        
        # Add self.wait() if missing and has animations
        if "self.play(" in code and "self.wait()" not in code:
            # Add wait at the end
            if not code.rstrip().endswith(')'):
                code += "\n        self.wait(1)"
        
        return code
    
    def _fallback_code_generation(self, parsed_content: ParsedContent) -> str:
        """Fallback code if AI generation fails"""
        return f'''from manim import *

class AIGeneratedScene(Scene):
    def construct(self):
        # Fallback scene - AI generation failed
        title = Text("{parsed_content.main_concept[:40]}", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        content = Text("Content type: {parsed_content.content_type}", font_size=28)
        content.move_to([0, 0, 0])
        self.play(FadeIn(content))
        self.wait(2)
        
        self.play(FadeOut(title), FadeOut(content))
'''


class AICodePostProcessor:
    """Post-processes AI-generated code for optimization"""
    
    def process(self, code: str, parsed_content: ParsedContent) -> str:
        """Apply dynamic adjustments to generated code"""
        
        # Adjust timing based on complexity
        code = self._adjust_timing(code, parsed_content.complexity)
        
        # Optimize animations
        code = self._optimize_animations(code)
        
        # Add audio sync markers (comments for timing)
        code = self._add_audio_markers(code, parsed_content)
        
        return code
    
    def _adjust_timing(self, code: str, complexity: str) -> str:
        """Adjust animation timing based on complexity"""
        timing_multipliers = {
            "beginner": 1.2,    # Slower for beginners
            "intermediate": 1.0,
            "advanced": 0.8     # Faster for advanced
        }
        
        multiplier = timing_multipliers.get(complexity, 1.0)
        
        # Adjust wait times
        pattern = r'self\.wait\((\d+\.?\d*)\)'
        
        def adjust_wait(match):
            original_time = float(match.group(1))
            new_time = original_time * multiplier
            return f'self.wait({new_time:.1f})'
        
        return re.sub(pattern, adjust_wait, code)
    
    def _optimize_animations(self, code: str) -> str:
        """Optimize animation sequences"""
        
        # Combine multiple FadeIn calls into one
        # Replace: self.play(FadeIn(a))\nself.play(FadeIn(b))
        # With: self.play(FadeIn(a), FadeIn(b))
        
        lines = code.split('\n')
        optimized_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check if current and next line are both FadeIn
            if 'self.play(FadeIn' in line and i + 1 < len(lines) and 'self.play(FadeIn' in lines[i + 1]:
                # Combine them
                combined = line.replace(')', ', ') + lines[i + 1].split('FadeIn')[1]
                optimized_lines.append(combined)
                i += 2
            else:
                optimized_lines.append(line)
                i += 1
        
        return '\n'.join(optimized_lines)
    
    def _add_audio_markers(self, code: str, parsed_content: ParsedContent) -> str:
        """Add timing comments for audio synchronization"""
        
        # Add comments indicating where each content block should align
        marker_code = f"\n        # Audio sync: {len(parsed_content.blocks)} blocks\n"
        
        for i, block in enumerate(parsed_content.blocks):
            marker_code += f"        # Block {i+1}: {block.timing:.1f}s - {block.content_type}\n"
        
        # Insert after construct method definition
        return code.replace("def construct(self):", f"def construct(self):{marker_code}")


class AIEnhancedSceneFactory:
    """Scene factory that uses AI-generated code"""
    
    def __init__(self, openai_api_key: str = None):
        self.code_generator = OpenAICodeGenerator(openai_api_key)
        self.post_processor = AICodePostProcessor()
    
    def create_ai_scene(self, parsed_content: ParsedContent, visual_strategy: Dict[str, Any]) -> tuple[str, str]:
        """Generate Manim scene code using AI"""
        
        print("🤖 Generating Manim code with AI...")
        raw_code = self.code_generator.generate_manim_code(parsed_content, visual_strategy)
        
        print("⚙️ Post-processing generated code...")
        processed_code = self.post_processor.process(raw_code, parsed_content)
        
        # Save to file for execution
        scene_file = "/Users/apple/Desktop/manim/.temp_ai_scene.py"
        with open(scene_file, 'w') as f:
            f.write(processed_code)
        
        print(f"✅ AI-generated scene saved to: {scene_file}")
        
        return processed_code, scene_file
    
    def render_ai_scene(self, scene_file: str, output_name: str = "ai_generated") -> str:
        """Render the AI-generated scene"""
        import subprocess
        
        try:
            cmd = [
                "/Users/apple/Desktop/manim/.venv/bin/python", "-m", "manim",
                "-pql", scene_file, "AIGeneratedScene"
            ]
            
            print("🎬 Rendering AI-generated scene...")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Find the generated video
            video_path = f"media/videos/{os.path.basename(scene_file).replace('.py', '')}/480p15/AIGeneratedScene.mp4"
            
            if os.path.exists(video_path):
                print(f"✅ Video rendered: {video_path}")
                return video_path
            else:
                print(f"❌ Video file not found at: {video_path}")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Video rendering failed: {e}")
            print(f"Error output: {e.stderr}")
            return None


# ============================================================================
# UPDATED PIPELINE WITH AI INTEGRATION
# ============================================================================

class AIEnhancedTextToVideoPipeline:
    """Enhanced pipeline with AI code generation"""
    
    def __init__(self, openai_api_key: str = None, use_ai: bool = True):
        self.semantic_analyzer = SemanticAnalyzer()
        self.visual_strategist = VisualStrategySelector()
        self.audio_generator = AudioGenerator()
        
        # AI or traditional scene factory
        self.use_ai = use_ai
        if use_ai:
            self.ai_scene_factory = AIEnhancedSceneFactory(openai_api_key)
        else:
            self.scene_factory = AdaptiveSceneFactory()
    
    def process(self, raw_text: str, output_name: str = "generated_video") -> Dict[str, Any]:
        """Process text into video using AI-generated Manim code"""
        
        print("🔍 Analyzing content...")
        parsed_content = self.semantic_analyzer.analyze(raw_text)
        
        print(f"📊 Content Type: {parsed_content.content_type}")
        print(f"🧠 Main Concept: {parsed_content.main_concept}")
        print(f"📈 Complexity: {parsed_content.complexity}")
        
        print("🎨 Selecting visual strategy...")
        visual_strategy = self.visual_strategist.select_strategy(
            parsed_content.content_type, parsed_content
        )
        
        if self.use_ai:
            print("🤖 Using AI to generate Manim code...")
            generated_code, scene_file = self.ai_scene_factory.create_ai_scene(
                parsed_content, visual_strategy
            )
            
            print("🎥 Rendering AI-generated video...")
            video_file = self.ai_scene_factory.render_ai_scene(scene_file, output_name)
        else:
            print("🎬 Using traditional scene generation...")
            video_file = self._render_traditional_video(raw_text, output_name)
        
        print("🔊 Generating audio...")
        audio_file = self.audio_generator.generate_audio(raw_text, f"{output_name}_audio.aiff")
        
        print("🎬 Combining video and audio...")
        final_video = self.combine_video_audio(video_file, audio_file, output_name)
        
        print(f"✅ FINAL VIDEO READY: {final_video}")
        
        result = {
            "parsed_content": parsed_content,
            "visual_strategy": visual_strategy,
            "audio_file": audio_file,
            "video_file": video_file,
            "final_video": final_video,
            "output_name": output_name
        }
        
        if self.use_ai:
            result["generated_code"] = generated_code
        
        return result
    
    def _render_traditional_video(self, raw_text: str, output_name: str) -> str:
        """Render video using traditional scene generation"""
        import os
        
        # Write content to file for pipeline.py to read (use same path as pipeline.py)
        CONTENT_FILE_PATH = "/Users/apple/Desktop/manim/.temp_content.txt"
        
        # Remove old content file if exists to force fresh render
        if os.path.exists(CONTENT_FILE_PATH):
            os.remove(CONTENT_FILE_PATH)
        
        with open(CONTENT_FILE_PATH, 'w') as f:
            f.write(raw_text)
        
        # Use the pipeline.py scene
        try:
            # Get the directory where app.py is located
            app_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Use the venv Python if it exists
            venv_python = os.path.join(app_dir, "venv", "bin", "python")
            if not os.path.exists(venv_python):
                venv_python = "python3"
            
            # Change to app directory to run manim
            original_dir = os.getcwd()
            os.chdir(app_dir)
            
            cmd = [
                venv_python, "-m", "manim",
                "--disable_caching",  # Force re-render every time
                "-pql", "pipeline.py", "UniversalContentScene"
            ]
            print("📹 Rendering video...")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=app_dir)
            
            # Change back to original directory
            os.chdir(original_dir)
            
            # Find the generated video file
            video_path = os.path.join(app_dir, "media", "videos", "pipeline", "480p15", "UniversalContentScene.mp4")
            
            if os.path.exists(video_path):
                print(f"✅ Video rendered: {video_path}")
                # Clean up temp file
                if os.path.exists(CONTENT_FILE_PATH):
                    os.remove(CONTENT_FILE_PATH)
                return video_path
            else:
                print(f"❌ Video file not found at: {video_path}")
                return None
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Video rendering failed: {e}")
            print(f"Error output: {e.stderr if hasattr(e, 'stderr') else 'No stderr'}")
            print(f"Return code: {e.returncode if hasattr(e, 'returncode') else 'Unknown'}")
            # Clean up temp file
            if os.path.exists(CONTENT_FILE_PATH):
                os.remove(CONTENT_FILE_PATH)
            return None
    
    def combine_video_audio(self, video_file: str, audio_file: str, output_name: str) -> str:
        """Combine video and audio"""
        if not video_file or not audio_file:
            print("❌ Missing video or audio file")
            return None
            
        final_output = f"{output_name}_final.mp4"
        
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_file,
                "-i", audio_file,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                final_output
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return final_output
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Video combination failed: {e}")
            return None


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def create_video_with_ai(text: str, output_name: str = "my_video", use_ai: bool = True):
    """Create video using AI-generated Manim code"""
    
    # Get API key from environment or pass directly
    api_key = os.getenv("OPENAI_API_KEY")
    
    if use_ai and not api_key:
        print("⚠️ No OpenAI API key found. Falling back to traditional generation.")
        use_ai = False
    
    pipeline = AIEnhancedTextToVideoPipeline(openai_api_key=api_key, use_ai=use_ai)
    result = pipeline.process(text, output_name)
    
    if result['final_video']:
        print(f"\n🎉 SUCCESS! Your video is ready:")
        print(f"📁 File: {result['final_video']}")
        if use_ai:
            print(f"🤖 Used AI-generated Manim code")
            print(f"📝 Generated code available in result['generated_code']")
        return result
    else:
        print("❌ Video creation failed!")
        return None


# Example usage:
if __name__ == "__main__":
    # Set your OpenAI API key
    # os.environ["OPENAI_API_KEY"] = "sk-your-key-here"
    
    sample_text = """
    Merge Sort Algorithm
    
    Merge sort is a divide-and-conquer algorithm that divides the input array into two halves,
    recursively sorts them, and then merges the sorted halves.
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    
    The algorithm works in three steps:
    1. Divide: Split the array into two halves
    2. Conquer: Recursively sort each half
    3. Combine: Merge the sorted halves
    """
    
    result = create_video_with_ai(sample_text, "merge_sort_demo", use_ai=True)
