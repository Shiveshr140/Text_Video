# ============================================================================
# AI-BASED MANIM CODE GENERATOR WITH HIGH-QUALITY TTS
# ============================================================================
#
# TTS (Text-to-Speech) Provider Options:
# - "openai": Premium quality, natural-sounding voices (requires OpenAI API key)
# - "macos": Built-in macOS TTS (free, decent quality)
# 
# OpenAI TTS Features:
# - Model: tts-1-hd (High Definition)
# - Voice: alloy (Neutral voice optimized for Indian accent)
# - Speed: 0.75 (Slower for Indian accent clarity)
# - Indian Pronunciation: Text preprocessing for Indian-English patterns
# - Audio-Video Sync: 1.5s initial delay for perfect synchronization
# - Format: Converted to AIFF for pipeline compatibility
# - AI Model: GPT-4o (Latest and most capable)
# - Content Coverage: 16-20 sections ensuring ALL content is included
#
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
        self.model = "gpt-4"  # Using GPT-4, the most capable available model
    
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
            
            response_content = response.choices[0].message.content
            
            # Try to extract JSON plan first
            json_plan = self._extract_json(response_content)
            
            print(f"🔍 AI Response (first 500 chars): {response_content[:500]}...")
            print(f"🔍 AI Response FULL JSON:")
            print(response_content)
            
            if json_plan:
                print("✅ Found JSON plan, generating code from plan")
                # Generate Manim code from JSON plan
                cleaned_code = self._generate_code_from_plan(json_plan)
            else:
                print("⚠️ No JSON found, falling back to direct code extraction")
                # Fallback to old method
                cleaned_code = self._extract_code(response_content)
            
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
        return """CRITICAL: You MUST return ONLY valid JSON. No code, no explanations, no markdown.

Your response should be EXACTLY this format:
{
  "title": "Educational Topic Name",
  "sections": [
    {"text": "Short sentence 1", "y_position": 2.0, "wait_time": 5},
    {"text": "Short sentence 2", "y_position": 1.5, "wait_time": 5},
    {"text": "Short sentence 3", "y_position": 1.0, "wait_time": 5}
  ]
}

RULES:
- Create exactly 16-20 UNIQUE sections (NO DUPLICATES ALLOWED!)
- Each text: maximum 15 words
- Y positions: 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0
- Wait times: 5-6 seconds each (longer for audio sync)
- CRITICAL: NEVER REPEAT THE SAME TEXT IN MULTIPLE SECTIONS!
- For titles/headings: Use them only ONCE at the start of their section
- For content: Each unique piece of information should appear exactly ONCE
- Organize content logically:
  1. Start with main concept
  2. Then architecture/structure
  3. Then steps/process
  4. Then advantages
  5. Finally applications
- PRESERVE ALL TECHNICAL DETAILS: Don't simplify, keep original complexity
- INCLUDE ALL EXAMPLES: But don't repeat similar examples

RETURN ONLY JSON - NO OTHER TEXT ALLOWED"""
    
    def _get_code_system_prompt(self) -> str:
        """Fallback system prompt for direct code generation"""
        return """CRITICAL RULES - FOLLOW EXACTLY:
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
        # Title
        title = Text("Main Title", font_size=42, color=BLUE)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1)
        self.wait(0.5)
        
        # Content section 1
        text1 = Text("First important concept", font_size=32)
        text1.move_to([0, 1, 0])
        self.play(FadeIn(text1), run_time=1)
        self.wait(2)
        
        # Content section 2
        text2 = Text("Second key concept", font_size=32)
        text2.move_to([0, -1, 0])
        self.play(FadeIn(text2), run_time=1)
        self.wait(2)
        
        # More sections for longer video
        text3 = Text("Third important point", font_size=32)
        text3.move_to([0, -2.5, 0])
        self.play(FadeIn(text3), run_time=1)
        self.wait(2)
        
        # Fade out
        self.play(FadeOut(VGroup(title, text1, text2, text3)), run_time=1)

CRITICAL INSTRUCTIONS - MUST FOLLOW:
1. Create EXACTLY 8-10 separate Text() objects (one for each sentence!)
2. Each Text() object should contain MAXIMUM 10-15 words
3. Use font_size=28 (smaller to fit more on screen)
4. Position vertically spread out: y=2.5, 1.5, 0.5, -0.5, -1.5, -2.5, -3.0
5. Each section gets self.wait(4-5) minimum
6. SHOW ALL CONTENT - don't skip any sentences!
7. Total script should have 10+ Text() objects and 40+ seconds of wait time

RETURN ONLY CODE - NO MARKDOWN, NO EXPLANATIONS"""
    
    def _build_prompt(self, parsed_content: ParsedContent, visual_strategy: Dict[str, Any]) -> str:
        """Build detailed prompt from parsed content"""
        
        # Get full text content
        full_text = parsed_content.main_concept + "\n\n" + "\n\n".join([b.text for b in parsed_content.blocks[:10]])
        
        # Smart splitting: handle steps, lists, and regular sentences
        smart_sections = self._smart_split_content(full_text)
        
        # Create numbered list for AI
        numbered_sentences = [f"{i+1}. {sent}" for i, sent in enumerate(smart_sections[:20])]
        
        prompt = f"""Generate a JSON plan for these sentences. INCLUDE EVERY SINGLE PIECE OF CONTENT:

CONTENT TO ANALYZE:
{chr(10).join(numbered_sentences)}

CRITICAL INSTRUCTIONS:
- CREATE A SECTION FOR EVERY IMPORTANT SENTENCE
- If you see "Architecture Overview:", create a section for it
- If you see "Step 1:", "Step 2:", etc., create separate sections for each step
- If you see numbered lists (1., 2., 3.), create separate sections for each item  
- Include ALL technical details like "input layer, hidden layers, output layer"
- Include ALL application examples like "recommendation systems", "financial modeling"
- Don't skip ANY content - every sentence should be represented
- Create 16-20 sections to ensure complete coverage

MISSING CONTENT IS NOT ACCEPTABLE - INCLUDE EVERYTHING!

Return JSON with title and 16-20 sections covering ALL content.

SCENE CLASS NAME: AIGeneratedScene
"""
        return prompt
    
    def _smart_split_content(self, text: str) -> List[str]:
        """Smart splitting of content to handle steps, lists, and sentences"""
        import re
        
        sections = []
        
        # First, split by major patterns
        # Look for step patterns in the entire text
        step_pattern = r'Step \d+:[^S]*(?=Step \d+:|$)'
        steps = re.findall(step_pattern, text, re.IGNORECASE)
        
        if steps:
            # Found steps, use them
            for step in steps:
                sections.append(step.strip())
            
            # Also get any content before the first step
            before_steps = re.split(r'Step \d+:', text, flags=re.IGNORECASE)[0].strip()
            if before_steps:
                sentences = [s.strip() for s in before_steps.split('.') if s.strip() and len(s.strip()) > 10]
                sections = sentences + sections  # Put before steps
        else:
            # No steps found, process line by line
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for numbered lists: "1.", "2.", etc.
                if re.match(r'^\d+\.', line):
                    sections.append(line)
                    continue
                
                # Check for section headers (with colons)
                if ':' in line and len(line) < 60:
                    sections.append(line)
                    continue
                
                # Regular sentence splitting
                sentences = [s.strip() for s in line.split('.') if s.strip() and len(s.strip()) > 10]
                sections.extend(sentences)
        
        return sections[:20]  # Limit to 20 sections
    
    def _extract_json(self, response: str) -> Dict:
        """Extract JSON plan from AI response"""
        import json
        import re
        
        # Look for JSON in code blocks first
        json_blocks = re.findall(r'```json\n?(.*?)```', response, re.DOTALL)
        if json_blocks:
            try:
                return json.loads(json_blocks[0].strip())
            except Exception as e:
                print(f"Failed to parse JSON block: {e}")
        
        # Look for JSON anywhere - more aggressive pattern
        json_patterns = [
            r'\{[^{}]*"title"[^{}]*"sections"[^{}]*\[[^\]]*\][^{}]*\}',  # Look for title+sections
            r'\{.*?"title".*?"sections".*?\}',  # Simpler pattern
            r'\{[\s\S]*\}',  # Any object
        ]
        
        for pattern in json_patterns:
            json_match = re.search(pattern, response, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group()
                    return json.loads(json_str)
                except Exception as e:
                    print(f"Failed to parse JSON with pattern {pattern}: {e}")
                    continue
        
        return None
    
    def _generate_code_from_plan(self, plan: Dict) -> str:
        """Generate Manim code from JSON plan - multiple slides approach with audio sync"""
        code_lines = [
            "from manim import *",
            "",
            "class AIGeneratedScene(Scene):",
            "    def construct(self):",
            "        # AUDIO SYNC FIX: Match audio start with slight delay",
            "        self.wait(1.2)  # Initial delay to match audio start",
            ""
        ]
        
        sections = plan.get('sections', [])
        
        # Ensure no duplicate sections and maintain logical order
        seen_texts = set()
        unique_sections = []
        
        # First, add the title section if it exists
        title_text = plan.get('title', '').strip()
        if title_text:
            seen_texts.add(title_text.lower())
        
        # Process other sections, skipping duplicates
        for section in sections:
            text = section.get('text', '').strip()
            text_lower = text.lower()
            
            # Skip if empty or duplicate
            if not text or text_lower in seen_texts:
                print(f"⚠️ Skipping duplicate or empty section: {text}")
                continue
                
            # Skip if it's just the title repeated
            if text_lower == title_text.lower():
                print(f"⚠️ Skipping repeated title: {text}")
                continue
            
            seen_texts.add(text_lower)
            unique_sections.append(section)
            print(f"✅ Added unique section: {text}")
        
        print(f"📊 Original sections: {len(sections)} → Unique sections: {len(unique_sections)}")
        
        # Group unique sections into slides
        slides = []
        slide_size = 5  # Sections per slide
        for i in range(0, len(unique_sections), slide_size):
            slides.append(unique_sections[i:i + slide_size])
        
        # Generate code for each slide
        for slide_num, slide_sections in enumerate(slides):
            code_lines.append(f'        # Slide {slide_num + 1}')
            code_lines.append('        slide_elements = []')
            code_lines.append('')
            
            # Add title (always at top)
            title = plan.get('title', 'Content')
            code_lines.append(f'        title = Text("{title}", font_size=36, color=BLUE, weight=BOLD)')
            code_lines.append('        title.to_edge(UP, buff=0.2)')
            code_lines.append('        self.play(FadeIn(title))')
            code_lines.append('        slide_elements.append(title)')
            code_lines.append('        self.wait(0.5)')
            code_lines.append('')
            
            # Add sections vertically spaced
            y_positions = [2.2, 1.1, 0.0, -1.1, -2.2]  # 5 sections per slide, evenly spaced
            
            for idx, section in enumerate(slide_sections):
                if idx >= len(y_positions):
                    break
                    
                text = section.get('text', '').replace('"', '\\"')
                y_pos = y_positions[idx]
                # Calculate wait time based on text length with better pacing
                text_length = len(text.split())
                wait = min(max(text_length * 0.45, 3.0), 4.5)  # Slightly longer waits (3.0-4.5 seconds) for better sync
                
                code_lines.append(f'        text_{slide_num}_{idx} = Text("{text}", font_size=28)')
                code_lines.append(f'        text_{slide_num}_{idx}.move_to([0, {y_pos}, 0])')
                code_lines.append(f'        self.play(FadeIn(text_{slide_num}_{idx}), run_time=0.5)')  # Faster animations
                code_lines.append(f'        slide_elements.append(text_{slide_num}_{idx})')
                code_lines.append(f'        self.wait({wait})')
                code_lines.append('')
            
            # Smooth slide transitions with better timing
            if slide_num < len(slides) - 1:  # Not the last slide
                code_lines.append('        self.wait(0.5)')  # Hold before transition
                code_lines.append('        self.play(FadeOut(VGroup(*slide_elements)), run_time=0.4)')  # Smooth fade out
                code_lines.append('        self.wait(0.2)')  # Brief pause after fade
                code_lines.append('')
            else:
                code_lines.append('        self.wait(2)')  # Longer final wait
                code_lines.append('')
        
        return '\n'.join(code_lines)
    
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
    
    def __init__(self, openai_api_key: str = None, use_ai: bool = True, tts_provider: str = "openai"):
        self.semantic_analyzer = SemanticAnalyzer()
        self.visual_strategist = VisualStrategySelector()
        self.audio_generator = AudioGenerator(provider=tts_provider)
        
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
        if self.use_ai:
            # Use improved AI TTS with moderate speed
            audio_file = self._generate_ai_audio_moderate(raw_text, f"{output_name}_audio.aiff")
        else:
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
    
    def _generate_ai_audio_moderate(self, text: str, output_file: str) -> str:
        """Generate AI audio with moderate speed for proper sync"""
        try:
            import openai
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return self.audio_generator.generate_audio(text, output_file)
            
            client = openai.OpenAI(api_key=api_key)
            
            # Clean text
            clean_text = re.sub(r'```[^`]*```', '', text)
            clean_text = re.sub(r'\n+', ' ', clean_text)
            clean_text = re.sub(r'\s+', ' ', clean_text)
            clean_text = clean_text.strip()
            
            print("🎙️ Generating AI audio (Moderate Speed + Indian Accent)...")
            
            # Add small silence at start for sync
            clean_text = "... " + clean_text  # Add slight pause at start
            
            # Use shimmer voice for fresh, engaging educational content
            response = client.audio.speech.create(
                model="tts-1-hd",
                voice="shimmer",  # Fresh, energetic voice perfect for educational content
                input=clean_text,
                speed=1.05  # Very slightly faster to match video
            )
            
            # Save and convert
            temp_mp3 = output_file.replace('.aiff', '_temp.mp3')
            with open(temp_mp3, 'wb') as f:
                f.write(response.content)
            
            convert_cmd = [
                "ffmpeg", "-y", "-i", temp_mp3,
                "-acodec", "pcm_s16be", output_file
            ]
            subprocess.run(convert_cmd, check=True, capture_output=True)
            
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
            
            print(f"✅ AI audio generated with proper sync: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"❌ AI audio failed: {e}, using traditional")
            return self.audio_generator.generate_audio(text, output_file)
    
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

def create_video_with_ai(text: str, output_name: str = "my_video", use_ai: bool = True, tts_provider: str = "openai"):
    """Create video using AI-generated Manim code with high-quality TTS"""
    
    # Get API key from environment or pass directly
    api_key = os.getenv("OPENAI_API_KEY")
    
    if use_ai and not api_key:
        print("⚠️ No OpenAI API key found. Falling back to traditional generation.")
        use_ai = False
    
    pipeline = AIEnhancedTextToVideoPipeline(openai_api_key=api_key, use_ai=use_ai, tts_provider=tts_provider)
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


# ============================================================================
# DEMO SCENES WITH ACTUAL ANIMATIONS
# ============================================================================

# class AnimationDemoScene(ThreeDScene):
#     """Showcase of Manim's animation capabilities with synchronized narration"""
#     def construct(self):
#         # Initial delay for audio sync
#         self.wait(1.2)
        
#         # Title with gradient and animation
#         title = Text("Animation Showcase", gradient=(BLUE, PURPLE), font_size=48)
#         title.to_edge(UP)
#         self.play(Write(title), run_time=1.5)
        
#         # Subtitle
#         subtitle = Text("Mathematical Animations Made Beautiful", font_size=32, color=BLUE_B)
#         subtitle.next_to(title, DOWN)
#         self.play(FadeIn(subtitle))
#         self.wait(2)  # Wait for intro narration

#         # Clear title for shape section
#         self.play(
#             FadeOut(title),
#             FadeOut(subtitle)
#         )
        
#         # 1. Shape Morphing with explanation
#         section_title = Text("Shape Transformations", gradient=(BLUE, GREEN), font_size=40)
#         section_title.to_edge(UP)
#         self.play(Write(section_title))
        
#         # Create and position shapes
#         circle = Circle(radius=2, color=BLUE)
#         square = Square(side_length=4, color=PURPLE)
#         triangle = Triangle().scale(2).set_color(GREEN)
        
#         # Show initial circle with glow effect
#         circle.set_stroke(width=4)
#         glow = circle.copy().set_stroke(BLUE_A, width=8, opacity=0.5)
#         self.play(
#             Create(circle),
#             Create(glow, rate_func=there_and_back)
#         )
#         self.wait(1)
        
#         # Transform with ripple effect
#         self.play(
#             Transform(circle, square),
#             Transform(glow, square.copy().set_stroke(PURPLE_A, width=8, opacity=0.5))
#         )
#         self.wait(1)
        
#         self.play(
#             Transform(circle, triangle),
#             Transform(glow, triangle.copy().set_stroke(GREEN_A, width=8, opacity=0.5))
#         )
#         self.wait(1)
        
#         # Clear for next animation with fade
#         self.play(
#             FadeOut(circle),
#             FadeOut(glow),
#             FadeOut(section_title)
#         )

#         # 2. Mathematical Functions
#         section_title = Text("Mathematical Visualization", gradient=(YELLOW, RED), font_size=40)
#         section_title.to_edge(UP)
#         self.play(Write(section_title))
        
#         # Create axes with labels
#         axes = Axes(
#             x_range=[-3, 3],
#             y_range=[-2, 2],
#             axis_config={"color": BLUE},
#             tips=True
#         ).scale(0.8)
        
#         # Add labels
#         x_label = Text("x", font_size=24).next_to(axes.x_axis, RIGHT)
#         y_label = Text("sin(x)", font_size=24).next_to(axes.y_axis, UP)
        
#         # Create and style the sine graph
#         sin_graph = axes.plot(lambda x: np.sin(x), color=YELLOW)
#         sin_graph.set_stroke(width=4)
        
#         # Add moving dot with trail
#         dot = Dot(color=RED)
#         dot.move_to(axes.c2p(-3, np.sin(-3)))
        
#         # Create trailing effect
#         trail = VMobject()
#         trail.set_points_as_corners([dot.get_center(), dot.get_center()])
#         trail.set_stroke(RED, 2, opacity=0.8)
        
#         def update_trail(trail):
#             previous_path = trail.copy()
#             previous_path.add_points_as_corners([dot.get_center()])
#             trail.become(previous_path)
        
#         # Animate everything with proper timing
#         self.play(
#             Create(axes),
#             Write(x_label),
#             Write(y_label)
#         )
#         self.wait(0.5)
        
#         self.play(Create(sin_graph, run_time=1.5))
#         self.play(Create(dot))
        
#         # Add trail and animate dot
#         trail.add_updater(update_trail)
#         self.add(trail)
#         self.play(MoveAlongPath(dot, sin_graph), run_time=4, rate_func=linear)
#         self.wait(0.5)
        
#         # Highlight the wave nature
#         wave_arrow = Arrow(start=axes.c2p(0, 0), end=axes.c2p(PI/2, 1), color=WHITE)
#         wave_text = Text("Wave Motion", font_size=24, color=YELLOW).next_to(wave_arrow, RIGHT)
#         self.play(
#             Create(wave_arrow),
#             Write(wave_text)
#         )
#         self.wait(1)
        
#         # Clear for next animation
#         self.play(
#             FadeOut(VGroup(axes, sin_graph, dot, trail, wave_arrow, wave_text, x_label, y_label, section_title))
#         )

#         # 3. 3D Rotation Animation
#         cube = Cube(side_length=2, fill_opacity=0.7, fill_color=BLUE)
#         cube.set_stroke(WHITE, 1)
        
#         # Setup 3D camera
#         self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
#         self.begin_ambient_camera_rotation(rate=0.2)  # Add smooth camera rotation
        
#         self.play(Create(cube))
#         # Create a more complex 3D scene
#         self.play(
#             Rotate(cube, angle=PI/2, axis=RIGHT),
#             cube.animate.set_color(RED),
#             run_time=2
#         )
#         self.wait(0.5)
        
#         # Add some spheres around the cube
#         spheres = VGroup(*[
#             Sphere(radius=0.3, fill_opacity=0.8).move_to(
#                 cube.get_center() + np.array([2*np.cos(t), 2*np.sin(t), 0])
#             ).set_color(BLUE)
#             for t in np.linspace(0, 2*PI, 8)
#         ])
        
#         self.play(Create(spheres))
#         self.wait(0.5)
        
#         # Animate the spheres
#         self.play(
#             Rotate(spheres, angle=2*PI, axis=OUT),
#             Rotate(cube, angle=2*PI, axis=UP),
#             run_time=3
#         )
#         self.wait(1)
        
#         # Stop camera rotation and reset view
#         self.stop_ambient_camera_rotation()
        
#         # Clear for next section
#         self.play(
#             FadeOut(cube),
#             FadeOut(spheres)
#         )

#         # 4. Text Effects
#         text1 = Text("Beautiful", font_size=36).shift(UP)
#         text2 = Text("Animations", font_size=48)
#         text3 = Text("With Manim", font_size=36).shift(DOWN)
        
#         self.play(AddTextLetterByLetter(text1))
#         self.play(AddTextLetterByLetter(text2))
#         self.play(AddTextLetterByLetter(text3))
#         self.wait(0.5)
        
#         # Final flourish
#         final_group = VGroup(text1, text2, text3)
#         self.play(
#             final_group.animate.scale(1.2).set_color(YELLOW),
#             run_time=1
#         )
#         self.wait(1)
        
#         # Fade everything out
#         self.play(
#             FadeOut(title),
#             FadeOut(final_group)
#         )
#         self.wait(0.5)

# def run_neural_network_demo():
#     """Run the neural networks explanation demo"""
#     neural_text = """
#     Understanding Neural Networks
    
#     Neural networks are the foundation of modern artificial intelligence and machine learning.
#     They are inspired by the structure of the human brain and how neurons process information.
    
#     Architecture Overview:
#     A neural network consists of layers: input layer, hidden layers, and output layer.
#     Each layer contains multiple neurons that process and transmit information.
#     Neurons are connected through weighted synapses that determine signal strength.
    
#     How It Works:
#     Step 1: Data enters the input layer
#     Step 2: Hidden layers perform complex calculations using weighted connections
#     Step 3: Output layer produces the final prediction or classification
#     Step 4: The network learns by adjusting weights through backpropagation
    
#     Key Advantages:
#     Neural networks can learn complex patterns in data that traditional algorithms cannot detect.
#     They excel at tasks like image recognition, natural language processing, and game playing.
#     Deep neural networks with multiple hidden layers can solve extremely complex problems.
    
#     Applications:
#     Image recognition systems used in self-driving cars and medical diagnosis
#     Natural language processing for chatbots and translation services
#     Recommendation systems that power online shopping and streaming platforms
#     Financial modeling for stock prediction and fraud detection
#     """
#     return create_video_with_ai(neural_text, "neural_networks_demo", use_ai=True, tts_provider="openai")

def create_animation_video(prompt: str, output_name: str = "output"):
    """
    Simple function: Give prompt -> Get animated video with audio
    Just like ChatGPT does it - one shot!
    """
    try:
        print(f"🎬 Starting video generation...")
        print(f"📝 Prompt: {prompt}\n")
        
        # Step 1: Ask AI to generate Manim code (like ChatGPT did)
        print("🤖 Asking AI to generate Manim animation code...")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ Please set OPENAI_API_KEY environment variable")
            return None
            
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert Manim animator. Generate complete, working Manim code.
                    
Rules:
- Use ManimCE (latest version)
- Create a Scene class
- Include beautiful animations with colors
- Add text explanations
- Use proper timing (self.wait())
- Return ONLY the Python code, nothing else
- Make it visually appealing and educational"""
                },
                {
                    "role": "user",
                    "content": f"Create a Manim animation for: {prompt}"
                }
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        # Extract the code
        manim_code = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if "```python" in manim_code:
            manim_code = manim_code.split("```python")[1].split("```")[0].strip()
        elif "```" in manim_code:
            manim_code = manim_code.split("```")[1].split("```")[0].strip()
        
        print("✅ AI generated the Manim code!\n")
        
        # Step 2: Create narration for the animation
        print("🎙️ Generating narration...")
        
        narration_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an educational narrator. Create clear, engaging narration for technical animations. Keep it concise and well-paced."
                },
                {
                    "role": "user",
                    "content": f"Create a narration script (2-3 minutes) explaining: {prompt}. Make it match the visual flow of the animation."
                }
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        narration_text = narration_response.choices[0].message.content.strip()
        print("✅ Narration created!\n")
        
        # Step 3: Generate audio from narration
        print("🔊 Generating audio...")
        audio_file = f"{output_name}_audio.aiff"
        
        tts_response = client.audio.speech.create(
            model="tts-1-hd",
            voice="shimmer",
            input="... " + narration_text,  # Small pause at start
            speed=1.05
        )
        
        tts_response.stream_to_file(audio_file)
        print(f"✅ Audio saved: {audio_file}\n")
        
        # Step 4: Save and render Manim code
        print("📝 Saving Manim scene...")
        scene_file = f".temp_{output_name}.py"
        
        with open(scene_file, 'w') as f:
            f.write(manim_code)
        
        print(f"✅ Scene saved: {scene_file}\n")
        
        # Extract scene class name from code
        import re
        scene_match = re.search(r'class\s+(\w+)\s*\(Scene\)', manim_code)
        if not scene_match:
            print("❌ Could not find Scene class in generated code")
            return None
        
        scene_class_name = scene_match.group(1)
        print(f"🎬 Rendering scene: {scene_class_name}...")
        
        # Step 5: Render the animation
        cmd = [
            "venv/bin/python",
            "-m", "manim",
            "-pql",
            scene_file,
            scene_class_name
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Animation rendered!\n")
        
        # Step 6: Find the rendered video
        video_path = f"media/videos/{scene_file.replace('.py', '')}/480p15/{scene_class_name}.mp4"
        
        if not os.path.exists(video_path):
            print(f"❌ Video not found at: {video_path}")
            return None
        
        # Step 7: Combine video with audio
        print("🎵 Combining video with audio...")
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
        print(f"✅ SUCCESS! Video created: {final_output}")
        print(f"{'='*60}\n")
        
        return {
            "final_video": final_output,
            "scene_file": scene_file,
            "audio_file": audio_file
        }
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Process failed: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"Error output: {e.stderr}")
        return None
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    

# ============================================================
# MAIN - Simple Usage
# ============================================================
if __name__ == "__main__":
    # Just give a simple prompt - AI does everything!
    
    result = create_animation_video(
        prompt="Bubble Sort Algorithm - show array sorting with visual comparisons and swaps",
        output_name="bubble_sort_demo"
    )
    
    if not result:
        sys.exit(1)
        
    print(f"\n🎉 Done! Watch your video: {result['final_video']}")
    
    # Want to create more videos? Just change the prompt!
    # Examples:
    # - "Quick Sort algorithm visualization"
    # - "Binary Search Tree operations"
    # - "Graph traversal BFS and DFS"
    # - "Neural network forward propagation"
    # - "Pythagorean theorem proof"
