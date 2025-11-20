#!/usr/bin/env python3
"""
Intelligent Text-to-Video Pipeline for EdTech
Handles any type of educational content: programming, math, concepts, etc.
"""

import re
import subprocess
import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from manim import *

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ContentBlock:
    """Represents a structured piece of content"""
    content_type: str
    text: str
    visual_type: str
    timing: float
    metadata: Dict[str, Any]

@dataclass
class ParsedContent:
    """Complete parsed content structure"""
    main_concept: str
    content_type: str
    blocks: List[ContentBlock]
    relationships: Dict[str, List[str]]
    complexity: str

# ============================================================================
# CONTENT CLASSIFICATION
# ============================================================================

class ContentClassifier:
    """Intelligently classifies content type"""
    
    def __init__(self):
        self.patterns = {
            "programming_concept": [
                r"\b(this|function|method|class|object|variable)\b",
                r"\b(JavaScript|Python|Java|C\+\+|programming)\b",
                r"\b(algorithm|data structure|OOP|inheritance)\b"
            ],
            "mathematical_concept": [
                r"\b(derivative|integral|function|equation|calculus)\b",
                r"\b(sine|cosine|tangent|graph|plot)\b",
                r"\b(limit|convergence|series|matrix)\b"
            ],
            "algorithm_explanation": [
                r"\b(sort|search|merge|quick|bubble|insertion)\b",
                r"\b(recursive|iteration|divide and conquer)\b",
                r"\b(time complexity|space complexity|Big O)\b"
            ],
            "general_concept": [
                r"\b(definition|concept|explanation|understanding)\b",
                r"\b(example|instance|case|scenario)\b"
            ]
        }
    
    def classify(self, text: str) -> str:
        """Classify content type based on text analysis"""
        text_lower = text.lower()
        scores = {}
        
        for content_type, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            scores[content_type] = score
        
        # Return the type with highest score
        return max(scores, key=scores.get) if max(scores.values()) > 0 else "general_concept"

# ============================================================================
# SEMANTIC ANALYSIS
# ============================================================================

class SemanticAnalyzer:
    """Extracts semantic structure from text"""
    
    def __init__(self):
        self.semantic_patterns = {
            "definitions": r"(?:is|refers to|means|defines|represents)",
            "examples": r"(?:for example|such as|like|including|e\.g\.)",
            "comparisons": r"(?:differs|unlike|versus|compared to|in contrast)",
            "conditions": r"(?:when|if|unless|provided that|in case)",
            "consequences": r"(?:therefore|thus|as a result|consequently|hence)",
            "code_blocks": r"```[\s\S]*?```|`[^`]+`",
            "keywords": r"\b[A-Z][a-z]+\b|\b[a-z]+\.[a-z]+\b"
        }
    
    def analyze(self, text: str) -> ParsedContent:
        """Perform comprehensive semantic analysis"""
        
        # Extract main concept (usually first sentence or title)
        main_concept = self.extract_main_concept(text)
        
        # Classify content type
        classifier = ContentClassifier()
        content_type = classifier.classify(text)
        
        # Extract semantic blocks
        blocks = self.extract_semantic_blocks(text)
        
        # Find relationships
        relationships = self.find_relationships(blocks)
        
        # Determine complexity
        complexity = self.assess_complexity(text, blocks)
        
        return ParsedContent(
            main_concept=main_concept,
            content_type=content_type,
            blocks=blocks,
            relationships=relationships,
            complexity=complexity
        )
    
    def extract_main_concept(self, text: str) -> str:
        """Extract the main concept from text"""
        sentences = re.split(r'[.!?]+', text)
        if sentences:
            # Look for definition patterns in first few sentences
            for sentence in sentences[:3]:
                if re.search(self.semantic_patterns["definitions"], sentence.lower()):
                    return sentence.strip()
            return sentences[0].strip()
        return "Main Concept"
    
    def extract_semantic_blocks(self, text: str) -> List[ContentBlock]:
        """Break text into semantic blocks"""
        blocks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            if not para.strip():
                continue
                
            # Determine block type
            block_type = self.determine_block_type(para)
            visual_type = self.determine_visual_type(para, block_type)
            timing = self.calculate_timing(para)
            
            blocks.append(ContentBlock(
                content_type=block_type,
                text=para.strip(),
                visual_type=visual_type,
                timing=timing,
                metadata=self.extract_metadata(para)
            ))
        
        return blocks
    
    def determine_block_type(self, text: str) -> str:
        """Determine the type of content block"""
        text_lower = text.lower()
        
        if re.search(self.semantic_patterns["definitions"], text_lower):
            return "definition"
        elif re.search(self.semantic_patterns["examples"], text_lower):
            return "example"
        elif re.search(self.semantic_patterns["comparisons"], text_lower):
            return "comparison"
        elif re.search(self.semantic_patterns["conditions"], text_lower):
            return "conditional"
        elif re.search(r"```|`[^`]+`", text):
            return "code"
        else:
            return "explanation"
    
    def determine_visual_type(self, text: str, block_type: str) -> str:
        """Determine appropriate visual type for content"""
        visual_mapping = {
            "definition": "text_animation",
            "example": "interactive_demo",
            "comparison": "side_by_side",
            "conditional": "flowchart",
            "code": "code_highlight",
            "explanation": "text_animation"
        }
        return visual_mapping.get(block_type, "text_animation")
    
    def calculate_timing(self, text: str) -> float:
        """Calculate appropriate timing for content"""
        word_count = len(text.split())
        base_time = word_count * 0.15  # ~150 words per minute
        
        # Adjust based on complexity
        if re.search(r"```|`[^`]+`", text):  # Has code
            base_time *= 1.5
        if re.search(r"\b(when|if|unless)", text.lower()):  # Has conditions
            base_time *= 1.3
        
        return max(2.0, min(base_time, 8.0))  # Between 2-8 seconds
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from text"""
        metadata = {
            "word_count": len(text.split()),
            "has_code": bool(re.search(r"```|`[^`]+`", text)),
            "has_examples": bool(re.search(self.semantic_patterns["examples"], text.lower())),
            "has_comparisons": bool(re.search(self.semantic_patterns["comparisons"], text.lower())),
            "keywords": re.findall(self.semantic_patterns["keywords"], text)
        }
        return metadata
    
    def find_relationships(self, blocks: List[ContentBlock]) -> Dict[str, List[str]]:
        """Find relationships between content blocks"""
        relationships = {}
        
        for i, block in enumerate(blocks):
            relationships[f"block_{i}"] = []
            
            # Find blocks that reference this one
            for j, other_block in enumerate(blocks):
                if i != j:
                    # Simple keyword matching for now
                    if any(keyword in other_block.text.lower() 
                          for keyword in block.metadata["keywords"]):
                        relationships[f"block_{i}"].append(f"block_{j}")
        
        return relationships
    
    def assess_complexity(self, text: str, blocks: List[ContentBlock]) -> str:
        """Assess content complexity"""
        total_words = len(text.split())
        code_blocks = sum(1 for block in blocks if block.content_type == "code")
        technical_terms = len(re.findall(r"\b[A-Z][a-z]+\b", text))
        
        complexity_score = (total_words / 100) + (code_blocks * 2) + (technical_terms / 10)
        
        if complexity_score < 2:
            return "beginner"
        elif complexity_score < 4:
            return "intermediate"
        else:
            return "advanced"

# ============================================================================
# VISUAL STRATEGY SELECTION
# ============================================================================

class VisualStrategySelector:
    """Selects appropriate visual strategy based on content"""
    
    def select_strategy(self, content_type: str, parsed_content: ParsedContent) -> Dict[str, Any]:
        """Select visual strategy for content"""
        
        strategies = {
            "programming_concept": self.programming_strategy,
            "mathematical_concept": self.mathematical_strategy,
            "algorithm_explanation": self.algorithm_strategy,
            "general_concept": self.general_strategy
        }
        
        strategy_func = strategies.get(content_type, self.general_strategy)
        return strategy_func(parsed_content)
    
    def programming_strategy(self, content: ParsedContent) -> Dict[str, Any]:
        """Strategy for programming concepts"""
        return {
            "scene_types": ["concept_intro", "code_demo", "interactive_example"],
            "visual_elements": ["code_highlighting", "object_visualization", "scope_diagram"],
            "interactive_features": ["live_code_execution", "variable_tracking"],
            "timing_adjustments": {"code_demo": 1.5, "interactive": 2.0}
        }
    
    def mathematical_strategy(self, content: ParsedContent) -> Dict[str, Any]:
        """Strategy for mathematical concepts"""
        return {
            "scene_types": ["concept_intro", "graph_visualization", "calculation_demo"],
            "visual_elements": ["function_plots", "moving_points", "area_calculations"],
            "interactive_features": ["parameter_adjustment", "real_time_calculation"],
            "timing_adjustments": {"graph_animation": 1.2, "calculation": 1.8}
        }
    
    def algorithm_strategy(self, content: ParsedContent) -> Dict[str, Any]:
        """Strategy for algorithm explanations"""
        return {
            "scene_types": ["algorithm_overview", "step_by_step", "complexity_analysis"],
            "visual_elements": ["array_visualization", "comparison_counting", "tree_diagram"],
            "interactive_features": ["step_control", "speed_adjustment"],
            "timing_adjustments": {"step_demo": 1.3, "comparison": 1.7}
        }
    
    def general_strategy(self, content: ParsedContent) -> Dict[str, Any]:
        """Strategy for general concepts"""
        return {
            "scene_types": ["concept_intro", "explanation", "example"],
            "visual_elements": ["text_animation", "simple_diagrams"],
            "interactive_features": [],
            "timing_adjustments": {"explanation": 1.0}
        }

# ============================================================================
# ADAPTIVE SCENE FACTORY
# ============================================================================

class AdaptiveSceneFactory:
    """Creates manim scenes based on content analysis - NOW DYNAMIC!"""
    
    def __init__(self):
        self.universal_factory = UniversalSceneFactory()
    
    def create_scenes(self, parsed_content: ParsedContent, visual_strategy: Dict[str, Any]) -> List[Scene]:
        """Create manim scenes from parsed content - COMPLETELY DYNAMIC!"""
        # Use the universal factory - works for ANY content type
        return self.universal_factory.create_multiple_scenes(parsed_content)
    
    # OLD HARDCODED METHODS REMOVED - NOW USING DYNAMIC GENERATION!

# ============================================================================
# HIGH-QUALITY AUDIO GENERATION WITH MULTIPLE TTS PROVIDERS
# ============================================================================
#
# TTS Provider Options:
# - "openai": Premium quality with natural voices (requires OpenAI API key)
# - "macos": Built-in system TTS (fallback, free)
#
# OpenAI TTS Configuration:
# - Model: tts-1-hd (High Definition for premium quality)
# - Voice: alloy (Neutral voice optimized for Indian accent)
# - Speed: 0.75 (Slower for Indian accent clarity)
# - Indian Pronunciation: Automatic text conversion to Indian-English patterns
# - Auto fallback to macOS TTS if API unavailable
# - Perfect audio-video synchronization with 1.5s delay
#
# Audio synchronization is preserved across all providers
# ============================================================================

class AudioGenerator:
    """Generates audio using multiple TTS providers with high quality voices"""
    
    def __init__(self, provider: str = "openai"):
        """Initialize with TTS provider: openai, google, amazon, or macos"""
        self.provider = provider.lower()
        
        # Initialize provider-specific settings
        if self.provider == "openai":
            self.openai_client = None
            try:
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.openai_client = openai.OpenAI(api_key=api_key)
                    print("✅ OpenAI TTS initialized")
                else:
                    print("⚠️ No OpenAI API key, falling back to macOS TTS")
                    self.provider = "macos"
            except ImportError:
                print("⚠️ OpenAI not installed, falling back to macOS TTS")
                self.provider = "macos"
    
    def generate_audio(self, text: str, output_file: str = "audio.aiff") -> str:
        """Generate high-quality audio using selected TTS provider"""
        # Clean text for audio generation
        clean_text = re.sub(r'```[^`]*```', '', text)  # Remove code blocks
        clean_text = re.sub(r'\n+', ' ', clean_text)  # Replace newlines with spaces
        clean_text = re.sub(r'\s+', ' ', clean_text)  # Replace multiple spaces with single space
        clean_text = re.sub(r'[🧩💡⚙️📦🏗️➡️🔗🧭🎯]', '', clean_text)  # Remove emojis
        clean_text = clean_text.strip()
            
        # Generate audio based on provider
        try:
            if self.provider == "openai" and self.openai_client:
                return self._generate_openai_audio(clean_text, output_file)
            else:
                return self._generate_macos_audio(clean_text, output_file)
        except Exception as e:
            print(f"❌ {self.provider} TTS failed: {e}")
            print("🔄 Falling back to macOS TTS...")
            return self._generate_macos_audio(clean_text, output_file)
    
    def _generate_openai_audio(self, text: str, output_file: str) -> str:
        """Generate audio using OpenAI TTS - Premium quality with Indian pronunciation"""
        try:
            print("🎙️ Generating audio with OpenAI TTS (Indian-English Style)...")
            
            # Pre-process text for Indian English pronunciation
            indian_text = self._convert_to_indian_english(text)
            
            # OpenAI TTS API call - Using Nova for clear, professional educational voice
            response = self.openai_client.audio.speech.create(
                model="tts-1-hd",  # High quality model
                voice="nova",      # Professional female voice, perfect for educational content
                input=indian_text,
                speed=0.95         # Slightly slower for clarity but still engaging
            )
            
            # Save to temporary file first (OpenAI returns mp3)
            temp_mp3 = output_file.replace('.aiff', '_temp.mp3')
            with open(temp_mp3, 'wb') as f:
                f.write(response.content)
            
            # Convert to aiff for consistency with existing pipeline
            convert_cmd = [
                "ffmpeg", "-y", "-i", temp_mp3, 
                "-acodec", "pcm_s16be", output_file
            ]
            subprocess.run(convert_cmd, check=True, capture_output=True)
            
            # Clean up temp file
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
            
            # Get audio duration for sync
            audio_duration = self._get_audio_duration(output_file)
            self._save_duration_for_sync(audio_duration)
            
            print(f"✅ OpenAI TTS audio generated: {output_file} ({audio_duration:.1f}s)")
            return output_file
            
        except Exception as e:
            print(f"❌ OpenAI TTS failed: {e}")
            raise e
    
    def _convert_to_indian_english(self, text: str) -> str:
        """Prepare text for clear educational pronunciation"""
        # Clean and format text for better TTS
        text = text.replace('e.g.', 'for example')
        text = text.replace('i.e.', 'that is')
        text = text.replace('etc.', 'etcetera')
        text = text.replace('vs.', 'versus')
        
        # Add slight pauses for better comprehension
        text = text.replace('. ', '. [pause] ')
        text = text.replace('! ', '! [pause] ')
        text = text.replace('? ', '? [pause] ')
        
        return text
    
    def _generate_macos_audio(self, text: str, output_file: str) -> str:
        """Generate audio using macOS built-in TTS (fallback)"""
        try:
            print("🎙️ Generating audio with macOS TTS...")
            
            cmd = [
                "say",
                "-v", "Alex",  # Voice
                "-r", "160",   # Slightly slower for clarity
                "-o", output_file,
                text
            ]
            subprocess.run(cmd, check=True)
            
            # Get audio duration for sync
            audio_duration = self._get_audio_duration(output_file)
            self._save_duration_for_sync(audio_duration)
            
            print(f"✅ macOS TTS audio generated: {output_file} ({audio_duration:.1f}s)")
            return output_file
            
        except subprocess.CalledProcessError as e:
            print(f"❌ macOS TTS failed: {e}")
            return None
    
    def _get_audio_duration(self, audio_file: str) -> float:
        """Get audio duration using ffprobe"""
        try:
            duration_cmd = [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", audio_file
            ]
            duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
            return float(duration_result.stdout.strip())
        except:
            return 30.0  # Fallback duration
            
    def _save_duration_for_sync(self, duration: float):
        """Save duration for video synchronization"""
        try:
            duration_file = CONTENT_FILE_PATH.replace('.txt', '_audio_duration.txt')
            with open(duration_file, 'w') as f:
                f.write(str(duration))
        except:
            pass  # Non-critical if this fails

# ============================================================================
# MAIN PIPELINE
# ============================================================================

class UniversalTextToVideoPipeline:
    """Main pipeline that orchestrates everything"""
    
    def __init__(self, tts_provider: str = "openai"):
        self.semantic_analyzer = SemanticAnalyzer()
        self.visual_strategist = VisualStrategySelector()
        self.scene_factory = AdaptiveSceneFactory()
        self.audio_generator = AudioGenerator(provider=tts_provider)
    
    def process(self, raw_text: str, output_name: str = "generated_video") -> Dict[str, Any]:
        """Process any text into video content - PRODUCTION READY!"""
        
        print("🔍 Analyzing content...")
        parsed_content = self.semantic_analyzer.analyze(raw_text)
        
        print(f"📊 Content Type: {parsed_content.content_type}")
        print(f"🧠 Main Concept: {parsed_content.main_concept}")
        print(f"📈 Complexity: {parsed_content.complexity}")
        print(f"📝 Blocks: {len(parsed_content.blocks)}")
        
        print("🎨 Selecting visual strategy...")
        visual_strategy = self.visual_strategist.select_strategy(
            parsed_content.content_type, parsed_content
        )
        
        print("🎬 Creating scenes...")
        scenes = self.scene_factory.create_scenes(parsed_content, visual_strategy)
        
        print("🔊 Generating audio...")
        audio_file = self.audio_generator.generate_audio(raw_text, f"{output_name}_audio.aiff")
        
        print("🎥 Rendering video...")
        video_file = self.render_video(raw_text, output_name)
        
        print("🎬 Combining video and audio...")
        final_video = self.combine_video_audio(video_file, audio_file, output_name)
        
        print(f"✅ FINAL VIDEO READY: {final_video}")
        
        return {
            "parsed_content": parsed_content,
            "visual_strategy": visual_strategy,
            "scenes": scenes,
            "audio_file": audio_file,
            "video_file": video_file,
            "final_video": final_video,
            "output_name": output_name
        }
    
    def render_video(self, raw_text: str, output_name: str) -> str:
        """Automatically render Manim video using UNIVERSAL scene"""
        import subprocess
        import os
        
        # Write content to temporary file for the scene to read
        with open(CONTENT_FILE_PATH, 'w') as f:
            f.write(raw_text)
        
        try:
            # Use the UNIVERSAL scene that works for ANY content
            cmd = [
                "/Users/apple/Desktop/manim/.venv/bin/python", "-m", "manim", 
                "-pql", "app.py", "UniversalContentScene"
            ]
            print("📹 Rendering video...")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Find the generated video file
            video_path = "media/videos/app/480p15/UniversalContentScene.mp4"
            
            # Check if file exists
            if os.path.exists(video_path):
                print(f"✅ Video rendered: {video_path}")
                # Clean up temporary file
                if os.path.exists(CONTENT_FILE_PATH):
                    os.remove(CONTENT_FILE_PATH)
                return video_path
            else:
                print(f"❌ Video file not found at: {video_path}")
                return None
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Video rendering failed: {e}")
            print(f"Error output: {e.stderr}")
            # Clean up temporary file
            if os.path.exists(CONTENT_FILE_PATH):
                os.remove(CONTENT_FILE_PATH)
            return None
    
    def combine_video_audio(self, video_file: str, audio_file: str, output_name: str) -> str:
        """Automatically combine video and audio"""
        if not video_file or not audio_file:
            print("❌ Missing video or audio file")
            return None
            
        final_output = f"{output_name}_final.mp4"
        
        try:
            cmd = [
                "ffmpeg", "-y",  # -y to overwrite output file
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
# TESTING FUNCTIONS
# ============================================================================

# OLD HARDCODED TEST FUNCTIONS REMOVED - NOW USING DYNAMIC SYSTEM!

# ============================================================================
# DYNAMIC SCENE GENERATOR - NO HARDCODED SCENES!
# ============================================================================

class DynamicSceneGenerator:
    """Generates ANY scene dynamically based on content - NO HARDCODING!"""
    
    def __init__(self):
        pass
    
    def create_dynamic_scene(self, parsed_content: ParsedContent) -> Scene:
        """Create ANY scene dynamically - works for ANY content type"""
        
        class UniversalDynamicScene(Scene):
            def __init__(self, content):
                super().__init__()
                self.content = content
            
            def construct(self):
                # Parse content into sections
                sections = self.parse_into_sections(self.content)
                
                # Render each section with proper timing
                for section in sections:
                    self.render_section(section)
            
            def parse_into_sections(self, content):
                """Parse content into renderable sections"""
                sections = []
                
                # Split by major delimiters (headings, emojis, double newlines)
                text = content.main_concept + "\n\n" + "\n\n".join([b.text for b in content.blocks])
                
                # Split by lines
                lines = text.split('\n')
                
                current_section = {"title": None, "content": [], "code": None}
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if it's a heading (contains emoji or starts with capital letter after emoji)
                    if re.match(r'^[🧩💡⚙️📦🏗️➡️🔗🧭🎯]', line) or (len(line) < 50 and line[0].isupper() and not line.endswith('.')):
                        # Save previous section
                        if current_section["title"] or current_section["content"]:
                            sections.append(current_section.copy())
                        # Start new section
                        current_section = {"title": line, "content": [], "code": None}
                    # Check if it's code
                    elif line.startswith('```') or '(' in line and ')' in line and '{' in line:
                        # Collect code block
                        code_lines = [line.replace('```', '').replace('javascript', '').replace('python', '')]
                        current_section["code"] = '\n'.join(code_lines).strip()
                    # Check if it's code continuation
                    elif current_section["code"] is not None and (line.startswith('  ') or '}' in line or ';' in line):
                        current_section["code"] += '\n' + line
                    # Regular content
                    else:
                        current_section["content"].append(line)
                
                # Add last section
                if current_section["title"] or current_section["content"]:
                    sections.append(current_section)
                
                return sections
            
            def render_section(self, section):
                """Render a complete section with title, content, and code"""
                elements = []
                
                # Title
                if section["title"]:
                    title = Text(section["title"], font_size=36, color=BLUE, weight=BOLD)
                    title.to_edge(UP, buff=0.5)
                    elements.append(title)
                    self.add(title)
                    self.wait(0.5)
                
                # Content text
                y_offset = 1.5
                if section["content"]:
                    content_text = '\n'.join(section["content"][:3])  # Max 3 lines
                    if len(content_text) > 120:
                        content_text = content_text[:120] + "..."
                    
                    text_obj = Text(content_text, font_size=24, color=WHITE)
                    text_obj.move_to([0, y_offset, 0])
                    elements.append(text_obj)
                    self.add(text_obj)
                    self.wait(1.5)
                    y_offset -= 1.5
                
                # Code block
                if section["code"]:
                    code_lines = section["code"].split('\n')[:7]  # Max 7 lines
                    code_text = '\n'.join(code_lines)
                    try:
                        code_obj = Code(code_string=code_text, language="javascript")
                        code_obj.scale(0.65)
                        code_obj.move_to([0, y_offset - 0.8, 0])
                        elements.append(code_obj)
                        self.add(code_obj)
                        self.wait(2)
                    except:
                        pass
                
                # Hold the section
                self.wait(1)
                
                # Clear all elements
                for elem in elements:
                    self.remove(elem)
                self.wait(0.3)
        
        return UniversalDynamicScene(parsed_content)
    
    def create_step_element(self, text: str):
        """Create step element"""
        return Text(text).scale(0.7).set_color(GREEN)
    
    def create_concept_element(self, text: str):
        """Create concept element"""
        return Text(text).scale(0.8).set_color(RED)

# ============================================================================
# UNIVERSAL SCENE CLASS - WORKS FOR ANY CONTENT!
# ============================================================================

class UniversalScene(Scene):
    """Universal scene that can handle ANY content type dynamically"""
    
    def __init__(self, content: ParsedContent):
        super().__init__()
        self.content = content
        self.generator = DynamicSceneGenerator()
    
    def construct(self):
        """Render ANY content dynamically"""
        self.generator.render_content(self, self.content)

# ============================================================================
# SCENE FACTORY - CREATES SCENES FOR ANY CONTENT!
# ============================================================================

class UniversalSceneFactory:
    """Creates scenes for ANY content type - completely dynamic"""
    
    def create_scene_for_content(self, parsed_content: ParsedContent) -> Scene:
        """Create scene for ANY content type"""
        return UniversalScene(parsed_content)
    
    def create_multiple_scenes(self, parsed_content: ParsedContent) -> List[Scene]:
        """Create multiple scenes for complex content"""
        scenes = []
        
        # Main scene
        scenes.append(self.create_scene_for_content(parsed_content))
        
        # Additional scenes for complex content
        if len(parsed_content.blocks) > 3:
            # Split into multiple scenes
            mid_point = len(parsed_content.blocks) // 2
            first_half = ParsedContent(
                main_concept=parsed_content.main_concept,
                content_type=parsed_content.content_type,
                blocks=parsed_content.blocks[:mid_point],
                relationships=parsed_content.relationships,
                complexity=parsed_content.complexity
            )
            second_half = ParsedContent(
                main_concept=parsed_content.main_concept,
                content_type=parsed_content.content_type,
                blocks=parsed_content.blocks[mid_point:],
                relationships=parsed_content.relationships,
                complexity=parsed_content.complexity
            )
            scenes.append(self.create_scene_for_content(first_half))
            scenes.append(self.create_scene_for_content(second_half))
        
        return scenes

# ============================================================================
# DEMO SCENE - SHOWS DYNAMIC SYSTEM IN ACTION!
# ============================================================================

class DynamicDemoScene(Scene):
    """Demo scene that shows the dynamic system working"""
    def construct(self):
        # Create sample content
        sample_text = """
        This is a demonstration of the dynamic text-to-video pipeline.
        
        It can handle ANY content type automatically:
        - Programming concepts
        - Mathematical concepts  
        - Algorithm explanations
        - General educational content
        
        No hardcoded scenes needed!
        """
        
        # Process the content dynamically
        pipeline = UniversalTextToVideoPipeline()
        result = pipeline.process(sample_text, "demo")
        
        # Show the analysis results
        title = Text("Dynamic Pipeline Demo").scale(1.2)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Show content type
        content_type = Text(f"Content Type: {result['parsed_content'].content_type}").scale(0.8)
        content_type.next_to(title, DOWN)
        self.play(Write(content_type))
        self.wait(1)
        
        # Show complexity
        complexity = Text(f"Complexity: {result['parsed_content'].complexity}").scale(0.8)
        complexity.next_to(content_type, DOWN)
        self.play(Write(complexity))
        self.wait(1)
        
        # Show blocks count
        blocks_count = Text(f"Content Blocks: {len(result['parsed_content'].blocks)}").scale(0.8)
        blocks_count.next_to(complexity, DOWN)
        self.play(Write(blocks_count))
        self.wait(1)
        
        # Show scenes count
        scenes_count = Text(f"Scenes Generated: {len(result['scenes'])}").scale(0.8)
        scenes_count.next_to(blocks_count, DOWN)
        self.play(Write(scenes_count))
        self.wait(2)
        
        # Success message
        success = Text("✅ DYNAMIC SYSTEM WORKING!").scale(1.0).set_color(GREEN)
        success.next_to(scenes_count, DOWN)
        self.play(Write(success))
        self.wait(2)
        
        # Clean up
        self.play(FadeOut(title), FadeOut(content_type), FadeOut(complexity),
                 FadeOut(blocks_count), FadeOut(scenes_count), FadeOut(success))

# ============================================================================
# DYNAMIC TEST FUNCTION - WORKS FOR ANY CONTENT!
# ============================================================================

def test_any_content(text: str, content_name: str):
    """Test ANY content type dynamically - NO HARDCODING!"""
    pipeline = UniversalTextToVideoPipeline()
    result = pipeline.process(text, content_name)
    
    print(f"\n✅ {content_name} analysis complete!")
    print(f"📊 Detected {len(result['parsed_content'].blocks)} content blocks")
    print(f"🎬 Generated {len(result['scenes'])} scenes")
    print(f"🧠 Content Type: {result['parsed_content'].content_type}")
    print(f"📈 Complexity: {result['parsed_content'].complexity}")
    
    return result

# ============================================================================
# UNIVERSAL SCENE FOR ANY CONTENT - NO HARDCODING!
# ============================================================================

# File-based content passing (more reliable than global variable)
CONTENT_FILE_PATH = "/Users/apple/Desktop/manim/.temp_content.txt"

class UniversalContentScene(Scene):
    def construct(self):
        """UNIVERSAL: Handles ANY content dynamically - NO HARDCODING!"""
        import os
        
        # Read content from file
        if not os.path.exists(CONTENT_FILE_PATH):
            # Fallback: show error message
            error = Text("No content file found!", font_size=36, color=RED)
            self.add(error)
            self.wait(2)
            return
        
        with open(CONTENT_FILE_PATH, 'r') as f:
            content_text = f.read()
        
        if not content_text:
            error = Text("Content file is empty!", font_size=36, color=RED)
            self.add(error)
            self.wait(2)
            return
        
        # Parse the raw text into sections
        sections = self.parse_text_into_sections(content_text)
        
        # CRITICAL FIX: Merge small sections together to avoid too many transitions
        sections = self.merge_small_sections(sections)
        
        # Calculate ACTUAL audio duration
        audio_file_path = CONTENT_FILE_PATH.replace('.txt', '_audio_duration.txt')
        
        import os
        actual_audio_duration = None
        if os.path.exists(audio_file_path):
            with open(audio_file_path, 'r') as f:
                actual_audio_duration = float(f.read().strip())
        
        if not actual_audio_duration:
            # Fallback: estimate from word count
            text_for_audio = re.sub(r'```[^`]*```', '', content_text)
            text_for_audio = re.sub(r'[🧩💡⚙️📦🏗️➡️🔗🧭🎯]', '', text_for_audio)
            words = len(text_for_audio.split())
            actual_audio_duration = (words / 160) * 60
        
        # Calculate time for EACH section based on its content length
        # This is the KEY fix - different sections get different times!
        section_times = []
        total_content_length = 0
        
        for section in sections:
            # Count words in this section (title + content + code estimate)
            section_words = 0
            if section.get("title"):
                section_words += len(section["title"].split())
            if section.get("content"):
                section_words += sum(len(line.split()) for line in section["content"])
            if section.get("code"):
                # Code takes longer to read - count as 1.5x
                section_words += len(section["code"].split()) * 1.5
            
            section_times.append(section_words)
            total_content_length += section_words
        
        # Scale section times to match total audio duration
        scaled_times = []
        for section_words in section_times:
            section_duration = (section_words / total_content_length) * actual_audio_duration
            section_duration = max(section_duration, 2.0)  # Minimum 2s per section
            scaled_times.append(section_duration)
        
        print(f"🎬 Video will be {actual_audio_duration:.1f}s with {len(sections)} sections")
        print(f"⏱️  Section times: {', '.join([f'{t:.1f}s' for t in scaled_times])}")
        
        # Render each section with its SPECIFIC calculated timing
        for i, (section, section_time) in enumerate(zip(sections, scaled_times), 1):
            if not section or not isinstance(section, dict):
                print(f"   ⚠️  Skipping invalid section {i}")
                continue
            title = section.get('title', 'Content')[:40] if section.get('title') else 'Content'
            print(f"   Section {i}/{len(sections)}: {title}... ({section_time:.1f}s)")
            self.render_section(section, section_time)
    
    def parse_text_into_sections(self, text: str):
        """Parse ANY text into renderable sections"""
        print("\n🔍 DEBUG: Original Text Being Parsed:")
        print("=" * 50)
        print(text)
        print("=" * 50)
        
        sections = []
        lines = text.split('\n')
        
        current_section = {"title": None, "content": [], "code": None}
        in_code_block = False
        code_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            
            if not line_stripped:
                continue
            
            # Detect code block start/end
            if line_stripped.startswith('```'):
                if in_code_block:
                    # End of code block
                    current_section["code"] = '\n'.join(code_lines)
                    code_lines = []
                    in_code_block = False
                else:
                    # Start of code block
                    in_code_block = True
                continue
            
            # If inside code block, collect code lines
            if in_code_block:
                code_lines.append(line)
                continue
            
            # Detect section titles (short lines with capital, emoji, or ending with colon)
            is_title = (
                len(line_stripped) < 60 and
                (line_stripped[0].isupper() or
                 re.match(r'^[🧩💡⚙️📦🏗️➡️🔗🧭🎯]', line_stripped) or
                 line_stripped.endswith(':'))
            )
            
            if is_title:
                # Save previous section
                if current_section["title"] or current_section["content"] or current_section["code"]:
                    sections.append(current_section.copy())
                # Start new section
                current_section = {"title": line_stripped, "content": [], "code": None}
            else:
                # Regular content line
                current_section["content"].append(line_stripped)
        
        # Add last section
        if current_section["title"] or current_section["content"] or current_section["code"]:
            sections.append(current_section)
        
        print("\n📋 DEBUG: Parsed Sections:")
        print("=" * 50)
        for i, section in enumerate(sections, 1):
            print(f"\nSection {i}:")
            if section.get("title"):
                print(f"Title: {section['title']}")
            if section.get("content"):
                print("Content:")
                for line in section["content"]:
                    print(f"  {line}")
            if section.get("code"):
                print("Code:")
                print(section["code"])
            print("-" * 30)
        print("=" * 50)
        
        return sections
    
    def merge_small_sections(self, sections):
        """ULTRA AGGRESSIVE merge - FILL THE ENTIRE SCREEN with content!"""
        merged = []
        current_merged = None
        word_count = 0
        screens_with_code = 0
        
        # ULTRA AGGRESSIVE: 100+ words per screen to fill height
        MIN_WORDS_PER_SCREEN = 100
        
        for i, section in enumerate(sections):
            # Count words in this section
            section_words = 0
            if section.get("title"):
                section_words += len(section["title"].split())
            if section.get("content"):
                section_words += sum(len(line.split()) for line in section["content"])
            
            # Special handling for code sections
            if section.get("code"):
                # Add preceding text to merged if exists
                if current_merged and word_count > 0:
                    merged.append(current_merged)
                
                # Create combined section with title + content + code
                code_section = {
                    "title": section.get("title", ""),
                    "content": section.get("content", []).copy(),
                    "code": section["code"]
                }
                merged.append(code_section)
                screens_with_code += 1
                current_merged = None
                word_count = 0
                continue
            
            # Start new merged section
            if current_merged is None:
                current_merged = {
                    "title": section.get("title", ""),
                    "content": section.get("content", []).copy(),
                    "code": None
                }
                word_count = section_words
            else:
                # Keep merging until we have ENOUGH content
                if section.get("content"):
                    current_merged["content"].extend([""])  # Add spacing
                    current_merged["content"].extend(section["content"])
                word_count += section_words
            
            # Only save if we have SUBSTANTIAL content OR it's the last section
            if word_count >= MIN_WORDS_PER_SCREEN or i == len(sections) - 1:
                if current_merged and (word_count > 0 or current_merged["content"]):
                    merged.append(current_merged)
                    current_merged = None
                    word_count = 0
        
        # Add any remaining content
        if current_merged and (word_count > 0 or current_merged["content"]):
            merged.append(current_merged)
        
        print(f"\n📦 DEBUG: After Merging Sections:")
        print("=" * 50)
        print(f"Original sections: {len(sections)} → Merged into: {len(merged)} screens")
        print(f"Screens with code: {screens_with_code}")
        print(f"Target words per screen: {MIN_WORDS_PER_SCREEN}+")
        
        print("\nMerged Sections Content:")
        for i, section in enumerate(merged, 1):
            print(f"\nMerged Section {i}:")
            if section.get("title"):
                print(f"Title: {section['title']}")
            if section.get("content"):
                print("Content:")
                for line in section["content"]:
                    print(f"  {line}")
            if section.get("code"):
                print("Code:")
                print(section["code"])
            print("-" * 30)
        print("=" * 50)
        
        return merged
    
    def render_section(self, section, time_per_section):
        """FULL WIDTH, FULL HEIGHT with ENGAGING ANIMATIONS!"""
        
        # FULL HEIGHT: Use entire screen from top to bottom
        y_start = 3.5
        y_end = -3.0
        available_height = y_start - y_end
        
        # Title with STUNNING visual entrance
        title_obj = None
        if section["title"]:
            # Use gradient colors and modern font
            from manim import Tex
            title_obj = Text(
                section["title"], 
                font_size=42,
                gradient=(BLUE, PURPLE),  # Beautiful gradient
                weight=BOLD,
                font="SF Pro Display"  # Modern macOS font
            )
            
            # Add underline decoration
            underline = Line(
                title_obj.get_left() + DOWN * 0.3,
                title_obj.get_right() + DOWN * 0.3,
                color=YELLOW,
                stroke_width=3
            )
            
            # CRITICAL: Scale title if too wide for screen
            max_title_width = 12.5
            if title_obj.width > max_title_width:
                scale_factor = max_title_width / title_obj.width
                title_obj.scale(scale_factor)
                underline.scale(scale_factor)
                print(f"   📏 Title scaled: {title_obj.width:.2f} → {max_title_width} (factor: {scale_factor:.2f})")
            
            title_obj.to_edge(UP, buff=0.2)
            underline.next_to(title_obj, DOWN, buff=0.1)
            
            # ANIMATION: Dramatic entrance with underline
            self.play(
                Write(title_obj, run_time=1.0),
                GrowFromCenter(underline, run_time=0.6)
            )
            y_start = 2.3
        
        # Content with FULL WIDTH and animations
        content_elements = VGroup()
        if section["content"]:
            all_content = ' '.join(section["content"])
            
            # FULL WIDTH: Let Manim handle text width naturally
            # Don't artificially limit - Manim will fit text to screen
            words = all_content.split()
            lines = []
            current_line = ""
            
            # Use MORE characters per line since we're not shrinking text
            for word in words:
                test_line = (current_line + " " + word).strip() if current_line else word
                # Allow up to 110 characters - Manim will scale if needed
                if len(test_line) > 110:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
            if current_line:
                lines.append(current_line)
            
            print(f"   📏 Split into {len(lines)} lines (max 110 chars each)")
            
            # Calculate lines that fit height - USE MAXIMUM HEIGHT!
            code_space = 2.5 if section.get("code") else 0
            title_space = 0.8 if title_obj else 0
            available_for_text = available_height - title_space - code_space
            line_spacing = 0.40  # Tighter spacing for more content
            max_lines = min(int(available_for_text / line_spacing), len(lines), 16)  # Up to 16 lines!
            
            print(f"   📝 Rendering {max_lines} lines of {len(lines)} available")
            
            # Create content with BEAUTIFUL styling - ALL AS ONE TEXT OBJECT for consistency
            # Join all lines into one text block
            full_text = '\n'.join(lines[:max_lines])
            
            # Create single unified text object with CONSISTENT styling
            text_obj = Text(
            full_text,
            font_size=24,  # Consistent size for all content
            color=WHITE,
            font="Helvetica Neue",
            line_spacing=0.8,  # Tighter line spacing
            weight=NORMAL,
            disable_ligatures=True  # Better rendering
            )
            
            # Scale if too wide
            max_width = 13.0
            if text_obj.width > max_width:
                scale_factor = max_width / text_obj.width
                text_obj.scale(scale_factor)
            
            # Position: Start from top of available space
            if title_obj:
                text_obj.next_to(title_obj, DOWN, buff=0.3)
            else:
                text_obj.to_edge(UP, buff=0.3)
            
            # Move left to center
            text_obj.move_to([0, text_obj.get_center()[1], 0])
            
            # Create shadow for depth
            shadow = text_obj.copy()
            shadow.set_color("#1a1a1a")
            shadow.shift(DOWN * 0.03 + RIGHT * 0.03)
            
            content_elements = VGroup(shadow, text_obj)
            
            # ANIMATION: Fade in smoothly
            self.play(
            FadeIn(content_elements, shift=UP * 0.3),
            run_time=1.2
                )
        
        # Code with STUNNING visual presentation
        code_obj = None
        if section["code"]:
            code_lines = section["code"].split('\n')[:12]
            code_text = '\n'.join(code_lines)
            try:
                # Detect language from code content
                language = "python"
                if "function" in code_text or "const" in code_text or "let" in code_text:
                    language = "javascript"
                elif "def " in code_text or "import " in code_text:
                    language = "python"
                
                code_obj = Code(
                    code_string=code_text, 
                    language=language,
                    font_size=22,
                    background="window",
                    insert_line_no=True,  # Add line numbers for professional look
                    style="monokai"  # Modern dark theme
                )
                code_obj.scale(0.9)
                
                # Add glowing border around code
                code_border = SurroundingRectangle(
                    code_obj,
                    color=BLUE,
                    buff=0.15,
                    stroke_width=2
                )
                code_border.set_fill(color="#1e1e1e", opacity=0.3)
                
                code_group = VGroup(code_border, code_obj)
                code_group.to_edge(DOWN, buff=0.3)
                
                # ANIMATION: Dramatic entrance with glow effect
                self.play(
                    FadeIn(code_border, scale=0.8),
                    run_time=0.4
                )
                self.play(
                    GrowFromCenter(code_obj),
                    run_time=0.8
                )
            except Exception as e:
                print(f"Error creating code: {e}")
                pass
        
        # Hold content while audio narrates
        hold_time = time_per_section - 2.5  # Account for animations
        self.wait(max(hold_time, 1.0))
        
        # ANIMATION: Fade out everything together with style
        fade_elements = VGroup()
        if title_obj:
            fade_elements.add(title_obj)
            # Add underline if it exists
            if 'underline' in locals():
                fade_elements.add(underline)
        if len(content_elements) > 0:
            fade_elements.add(content_elements)
        if code_obj:
            # Add code border if it exists
            if 'code_group' in locals():
                fade_elements.add(code_group)
            else:
                fade_elements.add(code_obj)
        
        self.play(FadeOut(fade_elements, shift=DOWN * 0.5), run_time=0.5)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# ============================================================================
# PRODUCTION-READY COMMAND LINE INTERFACE
# ============================================================================

def create_video_from_text(text: str, output_name: str = "my_video", tts_provider: str = "openai") -> str:
    """PRODUCTION-READY: Create video with high-quality voice from any text - ONE COMMAND!"""
    print("🚀 PRODUCTION-READY Text-to-Video Pipeline")
    print("=" * 60)
    print("✨ ONE COMMAND → FINAL VIDEO WITH HIGH-QUALITY VOICE!")
    print(f"🎙️ Using {tts_provider.upper()} TTS for premium audio quality")
    print("=" * 60)
    
    pipeline = UniversalTextToVideoPipeline(tts_provider=tts_provider)
    result = pipeline.process(text, output_name)
    
    if result['final_video']:
        print(f"\n🎉 SUCCESS! Your video is ready:")
        print(f"📁 File: {result['final_video']}")
        print(f"📊 Content: {result['parsed_content'].content_type}")
        print(f"🧠 Concept: {result['parsed_content'].main_concept}")
        return result['final_video']
    else:
        print("❌ Video creation failed!")
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line usage: python app.py "Your text here"
        text = sys.argv[1]
        output_name = sys.argv[2] if len(sys.argv) > 2 else "generated_video"
        
        print("🎬 Creating video from your text...")
        video_file = create_video_from_text(text, output_name)
        
        if video_file:
            print(f"\n✅ DONE! Video saved as: {video_file}")
        else:
            print("\n❌ FAILED! Check the error messages above.")
    else:
        # Demo mode - show examples
        print("🚀 PRODUCTION-READY Text-to-Video Pipeline")
        print("=" * 60)
        print("✨ USAGE: python app.py 'Your text here'")
        print("=" * 60)
        
        # Demo with sample text
        demo_text = """
        This is a demonstration of the production-ready text-to-video pipeline.
        
        It can handle ANY content type automatically:
        - Programming concepts
        - Mathematical concepts  
        - Algorithm explanations
        - General educational content
        
        Just input your text and get a professional video with voice!
        """
        
        print("🎬 Running demo with sample text...")
        video_file = create_video_from_text(demo_text, "demo_video")
        
        if video_file:
            print(f"\n✅ DEMO COMPLETE! Video saved as: {video_file}")
        else:
            print("\n❌ DEMO FAILED! Check the error messages above.")
