from manim import *

class CodeExplanationScene(Scene):
    def construct(self):
        self.wait(1.2)
        title = Text("Code Explanation", font_size=38, font="Helvetica", weight=BOLD, color=BLUE)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.5)
        self.wait(1)
        
        full_code = Text(
            """public class Test {
    public static void main(String[] args) {
        int x = 5;
        for (int i = 0; i < x; i++) {
            System.out.println(i);
        }
    }
}""",
            font_size=22,
            font="Courier",
            color=WHITE,
            line_spacing=1.0
        )
        
        if full_code.width > 13:
            full_code.scale_to_fit_width(13)
        
        full_code.to_edge(LEFT, buff=0.5)
        
        # Highlight rectangle for block: lines 4-6 (3 lines)
        # Block height calculation: 3 lines × 0.424 line_height = 1.273
        highlight_0 = Rectangle(
            width=12,
            height=1.273,
            fill_opacity=0.0,
            fill_color="#4A90E2",
            stroke_width=3,
            stroke_color="#4A90E2",
            stroke_opacity=0.0
        )
        # CRITICAL: Verify highlight height covers all 3 lines
        # If highlight appears too small, increase line_height or block_height
        # Add glow effect (slightly larger, semi-transparent)
        highlight_glow_0 = Rectangle(
            width=12.4,
            height=1.373,
            fill_opacity=0.0,
            fill_color="#4A90E2",
            stroke_width=5,
            stroke_color="#4A90E2",
            stroke_opacity=0.0
        )

        
        available_screen_height = 5.5
        code_height = full_code.height
        
        if code_height > available_screen_height:
            code_center_x = full_code.get_left()[0] + full_code.width/2
            start_center_y = 2.5 - (code_height / 2)
            full_code.move_to([code_center_x, start_center_y, 0])
            end_center_y = -3.5 + (code_height / 2)
            scroll_distance = end_center_y - start_center_y
            
            # Position highlight for lines 4-6 (0-indexed: 3-5)
            # Center of line 3 is at: full_code.get_top()[1] - ((3 + 0.5) * 0.424)
            # Center of block is at: center_of_start_line - ((3 - 1) * 0.424 / 2)
            block_0_start_line_center_y = full_code.get_top()[1] - ((3 + 0.5) * 0.424)
            block_0_center_y = block_0_start_line_center_y - ((3 - 1) * 0.424 / 2.0)
            highlight_0.move_to([full_code.get_center()[0], block_0_center_y, 0])
            highlight_0.stretch_to_fit_width(full_code.width + 0.3)
            # Highlight height: 1.273 (for 3 lines)
            highlight_0.set_z_index(-1)
            highlight_0.set_opacity(0)
            highlight_0.set_stroke_opacity(0)
            highlight_glow_0.move_to([full_code.get_center()[0], block_0_center_y, 0])
            highlight_glow_0.stretch_to_fit_width(full_code.width + 0.3)
            highlight_glow_0.set_z_index(-2)
            highlight_glow_0.set_opacity(0)
            highlight_glow_0.set_stroke_opacity(0)

            
            # Add highlights to scene
            all_highlights = VGroup(highlight_0, highlight_glow_0)
            self.add(all_highlights)
            self.play(FadeIn(full_code), run_time=0.5)
            
            self.wait(51.19)
            # Scroll only if needed (scroll_distance > 0)
            scroll_time = 0.3 if 68.64 > 0 and scroll_distance > 0 else 0.0
            if 68.64 > 0 and scroll_distance > 0:
                seg_target_y = start_center_y + (scroll_distance * 0.000)
                # CRITICAL: Update highlight positions relative to new code position after scroll
                code_top_y = seg_target_y + (full_code.height / 2)
                # Position highlight: center of start line, then adjust for block center
                block_0_start_line_center_y = code_top_y - ((3 + 0.5) * 0.424)
                block_0_center_y = block_0_start_line_center_y - ((3 - 1) * 0.424 / 2.0)
                highlight_0.move_to([code_center_x, block_0_center_y, 0])
                highlight_glow_0.move_to([code_center_x, block_0_center_y, 0])
                self.play(full_code.animate.move_to([code_center_x, seg_target_y, 0]), run_time=0.3)
            else:
                # No scrolling - highlights are already positioned correctly in highlight_positioning
                # Just ensure they're at the right position relative to current code position
                code_top_y = full_code.get_top()[1]
                # Position highlight: center of start line, then adjust for block center
                block_0_start_line_center_y = code_top_y - ((3 + 0.5) * 0.424)
                block_0_center_y = block_0_start_line_center_y - ((3 - 1) * 0.424 / 2.0)
                highlight_0.move_to([full_code.get_center()[0], block_0_center_y, 0])
                highlight_glow_0.move_to([full_code.get_center()[0], block_0_center_y, 0])
            
            # Enhanced highlight animation sequence
            # Step 1: Glow appears first (subtle background glow)
            self.play(
                highlight_glow_0.animate.set_stroke_opacity(0.3),
                run_time=0.2
            )
            # Step 2: Animated border drawing effect (stroke appears)
            self.play(
                highlight_0.animate.set_stroke_opacity(0.9),
                highlight_glow_0.animate.set_stroke_opacity(0.5),
                run_time=0.3
            )
            # Step 3: Fill fades in (background highlight)
            self.play(
                highlight_0.animate.set_opacity(0.25).set_stroke_opacity(0.95),
                highlight_glow_0.animate.set_stroke_opacity(0.4),
                run_time=0.2
            )
            # Step 4: Subtle pulse animation (breathing effect)
            self.play(
                highlight_0.animate.set_opacity(0.20).set_stroke_opacity(0.85),
                highlight_glow_0.animate.set_stroke_opacity(0.3),
                run_time=0.2
            )
            self.play(
                highlight_0.animate.set_opacity(0.28).set_stroke_opacity(0.95),
                highlight_glow_0.animate.set_stroke_opacity(0.45),
                run_time=0.2
            )
            remaining_time = 17.45 - scroll_time - 1.1
            if remaining_time > 0:
                self.wait(remaining_time)
            self.play(highlight_0.animate.set_opacity(0), run_time=0.2)

        else:
            # Code fits on screen - still need highlights!
            full_code.next_to(title, DOWN, buff=0.3)
            
            # Define variables needed for animation timeline (even without scrolling)
            code_center_x = full_code.get_center()[0]
            start_center_y = full_code.get_center()[1]
            scroll_distance = 0  # No scrolling needed
            
            # Position highlights even when code doesn't scroll
            
            # Position highlight for lines 4-6 (0-indexed: 3-5)
            # Center of line 3 is at: full_code.get_top()[1] - ((3 + 0.5) * 0.424)
            # Center of block is at: center_of_start_line - ((3 - 1) * 0.424 / 2)
            block_0_start_line_center_y = full_code.get_top()[1] - ((3 + 0.5) * 0.424)
            block_0_center_y = block_0_start_line_center_y - ((3 - 1) * 0.424 / 2.0)
            highlight_0.move_to([full_code.get_center()[0], block_0_center_y, 0])
            highlight_0.stretch_to_fit_width(full_code.width + 0.3)
            # Highlight height: 1.273 (for 3 lines)
            highlight_0.set_z_index(-1)
            highlight_0.set_opacity(0)
            highlight_0.set_stroke_opacity(0)
            highlight_glow_0.move_to([full_code.get_center()[0], block_0_center_y, 0])
            highlight_glow_0.stretch_to_fit_width(full_code.width + 0.3)
            highlight_glow_0.set_z_index(-2)
            highlight_glow_0.set_opacity(0)
            highlight_glow_0.set_stroke_opacity(0)

            
            # Add highlights to scene
            all_highlights = VGroup(highlight_0, highlight_glow_0)
            self.add(all_highlights)
            
            self.play(FadeIn(full_code), run_time=0.5)
            
            # Run animation timeline for highlights (even without scrolling)
            
            self.wait(51.19)
            # Scroll only if needed (scroll_distance > 0)
            scroll_time = 0.3 if 68.64 > 0 and scroll_distance > 0 else 0.0
            if 68.64 > 0 and scroll_distance > 0:
                seg_target_y = start_center_y + (scroll_distance * 0.000)
                # CRITICAL: Update highlight positions relative to new code position after scroll
                code_top_y = seg_target_y + (full_code.height / 2)
                # Position highlight: center of start line, then adjust for block center
                block_0_start_line_center_y = code_top_y - ((3 + 0.5) * 0.424)
                block_0_center_y = block_0_start_line_center_y - ((3 - 1) * 0.424 / 2.0)
                highlight_0.move_to([code_center_x, block_0_center_y, 0])
                highlight_glow_0.move_to([code_center_x, block_0_center_y, 0])
                self.play(full_code.animate.move_to([code_center_x, seg_target_y, 0]), run_time=0.3)
            else:
                # No scrolling - highlights are already positioned correctly in highlight_positioning
                # Just ensure they're at the right position relative to current code position
                code_top_y = full_code.get_top()[1]
                # Position highlight: center of start line, then adjust for block center
                block_0_start_line_center_y = code_top_y - ((3 + 0.5) * 0.424)
                block_0_center_y = block_0_start_line_center_y - ((3 - 1) * 0.424 / 2.0)
                highlight_0.move_to([full_code.get_center()[0], block_0_center_y, 0])
                highlight_glow_0.move_to([full_code.get_center()[0], block_0_center_y, 0])
            
            # Enhanced highlight animation sequence
            # Step 1: Glow appears first (subtle background glow)
            self.play(
                highlight_glow_0.animate.set_stroke_opacity(0.3),
                run_time=0.2
            )
            # Step 2: Animated border drawing effect (stroke appears)
            self.play(
                highlight_0.animate.set_stroke_opacity(0.9),
                highlight_glow_0.animate.set_stroke_opacity(0.5),
                run_time=0.3
            )
            # Step 3: Fill fades in (background highlight)
            self.play(
                highlight_0.animate.set_opacity(0.25).set_stroke_opacity(0.95),
                highlight_glow_0.animate.set_stroke_opacity(0.4),
                run_time=0.2
            )
            # Step 4: Subtle pulse animation (breathing effect)
            self.play(
                highlight_0.animate.set_opacity(0.20).set_stroke_opacity(0.85),
                highlight_glow_0.animate.set_stroke_opacity(0.3),
                run_time=0.2
            )
            self.play(
                highlight_0.animate.set_opacity(0.28).set_stroke_opacity(0.95),
                highlight_glow_0.animate.set_stroke_opacity(0.45),
                run_time=0.2
            )
            remaining_time = 17.45 - scroll_time - 1.1
            if remaining_time > 0:
                self.wait(remaining_time)
            self.play(highlight_0.animate.set_opacity(0), run_time=0.2)

        
        # CRITICAL: Only fade out code when code narration has ended
        # Code narration ends at code_narration_end_time, which is already reached by animation_timeline
        # So we can fade out immediately, but ensure we're at the right time
        self.play(FadeOut(full_code), FadeOut(title), run_time=0.5)
        self.wait(0.2)
        

        concepts_title = Text("Key Concepts", font_size=38, font="Helvetica", weight=BOLD, color=BLUE)
        concepts_title.to_edge(UP, buff=0.4)
        concept_items = VGroup(*[Text(f"• {concept}", font_size=24, font="Helvetica", color=WHITE) for concept in ["Java main method as entry point", "Initialization and use of integer variable", "For loop iteration and control", "Console output using System.out.println"]])
        concept_items.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        concept_items.next_to(concepts_title, DOWN, buff=0.6)
        concept_items.to_edge(LEFT, buff=0.8)
        self.play(Write(concepts_title), run_time=0.5)
        self.wait(0.3)
        self.play(Write(concept_items), run_time=1.0)
        self.wait(8.27600932426304)

        
        # Ensure video duration matches audio duration
        # Add remaining time to match full audio duration
        if 0.00 > 0:
            self.wait(0.00)
