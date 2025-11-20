from manim import *

class CodeExplanationScene(Scene):
    def construct(self):
        self.wait(1.2)
        title = Text("Code Explanation", font_size=38, font="Helvetica", weight=BOLD, color=BLUE)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.5)
        self.wait(1)
        
        full_code = Text(
            """// Java Program to Print the Pyramid pattern
// Main class
public class GFG {
    // Main driver method
    public static void main(String[] args)
    {
        int num = 5;
        int x = 0;
        // Outer loop for rows
        for (int i = 1; i <= num; i++) {
            x = i - 1;
            // inner loop for \"i\"th row printing
            for (int j = i; j <= num - 1; j++) {
                // First Number Space
                System.out.print(\" \");
                // Space between Numbers
                System.out.print(\"  \");
            }
            // Pyramid printing
            for (int j = 0; j <= x; j++)
                System.out.print((i + j) < 10
                                     ? (i + j) + \"  \"
                                     : (i + j) + \" \");
            for (int j = 1; j <= x; j++)
                System.out.print((i + x - j) < 10
                                     ? (i + x - j) + \"  \"
                                     : (i + x - j) + \" \");
            // By now we reach end for one row, so
            // new line to switch to next
            System.out.println();
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
        
        # Highlight rectangle for block: lines 17-24 (5 lines)
        # Block height calculation: 5 lines × 0.688 line_height = 3.438
        highlight_0 = Rectangle(
            width=12,
            height=3.438,
            fill_opacity=0.0,
            fill_color="#4A90E2",
            stroke_width=3,
            stroke_color="#4A90E2",
            stroke_opacity=0.0
        )
        # CRITICAL: Verify highlight height covers all 5 lines
        # If highlight appears too small, increase line_height or block_height
        # Add glow effect (slightly larger, semi-transparent)
        highlight_glow_0 = Rectangle(
            width=12.4,
            height=3.538,
            fill_opacity=0.0,
            fill_color="#4A90E2",
            stroke_width=5,
            stroke_color="#4A90E2",
            stroke_opacity=0.0
        )
        # Highlight rectangle for block: lines 27-28 (1 lines)
        # Block height calculation: 1 lines × 0.688 line_height = 0.688
        highlight_1 = Rectangle(
            width=12,
            height=0.688,
            fill_opacity=0.0,
            fill_color="#4A90E2",
            stroke_width=3,
            stroke_color="#4A90E2",
            stroke_opacity=0.0
        )
        # CRITICAL: Verify highlight height covers all 1 lines
        # If highlight appears too small, increase line_height or block_height
        # Add glow effect (slightly larger, semi-transparent)
        highlight_glow_1 = Rectangle(
            width=12.4,
            height=0.787,
            fill_opacity=0.0,
            fill_color="#4A90E2",
            stroke_width=5,
            stroke_color="#4A90E2",
            stroke_opacity=0.0
        )
        # Highlight rectangle for block: lines 32-33 (1 lines)
        # Block height calculation: 1 lines × 0.688 line_height = 0.688
        highlight_2 = Rectangle(
            width=12,
            height=0.688,
            fill_opacity=0.0,
            fill_color="#4A90E2",
            stroke_width=3,
            stroke_color="#4A90E2",
            stroke_opacity=0.0
        )
        # CRITICAL: Verify highlight height covers all 1 lines
        # If highlight appears too small, increase line_height or block_height
        # Add glow effect (slightly larger, semi-transparent)
        highlight_glow_2 = Rectangle(
            width=12.4,
            height=0.787,
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
            
            block_0_center_y = full_code.get_top()[1] - (13.0 * 0.688)
            highlight_0.move_to([full_code.get_center()[0], block_0_center_y, 0])
            highlight_0.stretch_to_fit_width(full_code.width + 0.3)
            # Highlight height: 0.688 (for 1 lines)
            highlight_0.set_z_index(-1)
            highlight_0.set_opacity(0)
            highlight_0.set_stroke_opacity(0)
            highlight_glow_0.move_to([full_code.get_center()[0], block_0_center_y, 0])
            highlight_glow_0.stretch_to_fit_width(full_code.width + 0.3)
            highlight_glow_0.set_z_index(-2)
            highlight_glow_0.set_opacity(0)
            highlight_glow_0.set_stroke_opacity(0)
            block_1_center_y = full_code.get_top()[1] - (18.0 * 0.688)
            highlight_1.move_to([full_code.get_center()[0], block_1_center_y, 0])
            highlight_1.stretch_to_fit_width(full_code.width + 0.3)
            # Highlight height: 0.688 (for 1 lines)
            highlight_1.set_z_index(-1)
            highlight_1.set_opacity(0)
            highlight_1.set_stroke_opacity(0)
            highlight_glow_1.move_to([full_code.get_center()[0], block_1_center_y, 0])
            highlight_glow_1.stretch_to_fit_width(full_code.width + 0.3)
            highlight_glow_1.set_z_index(-2)
            highlight_glow_1.set_opacity(0)
            highlight_glow_1.set_stroke_opacity(0)
            block_2_center_y = full_code.get_top()[1] - (22.0 * 0.688)
            highlight_2.move_to([full_code.get_center()[0], block_2_center_y, 0])
            highlight_2.stretch_to_fit_width(full_code.width + 0.3)
            # Highlight height: 0.688 (for 1 lines)
            highlight_2.set_z_index(-1)
            highlight_2.set_opacity(0)
            highlight_2.set_stroke_opacity(0)
            highlight_glow_2.move_to([full_code.get_center()[0], block_2_center_y, 0])
            highlight_glow_2.stretch_to_fit_width(full_code.width + 0.3)
            highlight_glow_2.set_z_index(-2)
            highlight_glow_2.set_opacity(0)
            highlight_glow_2.set_stroke_opacity(0)

            
            # Add highlights to scene
            all_highlights = VGroup(highlight_0, highlight_glow_0, highlight_1, highlight_glow_1, highlight_2, highlight_glow_2)
            self.add(all_highlights)
            self.play(FadeIn(full_code), run_time=0.5)
            
            self.wait(87.67)
            # Scroll only if needed (scroll_distance > 0)
            scroll_time = 0.3 if 163.22 > 0 and scroll_distance > 0 else 0.0
            if 163.22 > 0 and scroll_distance > 0:
                seg_target_y = start_center_y + (scroll_distance * 0.267)
                # CRITICAL: Update highlight positions relative to new code position after scroll
                code_top_y = seg_target_y + (full_code.height / 2)
                block_0_center_y = code_top_y - (22.0 * 0.688)
                highlight_0.move_to([code_center_x, block_0_center_y, 0])
                highlight_glow_0.move_to([code_center_x, block_0_center_y, 0])
                self.play(full_code.animate.move_to([code_center_x, seg_target_y, 0]), run_time=0.3)
            else:
                # No scrolling - highlights are already positioned correctly in highlight_positioning
                # Just ensure they're at the right position relative to current code position
                code_top_y = full_code.get_top()[1]
                block_0_center_y = code_top_y - (22.0 * 0.688)
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
            remaining_time = 20.02 - scroll_time - 1.1
            if remaining_time > 0:
                self.wait(remaining_time)
            # Fade out previous highlight
            self.play(
                highlight_0.animate.set_opacity(0).set_stroke_opacity(0),
                highlight_glow_0.animate.set_opacity(0).set_stroke_opacity(0),
                run_time=0.3
            )
            # Scroll only if needed (scroll_distance > 0)
            scroll_time = 0.3 if 163.22 > 0 and scroll_distance > 0 else 0.0
            if 163.22 > 0 and scroll_distance > 0:
                seg_target_y = start_center_y + (scroll_distance * 0.600)
                # CRITICAL: Update highlight positions relative to new code position after scroll
                code_top_y = seg_target_y + (full_code.height / 2)
                block_1_center_y = code_top_y - (22.0 * 0.688)
                highlight_1.move_to([code_center_x, block_1_center_y, 0])
                highlight_glow_1.move_to([code_center_x, block_1_center_y, 0])
                self.play(full_code.animate.move_to([code_center_x, seg_target_y, 0]), run_time=0.3)
            else:
                # No scrolling - highlights are already positioned correctly in highlight_positioning
                # Just ensure they're at the right position relative to current code position
                code_top_y = full_code.get_top()[1]
                block_1_center_y = code_top_y - (22.0 * 0.688)
                highlight_1.move_to([full_code.get_center()[0], block_1_center_y, 0])
                highlight_glow_1.move_to([full_code.get_center()[0], block_1_center_y, 0])
            
            # Enhanced highlight animation sequence
            # Step 1: Glow appears first (subtle background glow)
            self.play(
                highlight_glow_1.animate.set_stroke_opacity(0.3),
                run_time=0.2
            )
            # Step 2: Animated border drawing effect (stroke appears)
            self.play(
                highlight_1.animate.set_stroke_opacity(0.9),
                highlight_glow_1.animate.set_stroke_opacity(0.5),
                run_time=0.3
            )
            # Step 3: Fill fades in (background highlight)
            self.play(
                highlight_1.animate.set_opacity(0.25).set_stroke_opacity(0.95),
                highlight_glow_1.animate.set_stroke_opacity(0.4),
                run_time=0.2
            )
            # Step 4: Subtle pulse animation (breathing effect)
            self.play(
                highlight_1.animate.set_opacity(0.20).set_stroke_opacity(0.85),
                highlight_glow_1.animate.set_stroke_opacity(0.3),
                run_time=0.2
            )
            self.play(
                highlight_1.animate.set_opacity(0.28).set_stroke_opacity(0.95),
                highlight_glow_1.animate.set_stroke_opacity(0.45),
                run_time=0.2
            )
            remaining_time = 27.43 - scroll_time - 1.1
            if remaining_time > 0:
                self.wait(remaining_time)
            # Fade out previous highlight
            self.play(
                highlight_1.animate.set_opacity(0).set_stroke_opacity(0),
                highlight_glow_1.animate.set_opacity(0).set_stroke_opacity(0),
                run_time=0.3
            )
            # Scroll only if needed (scroll_distance > 0)
            scroll_time = 0.3 if 163.22 > 0 and scroll_distance > 0 else 0.0
            if 163.22 > 0 and scroll_distance > 0:
                seg_target_y = start_center_y + (scroll_distance * 0.867)
                # CRITICAL: Update highlight positions relative to new code position after scroll
                code_top_y = seg_target_y + (full_code.height / 2)
                block_2_center_y = code_top_y - (22.0 * 0.688)
                highlight_2.move_to([code_center_x, block_2_center_y, 0])
                highlight_glow_2.move_to([code_center_x, block_2_center_y, 0])
                self.play(full_code.animate.move_to([code_center_x, seg_target_y, 0]), run_time=0.3)
            else:
                # No scrolling - highlights are already positioned correctly in highlight_positioning
                # Just ensure they're at the right position relative to current code position
                code_top_y = full_code.get_top()[1]
                block_2_center_y = code_top_y - (22.0 * 0.688)
                highlight_2.move_to([full_code.get_center()[0], block_2_center_y, 0])
                highlight_glow_2.move_to([full_code.get_center()[0], block_2_center_y, 0])
            
            # Enhanced highlight animation sequence
            # Step 1: Glow appears first (subtle background glow)
            self.play(
                highlight_glow_2.animate.set_stroke_opacity(0.3),
                run_time=0.2
            )
            # Step 2: Animated border drawing effect (stroke appears)
            self.play(
                highlight_2.animate.set_stroke_opacity(0.9),
                highlight_glow_2.animate.set_stroke_opacity(0.5),
                run_time=0.3
            )
            # Step 3: Fill fades in (background highlight)
            self.play(
                highlight_2.animate.set_opacity(0.25).set_stroke_opacity(0.95),
                highlight_glow_2.animate.set_stroke_opacity(0.4),
                run_time=0.2
            )
            # Step 4: Subtle pulse animation (breathing effect)
            self.play(
                highlight_2.animate.set_opacity(0.20).set_stroke_opacity(0.85),
                highlight_glow_2.animate.set_stroke_opacity(0.3),
                run_time=0.2
            )
            self.play(
                highlight_2.animate.set_opacity(0.28).set_stroke_opacity(0.95),
                highlight_glow_2.animate.set_stroke_opacity(0.45),
                run_time=0.2
            )
            remaining_time = 28.10 - scroll_time - 1.1
            if remaining_time > 0:
                self.wait(remaining_time)
            self.play(highlight_2.animate.set_opacity(0), run_time=0.2)

        else:
            # Code fits on screen - still need highlights!
            full_code.next_to(title, DOWN, buff=0.3)
            
            # Define variables needed for animation timeline (even without scrolling)
            code_center_x = full_code.get_center()[0]
            start_center_y = full_code.get_center()[1]
            scroll_distance = 0  # No scrolling needed
            
            # Position highlights even when code doesn't scroll
            
            block_0_center_y = full_code.get_top()[1] - (13.0 * 0.688)
            highlight_0.move_to([full_code.get_center()[0], block_0_center_y, 0])
            highlight_0.stretch_to_fit_width(full_code.width + 0.3)
            # Highlight height: 0.688 (for 1 lines)
            highlight_0.set_z_index(-1)
            highlight_0.set_opacity(0)
            highlight_0.set_stroke_opacity(0)
            highlight_glow_0.move_to([full_code.get_center()[0], block_0_center_y, 0])
            highlight_glow_0.stretch_to_fit_width(full_code.width + 0.3)
            highlight_glow_0.set_z_index(-2)
            highlight_glow_0.set_opacity(0)
            highlight_glow_0.set_stroke_opacity(0)
            block_1_center_y = full_code.get_top()[1] - (18.0 * 0.688)
            highlight_1.move_to([full_code.get_center()[0], block_1_center_y, 0])
            highlight_1.stretch_to_fit_width(full_code.width + 0.3)
            # Highlight height: 0.688 (for 1 lines)
            highlight_1.set_z_index(-1)
            highlight_1.set_opacity(0)
            highlight_1.set_stroke_opacity(0)
            highlight_glow_1.move_to([full_code.get_center()[0], block_1_center_y, 0])
            highlight_glow_1.stretch_to_fit_width(full_code.width + 0.3)
            highlight_glow_1.set_z_index(-2)
            highlight_glow_1.set_opacity(0)
            highlight_glow_1.set_stroke_opacity(0)
            block_2_center_y = full_code.get_top()[1] - (22.0 * 0.688)
            highlight_2.move_to([full_code.get_center()[0], block_2_center_y, 0])
            highlight_2.stretch_to_fit_width(full_code.width + 0.3)
            # Highlight height: 0.688 (for 1 lines)
            highlight_2.set_z_index(-1)
            highlight_2.set_opacity(0)
            highlight_2.set_stroke_opacity(0)
            highlight_glow_2.move_to([full_code.get_center()[0], block_2_center_y, 0])
            highlight_glow_2.stretch_to_fit_width(full_code.width + 0.3)
            highlight_glow_2.set_z_index(-2)
            highlight_glow_2.set_opacity(0)
            highlight_glow_2.set_stroke_opacity(0)

            
            # Add highlights to scene
            all_highlights = VGroup(highlight_0, highlight_glow_0, highlight_1, highlight_glow_1, highlight_2, highlight_glow_2)
            self.add(all_highlights)
            
            self.play(FadeIn(full_code), run_time=0.5)
            
            # Run animation timeline for highlights (even without scrolling)
            
            self.wait(87.67)
            # Scroll only if needed (scroll_distance > 0)
            scroll_time = 0.3 if 163.22 > 0 and scroll_distance > 0 else 0.0
            if 163.22 > 0 and scroll_distance > 0:
                seg_target_y = start_center_y + (scroll_distance * 0.267)
                # CRITICAL: Update highlight positions relative to new code position after scroll
                code_top_y = seg_target_y + (full_code.height / 2)
                block_0_center_y = code_top_y - (22.0 * 0.688)
                highlight_0.move_to([code_center_x, block_0_center_y, 0])
                highlight_glow_0.move_to([code_center_x, block_0_center_y, 0])
                self.play(full_code.animate.move_to([code_center_x, seg_target_y, 0]), run_time=0.3)
            else:
                # No scrolling - highlights are already positioned correctly in highlight_positioning
                # Just ensure they're at the right position relative to current code position
                code_top_y = full_code.get_top()[1]
                block_0_center_y = code_top_y - (22.0 * 0.688)
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
            remaining_time = 20.02 - scroll_time - 1.1
            if remaining_time > 0:
                self.wait(remaining_time)
            # Fade out previous highlight
            self.play(
                highlight_0.animate.set_opacity(0).set_stroke_opacity(0),
                highlight_glow_0.animate.set_opacity(0).set_stroke_opacity(0),
                run_time=0.3
            )
            # Scroll only if needed (scroll_distance > 0)
            scroll_time = 0.3 if 163.22 > 0 and scroll_distance > 0 else 0.0
            if 163.22 > 0 and scroll_distance > 0:
                seg_target_y = start_center_y + (scroll_distance * 0.600)
                # CRITICAL: Update highlight positions relative to new code position after scroll
                code_top_y = seg_target_y + (full_code.height / 2)
                block_1_center_y = code_top_y - (22.0 * 0.688)
                highlight_1.move_to([code_center_x, block_1_center_y, 0])
                highlight_glow_1.move_to([code_center_x, block_1_center_y, 0])
                self.play(full_code.animate.move_to([code_center_x, seg_target_y, 0]), run_time=0.3)
            else:
                # No scrolling - highlights are already positioned correctly in highlight_positioning
                # Just ensure they're at the right position relative to current code position
                code_top_y = full_code.get_top()[1]
                block_1_center_y = code_top_y - (22.0 * 0.688)
                highlight_1.move_to([full_code.get_center()[0], block_1_center_y, 0])
                highlight_glow_1.move_to([full_code.get_center()[0], block_1_center_y, 0])
            
            # Enhanced highlight animation sequence
            # Step 1: Glow appears first (subtle background glow)
            self.play(
                highlight_glow_1.animate.set_stroke_opacity(0.3),
                run_time=0.2
            )
            # Step 2: Animated border drawing effect (stroke appears)
            self.play(
                highlight_1.animate.set_stroke_opacity(0.9),
                highlight_glow_1.animate.set_stroke_opacity(0.5),
                run_time=0.3
            )
            # Step 3: Fill fades in (background highlight)
            self.play(
                highlight_1.animate.set_opacity(0.25).set_stroke_opacity(0.95),
                highlight_glow_1.animate.set_stroke_opacity(0.4),
                run_time=0.2
            )
            # Step 4: Subtle pulse animation (breathing effect)
            self.play(
                highlight_1.animate.set_opacity(0.20).set_stroke_opacity(0.85),
                highlight_glow_1.animate.set_stroke_opacity(0.3),
                run_time=0.2
            )
            self.play(
                highlight_1.animate.set_opacity(0.28).set_stroke_opacity(0.95),
                highlight_glow_1.animate.set_stroke_opacity(0.45),
                run_time=0.2
            )
            remaining_time = 27.43 - scroll_time - 1.1
            if remaining_time > 0:
                self.wait(remaining_time)
            # Fade out previous highlight
            self.play(
                highlight_1.animate.set_opacity(0).set_stroke_opacity(0),
                highlight_glow_1.animate.set_opacity(0).set_stroke_opacity(0),
                run_time=0.3
            )
            # Scroll only if needed (scroll_distance > 0)
            scroll_time = 0.3 if 163.22 > 0 and scroll_distance > 0 else 0.0
            if 163.22 > 0 and scroll_distance > 0:
                seg_target_y = start_center_y + (scroll_distance * 0.867)
                # CRITICAL: Update highlight positions relative to new code position after scroll
                code_top_y = seg_target_y + (full_code.height / 2)
                block_2_center_y = code_top_y - (22.0 * 0.688)
                highlight_2.move_to([code_center_x, block_2_center_y, 0])
                highlight_glow_2.move_to([code_center_x, block_2_center_y, 0])
                self.play(full_code.animate.move_to([code_center_x, seg_target_y, 0]), run_time=0.3)
            else:
                # No scrolling - highlights are already positioned correctly in highlight_positioning
                # Just ensure they're at the right position relative to current code position
                code_top_y = full_code.get_top()[1]
                block_2_center_y = code_top_y - (22.0 * 0.688)
                highlight_2.move_to([full_code.get_center()[0], block_2_center_y, 0])
                highlight_glow_2.move_to([full_code.get_center()[0], block_2_center_y, 0])
            
            # Enhanced highlight animation sequence
            # Step 1: Glow appears first (subtle background glow)
            self.play(
                highlight_glow_2.animate.set_stroke_opacity(0.3),
                run_time=0.2
            )
            # Step 2: Animated border drawing effect (stroke appears)
            self.play(
                highlight_2.animate.set_stroke_opacity(0.9),
                highlight_glow_2.animate.set_stroke_opacity(0.5),
                run_time=0.3
            )
            # Step 3: Fill fades in (background highlight)
            self.play(
                highlight_2.animate.set_opacity(0.25).set_stroke_opacity(0.95),
                highlight_glow_2.animate.set_stroke_opacity(0.4),
                run_time=0.2
            )
            # Step 4: Subtle pulse animation (breathing effect)
            self.play(
                highlight_2.animate.set_opacity(0.20).set_stroke_opacity(0.85),
                highlight_glow_2.animate.set_stroke_opacity(0.3),
                run_time=0.2
            )
            self.play(
                highlight_2.animate.set_opacity(0.28).set_stroke_opacity(0.95),
                highlight_glow_2.animate.set_stroke_opacity(0.45),
                run_time=0.2
            )
            remaining_time = 28.10 - scroll_time - 1.1
            if remaining_time > 0:
                self.wait(remaining_time)
            self.play(highlight_2.animate.set_opacity(0), run_time=0.2)

        
        # CRITICAL: Only fade out code when code narration has ended
        # Code narration ends at code_narration_end_time, which is already reached by animation_timeline
        # So we can fade out immediately, but ensure we're at the right time
        self.play(FadeOut(full_code), FadeOut(title), run_time=0.5)
        self.wait(0.2)
        

        concepts_title = Text("Key Concepts", font_size=38, font="Helvetica", weight=BOLD, color=BLUE)
        concepts_title.to_edge(UP, buff=0.4)
        concept_items = VGroup(*[Text(f"• {concept}", font_size=24, font="Helvetica", color=WHITE) for concept in ["Pyramid pattern generation", "Nested loops for formatting", "Symmetric number alignment", "Java main method entry point", "Row and space management", "Conditional expression evaluation"]])
        concept_items.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        concept_items.next_to(concepts_title, DOWN, buff=0.6)
        concept_items.to_edge(LEFT, buff=0.8)
        self.play(Write(concepts_title), run_time=0.5)
        self.wait(0.3)
        self.play(Write(concept_items), run_time=1.0)
        self.wait(9.59600904308392)

        
        # Ensure video duration matches audio duration
        # Add remaining time to match full audio duration
        if 0.00 > 0:
            self.wait(0.00)
