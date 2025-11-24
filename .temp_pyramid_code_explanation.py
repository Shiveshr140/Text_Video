from manim import *

class CodeExplanationScene(Scene):
    def construct(self):
        self.wait(1.2)
        title = Text("Code Explanation", font_size=38, font="Helvetica", weight=BOLD, color=BLUE)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.5)
        self.wait(1)
        
        full_code = Text(
            """function areAnagrams(str1, str2) {
    if (str1.length !== str2.length) {
        return false; 
    }
    let count1 = {};
    let count2 = {};
    for (let i = 0; i < str1.length; i++) {
        let char = str1[i];
        count1[char] = (count1[char] || 0) + 1;
    }
    for (let i = 0; i < str2.length; i++) {
        let char = str2[i];
        count2[char] = (count2[char] || 0) + 1;
    }
    for (let char in count1) {
        if (count1[char] !== count2[char]) {
            return false; 
        }
    }
    return true; 
}
console.log(areAnagrams(\"listen\", \"silent\"));""",
            font_size=22,
            font="Courier",
            color=WHITE,
            line_spacing=1.0
        )
        
        if full_code.width > 13:
            full_code.scale_to_fit_width(13)
        
        full_code.to_edge(LEFT, buff=0.5)
        
        # Highlight rectangle for block: lines 7-10 (4 lines)
        # Block height calculation: 4 lines × 0.424 line_height = 1.695
        highlight_0 = Rectangle(
            width=12,
            height=1.695,
            fill_opacity=0.0,
            fill_color="#4A90E2",
            stroke_width=3,
            stroke_color="#4A90E2",
            stroke_opacity=0.0
        )
        # CRITICAL: Verify highlight height covers all 4 lines
        # If highlight appears too small, increase line_height or block_height
        # Add glow effect (slightly larger, semi-transparent)
        highlight_glow_0 = Rectangle(
            width=12.4,
            height=1.795,
            fill_opacity=0.0,
            fill_color="#4A90E2",
            stroke_width=5,
            stroke_color="#4A90E2",
            stroke_opacity=0.0
        )
        # Highlight rectangle for block: lines 11-14 (4 lines)
        # Block height calculation: 4 lines × 0.424 line_height = 1.695
        highlight_1 = Rectangle(
            width=12,
            height=1.695,
            fill_opacity=0.0,
            fill_color="#4A90E2",
            stroke_width=3,
            stroke_color="#4A90E2",
            stroke_opacity=0.0
        )
        # CRITICAL: Verify highlight height covers all 4 lines
        # If highlight appears too small, increase line_height or block_height
        # Add glow effect (slightly larger, semi-transparent)
        highlight_glow_1 = Rectangle(
            width=12.4,
            height=1.795,
            fill_opacity=0.0,
            fill_color="#4A90E2",
            stroke_width=5,
            stroke_color="#4A90E2",
            stroke_opacity=0.0
        )
        # Highlight rectangle for block: lines 15-19 (5 lines)
        # Block height calculation: 5 lines × 0.424 line_height = 2.119
        highlight_2 = Rectangle(
            width=12,
            height=2.119,
            fill_opacity=0.0,
            fill_color="#4A90E2",
            stroke_width=3,
            stroke_color="#4A90E2",
            stroke_opacity=0.0
        )
        # CRITICAL: Verify highlight height covers all 5 lines
        # If highlight appears too small, increase line_height or block_height
        # Add glow effect (slightly larger, semi-transparent)
        highlight_glow_2 = Rectangle(
            width=12.4,
            height=2.219,
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
            
            # Position highlight for lines 7-10 (0-indexed: 6-9)
            # Positioning at display line 7: full_code.get_top()[1] - (7 * 0.424)
            # Center of block is at: center_of_start_line - ((5 - 1) * 0.424 / 2)
            block_0_start_line_center_y = full_code.get_top()[1] - (7 * 0.424)
            block_0_center_y = block_0_start_line_center_y - ((5 - 1) * 0.424 / 2.0)
            highlight_0.move_to([full_code.get_center()[0], block_0_center_y, 0])
            highlight_0.stretch_to_fit_width(full_code.width + 0.3)
            # Highlight height: 2.119 (for 5 lines)
            highlight_0.set_z_index(-1)
            highlight_0.set_opacity(0)
            highlight_0.set_stroke_opacity(0)
            highlight_glow_0.move_to([full_code.get_center()[0], block_0_center_y, 0])
            highlight_glow_0.stretch_to_fit_width(full_code.width + 0.3)
            highlight_glow_0.set_z_index(-2)
            highlight_glow_0.set_opacity(0)
            highlight_glow_0.set_stroke_opacity(0)
            # Position highlight for lines 11-14 (0-indexed: 10-13)
            # Positioning at display line 11: full_code.get_top()[1] - (11 * 0.424)
            # Center of block is at: center_of_start_line - ((5 - 1) * 0.424 / 2)
            block_1_start_line_center_y = full_code.get_top()[1] - (11 * 0.424)
            block_1_center_y = block_1_start_line_center_y - ((5 - 1) * 0.424 / 2.0)
            highlight_1.move_to([full_code.get_center()[0], block_1_center_y, 0])
            highlight_1.stretch_to_fit_width(full_code.width + 0.3)
            # Highlight height: 2.119 (for 5 lines)
            highlight_1.set_z_index(-1)
            highlight_1.set_opacity(0)
            highlight_1.set_stroke_opacity(0)
            highlight_glow_1.move_to([full_code.get_center()[0], block_1_center_y, 0])
            highlight_glow_1.stretch_to_fit_width(full_code.width + 0.3)
            highlight_glow_1.set_z_index(-2)
            highlight_glow_1.set_opacity(0)
            highlight_glow_1.set_stroke_opacity(0)
            # Position highlight for lines 15-19 (0-indexed: 14-18)
            # Positioning at display line 15: full_code.get_top()[1] - (15 * 0.424)
            # Center of block is at: center_of_start_line - ((5 - 1) * 0.424 / 2)
            block_2_start_line_center_y = full_code.get_top()[1] - (15 * 0.424)
            block_2_center_y = block_2_start_line_center_y - ((5 - 1) * 0.424 / 2.0)
            highlight_2.move_to([full_code.get_center()[0], block_2_center_y, 0])
            highlight_2.stretch_to_fit_width(full_code.width + 0.3)
            # Highlight height: 2.119 (for 5 lines)
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
            
            self.wait(30.17)
            # Scroll only if needed (scroll_progress > 0 means actual scrolling)
            scroll_time = 0.3 if 0.000 > 0 and scroll_distance > 0 else 0.0
            if 0.000 > 0 and scroll_distance > 0:
                seg_target_y = start_center_y + (scroll_distance * 0.000)
                # CRITICAL: Scroll FIRST, then get the actual top position
                self.play(full_code.animate.move_to([code_center_x, seg_target_y, 0]), run_time=0.3)
                # Now get the ACTUAL top position after scroll
                code_top_y = full_code.get_top()[1]
                # Position highlight: use (N + 0.5) for scrolling case
                # (N+1 was 1 line below, N was 2 lines above, so N+0.5 should be perfect)
                block_0_start_line_center_y = code_top_y - (6.5 * 0.424)
                block_0_center_y = block_0_start_line_center_y - ((4 - 1) * 0.424 / 2.0)
                highlight_0.move_to([code_center_x, block_0_center_y, 0])
                highlight_glow_0.move_to([code_center_x, block_0_center_y, 0])
            else:
                # No scrolling - use N+1 (Confirmed correct for Loop 1)
                code_top_y = full_code.get_top()[1]
                block_0_start_line_center_y = code_top_y - (7 * 0.424)
                block_0_center_y = block_0_start_line_center_y - ((4 - 1) * 0.424 / 2.0)
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
            remaining_time = 24.00 - scroll_time - 1.1
            if remaining_time > 0:
                self.wait(remaining_time)
            # Fade out previous highlight
            self.play(
                highlight_0.animate.set_opacity(0).set_stroke_opacity(0),
                highlight_glow_0.animate.set_opacity(0).set_stroke_opacity(0),
                run_time=0.3
            )
            # Scroll only if needed (scroll_progress > 0 means actual scrolling)
            scroll_time = 0.3 if 0.625 > 0 and scroll_distance > 0 else 0.0
            if 0.625 > 0 and scroll_distance > 0:
                seg_target_y = start_center_y + (scroll_distance * 0.625)
                # CRITICAL: Scroll FIRST, then get the actual top position
                self.play(full_code.animate.move_to([code_center_x, seg_target_y, 0]), run_time=0.3)
                # Now get the ACTUAL top position after scroll
                code_top_y = full_code.get_top()[1]
                # Position highlight: use (N + 0.5) for scrolling case
                # (N+1 was 1 line below, N was 2 lines above, so N+0.5 should be perfect)
                block_1_start_line_center_y = code_top_y - (10.5 * 0.424)
                block_1_center_y = block_1_start_line_center_y - ((4 - 1) * 0.424 / 2.0)
                highlight_1.move_to([code_center_x, block_1_center_y, 0])
                highlight_glow_1.move_to([code_center_x, block_1_center_y, 0])
            else:
                # No scrolling - use N+1 (Confirmed correct for Loop 1)
                code_top_y = full_code.get_top()[1]
                block_1_start_line_center_y = code_top_y - (11 * 0.424)
                block_1_center_y = block_1_start_line_center_y - ((4 - 1) * 0.424 / 2.0)
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
            remaining_time = 25.63 - scroll_time - 1.1
            if remaining_time > 0:
                self.wait(remaining_time)
            # Fade out previous highlight
            self.play(
                highlight_1.animate.set_opacity(0).set_stroke_opacity(0),
                highlight_glow_1.animate.set_opacity(0).set_stroke_opacity(0),
                run_time=0.3
            )
            # Scroll only if needed (scroll_progress > 0 means actual scrolling)
            scroll_time = 0.3 if 1.000 > 0 and scroll_distance > 0 else 0.0
            if 1.000 > 0 and scroll_distance > 0:
                seg_target_y = start_center_y + (scroll_distance * 1.000)
                # CRITICAL: Scroll FIRST, then get the actual top position
                self.play(full_code.animate.move_to([code_center_x, seg_target_y, 0]), run_time=0.3)
                # Now get the ACTUAL top position after scroll
                code_top_y = full_code.get_top()[1]
                # Position highlight: use (N + 0.5) for scrolling case
                # (N+1 was 1 line below, N was 2 lines above, so N+0.5 should be perfect)
                block_2_start_line_center_y = code_top_y - (14.5 * 0.424)
                block_2_center_y = block_2_start_line_center_y - ((5 - 1) * 0.424 / 2.0)
                highlight_2.move_to([code_center_x, block_2_center_y, 0])
                highlight_glow_2.move_to([code_center_x, block_2_center_y, 0])
            else:
                # No scrolling - use N+1 (Confirmed correct for Loop 1)
                code_top_y = full_code.get_top()[1]
                block_2_start_line_center_y = code_top_y - (15 * 0.424)
                block_2_center_y = block_2_start_line_center_y - ((5 - 1) * 0.424 / 2.0)
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
            remaining_time = 23.90 - scroll_time - 1.1
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
            
            # Position highlight for lines 7-10 (0-indexed: 6-9)
            # Positioning at display line 7: full_code.get_top()[1] - (7 * 0.424)
            # Center of block is at: center_of_start_line - ((5 - 1) * 0.424 / 2)
            block_0_start_line_center_y = full_code.get_top()[1] - (7 * 0.424)
            block_0_center_y = block_0_start_line_center_y - ((5 - 1) * 0.424 / 2.0)
            highlight_0.move_to([full_code.get_center()[0], block_0_center_y, 0])
            highlight_0.stretch_to_fit_width(full_code.width + 0.3)
            # Highlight height: 2.119 (for 5 lines)
            highlight_0.set_z_index(-1)
            highlight_0.set_opacity(0)
            highlight_0.set_stroke_opacity(0)
            highlight_glow_0.move_to([full_code.get_center()[0], block_0_center_y, 0])
            highlight_glow_0.stretch_to_fit_width(full_code.width + 0.3)
            highlight_glow_0.set_z_index(-2)
            highlight_glow_0.set_opacity(0)
            highlight_glow_0.set_stroke_opacity(0)
            # Position highlight for lines 11-14 (0-indexed: 10-13)
            # Positioning at display line 11: full_code.get_top()[1] - (11 * 0.424)
            # Center of block is at: center_of_start_line - ((5 - 1) * 0.424 / 2)
            block_1_start_line_center_y = full_code.get_top()[1] - (11 * 0.424)
            block_1_center_y = block_1_start_line_center_y - ((5 - 1) * 0.424 / 2.0)
            highlight_1.move_to([full_code.get_center()[0], block_1_center_y, 0])
            highlight_1.stretch_to_fit_width(full_code.width + 0.3)
            # Highlight height: 2.119 (for 5 lines)
            highlight_1.set_z_index(-1)
            highlight_1.set_opacity(0)
            highlight_1.set_stroke_opacity(0)
            highlight_glow_1.move_to([full_code.get_center()[0], block_1_center_y, 0])
            highlight_glow_1.stretch_to_fit_width(full_code.width + 0.3)
            highlight_glow_1.set_z_index(-2)
            highlight_glow_1.set_opacity(0)
            highlight_glow_1.set_stroke_opacity(0)
            # Position highlight for lines 15-19 (0-indexed: 14-18)
            # Positioning at display line 15: full_code.get_top()[1] - (15 * 0.424)
            # Center of block is at: center_of_start_line - ((5 - 1) * 0.424 / 2)
            block_2_start_line_center_y = full_code.get_top()[1] - (15 * 0.424)
            block_2_center_y = block_2_start_line_center_y - ((5 - 1) * 0.424 / 2.0)
            highlight_2.move_to([full_code.get_center()[0], block_2_center_y, 0])
            highlight_2.stretch_to_fit_width(full_code.width + 0.3)
            # Highlight height: 2.119 (for 5 lines)
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
            
            self.wait(30.17)
            # Scroll only if needed (scroll_progress > 0 means actual scrolling)
            scroll_time = 0.3 if 0.000 > 0 and scroll_distance > 0 else 0.0
            if 0.000 > 0 and scroll_distance > 0:
                seg_target_y = start_center_y + (scroll_distance * 0.000)
                # CRITICAL: Scroll FIRST, then get the actual top position
                self.play(full_code.animate.move_to([code_center_x, seg_target_y, 0]), run_time=0.3)
                # Now get the ACTUAL top position after scroll
                code_top_y = full_code.get_top()[1]
                # Position highlight: use (N + 0.5) for scrolling case
                # (N+1 was 1 line below, N was 2 lines above, so N+0.5 should be perfect)
                block_0_start_line_center_y = code_top_y - (6.5 * 0.424)
                block_0_center_y = block_0_start_line_center_y - ((4 - 1) * 0.424 / 2.0)
                highlight_0.move_to([code_center_x, block_0_center_y, 0])
                highlight_glow_0.move_to([code_center_x, block_0_center_y, 0])
            else:
                # No scrolling - use N+1 (Confirmed correct for Loop 1)
                code_top_y = full_code.get_top()[1]
                block_0_start_line_center_y = code_top_y - (7 * 0.424)
                block_0_center_y = block_0_start_line_center_y - ((4 - 1) * 0.424 / 2.0)
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
            remaining_time = 24.00 - scroll_time - 1.1
            if remaining_time > 0:
                self.wait(remaining_time)
            # Fade out previous highlight
            self.play(
                highlight_0.animate.set_opacity(0).set_stroke_opacity(0),
                highlight_glow_0.animate.set_opacity(0).set_stroke_opacity(0),
                run_time=0.3
            )
            # Scroll only if needed (scroll_progress > 0 means actual scrolling)
            scroll_time = 0.3 if 0.625 > 0 and scroll_distance > 0 else 0.0
            if 0.625 > 0 and scroll_distance > 0:
                seg_target_y = start_center_y + (scroll_distance * 0.625)
                # CRITICAL: Scroll FIRST, then get the actual top position
                self.play(full_code.animate.move_to([code_center_x, seg_target_y, 0]), run_time=0.3)
                # Now get the ACTUAL top position after scroll
                code_top_y = full_code.get_top()[1]
                # Position highlight: use (N + 0.5) for scrolling case
                # (N+1 was 1 line below, N was 2 lines above, so N+0.5 should be perfect)
                block_1_start_line_center_y = code_top_y - (10.5 * 0.424)
                block_1_center_y = block_1_start_line_center_y - ((4 - 1) * 0.424 / 2.0)
                highlight_1.move_to([code_center_x, block_1_center_y, 0])
                highlight_glow_1.move_to([code_center_x, block_1_center_y, 0])
            else:
                # No scrolling - use N+1 (Confirmed correct for Loop 1)
                code_top_y = full_code.get_top()[1]
                block_1_start_line_center_y = code_top_y - (11 * 0.424)
                block_1_center_y = block_1_start_line_center_y - ((4 - 1) * 0.424 / 2.0)
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
            remaining_time = 25.63 - scroll_time - 1.1
            if remaining_time > 0:
                self.wait(remaining_time)
            # Fade out previous highlight
            self.play(
                highlight_1.animate.set_opacity(0).set_stroke_opacity(0),
                highlight_glow_1.animate.set_opacity(0).set_stroke_opacity(0),
                run_time=0.3
            )
            # Scroll only if needed (scroll_progress > 0 means actual scrolling)
            scroll_time = 0.3 if 1.000 > 0 and scroll_distance > 0 else 0.0
            if 1.000 > 0 and scroll_distance > 0:
                seg_target_y = start_center_y + (scroll_distance * 1.000)
                # CRITICAL: Scroll FIRST, then get the actual top position
                self.play(full_code.animate.move_to([code_center_x, seg_target_y, 0]), run_time=0.3)
                # Now get the ACTUAL top position after scroll
                code_top_y = full_code.get_top()[1]
                # Position highlight: use (N + 0.5) for scrolling case
                # (N+1 was 1 line below, N was 2 lines above, so N+0.5 should be perfect)
                block_2_start_line_center_y = code_top_y - (14.5 * 0.424)
                block_2_center_y = block_2_start_line_center_y - ((5 - 1) * 0.424 / 2.0)
                highlight_2.move_to([code_center_x, block_2_center_y, 0])
                highlight_glow_2.move_to([code_center_x, block_2_center_y, 0])
            else:
                # No scrolling - use N+1 (Confirmed correct for Loop 1)
                code_top_y = full_code.get_top()[1]
                block_2_start_line_center_y = code_top_y - (15 * 0.424)
                block_2_center_y = block_2_start_line_center_y - ((5 - 1) * 0.424 / 2.0)
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
            remaining_time = 23.90 - scroll_time - 1.1
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
        concept_items = VGroup(*[Text(f"• {concept}", font_size=24, font="Helvetica", color=WHITE) for concept in ["Anagram determination", "String length comparison", "Character frequency count", "Object iteration for comparison", "Discrepancy check in frequency counts"]])
        concept_items.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        concept_items.next_to(concepts_title, DOWN, buff=0.6)
        concept_items.to_edge(LEFT, buff=0.8)
        self.play(Write(concepts_title), run_time=0.5)
        self.wait(0.3)
        self.play(Write(concept_items), run_time=1.0)
        self.wait(9.596008718820865)

        
        # Ensure video duration matches audio duration
        # Add remaining time to match full audio duration
        if 0.00 > 0:
            self.wait(0.00)
