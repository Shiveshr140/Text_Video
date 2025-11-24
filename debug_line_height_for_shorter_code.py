from manim import *

class DebugLinePositions(Scene):
    def construct(self):
        # Create the code text
        code_lines = [
            "function findLargest(arr) {",      # Line 0
            "    let largest = arr[0]; ",       # Line 1
            "    for (let i = 1; i < arr.length; i++) {",  # Line 2 - FOR LOOP START
            "        if (arr[i] > largest) {",  # Line 3
            "            largest = arr[i]; ",   # Line 4
            "        }",                         # Line 5
            "    }",                             # Line 6 - FOR LOOP END
            "    return largest;",               # Line 7
            "}",                                 # Line 8
            "console.log(findLargest([99, 5, 3, 100, 1]));"  # Line 9
        ]
        
        full_code_text = "\n".join(code_lines)
        
        full_code = Text(
            full_code_text,
            font_size=22,
            font="Courier",
            color=WHITE,
            line_spacing=1.0
        )
        
        if full_code.width > 13:
            full_code.scale_to_fit_width(13)
        
        full_code.to_edge(LEFT, buff=0.5)
        self.add(full_code)
        
        # Get the actual top position
        code_top = full_code.get_top()[1]
        code_height = full_code.height
        code_bottom = full_code.get_bottom()[1]
        
        print(f"Code top: {code_top:.3f}")
        print(f"Code height: {code_height:.3f}")
        print(f"Code bottom: {code_bottom:.3f}")
        print(f"Number of lines: {len(code_lines)}")
        
        # Calculate line height
        # For 10 lines, there are 9 gaps between lines
        # Total height = 10 * line_height (approximately)
        calculated_line_height = code_height / len(code_lines)
        print(f"Calculated line height: {calculated_line_height:.3f}")
        
        # Now create individual line texts to see their actual positions
        individual_lines = VGroup()
        for i, line_text in enumerate(code_lines):
            line = Text(line_text, font_size=22, font="Courier", color=WHITE, line_spacing=1.0)
            individual_lines.add(line)
        
        individual_lines.arrange(DOWN, aligned_edge=LEFT, buff=0)
        individual_lines.to_edge(RIGHT, buff=0.5)
        
        # Measure the actual spacing
        if len(individual_lines) > 1:
            actual_line_height = individual_lines[0].get_bottom()[1] - individual_lines[1].get_top()[1]
            print(f"Actual spacing between lines: {actual_line_height:.3f}")
            actual_line_height_center = individual_lines[0].get_center()[1] - individual_lines[1].get_center()[1]
            print(f"Actual line height (center to center): {actual_line_height_center:.3f}")
        
        # Draw markers at calculated positions
        line_height = 0.688
        markers = VGroup()
        for i in range(len(code_lines)):
            y_pos = code_top - (i * line_height)
            marker = Line(
                start=[full_code.get_left()[0] - 0.3, y_pos, 0],
                end=[full_code.get_left()[0] - 0.1, y_pos, 0],
                color=RED
            )
            label = Text(str(i), font_size=12, color=RED)
            label.next_to(marker, LEFT, buff=0.05)
            markers.add(VGroup(marker, label))
        
        self.add(markers)
        
        # Highlight the for loop area (lines 2-6)
        for_loop_start = 2
        for_loop_end = 6
        for_loop_height = (for_loop_end - for_loop_start + 1) * line_height
        
        highlight = Rectangle(
            width=full_code.width + 0.3,
            height=for_loop_height,
            fill_opacity=0.2,
            fill_color=BLUE,
            stroke_width=2,
            stroke_color=BLUE,
            stroke_opacity=0.8
        )
        
        block_top_y = code_top - (for_loop_start * line_height)
        block_center_y = block_top_y - (for_loop_height / 2.0)
        highlight.move_to([full_code.get_center()[0], block_center_y, 0])
        
        self.add(highlight)
        
        # Add title
        title = Text(f"Line height: {line_height:.3f} | Actual: {actual_line_height_center:.3f}", 
                     font_size=18, color=YELLOW)
        title.to_edge(UP)
        self.add(title)
        
        self.wait(2)
