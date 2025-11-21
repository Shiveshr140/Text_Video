from manim import *

class TestHighlight(Scene):
    def construct(self):
        # Same code as in the video
        full_code = Text(
            """function findLargest(arr) {
    let largest = arr[0]; 
    for (let i = 1; i < arr.length; i++) {
        if (arr[i] > largest) {
            largest = arr[i]; 
        }
    }
    return largest;
}
console.log(findLargest([99, 5, 3, 100, 1]));""",
            font_size=22,
            font="Courier",
            color=WHITE,
            line_spacing=1.0
        )
        
        if full_code.width > 13:
            full_code.scale_to_fit_width(13)
        
        full_code.to_edge(LEFT, buff=0.5)
        
        # Calculate line height
        line_height = 0.688
        
        # Highlight for lines 2-6 (0-indexed) - the for loop
        # This should be 5 lines: for, if, largest=, }, }
        block_height = 5 * line_height  # 3.438
        
        highlight = Rectangle(
            width=12,
            height=block_height,
            fill_opacity=0.25,
            fill_color="#4A90E2",
            stroke_width=3,
            stroke_color="#4A90E2",
            stroke_opacity=0.95
        )
        
        # Position at line 2 (0-indexed) - the "for" line
        start_line_0indexed = 2
        block_top_y = full_code.get_top()[1] - (start_line_0indexed * line_height)
        block_center_y = block_top_y - (block_height / 2.0)
        
        highlight.move_to([full_code.get_center()[0], block_center_y, 0])
        highlight.stretch_to_fit_width(full_code.width + 0.3)
        
        # Add line numbers for debugging
        line_numbers = VGroup()
        for i in range(10):
            line_y = full_code.get_top()[1] - (i * line_height)
            num = Text(str(i), font_size=16, color=RED)
            num.move_to([full_code.get_left()[0] - 0.5, line_y, 0])
            line_numbers.add(num)
        
        # Show everything
        self.add(full_code)
        self.add(highlight)
        self.add(line_numbers)
        
        # Add labels
        title = Text("Highlight Test - Should cover FOR loop (lines 2-6)", font_size=20, color=YELLOW)
        title.to_edge(UP)
        self.add(title)
        
        # Mark the expected lines
        expected_start = Text("← Start (for)", font_size=14, color=GREEN)
        expected_start.next_to(full_code, RIGHT, buff=0.1)
        expected_start_y = full_code.get_top()[1] - (2 * line_height)
        expected_start.move_to([expected_start.get_center()[0], expected_start_y, 0])
        
        expected_end = Text("← End (})", font_size=14, color=GREEN)
        expected_end.next_to(full_code, RIGHT, buff=0.1)
        expected_end_y = full_code.get_top()[1] - (6 * line_height)
        expected_end.move_to([expected_end.get_center()[0], expected_end_y, 0])
        
        self.add(expected_start, expected_end)
        
        self.wait(2)
