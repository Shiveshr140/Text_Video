from manim import *

class TestHighlightPositions(Scene):
    def construct(self):
        # Same code as in the video
        full_code = Text(
            """public class GFG {
    public static void main(String[] args)
    {
        int num = 5;
        int x = 0;
        for (int i = 1; i <= num; i++) {
            x = i - 1;
            for (int j = i; j <= num - 1; j++) {
                System.out.print(" ");
                System.out.print("  ");
            }
            for (int j = 0; j <= x; j++)
                System.out.print((i + j) < 10
                                     ? (i + j) + "  "
                                     : (i + j) + " ");
            for (int j = 1; j <= x; j++)
                System.out.print((i + x - j) < 10
                                     ? (i + x - j) + "  "
                                     : (i + x - j) + " ");
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
        
        # Position code at start position (same as video)
        code_center_x = full_code.get_left()[0] + full_code.width/2
        start_center_y = 2.5 - (full_code.height / 2)
        full_code.move_to([code_center_x, start_center_y, 0])
        
        self.add(full_code)
        
        # Measure actual line positions
        code_top_y = full_code.get_top()[1]
        print(f"\n{'='*80}")
        print(f"CODE POSITIONING DEBUG")
        print(f"{'='*80}")
        print(f"Code top Y: {code_top_y:.3f}")
        print(f"Code center Y: {full_code.get_center()[1]:.3f}")
        print(f"Code height: {full_code.height:.3f}")
        
        # Calculate line height by measuring actual text
        lines = full_code.text.split('\n')
        measured_line_height = full_code.height / len(lines)
        print(f"\nTotal lines: {len(lines)}")
        print(f"Measured line height: {measured_line_height:.3f}")
        print(f"Expected line height (0.421): 0.421")
        print(f"Difference: {abs(measured_line_height - 0.421):.3f}")
        
        # Test highlight positions for the 3 loops
        loops = [
            {"name": "Loop 1", "start": 8, "end": 11},
            {"name": "Loop 2", "start": 12, "end": 15},
            {"name": "Loop 3", "start": 16, "end": 19},
        ]
        
        print(f"\n{'='*80}")
        print(f"HIGHLIGHT POSITIONS")
        print(f"{'='*80}")
        
        for loop in loops:
            start_0idx = loop['start'] - 1
            end_0idx = loop['end'] - 1
            num_lines = loop['end'] - loop['start'] + 1
            
            # Calculate using MEASURED line height
            block_top_y_measured = code_top_y - (start_0idx * measured_line_height)
            block_height_measured = num_lines * measured_line_height
            block_center_y_measured = block_top_y_measured - (block_height_measured / 2.0)
            
            # Calculate using HARDCODED line height (0.421)
            block_top_y_hardcoded = code_top_y - (start_0idx * 0.421)
            block_height_hardcoded = num_lines * 0.421
            block_center_y_hardcoded = block_top_y_hardcoded - (block_height_hardcoded / 2.0)
            
            print(f"\n{loop['name']} (lines {loop['start']}-{loop['end']}, {num_lines} lines):")
            print(f"  Using MEASURED line height ({measured_line_height:.3f}):")
            print(f"    Top Y: {block_top_y_measured:.3f}")
            print(f"    Center Y: {block_center_y_measured:.3f}")
            print(f"  Using HARDCODED line height (0.421):")
            print(f"    Top Y: {block_top_y_hardcoded:.3f}")
            print(f"    Center Y: {block_center_y_hardcoded:.3f}")
            print(f"  Difference in center Y: {abs(block_center_y_measured - block_center_y_hardcoded):.3f}")
            
            # Create highlight with MEASURED line height
            highlight = Rectangle(
                width=full_code.width + 0.3,
                height=block_height_measured,
                fill_opacity=0.2,
                fill_color=YELLOW,
                stroke_width=2,
                stroke_color=YELLOW
            )
            highlight.move_to([code_center_x, block_center_y_measured, 0])
            self.add(highlight)
        
        self.wait(5)
