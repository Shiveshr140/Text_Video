from manim import *

class DebugLinePositions(Scene):
    def construct(self):
        # Same code as in the video
        code_text = """public class GFG {
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
}"""
        
        full_code = Text(
            code_text,
            font_size=22,
            font="Courier",
            color=WHITE,
            line_spacing=1.0
        )
        
        if full_code.width > 13:
            full_code.scale_to_fit_width(13)
        
        full_code.to_edge(LEFT, buff=0.5)
        
        # Position code same as video
        code_height = full_code.height
        code_center_x = full_code.get_left()[0] + full_code.width/2
        start_center_y = 2.5 - (code_height / 2)
        full_code.move_to([code_center_x, start_center_y, 0])
        
        self.add(full_code)
        
        # Split the text and create individual Text objects for each line
        lines = code_text.split('\n')
        
        print(f"\n{'='*80}")
        print(f"ACTUAL LINE POSITIONS IN MANIM")
        print(f"{'='*80}")
        print(f"Code top: {full_code.get_top()[1]:.3f}")
        print(f"Code center: {full_code.get_center()[1]:.3f}")
        print(f"Code bottom: {full_code.get_bottom()[1]:.3f}")
        print(f"Code height: {full_code.height:.3f}")
        print(f"Total lines: {len(lines)}")
        
        # Create individual line objects to measure their actual positions
        y_positions = []
        for i, line_text in enumerate(lines):
            line_obj = Text(
                line_text,
                font_size=22,
                font="Courier",
                color=YELLOW,
                line_spacing=1.0
            )
            # Scale to match full_code scaling
            if full_code.width > 13:
                line_obj.scale_to_fit_width(13)
            
            # Position at left edge, same x as full_code
            line_obj.move_to([code_center_x, 0, 0])
            
            # Try to find this line's position by matching to full_code
            # We'll estimate based on even spacing
            estimated_y = full_code.get_top()[1] - (i * (full_code.height / len(lines)))
            y_positions.append(estimated_y)
            
            marker = ""
            if "for (int j = i; j <= num - 1" in line_text:
                marker = " ← LOOP 1"
            elif "for (int j = 0; j <= x" in line_text:
                marker = " ← LOOP 2"
            elif "for (int j = 1; j <= x" in line_text:
                marker = " ← LOOP 3"
            
            print(f"Line {i+1:2d} (0-idx {i:2d}): Y={estimated_y:6.3f} | {line_text[:50]}{marker}")
        
        # Now test our formula
        print(f"\n{'='*80}")
        print(f"TESTING OUR FORMULA")
        print(f"{'='*80}")
        
        line_height = full_code.height / len(lines)
        print(f"Calculated line_height: {line_height:.3f}")
        
        loops = [
            {"name": "Loop 1", "line_1indexed": 8, "line_0indexed": 7},
            {"name": "Loop 2", "line_1indexed": 12, "line_0indexed": 11},
            {"name": "Loop 3", "line_1indexed": 16, "line_0indexed": 15},
        ]
        
        for loop in loops:
            idx = loop['line_0indexed']
            
            # Our current formula
            our_formula_y = full_code.get_top()[1] - ((idx + 0.5) * line_height)
            
            # Actual position from even spacing
            actual_y = y_positions[idx]
            
            print(f"\n{loop['name']} (line {loop['line_1indexed']}, 0-indexed {idx}):")
            print(f"  Our formula: {our_formula_y:.3f}")
            print(f"  Actual Y:    {actual_y:.3f}")
            print(f"  Difference:  {abs(our_formula_y - actual_y):.3f}")
            
            # Draw a line marker at our formula position
            marker = Line(
                start=[full_code.get_left()[0] - 0.5, our_formula_y, 0],
                end=[full_code.get_left()[0] - 0.2, our_formula_y, 0],
                color=RED,
                stroke_width=4
            )
            self.add(marker)
            
            # Draw a line marker at actual position
            actual_marker = Line(
                start=[full_code.get_right()[0] + 0.2, actual_y, 0],
                end=[full_code.get_right()[0] + 0.5, actual_y, 0],
                color=GREEN,
                stroke_width=4
            )
            self.add(actual_marker)
        
        # Add legend
        legend = VGroup(
            Text("RED = Our formula", font_size=16, color=RED),
            Text("GREEN = Actual", font_size=16, color=GREEN)
        ).arrange(DOWN, aligned_edge=LEFT).to_corner(UR)
        self.add(legend)
        
        self.wait(10)
