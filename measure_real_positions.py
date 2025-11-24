from manim import *
import sys

class MeasureRealPositions(Scene):
    def construct(self):
        # Exact same code as video
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
        
        # Create full code exactly as in video
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
        
        # Position exactly as in video
        code_height = full_code.height
        code_center_x = full_code.get_left()[0] + full_code.width/2
        start_center_y = 2.5 - (code_height / 2)
        full_code.move_to([code_center_x, start_center_y, 0])
        
        self.add(full_code)
        
        # Now create INDIVIDUAL text objects for each line to measure their REAL positions
        lines = code_text.split('\n')
        
        print(f"\n{'='*80}")
        print(f"MEASURING REAL LINE POSITIONS")
        print(f"{'='*80}")
        print(f"Full code top: {full_code.get_top()[1]:.4f}")
        print(f"Full code center: {full_code.get_center()[1]:.4f}")
        print(f"Full code height: {full_code.height:.4f}")
        print(f"Total lines: {len(lines)}")
        print(f"Calculated line_height: {full_code.height / len(lines):.4f}")
        print(f"\n{'='*80}")
        
        # Create a VGroup with individual lines to see their actual positions
        individual_lines = VGroup()
        for line_text in lines:
            line_obj = Text(
                line_text,
                font_size=22,
                font="Courier",
                color=YELLOW,
                line_spacing=1.0
            )
            individual_lines.add(line_obj)
        
        # Arrange them vertically with same spacing
        individual_lines.arrange(DOWN, buff=0, aligned_edge=LEFT)
        
        # Scale to match full_code
        if individual_lines.width > 13:
            individual_lines.scale_to_fit_width(13)
        
        # Position to match full_code
        individual_lines.move_to(full_code.get_center())
        
        # Now measure each line's actual position
        for i, line_obj in enumerate(individual_lines):
            line_text = lines[i]
            actual_y = line_obj.get_center()[1]
            
            marker = ""
            if "for (int j = i; j <= num - 1" in line_text:
                marker = " ← LOOP 1 (should be line 8)"
            elif "for (int j = 0; j <= x" in line_text:
                marker = " ← LOOP 2 (should be line 12)"
            elif "for (int j = 1; j <= x" in line_text:
                marker = " ← LOOP 3 (should be line 16)"
            
            print(f"Line {i+1:2d} (0-idx {i:2d}): Y={actual_y:7.4f} | {line_text[:40]}{marker}")
        
        # Now test all our formulas
        print(f"\n{'='*80}")
        print(f"TESTING FORMULAS")
        print(f"{'='*80}")
        
        line_height = full_code.height / len(lines)
        code_top = full_code.get_top()[1]
        
        loops = [
            {"name": "Loop 1", "line_1idx": 8, "line_0idx": 7, "actual_obj": individual_lines[7]},
            {"name": "Loop 2", "line_1idx": 12, "line_0idx": 11, "actual_obj": individual_lines[11]},
            {"name": "Loop 3", "line_1idx": 16, "line_0idx": 15, "actual_obj": individual_lines[15]},
        ]
        
        for loop in loops:
            idx = loop['line_0idx']
            actual_y = loop['actual_obj'].get_center()[1]
            
            # Test different formulas
            formula_n = code_top - (idx * line_height)
            formula_n_plus_05 = code_top - ((idx + 0.5) * line_height)
            formula_n_minus_05 = code_top - ((idx - 0.5) * line_height)
            formula_n_plus_1 = code_top - ((idx + 1) * line_height)
            
            print(f"\n{loop['name']} (line {loop['line_1idx']}, 0-indexed {idx}):")
            print(f"  ACTUAL Y:        {actual_y:7.4f}")
            print(f"  Formula (N):     {formula_n:7.4f}  diff: {abs(actual_y - formula_n):6.4f}")
            print(f"  Formula (N+0.5): {formula_n_plus_05:7.4f}  diff: {abs(actual_y - formula_n_plus_05):6.4f}")
            print(f"  Formula (N-0.5): {formula_n_minus_05:7.4f}  diff: {abs(actual_y - formula_n_minus_05):6.4f}")
            print(f"  Formula (N+1):   {formula_n_plus_1:7.4f}  diff: {abs(actual_y - formula_n_plus_1):6.4f}")
            
            # Find which formula is closest
            diffs = {
                "N": abs(actual_y - formula_n),
                "N+0.5": abs(actual_y - formula_n_plus_05),
                "N-0.5": abs(actual_y - formula_n_minus_05),
                "N+1": abs(actual_y - formula_n_plus_1),
            }
            best = min(diffs, key=diffs.get)
            print(f"  ✅ BEST FORMULA: {best} (diff: {diffs[best]:.4f})")
        
        self.wait(1)
