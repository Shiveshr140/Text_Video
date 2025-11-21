from simple_app import parse_code_to_blocks

sample_code = """
// Java Program to Print the Pyramid pattern

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

            // inner loop for "i"th row printing
            for (int j = i; j <= num - 1; j++) {

                // First Number Space
                System.out.print(" ");

                // Space between Numbers
                System.out.print("  ");
            }

            // Pyramid printing
            for (int j = 0; j <= x; j++)
                System.out.print((i + j) < 10
                                     ? (i + j) + "  "
                                     : (i + j) + " ");

            for (int j = 1; j <= x; j++)
                System.out.print((i + x - j) < 10
                                     ? (i + x - j) + "  "
                                     : (i + x - j) + " ");

            // By now we reach end for one row, so
            // new line to switch to next
            System.out.println();
        }
    }
}
"""

print("Parsing code...")
blocks = parse_code_to_blocks(sample_code, language="java")

print(f"\nFound {len(blocks)} blocks:")
for i, block in enumerate(blocks, 1):
    print(f"Block {i}: {block['type']} (lines {block['start_line']}-{block['end_line']})")
    print(f"Code preview: {repr(block['code'][:50])}...")
    print("-" * 40)
