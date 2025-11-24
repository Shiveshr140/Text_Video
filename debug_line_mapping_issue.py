#!/usr/bin/env python3
"""
Debug script to trace the line number mapping issue
"""

# Simulate the Java code
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

print("=" * 80)
print("ORIGINAL CODE (with comments and blank lines)")
print("=" * 80)
original_lines = sample_code.strip().split('\n')
for i, line in enumerate(original_lines, 1):
    print(f"{i:3d}: {line}")

print("\n" + "=" * 80)
print("STEP 1: Remove comments")
print("=" * 80)

def remove_comments(line):
    """Remove // comments from a line"""
    if '//' in line:
        in_string = False
        quote_char = None
        for i, char in enumerate(line):
            if char in ['"', "'"] and (i == 0 or line[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    quote_char = char
                elif char == quote_char:
                    in_string = False
                    quote_char = None
            elif char == '/' and i < len(line) - 1 and line[i+1] == '/' and not in_string:
                return line[:i].rstrip()
        return line
    return line

lines_without_comments = [remove_comments(line) for line in original_lines]
for i, line in enumerate(lines_without_comments, 1):
    marker = " <-- REMOVED" if line != original_lines[i-1] else ""
    print(f"{i:3d}: {line}{marker}")

print("\n" + "=" * 80)
print("STEP 2: Remove blank lines and create mapping")
print("=" * 80)

non_blank_lines = []
line_number_mapping = {}
new_line_num = 1

for orig_line_num, line in enumerate(lines_without_comments, 1):
    if line.strip():
        non_blank_lines.append(line)
        line_number_mapping[orig_line_num] = new_line_num
        new_line_num += 1

print(f"Line number mapping (original -> cleaned):")
for orig, new in line_number_mapping.items():
    print(f"  {orig:3d} -> {new:3d}")

print("\n" + "=" * 80)
print("CLEANED CODE (displayed in video)")
print("=" * 80)
for i, line in enumerate(non_blank_lines, 1):
    print(f"{i:3d}: {line}")

print("\n" + "=" * 80)
print("BLOCK DETECTION (on ORIGINAL code with comments)")
print("=" * 80)

# The blocks are detected on the ORIGINAL code (lines 17-24, 27-30, 32-35)
blocks = [
    {"name": "Loop 1", "start": 17, "end": 24},
    {"name": "Loop 2", "start": 27, "end": 30},
    {"name": "Loop 3", "start": 32, "end": 35},
]

for block in blocks:
    print(f"\n{block['name']}: Original lines {block['start']}-{block['end']}")
    print(f"  Original code:")
    for i in range(block['start'], block['end'] + 1):
        print(f"    {i:3d}: {original_lines[i-1]}")
    
    # Map to cleaned code
    mapped_start = line_number_mapping.get(block['start'])
    mapped_end = line_number_mapping.get(block['end'])
    
    print(f"  Mapped to cleaned lines: {mapped_start}-{mapped_end}")
    if mapped_start and mapped_end:
        print(f"  Cleaned code:")
        for i in range(mapped_start, mapped_end + 1):
            print(f"    {i:3d}: {non_blank_lines[i-1]}")

print("\n" + "=" * 80)
print("THE PROBLEM")
print("=" * 80)
print("Blocks are detected at lines 17-24, 27-30, 32-35 in ORIGINAL code")
print("But after removing comments/blanks, these map to DIFFERENT line numbers!")
print("The highlight positioning uses the ORIGINAL line numbers on CLEANED code")
print("This causes the mismatch!")
