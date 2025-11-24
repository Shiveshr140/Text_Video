#!/usr/bin/env python3
"""
Debug script to check the exact line numbers
"""

code = """public class GFG {
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

lines = code.split('\n')
print("Line numbers (1-indexed) and content:")
print("=" * 80)
for i, line in enumerate(lines, 1):
    marker = ""
    if "for (int j = i; j <= num - 1" in line:
        marker = " ← Loop 1 START"
    elif "for (int j = 0; j <= x" in line:
        marker = " ← Loop 2 START"
    elif "for (int j = 1; j <= x" in line:
        marker = " ← Loop 3 START"
    print(f"{i:2d} (0-idx: {i-1:2d}): {line}{marker}")

print("\n" + "=" * 80)
print("Expected highlights:")
print("  Loop 1: Lines 8-11 (1-indexed) = Lines 7-10 (0-indexed)")
print("  Loop 2: Lines 12-15 (1-indexed) = Lines 11-14 (0-indexed)")
print("  Loop 3: Lines 16-19 (1-indexed) = Lines 15-18 (0-indexed)")

print("\n" + "=" * 80)
print("Actual loop start lines:")
print(f"  Loop 1 starts at line: {[i for i, line in enumerate(lines, 1) if 'for (int j = i; j <= num - 1' in line][0]} (1-indexed)")
print(f"  Loop 2 starts at line: {[i for i, line in enumerate(lines, 1) if 'for (int j = 0; j <= x' in line][0]} (1-indexed)")
print(f"  Loop 3 starts at line: {[i for i, line in enumerate(lines, 1) if 'for (int j = 1; j <= x' in line][0]} (1-indexed)")
