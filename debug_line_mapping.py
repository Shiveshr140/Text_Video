sample_code = """

function findLargest(arr) {
    let largest = arr[0]; 
    for (let i = 1; i < arr.length; i++) {
        if (arr[i] > largest) {
            largest = arr[i]; 
        }
    }
    return largest;
}

console.log(findLargest([99, 5, 3, 100, 1]));

"""

print("ORIGINAL CODE:")
print("="*60)
original_lines = sample_code.split('\n')
for i, line in enumerate(original_lines, 1):
    print(f"Line {i:2d}: {repr(line)}")

print(f"\n{'='*60}")
print("CLEANING CODE (removing blank lines)")
print("="*60)

non_blank_lines = []
line_number_mapping = {}
new_line_num = 1

for orig_line_num, line in enumerate(original_lines, start=1):
    if line.strip():  # Non-blank line
        non_blank_lines.append(line)
        line_number_mapping[orig_line_num] = new_line_num
        print(f"Original line {orig_line_num:2d} → Cleaned line {new_line_num:2d}: {repr(line[:50])}")
        new_line_num += 1
    else:
        print(f"Original line {orig_line_num:2d} → SKIPPED (blank)")

cleaned_code = '\n'.join(non_blank_lines)

print(f"\n{'='*60}")
print("CLEANED CODE:")
print("="*60)
for i, line in enumerate(cleaned_code.split('\n'), 1):
    print(f"Line {i:2d}: {repr(line)}")

print(f"\n{'='*60}")
print("BLOCK DETECTION (on original code)")
print("="*60)
print("Function: lines 1-9 (in original)")
print("For loop: lines 3-7 (in original)")

print(f"\n{'='*60}")
print("MAPPING BLOCKS TO CLEANED CODE")
print("="*60)

# For loop is lines 3-7 in original
for_loop_original_start = 3
for_loop_original_end = 7

print(f"For loop in ORIGINAL: lines {for_loop_original_start}-{for_loop_original_end}")

# Map to cleaned code
for_loop_cleaned_start = None
for_loop_cleaned_end = None

for orig_line in range(for_loop_original_start, for_loop_original_end + 1):
    if orig_line in line_number_mapping:
        if for_loop_cleaned_start is None:
            for_loop_cleaned_start = line_number_mapping[orig_line]
        for_loop_cleaned_end = line_number_mapping[orig_line]
        print(f"  Original line {orig_line} → Cleaned line {line_number_mapping[orig_line]}")

print(f"\nFor loop in CLEANED: lines {for_loop_cleaned_start}-{for_loop_cleaned_end}")
print(f"For loop in CLEANED (0-indexed): lines {for_loop_cleaned_start-1}-{for_loop_cleaned_end-1}")
