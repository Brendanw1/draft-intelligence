import re

def parse_height(height_str):
    if not height_str or height_str == "":
        return None
    try:
        s = str(height_str).strip()
        # Try "6-7" format (roster)
        if "-" in s:
            parts = s.split("-")
            if len(parts) == 2:
                feet, inches = int(parts[0]), int(parts[1])
                return feet * 12 + inches
        # Try "6' 7"" or "6' 7'" format (draft records)
        m = re.match(r"(\d+)\s*['\u2019\u2018]\s*(\d*)\s*[\"\u201d\u201c]?", s)
        if m:
            feet = int(m.group(1))
            inches = int(m.group(2)) if m.group(2) else 0
            return feet * 12 + inches
        # Try bare number (already inches)
        return int(float(s))
    except (ValueError, IndexError):
        pass
    return None

# Test cases
tests = [
    ("6-2", "roster format"),
    ("6' 2\"", "draft format"),
    ("6' 2'", "draft with single quote"),
    ("6'2\"", "draft no space"),
    ("6-7", "roster tall"),
    ("-", "dash only"),
    ("", "empty"),
    (None, "None"),
]
for t, desc in tests:
    result = parse_height(t)
    print(f"  '{t}' ({desc}) -> {result}")
