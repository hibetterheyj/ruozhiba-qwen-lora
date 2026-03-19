import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
file_path = _REPO / "data" / "tieba" / "best365_2025_classified.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)


def fix_unescaped_quotes(s: str) -> str:
    """Fix unescaped quotes inside a JSON string."""
    result = []
    in_string = False
    escape_next = False

    for i, char in enumerate(s):
        if escape_next:
            result.append(char)
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            result.append(char)
            continue

        if char == '"':
            if not in_string:
                in_string = True
                result.append(char)
            else:
                if i + 1 < len(s) and s[i + 1] in [",", "}", "]", ":", " ", "\n", "\t"]:
                    in_string = False
                    result.append(char)
                else:
                    result.append("\\\"")
        else:
            result.append(char)

    return "".join(result)


def extract_json_from_response(content: str):
    """Try multiple methods to extract valid JSON from response."""
    if not content:
        return None

    content = content.strip()

    if content.startswith("```"):
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    try:
        fixed = fix_unescaped_quotes(content)
        print("After fix_unescaped_quotes:")
        print(repr(fixed[:300]))
        return json.loads(fixed)
    except json.JSONDecodeError as e:
        print(f"Still failed: {e}")
        pass

    return None


for item in data:
    if item.get("no") == 57:
        print("Found item no. 57:")
        classification = item.get("classification", {})
        raw_response = classification.get("raw_response")
        
        if raw_response:
            print("\n" + "=" * 60)
            print("Attempting to parse raw_response:")
            result = extract_json_from_response(raw_response)
            
            if result:
                print("\n✓ SUCCESS!")
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print("\n✗ FAILED")
        break
