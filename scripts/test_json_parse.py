import json
import re

test_raw_response = r'''```json
{
  "thought_process": "这句话的核心笑点在于\"放在眼里\"这个词的双关。日常语境中，\"没把他放在眼里\"意思是不重视某人。但在这个语境中，患者错误使用痔疮膏——痔疮膏被误涂到了眼睛上（或者说本该用在肛门的药膏被用在了眼睛附近），医生说\"你根本就没把他放在眼里\"，字面意思变成了\"你没有把（痔疮膏）放到眼睛里\"，形成了一语双关的效果。这是一个典型的文字游戏，利用\"放在眼里\"的惯用义和字面义之间的错位制造笑点。同时，整个场景设定——医生批评患者错误使用痔疮膏——带有一种荒诞的画面感，属于古典弱智的荒谬情境构建。此外，涉及痔疮、误用药膏等略带身体不适/尴尬的话题，有一点点冒犯性的黑色幽默色彩，但程度较轻。",
  "top3_categories": [
    {
      "rank": 1,
      "category": "文字游戏",
      "confidence_score": 0.80,
      "reason": "核心笑点完全建立在\"放在眼里\"这一中文惯用语的双关之上——惯用义（不重视）与字面义（放进眼睛里）的错位是整句话的幽默引擎。"
    },
    {
      "rank": 2,
      "category": "古典弱智",
      "confidence_score": 0.12,
      "reason": "整个情境设定——医生因患者错误使用痔疮膏而说出一句日常成语——具有荒诞的场景构建感，属于\"精神病人思路广\"式的情境幽默。"
    },
    {
      "rank": 3,
      "category": "地狱笑话",
      "confidence_score": 0.08,
      "reason": "痔疮膏涂眼睛的画面隐含身体伤害和医疗事故的冒犯性，带有轻微的黑色幽默成分，但冒犯程度不高，不构成典型地狱笑话。"
    }
  ]
}
```'''

def fix_json_string(content: str) -> str:
    """Fix common JSON issues like Chinese quotes."""
    content = content.replace('"', '"').replace('"', '"')
    content = content.replace(''', "'").replace(''', "'")
    content = content.replace('，', ',')
    content = content.replace('：', ':')
    content = content.replace('【', '[').replace('】', ']')
    return content

def extract_json_from_response(content: str):
    """Try multiple methods to extract valid JSON from response."""
    if not content:
        return None

    content = content.strip()

    print("Step 1: Original content:")
    print(repr(content[:200]))
    print()

    if content.startswith("```"):
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    print("Step 2: After removing ```:")
    print(repr(content[:200]))
    print()

    content = fix_json_string(content)

    print("Step 3: After fixing JSON string:")
    print(repr(content[:200]))
    print()

    try:
        result = json.loads(content)
        print("✓ Success: Direct parse worked!")
        return result
    except json.JSONDecodeError as e:
        print(f"✗ Direct parse failed: {e}")
        pass

    json_patterns = [
        r'\{[\s\S]*"thought_process"[\s\S]*"top3_categories"[\s\S]*\}',
        r'\{[\s\S]*\}',
    ]

    for i, pattern in enumerate(json_patterns):
        print(f"\nTrying pattern {i}: {pattern}")
        matches = re.findall(pattern, content)
        print(f"Found {len(matches)} matches")
        for j, match in enumerate(matches):
            print(f"  Match {j}: {repr(match[:100])}")
            try:
                fixed_match = fix_json_string(match)
                result = json.loads(fixed_match)
                print(f"  ✓ Success!")
                return result
            except json.JSONDecodeError as e:
                print(f"  ✗ Failed: {e}")
                continue

    return None

result = extract_json_from_response(test_raw_response)
print("\n" + "=" * 60)
print("Final result:")
if result:
    print(json.dumps(result, ensure_ascii=False, indent=2))
else:
    print("None")
