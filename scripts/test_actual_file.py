import json
from pathlib import Path

file_path = Path("/Users/heyujie/Documents/cuhksz-all-sync/course_materials/CSS 5120 - Computational Linguistics/Lab3_SFT/data/tieba/best365_2025_classified.json")

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Find item no. 57
for item in data:
    if item.get("no") == 57:
        print("Found item no. 57:")
        print(json.dumps(item, ensure_ascii=False, indent=2))
        
        classification = item.get("classification", {})
        raw_response = classification.get("raw_response")
        
        if raw_response:
            print("\n" + "=" * 60)
            print("RAW RESPONSE:")
            print(repr(raw_response))
        break
