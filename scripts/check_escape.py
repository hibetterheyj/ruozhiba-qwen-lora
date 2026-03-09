import json
from pathlib import Path

script_dir = Path(__file__).parent
data_dir = script_dir.parent / 'data' / 'tieba'

# Check all items to see if thought_process uses ASCII quotes
with open(data_dir / 'best365_2025_classified.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

ascii_quote = chr(34)
chinese_left = chr(8220)
chinese_right = chr(8221)

ascii_in_thought = 0
chinese_in_thought = 0

for item in data:
    tp = item.get('classification', {}).get('thought_process', '')
    if ascii_quote in tp:
        ascii_in_thought += 1
    if chinese_left in tp or chinese_right in tp:
        chinese_in_thought += 1

print(f'Total items: {len(data)}')
print(f'Items with ASCII quotes in thought_process: {ascii_in_thought}')
print(f'Items with Chinese quotes in thought_process: {chinese_in_thought}')

# Check a few examples
print('\n=== Sample thought_process values ===')
for i, item in enumerate(data[:5]):
    tp = item.get('classification', {}).get('thought_process', '')
    if ascii_quote in tp:
        print(f'\nItem {item.get("no")}:')
        print(tp[:100] + '...')
