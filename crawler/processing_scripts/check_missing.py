import json
import re
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

input_file = os.path.join(PROJECT_ROOT, 'crawler/threads/10354221105_弱智吧2025年度365佳贴/10354221105_dump.jsonl')

pattern = re.compile(r'^(\d+)\.')

found_nums = set()

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            data = json.loads(line)
            text = data.get('text', '')
            match = pattern.match(text)
            if match:
                num = int(match.group(1))
                found_nums.add(num)

print(f'找到的编号数量: {len(found_nums)}')
print(f'编号范围: {min(found_nums)} 到 {max(found_nums)}')

missing = []
for i in range(1, 366):
    if i not in found_nums:
        missing.append(i)

print(f'缺失的编号: {missing}')
