import json
import re
import os
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

input_file = os.path.join(PROJECT_ROOT, 'crawler/threads/10354221105_弱智吧2025年度365佳贴/extracted.json')
output_file = os.path.join(PROJECT_ROOT, 'data/tieba/best365_2025.json')

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

pattern = re.compile(r'^(\d+)\.[\s\n]*(.+?)[\s\n]*评分：(\d+\.\d+)$', re.DOTALL)

extracted: List[Dict] = []
for item in data:
    text = item.get('text', '')
    match = pattern.match(text)
    if match:
        no = int(match.group(1))
        content = match.group(2).strip()
        score = float(match.group(3))
        extracted.append({
            'no': no,
            'text': content,
            'score': score
        })

print(f'匹配到的帖子数量: {len(extracted)}')

extracted.sort(key=lambda x: x['no'])

print(f'编号范围: {extracted[0]["no"]} 到 {extracted[-1]["no"]}')

missing = []
for i in range(1, 366):
    if not any(item['no'] == i for item in extracted):
        missing.append(i)
if missing:
    print(f'缺失的编号: {missing}')
else:
    print('编号完整，共365条')

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(extracted, f, ensure_ascii=False, indent=2)

print(f'已保存到: {output_file}')
