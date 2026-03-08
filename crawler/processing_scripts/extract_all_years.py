import json
import re
import os
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

def extract_posts(thread_id: str, year: str) -> None:
    input_file = os.path.join(PROJECT_ROOT, f'crawler/threads/{thread_id}_弱智吧{year}年度365佳贴/{thread_id}_dump.jsonl')
    output_file = os.path.join(PROJECT_ROOT, f'data/tieba/best365_{year}.json')
    
    pattern = re.compile(r'^(\d+)\.[\s\n]*(.+?)[\s\n]*评分：(\d+\.\d+)$', re.DOTALL)
    
    extracted: List[Dict] = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                text = data.get('text', '')
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
    
    print(f'{year}年匹配到的帖子数量: {len(extracted)}')
    
    if extracted:
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
    else:
        print(f'{year}年没有匹配到任何帖子')

if __name__ == '__main__':
    extract_posts('9354404050', '2024')
    print()
    extract_posts('8807455743', '2023')
