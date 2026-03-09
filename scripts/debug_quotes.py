import json
from pathlib import Path
from typing import Any

script_dir = Path(__file__).parent
data_dir = script_dir.parent / 'data' / 'tieba'

# Test the conversion function
test_text = '这句话的核心笑点在于"放在眼里"这个词的双关。'
print('=== Test conversion ===')
print(f'Input: {repr(test_text)}')
print(f'ASCII quote char: {repr(chr(34))}')
print(f'Chinese left quote char: {repr(chr(8220))}')
print(f'Chinese right quote char: {repr(chr(8221))}')

# Check if the text contains ASCII quotes
print(f'\nContains ASCII quote: {chr(34) in test_text}')
print(f'ASCII quote positions: {[i for i, c in enumerate(test_text) if c == chr(34)]}')

# Manual conversion
def convert_ascii_quotes_to_chinese(text: str) -> str:
    if not text or '"' not in text:
        return text
    
    result = []
    in_quote = False
    
    for char in text:
        if char == '"':
            if not in_quote:
                result.append('"')
                in_quote = True
            else:
                result.append('"')
                in_quote = False
        else:
            result.append(char)
    
    return ''.join(result)

converted = convert_ascii_quotes_to_chinese(test_text)
print(f'\nConverted: {repr(converted)}')
print(f'Converted contains ASCII quote: {chr(34) in converted}')
print(f'Converted contains Chinese left quote: {chr(8220) in converted}')
