"""Adhoc debug script for validating ASCII-to-Chinese quote conversion."""

import logging
from typing import Any

from scripts.data.fix_quotes import convert_ascii_quotes_to_chinese


LOGGER = logging.getLogger(__name__)


def process_value(value: Any) -> Any:
    """Recursively apply quote conversion for quick manual inspection."""
    if isinstance(value, str):
        return convert_ascii_quotes_to_chinese(value)
    if isinstance(value, dict):
        return {key: process_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [process_value(item) for item in value]
    return value


def main() -> None:
    """Run a few representative quote-conversion cases."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    test_cases = [
        '这句话的核心笑点在于"放在眼里"这个词的双关。',
        '简单的测试"引号"转换。',
        '没有引号的文本',
        '多个"引号"测试"另一个"引号',
    ]

    LOGGER.info("=== Quote conversion test cases ===")
    for case in test_cases:
        result = convert_ascii_quotes_to_chinese(case)
        LOGGER.info("Input:  %r", case)
        LOGGER.info("Output: %r", result)


if __name__ == '__main__':
    main()
