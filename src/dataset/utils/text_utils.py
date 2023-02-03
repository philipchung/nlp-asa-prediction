from __future__ import annotations

import re
from collections import namedtuple
from typing import Union

Match = namedtuple("Match", ["text", "start", "end"])


def find_pattern(
    text: str, pattern: str, flags: re.RegexFlag = 0
) -> Union[None, list[Match]]:
    "Returns matched strings as Match namedtuples if mentioned in input text."
    patt = re.compile(pattern, flags=flags)
    if text and (type(text) == str):
        result = re.finditer(patt, text)
        return [Match(x.group(), x.start(), x.end()) for x in result]
    else:
        return None


def extract_field(text: str, pre: str, post: str) -> list[str]:
    """Extracts text value that follows given `field_text`.

    Args:
        text: string text from which to extract field value
        pre: regex pattern preceeding value to extract
        post: regex pattern following value to extract

    Returns:
        List of all substring text matched by regex pattern.
        If no matches, then returns `None`.
    """
    if text and (type(text) == str):
        pattern = rf"(?<={pre})(.*)(?={post})"
        patt = re.compile(pattern)
        result = re.finditer(patt, text)
        return [Match(x.group(), x.start(), x.end()) for x in result]
    else:
        return None
