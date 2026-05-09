import re
from typing import List


class ArabicArticleExtractor:
    """Extracts article numbers from a user query string."""

    _AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

    _PATTERNS: List[re.Pattern] = [
        re.compile(
            r"المواد\s+((?:[\d٠-٩]+\s*(?:و|،|,)\s*)+[\d٠-٩]+)",
            re.UNICODE,
        ),
        re.compile(
            r"(?:ال)?ماد(?:ة|تين|ات)\s+(?:رقم\s+)?[\d٠-٩]+"
            r"(?:\s*(?:و|،|,)\s*(?:(?:ال)?ماد(?:ة|تين|ات)\s+)?(?:رقم\s+)?[\d٠-٩]+)*",
            re.UNICODE,
        ),
    ]

    @classmethod
    def _normalize(cls, text: str) -> str:
        return text.translate(cls._AR_DIGITS)

    @classmethod
    def extract(cls, query: str) -> List[int]:
        query = cls._normalize(query)
        found: set[int] = set()
        for pattern in cls._PATTERNS:
            for match in pattern.finditer(query):
                nums = re.findall(r"\d+", match.group(0))
                found.update(int(n) for n in nums)
        return sorted(found)


class ChunkArticleParser:
    """Extracts article numbers from a chunk of text (used at index-build time)."""

    _AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

    _HEADER_PATTERN = re.compile(
        r"(?:^|\n)\s*(?:الماد[ةة]|مادة)\s*(?:رقم\s*)?([\d٠-٩]+)",
        re.UNICODE | re.MULTILINE,
    )

    @classmethod
    def parse(cls, text: str) -> List[int]:
        text = text.translate(cls._AR_DIGITS)
        nums = [int(m.group(1)) for m in cls._HEADER_PATTERN.finditer(text)]
        return sorted(set(nums))


class TextAnalyzer:
    """Classifies a query as 'legal' or 'general'."""

    _LEGAL_KEYWORDS = re.compile(
        r"(?:الماد[ةة]|مادة|المواد|قانون|تشريع|لائحة|نظام|بند|فقرة|الفصل|الباب|القسم"
        r"|أحكام|عقوبة|جريمة|جزاء|اتفاقية|مرسوم|قرار\s+وزاري)",
        re.UNICODE,
    )

    @classmethod
    def classify(cls, query: str) -> str:
        if ArabicArticleExtractor.extract(query):
            return "legal"
        if cls._LEGAL_KEYWORDS.search(query):
            return "legal"
        return "general"