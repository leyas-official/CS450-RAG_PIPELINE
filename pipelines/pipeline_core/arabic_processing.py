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
            r"(?:ال)?ماد(?:ة|تين|ات)\s*(?:رقم\s*)?[\(\[]?\s*[\d٠-٩]+\s*[\)\]]?"
            r"(?:\s*(?:و|،|,)\s*(?:(?:ال)?ماد(?:ة|تين|ات)\s*)?(?:رقم\s*)?[\(\[]?\s*[\d٠-٩]+\s*[\)\]]?)*",
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


import re
from typing import List

class ChunkArticleParser:
    """Extracts article numbers from a chunk of text (used at index-build time)."""

    _AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

    _HEADER_PATTERN = re.compile(
        r"^[ \t]*(?:[-–•]\s*)?"
        r"[(\[]?\s*([\d٠-٩]+)\s*[)\]]?"
        r"\s*(?:رقم\s*)?" + r"(?:ال)?ماد(?:ة|ه|تين|ات)"
        r"(\s*مكرر[ةا]?(?:\s*\([\d٠-٩]+\))?)?",
        re.UNICODE | re.MULTILINE,
    )

    @classmethod
    def parse(cls, text: str) -> List[int]:
        """Unique article numbers found in the chunk."""
        text = text.translate(cls._AR_DIGITS)
        nums = [int(m.group(1)) for m in cls._HEADER_PATTERN.finditer(text)]
        return sorted(set(nums))

    @classmethod
    def parse_detailed(cls, text: str) -> List[str]:
        text = text.translate(cls._AR_DIGITS)
        results = []
        for m in cls._HEADER_PATTERN.finditer(text):
            num = m.group(1)
            suffix = (m.group(2) or "").strip()
            key = f"{num} {suffix}".strip() if suffix else num
            results.append(key)
        seen, out = set(), []
        for r in results:
            if r not in seen:
                seen.add(r)
                out.append(r)
        return out


class TextAnalyzer:
    """Classifies a query as 'legal' or 'general'."""

    # structural section words are only unambiguously legal when followed
    # directly by an ordinal — their bare form is a common, highly ambiguous
    # everyday word (الباب=door, الفصل=season/school term, القسم=department/oath)
    _ORDINAL = (
        r"(?:الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر"
        r"|[\d٠-٩]+)"
    )

    _LEGAL_KEYWORDS = re.compile(
        r"\b(?:"
        r"(?:ال)?ماد[ةه]|(?:ال)?مواد"
        r"|(?:ال)?قانون|(?:ال)?تشريع|(?:ال)?لائحة|(?:ال)?نظام"
        r"|(?:ال)?بند|(?:ال)?فقرة"
        r"|(?:الباب|الفصل|القسم)\s+" + _ORDINAL +
        r"|(?:ال)?أحكام|(?:ال)?عقوبة|(?:ال)?جريمة|(?:ال)?جزاء"
        r"|(?:ال)?اتفاقية|(?:ال)?مرسوم"
        r"|(?:ال)?قرار\s+(?:وزاري|إداري|جمهوري|أميري|تنظيمي)"
        r")\b",
        re.UNICODE,
    )

    # _DISCIPLINARY_KEYWORDS = [...]  # unused/dead code — let me know if you
    # want this wired in as a sub-category, or should I just remove it?

    @classmethod
    def classify(cls, query: str) -> str:
        if ArabicArticleExtractor.extract(query):
            return "legal"
        if cls._LEGAL_KEYWORDS.search(query):
            return "legal"
        return "general"

    # @classmethod
    # def is_disciplinary(cls, article_text: str) -> bool:
    #     return sum(kw in article_text for kw in cls._DISCIPLINARY_KEYWORDS) >= 2

class ActorExtractor:
    """Extracts legal actors/parties mentioned in a user's question.

    Uses an LLM with a few-shot prompt to do the semantic extraction, then
    applies a deterministic blacklist filter as a safety net -- small models
    (e.g. Gemma 9B) don't reliably follow "don't do X" instructions 100% of
    the time, especially when X is the grammatical subject of the sentence
    (e.g. "يفرضها القانون"), so the prompt alone isn't enough.
    """

    _prompt = """أنت محلل نصوص قانونية. مهمتك تحديد الأطراف/الفاعلين المذكورين في سؤال المستخدم.

الفاعل هو شخص أو جهة أو صفة قانونية تكون طرفًا في العلاقة القانونية أو محل حق أو التزام،
مثل: الطالب، المرخص له، الجهة المختصة، صاحب المعاش.

لا تعتبر الكلمات التالية فاعلين أبدًا، حتى لو وردت في الجملة كفاعل نحوي (مثال: "يفرضها القانون"):
القانون، النظام، اللائحة، التشريع، الدستور، القرار، المادة، النص، البند، الفقرة.
فهذه مصادر الحكم القانوني وليست أطرافًا فعليين.

أمثلة:
السؤال: ما هي القيود التي يفرضها القانون على الجمع بين أكثر من معاش أو بين معاش ومرتب آخر؟
الفاعلون:
لا يوجد

السؤال: ما هي شروط ترخيص المنشآت الصناعية من الجهة المختصة؟
الفاعلون:
الجهة المختصة

أعد أسماء الفاعلين فقط، كل فاعل في سطر مستقل، بدون ترقيم أو شرح.
إذا لم يوجد فاعل واضح في السؤال، أعد كلمة: لا يوجد

السؤال:
{text}"""

    # deterministic safety net: legal-source terms that must never be
    # treated as actors, regardless of what the model returns
    _NON_ACTOR_TERMS = {
        "القانون", "قانون",
        "النظام", "نظام",
        "اللائحة", "لائحة",
        "التشريع", "تشريع",
        "الدستور", "دستور",
        "القرار", "قرار",
        "النص القانوني", "النص", "نص",
        "المادة", "مادة",
        "البند", "بند",
        "الفقرة", "فقرة",
    }

    _STRIP_CHARS = "،.:؛!؟ \t"

    @classmethod
    def _clean(cls, term: str) -> str:
        return term.strip().strip(cls._STRIP_CHARS)

    @classmethod
    def _is_actor(cls, term: str) -> bool:
        cleaned = cls._clean(term)
        if not cleaned:
            return False
        return cleaned not in cls._NON_ACTOR_TERMS

    @classmethod
    def _is_present_in_source(cls, term: str, source_text: str) -> bool:
        """Extra guard: an actor should actually appear in the question,
        not be invented/hallucinated by the model."""
        return cls._clean(term) in source_text

    @classmethod
    def extract(cls, text: str, llm, verify_in_source: bool = True) -> List[str]:
        try:
            response = llm.complete(
                cls._prompt.format(text=text),
                temperature=0,  # deterministic output, less drift on small models
            )
            raw = response.text.strip()

            if "لا يوجد" in raw:
                return []

            candidates = [cls._clean(line) for line in raw.splitlines() if line.strip()]
            candidates = [c for c in candidates if c]

            # filter 1: blacklist of legal-source terms
            candidates = [c for c in candidates if cls._is_actor(c)]

            # filter 2 (optional): must actually appear in the original question
            if verify_in_source:
                candidates = [c for c in candidates if cls._is_present_in_source(c, text)]

            # dedupe while preserving order
            seen = set()
            result = []
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    result.append(c)

            return result
        except Exception as e:
            print(f"[ActorExtractor ERROR] {e}")
            return []