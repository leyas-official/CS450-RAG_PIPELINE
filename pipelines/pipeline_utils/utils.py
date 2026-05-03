import os
import re
from tika import parser
from llama_index.core import Document


def load_documents_with_tika(directory: str):
    documents = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if not os.path.isfile(file_path):
            continue

        try:
            parsed = parser.from_file(file_path)
            text = parsed.get("content")

            if not text:
                continue

            text = re.sub(r"[ًٌٍَُِّْـ]", "", text).strip()

            documents.append(
                Document(
                    text=text,
                    metadata={"file_name": filename}
                )
            )

        except Exception as e:
            print(f"[TIKA ERROR] {filename}: {e}")

    return documents


def split_by_articles(text):
    def sanitize(text: str) -> str:
        text = text.replace("\x00", "")
        text = re.sub(r"[\u00a0\u00ad\u200b\u200c\u200d\u200e\u200f\ufeff]", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        return text.strip()

    _ARTICLE_LINE_RE = re.compile(
        r"^[ \t]*"
        r"(?:ال)?[مM]ادة"
        r"(?:\s+رقم)?"
        r"\s+"
        r"(\d+)"
        r"[ \t]*$",
        re.MULTILINE | re.UNICODE,
    )

    _BUND_LINE_RE = re.compile(
        r"^[ \t]*"
        r"البند"
        r"(?:\s+رقم)?"
        r"\s+"
        r"(\d+)"
        r"[ \t]*$",
        re.MULTILINE | re.UNICODE,
    )

    def _find_headers(text: str):
        headers = []
        for pattern, prefix in ((_ARTICLE_LINE_RE, "المادة"), (_BUND_LINE_RE, "البند")):
            for m in pattern.finditer(text):
                headers.append({
                    "start": m.start(),
                    "end": m.end(),
                    "number": int(m.group(1)),
                    "label": f"{prefix} {m.group(1)}",
                })
        headers.sort(key=lambda h: h["start"])
        return headers

    _NOISE_LINE_RE = re.compile(
        r"^("
        r"www\.\S+|"
        r"https?://\S+|"
        r"LAWSOCIETY\.LY.*|"
        r".*lawsociety\.ly.*|"
        r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s+[APap][Mm].*|"
        r"صفحة\s+\d+|"
        r"[a-zA-Z0-9\-]+صفحة\s+\d+|"
        r"REG-GPC-\S+|"
        r"المحتويات.*"
        r")$",
        re.MULTILINE | re.UNICODE,
    )

    def _remove_noise_lines(text: str) -> str:
        cleaned = _NOISE_LINE_RE.sub("", text)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned

    def parse_legal_document(raw_text: str):
        text = sanitize(raw_text)
        text = _remove_noise_lines(text)
        headers = _find_headers(text)

        if not headers:
            return []

        segments = []
        for i, hdr in enumerate(headers):
            content_start = hdr["end"]
            content_end = headers[i + 1]["start"] if i + 1 < len(headers) else len(text)
            body = text[content_start:content_end].strip()
            entry = f"{hdr['label']} {body}" if body else hdr["label"]
            entry = re.sub(r" {2,}", " ", entry)
            entry = re.sub(r"\n{3,}", "\n\n", entry)
            segments.append(entry.strip())

        return segments

    return parse_legal_document(text)


def fixed_width_chunking(articles, chunk_size=3, overlap=1):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(articles), step):
        chunk = articles[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)

    return chunks


def boost_nodes_with_articles(nodes, query):
    from pipelines.pipeline_core import ArabicArticleExtractor
    query_articles = ArabicArticleExtractor.extract(query)

    for node in nodes:
        node.score = node.score or 0.0
        if query_articles:
            chunk_articles = node.metadata.get("Article", [])
            if any(a in chunk_articles for a in query_articles):
                node.score += 1.0

    return nodes