import os
import glob
import re
import urllib.parse
import urllib.request


class SearchTimeoutError(Exception):
    pass


STOP_WORDS = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "did", "do", "does", "doing", "don", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off",
    "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "write", "function",
}

IGNORED_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "node_modules", ".pytest_cache",
    ".mypy_cache", "data", "dist", "build", ".idea", ".vscode"
}

IGNORED_EXTENSIONS = {
    ".pyc", ".pyo", ".sqlite", ".sqlite3", ".db", ".bin", ".exe", ".so",
    ".dylib", ".dll", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".pdf", ".woff", ".woff2",
    ".ttf", ".eot"
}


def local_or_web_search(query: str, timeout: float = 15.0):
    """Minimal, offline-first search tool.

    1. If the query references a local file/dir (or one that exists), search the local
       filesystem and return matching paths + snippets.
    2. Otherwise return the intended web query so a real search backend can be swapped in
       later (see README). Never silently phones home — fully offline by default.

    Returns a string. Swap this for a real engine (DuckDuckGo/tavily/local doc index)
    without touching agent code.
    """
    query = query.strip()

    local_hits = _search_local(query)
    if local_hits:
        return _format_hits(local_hits)

    # Offline placeholder: surface the query plus why we can't search the web.
    return (
        f"[offline search] No local match found for query: {query}\n"
        f"Note: web search is disabled in this offline build to guarantee no external "
        f"network calls. To enable, swap `local_or_web_search` in tools/search_tool.py "
        f"for a real backend.\n"
    )


def _search_local(query: str, top_n: int = 5):
    """Search under the CWD for text files matching query terms."""
    all_tokens = [t for t in re.split(r"[\s,_\-./\\]+", query.lower()) if t]
    # Filter out stop words and short tokens to avoid matching everything
    terms = [t for t in all_tokens if len(t) >= 3 and t not in STOP_WORDS]
    if not terms:
        terms = [t for t in all_tokens if len(t) >= 3]
    if not terms:
        return []

    root = os.getcwd()
    hits = []  # (path, snippet)

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored directories in-place
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS and not d.startswith(".")]

        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in IGNORED_EXTENSIONS or fname.startswith("."):
                continue

            path = os.path.join(dirpath, fname)
            if not os.access(path, os.R_OK):
                continue

            lower_name = fname.lower()
            name_match = any(t in lower_name for t in terms)

            # Check content if filename doesn't match or to get a snippet
            snippet = _read_snippet_matching(path, terms)
            if snippet is None:
                # File is binary or unreadable
                continue

            if name_match or snippet:
                display_snippet = snippet or _read_head(path, max_chars=300)
                hits.append((path, display_snippet))
                if len(hits) >= top_n:
                    return hits

    return hits


def _read_head(path: str, max_chars: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            chunk = f.read(max_chars)
            if "\x00" in chunk:
                return ""
            return chunk.strip()
    except Exception:
        return ""


def _read_snippet_matching(path: str, terms: list[str], max_chars: int = 300) -> str | None:
    """Reads a text file and looks for matches of terms. Returns snippet or None if binary."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read(8192)
            if "\x00" in text:
                return None  # binary file
            lower_text = text.lower()
            for term in terms:
                idx = lower_text.find(term)
                if idx != -1:
                    start = max(0, idx - 60)
                    end = min(len(text), idx + max_chars)
                    snippet = text[start:end].replace("\n", " ").strip()
                    return f"...{snippet}..."
            return ""
    except Exception:
        return None


def _format_hits(hits) -> str:
    lines = [f"Found {len(hits)} local file(s):"]
    for path, snippet in hits:
        lines.append(f"--- {path}")
        if snippet:
            lines.append(snippet)
    return "\n".join(lines)


def _web_search_placeholder(query: str, timeout: float):
    """Reference implementation for a real backend — not enabled by default."""
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        raise SearchTimeoutError(f"web search failed: {e}") from e
