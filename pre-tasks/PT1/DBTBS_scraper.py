#!/usr/bin/env python3
"""
dbtbs_scraper.py - extract real promoter sequences from DBTBS (Bacillus subtilis).

Crawls the DBTBS ver3 promoter classification table, follows every per-gene
promoter page, and parses out the promoter / cis-element sequences, preserving
the transcription start site (shown in DBTBS as bold) and the exact binding
sequence (shown in red).

Output: a TSV with one row per sequence found, plus an optional FASTA of the
positive (promoter) sequences for downstream ProkBERT fine-tuning.

Dependencies:  pip install requests beautifulsoup4

Examples
--------
  # inspect one page's raw markup to confirm the bold/red tags:
  python dbtbs_scraper.py --debug dltA

  # small test crawl (first 20 promoter pages):
  python dbtbs_scraper.py --limit 20 --out test.tsv --fasta test.fasta

  # full crawl:
  python dbtbs_scraper.py --out dbtbs_promoters.tsv --fasta dbtbs_promoters.fasta
"""

import argparse
import csv
import hashlib
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

BASE = "https://dbtbs.hgc.jp"
INDEX = BASE + "/ver3/promtable.html"
PROM_HREF = "/ver3/COG/prom/"

# Tokens that identify the "red" highlight DBTBS uses for the exact binding seq.
RED_TOKENS = ("red", "#ff0000", "#f00", "#e00", "#cc0000")
META_KEYS = {
    "direction",
    "start position",
    "operon",
    "transcription start site",
    "cog id",
    "profile",
}
NUC = set("ACGTNRYSWKMBDHVacgtnryswkmbdhv")  # IUPAC, kept permissive


# --------------------------------------------------------------------------- #
# Networking (with on-disk cache + retries)
# --------------------------------------------------------------------------- #
def get_html(url, session, cache_dir, delay, retries=3):
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(url.encode()).hexdigest() + ".html"
    path = cache_dir / key
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")

    last = None
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=30)
            resp.encoding = "iso-8859-1"  # DBTBS pages are Latin-1
            if resp.status_code == 200:
                path.write_text(resp.text, encoding="utf-8", errors="replace")
                time.sleep(delay)
                return resp.text
            last = f"HTTP {resp.status_code}"
        except requests.RequestException as exc:
            last = str(exc)
        time.sleep(delay * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


# --------------------------------------------------------------------------- #
# Index parsing
# --------------------------------------------------------------------------- #
def find_prom_pages(index_html, base_url=INDEX):
    """Collect per-gene promoter page URLs, resolving relative hrefs first."""
    soup = BeautifulSoup(index_html, "html.parser")
    urls = set()
    anchors = soup.find_all("a", href=True)
    for a in anchors:
        href = a["href"].strip()
        if not href.lower().endswith(".html"):
            continue
        full = urljoin(base_url, href)   # resolves "COG/prom/x.html" -> absolute
        if PROM_HREF in full:            # keep only .../ver3/COG/prom/*.html
            urls.add(full)
    # Fallback: scan raw HTML if the parser somehow missed the anchors.
    if not urls:
        for m in re.finditer(r'href=["\']?([^"\'> ]+\.html)', index_html, re.I):
            full = urljoin(base_url, m.group(1))
            if PROM_HREF in full:
                urls.add(full)
    if not urls:
        sample = [a["href"] for a in anchors[:8]]
        print(
            f"[diag] {len(anchors)} anchors found but 0 promoter links matched; "
            f"sample hrefs: {sample}",
            file=sys.stderr,
        )
    return sorted(urls)


def gene_from_url(url):
    return url.rsplit("/", 1)[-1].rsplit(".", 1)[0]


# --------------------------------------------------------------------------- #
# Per-page parsing
# --------------------------------------------------------------------------- #
def parse_meta(soup):
    """Pull the key/value metadata table (Direction, Operon, COG, ...)."""
    meta = {}
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            if len(cells) == 2:
                key = cells[0].get_text(" ", strip=True).lower()
                if key in META_KEYS:
                    meta[key] = cells[1].get_text(" ", strip=True)
    return meta


def _flags_for(node, cell):
    """Walk ancestors up to `cell` and decide if this text node is bold/red."""
    bold = red = False
    p = node.parent
    while p is not None:
        if isinstance(p, Tag):
            name = p.name.lower()
            style = (p.get("style") or "").lower()
            if name in ("b", "strong"):
                bold = True
            if "font-weight" in style and (
                "bold" in style or any(w in style for w in ("600", "700", "800", "900"))
            ):
                bold = True
            color = (p.get("color") or "").lower()
            if name == "font" and any(t in color for t in RED_TOKENS):
                red = True
            if "color:" in style and any(t in style for t in RED_TOKENS):
                red = True
        if p is cell:
            break
        p = p.parent
    return bold, red


def annotate_cell(cell):
    """Return list of (char, is_bold, is_red) for the nucleotides in a cell."""
    out = []
    for s in cell.descendants:
        if isinstance(s, NavigableString):
            bold, red = _flags_for(s, cell)
            for ch in str(s):
                out.append((ch, bold, red))
    return out


def extract_seq(chars):
    """Collapse annotated chars into a clean sequence + TSS index + red spans."""
    seq, tss, spans = [], -1, []
    in_red, start = False, 0
    for ch, bold, red in chars:
        if ch in NUC:
            i = len(seq)
            seq.append(ch.upper())
            if bold and tss < 0:
                tss = i
            if red and not in_red:
                in_red, start = True, i
            elif not red and in_red:
                in_red = False
                spans.append((start, i))
    if in_red:
        spans.append((start, len(seq)))
    return "".join(seq), tss, spans


def looks_like_seq(text):
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 15:
        return False
    nuc = [c for c in letters if c in NUC]
    return len(nuc) / len(letters) > 0.85


def parse_page(gene, url, html):
    soup = BeautifulSoup(html, "html.parser")
    meta = parse_meta(soup)
    rows = []
    for ti, table in enumerate(soup.find_all("table")):
        for ri, tr in enumerate(table.find_all("tr")):
            cells = tr.find_all(["td", "th"])
            for ci, cell in enumerate(cells):
                ctext = cell.get_text(" ", strip=True)
                if not looks_like_seq(ctext):
                    continue
                seq, tss, spans = extract_seq(annotate_cell(cell))
                if not seq:
                    continue
                binding = ";".join(seq[a:b] for a, b in spans)
                context = " | ".join(
                    cells[cj].get_text(" ", strip=True)
                    for cj in range(len(cells))
                    if cj != ci
                )
                rows.append(
                    {
                        "gene": gene,
                        "direction": meta.get("direction", ""),
                        "genomic_span": meta.get("start position", ""),
                        "operon": meta.get("operon", ""),
                        "tss_field": meta.get("transcription start site", ""),
                        "cog_id": meta.get("cog id", ""),
                        "profile": meta.get("profile", ""),
                        "table_idx": ti,
                        "row_idx": ri,
                        "row_context": context,
                        "sequence": seq,
                        "seq_len": len(seq),
                        "tss_index": tss,
                        "binding_subseq": binding,
                        "source_url": url,
                    }
                )
    return rows


# --------------------------------------------------------------------------- #
# Debug helper
# --------------------------------------------------------------------------- #
def debug_page(gene, session, cache_dir, delay):
    url = f"{BASE}{PROM_HREF}{gene}.html"
    html = get_html(url, session, cache_dir, delay)
    soup = BeautifulSoup(html, "html.parser")
    print(f"# raw tables for {gene}  ({url})\n")
    for ti, table in enumerate(soup.find_all("table")):
        print(f"----- table {ti} " + "-" * 50)
        print(table.prettify()[:4000])
        print()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
FIELDS = [
    "gene", "direction", "genomic_span", "operon", "tss_field", "cog_id",
    "profile", "table_idx", "row_idx", "row_context", "sequence", "seq_len",
    "tss_index", "binding_subseq", "source_url",
]


def main():
    ap = argparse.ArgumentParser(description="Scrape promoter sequences from DBTBS.")
    ap.add_argument("--out", default="dbtbs_promoters.tsv", help="output TSV path")
    ap.add_argument("--fasta", default=None, help="optional FASTA of sequences")
    ap.add_argument("--cache-dir", default=".dbtbs_cache", help="page cache dir")
    ap.add_argument("--delay", type=float, default=0.7, help="seconds between fetches")
    ap.add_argument("--limit", type=int, default=0, help="max pages (0 = all)")
    ap.add_argument("--debug", metavar="GENE", help="dump raw tables for one gene and exit")
    args = ap.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = (
        "dbtbs-promoter-scraper/1.0 (research use; contact: you@example.org)"
    )
    cache_dir = Path(args.cache_dir)

    if args.debug:
        debug_page(args.debug, session, cache_dir, args.delay)
        return

    print(f"[*] fetching index: {INDEX}", file=sys.stderr)
    pages = find_prom_pages(get_html(INDEX, session, cache_dir, args.delay))
    if args.limit:
        pages = pages[: args.limit]
    print(f"[*] {len(pages)} promoter pages to crawl", file=sys.stderr)

    all_rows, failures = [], []
    for n, url in enumerate(pages, 1):
        gene = gene_from_url(url)
        try:
            html = get_html(url, session, cache_dir, args.delay)
            rows = parse_page(gene, url, html)
            all_rows.extend(rows)
            print(f"[{n}/{len(pages)}] {gene}: {len(rows)} seq(s)", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - keep crawling on any page error
            failures.append((url, str(exc)))
            print(f"[{n}/{len(pages)}] {gene}: FAILED ({exc})", file=sys.stderr)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_rows)

    if args.fasta:
        seen = set()
        with open(args.fasta, "w", encoding="utf-8") as fh:
            for r in all_rows:
                seq = r["sequence"]
                if seq in seen:
                    continue
                seen.add(seq)
                header = f"{r['gene']}|tss{r['tss_index']}|{gene_from_url(r['source_url'])}"
                fh.write(f">{header}\n{seq}\n")

    uniq = len({r["sequence"] for r in all_rows})
    print(
        f"\n[done] {len(all_rows)} rows  ({uniq} unique sequences)  "
        f"from {len(pages)} pages -> {args.out}",
        file=sys.stderr,
    )
    if args.fasta:
        print(f"[done] FASTA -> {args.fasta}", file=sys.stderr)
    if failures:
        print(f"[warn] {len(failures)} page(s) failed:", file=sys.stderr)
        for url, err in failures:
            print(f"        {url}  ({err})", file=sys.stderr)


if __name__ == "__main__":
    main()