from __future__ import annotations

import argparse
import base64
import html
import unicodedata
import mimetypes
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse


class MarkdownConversionError(RuntimeError):
    """Markdown→HTML変換に失敗した場合に送出する例外。"""


class ImageEmbeddingError(RuntimeError):
    """画像埋め込みに失敗した場合に送出する例外。"""


# 「前後が半角スペース」で挟まれた `$...$` のみを対象（ユーザー運用ルール）
_INLINE_DOLLAR_MATH_RE = re.compile(r"(?<!\\)\s\$(.+?)\$\s", re.DOTALL)
# ```math ... ``` ブロック
_MATH_FENCE_RE = re.compile(r"```math\s*\n(.*?)\n```", re.DOTALL)

# 置換用プレースホルダ（Markdownパーサに数式中身を触らせない）
_INLINE_PLACEHOLDER_FMT = "%%MATHINLINE:{idx}%%"
_DISPLAY_PLACEHOLDER_FMT = "%%MATHDISPLAY:{idx}%%"


@dataclass(frozen=True)
class _MathPlaceholders:
    """Markdown→HTML変換前に退避した数式を復元するためのコンテナ。"""

    inline_tex: List[str]
    display_tex: List[str]


def _is_remote_url(src: str) -> bool:
    """src が http(s) URL なら True を返す。"""
    u = urlparse(src)
    return u.scheme in {"http", "https"}


def _is_data_uri(src: str) -> bool:
    """src が data URI なら True を返す。"""
    return src.strip().startswith("data:")


def _guess_mime_type(path: Path) -> str:
    """ファイル拡張子から MIME type を推定する（推定不能なら安全側フォールバック）。"""
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        ext = path.suffix.lower()
        if ext == ".svg":
            return "image/svg+xml"
        if ext == ".png":
            return "image/png"
        if ext in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if ext == ".gif":
            return "image/gif"
        if ext == ".webp":
            return "image/webp"
        return "application/octet-stream"
    return mime


def _file_to_data_uri(path: Path) -> str:
    """画像ファイルを data URI に変換する。"""
    if not path.is_file():
        raise FileNotFoundError(f"画像ファイルが見つかりません: {path}")
    mime = _guess_mime_type(path)
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _extract_math_to_placeholders(md_text: str) -> tuple[str, _MathPlaceholders]:
    """Markdown中の数式をプレースホルダに退避して返す。

    目的は、Markdownパーサ（CommonMark）が `_` や `*` を強調として解釈して
    LaTeX を破壊するのを防ぐことである。

    対象
    ----
    1) ` ```math ... ``` ` ブロック → display 数式として退避
    2) 半角スペースで挟まれた ` $...$ ` → inline 数式として退避

    Returns
    -------
    md_out, placeholders
        md_out はプレースホルダに置換済みのMarkdown。
        placeholders は復元用の LaTeX 生文字列リスト。
    """
    if not md_text:
        return md_text, _MathPlaceholders(inline_tex=[], display_tex=[])

    inline_tex: List[str] = []
    display_tex: List[str] = []

    # 1) display（math fence）を先に抜く（inlineの $...$ と干渉しないため）
    def _repl_display(m: re.Match[str]) -> str:
        body = m.group(1).strip()
        idx = len(display_tex)
        display_tex.append(body)
        return f"\n{_DISPLAY_PLACEHOLDER_FMT.format(idx=idx)}\n"

    md_mid = _MATH_FENCE_RE.sub(_repl_display, md_text)

    # 2) inline（空白 $...$ 空白）
    def _repl_inline(m: re.Match[str]) -> str:
        inner = m.group(1).strip()
        idx = len(inline_tex)
        inline_tex.append(inner)
        # プレースホルダ前後の空白は維持（m全体に空白が含まれるため）
        return f" {_INLINE_PLACEHOLDER_FMT.format(idx=idx)} "

    md_out = _INLINE_DOLLAR_MATH_RE.sub(_repl_inline, md_mid)

    return md_out, _MathPlaceholders(inline_tex=inline_tex, display_tex=display_tex)


def _restore_math_placeholders(html_text: str, ph: _MathPlaceholders) -> str:
    """HTML中のプレースホルダを MathJax 形式に復元する。

    Notes
    -----
    - バックスラッシュは Markdown/HTML の途中で失われやすいので、`&#92;` を用いて確実に出す。
    - LaTeX 本文は `& < >` をエスケープする（MathJax的に不要なクォートは触らない）。
    """
    out = html_text

    # display
    for i, tex in enumerate(ph.display_tex):
        tex_escaped = html.escape(tex, quote=False)
        token = _DISPLAY_PLACEHOLDER_FMT.format(idx=i)
        repl = f'<div class="mathjax-display">&#92;[\n{tex_escaped}\n&#92;]</div>'
        out = out.replace(token, repl)

    # inline
    for i, tex in enumerate(ph.inline_tex):
        tex_escaped = html.escape(tex, quote=False)
        token = _INLINE_PLACEHOLDER_FMT.format(idx=i)
        repl = f'<span class="mathjax-inline">&#92;({tex_escaped}&#92;)</span>'
        out = out.replace(token, repl)

    return out


def _slugify_heading_text(text: str) -> str:
    """見出しテキストから id として使う文字列（スラッグ）を生成する。

    目的
    ----
    - VS Code / GitHub Flavored Markdown が生成する見出しアンカーに近い規則で `id` を付与し、
      Markdown内の `[...](#...)`（自動生成TOCを含む）が安定して飛ぶようにする。

    仕様（近似）
    ------------
    - UnicodeをNFKC正規化し、小文字化する。
    - 句読点・記号（Unicodeカテゴリ: P*, S*）は基本的に除去する。
      ただし `-` と `_` は保持する（連結用）。
    - 空白類は `-` に置換し、連続する `-` は1つに圧縮する。
    - 先頭/末尾の `-` は除去する。
    - 日本語など非ASCII文字は保持する（URLエンコードされればブラウザ側で一致する）。

    Notes
    -----
    - GitHubの完全一致実装ではなく「TOCが飛ぶ」ことを最優先した実用近似である。
    """
    t = unicodedata.normalize("NFKC", text).strip().lower()

    out_chars: list[str] = []
    for ch in t:
        if ch.isspace():
            out_chars.append("-")
            continue

        if ch in {"-", "_"}:
            out_chars.append(ch)
            continue

        cat = unicodedata.category(ch)  # e.g., 'Ll', 'Nd', 'Po', 'Sm'
        if cat.startswith("P") or cat.startswith("S"):
            # punctuation / symbol -> drop (., +, (), （）, ・, …, etc.)
            continue

        out_chars.append(ch)

    slug = "".join(out_chars)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug


class _HeadingIdInjector(HTMLParser):
    """HTML中の見出し(h1〜h6)へ id を自動付与する。

    仕様
    ----
    - 既に id がある見出しは変更しない。
    - id が無い場合、見出しの表示テキスト（タグ除去後）から id を生成する。
    - 同一 id が衝突した場合は `-2`, `-3`, ... を付与して回避する。
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._out: list[str] = []
        self._id_counts: dict[str, int] = {}

        self._capturing: bool = False
        self._cap_tag: str = ""
        self._cap_attrs: list[tuple[str, str | None]] = []
        self._cap_inner_chunks: list[str] = []
        self._cap_text_chunks: list[str] = []
        self._cap_has_id: bool = False

    def html(self) -> str:
        return "".join(self._out)

    def _emit_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._out.append("<" + tag)
        for k, v in attrs:
            if v is None:
                self._out.append(f" {k}")
            else:
                esc = (
                    v.replace("&", "&amp;")
                    .replace('"', "&quot;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                self._out.append(f' {k}="{esc}"')
        self._out.append(">")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in {"h1", "h2", "h3", "h4", "h5", "h6"} and not self._capturing:
            self._capturing = True
            self._cap_tag = t
            self._cap_attrs = attrs
            self._cap_inner_chunks = []
            self._cap_text_chunks = []
            self._cap_has_id = any(k.lower() == "id" for k, _ in attrs)
            return

        if self._capturing:
            self._cap_inner_chunks.append(self.get_starttag_text()) # type: ignore
            return

        self._out.append(self.get_starttag_text()) # type: ignore

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if self._capturing and t == self._cap_tag:
            attrs = list(self._cap_attrs)
            if not self._cap_has_id:
                text = "".join(self._cap_text_chunks)
                base = _slugify_heading_text(text) or "section"
                n = self._id_counts.get(base, 0) + 1
                self._id_counts[base] = n
                hid = base if n == 1 else f"{base}-{n}"
                attrs.append(("id", hid))

            self._emit_starttag(self._cap_tag, attrs)
            self._out.append("".join(self._cap_inner_chunks))
            self._out.append(f"</{self._cap_tag}>")

            self._capturing = False
            self._cap_tag = ""
            self._cap_attrs = []
            self._cap_inner_chunks = []
            self._cap_text_chunks = []
            self._cap_has_id = False
            return

        if self._capturing:
            self._cap_inner_chunks.append(f"</{t}>")
            return

        self._out.append(f"</{t}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._capturing:
            self._cap_inner_chunks.append(self.get_starttag_text()) # type: ignore
            return
        self._out.append(self.get_starttag_text()) # type: ignore

    def handle_data(self, data: str) -> None:
        if self._capturing:
            self._cap_inner_chunks.append(data)
            self._cap_text_chunks.append(data)
            return
        self._out.append(data)

    def handle_entityref(self, name: str) -> None:
        ent = f"&{name};"
        if self._capturing:
            self._cap_inner_chunks.append(ent)
            self._cap_text_chunks.append(html.unescape(ent))
            return
        self._out.append(ent)

    def handle_charref(self, name: str) -> None:
        ent = f"&#{name};"
        if self._capturing:
            self._cap_inner_chunks.append(ent)
            self._cap_text_chunks.append(html.unescape(ent))
            return
        self._out.append(ent)

    def handle_comment(self, data: str) -> None:
        com = f"<!--{data}-->"
        if self._capturing:
            self._cap_inner_chunks.append(com)
            return
        self._out.append(com)

    def handle_decl(self, decl: str) -> None:
        dec = f"<!{decl}>"
        if self._capturing:
            self._cap_inner_chunks.append(dec)
            return
        self._out.append(dec)


def _inject_heading_ids(html_text: str) -> str:
    """HTML内の見出し(h1〜h6)へ id を自動付与する。"""
    parser = _HeadingIdInjector()
    parser.feed(html_text)
    parser.close()
    return parser.html()


def _convert_markdown_to_html(md_text: str) -> str:
    """Markdown を HTML に変換する（表/箇条書き/強調/コードフェンスを含む）。"""
    if not md_text.strip():
        raise ValueError("Markdownが空です。")

    try:
        from markdown_it import MarkdownIt  # type: ignore
    except ModuleNotFoundError as e:
        raise MarkdownConversionError(
            "markdown-it-py が必要です。\n"
            "  pip install markdown-it-py"
        ) from e

    md = MarkdownIt("default", {"html": True, "linkify": False, "typographer": True})
    md.enable("table")
    md.enable("strikethrough")
    return md.render(md_text)


@dataclass(frozen=True)
class EmbedReport:
    embedded: int
    skipped_remote: int
    skipped_data_uri: int
    skipped_missing: int
    errors: int


class _ImgSrcEmbeddingParser(HTMLParser):
    """HTML中の <img src="..."> を走査し、ローカル画像を data URI に置換する。"""

    def __init__(self, base_dir: Path, strict: bool) -> None:
        super().__init__(convert_charrefs=False)
        self._base_dir = base_dir
        self._strict = strict
        self._chunks: List[str] = []

        self._embedded = 0
        self._skipped_remote = 0
        self._skipped_data_uri = 0
        self._skipped_missing = 0
        self._errors = 0

    def get_html(self) -> str:
        return "".join(self._chunks)

    def get_report(self) -> EmbedReport:
        return EmbedReport(
            embedded=self._embedded,
            skipped_remote=self._skipped_remote,
            skipped_data_uri=self._skipped_data_uri,
            skipped_missing=self._skipped_missing,
            errors=self._errors,
        )

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() != "img":
            self._chunks.append(self.get_starttag_text()) # type: ignore
            return

        attr_dict: Dict[str, str] = {}
        for k, v in attrs:
            attr_dict[k] = "" if v is None else v

        src = attr_dict.get("src", "")
        if not src:
            self._chunks.append(self.get_starttag_text()) # type: ignore
            return

        if _is_data_uri(src):
            self._skipped_data_uri += 1
            self._chunks.append(self.get_starttag_text()) # type: ignore
            return

        if _is_remote_url(src):
            self._skipped_remote += 1
            self._chunks.append(self.get_starttag_text()) # type: ignore
            return

        if src.startswith("file://"):
            src_path = Path(src.replace("file://", "", 1))
            if not src_path.is_absolute():
                src_path = (self._base_dir / src_path).resolve()
        else:
            src_path = (self._base_dir / src).resolve()

        try:
            data_uri = _file_to_data_uri(src_path)
            attr_dict["src"] = data_uri
            self._embedded += 1
        except FileNotFoundError:
            self._skipped_missing += 1
            if self._strict:
                self._errors += 1
                raise ImageEmbeddingError(f"参照画像が存在しません: {src} -> {src_path}")
        except Exception as e:
            self._errors += 1
            if self._strict:
                raise ImageEmbeddingError(f"画像埋め込みに失敗: {src} -> {src_path}: {e}") from e
            attr_dict["src"] = src

        self._chunks.append("<img")
        for k, v in attr_dict.items():
            escaped = (
                v.replace("&", "&amp;")
                .replace('"', "&quot;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            self._chunks.append(f' {k}="{escaped}"')
        self._chunks.append(">")

    def handle_endtag(self, tag: str) -> None:
        self._chunks.append(f"</{tag}>")

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_data(self, data: str) -> None:
        self._chunks.append(data)

    def handle_entityref(self, name: str) -> None:
        self._chunks.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._chunks.append(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        self._chunks.append(f"<!--{data}-->")

    def handle_decl(self, decl: str) -> None:
        self._chunks.append(f"<!{decl}>")


def md_to_self_contained_html(
    md_path: Path,
    *,
    out_path: Path,
    title: Optional[str] = None,
    strict: bool = True,
    mathjax: str = "cdn",
    mathjax_js: Optional[Path] = None,
) -> EmbedReport:
    """Markdown→HTML 変換（画像は data URI で埋め込み、数式は MathJax で描画）。

    数式の仕様
    ----------
    - display: ```math ... ```
    - inline: 半角スペースで挟まれた ` $...$ ` のみ（ユーザー運用ルール）
      ただし、Markdownの強調パースを避けるため、Markdown→HTML変換の前に
      数式をプレースホルダへ退避し、HTML生成後に復元する。

    Parameters
    ----------
    md_path:
        入力Markdownパス。
    out_path:
        出力HTMLパス。
    title:
        HTML <title>（省略時は入力ファイル名）。
    strict:
        Trueの場合、参照画像が見つからない/埋め込み失敗で例外を送出する。
    mathjax:
        'cdn' はCDN参照（ネットワーク必須）。'inline' はローカルJSをHTMLに埋め込む（--mathjax-js必須）。
    mathjax_js:
        mathjax='inline' のときに埋め込むJSファイル（例: tex-chtml.js）。

    Returns
    -------
    EmbedReport
        埋め込んだ画像数などのレポート。
    """
    if md_path.suffix.lower() not in {".md", ".markdown"}:
        raise ValueError(f"入力がMarkdownに見えません: {md_path}")
    if not md_path.is_file():
        raise FileNotFoundError(f"Markdownが見つかりません: {md_path}")

    md_text = md_path.read_text(encoding="utf-8")
    md_text, ph = _extract_math_to_placeholders(md_text)

    body_html = _convert_markdown_to_html(md_text)
    body_html = _restore_math_placeholders(body_html, ph)
    body_html = _inject_heading_ids(body_html)

    base_dir = md_path.parent.resolve()
    parser = _ImgSrcEmbeddingParser(base_dir=base_dir, strict=strict)
    parser.feed(body_html)
    parser.close()

    report = parser.get_report()
    embedded_body = parser.get_html()

    page_title = title if title is not None else md_path.stem

    css = (
        "  <style>\n"
        "    .page {\n"
        "      width: 1000px;\n"
        "      margin: 0 auto;\n"
        "      box-sizing: border-box;\n"
        "    }\n"
        "    table {\n"
        "      border-collapse: collapse;\n"
        "      width: 100%;\n"
        "      display: block;\n"
        "      overflow-x: auto;\n"
        "    }\n"
        "    th, td {\n"
        "      border: 1px solid #999;\n"
        "      padding: 6px 10px;\n"
        "      vertical-align: top;\n"
        "      white-space: nowrap;\n"
        "    }\n"
        "    thead th {\n"
        "      background: #f3f3f3;\n"
        "    }\n"
        "  </style>\n"
    )

    # MathJax 設定：$...$ を無効化し、\( ... \), \[ ... \] のみ有効
    mathjax_config = (
        "  <script>\n"
        "    window.MathJax = {\n"
        "      tex: {\n"
        "        inlineMath: [['\\\\(', '\\\\)']],\n"
        "        displayMath: [['\\\\[', '\\\\]']],\n"
        "      },\n"
        "      options: {\n"
        "        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],\n"
        "      }\n"
        "    };\n"
        "  </script>\n"
    )

    if mathjax == "cdn":
        mathjax_loader = '  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>\n'
    elif mathjax == "inline":
        if mathjax_js is None:
            raise ValueError("--mathjax=inline の場合は --mathjax-js を指定してください。")
        if not mathjax_js.is_file():
            raise FileNotFoundError(f"MathJax JS が見つかりません: {mathjax_js}")
        js_code = mathjax_js.read_text(encoding="utf-8")
        mathjax_loader = "  <script>\n" + js_code + "\n  </script>\n"
    else:
        raise ValueError(f"mathjax は 'cdn' または 'inline' を指定してください: {mathjax}")

    full_html = (
        "<!doctype html>\n"
        "<html>\n"
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"  <title>{page_title}</title>\n"
        f"{css}"
        f"{mathjax_config}"
        f"{mathjax_loader}"
        "</head>\n"
        "<body>\n"
        '  <div class="page">\n'
        f"{embedded_body}\n"
        "  </div>\n"
        "</body>\n"
        "</html>\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(full_html, encoding="utf-8")
    return report


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MarkdownをHTMLへ変換し、ローカル画像をdata URIとして埋め込む（数式はMathJax）。"
    )
    p.add_argument("md", type=str, help="入力Markdownファイルパス")
    p.add_argument("-o", "--out", type=str, required=True, help="出力HTMLファイルパス")
    p.add_argument("--title", type=str, default=None, help="HTML <title>（省略可）")
    p.add_argument(
        "--non-strict",
        action="store_true",
        help="画像が見つからない場合でもエラーにせず続行する（srcは保持）。",
    )
    p.add_argument(
        "--mathjax",
        type=str,
        default="cdn",
        choices=["cdn", "inline"],
        help="数式描画用MathJaxの読み込み方法。cdn=CDN参照, inline=ローカルJSをHTMLに埋め込む（--mathjax-js必須）。",
    )
    p.add_argument(
        "--mathjax-js",
        type=str,
        default=None,
        help="--mathjax=inline のときに埋め込む MathJax v3 の tex-chtml.js（または互換バンドル）へのパス。",
    )
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    md_path = Path(args.md)
    out_path = Path(args.out)
    strict = not bool(args.non_strict)

    try:
        report = md_to_self_contained_html(
            md_path,
            out_path=out_path,
            title=args.title,
            strict=strict,
            mathjax=args.mathjax,
            mathjax_js=Path(args.mathjax_js) if args.mathjax_js is not None else None,
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise

    print(
        "[OK] wrote: "
        f"{out_path}\n"
        f"  embedded={report.embedded}, "
        f"skipped_remote={report.skipped_remote}, "
        f"skipped_data_uri={report.skipped_data_uri}, "
        f"skipped_missing={report.skipped_missing}, "
        f"errors={report.errors}"
    )


if __name__ == "__main__":
    main()
# python md2html.py index.md -o index.html --non-strict