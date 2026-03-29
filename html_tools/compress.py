import argparse
import sys
from pathlib import Path

try:
    from .hash_id import generate_hash
    from .spec import CompressionSpec
    from .transform import attrs_to_string, filter_attrs, iter_kept_nodes, parse_html_root
except ImportError:  # pragma: no cover - direct script execution fallback
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from html_tools.hash_id import generate_hash
    from html_tools.spec import CompressionSpec
    from html_tools.transform import attrs_to_string, filter_attrs, iter_kept_nodes, parse_html_root


def dom_to_compressed_lines(html: str, spec: CompressionSpec) -> str:
    """HTML을 정책 기반으로 순회해 `태그 + text + hash id` 라인 텍스트로 변환합니다."""
    root = parse_html_root(html, spec)
    lines = []

    for node, path, text in iter_kept_nodes(root, spec):
        attrs = filter_attrs(node.attrs, spec)
        attr_str = attrs_to_string(attrs)

        tag_str = f"<{node.name}"
        if attr_str:
            tag_str += f" {attr_str}"
        tag_str += ">"

        node_hash = generate_hash(path)
        if text:
            lines.append(f'{tag_str} text="{text}" id={node_hash}')
        else:
            lines.append(f"{tag_str} id={node_hash}")

    return "\n".join(lines)


def convert_html_file_to_txt(
    input_html_path: str,
    output_txt_path: str,
    spec: CompressionSpec,
) -> None:
    """입력 HTML 파일을 압축 포맷으로 변환해 출력 텍스트 파일로 저장합니다."""
    with open(input_html_path, "r", encoding="utf-8") as f:
        html = f.read()

    result = dom_to_compressed_lines(html, spec=spec)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"✅ 변환 완료: {output_txt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert HTML file to compressed HTML text format.")
    parser.add_argument(
        "--input-html",
        default="input.html",
        help="Path to source HTML file (default: input.html)",
    )
    parser.add_argument(
        "--output-txt",
        help="Path to output TXT file (default: <input_basename>_compressed.txt)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_html)

    if args.output_txt:
        output_path = Path(args.output_txt)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_compressed.txt")

    spec = CompressionSpec()
    convert_html_file_to_txt(str(input_path), str(output_path), spec)


if __name__ == "__main__":
    main()
