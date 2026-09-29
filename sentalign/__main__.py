"""CLI: python3 -m sentalign <src.txt> <tgt.txt>"""

import sys

from . import split_sentences, align


def main(argv):
    if len(argv) != 3:
        sys.stderr.write("usage: python3 -m sentalign <src.txt> <tgt.txt>\n")
        return 2
    with open(argv[1], encoding="utf-8") as fh:
        src = split_sentences(fh.read())
    with open(argv[2], encoding="utf-8") as fh:
        tgt = split_sentences(fh.read())
    print(align(src, tgt).path())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
