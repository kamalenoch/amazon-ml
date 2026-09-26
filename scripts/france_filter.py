"""Post-processing for the unseen country: drop matched pairs whose pool name ADDS a decoy marker word.

Train evidence (dev candidates, close pairs): a name that gains the country name ('india'), 'group' or 'holdings'
is a true copy 0.0% of the time (9.5k, 19.9k, 16.6k pairs), with or without a dropped word. The French test
equivalents are 'france', 'groupe', 'holding(s)', 'participations', 'developpement'; the v3 stem fallback only
partly catches them (v3.1 still accepts ~10k French pairs that add one). Generic suffixes that true copies do
add ('services', 'associes', 'fils', 'cie', legal forms) are NOT touched.

  python scripts/france_filter.py --matching <in matching_results.tsv> --test-dir <data>/test --out <out.tsv>
Plain Python; streams the files."""
import argparse
import re
import unicodedata
from collections import Counter

WORDS = {"france", "groupe", "holding", "holdings", "participations", "participation", "developpement"}
TOK = re.compile(r"[a-z0-9]+")


def toks(s):
    return set(TOK.findall(unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()))


def names(path, keep_country):
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        head = f.readline().rstrip("\r\n").split("\t")
        i_id, i_n, i_c = head.index("entity_id"), head.index("business_name"), head.index("country")
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) > max(i_id, i_n, i_c) and p[i_c] in keep_country:
                out[p[i_id]] = p[i_n]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--countries", default="France")
    ap.add_argument("--words", default=",".join(sorted(WORDS)))
    a = ap.parse_args()
    keep = {c.strip() for c in a.countries.split(",")}
    words = {w.strip() for w in a.words.split(",") if w.strip()}
    s1 = names(f"{a.test_dir}/test_source1.tsv", keep)
    pool = names(f"{a.test_dir}/test_source2.tsv", keep)
    pool.update(names(f"{a.test_dir}/test_source3.tsv", keep))
    s1_toks = {k: toks(v) for k, v in s1.items()}
    removed, kept, rows_changed = Counter(), 0, 0
    with open(a.matching, encoding="utf-8", newline="") as f, open(a.out, "w", encoding="utf-8", newline="") as g:
        g.write(f.readline())
        for line in f:
            sid, _, m = line.rstrip("\r\n").partition("\t")
            if sid not in s1_toks or not m:
                g.write(line if line.endswith("\n") else line + "\n")
                continue
            base = s1_toks[sid]
            out = []
            for p in m.split(","):
                hit = (toks(pool.get(p, "")) - base) & words
                if hit:
                    removed.update(hit)
                else:
                    out.append(p)
                    kept += 1
            rows_changed += len(out) != len(m.split(","))
            g.write(f"{sid}\t{','.join(out)}\n")
    print(f"{a.countries}: kept {kept:,} pairs, removed {sum(removed.values()):,} marker hits in {rows_changed:,} S1 rows: {dict(removed)}")


if __name__ == "__main__":
    main()
