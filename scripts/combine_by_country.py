"""Combine two complete submissions by country: rows of the S1 in --countries come from --take, all other rows
from --base. Valid because blocking keys are country-scoped: a pool record is only ever matched to an S1 of
its own country, so rows of different countries never compete for the same record.
Plain Python (no pandas); streams the files.

  python scripts/combine_by_country.py --test-s1 <data>/test/test_source1.tsv \
      --base output_v4 --take output_teammate --countries France --out output_combined

Each of --base / --take is a folder with matching_results.tsv and candidate_pairs.tsv."""
import argparse
import os
from collections import Counter


def s1_countries(path):
    with open(path, encoding="utf-8", newline="") as f:
        head = f.readline().rstrip("\r\n").split("\t")
        i_id, i_ct = head.index("entity_id"), head.index("country")
        return {p[i_id]: p[i_ct] for p in (l.rstrip("\r\n").split("\t") for l in f) if len(p) > max(i_id, i_ct)}


def combine(name, base, take, out, ctry, keep):
    rows = {}
    with open(os.path.join(take, name), encoding="utf-8", newline="") as f:
        header_take = f.readline()
        for line in f:
            s1 = line.split("\t", 1)[0]
            if ctry.get(s1) in keep:
                rows[s1] = line if line.endswith("\n") else line + "\n"
    stats = Counter()
    with open(os.path.join(base, name), encoding="utf-8", newline="") as f, \
            open(os.path.join(out, name), "w", encoding="utf-8", newline="") as g:
        header = f.readline()
        if header.strip() != header_take.strip():
            raise SystemExit(f"{name}: headers differ: {header.strip()!r} vs {header_take.strip()!r}")
        g.write(header)
        seen = set()
        for line in f:
            s1 = line.split("\t", 1)[0]
            seen.add(s1)
            c = ctry.get(s1, "?")
            if c in keep:
                new = rows.get(s1)
                if new is None:
                    stats["missing_in_take"] += 1
                    new = line
                else:
                    stats["changed" if new.rstrip("\r\n") != line.rstrip("\r\n") else "unchanged"] += 1
                g.write(new if new.endswith("\n") else new + "\n")
                stats[f"from take ({c})"] += 1
            else:
                g.write(line if line.endswith("\n") else line + "\n")
                stats[f"from base ({c})"] += 1
    stats["take rows not in base"] = len(set(rows) - seen)
    print(f"{name}: " + ", ".join(f"{k} {v:,}" for k, v in sorted(stats.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-s1", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--take", required=True)
    ap.add_argument("--countries", required=True, help="comma-separated, e.g. France")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ctry = s1_countries(a.test_s1)
    keep = {c.strip() for c in a.countries.split(",")}
    unknown = keep - set(ctry.values())
    if unknown:
        raise SystemExit(f"unknown countries {unknown}; test has {sorted(set(ctry.values()))}")
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        combine(name, a.base, a.take, a.out, ctry, keep)
    print(f"wrote {a.out}/matching_results.tsv and candidate_pairs.tsv - run utils/validate_submission.py on them")


if __name__ == "__main__":
    main()
