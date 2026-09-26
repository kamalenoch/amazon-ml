"""Text normalization for business names and addresses.

Country-agnostic: every rule is language-level (Latin transliteration, punctuation,
abbreviation folding), never keyed on a fixed country list, so unseen countries
(France in test) go through the same path.
"""
import re
from .translit import to_ascii as _romanize

# ----------------------------------------------------------------------------- tokens
LEGAL = {
    # EN / IN
    "pvt": "pvt", "private": "pvt", "pte": "pvt", "prvt": "pvt",
    "ltd": "ltd", "limited": "ltd", "ltda": "ltd", "lt": "ltd",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc", "pc": "pc",
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "cos": "co", "tbk": "tbk", "gmbh": "gmbh", "ag": "ag",
    "public": "public", "opc": "opc", "holdings": None, "group": None,
    # FR (and generic civil-law)
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "sci": "sci",
    "snc": "snc", "sca": "sca", "scop": "scop", "selarl": "selarl", "scp": "scp", "gie": "gie",
}
LEGAL_FORMS = {v for v in LEGAL.values() if v}
NAME_STOP = {"the", "and", "of", "de", "des", "du", "la", "le", "les", "d", "l", "et", "a",
             "an", "www", "com", "net", "org", "in", "co.in", "biz", "info", "fr", "us", "dba",
             "nee", "id", "m/s", "ms", "mr", "mrs", "sri", "shri", "shree"}

ADDR_MAP = {
    "street": "st", "str": "st", "st": "st", "avenue": "ave", "av": "ave", "ave": "ave", "avn": "ave",
    "road": "rd", "rd": "rd", "drive": "dr", "dr": "dr", "lane": "ln", "ln": "ln",
    "circle": "cir", "cir": "cir", "court": "ct", "ct": "ct", "boulevard": "blvd", "blvd": "blvd",
    "bd": "blvd", "bld": "blvd", "place": "pl", "pl": "pl", "parkway": "pkwy", "pkwy": "pkwy",
    "highway": "hwy", "hwy": "hwy", "terrace": "ter", "ter": "ter", "trail": "trl", "trl": "trl",
    "square": "sq", "sq": "sq", "cove": "cv", "cv": "cv", "way": "way", "loop": "loop",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw", "suite": "ste", "ste": "ste", "apartment": "apt", "apt": "apt",
    "floor": "flr", "flr": "flr", "fl": "flr", "etage": "flr", "building": "bldg", "bldg": "bldg",
    "sector": "sec", "sec": "sec", "nagar": "ngr", "ngr": "ngr", "colony": "col", "marg": "marg",
    # FR
    "rue": "rue", "r": "rue", "allee": "all", "all": "all", "chemin": "chem", "chem": "chem",
    "impasse": "imp", "imp": "imp", "route": "rte", "rte": "rte", "faubourg": "fbg", "fbg": "fbg",
    "quai": "quai", "cours": "crs", "crs": "crs", "residence": "res", "res": "res",
}
ADDR_STOP = {"no", "num", "number", "nr", "of", "the", "de", "des", "du", "la", "le", "les", "d", "l",
             "city", "cdp", "borough", "village", "town", "township", "bis", "ter", "unit", "near",
             "opp", "opposite", "behind", "landmark", "house", "door", "flat", "plot", "shop", "h",
             "hno", "dno", "p", "s/o", "so", "at", "po", "post", "dist", "district", "tehsil", "taluka",
             "eme", "er", "nd", "rd_", "th"}

# state / region names -> canonical code, matched by phonetic skeleton so that
# "Maharashtra", "महाराष्ट्र" (-> "mhaaraassttr") and typos land on the same code.
_STATES = {
    # US
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "dc": "district of columbia",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho", "il": "illinois",
    "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon",
    "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina", "sd": "south dakota",
    "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia",
    "wa": "washington", "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
}
_IN_STATES = {
    "mh": "maharashtra", "dl": "delhi", "up": "uttar pradesh", "ka": "karnataka", "tn": "tamil nadu",
    "gj": "gujarat", "wb": "west bengal", "tg": "telangana", "ts": "telangana", "hr": "haryana",
    "kl": "kerala", "rj": "rajasthan", "br": "bihar", "mp": "madhya pradesh", "ap": "andhra pradesh",
    "od": "odisha", "or": "orissa", "pb": "punjab", "jh": "jharkhand", "ch": "chandigarh",
    "ga": "goa", "as": "assam", "uk": "uttarakhand", "hp": "himachal pradesh", "jk": "jammu and kashmir",
    "ct": "chhattisgarh", "cg": "chhattisgarh", "py": "puducherry",
}

# ----------------------------------------------------------------------------- helpers
_RE_ID = re.compile(r"\(\s*id\s*:?\s*\d+\s*\)", re.I)
_RE_DOT_SPLIT = re.compile(r"\.(?=[a-z0-9]{2,})")
_RE_NONALNUM = re.compile(r"[^a-z0-9/ ]+")
_RE_SPACES = re.compile(r"\s+")
_RE_ORD = re.compile(r"\b(\d+)(st|nd|rd|th|eme|er|e)\b")
_RE_NUM = re.compile(r"\d+")
_OCR = str.maketrans({"0": "o", "1": "l", "5": "s", "3": "e", "4": "a", "8": "b", "7": "t"})


def to_ascii(s: str) -> str:
    if not s:
        return ""
    if not s.isascii():
        s = _romanize(s)
    return s.lower()


def _basic(s: str) -> str:
    s = to_ascii(s)
    s = _RE_ID.sub(" ", s)
    s = s.replace("&", " and ").replace("+", " plus ").replace("@", " at ")
    s = _RE_DOT_SPLIT.sub(" ", s)      # "pvt.ltd" -> "pvt ltd", "ironl.com" -> "ironl com"
    s = s.replace(".", "")             # "s.a.r.l." -> "sarl", "inc." -> "inc"
    s = s.replace("'", "")
    s = _RE_NONALNUM.sub(" ", s)
    return _RE_SPACES.sub(" ", s).strip()


def collapse(t: str) -> str:
    out = []
    for c in t:
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


_DIGRAPHS = [("ph", "f"), ("bh", "b"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"),
             ("sh", "s"), ("ch", "c"), ("ck", "k"), ("nb", "mb"), ("np", "mp"), ("x", "ks"),
             ("q", "k"), ("c", "k"), ("v", "b"), ("w", ""), ("z", "j"), ("g", "j"), ("h", "")]
_VOWELS = set("aeiouy")


def skeleton(t: str) -> str:
    """Phonetic consonant skeleton; makes transliterations / vowel typos collide.
    'private'/'praaivett'/'praibhet' -> 'prbt';  'management'/'myaanejmentt' -> 'mnjmnt'."""
    t = collapse(t)
    for a, b in _DIGRAPHS:
        if a in t:
            t = t.replace(a, b)
    return collapse("".join(c for c in t if c not in _VOWELS))


def _fix_ocr(tok: str) -> str:
    # tokens mixing letters and digits with a majority of letters: '0ne' -> 'one'
    if tok.isalnum() and not tok.isalpha() and not tok.isdigit():
        n_alpha = sum(c.isalpha() for c in tok)
        if n_alpha >= len(tok) / 2:
            return tok.translate(_OCR)
    return tok


_STATE_FULL = {}   # exact full name -> code
_STATE_SKEL = {}   # skeleton -> code, only for skeletons that are unambiguous and >= 2 chars
_skel_codes = {}
for _code, _name in list(_STATES.items()) + list(_IN_STATES.items()):
    _code = "od" if _name == "orissa" else _code          # orissa/odisha are one state
    _STATE_FULL.setdefault(_name.replace(" ", ""), _code)
    _skel_codes.setdefault(skeleton(_name.replace(" ", "")), set()).add(_code)
for _sk, _codes in _skel_codes.items():
    # '' (hawaii/iowa/ohio), 'd', 't', 'j' match arbitrary words; 'rjn' is arizona AND oregon
    if len(_sk) >= 2 and len(_codes) == 1:
        _STATE_SKEL[_sk] = next(iter(_codes))


def state_of(part: str):
    """Map one comma-separated address component to a state code if it is one."""
    p = part.replace(" ", "")
    if len(p) == 2 and p.isalpha():
        return p if (p in _STATES or p in _IN_STATES) else None
    if p in _STATE_FULL:
        return _STATE_FULL[p]
    return _STATE_SKEL.get(skeleton(p)) if len(p) >= 4 else None


# ----------------------------------------------------------------------------- public API
_LEGAL_SKEL = {skeleton(k): v for k, v in LEGAL.items() if v and len(k) >= 5}


def _legal(t: str):
    """Legal-form lookup that survives transliteration ('praivet', 'limittedd')."""
    if t in LEGAL:
        return LEGAL[t]
    if len(t) >= 5:
        c = collapse(t)
        if c in LEGAL:
            return LEGAL[c]
        return _LEGAL_SKEL.get(skeleton(t))
    return None


def norm_name(s: str):
    """Returns (clean_string, core_tokens, legal_forms)."""
    b = _basic(s)
    toks = [_fix_ocr(t) for t in b.replace("/", " ").split()]
    core, legal = [], []
    for t in toks:
        v = _legal(t)
        if v:
            legal.append(v)
            continue
        if t in NAME_STOP:
            continue
        core.append(collapse(t) if len(t) > 3 else t)
    return " ".join(core), core, sorted(set(legal))


def norm_addr(s: str):
    """Returns (clean_string, alpha_tokens, numbers, state_code)."""
    if not s:
        return "", [], [], ""
    a = to_ascii(s).replace("#", " ")
    state, keep = "", []
    for part in a.split(","):
        p = _RE_NONALNUM.sub(" ", part.replace(".", " ")).strip()
        st = state_of(p) if p else None
        if st:
            state = st
        else:
            keep.append(part)
    b = _basic(",".join(keep))
    b = _RE_ORD.sub(r"\1", b)
    b = b.replace("/", " ")
    alpha, nums = [], []
    for t in b.split():
        if t.isdigit():
            t = t.lstrip("0") or "0"
            nums.append(t)
            continue
        t = _fix_ocr(t)
        if any(c.isdigit() for c in t):          # 1715b, 8c5 -> keep leading number too
            m = _RE_NUM.match(t)
            if m:
                nums.append(m.group(0).lstrip("0") or "0")
            alpha.append(t)
            continue
        t = ADDR_MAP.get(t, t)
        if t in ADDR_STOP or len(t) < 2:
            continue
        alpha.append(collapse(t))
    return " ".join(alpha), alpha, nums, state
