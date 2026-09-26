"""Romanization to ASCII without GPL dependencies.

Indic scripts: the nine Brahmic blocks in Unicode (Devanagari .. Malayalam) share the
ISCII layout, so one offset -> Latin table covers all of them. Handles the inherent
vowel, virama, anusvara (n / m before labials), nukta, Malayalam chillus.
Latin with diacritics: NFKD + drop combining marks. Anything else: anyascii (ISC).
"""
import unicodedata

from anyascii import anyascii

_BLOCKS = (0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00)
_TAMIL, _MALAYALAM = 0x0B80, 0x0D00

# offset within block -> latin; consonants carry an inherent 'a'
_CONS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
    0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
    0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b",
    0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh",
    0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
_VOWEL = {  # independent vowels
    0x04: "a", 0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri",
    0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o",
    0x14: "au", 0x60: "ri", 0x61: "li", 0x72: "i", 0x73: "u",
}
_SIGN = {  # dependent vowel signs (replace the inherent 'a')
    0x3A: "e", 0x3B: "e", 0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri",
    0x44: "ri", 0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
    0x4C: "au", 0x4F: "aw", 0x62: "li", 0x63: "li", 0x56: "ai", 0x57: "au",
}
_NASAL = {0x01, 0x02, 0x70}          # candrabindu, anusvara, gurmukhi tippi
_MARKS = {0x3C, 0x4D, 0x71, 0x51, 0x52, 0x53, 0x54, 0x55, 0x3D}   # nukta, virama, addak, stress, avagraha
_VIRAMA, _NUKTA = 0x4D, 0x3C
_NUKTA_OF = {"ph": "f", "j": "z", "k": "q", "kh": "kh", "g": "g", "d": "r", "dh": "rh"}
_EXTRA = {  # script-specific code points (absolute)
    0x09F0: "r", 0x09F1: "w", 0x0B71: "w", 0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l",
    0x0D7E: "l", 0x0D7F: "k", 0x0D54: "m", 0x0D55: "y", 0x0D56: "l", 0x0D4E: "r", 0x0C58: "ts",
    0x0C59: "dz", 0x0B83: "",
}
_LATIN = {"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ß": "ss", "đ": "d",
          "Đ": "D", "ł": "l", "Ł": "L", "ı": "i", "þ": "th", "ð": "d", "°": " ", "’": "'", "‘": "'",
          "“": '"', "”": '"', "–": "-", "—": "-", " ": " "}
_ZW = {"‌", "‍", "​", "﻿"}
_LABIAL = {"p", "b", "m", "f"}


def _block(cp):
    if 0x0900 <= cp < 0x0D80:
        base = cp & ~0x7F
        return base, cp - base
    return None, None


def _indic_run(chars, i):
    """Transliterate a maximal run of Indic code points starting at i. Returns (text, next_i)."""
    out = []           # list of syllable pieces
    pending = None     # consonant waiting for its vowel
    n = len(chars)
    while i < n:
        ch = chars[i]
        if ch in _ZW:
            i += 1
            continue
        cp = ord(ch)
        base, off = _block(cp)
        if base is None:
            break
        if cp in _EXTRA:
            if pending:
                out.append(pending + "a")
                pending = None
            out.append(_EXTRA[cp])
        elif off in _CONS:
            if pending:
                out.append(pending + "a")
            pending = _CONS[off]
            # Malayalam റ്റ (rra virama rra) is "tt"
            if base == _MALAYALAM and off == 0x31 and i + 2 < n and ord(chars[i + 1]) == 0x0D4D \
                    and ord(chars[i + 2]) == 0x0D31:
                pending = "tt"
                i += 2
        elif off in _SIGN:
            out.append((pending or "") + _SIGN[off])
            pending = None
        elif off == _VIRAMA:
            if pending:
                out.append(pending)
                pending = None
        elif off == _NUKTA:
            if pending:
                pending = _NUKTA_OF.get(pending, pending)
        elif off in _NASAL:
            if pending:
                out.append(pending + "a")
                pending = None
            out.append("\x00" if off == 0x02 else "\x01")   # anusvara / other nasal, resolved below
        elif off == 0x03:             # visarga
            if pending:
                out.append(pending + "a")
                pending = None
            out.append("h")
        elif off in _VOWEL:
            if pending:
                out.append(pending + "a")
                pending = None
            out.append(_VOWEL[off])
        elif 0x66 <= off <= 0x6F:     # digits
            if pending:
                out.append(pending + "a")
                pending = None
            out.append(str(off - 0x66))
        elif off in (0x64, 0x65):     # danda
            if pending:
                out.append(pending + "a")
                pending = None
            out.append(" ")
        elif off in _MARKS:
            pass
        else:
            if pending:
                out.append(pending + "a")
                pending = None
        i += 1
    if pending:
        out.append(pending)           # word-final schwa deletion
    s = "".join(out)
    if "\x00" in s or "\x01" in s:
        s = "".join(_nasal(s, j) if c in "\x00\x01" else c for j, c in enumerate(s))
    return s, i


def _nasal(s, j):
    nxt = s[j + 1:j + 3]
    if not nxt:
        return "ng" if s[j] == "\x00" else "n"   # word-final anusvara: 'marketing', 'building'
    return "m" if (nxt[0] in _LABIAL and nxt != "ph") else "n"


def to_ascii(s: str) -> str:
    if not s or s.isascii():
        return s
    chars, out, i = list(s), [], 0
    while i < len(chars):
        ch = chars[i]
        cp = ord(ch)
        if cp < 128:
            out.append(ch)
            i += 1
        elif 0x0900 <= cp < 0x0D80:
            t, i = _indic_run(chars, i)
            out.append(t)
        elif ch in _ZW:
            i += 1
        else:
            if ch in _LATIN:
                out.append(_LATIN[ch])
            else:
                d = unicodedata.normalize("NFKD", ch)
                base = "".join(c for c in d if not unicodedata.combining(c))
                out.append(base if base.isascii() else anyascii(base))
            i += 1
    return "".join(out)
