"""Stage 1+2: Normalization and Transliteration of business names and addresses.

Everything is rule based, offline and deterministic. Nothing depends on a specific
country label: the same rules run for US, India, France or any unseen country.
"""
import re

from anyascii import anyascii  # ISC licence; offline Unicode -> ASCII transliteration

# Legal / generic suffixes and filler words dropped from the "core" name (kept in name_norm).
LEGAL = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "cos", "llc", "llp", "lp", "ltd", "limited",
    "pvt", "private", "plc", "pc", "pllc", "pa", "sa", "sas", "sasu", "sarl", "sci", "eurl", "ei", "snc", "scp",
    "gmbh", "ag", "group", "groupe", "the", "of", "and", "et", "de", "du", "des", "la", "le", "les", "ms", "dba",
    "aka", "fka", "www", "com", "net", "org", "in", "fr", "us", "a", "an", "at", "by", "esq",
}
# Name synonyms mapped to one canonical spelling before comparison.
NAME_SYN = {
    "pvt": "private", "ltd": "limited", "corp": "corporation", "co": "company", "inc": "incorporated",
    "intl": "international", "mfg": "manufacturing", "svcs": "services", "svc": "services", "mgmt": "management",
    "assoc": "associates", "assn": "association", "bros": "brothers", "ctr": "center", "centre": "center",
    "tech": "technologies", "natl": "national", "dept": "department", "univ": "university", "hosp": "hospital",
    "cie": "compagnie", "ste": "societe", "st": "saint",
}
# Address words mapped to one short canonical form (long and short spellings collide on purpose).
ADDR_SYN = {
    "street": "st", "str": "st", "saint": "st", "road": "rd", "avenue": "av", "ave": "av", "boulevard": "bd",
    "blvd": "bd", "boul": "bd", "drive": "dr", "lane": "ln", "court": "ct", "place": "pl", "circle": "cir",
    "highway": "hwy", "parkway": "pkwy", "pky": "pkwy", "terrace": "ter", "trail": "trl", "square": "sq",
    "suite": "ste", "apartment": "apt", "building": "bldg", "floor": "fl", "flr": "fl", "north": "n",
    "south": "s", "east": "e", "west": "w", "near": "nr", "opposite": "opp", "number": "no", "rue": "r",
    "chemin": "ch", "impasse": "imp", "allee": "all", "route": "rte", "faubourg": "fbg", "mount": "mt",
    "fort": "ft", "county": "cty", "cnty": "cty", "township": "twp", "sector": "sec", "nagar": "ngr",
    "colony": "col", "marg": "mg", "extension": "ext", "cross": "crs", "main": "mn", "block": "blk",
    "plot": "plt", "house": "h", "hno": "h", "flat": "flt", "way": "wy", "point": "pt", "heights": "hts",
    "center": "ctr", "centre": "ctr", "expressway": "expy", "freeway": "fwy", "junction": "jct",
}
NULLS = re.compile(r"<null>|\bn/?a\b|\bnull\b|\bnone\b", re.I)
DOMAIN = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9-]*)\.(?:com|in|net|org|co|fr|us|biz|info|io)\b")
NON_ALNUM = re.compile(r"[^a-z0-9]+")
ORDINAL = re.compile(r"^(\d+)(?:st|nd|rd|th)$")
LONG_DIGITS = re.compile(r"^\d{7,}$")  # phone numbers glued to names


def translit(s):
    """Unicode -> lowercase ASCII (Devanagari, Kannada, Tamil, accents ... all handled)."""
    return anyascii(s).lower() if not s.isascii() else s.lower()


def num_token(t):
    """'03520' -> '3520', '35th' -> '35'."""
    m = ORDINAL.match(t)
    if m:
        t = m.group(1)
    return (t.lstrip("0") or "0") if t.isdigit() else t


def name_tokens(raw):
    """Normalized name tokens (synonyms canonicalized, phone numbers dropped)."""
    s = NULLS.sub(" ", translit(raw)).replace("&", " and ")
    s = DOMAIN.sub(r" \1 ", s)
    s = s.replace(".", "").replace("'", "")  # L.L.C. -> llc, Moyna's -> moynas
    out = []
    for t in NON_ALNUM.split(s):
        if not t or LONG_DIGITS.match(t):
            continue
        out.append(NAME_SYN.get(t, num_token(t)))
    return out


def addr_tokens(raw):
    """Normalized address tokens (abbreviations canonicalized, numbers de-padded)."""
    s = NULLS.sub(" ", translit(raw)).replace("&", " and ").replace("'", "")
    return [ADDR_SYN.get(t, num_token(t)) for t in NON_ALNUM.split(s) if t]


_DIGRAPH = [("ph", "f"), ("gh", "g"), ("kh", "k"), ("bh", "b"), ("dh", "d"), ("th", "t"), ("sh", "s"),
            ("ch", "c"), ("ck", "k"), ("jh", "j")]
# (tested: also merging b/p, d/t, g/k lowered recall on non-ASCII names 0.9020 -> 0.8977, so not used)
_SKEL = str.maketrans({"c": "k", "q": "k", "z": "s", "v": "w", "m": "n", "x": "ks",
                       "a": None, "e": None, "i": None, "o": None, "u": None, "y": None, "h": None})


def skeleton(t):
    """Phonetic consonant skeleton: 'marketimg'/'marketing' -> 'nrktng', 'praivet'/'private' -> 'prwt'.

    Bridges transliteration spelling differences and vowel typos.
    """
    if t.isdigit():
        return ""
    for a, b in _DIGRAPH:
        t = t.replace(a, b)
    t = t.translate(_SKEL)
    return re.sub(r"(.)\1+", r"\1", t)


_LEGAL_WORDS = {"private", "limited", "corporation", "company", "incorporated"}
# Transliterated legal words ('praivet', 'limitedd') are caught by their skeleton.
_LEGAL_SKEL = {skeleton(w) for w in _LEGAL_WORDS | {"llp", "llc"}}


def normalize_record(name, addr):
    """Return (name_norm, name_core, addr_norm, name_skel)."""
    nt = name_tokens(name)
    translit_name = not name.isascii()
    core = [t for t in nt if t not in LEGAL and t not in _LEGAL_WORDS and not (translit_name and skeleton(t) in _LEGAL_SKEL)]
    at = addr_tokens(addr)
    sk = [k for k in (skeleton(t) for t in core) if len(k) >= 2]
    return " ".join(nt), " ".join(core), " ".join(at), " ".join(sk)
