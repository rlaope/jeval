"""Money formatting that respects the currency it is formatting, and the shares printed beside it.

A cost matrix is written in whatever currency the business thinks in, and each currency has its own
smallest unit. ``KRW 38,524,590.16`` is not a figure anyone in Seoul has ever read: the won has no
minor unit, so the ``.16`` is noise the report invented. ``JPY`` is the same; ``BHD`` and ``KWD``
carry three decimals, and most others carry two.

This module is presentation only. It never converts between currencies, never guesses a currency
from a locale, and never reads the network. The code the user passes (``--currency``) is printed
as-is in front of the amount, so an unknown code such as ``credits`` still works: it simply gets the
two-decimal default.

The report's inline script mirrors :func:`format_amount` and :func:`format_compact`; the digits it
needs travel in a ``data-currency-digits`` attribute written from :func:`minor_units`, so the two
implementations cannot disagree about how many decimals a currency has.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal

__all__ = [
    "DEFAULT_CURRENCY",
    "format_amount",
    "format_compact",
    "format_delta",
    "format_percent",
    "minor_units",
    "normalise_code",
]

DEFAULT_CURRENCY = "USD"

# ISO 4217 minor units for the currencies that are not the two-decimal default. Listed as data so a
# missing code is a one-line addition, never a branch.
_MINOR_UNITS: dict[str, int] = {
    # no minor unit in circulation
    "BIF": 0,
    "CLP": 0,
    "DJF": 0,
    "GNF": 0,
    "ISK": 0,
    "JPY": 0,
    "KMF": 0,
    "KRW": 0,
    "PYG": 0,
    "RWF": 0,
    "UGX": 0,
    "UYI": 0,
    "VND": 0,
    "VUV": 0,
    "XAF": 0,
    "XOF": 0,
    "XPF": 0,
    # three decimals
    "BHD": 3,
    "IQD": 3,
    "JOD": 3,
    "KWD": 3,
    "LYD": 3,
    "OMR": 3,
    "TND": 3,
}

# An expected cost is an average, not a payment: a per-case figure of 0.4 won is a real result
# of the arithmetic, and printing it as "KRW 0" would erase it. An average smaller than the
# currency's own minor unit keeps two significant figures, up to two decimals past that unit.
_SIGNIFICANT = 2
_EXTRA_DIGITS = 2

_SUFFIXES: tuple[tuple[float, str], ...] = (
    (1e12, "T"),
    (1e9, "B"),
    (1e6, "M"),
    (1e3, "k"),
)


def normalise_code(code: str | None) -> str:
    """The code as it will be printed: trimmed, upper-cased when it looks like ISO 4217.

    A blank code falls back to :data:`DEFAULT_CURRENCY`, because an amount with a leading space and
    no unit reads as a formatting bug. A free-form unit such as ``credits`` is kept as written.
    """
    text = (code or "").strip()
    if not text:
        return DEFAULT_CURRENCY
    if len(text) == 3 and text.isalpha():
        return text.upper()
    return text


def minor_units(code: str | None) -> int:
    """How many decimals an amount in this currency is written with."""
    return _MINOR_UNITS.get(normalise_code(code), 2)


def _digits_for(value: float, code: str | None) -> int:
    base = minor_units(code)
    magnitude = abs(value)
    if magnitude == 0 or magnitude >= 10.0**-base:
        return base
    needed = _SIGNIFICANT - 1 - math.floor(math.log10(magnitude))
    return min(base + _EXTRA_DIGITS, needed)


def _fixed(value: float, digits: int, *, grouped: bool = True) -> str:
    """``value`` to ``digits`` decimals, halves rounded away from zero, without a ``-0``.

    Python's own formatting rounds halves to even and a browser's ``toFixed`` rounds them up, so
    the same 2.5 won would print as 2 in the table and 3 in the slider under it. Rounding the exact
    binary value half-up here is what ``toFixed`` does, so the two sides agree on every tie.
    """
    quantum = Decimal(1).scaleb(-digits)
    rounded = Decimal(value).quantize(quantum, rounding=ROUND_HALF_UP)
    if rounded == 0:
        rounded = abs(rounded)
    return f"{rounded:,.{digits}f}" if grouped else f"{rounded:.{digits}f}"


def _prefix(code: str | None, value: str) -> str:
    return f"{normalise_code(code)} {value}"


def format_amount(value: float, code: str | None, *, unit: bool = True) -> str:
    """A full amount, grouped, with the currency's own number of decimals.

    ``format_amount(38524590.16, "KRW") == "KRW 38,524,590"`` and
    ``format_amount(1926.23, "USD") == "USD 1,926.23"``. ``unit=False`` drops the code, for a column
    whose header already names it. A non-finite value reads ``n/a``.
    """
    if not math.isfinite(value):
        return "n/a"
    digits = _digits_for(value, code)
    text = _fixed(value, digits)
    if digits > minor_units(code) and float(text.replace(",", "")) == 0:
        # Too small even for the widened window: write the currency's plain zero, not "0.0000".
        text = _fixed(0.0, minor_units(code))
    return _prefix(code, text) if unit else text


def format_compact(value: float, code: str | None = None, *, unit: bool = True) -> str:
    """A short amount for chart labels and live readouts: ``KRW 34.8M``, ``USD 1.93k``.

    Three significant figures above a thousand, and :func:`format_amount` below it, so a compact
    figure and a full figure of the same amount never contradict each other.
    """
    if not math.isfinite(value):
        return "n/a"
    magnitude = abs(value)
    ladder = list(reversed(_SUFFIXES))  # k, M, B, T
    for index, (size, suffix) in enumerate(ladder):
        if magnitude < size or (index + 1 < len(ladder) and magnitude >= ladder[index + 1][0]):
            continue
        text = _scaled(value / size)
        # 999,999.5 rounds to "1000k": a figure that reaches the next suffix moves up to it.
        if abs(float(text)) >= 1000 and index + 1 < len(ladder):
            size, suffix = ladder[index + 1]
            text = _scaled(value / size)
        text += suffix
        return _prefix(code, text) if (unit and code is not None) else text
    if code is None:
        return _fixed(value, 0 if magnitude >= 10 or magnitude == 0 else 2)
    return format_amount(value, code, unit=unit)


def _scaled(scaled: float) -> str:
    """Three significant figures, trailing zeros dropped: 34.8, 1.74, 250."""
    size = abs(scaled)
    text = _fixed(scaled, 0 if size >= 100 else (1 if size >= 10 else 2), grouped=False)
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def format_delta(value: float, code: str | None, *, unit: bool = True) -> str:
    """A signed amount, for "worse by" and "vs global" columns: ``+KRW 189``, ``-USD 32.00``."""
    if not math.isfinite(value):
        return "n/a"
    body = format_amount(abs(value), code, unit=False)
    if float(body.replace(",", "")) == 0:
        sign, body = "±", format_amount(0.0, code, unit=False)
    else:
        sign = "+" if value > 0 else "-"
    return f"{sign}{normalise_code(code)} {body}" if unit else f"{sign}{body}"


def format_percent(value: float, digits: int = 0) -> str:
    """A share as a percentage, rounded the way the report's slider rounds it.

    Not money, but it sits in the same table as money and is re-rendered by the same script: with
    Python's half-to-even rule a share of exactly 12.5% read "12%" in the table and "13%" in the
    live readout under it.
    """
    if not math.isfinite(value):
        return "n/a"
    return _fixed(value * 100.0, digits, grouped=False) + "%"
