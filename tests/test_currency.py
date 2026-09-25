"""Money is printed with the decimals its currency actually has."""

from __future__ import annotations

import math

import pytest

from jeval.currency import (
    format_amount,
    format_compact,
    format_delta,
    minor_units,
    normalise_code,
)


def test_won_and_yen_have_no_minor_unit() -> None:
    assert format_amount(38_524_590.16, "KRW") == "KRW 38,524,590"
    assert format_amount(1_926.23, "KRW") == "KRW 1,926"
    assert format_amount(1_926.23, "JPY") == "JPY 1,926"


def test_two_and_three_decimal_currencies() -> None:
    assert format_amount(1_926.234, "USD") == "USD 1,926.23"
    assert format_amount(1_926.2345, "KWD") == "KWD 1,926.235"
    assert minor_units("EUR") == 2
    assert minor_units("BHD") == 3


def test_averages_below_the_minor_unit_keep_two_significant_figures() -> None:
    # A per-case average of under one won is a real result; rounding it to "KRW 0" would erase it.
    assert format_amount(0.4321, "KRW") == "KRW 0.43"
    assert format_amount(0.04321, "KRW") == "KRW 0.04"
    assert format_amount(12.46, "KRW") == "KRW 12"
    assert format_amount(0.004321, "USD") == "USD 0.0043"
    assert format_amount(0.25, "USD") == "USD 0.25"
    assert format_amount(0.0, "KRW") == "KRW 0"


def test_code_is_normalised_and_free_form_units_survive() -> None:
    assert normalise_code(" krw ") == "KRW"
    assert normalise_code("") == "USD"
    assert normalise_code("credits") == "credits"
    assert format_amount(12.5, "credits") == "credits 12.50"


def test_compact_uses_three_significant_figures() -> None:
    assert format_compact(34_754_098.36, "KRW") == "KRW 34.8M"
    assert format_compact(1_737.7, "KRW") == "KRW 1.74k"
    assert format_compact(189.0, "KRW") == "KRW 189"
    assert format_compact(250_000.0, None) == "250k"
    assert format_compact(890.0, None) == "890"


def test_delta_carries_its_sign_in_front_of_the_unit() -> None:
    assert format_delta(188.53, "KRW") == "+KRW 189"
    assert format_delta(-32.0, "USD") == "-USD 32.00"
    assert format_delta(0.00001, "KRW") == "±KRW 0"
    assert format_delta(0.0, "KRW") == "±KRW 0"


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_reads_na(value: float) -> None:
    assert format_amount(value, "KRW") == "n/a"
    assert format_compact(value, "KRW") == "n/a"
    assert format_delta(value, "KRW") == "n/a"


def test_halves_round_away_from_zero_like_the_browser_does() -> None:
    # The slider under the table formats with toFixed, which rounds a tie up; Python's own format
    # would round it to even and print a different number one line below.
    assert format_amount(0.125, "USD") == "USD 0.13"
    assert format_amount(2.5, "KRW") == "KRW 3"
    assert format_amount(1_926.5, "KRW") == "KRW 1,927"
    assert format_amount(-12.5, "KRW") == "KRW -13"
    assert format_compact(1_125.0, None) == "1.13k"
    assert format_compact(12_250.0, None) == "12.3k"


def test_a_compact_figure_moves_up_a_suffix_instead_of_reading_1000() -> None:
    assert format_compact(999_999.5, "KRW") == "KRW 1M"
    assert format_compact(999_500_000.0, None) == "1B"
    assert format_compact(999_499.0, None) == "999k"


def test_values_below_the_widened_window_read_as_the_plain_zero() -> None:
    assert format_amount(1e-9, "USD") == "USD 0.00"
    assert format_amount(0.0049, "KRW") == "KRW 0"
    assert format_compact(-0.001, None) == "0.00"
