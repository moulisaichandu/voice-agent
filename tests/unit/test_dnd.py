import pytest

from app.compliance.dnd import is_dnd, normalize_phone_e164


@pytest.mark.parametrize("raw,expected", [
    ("9876543210", "+919876543210"),          # bare 10-digit national
    ("09876543210", "+919876543210"),          # leading trunk 0
    ("+919876543210", "+919876543210"),        # already E.164
    ("919876543210", "+919876543210"),         # country code, no '+'
    ("+91 98765 43210", "+919876543210"),      # spaces stripped
    ("98765-43210", "+919876543210"),          # dashes stripped
    (None, None),
    ("", None),
    ("   ", None),
    ("12345", None),                           # too short — an ID, not a phone
    ("+15551234567", None),                    # non-India +country — out of scope
    ("987654321", None),                       # 9 digits — not a valid national number
])
def test_normalize_phone_e164(raw, expected):
    assert normalize_phone_e164(raw) == expected


def test_is_dnd_membership():
    dnd_set = {"+919876543210", "+919812345678"}
    assert is_dnd("+919876543210", dnd_set) is True
    assert is_dnd("+919711111111", dnd_set) is False
