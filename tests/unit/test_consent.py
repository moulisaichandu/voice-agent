from datetime import datetime, timedelta

from app.compliance.consent import has_valid_consent


def test_valid_explicit_consent():
    assert has_valid_consent("explicit", datetime.now() - timedelta(days=1)) is True


def test_valid_inferred_consent():
    assert has_valid_consent("inferred", datetime.now() - timedelta(hours=1)) is True


def test_missing_basis_is_invalid():
    assert has_valid_consent(None, datetime.now()) is False


def test_unrecognised_basis_is_invalid():
    assert has_valid_consent("verbal", datetime.now()) is False


def test_missing_timestamp_is_invalid():
    assert has_valid_consent("explicit", None) is False


def test_future_timestamp_is_invalid():
    assert has_valid_consent("explicit", datetime.now() + timedelta(days=1)) is False
