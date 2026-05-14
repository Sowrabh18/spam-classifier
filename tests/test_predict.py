# ─────────────────────────────────────────────────────
# tests/test_predict.py
#
# Tests for the prediction function.
# Run with: pytest tests/
# ─────────────────────────────────────────────────────

from src.predict import predict


def test_spam_prediction():
    """
    Test that an obvious spam email
    gets classified as spam.
    """
    result = predict("Congratulations you won a FREE prize click now")
    assert result['prediction'] == 'spam'
    assert result['is_spam'] == True
    assert result['confidence'] > 0.5


def test_ham_prediction():
    """
    Test that an obvious ham email
    gets classified as ham.
    """
    result = predict("Hey are we still meeting for lunch tomorrow")
    assert result['prediction'] == 'ham'
    assert result['is_spam'] == False


def test_result_has_required_fields():
    """
    Test that the result always contains
    all required fields regardless of input.
    """
    result = predict("some random email text")
    assert 'prediction' in result
    assert 'confidence' in result
    assert 'is_spam' in result


def test_confidence_is_between_0_and_1():
    """
    Test that confidence score is always
    a valid probability between 0 and 1.
    """
    result = predict("Free money click here now")
    assert 0 <= result['confidence'] <= 1


def test_empty_string_handled():
    """
    Test that empty string doesn't crash
    the prediction function.
    """
    result = predict("")
    assert result['prediction'] in ['spam', 'ham']