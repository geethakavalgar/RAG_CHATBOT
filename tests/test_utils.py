from src.utils import clean_text

def test_clean_text():
    assert clean_text("hello   world\nagain") == "hello world again"
