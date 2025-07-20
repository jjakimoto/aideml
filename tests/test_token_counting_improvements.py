"""Tests for improved token counting in Claude Code backend."""

import pytest
from aide.backend.backend_claude_code import _estimate_claude_tokens


class TestClaudeTokenEstimation:
    """Test the improved Claude token estimation function."""
    
    def test_empty_text(self):
        """Test token estimation for empty text."""
        assert _estimate_claude_tokens("") == 0
        assert _estimate_claude_tokens(None) == 0
    
    def test_simple_text(self):
        """Test token estimation for simple text."""
        # "Hello world" - 11 chars / 3.7 ≈ 3 tokens
        result = _estimate_claude_tokens("Hello world")
        assert 2 <= result <= 4
    
    def test_text_with_punctuation(self):
        """Test token estimation accounts for punctuation."""
        # Punctuation should increase token count
        text_no_punct = "Hello world how are you"
        text_with_punct = "Hello, world! How are you?"
        
        tokens_no_punct = _estimate_claude_tokens(text_no_punct)
        tokens_with_punct = _estimate_claude_tokens(text_with_punct)
        
        # Text with punctuation should have more tokens
        assert tokens_with_punct > tokens_no_punct
    
    def test_text_with_newlines(self):
        """Test token estimation accounts for newlines."""
        text_single_line = "This is a test sentence for token counting"
        text_multi_line = "This is a test\nsentence for\ntoken counting"
        
        tokens_single = _estimate_claude_tokens(text_single_line)
        tokens_multi = _estimate_claude_tokens(text_multi_line)
        
        # Text with newlines should have more tokens
        assert tokens_multi > tokens_single
    
    def test_code_snippet(self):
        """Test token estimation for code snippets."""
        code = '''def hello_world():
    print("Hello, world!")
    return True'''
        
        tokens = _estimate_claude_tokens(code)
        # Code with special chars, newlines, punctuation should have reasonable count
        # ~50 chars / 3.7 + adjustments ≈ 15-20 tokens
        assert 10 <= tokens <= 25
    
    def test_long_text(self):
        """Test token estimation for longer text."""
        # Create a text with ~1000 characters
        long_text = "This is a longer piece of text. " * 30
        tokens = _estimate_claude_tokens(long_text)
        
        # ~960 chars / 3.7 ≈ 260 tokens (plus adjustments)
        assert 250 <= tokens <= 280
    
    def test_json_text(self):
        """Test token estimation for JSON-like text."""
        json_text = '''{
    "name": "test",
    "value": 123,
    "items": ["a", "b", "c"],
    "nested": {
        "key": "value"
    }
}'''
        tokens = _estimate_claude_tokens(json_text)
        # JSON has lots of special characters and structure
        # Should be higher than simple text of same length
        char_count = len(json_text)
        simple_estimate = char_count / 3.7
        assert tokens > simple_estimate
    
    def test_comparison_with_old_method(self):
        """Compare new estimation with old word-based method."""
        text = "The quick brown fox jumps over the lazy dog"
        
        # Old method: word count * 1.3
        old_estimate = len(text.split()) * 1.3
        new_estimate = _estimate_claude_tokens(text)
        
        # New method should give different (likely higher) estimate
        # since Claude produces more tokens than simple word count suggests
        assert new_estimate != int(old_estimate)
    
    def test_unicode_text(self):
        """Test token estimation handles unicode text."""
        unicode_text = "Hello 世界! How are you? 🌍"
        tokens = _estimate_claude_tokens(unicode_text)
        # Should handle unicode gracefully
        assert tokens > 0
    
    def test_realistic_prompt(self):
        """Test token estimation for a realistic AI prompt."""
        prompt = """System: You are a helpful AI assistant.

User: Please analyze the following code and identify any potential issues:

```python
def calculate_average(numbers):
    total = sum(numbers)
    return total / len(numbers)
```

What happens if the input list is empty?"""
        
        tokens = _estimate_claude_tokens(prompt)
        # This prompt has ~250 chars, multiple newlines, code block
        # Should be reasonable token count
        assert 60 <= tokens <= 90


if __name__ == "__main__":
    pytest.main([__file__, "-v"])