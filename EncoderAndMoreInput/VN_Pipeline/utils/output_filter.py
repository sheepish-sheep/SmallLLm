"""
Output filtering for seq2seq paraphrase model.

This module provides filters to reject low-quality outputs from the paraphrase model.
Each filter returns True if the output PASSES (is good), False if it should be REJECTED.

Usage:
    from EncoderAndMoreInput.VN_Pipeline.utils.output_filter import OutputFilter

    filter = OutputFilter()
    result = filter.filter("I love you.", "I adore you greatly.")

    if result.passed:
        print(f"Good output: {result.output}")
    else:
        print(f"Rejected: {result.rejection_reasons}")

To customize: Edit the filter functions or add your own in the CUSTOM FILTERS section.
"""

import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class FilterResult:
    """Result of filtering an output."""
    passed: bool
    output: str
    input_text: str
    rejection_reasons: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


# =============================================================================
# BUILT-IN FILTERS
# =============================================================================

def filter_min_length(input_text: str, output: str, min_chars: int = 5) -> tuple[bool, str]:
    if len(output.strip()) < min_chars:
        return False, f"Output too short ({len(output)} chars, min {min_chars})"
    return True, ""


def filter_max_length(input_text: str, output: str, max_ratio: float = 3.0) -> tuple[bool, str]:
    input_len = max(len(input_text), 1)
    ratio = len(output) / input_len
    if ratio > max_ratio:
        return False, f"Output too long (ratio {ratio:.1f}, max {max_ratio})"
    return True, ""


def filter_repetition(input_text: str, output: str, max_repeat: int = 3) -> tuple[bool, str]:
    words = output.lower().split()
    if len(words) < 2:
        return True, ""
    repeat_count = 1
    for i in range(1, len(words)):
        if words[i] == words[i-1]:
            repeat_count += 1
            if repeat_count > max_repeat:
                return False, f"Word '{words[i]}' repeated {repeat_count}+ times consecutively"
        else:
            repeat_count = 1
    from collections import Counter
    word_counts = Counter(words)
    total_words = len(words)
    for word, count in word_counts.items():
        if len(word) > 2 and count > max(3, total_words * 0.4):
            return False, f"Word '{word}' appears {count} times ({count/total_words*100:.0f}% of output)"
    return True, ""


def filter_incomplete_sentence(input_text: str, output: str) -> tuple[bool, str]:
    output = output.strip()
    if not output:
        return False, "Empty output"
    if re.search(r'\s[a-zA-Z]$', output):
        return False, "Output appears cut off mid-word"
    incomplete_patterns = [
        r'\s(the|a|an|to|of|in|on|at|for|and|or|but|with|is|are|was|were)\s*$',
        r'[,;:]\s*$', 
        r'\s(I|you|he|she|it|we|they)\s*$',
    ]
    for pattern in incomplete_patterns:
        if re.search(pattern, output, re.IGNORECASE):
            return False, "Output appears incomplete"
    return True, ""


def filter_nonsense_patterns(input_text: str, output: str) -> tuple[bool, str]:
    if re.search(r'[.!?,;:]{3,}', output):
        return False, "Excessive punctuation"
    gibberish_patterns = [
        r'[bcdfghjklmnpqrstvwxz]{5,}',
        r'(.)\1{3,}',
    ]
    for pattern in gibberish_patterns:
        if re.search(pattern, output, re.IGNORECASE):
            return False, "Contains gibberish patterns"
    words = output.split()
    for word in words:
        clean_word = re.sub(r'[^a-zA-Z]', '', word)
        if len(clean_word) > 4 and not re.search(r'[aeiouAEIOU]', clean_word):
            return False, f"Nonsense word detected: '{word}'"
    return True, ""


def filter_meaning_preserved(input_text: str, output: str, min_overlap: float = 0.1) -> tuple[bool, str]:
    input_words = set(re.findall(r'\b\w+\b', input_text.lower()))
    output_words = set(re.findall(r'\b\w+\b', output.lower()))
    stop_words = {'i', 'me', 'my', 'you', 'your', 'he', 'she', 'it', 'we', 'they',
                  'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
                  'to', 'of', 'in', 'on', 'at', 'for', 'and', 'or', 'but', 'with',
                  'this', 'that', 'these', 'those', 'do', 'does', 'did', 'have', 'has'}
    input_content = input_words - stop_words
    output_content = output_words - stop_words
    if not input_content:
        return True, ""
    overlap = len(input_content & output_content) / len(input_content)
    if overlap < min_overlap and len(input_content) > 3:
        return False, f"Low word overlap ({overlap*100:.0f}%), meaning may be lost"
    return True, ""


# =============================================================================
# CUSTOM FILTERS - ADD YOUR OWN HERE
# =============================================================================

def filter_custom_blocklist(input_text: str, output: str) -> tuple[bool, str]:
    """
    Reject outputs containing blocked words/phrases.

    TODO: Add your own blocked words here.

    Returns:
        (passed, reason)
    """
    # Add words/phrases you want to block
    blocklist = [
        # Examples - uncomment or add your own:
        # "error",
        # "undefined",
        # "null",
    ]

    output_lower = output.lower()
    for blocked in blocklist:
        if blocked.lower() in output_lower:
            return False, f"Contains blocked word: '{blocked}'"

    return True, ""


def filter_grammar_check(input_text: str, output: str) -> tuple[bool, str]:
    """
    STUB: Check grammar quality.

    TODO: Implement using a grammar checker library like:
    - language_tool_python
    - grammarbot
    - Or a simple rule-based approach

    Example with language_tool_python:
        import language_tool_python
        tool = language_tool_python.LanguageTool('en-US')
        matches = tool.check(output)
        if len(matches) > threshold:
            return False, f"Grammar errors: {len(matches)}"

    Returns:
        (passed, reason)
    """
    # Currently passes everything - implement your own logic
    return True, ""


def filter_semantic_similarity(input_text: str, output: str, min_similarity: float = 0.3) -> tuple[bool, str]:
    """
    STUB: Check semantic similarity between input and output.

    TODO: Implement using sentence embeddings like:
    - sentence-transformers
    - OpenAI embeddings
    - spaCy similarity

    Example with sentence-transformers:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer('all-MiniLM-L6-v2')
        emb1 = model.encode(input_text)
        emb2 = model.encode(output)
        similarity = util.cos_sim(emb1, emb2).item()
        if similarity < min_similarity:
            return False, f"Low semantic similarity: {similarity:.2f}"

    Returns:
        (passed, reason)
    """
    # Currently passes everything - implement your own logic
    return True, ""


# =============================================================================
# MAIN FILTER CLASS
# =============================================================================

class OutputFilter:
    """
    Combines multiple filters to validate paraphrase outputs.

    Usage:
        filter = OutputFilter()
        result = filter.filter("input", "output")

        # Or generate multiple and pick best:
        candidates = ["output1", "output2", "output3"]
        best = filter.filter_best("input", candidates)
    """

    def __init__(self,
                 min_length: int = 5,
                 max_length_ratio: float = 3.0,
                 max_repetition: int = 3,
                 check_incomplete: bool = True,
                 check_nonsense: bool = True,
                 check_meaning: bool = True,
                 min_word_overlap: float = 0.1,
                 use_custom_blocklist: bool = True,
                 use_grammar_check: bool = False,  # Disabled by default (stub)
                 use_semantic_check: bool = False,  # Disabled by default (stub)
                 ):
        """
        Initialize filter with settings.

        Args:
            min_length: Minimum output character length
            max_length_ratio: Maximum output/input length ratio
            max_repetition: Maximum consecutive word repetitions allowed
            check_incomplete: Check for cut-off sentences
            check_nonsense: Check for gibberish patterns
            check_meaning: Check word overlap with input
            min_word_overlap: Minimum word overlap ratio
            use_custom_blocklist: Use custom blocklist filter
            use_grammar_check: Use grammar check (stub - implement yourself)
            use_semantic_check: Use semantic similarity (stub - implement yourself)
        """
        self.filters: list[tuple[str, Callable]] = []

        # Add enabled filters
        if min_length > 0:
            self.filters.append(("min_length",
                lambda i, o, ml=min_length: filter_min_length(i, o, ml)))

        if max_length_ratio > 0:
            self.filters.append(("max_length",
                lambda i, o, mr=max_length_ratio: filter_max_length(i, o, mr)))

        if max_repetition > 0:
            self.filters.append(("repetition",
                lambda i, o, mr=max_repetition: filter_repetition(i, o, mr)))

        if check_incomplete:
            self.filters.append(("incomplete", filter_incomplete_sentence))

        if check_nonsense:
            self.filters.append(("nonsense", filter_nonsense_patterns))

        if check_meaning:
            self.filters.append(("meaning",
                lambda i, o, mo=min_word_overlap: filter_meaning_preserved(i, o, mo)))

        if use_custom_blocklist:
            self.filters.append(("blocklist", filter_custom_blocklist))

        if use_grammar_check:
            self.filters.append(("grammar", filter_grammar_check))

        if use_semantic_check:
            self.filters.append(("semantic", filter_semantic_similarity))

    def filter(self, input_text: str, output: str) -> FilterResult:
        """
        Run all filters on an output.

        Args:
            input_text: Original input text
            output: Generated paraphrase output

        Returns:
            FilterResult with passed status and rejection reasons
        """
        result = FilterResult(
            passed=True,
            output=output,
            input_text=input_text,
            rejection_reasons=[],
            scores={}
        )

        for filter_name, filter_fn in self.filters:
            try:
                passed, reason = filter_fn(input_text, output)
                result.scores[filter_name] = 1.0 if passed else 0.0

                if not passed:
                    result.passed = False
                    result.rejection_reasons.append(f"[{filter_name}] {reason}")
            except Exception as e:
                # Filter failed - log but don't reject
                result.scores[filter_name] = -1.0  # Error indicator

        return result

    def filter_best(self, input_text: str, candidates: list[str]) -> FilterResult | None:
        """
        Filter multiple candidates and return the best passing one.

        Args:
            input_text: Original input text
            candidates: List of generated outputs to choose from

        Returns:
            Best FilterResult that passed, or None if all rejected
        """
        passing = []

        for candidate in candidates:
            result = self.filter(input_text, candidate)
            if result.passed:
                passing.append(result)

        if not passing:
            return None

        # Return shortest passing output (usually cleaner)
        return min(passing, key=lambda r: len(r.output))

    def filter_with_fallback(self, input_text: str, candidates: list[str]) -> FilterResult:
        """
        Filter candidates, returning best passing one or least-bad if all fail.

        Args:
            input_text: Original input text
            candidates: List of generated outputs

        Returns:
            Best passing result, or least-rejected result if all fail
        """
        results = [self.filter(input_text, c) for c in candidates]

        # Try to find a passing one
        passing = [r for r in results if r.passed]
        if passing:
            return min(passing, key=lambda r: len(r.output))

        # All failed - return one with fewest rejections
        return min(results, key=lambda r: len(r.rejection_reasons))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_filter(input_text: str, output: str) -> bool:
    """Quick check if output passes basic filters."""
    f = OutputFilter()
    return f.filter(input_text, output).passed


def filter_and_explain(input_text: str, output: str) -> None:
    """Print detailed filter results for debugging."""
    f = OutputFilter()
    result = f.filter(input_text, output)

    print(f"Input:  {input_text}")
    print(f"Output: {output}")
    print(f"Passed: {result.passed}")

    if result.rejection_reasons:
        print("Rejection reasons:")
        for reason in result.rejection_reasons:
            print(f"  - {reason}")

    print("Filter scores:")
    for name, score in result.scores.items():
        status = "PASS" if score == 1.0 else "FAIL" if score == 0.0 else "ERROR"
        print(f"  {name}: {status}")


# =============================================================================
# MAIN - Test the filters
# =============================================================================

if __name__ == "__main__":
    # Test examples
    test_cases = [
        ("I love you so much.", "I adore you greatly."),  # Good
        ("I love you so much.", "I I I I love love love"),  # Repetition
        ("I love you so much.", "xyz"),
        ("Hello there.", "Hello there my friend how are you doing today I hope you are well and having a great time with your family and friends"),  # Too long
        ("What are you doing?", "The moon shines bright"),  # Meaning lost
        ("Test input.", "This is a test with bbbbbbb nonsense"),  # Gibberish
        ("Hello world.", "Hello there, this is a"),  # Incomplete
    ]

    print("=" * 60)
    print("OUTPUT FILTER TEST")
    print("=" * 60)

    for input_text, output in test_cases:
        print()
        filter_and_explain(input_text, output)
        print("-" * 40)
