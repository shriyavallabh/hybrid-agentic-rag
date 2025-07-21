"""
PII (Personally Identifiable Information) redaction utilities.
Scrubs emails, phone numbers, SSNs, credit cards, and tokens from text.
"""

import re
from typing import List, Tuple


# Compiled regex patterns for PII detection
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')

# Phone number patterns (US and international)
PHONE_PATTERNS = [
    re.compile(r'\b\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),  # US format
    re.compile(r'\b\+[1-9]\d{1,14}\b'),  # International format
]

# Government ID patterns
SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')  # US SSN
PAN_PATTERN = re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b')  # Indian PAN
AADHAAR_PATTERN = re.compile(r'\b\d{4}\s\d{4}\s\d{4}\b')  # Indian Aadhaar

# Credit card pattern (16 digits)
CREDIT_CARD_PATTERN = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')

# Token patterns
HEX32_PATTERN = re.compile(r'\b[a-fA-F0-9]{32}\b')  # 32-character hex
HEX64_PATTERN = re.compile(r'\b[a-fA-F0-9]{64}\b')  # 64-character hex
JWT_PATTERN = re.compile(r'\beyJ[A-Za-z0-9+/=]+\.[A-Za-z0-9+/=]+\.[A-Za-z0-9+/=_-]+\b')  # JWT tokens

# IP Address pattern
IP_PATTERN = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')

# All patterns list for easy iteration
ALL_PATTERNS = [
    EMAIL_PATTERN,
    SSN_PATTERN,
    PAN_PATTERN, 
    AADHAAR_PATTERN,
    CREDIT_CARD_PATTERN,
    HEX32_PATTERN,
    HEX64_PATTERN,
    JWT_PATTERN,
    IP_PATTERN,
] + PHONE_PATTERNS

REDACTION_TOKEN = "[PII-REDACTED]"


def _luhn_check(card_number: str) -> bool:
    """
    Validate credit card number using Luhn algorithm.
    
    Args:
        card_number: Card number string (digits only)
        
    Returns:
        True if valid according to Luhn algorithm
    """
    # Remove non-digits
    digits = [int(d) for d in card_number if d.isdigit()]
    
    if len(digits) != 16:
        return False
    
    # Luhn algorithm
    checksum = 0
    reverse_digits = digits[::-1]
    
    for i, digit in enumerate(reverse_digits):
        if i % 2 == 1:  # Every second digit from right
            digit *= 2
            if digit > 9:
                digit = digit // 10 + digit % 10
        checksum += digit
    
    return checksum % 10 == 0


def _validate_credit_card(match: re.Match) -> bool:
    """
    Validate if a credit card match is a real credit card number.
    
    Args:
        match: Regex match object
        
    Returns:
        True if appears to be valid credit card
    """
    card_number = match.group()
    # Remove separators
    clean_number = re.sub(r'[-\s]', '', card_number)
    
    # Check if it's all digits and passes Luhn check
    if clean_number.isdigit() and len(clean_number) == 16:
        return _luhn_check(clean_number)
    
    return False


def redact(text: str) -> str:
    """
    Redact PII from text, replacing with [PII-REDACTED] tokens.
    
    Args:
        text: Input text that may contain PII
        
    Returns:
        Text with PII replaced by redaction tokens
    """
    if not text:
        return text
    
    result = text
    
    # Special handling for credit cards (validate with Luhn)
    def replace_valid_cards(match):
        if _validate_credit_card(match):
            return REDACTION_TOKEN
        return match.group()  # Keep invalid card numbers
    
    result = CREDIT_CARD_PATTERN.sub(replace_valid_cards, result)
    
    # Apply other patterns
    for pattern in ALL_PATTERNS:
        if pattern == CREDIT_CARD_PATTERN:
            continue  # Already handled above
        result = pattern.sub(REDACTION_TOKEN, result)
    
    return result


def detect_pii_types(text: str) -> List[str]:
    """
    Detect what types of PII are present in text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        List of PII types found
    """
    found_types = []
    
    if EMAIL_PATTERN.search(text):
        found_types.append("email")
    
    for phone_pattern in PHONE_PATTERNS:
        if phone_pattern.search(text):
            found_types.append("phone")
            break
    
    if SSN_PATTERN.search(text):
        found_types.append("ssn")
    
    if PAN_PATTERN.search(text):
        found_types.append("pan")
    
    if AADHAAR_PATTERN.search(text):
        found_types.append("aadhaar")
    
    # Check credit cards with validation
    for match in CREDIT_CARD_PATTERN.finditer(text):
        if _validate_credit_card(match):
            found_types.append("credit_card")
            break
    
    if HEX32_PATTERN.search(text):
        found_types.append("hex32_token")
    
    if HEX64_PATTERN.search(text):
        found_types.append("hex64_token")
    
    if JWT_PATTERN.search(text):
        found_types.append("jwt_token")
    
    if IP_PATTERN.search(text):
        found_types.append("ip_address")
    
    return found_types


if __name__ == "__main__":
    """CLI demo for PII redaction."""
    import sys
    
    # Test cases
    test_cases = [
        "Contact me at john.doe@example.com or call 555-123-4567",
        "My SSN is 123-45-6789 and PAN is ABCDE1234F",
        "Card number: 4532-1234-5678-9012 (valid Luhn)",
        "Card number: 1234-5678-9012-3456 (invalid)",
        "Token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature",
        "API key: a1b2c3d4e5f6789012345678901234567890abcd and secret: 1234567890abcdef1234567890abcdef12345678",
        "Server IP: 192.168.1.100 on port 8080"
    ]
    
    if len(sys.argv) > 1 and sys.argv[1] == "--stdin":
        # Read from stdin
        for line in sys.stdin:
            print(redact(line.rstrip()))
    else:
        # Run test cases
        print("🔍 PII Redaction Demo")
        print("=" * 50)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest {i}:")
            print(f"Original:  {test_case}")
            
            detected = detect_pii_types(test_case)
            if detected:
                print(f"Detected:  {', '.join(detected)}")
            
            redacted = redact(test_case)
            print(f"Redacted:  {redacted}")
        
        print(f"\n✅ All tests completed. PII replaced with '{REDACTION_TOKEN}'")