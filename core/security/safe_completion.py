"""
Safe completion responses for security violations.
Provides standardized refusal messages and error handling.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def refuse(reason: Optional[str] = None) -> str:
    """
    Generate a safe refusal response.
    
    Args:
        reason: Optional specific reason for refusal (for logging)
        
    Returns:
        Standard refusal message
    """
    if reason:
        logger.warning(f"Request refused: {reason}")
    else:
        logger.warning("Request refused due to safety policy")
    
    return "⚠️ I'm sorry – I can't help with that."


def refuse_with_guidance(category: str) -> str:
    """
    Generate a refusal with category-specific guidance.
    
    Args:
        category: Type of violation (moderation, rate_limit, etc.)
        
    Returns:
        Refusal message with appropriate guidance
    """
    guidance_messages = {
        'moderation': (
            "⚠️ I'm sorry – I can't help with that. "
            "Please ensure your request follows our content policy."
        ),
        'rate_limit': (
            "⚠️ You're sending requests too quickly. "
            "Please wait a moment before trying again."
        ),
        'pii': (
            "⚠️ I noticed your message may contain personal information. "
            "Please remove any sensitive details and try again."
        ),
        'token_limit': (
            "⚠️ Your request is too long. "
            "Please try a shorter message or break it into parts."
        ),
        'api_error': (
            "⚠️ I'm experiencing technical difficulties. "
            "Please try again in a moment."
        )
    }
    
    message = guidance_messages.get(category, refuse())
    logger.warning(f"Request refused - {category}: {message}")
    return message


def safe_error_response(error: Exception, include_details: bool = False) -> str:
    """
    Generate a safe error response that doesn't leak sensitive information.
    
    Args:
        error: The exception that occurred
        include_details: Whether to include error details (only in development)
        
    Returns:
        Safe error message for user display
    """
    # Log the full error for debugging
    logger.error(f"Error generating response: {type(error).__name__}: {error}")
    
    # Return safe message to user
    if include_details:
        return f"⚠️ An error occurred: {type(error).__name__}"
    else:
        return "⚠️ I'm sorry – I encountered an error while processing your request."


def format_system_status(status: str, details: str = "") -> str:
    """
    Format system status messages safely.
    
    Args:
        status: Status level (info, warning, error)
        details: Additional details to include
        
    Returns:
        Formatted status message
    """
    status_emojis = {
        'info': 'ℹ️',
        'warning': '⚠️', 
        'error': '❌',
        'success': '✅'
    }
    
    emoji = status_emojis.get(status, 'ℹ️')
    
    if details:
        return f"{emoji} {details}"
    else:
        return f"{emoji} System notification"


if __name__ == "__main__":
    """CLI demo for safe completion responses."""
    
    print("🛡️ Safe Completion Demo")
    print("=" * 50)
    
    # Test basic refusal
    print("\n1. Basic refusal:")
    print(refuse())
    
    # Test refusal with reason
    print("\n2. Refusal with reason:")
    print(refuse("Test violation detected"))
    
    # Test category-specific guidance
    print("\n3. Category-specific guidance:")
    categories = ['moderation', 'rate_limit', 'pii', 'token_limit', 'api_error']
    
    for category in categories:
        print(f"\n   {category.upper()}:")
        print(f"   {refuse_with_guidance(category)}")
    
    # Test error responses
    print("\n4. Error responses:")
    try:
        raise ValueError("Test error message")
    except Exception as e:
        print(f"\n   Safe (production): {safe_error_response(e)}")
        print(f"   With details (dev): {safe_error_response(e, include_details=True)}")
    
    # Test status formatting
    print("\n5. Status formatting:")
    statuses = [
        ('info', 'System running normally'),
        ('warning', 'Rate limit approaching'),
        ('error', 'Service temporarily unavailable'),
        ('success', 'Request processed successfully')
    ]
    
    for status, message in statuses:
        print(f"   {format_system_status(status, message)}")
    
    print("\n✅ Safe completion demo completed")