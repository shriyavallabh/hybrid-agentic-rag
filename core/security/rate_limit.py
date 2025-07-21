"""
Rate limiting decorator using in-memory TTL tracking.
Enforces request limits per IP address with time-based cleanup.
"""

import time
import logging
from collections import defaultdict, deque
from functools import wraps
from typing import Dict, Deque, Callable, Any

import streamlit as st

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""
    pass


# Rate limiting configuration
REQUEST_LIMIT = 10  # requests per window
TIME_WINDOW = 60   # seconds (1 minute)

# Global tracking storage
_request_history: Dict[str, Deque[float]] = defaultdict(deque)
_last_cleanup = time.time()
CLEANUP_INTERVAL = 300  # Clean up every 5 minutes


def _get_client_ip() -> str:
    """
    Get client IP address from Streamlit session state or headers.
    
    Returns:
        Client IP address string
    """
    # Try to get IP from session state first
    if hasattr(st, 'session_state') and 'client_ip' in st.session_state:
        return st.session_state.client_ip
    
    # Fallback to getting from headers if available
    try:
        # In Streamlit Cloud or when behind proxy
        headers = st.context.headers if hasattr(st, 'context') else {}
        client_ip = (
            headers.get('x-forwarded-for', '').split(',')[0].strip() or
            headers.get('x-real-ip', '') or
            headers.get('remote-addr', '') or
            '127.0.0.1'  # Default fallback
        )
        
        # Store in session state for future use
        if hasattr(st, 'session_state'):
            st.session_state.client_ip = client_ip
            
        return client_ip
        
    except Exception as e:
        logger.warning(f"Could not determine client IP: {e}")
        return '127.0.0.1'  # Safe fallback


def _cleanup_old_requests() -> None:
    """
    Clean up old request history entries to prevent memory leaks.
    """
    global _last_cleanup
    
    current_time = time.time()
    
    # Only cleanup every CLEANUP_INTERVAL seconds
    if current_time - _last_cleanup < CLEANUP_INTERVAL:
        return
    
    cutoff_time = current_time - TIME_WINDOW
    removed_ips = []
    
    for ip, history in _request_history.items():
        # Remove old requests outside time window
        while history and history[0] <= cutoff_time:
            history.popleft()
        
        # Remove IP entries with no recent requests
        if not history:
            removed_ips.append(ip)
    
    # Clean up empty IP entries
    for ip in removed_ips:
        del _request_history[ip]
    
    _last_cleanup = current_time
    
    if removed_ips:
        logger.debug(f"Cleaned up rate limit history for {len(removed_ips)} IPs")


def _check_rate_limit(client_ip: str) -> None:
    """
    Check if client IP is within rate limits.
    
    Args:
        client_ip: Client IP address
        
    Raises:
        RateLimitError: If rate limit exceeded
    """
    current_time = time.time()
    cutoff_time = current_time - TIME_WINDOW
    
    # Get request history for this IP
    history = _request_history[client_ip]
    
    # Remove old requests outside time window
    while history and history[0] <= cutoff_time:
        history.popleft()
    
    # Check if we're at the limit
    if len(history) >= REQUEST_LIMIT:
        oldest_request = history[0]
        wait_time = TIME_WINDOW - (current_time - oldest_request)
        
        logger.warning(f"Rate limit exceeded for IP {client_ip}")
        raise RateLimitError(
            f"Too many requests – wait {wait_time:.0f} seconds. "
            f"Limit: {REQUEST_LIMIT} requests per {TIME_WINDOW} seconds."
        )
    
    # Add current request to history
    history.append(current_time)


def enforce(func: Callable) -> Callable:
    """
    Decorator to enforce rate limiting on a function.
    
    Args:
        func: Function to rate limit
        
    Returns:
        Wrapped function with rate limiting
        
    Raises:
        RateLimitError: If rate limit exceeded
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Periodic cleanup
        _cleanup_old_requests()
        
        # Get client IP
        client_ip = _get_client_ip()
        
        # Check rate limit
        _check_rate_limit(client_ip)
        
        # Call original function
        return func(*args, **kwargs)
    
    return wrapper


def get_rate_limit_status(client_ip: str = None) -> Dict[str, Any]:
    """
    Get current rate limit status for an IP.
    
    Args:
        client_ip: IP address to check (defaults to current client)
        
    Returns:
        Dictionary with rate limit status information
    """
    if client_ip is None:
        client_ip = _get_client_ip()
    
    current_time = time.time()
    cutoff_time = current_time - TIME_WINDOW
    
    history = _request_history[client_ip]
    
    # Count recent requests
    recent_requests = 0
    oldest_request_time = None
    
    for request_time in history:
        if request_time > cutoff_time:
            recent_requests += 1
            if oldest_request_time is None:
                oldest_request_time = request_time
    
    remaining = max(0, REQUEST_LIMIT - recent_requests)
    
    # Calculate reset time
    reset_time = 0
    if oldest_request_time and recent_requests >= REQUEST_LIMIT:
        reset_time = oldest_request_time + TIME_WINDOW - current_time
    
    return {
        'client_ip': client_ip,
        'requests_made': recent_requests,
        'requests_remaining': remaining,
        'limit': REQUEST_LIMIT,
        'window_seconds': TIME_WINDOW,
        'reset_in_seconds': max(0, reset_time),
        'limited': recent_requests >= REQUEST_LIMIT
    }


def reset_rate_limit(client_ip: str = None) -> None:
    """
    Reset rate limit for a specific IP (admin function).
    
    Args:
        client_ip: IP address to reset (defaults to current client)
    """
    if client_ip is None:
        client_ip = _get_client_ip()
    
    if client_ip in _request_history:
        del _request_history[client_ip]
        logger.info(f"Rate limit reset for IP {client_ip}")


if __name__ == "__main__":
    """CLI demo for rate limiting."""
    import sys
    
    # Mock function for testing
    @enforce
    def test_function(msg: str) -> str:
        return f"Processed: {msg}"
    
    # Simulate Streamlit session state
    class MockSessionState:
        def __init__(self):
            self.client_ip = "127.0.0.1"
    
    if not hasattr(st, 'session_state'):
        st.session_state = MockSessionState()
    
    print("🚦 Rate Limiting Demo")
    print("=" * 50)
    
    # Test normal operation
    print("\n📝 Testing normal operation:")
    for i in range(5):
        try:
            result = test_function(f"Message {i+1}")
            print(f"✅ {result}")
            status = get_rate_limit_status()
            print(f"   Remaining: {status['requests_remaining']}/{status['limit']}")
        except RateLimitError as e:
            print(f"❌ {e}")
    
    # Test rate limiting
    print(f"\n⚡ Testing rate limit (attempting {REQUEST_LIMIT + 2} rapid requests):")
    for i in range(REQUEST_LIMIT + 2):
        try:
            result = test_function(f"Rapid message {i+1}")
            print(f"✅ {result}")
        except RateLimitError as e:
            print(f"❌ Request {i+1}: {e}")
            break
    
    # Show final status
    status = get_rate_limit_status()
    print(f"\n📊 Final status: {status['requests_made']}/{status['limit']} requests used")
    if status['limited']:
        print(f"⏰ Reset in {status['reset_in_seconds']:.1f} seconds")
    
    print("✅ Rate limiting demo completed")