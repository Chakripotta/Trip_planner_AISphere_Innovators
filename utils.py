def format_list(items: list[str]) -> str:
    """Format a list of strings into markdown bullets."""
    return "\n".join([f"- {item}" for item in items])
