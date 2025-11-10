import logging

def boxed_log(message: str, logger: logging.Logger = None, level: str = "info"):
    """
    Prints a message with a top and bottom border only.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    lines = message.split("\n")
    max_len = max(len(line) for line in lines)
    
    # Top and bottom borders
    border = "═" * (max_len + 4)  # 2 spaces padding on each side
    top = f"╔{border}╗"
    bottom = f"╚{border}╝"

    # Build message with padding, no side borders
    boxed_message = top + "\n"
    for line in lines:
        boxed_message += f"  {line.ljust(max_len)}  \n"
    boxed_message += bottom

    # Start on a new line after the logger prefix
    boxed_message = "\n" + boxed_message

    getattr(logger, level.lower())(boxed_message)
