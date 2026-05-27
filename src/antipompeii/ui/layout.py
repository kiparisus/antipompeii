from typing import Optional
import time
import random

# Terminal width used by every banner/section header in the CLI.
TERMINAL_WIDTH = 110


def print_banner(
    title: str,
    subtitle: str,
    author: str | None = None,
    version: str | None = None,
    width: int = TERMINAL_WIDTH,
):
    print()
    print("=" * width)
    print(f"\033[1m{title.center(width)}\033[0m")
    print(subtitle.center(width))
    if author:
        print(author.center(width))
    if version:
        print(f"Version {version}".center(width))
    print("=" * width)
    print()

def print_section(title: str, width: int = TERMINAL_WIDTH):
    print()
    print(f"{'─' * width}")
    print(f"  \033[1m{title}\033[0m")
    print(f"{'─' * width}")

def wait_for_enter(message: str = "Press Enter to continue...") -> None:
    input(message)

def print_progress(message: str, done: bool = False):
    symbol = "✓" if done else "→"
    print(f"{symbol} {message}")


def typing(text, min_delay=0.009, max_delay=0.03, punctuation_multiplier=3):
    for char in text:
        print(char, end='', flush=True)

        # Base random delay
        delay = random.uniform(min_delay, max_delay)

        # Multiply delay after punctuation
        if char in '.,!?;:':
            delay *= punctuation_multiplier

        time.sleep(delay)
    print()

def paint(text, delay=0.0002):
    typing(text, min_delay=delay, max_delay=delay, punctuation_multiplier=1)

def typereal(text, min_delay=0.01, max_delay=0.07, punctuation_multiplier=5):
    typing(text, min_delay=min_delay, max_delay=max_delay, punctuation_multiplier=punctuation_multiplier)

def get_user_input(prompt: str, default: Optional[str] = None) -> str:
    import sys

    if default:
        full_prompt = f"{prompt} [{default}]: "
    else:
        full_prompt = f"{prompt}: "

    typing(full_prompt)

    try:
        response = input().strip()  # Empty input() since prompt already displayed
        return response if response else (default or "")
    except EOFError:
        return default or ""
    except UnicodeDecodeError:
        # The terminal sent bytes that don't form valid UTF-8 (e.g. a split
        # multi-byte sequence or a non-UTF-8 keyboard encoding).  Read whatever
        # raw bytes remain on the line, decode with replacement so we get a
        # printable string, and fall back to the default if the result is empty.
        try:
            raw = sys.stdin.buffer.readline()
            response = raw.decode("utf-8", errors="replace").strip()
            return response if response else (default or "")
        except Exception:
            return default or ""
