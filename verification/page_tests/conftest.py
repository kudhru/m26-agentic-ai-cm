"""
Session-scoped Playwright fixtures shared across all page test modules.

A single sync_playwright() / browser instance is created once for the whole
test session.  Individual test modules get page objects from this browser.

Background: anyio (installed as a transitive dep) creates an asyncio event
loop for the test session.  Opening a second sync_playwright() context inside
that loop raises "Playwright Sync API inside the asyncio loop."  Using one
session-scoped context avoids the collision.
"""

import pytest
from playwright.sync_api import sync_playwright


@pytest.fixture(scope="session")
def pw_browser():
    """One headless Chromium browser for the entire pytest session."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()
