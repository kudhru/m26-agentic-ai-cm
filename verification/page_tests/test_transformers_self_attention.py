"""
Browser-level smoke tests for topics/transformers-self-attention.html.

Uses Playwright (headless Chromium) to verify:
  1. Page loads without JavaScript console errors
  2. Math verification badge reaches passing state
  3. All six walkthrough step panels have content after load
  4. Scaling slider changes step 4 values and step 5 heatmap
  5. Head tab switching changes step 1/2 content
  6. Causal mask toggle changes step 4 values and insight text
  7. Font size is 17px for .step-desc elements (regression guard)
  8. Real attention section: graceful error state on file:// (no fetch)
  9. Real attention section: full load via page.route() (no server needed)

Run:
  cd <project-root>
  bash verification/run_page_tests.sh
"""

import pytest
from pathlib import Path

PROJECT_ROOT    = (Path(__file__).parent.parent.parent).resolve()
HTML_PATH       = PROJECT_ROOT / "topics" / "transformers-self-attention.html"
PAGE_URL        = f"file://{HTML_PATH}"
REAL_ATT_PATH   = PROJECT_ROOT / "verification" / "transformers-self-attention" / "real_attention.json"


@pytest.fixture(scope="module")
def page_with_errors(pw_browser):
    """Open the page once and share across all tests in this module."""
    ctx  = pw_browser.new_context()
    page = ctx.new_page()

    js_errors = []
    page.on("pageerror", lambda e: js_errors.append(str(e)))

    page.goto(PAGE_URL)

    # Wait for our own verification to finish: badge leaves "checking…"
    page.wait_for_function(
        "() => document.getElementById('mathBadge').textContent.trim() !== 'checking…'",
        timeout=15_000,
    )

    yield page, js_errors

    ctx.close()


# ===========================================================================
# 1. Page load
# ===========================================================================

class TestPageLoad:
    def test_no_js_errors_on_load(self, page_with_errors):
        _, errors = page_with_errors
        assert errors == [], "Unexpected JS errors on load:\n" + "\n".join(errors)

    def test_math_badge_is_passing(self, page_with_errors):
        page, _ = page_with_errors
        classes = page.locator("#mathBadge").get_attribute("class") or ""
        assert "badge-pass" in classes, (
            f"Math badge not passing. classes={classes!r}, "
            f"text={page.locator('#mathBadge').text_content()!r}"
        )

    def test_math_badge_text_contains_verified(self, page_with_errors):
        page, _ = page_with_errors
        text = page.locator("#mathBadge").text_content()
        assert "verified" in text.lower(), f"Unexpected badge text: {text!r}"

    def test_page_title_mentions_attention(self, page_with_errors):
        page, _ = page_with_errors
        title = page.title()
        assert "Transformer" in title or "Attention" in title, (
            f"Unexpected page title: {title!r}"
        )


# ===========================================================================
# 2. Walkthrough step panels are populated
# ===========================================================================

class TestWalkthroughContent:
    @pytest.mark.parametrize("step_id", [
        "step1Content", "step2Content", "step3Content",
        "step4Content", "step5Content", "step6Content",
    ])
    def test_step_panel_is_populated(self, page_with_errors, step_id):
        page, _ = page_with_errors
        html = page.locator(f"#{step_id}").inner_html()
        assert html.strip(), f"#{step_id} is empty after page load"

    def test_step3_range_shows_two_numbers(self, page_with_errors):
        page, _ = page_with_errors
        text = page.locator("#step3Range").text_content().strip()
        assert text and "to" in text, f"#step3Range looks wrong: {text!r}"

    def test_step5_has_nine_heatmap_cells(self, page_with_errors):
        """3 tokens × 3 tokens = 9 cells."""
        page, _ = page_with_errors
        count = page.locator("#step5Content .hm-cell").count()
        assert count == 9, f"Expected 9 hm-cells (3×3), got {count}"

    def test_step4_insight_has_text(self, page_with_errors):
        page, _ = page_with_errors
        text = page.locator("#step4Insight").text_content().strip()
        assert text, "#step4Insight is empty after load"

    def test_step5_insight_has_text(self, page_with_errors):
        page, _ = page_with_errors
        text = page.locator("#step5Insight").text_content().strip()
        assert text, "#step5Insight is empty after load"


# ===========================================================================
# 3. Scaling slider interaction
# ===========================================================================

class TestScalingSlider:
    def _set_t(self, page, value: str):
        page.locator("#scalingSlider").fill(value)
        page.locator("#scalingSlider").dispatch_event("input")

    def test_slider_changes_step4_content(self, page_with_errors):
        page, _ = page_with_errors
        before = page.locator("#step4Content").inner_html()
        self._set_t(page, "6")
        after = page.locator("#step4Content").inner_html()
        self._set_t(page, "2")
        assert before != after, "Step 4 content did not change when slider moved to T=6"

    def test_slider_changes_step5_heatmap(self, page_with_errors):
        page, _ = page_with_errors
        cells_before = [
            page.locator("#step5Content .hm-cell").nth(i).get_attribute("style")
            for i in range(9)
        ]
        self._set_t(page, "0.25")
        cells_after = [
            page.locator("#step5Content .hm-cell").nth(i).get_attribute("style")
            for i in range(9)
        ]
        self._set_t(page, "2")
        assert cells_before != cells_after, (
            "Heatmap cell colors did not change when slider moved to T=0.25"
        )

    def test_slider_val_display_updates(self, page_with_errors):
        page, _ = page_with_errors
        self._set_t(page, "4")
        val = page.locator("#scalingVal").text_content()
        self._set_t(page, "2")
        assert val.startswith("4"), f"Slider value display did not update: {val!r}"

    def test_annotation_is_default_at_sqrt_dk(self, page_with_errors):
        """At T=2 (= √d_k = √4), annotation should carry annot-default class."""
        page, _ = page_with_errors
        self._set_t(page, "2")
        classes = page.locator("#scalingAnnotation").get_attribute("class") or ""
        assert "annot-default" in classes, f"Annotation classes at T=2: {classes!r}"

    def test_annotation_is_flat_at_high_t(self, page_with_errors):
        """T > √d_k → each score divided by more → distribution flatter."""
        page, _ = page_with_errors
        self._set_t(page, "6")
        classes = page.locator("#scalingAnnotation").get_attribute("class") or ""
        self._set_t(page, "2")
        assert "annot-flat" in classes, (
            f"Annotation at T=6 should be annot-flat, got: {classes!r}"
        )

    def test_annotation_is_sharp_at_low_t(self, page_with_errors):
        """T < √d_k → scores divided by less → distribution sharper (more peaked)."""
        page, _ = page_with_errors
        self._set_t(page, "0.25")
        classes = page.locator("#scalingAnnotation").get_attribute("class") or ""
        self._set_t(page, "2")
        assert "annot-sharp" in classes, (
            f"Annotation at T=0.25 should be annot-sharp, got: {classes!r}"
        )


# ===========================================================================
# 4. Head tab interaction
# ===========================================================================

class TestHeadTabs:
    def test_head0_is_active_on_load(self, page_with_errors):
        page, _ = page_with_errors
        classes = page.locator(".head-tab").nth(0).get_attribute("class") or ""
        assert "active" in classes, f"Head 0 tab should be active on load, got: {classes!r}"

    def test_head1_is_inactive_on_load(self, page_with_errors):
        page, _ = page_with_errors
        classes = page.locator(".head-tab").nth(1).get_attribute("class") or ""
        assert "active" not in classes, (
            f"Head 1 tab should not be active on load, got: {classes!r}"
        )

    def test_switching_to_head1_changes_step2(self, page_with_errors):
        page, _ = page_with_errors
        before = page.locator("#step2Content").inner_html()
        page.locator(".head-tab").nth(1).click()
        after = page.locator("#step2Content").inner_html()
        page.locator(".head-tab").nth(0).click()
        assert before != after, "Step 2 content did not change when switching to Head 1"

    def test_switching_to_head1_changes_step5(self, page_with_errors):
        page, _ = page_with_errors
        before = page.locator("#step5Content").inner_html()
        page.locator(".head-tab").nth(1).click()
        after = page.locator("#step5Content").inner_html()
        page.locator(".head-tab").nth(0).click()
        assert before != after, "Step 5 heatmap did not change when switching to Head 1"

    def test_switching_back_to_head0_restores_step2(self, page_with_errors):
        page, _ = page_with_errors
        original = page.locator("#step2Content").inner_html()
        page.locator(".head-tab").nth(1).click()
        page.locator(".head-tab").nth(0).click()
        restored = page.locator("#step2Content").inner_html()
        assert original == restored, "Step 2 did not restore when switching back to Head 0"


# ===========================================================================
# 5. Causal mask toggle
# ===========================================================================

class TestCausalMaskToggle:
    def test_mask_is_off_on_load(self, page_with_errors):
        page, _ = page_with_errors
        assert not page.locator("#maskToggle").is_checked(), (
            "Causal mask checkbox should be unchecked on load"
        )

    def test_checking_mask_changes_step4(self, page_with_errors):
        page, _ = page_with_errors
        before = page.locator("#step4Content").inner_html()
        page.locator("#maskToggle").check()
        after = page.locator("#step4Content").inner_html()
        page.locator("#maskToggle").uncheck()
        assert before != after, "Step 4 content did not change when causal mask enabled"

    def test_checking_mask_changes_step5(self, page_with_errors):
        page, _ = page_with_errors
        before = page.locator("#step5Content").inner_html()
        page.locator("#maskToggle").check()
        after = page.locator("#step5Content").inner_html()
        page.locator("#maskToggle").uncheck()
        assert before != after, "Step 5 heatmap did not change when causal mask enabled"

    def test_unchecking_mask_restores_step4(self, page_with_errors):
        page, _ = page_with_errors
        original = page.locator("#step4Content").inner_html()
        page.locator("#maskToggle").check()
        page.locator("#maskToggle").uncheck()
        restored = page.locator("#step4Content").inner_html()
        assert original == restored, "Step 4 content did not restore after unchecking mask"

    def test_mask_insight_mentions_masking(self, page_with_errors):
        """With mask on, step 4 insight must say something about causal/masking."""
        page, _ = page_with_errors
        page.locator("#maskToggle").check()
        text = page.locator("#step4Insight").text_content().lower()
        page.locator("#maskToggle").uncheck()
        assert any(kw in text for kw in ("causal", "mask", "upper", "future")), (
            f"Step 4 insight with mask enabled doesn't mention masking: {text!r}"
        )


# ===========================================================================
# 6. Font sizes — regression guard
# ===========================================================================

class TestFontSizes:
    def test_step_desc_is_17px(self, page_with_errors):
        page, _ = page_with_errors
        size = page.locator(".step-desc").first.evaluate(
            "el => getComputedStyle(el).fontSize"
        )
        assert size == "17px", f".step-desc font size is {size!r}, expected '17px'"

    def test_step_insight_is_visible_and_nonzero(self, page_with_errors):
        page, _ = page_with_errors
        box = page.locator("#step4Insight").bounding_box()
        assert box and box["height"] > 0 and box["width"] > 0, (
            f"#step4Insight bounding box looks wrong: {box}"
        )


# ===========================================================================
# 7. Real attention section — graceful error state on file://
#    (fetch always fails on file:// due to CORS; we test the fallback UI)
# ===========================================================================

class TestRealAttentionFallback:
    def test_sentence_select_exists(self, page_with_errors):
        page, _ = page_with_errors
        assert page.locator("#sentSelect").count() == 1, "#sentSelect not found"

    def test_layer_tabs_container_exists(self, page_with_errors):
        page, _ = page_with_errors
        assert page.locator("#layerTabs").count() == 1, "#layerTabs not found"

    def test_error_message_is_visible(self, page_with_errors):
        """On file://, fetch fails; error message should appear in #realAttLoading."""
        page, _ = page_with_errors
        # Wait briefly for the async fetch to fail
        page.wait_for_function(
            "() => !document.getElementById('realAttLoading').textContent.includes('Loading')",
            timeout=5_000,
        )
        text = page.locator("#realAttLoading").text_content()
        assert "http.server" in text or "Could not load" in text, (
            f"Expected error message, got: {text!r}"
        )

    def test_sentselect_shows_error_not_loading(self, page_with_errors):
        """sentSelect must not be stuck at 'Loading…' — it should show the error option."""
        page, _ = page_with_errors
        page.wait_for_function(
            "() => !document.getElementById('realAttLoading').textContent.includes('Loading')",
            timeout=5_000,
        )
        opt_text = page.locator("#sentSelect option").nth(0).text_content()
        assert "Loading" not in opt_text, (
            f"sentSelect is still stuck at 'Loading…' after fetch failure. "
            f"First option text: {opt_text!r}"
        )

    def test_sentselect_is_disabled_on_error(self, page_with_errors):
        """sentSelect should be disabled when data is unavailable."""
        page, _ = page_with_errors
        page.wait_for_function(
            "() => !document.getElementById('realAttLoading').textContent.includes('Loading')",
            timeout=5_000,
        )
        disabled = page.locator("#sentSelect").is_disabled()
        assert disabled, "sentSelect should be disabled when real_attention.json could not load"


# ===========================================================================
# 8. Real attention section — full load via page.route()
#    Playwright intercepts the fetch() call and responds with the local file.
#    No HTTP server needed; works on file:// and on GitHub Pages alike.
# ===========================================================================

@pytest.fixture(scope="module")
def served_page(pw_browser):
    """
    Open the page via file:// with real_attention.json pre-loaded.

    page.route() cannot intercept file:// requests (Chromium's file protocol
    bypasses the network stack entirely, so routes never fire).  Instead we
    use add_init_script() to shadow window.fetch before any page JS runs:
    when the page calls fetch('...real_attention.json'), it receives the file
    content immediately from the injected stub — no server needed.
    """
    json_body = REAL_ATT_PATH.read_text(encoding="utf-8")

    # Escape backticks so the JSON can safely sit inside a JS template literal
    json_escaped = json_body.replace("\\", "\\\\").replace("`", "\\`")

    init_script = f"""
(function () {{
    const _data = JSON.parse(`{json_escaped}`);
    const _fetch = window.fetch;
    window.fetch = function (url, ...args) {{
        if (String(url).includes('real_attention.json')) {{
            return Promise.resolve(
                new Response(JSON.stringify(_data), {{
                    status: 200,
                    headers: {{ 'Content-Type': 'application/json' }},
                }})
            );
        }}
        return _fetch.call(this, url, ...args);
    }};
}})();
"""

    ctx  = pw_browser.new_context()
    page = ctx.new_page()
    page.add_init_script(init_script)
    page.goto(PAGE_URL)

    # Wait for the grid to appear (JS sets display:grid after parsing the JSON)
    page.wait_for_function(
        "() => document.getElementById('headGrid').style.display !== 'none'",
        timeout=20_000,
    )

    yield page
    ctx.close()


class TestRealAttentionLoaded:
    def test_head_grid_is_visible(self, served_page):
        assert served_page.locator("#headGrid").is_visible(), "#headGrid should be visible after load"

    def test_twelve_head_cards_rendered(self, served_page):
        count = served_page.locator(".head-card").count()
        assert count == 12, f"Expected 12 head cards, got {count}"

    def test_sentselect_has_five_options(self, served_page):
        count = served_page.locator("#sentSelect option").count()
        assert count == 5, f"Expected 5 sentence options, got {count}"

    def test_sentselect_is_enabled(self, served_page):
        assert not served_page.locator("#sentSelect").is_disabled(), (
            "sentSelect should be enabled when data loaded"
        )

    def test_layer_tabs_built(self, served_page):
        count = served_page.locator("#layerTabs .layer-tab").count()
        assert count == 6, f"Expected 6 layer tabs (DistilBERT layers), got {count}"

    def test_layer0_is_active_on_load(self, served_page):
        classes = served_page.locator("#layerTabs .layer-tab").nth(0).get_attribute("class") or ""
        assert "active" in classes, f"Layer 0 tab should be active on load, got: {classes!r}"

    def test_switching_layer_rerenders_grid(self, served_page):
        before = served_page.locator("#headGrid").inner_html()
        served_page.locator("#layerTabs .layer-tab").nth(3).click()
        after = served_page.locator("#headGrid").inner_html()
        served_page.locator("#layerTabs .layer-tab").nth(0).click()
        assert before != after, "Head grid did not change when switching to layer 3"

    def test_switching_sentence_rerenders_grid(self, served_page):
        before = served_page.locator("#headGrid").inner_html()
        served_page.locator("#sentSelect").select_option(index=1)
        after = served_page.locator("#headGrid").inner_html()
        served_page.locator("#sentSelect").select_option(index=0)
        assert before != after, "Head grid did not change when switching to sentence 1"

    def test_sent_note_appears_after_selection(self, served_page):
        served_page.locator("#sentSelect").select_option(index=2)
        note = served_page.locator("#sentNote")
        assert note.is_visible(), "#sentNote should be visible after selecting a sentence"
        text = note.text_content().strip()
        assert text, "#sentNote is empty"
        served_page.locator("#sentSelect").select_option(index=0)

    def test_mini_heatmap_cells_present_in_first_head(self, served_page):
        cells = served_page.locator(".head-card").nth(0).locator(".mini-cell").count()
        assert cells > 0, "First head card has no mini-cell elements"
