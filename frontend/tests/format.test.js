import { test } from "node:test";
import assert from "node:assert/strict";
import { compactNumber, formatPrice, highlightParts, pct, shortTitle } from "../src/lib/format.js";

test("formatPrice handles missing and small values", () => {
  assert.equal(formatPrice(null), "Price n/a");
  assert.equal(formatPrice(89.5), "$89.50");
  assert.equal(formatPrice(320), "$320");
});

test("pct and compactNumber", () => {
  assert.equal(pct(0.456), "46%");
  assert.equal(pct(null), "–");
  assert.equal(compactNumber(1234), "1.2k");
  assert.equal(compactNumber(25000), "25k");
});

test("shortTitle truncates long titles", () => {
  assert.equal(shortTitle("abc", 10), "abc");
  assert.equal(shortTitle("a".repeat(20), 10).length, 10);
});

test("highlightParts marks the aspect term case-insensitively", () => {
  const parts = highlightParts("The Camera is great, camera!", "camera");
  assert.deepEqual(
    parts.map((p) => [p.text, p.match]),
    [["The ", false], ["Camera", true], [" is great, ", false], ["camera", true], ["!", false]],
  );
  assert.deepEqual(highlightParts("no term", ""), [{ text: "no term", match: false }]);
});
