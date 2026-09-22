#!/usr/bin/env node
/**
 * Capture FieldLens dashboard pages at desktop and mobile widths.
 * Requires: npm install -D playwright && npx playwright install chromium
 * Usage: node scripts/screenshot.mjs [baseUrl]
 */
const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const BASE = process.argv[2] || "http://localhost:3000";
const OUT = path.join(__dirname, "..", "..", "docs", "screenshots");
const PAGES = [
  "overview",
  "explorer",
  "compare",
  "results",
  "method",
  "terms",
  "privacy",
];
const PATHS = {
  overview: "/",
  explorer: "/explorer",
  compare: "/compare",
  results: "/results",
  method: "/method",
  terms: "/terms",
  privacy: "/privacy",
};
const WIDTHS = [
  { name: "desktop", width: 1280, height: 800 },
  { name: "mobile", width: 390, height: 844 },
];

async function main() {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  for (const pageName of PAGES) {
    for (const vp of WIDTHS) {
      const page = await browser.newPage({
        viewport: { width: vp.width, height: vp.height },
      });
      const url = BASE + PATHS[pageName];
      await page.goto(url, { waitUntil: "networkidle" });
      const file = path.join(OUT, `${pageName}_${vp.name}.png`);
      await page.screenshot({ path: file, fullPage: true });
      console.log("wrote", file);
      await page.close();
    }
  }
  await browser.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
