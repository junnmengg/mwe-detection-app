/**
 * Keep the hosted Streamlit demo awake, and wake it if it has already slept.
 *
 * Streamlit Community Cloud suspends an app after 12 hours without traffic.
 * A plain HTTPS request fetches the page shell but never opens the WebSocket
 * that carries a real Streamlit session, and Streamlit does not document
 * whether that counts as traffic. A real browser removes the ambiguity.
 *
 * Design notes, both learned from a failing run:
 *
 *   * Every individual Puppeteer call must finish inside `protocolTimeout`
 *     (180s by default). A single `waitForSelector` longer than that dies with
 *     an opaque `ProtocolError` instead of a clean timeout, so readiness is
 *     polled in short slices and `protocolTimeout` is raised explicitly.
 *
 *   * `networkidle2` is the wrong wait for Streamlit: the app holds a
 *     WebSocket open, so "the network went quiet" may never happen. The script
 *     waits for `domcontentloaded` and then polls the DOM instead.
 *
 * On failure it writes a screenshot and the page HTML to `artifacts/` so the
 * workflow can upload them; a mystery timeout then becomes something you can
 * actually look at.
 */

import { mkdir, writeFile, appendFile } from "node:fs/promises";
import path from "node:path";

import puppeteer from "puppeteer";

const APP_URL = process.env.APP_URL;
const ARTIFACT_DIR = process.env.ARTIFACT_DIR ?? "artifacts";

/** Raised above every wait below so CDP never dies mid-wait. */
const PROTOCOL_TIMEOUT_MS = 300_000;
const NAVIGATION_TIMEOUT_MS = 60_000;
/** Total budget for the app to become interactive, polled in slices. */
const READY_BUDGET_MS = 210_000;
const POLL_INTERVAL_MS = 3_000;
const SETTLE_MS = 6_000;

/** Any of these means Streamlit has mounted and is running. */
const APP_SELECTORS = [
  '[data-testid="stApp"]',
  '[data-testid="stAppViewContainer"]',
  ".stApp",
  "section.main",
];

/** Text Community Cloud shows on the sleeping-app interstitial. */
const SLEEP_PATTERNS = [
  "get this app back up",
  "has gone to sleep",
  "is currently sleeping",
  "wake it back up",
  "zzzz",
];

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

if (!APP_URL) {
  console.error("APP_URL is not set.");
  process.exit(1);
}

/**
 * Report what the page currently looks like.
 *
 * Returns which readiness selector matched (if any), whether the sleep
 * interstitial is showing, the title and a short text excerpt. Logging this
 * on every attempt means a future failure is diagnosable from the run log
 * alone.
 */
async function probe(page, selectors, sleepPatterns) {
  return page.evaluate(
    (appSelectors, patterns) => {
      const matched = appSelectors.find((selector) =>
        document.querySelector(selector),
      );
      const text = (document.body?.innerText ?? "").trim();
      const haystack = text.toLowerCase();
      return {
        matchedSelector: matched ?? null,
        asleep: patterns.some((pattern) => haystack.includes(pattern)),
        title: document.title,
        textLength: text.length,
        excerpt: text.slice(0, 400),
      };
    },
    selectors,
    sleepPatterns,
  );
}

/** Click the wake control on the sleeping-app page. Returns true if clicked. */
async function clickWakeControl(page, sleepPatterns) {
  return page.evaluate((patterns) => {
    const candidates = [
      ...document.querySelectorAll("button, a, [role='button'], input[type='submit']"),
    ];
    const target = candidates.find((element) => {
      const label = (
        element.innerText ||
        element.textContent ||
        element.value ||
        ""
      ).toLowerCase();
      return patterns.some((pattern) => label.includes(pattern));
    });
    if (!target) return false;
    target.click();
    return true;
  }, sleepPatterns);
}

/** Save a screenshot and the raw HTML so a failed run can be inspected. */
async function captureDiagnostics(page, label) {
  try {
    await mkdir(ARTIFACT_DIR, { recursive: true });
    await page.screenshot({
      path: path.join(ARTIFACT_DIR, `${label}.png`),
      fullPage: true,
    });
    await writeFile(path.join(ARTIFACT_DIR, `${label}.html`), await page.content());
    console.log(`Saved diagnostics to ${ARTIFACT_DIR}/${label}.{png,html}`);
  } catch (error) {
    console.warn(`Could not capture diagnostics: ${error.message}`);
  }
}

async function summarise(line) {
  if (!process.env.GITHUB_STEP_SUMMARY) return;
  await appendFile(process.env.GITHUB_STEP_SUMMARY, `${line}\n`);
}

async function main() {
  const browser = await puppeteer.launch({
    protocolTimeout: PROTOCOL_TIMEOUT_MS,
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
      "--disable-dev-shm-usage",
      "--disable-gpu",
    ],
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 900 });
    page.setDefaultTimeout(NAVIGATION_TIMEOUT_MS);

    console.log(`Visiting ${APP_URL}`);
    // domcontentloaded, not networkidle2: Streamlit's WebSocket keeps the
    // connection open, so the network may never go idle.
    await page.goto(APP_URL, {
      waitUntil: "domcontentloaded",
      timeout: NAVIGATION_TIMEOUT_MS,
    });

    const deadline = Date.now() + READY_BUDGET_MS;
    let wokeIt = false;
    let last = null;

    while (Date.now() < deadline) {
      last = await probe(page, APP_SELECTORS, SLEEP_PATTERNS);

      if (last.matchedSelector) {
        await sleep(SETTLE_MS); // let the WebSocket session register a visit
        const verdict = wokeIt ? "Woke a sleeping app" : "App already awake";
        console.log(`${verdict} — matched ${last.matchedSelector}, title "${last.title}"`);
        await summarise(`- ${verdict} — \`${last.title}\``);
        return;
      }

      if (last.asleep && !wokeIt) {
        console.log("Sleeping-app page detected; clicking the wake control.");
        await captureDiagnostics(page, "before-wake");
        wokeIt = await clickWakeControl(page, SLEEP_PATTERNS);
        console.log(
          wokeIt
            ? "Wake control clicked; waiting for the app to boot."
            : "No wake control found on the page; continuing to poll.",
        );
      }

      const remaining = Math.round((deadline - Date.now()) / 1000);
      console.log(
        `Not ready yet (title="${last.title}", text=${last.textLength} chars, ` +
          `asleep=${last.asleep}); ${remaining}s left.`,
      );
      await sleep(POLL_INTERVAL_MS);
    }

    await captureDiagnostics(page, "failure");
    console.error("Timed out waiting for the app to become ready.");
    console.error(`Last probe: ${JSON.stringify(last, null, 2)}`);
    await summarise(`- Failed to confirm the app — last title \`${last?.title ?? "?"}\``);
    process.exitCode = 1;
  } finally {
    await browser.close();
  }
}

main().catch(async (error) => {
  console.error("Unexpected failure while visiting the demo:");
  console.error(error);
  await summarise(`- Errored: \`${error.message}\``);
  process.exit(1);
});
