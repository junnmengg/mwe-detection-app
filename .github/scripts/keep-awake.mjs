/**
 * Keep the hosted Streamlit demo awake, and wake it if it has already slept.
 *
 * Streamlit Community Cloud suspends an app after 12 hours without traffic.
 * A plain HTTPS request fetches the page shell but never opens the WebSocket
 * that carries a real Streamlit session, and Streamlit does not document
 * whether that counts as traffic. A real browser removes the ambiguity.
 *
 * Three things about this page were learned the hard way from failing runs:
 *
 *   1. Every Puppeteer call must finish inside `protocolTimeout` (180s by
 *      default). One long `waitForSelector` dies with an opaque
 *      `ProtocolError`, so readiness is polled in short slices instead.
 *
 *   2. `networkidle2` never settles here, because Streamlit holds a WebSocket
 *      open for the session. Navigation waits for `domcontentloaded`.
 *
 *   3. Community Cloud serves the app inside a nested browsing context, so the
 *      top-level document has the right `<title>` but an empty body and none
 *      of Streamlit's elements. Every frame is probed, not just the main one.
 *
 * Success is judged on two levels. A matched Streamlit element in any frame is
 * proof the app rendered. Failing that, the expected page title with no sleep
 * interstitial is strong evidence the app served the request, which is what
 * actually matters for keeping it awake; the run passes and says it used the
 * weaker signal. Anything else fails and uploads a screenshot plus the page
 * HTML.
 */

import { mkdir, writeFile, appendFile } from "node:fs/promises";
import path from "node:path";

import puppeteer from "puppeteer";

const APP_URL = process.env.APP_URL;
const ARTIFACT_DIR = process.env.ARTIFACT_DIR ?? "artifacts";
/** Substring of the title Streamlit renders for this app. */
const EXPECTED_TITLE = process.env.EXPECTED_TITLE ?? "MWEs Prediction App";

const PROTOCOL_TIMEOUT_MS = 300_000;
const NAVIGATION_TIMEOUT_MS = 60_000;
const READY_BUDGET_MS = 180_000;
const POLL_INTERVAL_MS = 3_000;
/** Only log every Nth poll; 60 identical lines help nobody. */
const LOG_EVERY = 5;
const SETTLE_MS = 6_000;

const APP_SELECTORS = [
  '[data-testid="stApp"]',
  '[data-testid="stAppViewContainer"]',
  '[data-testid="stSidebar"]',
  ".stApp",
  "section.main",
];

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

/** Inspect one frame for Streamlit's elements and the sleep interstitial. */
async function probeFrame(frame, selectors, patterns) {
  return frame.evaluate(
    (appSelectors, sleepPatterns) => {
      const matched = appSelectors.find((selector) => document.querySelector(selector));
      const text = (document.body?.innerText ?? "").trim();
      const haystack = text.toLowerCase();
      return {
        matchedSelector: matched ?? null,
        asleep: sleepPatterns.some((pattern) => haystack.includes(pattern)),
        textLength: text.length,
        excerpt: text.slice(0, 200),
        frameCount: document.querySelectorAll("iframe").length,
      };
    },
    selectors,
    patterns,
  );
}

/** Probe every frame on the page, tolerating ones that detach mid-check. */
async function probeAllFrames(page) {
  const frames = page.frames();
  const results = [];
  for (const frame of frames) {
    try {
      results.push({ url: frame.url(), ...(await probeFrame(frame, APP_SELECTORS, SLEEP_PATTERNS)) });
    } catch (error) {
      results.push({ url: frame.url(), error: error.message });
    }
  }
  return results;
}

/** Click the wake control, searching every frame. Returns true if clicked. */
async function clickWakeControl(page) {
  for (const frame of page.frames()) {
    try {
      const clicked = await frame.evaluate((patterns) => {
        const candidates = [
          ...document.querySelectorAll(
            "button, a, [role='button'], input[type='submit']",
          ),
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
      }, SLEEP_PATTERNS);
      if (clicked) return true;
    } catch {
      // Frame detached while we were looking at it; try the next one.
    }
  }
  return false;
}

async function captureDiagnostics(page, label) {
  try {
    await mkdir(ARTIFACT_DIR, { recursive: true });
    await page.screenshot({ path: path.join(ARTIFACT_DIR, `${label}.png`), fullPage: true });
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
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 900 });
    page.setDefaultTimeout(NAVIGATION_TIMEOUT_MS);

    console.log(`Visiting ${APP_URL}`);
    await page.goto(APP_URL, { waitUntil: "domcontentloaded", timeout: NAVIGATION_TIMEOUT_MS });

    const deadline = Date.now() + READY_BUDGET_MS;
    let wokeIt = false;
    let attempt = 0;
    let frames = [];
    let title = "";

    while (Date.now() < deadline) {
      attempt += 1;
      frames = await probeAllFrames(page);
      title = await page.title();

      const rendered = frames.find((frame) => frame.matchedSelector);
      const asleep = frames.some((frame) => frame.asleep);

      if (rendered) {
        await sleep(SETTLE_MS);
        const verdict = wokeIt ? "Woke a sleeping app" : "App already awake";
        console.log(`${verdict} — matched ${rendered.matchedSelector} in ${rendered.url}`);
        await summarise(`- ${verdict} — \`${title}\``);
        return;
      }

      // Fallback: the app answered with its own title and is not showing the
      // sleep page. The request reached a running container, which is the
      // thing that keeps it awake, even if the DOM probe cannot see inside.
      if (!asleep && EXPECTED_TITLE && title.includes(EXPECTED_TITLE) && attempt >= 3) {
        await sleep(SETTLE_MS);
        console.log(`App served "${title}" but no Streamlit element was visible to the probe.`);
        console.log(`Frames seen: ${JSON.stringify(frames.map((f) => f.url))}`);
        await summarise(`- App responded (title match, DOM probe blind) — \`${title}\``);
        return;
      }

      if (asleep && !wokeIt) {
        console.log("Sleeping-app page detected; clicking the wake control.");
        await captureDiagnostics(page, "before-wake");
        wokeIt = await clickWakeControl(page);
        console.log(wokeIt ? "Wake control clicked." : "No wake control found; still polling.");
      }

      if (attempt === 1 || attempt % LOG_EVERY === 0) {
        const remaining = Math.round((deadline - Date.now()) / 1000);
        console.log(
          `Not ready (title="${title}", frames=${frames.length}, ` +
            `asleep=${asleep}); ${remaining}s left.`,
        );
      }
      await sleep(POLL_INTERVAL_MS);
    }

    await captureDiagnostics(page, "failure");
    console.error("Timed out waiting for the app to become ready.");
    console.error(`Title: ${title}`);
    console.error(`Frames: ${JSON.stringify(frames, null, 2)}`);
    await summarise(`- Failed to confirm the app — last title \`${title}\``);
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
