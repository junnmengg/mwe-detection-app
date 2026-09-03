/**
 * Keep the hosted Streamlit demo awake, and wake it if it has already slept.
 *
 * Streamlit Community Cloud suspends an app after 12 hours without traffic.
 * A plain HTTPS request fetches the page shell but does not open the WebSocket
 * that Streamlit uses for real sessions, and Streamlit does not document
 * whether that counts as traffic. A real browser removes the ambiguity: it
 * loads the page, opens the socket and renders the app exactly as a visitor
 * would.
 *
 * The script exits non-zero if it cannot confirm the app is running, so a
 * silent failure becomes a red run and a notification email.
 */

import puppeteer from "puppeteer";

const APP_URL = process.env.APP_URL;

/** Streamlit renders the app body inside one of these. */
const APP_SELECTOR = '[data-testid="stApp"], .stApp';

/** Text on the button Community Cloud shows for a sleeping app. */
const WAKE_BUTTON_PATTERN = /get this app back up/i;

const NAVIGATION_TIMEOUT_MS = 90_000;
const BOOT_TIMEOUT_MS = 240_000;
const SETTLE_MS = 5_000;

if (!APP_URL) {
  console.error("APP_URL is not set.");
  process.exit(1);
}

/** Click the "wake up" button if this is the sleeping-app page. */
async function clickWakeButtonIfPresent(page) {
  return page.evaluate((source) => {
    const pattern = new RegExp(source, "i");
    const clickable = [...document.querySelectorAll("button, a, [role='button']")];
    const target = clickable.find((element) => pattern.test(element.textContent ?? ""));
    if (!target) return false;
    target.click();
    return true;
  }, WAKE_BUTTON_PATTERN.source);
}

async function main() {
  const browser = await puppeteer.launch({
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 800 });
    page.setDefaultTimeout(NAVIGATION_TIMEOUT_MS);

    console.log(`Visiting ${APP_URL}`);
    await page.goto(APP_URL, {
      waitUntil: "networkidle2",
      timeout: NAVIGATION_TIMEOUT_MS,
    });

    const wasAsleep = await clickWakeButtonIfPresent(page);
    if (wasAsleep) {
      console.log("App was asleep. Clicked the wake button; waiting for it to boot.");
      await page.waitForSelector(APP_SELECTOR, { timeout: BOOT_TIMEOUT_MS });
    }

    // Confirm the real app rendered rather than an error or interstitial page.
    await page.waitForSelector(APP_SELECTOR, { timeout: BOOT_TIMEOUT_MS });

    // Let the WebSocket session settle so Community Cloud records a visit.
    await new Promise((resolve) => setTimeout(resolve, SETTLE_MS));

    const title = await page.title();
    console.log(
      wasAsleep
        ? `Woke the app successfully. Page title: ${title}`
        : `App was already awake. Page title: ${title}`,
    );

    // Surfaced in the job summary so the history is readable at a glance.
    if (process.env.GITHUB_STEP_SUMMARY) {
      const { appendFileSync } = await import("node:fs");
      appendFileSync(
        process.env.GITHUB_STEP_SUMMARY,
        `- ${wasAsleep ? "Woke a sleeping app" : "App already awake"} — \`${title}\`\n`,
      );
    }
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error("Failed to confirm the demo is running:");
  console.error(error);
  process.exit(1);
});
