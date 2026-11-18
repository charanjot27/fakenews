// Prints docs/guide/guide.html to docs/SatyaCheck_Text_Team_Guide.pdf with headless Chromium (Playwright).
// Usage: NODE_PATH=$(npm root -g) node scripts/print_pdf.js
const path = require("path");
const { chromium } = require("playwright");
(async () => {
  const root = path.resolve(__dirname, "..");
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto("file://" + path.join(root, "docs/guide/guide.html"), { waitUntil: "load" });
  await page.waitForTimeout(800);
  await page.pdf({
    path: path.join(root, "docs/SatyaCheck_Text_Team_Guide.pdf"), format: "A4", printBackground: true,
    displayHeaderFooter: true,
    headerTemplate: "<div></div>",
    footerTemplate: '<div style="font-size:8px;color:#888;width:100%;padding:0 14mm;display:flex;justify-content:space-between">' +
      '<span>SatyaCheck-Text · Team Guide</span><span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>',
    margin: { top: "14mm", bottom: "16mm", left: "13mm", right: "13mm" },
  });
  await browser.close();
  console.log("docs/SatyaCheck_Text_Team_Guide.pdf");
})();
