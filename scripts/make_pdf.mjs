// Make s1.pdf from the S-1 on EDGAR: download the filing and its images, shrink the images,
// and print the filing to PDF with headless Chrome, one filing page per PDF page.
//
// EDGAR lays the filing out as fixed 612x792pt pages (a few are 792x612pt landscape), so each one
// becomes a PDF page of the same size. The images are 600 dpi scans; they are cut to 1650 pixels on
// the long side (about 150 dpi) with macOS's sips, which takes the PDF from 169 MB to 34 MB.
//
// Usage: node scripts/make_pdf.mjs   (macOS, with Google Chrome installed)
import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { ROOT } from "./lib/colbert.mjs";

const BASE = "https://www.sec.gov/Archives/edgar/data/2130357/000162828026067168/";
const FILING = "anthropicpbc-sx1.htm";
// EDGAR asks automated clients to identify themselves.
const HEADERS = { "User-Agent": "ant-s1-search-demo admin@example.com" };
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const RAW = join(ROOT, "cache", "s1");
const SMALL = join(ROOT, "cache", "s1-small");

async function fetchOnce(name) {
  const path = join(RAW, name);
  if (!existsSync(path)) {
    const res = await fetch(BASE + name, { headers: HEADERS });
    if (!res.ok) throw new Error(`HTTP ${res.status} for ${BASE + name}`);
    await writeFile(path, Buffer.from(await res.arrayBuffer()));
    await new Promise((r) => setTimeout(r, 150)); // EDGAR allows 10 requests a second
  }
  return path;
}

await mkdir(RAW, { recursive: true });
await mkdir(SMALL, { recursive: true });
let html = await readFile(await fetchOnce(FILING), "utf8");
html = html.slice(html.indexOf("<html"), html.lastIndexOf("</html>") + 7); // drop EDGAR's SGML wrapper

const images = [...new Set([...html.matchAll(/src="([\w.-]+\.jpg)"/g)].map((m) => m[1]))];
for (const name of images) {
  const raw = await fetchOnce(name);
  const small = join(SMALL, name);
  if (!existsSync(small)) execFileSync("sips", ["-s", "format", "jpeg", "-s", "formatOptions", "70", "-Z", "1650", raw, "--out", small], { stdio: "ignore" });
}
console.error(`${images.length} images`);

// Each filing page is a div with margin:10px auto; landscape ones are 792pt wide.
let pages = 0;
let landscape = 0;
html = html.replace(/<div style="(--justify:[^"]*?margin:10px auto 10px auto;[^"]*)"/g, (_, style) => {
  pages++;
  const land = /(?<![-\w])width:792pt/.test(style);
  if (land) landscape++;
  return `<div class="pg${land ? " land" : ""}" style="${style}"`;
});
const css =
  "<style>@page{size:612pt 792pt;margin:0}@page land{size:792pt 612pt;margin:0}" +
  "html,body{margin:0;padding:0}body>div{background:none!important}" +
  ".pg{margin:0!important;border:0!important;break-after:page;content-visibility:visible!important}" +
  ".land{page:land}</style>";
await writeFile(join(SMALL, "print.html"), html.replace("</head>", css + "</head>"));
console.error(`${pages} pages (${landscape} landscape)`);

execFileSync(CHROME, [
  "--headless=new",
  "--disable-gpu",
  "--no-pdf-header-footer",
  "--virtual-time-budget=20000",
  `--print-to-pdf=${join(ROOT, "s1.pdf")}`,
  `file://${join(SMALL, "print.html")}`,
], { stdio: "ignore" });
console.error("Wrote s1.pdf");
