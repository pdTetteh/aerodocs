import { readFile, stat } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const assets = [
  "app/ui/static/app.css",
  "app/ui/static/app.js",
  "app/ui/templates/base.html",
  "app/ui/templates/home.html",
  "app/ui/templates/documents.html",
  "app/ui/templates/search.html",
  "app/ui/templates/ask.html",
  "app/ui/templates/not_found.html",
];

for (const relative of assets) {
  const path = resolve(root, relative);
  const metadata = await stat(path);
  if (!metadata.isFile() || metadata.size === 0 || metadata.size > 256 * 1024) {
    throw new Error("ui_asset_invalid");
  }
  const text = await readFile(path, "utf8");
  if (
    text.includes("http://") ||
    text.includes("https://") ||
    text.includes("localStorage") ||
    text.includes("sessionStorage") ||
    text.includes("serviceWorker") ||
    text.includes(".innerHTML")
  ) {
    throw new Error("ui_asset_policy_violation");
  }
}

const base = await readFile(
  resolve(root, "app/ui/templates/base.html"),
  "utf8",
);
for (const asset of ["/ui/assets/app.css", "/ui/assets/app.js"]) {
  if (!base.includes(asset)) {
    throw new Error("ui_asset_reference_missing");
  }
}
