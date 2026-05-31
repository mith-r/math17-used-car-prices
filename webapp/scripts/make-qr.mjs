// Generate a QR code PNG pointing at the deployed site.
// Usage: node scripts/make-qr.mjs <url>
import QRCode from "qrcode";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const url = process.argv[2];
if (!url) {
  console.error("Usage: node scripts/make-qr.mjs <deployed-url>");
  process.exit(1);
}

const here = dirname(fileURLToPath(import.meta.url));
const out = join(here, "..", "..", "qr.png"); // save into the project folder

await QRCode.toFile(out, url, {
  width: 900,
  margin: 2,
  color: { dark: "#0b1220", light: "#ffffff" },
});
console.log(`QR for ${url} -> ${out}`);
