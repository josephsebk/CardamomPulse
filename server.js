import express from "express";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;
const WEBAPP_DIR = path.join(__dirname, "cardamom_webapp");

app.use(express.static(WEBAPP_DIR));

app.get("*", (_req, res) => {
  res.sendFile(path.join(WEBAPP_DIR, "index.html"));
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`CardamomPulse server running at http://0.0.0.0:${PORT}`);
});
