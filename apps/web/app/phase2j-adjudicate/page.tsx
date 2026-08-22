import { readFileSync } from "node:fs";
import path from "node:path";
import type { Metadata } from "next";
import { Phase2JAdjudicationClient } from "../../components/phase2j-adjudication-client";
import { sanitizeAdjudicationPacket } from "../../lib/phase2j-adjudication";

export const metadata: Metadata = {
  title: "Phase 2J — Human vs Sol Adjudication",
  description:
    "Post-Pass-A human adjudication of Sol proposals for the 30 locked Phase 2J Bronze windows. Sol is a second opinion, never gold.",
};

// The generated sanitized adjudication packet is read once at static-build
// time; only the strictly validated payload is serialized into the page.
function loadSanitizedAdjudicationPayload() {
  const packetPath = path.join(
    process.cwd(),
    "../../data/phase2j/phase2j-adjudication-packet-v1.json",
  );
  const raw = JSON.parse(readFileSync(packetPath, "utf8")) as unknown;
  return sanitizeAdjudicationPacket(raw);
}

export default function Phase2JAdjudicationPage() {
  const payload = loadSanitizedAdjudicationPayload();
  return <Phase2JAdjudicationClient payload={payload} />;
}
