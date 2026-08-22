import { readFileSync } from "node:fs";
import path from "node:path";
import type { Metadata } from "next";
import { Phase2JReviewClient } from "../../components/phase2j-review-client";
import { sanitizePacket } from "../../lib/phase2j-review";

export const metadata: Metadata = {
  title: "Phase 2J — Bronze Endpoint Review",
  description:
    "Pass A human endpoint annotation for the 30 locked Phase 2J Bronze windows.",
};

// The packet is read once at static-build time; only the sanitized payload is
// serialized into the page for the client.
function loadSanitizedPayload() {
  const packetPath = path.join(
    process.cwd(),
    "../../data/phase2j/endpoint-annotation-packet-v1.json",
  );
  const raw = JSON.parse(readFileSync(packetPath, "utf8")) as unknown;
  return sanitizePacket(raw);
}

export default function Phase2JReviewPage() {
  const payload = loadSanitizedPayload();
  return <Phase2JReviewClient payload={payload} />;
}
