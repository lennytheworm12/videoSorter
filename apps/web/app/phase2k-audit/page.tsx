import type { Metadata } from "next";
import { Phase2KAuditClient } from "../../components/phase2k-audit-client";

export const metadata: Metadata = {
  title: "Phase 2K — Transformation Audit Review",
  description:
    "Live transformation-audit human review: operation-level decisions over exact evidence spans, with no downstream results.",
};

export default function Phase2KAuditPage() {
  return <Phase2KAuditClient />;
}
