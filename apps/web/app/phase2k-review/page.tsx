import type { Metadata } from "next";
import { Phase2KReviewClient } from "../../components/phase2k-review-client";

export const metadata: Metadata = {
  title: "Phase 2K — Blinded Semantic-Recoverability Review",
  description:
    "Blinded human review of Phase 2K semantic-recoverability presentations. The review packet is loaded locally and never uploaded; the condition/radius mapping is never read.",
};

// Client-only route: the live blank packet lives in an external immutable
// directory that does not exist yet, so nothing is read at build time.
export default function Phase2KReviewPage() {
  return <Phase2KReviewClient />;
}
