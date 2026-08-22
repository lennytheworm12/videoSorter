import type { Metadata } from "next";
import { Phase2KAlignmentClient } from "../../components/phase2k-alignment-client";

export const metadata: Metadata = {
  title: "Phase 2K — Downstream Semantic-Target Alignment",
  description:
    "Post-human-review semantic-target alignment over the sealed Phase 2K D representations. The packet is loaded locally and validated strictly; the workspace is model/scorer blind and never loads predictions or results.",
};

// Client-only route: the live blank packet lives in an external immutable
// directory that does not exist yet, so nothing is read at build time.
export default function Phase2KAlignmentPage() {
  return <Phase2KAlignmentClient />;
}
