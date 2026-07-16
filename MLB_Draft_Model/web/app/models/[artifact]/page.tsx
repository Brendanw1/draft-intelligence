import { ModelCardView } from "@/components/models/ModelCardView";

export function generateStaticParams() {
  return [
    { artifact: "fg-draft-hitter" },
    { artifact: "fg-draft-pitcher" },
    { artifact: "tier2-predraft-hitter" },
    { artifact: "tier2-predraft-pitcher" },
  ];
}

export default async function ModelCardPage({
  params,
}: {
  params: Promise<{ artifact: string }>;
}) {
  const { artifact } = await params;
  return <ModelCardView artifactKey={artifact} />;
}
