import { ClassView } from "@/components/classes/ClassView";

export function generateStaticParams() {
  return ["2021", "2022", "2023", "2024", "2025", "2026"].map((year) => ({ year }));
}

export default async function ClassYearPage({
  params,
}: {
  params: Promise<{ year: string }>;
}) {
  const { year } = await params;
  return <ClassView year={Number(year)} />;
}
