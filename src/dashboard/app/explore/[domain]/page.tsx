import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft, ArrowRight, Clock, Signal } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";

interface DomainPageProps {
  params: { domain: string };
}

const difficultyColors: Record<string, string> = {
  beginner: "bg-green-500",
  intermediate: "bg-yellow-500",
  advanced: "bg-red-500",
};

const statusVariants: Record<string, "default" | "secondary" | "outline"> = {
  ready: "default",
  beta: "secondary",
  "coming-soon": "outline",
};

async function getDomainData(domainId: string) {
  try {
    const domain = await api.getDomain(domainId);
    return domain;
  } catch (error) {
    console.error("Failed to fetch domain:", error);
    return null;
  }
}

export default async function DomainPage({ params }: DomainPageProps) {
  const domain = await getDomainData(params.domain);

  if (!domain) {
    notFound();
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Button variant="ghost" size="icon" asChild>
          <Link href="/explore">
            <ArrowLeft className="h-4 w-4" />
          </Link>
        </Button>
        <div>
          <h1 className="text-3xl font-bold tracking-tight">{domain.name}</h1>
          <p className="text-muted-foreground mt-1">{domain.description}</p>
        </div>
      </div>

      {/* Stats */}
      <div className="flex gap-4">
        <Badge variant="secondary" className="text-base py-1 px-3">
          {domain.technologies.length} Technologies
        </Badge>
      </div>

      {/* Technologies Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {domain.technologies.map((tech) => (
          <Link
            key={tech.id}
            href={`/explore/${params.domain}/${tech.id}`}
          >
            <Card className="hover:bg-accent/50 transition-colors cursor-pointer h-full group">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <CardTitle className="text-lg flex items-center gap-2">
                    {tech.name}
                    <ArrowRight className="h-4 w-4 opacity-0 group-hover:opacity-100 transition-opacity" />
                  </CardTitle>
                  <Badge variant={statusVariants[tech.status] || "default"}>
                    {tech.status}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <CardDescription className="line-clamp-2">
                  {tech.tagline}
                </CardDescription>
                <div className="flex items-center gap-4 text-sm text-muted-foreground">
                  <div className="flex items-center gap-1">
                    <Signal className="h-3 w-3" />
                    <span className="capitalize">{tech.difficulty}</span>
                  </div>
                  {tech.quick_start_time && (
                    <div className="flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      <span>{tech.quick_start_time}</span>
                    </div>
                  )}
                </div>
                <div className="flex gap-1">
                  {["L0", "L1", "L2", "L3", "L4"].map((level, i) => (
                    <div
                      key={level}
                      className={`h-1.5 flex-1 rounded-full ${
                        i === 0 ? "bg-primary" : "bg-muted"
                      }`}
                    />
                  ))}
                </div>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>

      {domain.technologies.length === 0 && (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <p className="text-muted-foreground">
              No technologies available in this domain yet.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
