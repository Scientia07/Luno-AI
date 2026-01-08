"use client";

import { useEffect, useState } from "react";
import { FlaskConical, Play, CheckCircle2, Code } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";

interface Lab {
  id: string;
  domain: string;
  path: string;
  filename: string;
  title: string;
  description: string;
  code_cells: number;
  markdown_cells: number;
  total_cells: number;
  kernel: string;
}

export default function LabsPage() {
  const [labs, setLabs] = useState<Lab[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedDomain, setSelectedDomain] = useState<string | null>(null);

  useEffect(() => {
    async function loadLabs() {
      try {
        const data = await api.getLabs(selectedDomain || undefined);
        setLabs(data.labs);
      } catch (error) {
        console.error("Failed to load labs:", error);
      } finally {
        setLoading(false);
      }
    }
    loadLabs();
  }, [selectedDomain]);

  // Group labs by domain
  const labsByDomain = labs.reduce((acc, lab) => {
    if (!acc[lab.domain]) {
      acc[lab.domain] = [];
    }
    acc[lab.domain].push(lab);
    return acc;
  }, {} as Record<string, Lab[]>);

  const domains = Object.keys(labsByDomain).sort();

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-pulse text-muted-foreground">Loading...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Interactive Labs</h1>
        <p className="text-muted-foreground mt-2">
          Hands-on Jupyter notebooks for practical experimentation
        </p>
      </div>

      <div className="flex items-center gap-2">
        <FlaskConical className="h-5 w-5 text-muted-foreground" />
        <span className="text-muted-foreground">
          {labs.length} notebooks available
        </span>
      </div>

      {/* Domain Filter */}
      {domains.length > 1 && (
        <div className="flex flex-wrap gap-2">
          <Button
            variant={selectedDomain === null ? "default" : "outline"}
            size="sm"
            onClick={() => setSelectedDomain(null)}
          >
            All
          </Button>
          {domains.map((domain) => (
            <Button
              key={domain}
              variant={selectedDomain === domain ? "default" : "outline"}
              size="sm"
              onClick={() => setSelectedDomain(domain)}
            >
              {domain}
            </Button>
          ))}
        </div>
      )}

      {/* Labs Grid */}
      {selectedDomain ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {labsByDomain[selectedDomain]?.map((lab) => (
            <LabCard key={lab.id} lab={lab} />
          ))}
        </div>
      ) : (
        <div className="space-y-8">
          {domains.map((domain) => (
            <div key={domain}>
              <h2 className="text-xl font-semibold mb-4 capitalize">
                {domain}
              </h2>
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {labsByDomain[domain].map((lab) => (
                  <LabCard key={lab.id} lab={lab} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {labs.length === 0 && (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <FlaskConical className="h-12 w-12 text-muted-foreground mb-4" />
            <p className="text-muted-foreground">No labs available yet</p>
            <p className="text-sm text-muted-foreground mt-1">
              Labs will appear here once created in the labs directory
            </p>
          </CardContent>
        </Card>
      )}

      {/* Getting Started */}
      <Card>
        <CardHeader>
          <CardTitle>Running Labs</CardTitle>
          <CardDescription>
            How to work with interactive notebooks
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-start gap-3">
            <div className="rounded-full bg-primary/10 p-2">
              <span className="text-sm font-medium">1</span>
            </div>
            <div>
              <p className="font-medium">Start Jupyter</p>
              <code className="text-sm bg-muted px-2 py-1 rounded mt-1 inline-block">
                jupyter lab labs/
              </code>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="rounded-full bg-primary/10 p-2">
              <span className="text-sm font-medium">2</span>
            </div>
            <div>
              <p className="font-medium">Open a notebook</p>
              <p className="text-sm text-muted-foreground">
                Navigate to the lab you want to run
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="rounded-full bg-primary/10 p-2">
              <span className="text-sm font-medium">3</span>
            </div>
            <div>
              <p className="font-medium">Run cells</p>
              <p className="text-sm text-muted-foreground">
                Execute code cells and experiment
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function LabCard({ lab }: { lab: Lab }) {
  return (
    <Card className="hover:bg-accent/50 transition-colors">
      <CardHeader>
        <div className="flex items-start justify-between">
          <CardTitle className="text-base">{lab.title}</CardTitle>
          <Badge variant="outline">{lab.kernel}</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <CardDescription className="line-clamp-2">
          {lab.description || "Interactive notebook"}
        </CardDescription>
        <div className="flex items-center gap-4 text-sm text-muted-foreground">
          <div className="flex items-center gap-1">
            <Code className="h-3 w-3" />
            <span>{lab.code_cells} code cells</span>
          </div>
          <div className="flex items-center gap-1">
            <CheckCircle2 className="h-3 w-3" />
            <span>{lab.total_cells} total</span>
          </div>
        </div>
        <p className="text-xs text-muted-foreground font-mono truncate">
          {lab.path}
        </p>
      </CardContent>
    </Card>
  );
}
