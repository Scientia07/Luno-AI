"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { Markdown } from "@/components/ui/markdown";
import {
  ArrowLeft,
  BookOpen,
  CheckCircle2,
  Circle,
  Clock,
  ExternalLink,
  Bookmark,
  BookmarkCheck,
  Copy,
  Check,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Checkbox } from "@/components/ui/checkbox";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { api, TechnologyDetail, TechnologyProgress } from "@/lib/api";

const layerNames = [
  "Overview",
  "Getting Started",
  "Core Concepts",
  "Advanced Topics",
  "Production",
];

const layerDescriptions = [
  "What is it and why use it",
  "Installation and first steps",
  "Deep dive into key concepts",
  "Advanced patterns and techniques",
  "Production optimization and deployment",
];

export default function TechnologyPage() {
  const params = useParams();
  const domain = params.domain as string;
  const techId = params.tech as string;

  const [tech, setTech] = useState<TechnologyDetail | null>(null);
  const [progress, setProgress] = useState<TechnologyProgress | null>(null);
  const [loading, setLoading] = useState(true);
  const [copiedCode, setCopiedCode] = useState<string | null>(null);
  const [activeLayer, setActiveLayer] = useState("0");

  useEffect(() => {
    async function loadData() {
      try {
        const [techData, progressData] = await Promise.all([
          api.getTechnology(domain, techId),
          api.getTechnologyProgress(domain, techId),
        ]);
        setTech(techData);
        setProgress(progressData);
      } catch (error) {
        console.error("Failed to load technology:", error);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, [domain, techId]);

  const handleLayerComplete = async (layer: number, completed: boolean) => {
    try {
      await api.updateProgress(domain, techId, layer, completed);
      const updated = await api.getTechnologyProgress(domain, techId);
      setProgress(updated);
    } catch (error) {
      console.error("Failed to update progress:", error);
    }
  };

  const handleBookmark = async () => {
    try {
      await api.toggleBookmark(domain, techId);
      const updated = await api.getTechnologyProgress(domain, techId);
      setProgress(updated);
    } catch (error) {
      console.error("Failed to toggle bookmark:", error);
    }
  };

  const copyCode = (code: string, id: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-pulse text-muted-foreground">Loading...</div>
      </div>
    );
  }

  if (!tech) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-4">
        <p className="text-muted-foreground">Technology not found</p>
        <Button asChild>
          <Link href={`/explore/${domain}`}>Back to domain</Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" asChild>
            <Link href={`/explore/${domain}`}>
              <ArrowLeft className="h-4 w-4" />
            </Link>
          </Button>
          <div>
            <h1 className="text-3xl font-bold tracking-tight">{tech.name}</h1>
            <p className="text-muted-foreground mt-1">{tech.tagline}</p>
          </div>
        </div>
        <Button
          variant={progress?.bookmarked ? "default" : "outline"}
          size="icon"
          onClick={handleBookmark}
        >
          {progress?.bookmarked ? (
            <BookmarkCheck className="h-4 w-4" />
          ) : (
            <Bookmark className="h-4 w-4" />
          )}
        </Button>
      </div>

      {/* Progress Bar */}
      <Card>
        <CardContent className="py-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Learning Progress</span>
            <span className="text-sm text-muted-foreground">
              Layer {progress?.current_layer || 0} of 4
            </span>
          </div>
          <div className="flex gap-2">
            {[0, 1, 2, 3, 4].map((level) => {
              const isCompleted = progress?.layers?.[level]?.completed;
              return (
                <div
                  key={level}
                  className={`h-2 flex-1 rounded-full transition-colors ${
                    isCompleted ? "bg-primary" : "bg-muted"
                  }`}
                />
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Overview Section */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">What is it?</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              {tech.overview.what || "No description available"}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Why use it?</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              {tech.overview.why || "No description available"}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Key Tools</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-1">
              {tech.overview.tools?.slice(0, 5).map((tool) => (
                <Badge key={tool} variant="secondary" className="text-xs">
                  {tool}
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Best For</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              {tech.overview.best_for || "General purpose"}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Quick Start */}
      {tech.quick_start && (tech.quick_start.install || tech.quick_start.code) && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="h-4 w-4" />
              Quick Start
              <Badge variant="secondary">{tech.quick_start.time}</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {tech.quick_start.install && (
              <div>
                <p className="text-sm font-medium mb-2">Installation</p>
                <div className="relative">
                  <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                    <code>{tech.quick_start.install}</code>
                  </pre>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="absolute top-2 right-2"
                    onClick={() => copyCode(tech.quick_start.install, "install")}
                  >
                    {copiedCode === "install" ? (
                      <Check className="h-4 w-4" />
                    ) : (
                      <Copy className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </div>
            )}
            {tech.quick_start.code && (
              <div>
                <p className="text-sm font-medium mb-2">Example</p>
                <div className="relative">
                  <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                    <code>{tech.quick_start.code}</code>
                  </pre>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="absolute top-2 right-2"
                    onClick={() => copyCode(tech.quick_start.code, "code")}
                  >
                    {copiedCode === "code" ? (
                      <Check className="h-4 w-4" />
                    ) : (
                      <Copy className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Learning Layers */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="h-4 w-4" />
            Learning Layers
          </CardTitle>
          <CardDescription>
            Progress through each layer to master this technology
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs value={activeLayer} onValueChange={setActiveLayer}>
            <TabsList className="grid grid-cols-5 w-full">
              {[0, 1, 2, 3, 4].map((level) => {
                const layer = tech.layers.find((l) => l.level === level);
                const isCompleted = progress?.layers?.[level]?.completed;
                return (
                  <TabsTrigger
                    key={level}
                    value={level.toString()}
                    className="flex items-center gap-1"
                  >
                    {isCompleted ? (
                      <CheckCircle2 className="h-3 w-3 text-green-500" />
                    ) : (
                      <Circle className="h-3 w-3" />
                    )}
                    L{level}
                  </TabsTrigger>
                );
              })}
            </TabsList>

            {[0, 1, 2, 3, 4].map((level) => {
              const layer = tech.layers.find((l) => l.level === level);
              const isCompleted = progress?.layers?.[level]?.completed;

              return (
                <TabsContent key={level} value={level.toString()} className="mt-4">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="font-semibold">
                          {layer?.name || layerNames[level]}
                        </h3>
                        <p className="text-sm text-muted-foreground">
                          {layerDescriptions[level]}
                        </p>
                      </div>
                      <div className="flex items-center gap-2">
                        <Checkbox
                          checked={isCompleted}
                          onCheckedChange={(checked) =>
                            handleLayerComplete(level, checked as boolean)
                          }
                        />
                        <span className="text-sm">Mark complete</span>
                      </div>
                    </div>

                    <Separator />

                    {layer?.content ? (
                      <ScrollArea className="h-[500px] pr-4">
                        <Markdown content={layer.content} />
                      </ScrollArea>
                    ) : (
                      <div className="flex items-center justify-center h-32 text-muted-foreground">
                        Content coming soon for this layer
                      </div>
                    )}
                  </div>
                </TabsContent>
              );
            })}
          </Tabs>
        </CardContent>
      </Card>

      {/* Code Examples */}
      {tech.code_examples && tech.code_examples.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Code Examples</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {tech.code_examples.slice(0, 5).map((example, i) => (
                <div key={i}>
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-sm font-medium">{example.title}</p>
                    <Badge variant="outline">{example.language}</Badge>
                  </div>
                  <div className="relative">
                    <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm">
                      <code>{example.code}</code>
                    </pre>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="absolute top-2 right-2"
                      onClick={() => copyCode(example.code, `example-${i}`)}
                    >
                      {copiedCode === `example-${i}` ? (
                        <Check className="h-4 w-4" />
                      ) : (
                        <Copy className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Resources & Related */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Resources */}
        {tech.resources && tech.resources.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Resources</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {tech.resources.map((resource, i) => (
                  <a
                    key={i}
                    href={resource.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 text-sm text-primary hover:underline"
                  >
                    <ExternalLink className="h-3 w-3" />
                    {resource.title}
                  </a>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Related Technologies */}
        {tech.related_tech && tech.related_tech.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Related Technologies</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {tech.related_tech.map((related) => (
                  <Badge key={related} variant="secondary">
                    {related}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
