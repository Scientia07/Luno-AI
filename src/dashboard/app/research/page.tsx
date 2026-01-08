"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  FlaskConical,
  Calendar,
  FileText,
  ArrowLeft,
  Tag,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api, ResearchSession } from "@/lib/api";

function ResearchContent() {
  const searchParams = useSearchParams();
  const selectedSession = searchParams.get("session");

  const [sessions, setSessions] = useState<ResearchSession[]>([]);
  const [sessionDetail, setSessionDetail] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadSessions() {
      try {
        const data = await api.getResearchSessions();
        setSessions(data.sessions.filter(s => s.id !== "topics"));
      } catch (error) {
        console.error("Failed to load sessions:", error);
      } finally {
        setLoading(false);
      }
    }
    loadSessions();
  }, []);

  useEffect(() => {
    async function loadSessionDetail() {
      if (!selectedSession) {
        setSessionDetail(null);
        return;
      }
      try {
        const data = await api.getResearchSession(selectedSession);
        setSessionDetail(data);
      } catch (error) {
        console.error("Failed to load session:", error);
      }
    }
    loadSessionDetail();
  }, [selectedSession]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-pulse text-muted-foreground">Loading...</div>
      </div>
    );
  }

  // Session Detail View
  if (sessionDetail) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" asChild>
            <a href="/research">
              <ArrowLeft className="h-4 w-4" />
            </a>
          </Button>
          <div>
            <h1 className="text-2xl font-bold tracking-tight">
              {sessionDetail.title}
            </h1>
            <div className="flex items-center gap-2 mt-1 text-muted-foreground">
              <Calendar className="h-4 w-4" />
              <span>{sessionDetail.date}</span>
            </div>
          </div>
        </div>

        {sessionDetail.tags && sessionDetail.tags.length > 0 && (
          <div className="flex items-center gap-2">
            <Tag className="h-4 w-4 text-muted-foreground" />
            {sessionDetail.tags.map((tag: string) => (
              <Badge key={tag} variant="secondary">
                {tag}
              </Badge>
            ))}
          </div>
        )}

        <Tabs defaultValue="readme">
          <TabsList>
            {sessionDetail.content?.["README.md"] && (
              <TabsTrigger value="readme">Overview</TabsTrigger>
            )}
            {sessionDetail.content?.["findings.md"] && (
              <TabsTrigger value="findings">Findings</TabsTrigger>
            )}
            {sessionDetail.content?.["sources.md"] && (
              <TabsTrigger value="sources">Sources</TabsTrigger>
            )}
          </TabsList>

          {sessionDetail.content?.["README.md"] && (
            <TabsContent value="readme" className="mt-4">
              <Card>
                <CardContent className="pt-6">
                  <ScrollArea className="h-[600px]">
                    <div className="prose dark:prose-invert max-w-none pr-4">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {sessionDetail.content["README.md"]}
                      </ReactMarkdown>
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </TabsContent>
          )}

          {sessionDetail.content?.["findings.md"] && (
            <TabsContent value="findings" className="mt-4">
              <Card>
                <CardContent className="pt-6">
                  <ScrollArea className="h-[600px]">
                    <div className="prose dark:prose-invert max-w-none pr-4">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {sessionDetail.content["findings.md"]}
                      </ReactMarkdown>
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </TabsContent>
          )}

          {sessionDetail.content?.["sources.md"] && (
            <TabsContent value="sources" className="mt-4">
              <Card>
                <CardContent className="pt-6">
                  <ScrollArea className="h-[600px]">
                    <div className="prose dark:prose-invert max-w-none pr-4">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {sessionDetail.content["sources.md"]}
                      </ReactMarkdown>
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </TabsContent>
          )}
        </Tabs>

        {/* Files List */}
        {sessionDetail.files && sessionDetail.files.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-4 w-4" />
                Session Files
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-2 md:grid-cols-2 lg:grid-cols-3">
                {sessionDetail.files.map((file: any) => (
                  <div
                    key={file.name}
                    className="flex items-center gap-2 p-2 rounded-lg bg-muted"
                  >
                    <FileText className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm">{file.name}</span>
                    {file.size && (
                      <span className="text-xs text-muted-foreground ml-auto">
                        {(file.size / 1024).toFixed(1)} KB
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    );
  }

  // Sessions List View
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Research Vault</h1>
        <p className="text-muted-foreground mt-2">
          Browse deep research sessions and findings
        </p>
      </div>

      <div className="flex items-center gap-2">
        <FlaskConical className="h-5 w-5 text-muted-foreground" />
        <span className="text-muted-foreground">
          {sessions.length} research sessions
        </span>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {sessions.map((session) => (
          <a key={session.id} href={`/research?session=${session.id}`}>
            <Card className="hover:bg-accent/50 transition-colors cursor-pointer h-full">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <CardTitle className="text-lg">{session.title}</CardTitle>
                  <Badge variant="outline">{session.date}</Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <CardDescription className="line-clamp-2">
                  {session.description || "Deep research session"}
                </CardDescription>
                {session.tags && session.tags.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {session.tags.slice(0, 4).map((tag) => (
                      <Badge key={tag} variant="secondary" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </a>
        ))}
      </div>

      {sessions.length === 0 && (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <FlaskConical className="h-12 w-12 text-muted-foreground mb-4" />
            <p className="text-muted-foreground">No research sessions yet</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export default function ResearchPage() {
  return (
    <Suspense fallback={
      <div className="flex items-center justify-center h-64">
        <div className="animate-pulse text-muted-foreground">Loading...</div>
      </div>
    }>
      <ResearchContent />
    </Suspense>
  );
}
