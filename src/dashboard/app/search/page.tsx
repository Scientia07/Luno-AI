"use client";

import { Suspense, useEffect, useState, useCallback } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import Link from "next/link";
import { Search, BookOpen, FlaskConical, ArrowRight } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api } from "@/lib/api";

function SearchContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const initialQuery = searchParams.get("q") || "";

  const [query, setQuery] = useState(initialQuery);
  const [results, setResults] = useState<{
    integrations: any[];
    research: any[];
    total: number;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("all");

  const performSearch = useCallback(async (searchQuery: string) => {
    if (!searchQuery.trim()) {
      setResults(null);
      return;
    }

    setLoading(true);
    try {
      const data = await api.search(searchQuery, {
        type: activeTab === "all" ? "all" : activeTab,
        limit: 20,
      });
      setResults(data);
    } catch (error) {
      console.error("Search failed:", error);
    } finally {
      setLoading(false);
    }
  }, [activeTab]);

  useEffect(() => {
    if (initialQuery) {
      performSearch(initialQuery);
    }
  }, [initialQuery, performSearch]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    router.push(`/search?q=${encodeURIComponent(query)}`);
    performSearch(query);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Search</h1>
        <p className="text-muted-foreground mt-2">
          Search across technologies, concepts, and research sessions
        </p>
      </div>

      {/* Search Form */}
      <form onSubmit={handleSubmit} className="flex gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
          <Input
            type="search"
            placeholder="Search for technologies, concepts, frameworks..."
            className="pl-9"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <Button type="submit" disabled={loading}>
          {loading ? "Searching..." : "Search"}
        </Button>
      </form>

      {/* Results */}
      {results && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">
              {results.total} results for "{initialQuery}"
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList>
              <TabsTrigger value="all">
                All ({results.total})
              </TabsTrigger>
              <TabsTrigger value="integrations">
                Technologies ({results.integrations.length})
              </TabsTrigger>
              <TabsTrigger value="research">
                Research ({results.research.length})
              </TabsTrigger>
            </TabsList>

            <TabsContent value="all" className="space-y-4 mt-4">
              {results.integrations.length > 0 && (
                <div>
                  <h3 className="font-semibold flex items-center gap-2 mb-3">
                    <BookOpen className="h-4 w-4" />
                    Technologies
                  </h3>
                  <div className="grid gap-3">
                    {results.integrations.slice(0, 5).map((result, i) => (
                      <SearchResultCard key={i} result={result} type="integration" />
                    ))}
                  </div>
                </div>
              )}

              {results.research.length > 0 && (
                <div>
                  <h3 className="font-semibold flex items-center gap-2 mb-3">
                    <FlaskConical className="h-4 w-4" />
                    Research
                  </h3>
                  <div className="grid gap-3">
                    {results.research.slice(0, 5).map((result, i) => (
                      <SearchResultCard key={i} result={result} type="research" />
                    ))}
                  </div>
                </div>
              )}

              {results.total === 0 && (
                <Card>
                  <CardContent className="flex flex-col items-center justify-center py-12">
                    <Search className="h-12 w-12 text-muted-foreground mb-4" />
                    <p className="text-muted-foreground">No results found</p>
                    <p className="text-sm text-muted-foreground mt-1">
                      Try different keywords or browse the explore page
                    </p>
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            <TabsContent value="integrations" className="space-y-3 mt-4">
              {results.integrations.map((result, i) => (
                <SearchResultCard key={i} result={result} type="integration" />
              ))}
              {results.integrations.length === 0 && (
                <p className="text-muted-foreground text-center py-8">
                  No technology results
                </p>
              )}
            </TabsContent>

            <TabsContent value="research" className="space-y-3 mt-4">
              {results.research.map((result, i) => (
                <SearchResultCard key={i} result={result} type="research" />
              ))}
              {results.research.length === 0 && (
                <p className="text-muted-foreground text-center py-8">
                  No research results
                </p>
              )}
            </TabsContent>
          </Tabs>
        </div>
      )}

      {/* Empty State */}
      {!results && !loading && (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16">
            <Search className="h-16 w-16 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium">Start searching</h3>
            <p className="text-muted-foreground mt-1">
              Enter a query to search across all content
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export default function SearchPage() {
  return (
    <Suspense fallback={
      <div className="flex items-center justify-center h-64">
        <div className="animate-pulse text-muted-foreground">Loading...</div>
      </div>
    }>
      <SearchContent />
    </Suspense>
  );
}

function SearchResultCard({
  result,
  type,
}: {
  result: any;
  type: "integration" | "research";
}) {
  if (type === "integration") {
    const domain = result.domain || result.metadata?.domain;
    const technology = result.technology || result.metadata?.technology;

    return (
      <Link href={`/explore/${domain}/${technology}`}>
        <Card className="hover:bg-accent/50 transition-colors cursor-pointer">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base flex items-center gap-2">
                {technology?.replace(/-/g, " ") || "Unknown"}
                <ArrowRight className="h-4 w-4 opacity-50" />
              </CardTitle>
              <Badge variant="secondary">{domain}</Badge>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground line-clamp-2">
              {result.snippet || result.content?.slice(0, 200)}
            </p>
          </CardContent>
        </Card>
      </Link>
    );
  }

  return (
    <Link href={`/research?session=${result.session}`}>
      <Card className="hover:bg-accent/50 transition-colors cursor-pointer">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base flex items-center gap-2">
              {result.session?.replace(/-/g, " ") || "Research"}
              <ArrowRight className="h-4 w-4 opacity-50" />
            </CardTitle>
            <Badge variant="outline">{result.file}</Badge>
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground line-clamp-2">
            {result.snippet?.slice(0, 200)}
          </p>
        </CardContent>
      </Card>
    </Link>
  );
}
