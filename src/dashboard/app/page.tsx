import Link from "next/link";
import {
  Eye,
  Sparkles,
  Volume2,
  MessageSquare,
  Bot,
  TrendingUp,
  Cloud,
  Cpu,
  Zap,
  BookOpen,
  FlaskConical,
  ArrowRight,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";

const domainIcons: Record<string, React.ElementType> = {
  "visual-ai": Eye,
  "generative": Sparkles,
  "audio": Volume2,
  "llms": MessageSquare,
  "agents": Bot,
  "ml": TrendingUp,
  "deploy": Cloud,
  "robotics": Cpu,
  "specialized": Zap,
};

const domainColors: Record<string, string> = {
  "visual-ai": "bg-blue-500",
  "generative": "bg-pink-500",
  "audio": "bg-amber-500",
  "llms": "bg-emerald-500",
  "agents": "bg-violet-500",
  "ml": "bg-cyan-500",
  "deploy": "bg-indigo-500",
  "robotics": "bg-red-500",
  "specialized": "bg-lime-500",
};

async function getHomeData() {
  try {
    const [domainsRes, stats, progress] = await Promise.all([
      api.getDomains(),
      api.getStats(),
      api.getProgressSummary(),
    ]);
    return { domains: domainsRes.domains, stats, progress };
  } catch (error) {
    console.error("Failed to fetch home data:", error);
    return { domains: [], stats: null, progress: null };
  }
}

export default async function HomePage() {
  const { domains, stats, progress } = await getHomeData();

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="flex flex-col gap-4">
        <h1 className="text-3xl font-bold tracking-tight">
          Welcome to Luno-AI
        </h1>
        <p className="text-muted-foreground max-w-2xl">
          Your gateway to AI technology exploration. Learn through layered depth
          (L0-L4), from high-level concepts to production implementations.
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Domains</CardTitle>
            <BookOpen className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats?.domains || 9}</div>
            <p className="text-xs text-muted-foreground">AI technology categories</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Technologies</CardTitle>
            <Cpu className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats?.technologies || 60}</div>
            <p className="text-xs text-muted-foreground">Integration PRDs available</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Research</CardTitle>
            <FlaskConical className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats?.research_sessions || 12}</div>
            <p className="text-xs text-muted-foreground">Deep research sessions</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Progress</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {progress?.overall?.percentage?.toFixed(1) || 0}%
            </div>
            <Progress value={progress?.overall?.percentage || 0} className="mt-2" />
          </CardContent>
        </Card>
      </div>

      {/* Domains Grid */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-semibold tracking-tight">Explore Domains</h2>
          <Button variant="ghost" asChild>
            <Link href="/explore">
              View all <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {domains.map((domain) => {
            const Icon = domainIcons[domain.id] || BookOpen;
            const bgColor = domainColors[domain.id] || "bg-gray-500";
            return (
              <Link key={domain.id} href={`/explore/${domain.id}`}>
                <Card className="hover:bg-accent/50 transition-colors cursor-pointer h-full">
                  <CardHeader>
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-lg ${bgColor}`}>
                        <Icon className="h-5 w-5 text-white" />
                      </div>
                      <div>
                        <CardTitle className="text-lg">{domain.name}</CardTitle>
                        <Badge variant="secondary" className="mt-1">
                          {domain.tech_count} technologies
                        </Badge>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <CardDescription>{domain.description}</CardDescription>
                  </CardContent>
                </Card>
              </Link>
            );
          })}
        </div>
      </div>

      {/* Quick Actions */}
      <div>
        <h2 className="text-2xl font-semibold tracking-tight mb-4">Quick Actions</h2>
        <div className="grid gap-4 md:grid-cols-3">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Bot className="h-5 w-5 text-violet-500" />
                Learn Agents
              </CardTitle>
              <CardDescription>
                Start with LangGraph, CrewAI, and autonomous AI systems
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button asChild>
                <Link href="/explore/agents">
                  Start Learning <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MessageSquare className="h-5 w-5 text-emerald-500" />
                Explore LLMs
              </CardTitle>
              <CardDescription>
                Foundation models, fine-tuning, and local inference
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button asChild>
                <Link href="/explore/llms">
                  Start Learning <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FlaskConical className="h-5 w-5 text-blue-500" />
                Research Vault
              </CardTitle>
              <CardDescription>
                Browse deep research sessions and findings
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button variant="outline" asChild>
                <Link href="/research">
                  Browse Research <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
