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
  ArrowRight,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
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

async function getDomains() {
  try {
    const response = await api.getDomains();
    return response.domains;
  } catch (error) {
    console.error("Failed to fetch domains:", error);
    return [];
  }
}

export default async function ExplorePage() {
  const domains = await getDomains();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Explore AI Technologies</h1>
        <p className="text-muted-foreground mt-2">
          Browse 60+ AI technologies across 9 domains. Each technology includes layered
          learning from L0 (overview) to L4 (production optimization).
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {domains.map((domain) => {
          const Icon = domainIcons[domain.id] || BookOpen;
          const bgColor = domainColors[domain.id] || "bg-gray-500";

          return (
            <Link key={domain.id} href={`/explore/${domain.id}`}>
              <Card className="hover:bg-accent/50 transition-colors cursor-pointer h-full group">
                <CardHeader>
                  <div className="flex items-center gap-4">
                    <div className={`p-3 rounded-xl ${bgColor}`}>
                      <Icon className="h-6 w-6 text-white" />
                    </div>
                    <div className="flex-1">
                      <CardTitle className="text-xl flex items-center gap-2">
                        {domain.name}
                        <ArrowRight className="h-4 w-4 opacity-0 group-hover:opacity-100 transition-opacity" />
                      </CardTitle>
                      <Badge variant="secondary" className="mt-1">
                        {domain.tech_count} technologies
                      </Badge>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-base">
                    {domain.description}
                  </CardDescription>
                </CardContent>
              </Card>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
