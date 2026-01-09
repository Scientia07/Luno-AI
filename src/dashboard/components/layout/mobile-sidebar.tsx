"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { X } from "lucide-react";
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
  Home,
  Search,
  BookOpen,
  FlaskConical,
  BarChart3,
  Bookmark,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";

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
  "visual-ai": "text-blue-500",
  "generative": "text-pink-500",
  "audio": "text-amber-500",
  "llms": "text-emerald-500",
  "agents": "text-violet-500",
  "ml": "text-cyan-500",
  "deploy": "text-indigo-500",
  "robotics": "text-red-500",
  "specialized": "text-lime-500",
};

const mainNav = [
  { name: "Home", href: "/", icon: Home },
  { name: "Explore", href: "/explore", icon: BookOpen },
  { name: "Search", href: "/search", icon: Search },
  { name: "Progress", href: "/progress", icon: BarChart3 },
  { name: "Research", href: "/research", icon: FlaskConical },
  { name: "Labs", href: "/labs", icon: FlaskConical },
  { name: "Bookmarks", href: "/progress?tab=bookmarks", icon: Bookmark },
];

const domains = [
  { id: "visual-ai", name: "Visual AI" },
  { id: "generative", name: "Generative" },
  { id: "audio", name: "Audio AI" },
  { id: "llms", name: "LLMs" },
  { id: "agents", name: "Agents" },
  { id: "ml", name: "Classical ML" },
  { id: "deploy", name: "Deployment" },
  { id: "robotics", name: "Robotics" },
  { id: "specialized", name: "Specialized" },
];

interface MobileSidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

export function MobileSidebar({ isOpen, onClose }: MobileSidebarProps) {
  const pathname = usePathname();

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-40 bg-black/50 md:hidden"
        onClick={onClose}
      />

      {/* Sidebar */}
      <div className="fixed inset-y-0 left-0 z-50 w-64 bg-background border-r md:hidden">
        <div className="flex h-full max-h-screen flex-col gap-2">
          <div className="flex h-14 items-center justify-between border-b px-4">
            <Link href="/" className="flex items-center gap-2 font-semibold" onClick={onClose}>
              <Sparkles className="h-6 w-6 text-primary" />
              <span>Luno-AI</span>
            </Link>
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="h-5 w-5" />
              <span className="sr-only">Close menu</span>
            </Button>
          </div>
          <ScrollArea className="flex-1 px-3">
            <div className="space-y-1 py-2">
              {mainNav.map((item) => {
                const Icon = item.icon;
                const isActive = pathname === item.href ||
                  (item.href !== "/" && pathname.startsWith(item.href.split("?")[0]));
                return (
                  <Link
                    key={item.name}
                    href={item.href}
                    onClick={onClose}
                    className={cn(
                      "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-all hover:bg-accent",
                      isActive
                        ? "bg-accent text-accent-foreground"
                        : "text-muted-foreground"
                    )}
                  >
                    <Icon className="h-4 w-4" />
                    {item.name}
                  </Link>
                );
              })}
            </div>
            <Separator className="my-4" />
            <div className="py-2">
              <h4 className="mb-2 px-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                Domains
              </h4>
              <div className="space-y-1">
                {domains.map((domain) => {
                  const Icon = domainIcons[domain.id] || BookOpen;
                  const colorClass = domainColors[domain.id] || "text-gray-500";
                  const isActive = pathname.includes(`/explore/${domain.id}`);
                  return (
                    <Link
                      key={domain.id}
                      href={`/explore/${domain.id}`}
                      onClick={onClose}
                      className={cn(
                        "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-all hover:bg-accent",
                        isActive
                          ? "bg-accent text-accent-foreground"
                          : "text-muted-foreground"
                      )}
                    >
                      <Icon className={cn("h-4 w-4", colorClass)} />
                      {domain.name}
                    </Link>
                  );
                })}
              </div>
            </div>
          </ScrollArea>
        </div>
      </div>
    </>
  );
}
