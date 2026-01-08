"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  BarChart3,
  BookmarkCheck,
  Clock,
  TrendingUp,
  CheckCircle2,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api, ProgressSummary } from "@/lib/api";
import { formatDate, getDomainColor } from "@/lib/utils";

export default function ProgressPage() {
  const [progress, setProgress] = useState<ProgressSummary | null>(null);
  const [bookmarks, setBookmarks] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [progressData, bookmarksData] = await Promise.all([
          api.getProgressSummary(),
          api.getBookmarks(),
        ]);
        setProgress(progressData);
        setBookmarks(bookmarksData.bookmarks || []);
      } catch (error) {
        console.error("Failed to load progress:", error);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

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
        <h1 className="text-3xl font-bold tracking-tight">Your Progress</h1>
        <p className="text-muted-foreground mt-2">
          Track your learning journey across all AI technologies
        </p>
      </div>

      {/* Overview Stats */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Overall Progress</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {progress?.overall?.percentage?.toFixed(1) || 0}%
            </div>
            <Progress
              value={progress?.overall?.percentage || 0}
              className="mt-2"
            />
            <p className="text-xs text-muted-foreground mt-2">
              {progress?.overall?.completed || 0} of {progress?.overall?.total || 0} layers completed
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Completed Layers</CardTitle>
            <CheckCircle2 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {progress?.overall?.completed || 0}
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              Across all technologies
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Bookmarks</CardTitle>
            <BookmarkCheck className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{bookmarks.length}</div>
            <p className="text-xs text-muted-foreground mt-2">
              Saved for later
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Recent Activity</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {progress?.recent_activity?.length || 0}
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              Completions this week
            </p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="domains">
        <TabsList>
          <TabsTrigger value="domains">By Domain</TabsTrigger>
          <TabsTrigger value="activity">Recent Activity</TabsTrigger>
          <TabsTrigger value="bookmarks">Bookmarks</TabsTrigger>
        </TabsList>

        {/* Domain Progress */}
        <TabsContent value="domains" className="space-y-4 mt-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {progress?.by_domain?.map((domain) => (
              <Card key={domain.domain}>
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base capitalize">
                      {domain.domain.replace("-", " ")}
                    </CardTitle>
                    <Badge variant="secondary">
                      {domain.percentage.toFixed(0)}%
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <Progress
                    value={domain.percentage}
                    className="h-2"
                    indicatorClassName={`bg-[${getDomainColor(domain.domain)}]`}
                  />
                  <p className="text-xs text-muted-foreground mt-2">
                    {domain.completed} of {domain.total} technologies mastered
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Recent Activity */}
        <TabsContent value="activity" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Completions</CardTitle>
              <CardDescription>
                Your latest learning achievements
              </CardDescription>
            </CardHeader>
            <CardContent>
              {progress?.recent_activity && progress.recent_activity.length > 0 ? (
                <div className="space-y-4">
                  {progress.recent_activity.map((activity, i) => (
                    <Link
                      key={i}
                      href={`/explore/${activity.domain}/${activity.tech}`}
                    >
                      <div className="flex items-center justify-between p-3 rounded-lg hover:bg-accent transition-colors">
                        <div className="flex items-center gap-3">
                          <CheckCircle2 className="h-5 w-5 text-green-500" />
                          <div>
                            <p className="font-medium capitalize">
                              {activity.tech.replace(/-/g, " ")}
                            </p>
                            <p className="text-sm text-muted-foreground">
                              Completed Layer {activity.layer}
                            </p>
                          </div>
                        </div>
                        <div className="text-right">
                          <Badge variant="outline" className="capitalize">
                            {activity.domain.replace("-", " ")}
                          </Badge>
                          <p className="text-xs text-muted-foreground mt-1">
                            {formatDate(activity.completed_at)}
                          </p>
                        </div>
                      </div>
                    </Link>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-12">
                  <Clock className="h-12 w-12 text-muted-foreground mb-4" />
                  <p className="text-muted-foreground">No activity yet</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Start learning to track your progress
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Bookmarks */}
        <TabsContent value="bookmarks" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Bookmarked Technologies</CardTitle>
              <CardDescription>
                Technologies you've saved for later
              </CardDescription>
            </CardHeader>
            <CardContent>
              {bookmarks.length > 0 ? (
                <div className="space-y-3">
                  {bookmarks.map((bookmark, i) => (
                    <Link
                      key={i}
                      href={`/explore/${bookmark.domain}/${bookmark.tech}`}
                    >
                      <div className="flex items-center justify-between p-3 rounded-lg hover:bg-accent transition-colors">
                        <div className="flex items-center gap-3">
                          <BookmarkCheck className="h-5 w-5 text-primary" />
                          <div>
                            <p className="font-medium capitalize">
                              {bookmark.tech?.replace(/-/g, " ")}
                            </p>
                            <p className="text-sm text-muted-foreground capitalize">
                              {bookmark.domain?.replace("-", " ")}
                            </p>
                          </div>
                        </div>
                        <p className="text-xs text-muted-foreground">
                          {formatDate(bookmark.created_at)}
                        </p>
                      </div>
                    </Link>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-12">
                  <BookmarkCheck className="h-12 w-12 text-muted-foreground mb-4" />
                  <p className="text-muted-foreground">No bookmarks yet</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Bookmark technologies while browsing
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
