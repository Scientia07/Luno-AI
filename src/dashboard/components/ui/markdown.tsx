"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { useState } from "react";
import { Copy, Check } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "./button";

interface MarkdownProps {
  content: string;
  className?: string;
}

export function Markdown({ content, className }: MarkdownProps) {
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  const copyCode = (code: string, id: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  return (
    <div
      className={cn(
        "prose prose-sm dark:prose-invert max-w-none",
        // Headings
        "prose-headings:font-semibold prose-headings:tracking-tight",
        "prose-h1:text-2xl prose-h2:text-xl prose-h3:text-lg prose-h4:text-base",
        "prose-h1:mt-6 prose-h1:mb-4 prose-h2:mt-5 prose-h2:mb-3 prose-h3:mt-4 prose-h3:mb-2",
        // Paragraphs
        "prose-p:text-muted-foreground prose-p:leading-relaxed",
        // Links
        "prose-a:text-primary prose-a:no-underline hover:prose-a:underline",
        // Lists
        "prose-ul:my-2 prose-ol:my-2 prose-li:my-0.5",
        "prose-li:text-muted-foreground",
        // Code inline
        "prose-code:text-primary prose-code:bg-muted prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sm",
        "prose-code:before:content-none prose-code:after:content-none",
        // Strong and emphasis
        "prose-strong:text-foreground prose-strong:font-semibold",
        "prose-em:text-muted-foreground",
        // Blockquotes
        "prose-blockquote:border-l-primary prose-blockquote:text-muted-foreground prose-blockquote:not-italic",
        // Tables
        "prose-table:text-sm",
        "prose-th:text-left prose-th:font-semibold prose-th:text-foreground prose-th:border-b prose-th:border-border prose-th:pb-2",
        "prose-td:py-2 prose-td:border-b prose-td:border-border/50",
        // Horizontal rule
        "prose-hr:border-border",
        className
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          // Custom code block with copy button
          pre({ children, ...props }) {
            // Extract code content for copy button
            const codeElement = children as React.ReactElement;
            const codeContent = extractTextContent(codeElement);
            const codeId = `code-${Math.random().toString(36).substr(2, 9)}`;

            return (
              <div className="relative group not-prose my-4">
                <pre
                  className="bg-muted/50 border border-border rounded-lg p-4 overflow-x-auto text-sm"
                  {...props}
                >
                  {children}
                </pre>
                <Button
                  variant="ghost"
                  size="icon"
                  className="absolute top-2 right-2 h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity"
                  onClick={() => copyCode(codeContent, codeId)}
                >
                  {copiedCode === codeId ? (
                    <Check className="h-4 w-4 text-green-500" />
                  ) : (
                    <Copy className="h-4 w-4" />
                  )}
                </Button>
              </div>
            );
          },
          // Custom code element for syntax highlighting
          code({ className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || "");
            const isInline = !match;

            if (isInline) {
              return (
                <code className={className} {...props}>
                  {children}
                </code>
              );
            }

            return (
              <code className={cn(className, "hljs")} {...props}>
                {children}
              </code>
            );
          },
          // Custom table styling
          table({ children, ...props }) {
            return (
              <div className="not-prose my-4 overflow-x-auto">
                <table className="w-full text-sm border-collapse" {...props}>
                  {children}
                </table>
              </div>
            );
          },
          // Custom heading anchors
          h2({ children, ...props }) {
            const id = children?.toString().toLowerCase().replace(/\s+/g, "-");
            return (
              <h2 id={id} {...props}>
                {children}
              </h2>
            );
          },
          h3({ children, ...props }) {
            const id = children?.toString().toLowerCase().replace(/\s+/g, "-");
            return (
              <h3 id={id} {...props}>
                {children}
              </h3>
            );
          },
          // Checkbox list items
          li({ children, ...props }) {
            const text = children?.toString() || "";
            if (text.startsWith("[ ] ") || text.startsWith("[x] ")) {
              const checked = text.startsWith("[x] ");
              const label = text.substring(4);
              return (
                <li className="flex items-start gap-2 list-none" {...props}>
                  <span
                    className={cn(
                      "inline-flex items-center justify-center w-4 h-4 mt-0.5 rounded border",
                      checked
                        ? "bg-primary border-primary text-primary-foreground"
                        : "border-muted-foreground"
                    )}
                  >
                    {checked && (
                      <svg
                        className="w-3 h-3"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                    )}
                  </span>
                  <span className={checked ? "line-through opacity-60" : ""}>
                    {label}
                  </span>
                </li>
              );
            }
            return <li {...props}>{children}</li>;
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

// Helper to extract text content from React elements
function extractTextContent(element: React.ReactNode): string {
  if (typeof element === "string") {
    return element;
  }
  if (typeof element === "number") {
    return String(element);
  }
  if (Array.isArray(element)) {
    return element.map(extractTextContent).join("");
  }
  if (element && typeof element === "object" && "props" in element) {
    return extractTextContent((element as React.ReactElement).props.children);
  }
  return "";
}
