import type { Metadata } from "next";
import type { ReactNode } from "react";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Voice Agent — Test Console",
  description: "Local test frontend for the ElevenLabs voice agent backend.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <nav className="nav">
          <span className="nav-title">Voice Agent Test Console</span>
          <Link href="/">Campaigns</Link>
          <Link href="/rag">RAG Search</Link>
        </nav>
        <main className="main">{children}</main>
      </body>
    </html>
  );
}
