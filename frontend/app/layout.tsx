import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";
import { NavLink } from "@/components/NavLink";

export const metadata: Metadata = {
  title: "Voice Agent Console",
  description: "Ops console for the Digital Brolly outbound voice agent backend.",
};

const NAV_ITEMS = [
  { href: "/", label: "Dashboard" },
  { href: "/campaigns", label: "Campaigns" },
  { href: "/leads", label: "Leads" },
  { href: "/calls", label: "Calls" },
  { href: "/settings", label: "Settings" },
];

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="mx-auto flex min-h-screen max-w-6xl flex-col">
          <header className="flex flex-wrap items-center gap-1 border-b border-neutral-200 px-4 py-3 dark:border-neutral-800 sm:px-6">
            <span className="mr-4 text-sm font-semibold text-neutral-900 dark:text-neutral-50">
              Voice Agent Console
            </span>
            <nav className="flex flex-wrap gap-1">
              {NAV_ITEMS.map((item) => (
                <NavLink key={item.href} href={item.href}>
                  {item.label}
                </NavLink>
              ))}
            </nav>
          </header>
          <main className="flex-1 px-4 py-6 sm:px-6">{children}</main>
        </div>
      </body>
    </html>
  );
}
