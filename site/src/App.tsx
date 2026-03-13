import { ArrowUpRight, Github, Moon, Sun } from "lucide-react";
import { useState } from "react";

import { ModelPanel, type ModelPanelTab } from "@/components/model-panel";
import { Button } from "@/components/ui/button";
import { PreviewCard, PreviewCardPopup, PreviewCardTrigger } from "@/components/ui/preview-card";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useAppTheme } from "@/lib/app-theme";
import { MODEL_PANELS } from "@/lib/model-catalog";
import { resolveBasePath } from "@/lib/utils";

const REPO_URL = "https://github.com/cpauldev/dreamphrase-gpt";
const PROFILE_URL = "https://github.com/cpauldev";
const BANNER_SRC = resolveBasePath("dreamphrasegpt.png");

export default function App() {
  const [activeTab, setActiveTab] = useState<ModelPanelTab>("generated");
  const { resolvedTheme, setPreference } = useAppTheme();

  function toggleTheme() {
    setPreference(resolvedTheme === "dark" ? "light" : "dark");
  }

  return (
    <TooltipProvider delay={200}>
      <main className="min-h-screen bg-background text-foreground">
        <div className="mx-auto flex w-full max-w-5xl flex-col px-4 py-5 sm:px-6">
          <header className="mb-8 flex flex-wrap items-start justify-between gap-4">
            <div className="max-w-3xl space-y-3">
              <h1 className="leading-none">
                <img
                  src={BANNER_SRC}
                  alt="DreamPhraseGPT"
                  className="h-auto w-full max-w-md sm:max-w-lg"
                />
              </h1>
              <p className="text-sm leading-6 text-muted-foreground sm:text-base">
                DreamPhraseGPT trains a character-level transformer on any newline-delimited text
                file and can generate strings that follow the character patterns, structure, and
                common sequences learned from that dataset.
              </p>
              <p className="text-sm text-muted-foreground">
                Research and implementation by Christian Paul{" "}
                <HeaderLinkButton
                  ariaLabel="Open Christian Paul's GitHub profile"
                  href={PROFILE_URL}
                  label="@cpauldev"
                  size="xs"
                  variant="link"
                  className="h-auto px-0 text-sm text-foreground"
                />
              </p>
            </div>

            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="icon"
                aria-label={
                  resolvedTheme === "dark" ? "Switch to light mode" : "Switch to dark mode"
                }
                onClick={toggleTheme}
              >
                {resolvedTheme === "dark" ? <Sun /> : <Moon />}
              </Button>

              <PreviewCard>
                <PreviewCardTrigger
                  delay={300}
                  render={
                    <Button
                      render={
                        <a
                          href={REPO_URL}
                          target="_blank"
                          rel="noreferrer"
                          aria-label="Open the DreamPhraseGPT repository"
                        >
                          <Github />
                          Repo
                          <ArrowUpRight />
                        </a>
                      }
                      variant="outline"
                      className="gap-2"
                    />
                  }
                />
                <PreviewCardPopup
                  align="end"
                  sideOffset={8}
                  className="w-80 max-w-[calc(100vw-2rem)] text-wrap"
                >
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <Github className="size-4 shrink-0" />
                      <span className="font-semibold text-sm">cpauldev/dreamphrase-gpt</span>
                    </div>
                    <p className="text-xs leading-5 text-muted-foreground">
                      Train a character-level GPT on newline-delimited text files and generate new
                      strings that follow the character patterns, structure, and common sequences
                      learned from the dataset.
                    </p>
                    <a
                      href={REPO_URL}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1 text-xs text-foreground underline-offset-4 hover:underline"
                    >
                      View on GitHub
                      <ArrowUpRight className="size-3" />
                    </a>
                  </div>
                </PreviewCardPopup>
              </PreviewCard>
            </div>
          </header>

          <section className="grid gap-6 lg:grid-cols-2">
            {MODEL_PANELS.map((model) => (
              <ModelPanel
                key={model.title}
                activeTab={activeTab}
                onTabChange={setActiveTab}
                {...model}
              />
            ))}
          </section>
        </div>
      </main>
    </TooltipProvider>
  );
}

type HeaderLinkButtonProps = {
  ariaLabel: string;
  className?: string;
  href: string;
  icon?: React.ReactNode;
  label: string;
  size?: React.ComponentProps<typeof Button>["size"];
  variant: React.ComponentProps<typeof Button>["variant"];
};

function HeaderLinkButton({
  ariaLabel,
  className,
  href,
  icon,
  label,
  size,
  variant,
}: HeaderLinkButtonProps) {
  return (
    <Button
      render={
        <a href={href} target="_blank" rel="noreferrer" aria-label={ariaLabel}>
          {icon}
          {label}
          <ArrowUpRight />
        </a>
      }
      variant={variant}
      size={size}
      className={className}
    />
  );
}
