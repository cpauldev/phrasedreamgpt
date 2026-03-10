import { ArrowUpRight, Github } from "lucide-react";
import { useState } from "react";

import { ModelPanel, type ModelPanelTab } from "@/components/model-panel";
import { Button } from "@/components/ui/button";
import { TooltipProvider } from "@/components/ui/tooltip";
import { MODEL_PANELS } from "@/lib/model-catalog";
import { resolveBasePath } from "@/lib/utils";

const REPO_URL = "https://github.com/cpauldev/dreamphrase-gpt";
const PROFILE_URL = "https://github.com/cpauldev";
const BANNER_SRC = resolveBasePath("dreamphrasegpt.png");

export default function App() {
  const [activeTab, setActiveTab] = useState<ModelPanelTab>("generated");

  return (
    <TooltipProvider delay={200}>
      <main className="min-h-screen bg-background text-foreground">
        <div className="mx-auto flex w-full max-w-5xl flex-col px-4 py-5 sm:px-6">
          <header className="my-8 flex flex-wrap items-start justify-between gap-4">
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

            <HeaderLinkButton
              ariaLabel="Open the DreamPhraseGPT repository"
              href={REPO_URL}
              icon={<Github />}
              label="Repo"
              variant="outline"
              className="gap-2"
            />
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
