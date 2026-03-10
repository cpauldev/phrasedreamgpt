import { ArrowUpRight, DownloadIcon, HeartIcon, InfoIcon, SparklesIcon, XIcon } from "lucide-react";
import { useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import { Field, FieldLabel } from "@/components/ui/field";
import {
  Frame,
  FrameDescription,
  FrameHeader,
  FramePanel,
  FrameTitle,
} from "@/components/ui/frame";
import { Slider, SliderValue } from "@/components/ui/slider";
import { Spinner } from "@/components/ui/spinner";
import { Table, TableBody, TableCell, TableRow } from "@/components/ui/table";
import { Tabs, TabsList, TabsPanel, TabsTab } from "@/components/ui/tabs";
import { Tooltip, TooltipPopup, TooltipTrigger } from "@/components/ui/tooltip";
import type { ModelMetadataItem, ModelPanelConfig } from "@/lib/model-catalog";
import {
  DEFAULT_TEMPERATURE,
  DISPLAY_COLUMN_COUNT,
  DISPLAY_ROW_COUNT,
  formatTemperature,
  MAX_TEMPERATURE,
  MIN_TEMPERATURE,
  normalizeLikeValue,
  normalizeTemperature,
  TEMPERATURE_TICK_STEP,
  TEMPERATURE_TICKS,
  useModelResults,
  usePersistentLikes,
} from "@/lib/model-results";
import { cn } from "@/lib/utils";

const TEMPERATURE_TICK_LABEL_INTERVAL = 2;
const TEMPERATURE_FORMAT = {
  maximumFractionDigits: 1,
  minimumFractionDigits: 1,
} as const;
const EMPTY_RESULTS_AREA_CLASS = "min-h-[13.75rem] sm:min-h-[12.5rem]";

const MODEL_PANEL_TABS = ["generated", "likes", "details"] as const;

export type ModelPanelTab = (typeof MODEL_PANEL_TABS)[number];

export function ModelPanel({
  activeTab,
  bundlePath,
  description,
  metadata,
  onTabChange,
  title,
}: ModelPanelConfig & {
  activeTab: ModelPanelTab;
  onTabChange: (value: ModelPanelTab) => void;
}) {
  const [temperature, setTemperature] = useState(DEFAULT_TEMPERATURE);
  const temperatureKey = formatTemperature(temperature);
  const {
    error,
    generateNextPage,
    isInitialLoading,
    isRefreshingSet,
    isWaitingForNextBatch,
    visibleResults,
  } = useModelResults(bundlePath, temperatureKey);
  const { likes, toggleLike } = usePersistentLikes(`dreamphrasegpt:likes:${bundlePath}`);

  const likedResults = useMemo(() => new Set(likes.map(normalizeLikeValue)), [likes]);
  const hasVisibleResults = visibleResults.some(Boolean);
  const isGenerateDisabled = isInitialLoading || isRefreshingSet || isWaitingForNextBatch;

  return (
    <Frame>
      <FrameHeader>
        <FrameTitle className="text-3xl font-semibold tracking-tight">{title}</FrameTitle>
        <FrameDescription className="mt-2">{description}</FrameDescription>
      </FrameHeader>

      <FramePanel className="space-y-6">
        <Field>
          <Slider
            aria-label={`${title} temperature`}
            min={MIN_TEMPERATURE}
            max={MAX_TEMPERATURE}
            step={TEMPERATURE_TICK_STEP}
            format={TEMPERATURE_FORMAT}
            defaultValue={DEFAULT_TEMPERATURE}
            onValueChange={(value) => {
              const nextValue = Array.isArray(value) ? value[0] : value;
              setTemperature(normalizeTemperature(Number(nextValue ?? DEFAULT_TEMPERATURE)));
            }}
            className="w-full"
          >
            <div className="mb-2 flex items-center justify-between gap-1">
              <FieldLabel className="font-medium text-sm">Temperature</FieldLabel>
              <Badge render={<SliderValue />} variant="outline" className="text-muted-foreground" />
            </div>
          </Slider>

          {/* biome-ignore lint/a11y/useSemanticElements: Match the documented COSS slider scale pattern. */}
          <div
            aria-label={`Temperature scale from ${MIN_TEMPERATURE.toFixed(1)} to ${MAX_TEMPERATURE.toFixed(1)}`}
            className="mt-3 flex w-full items-center justify-between gap-1 px-2.5 font-medium text-muted-foreground text-xs"
            role="group"
          >
            {TEMPERATURE_TICKS.map((tick, index) => (
              <span
                className="flex w-0 flex-col items-center justify-center gap-2"
                key={String(tick)}
              >
                <span
                  className={cn(
                    "h-1 w-px bg-muted-foreground/70",
                    index % TEMPERATURE_TICK_LABEL_INTERVAL !== 0 && "h-0.5",
                  )}
                />
                <span className={cn(index % TEMPERATURE_TICK_LABEL_INTERVAL !== 0 && "opacity-0")}>
                  {tick}
                </span>
              </span>
            ))}
          </div>
        </Field>

        <Button
          onClick={generateNextPage}
          disabled={isGenerateDisabled}
          size="xl"
          className="w-full"
        >
          {getButtonLabel({
            error,
            hasVisibleResults,
            isInitialLoading,
            isRefreshingSet,
            isWaitingForNextBatch,
          })}
        </Button>
      </FramePanel>

      <FramePanel className="p-0">
        <Tabs
          value={activeTab}
          onValueChange={(value) => {
            if (isModelPanelTab(value)) {
              onTabChange(value);
            }
          }}
          className="gap-0"
        >
          <div className="px-5 pt-4">
            <TabsList variant="underline">
              <TabsTab value="generated">
                <SparklesIcon aria-hidden="true" className="opacity-60" />
                Generated
              </TabsTab>
              <TabsTab value="likes">
                <HeartIcon aria-hidden="true" className="opacity-60" />
                Likes
                {likes.length > 0 ? (
                  <Badge className="not-in-data-active:text-muted-foreground" variant="outline">
                    {likes.length}
                  </Badge>
                ) : null}
              </TabsTab>
              <TabsTab value="details">
                <InfoIcon aria-hidden="true" className="opacity-60" />
                Details
              </TabsTab>
            </TabsList>
          </div>

          <TabsPanel value="generated" className="p-0">
            {isInitialLoading && !hasVisibleResults ? (
              <Empty className={cn("px-5 py-5", EMPTY_RESULTS_AREA_CLASS)}>
                <EmptyHeader>
                  <EmptyTitle>Loading results</EmptyTitle>
                  <EmptyDescription>This first batch is generating now.</EmptyDescription>
                </EmptyHeader>
              </Empty>
            ) : error && !hasVisibleResults ? (
              <Empty className={cn("px-5 py-5", EMPTY_RESULTS_AREA_CLASS)}>
                <EmptyHeader>
                  <EmptyTitle>Generation failed</EmptyTitle>
                  <EmptyDescription>{error}</EmptyDescription>
                </EmptyHeader>
              </Empty>
            ) : (
              <>
                <ResultsTable
                  ariaLabel={`${title} generated results`}
                  items={visibleResults}
                  likedResults={likedResults}
                  minimumRowCount={DISPLAY_ROW_COUNT}
                  onToggleLike={toggleLike}
                  rowAction="like"
                />

                {error ? (
                  <p className="px-5 pb-5 pt-2 text-sm text-destructive-foreground">{error}</p>
                ) : null}
              </>
            )}
          </TabsPanel>

          <TabsPanel value="likes" className="p-0">
            {likes.length === 0 ? (
              <Empty className={cn("px-5 py-5", EMPTY_RESULTS_AREA_CLASS)}>
                <EmptyHeader>
                  <EmptyTitle>No likes yet</EmptyTitle>
                  <EmptyDescription>Heart a result to keep it here.</EmptyDescription>
                </EmptyHeader>
              </Empty>
            ) : (
              <ResultsTable
                ariaLabel={`${title} liked results`}
                items={likes}
                likedResults={likedResults}
                onToggleLike={toggleLike}
                rowAction="remove"
              />
            )}
          </TabsPanel>

          <TabsPanel value="details" className="px-5 py-5">
            <ModelDetailsList items={metadata} panelTitle={title} />
          </TabsPanel>
        </Tabs>
      </FramePanel>
    </Frame>
  );
}

type ModelDetailsListProps = {
  items: ModelMetadataItem[];
  panelTitle: string;
};

function ModelDetailsList({ items, panelTitle }: ModelDetailsListProps) {
  return (
    <dl className="grid gap-x-6 gap-y-3 sm:grid-cols-2">
      {items.map((item) => (
        <div key={`${panelTitle}-${item.label}`} className="min-w-0 space-y-1">
          <dt className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <span>{item.label}</span>
            {item.helpText ? (
              <Tooltip>
                <TooltipTrigger
                  render={
                    <span className="inline-flex size-4 items-center justify-center text-muted-foreground transition-colors hover:text-foreground focus-visible:text-foreground focus-visible:outline-none" />
                  }
                >
                  <InfoIcon aria-hidden="true" className="size-3.5" />
                </TooltipTrigger>
                <TooltipPopup>{item.helpText}</TooltipPopup>
              </Tooltip>
            ) : null}
          </dt>
          <dd className="min-w-0 text-sm text-foreground">
            {item.actions?.length ? (
              <div className="flex flex-col justify-between gap-2">
                <span className="truncate" title={item.value}>
                  {item.value}
                </span>
                <div className="flex flex-wrap gap-2">
                  {item.actions.map((action) => (
                    <MetadataActionButton
                      key={`${panelTitle}-${item.label}-${action.label}`}
                      action={action}
                    />
                  ))}
                </div>
              </div>
            ) : (
              item.value
            )}
          </dd>
        </div>
      ))}
    </dl>
  );
}

type MetadataActionButtonProps = {
  action: NonNullable<ModelMetadataItem["actions"]>[number];
};

function MetadataActionButton({ action }: MetadataActionButtonProps) {
  const actionIcon =
    action.target === "_blank" ? (
      <ArrowUpRight aria-hidden="true" />
    ) : (
      <DownloadIcon aria-hidden="true" />
    );

  return (
    <Button
      render={
        <a
          href={action.href}
          target={action.target}
          rel={action.target === "_blank" ? "noreferrer" : undefined}
          aria-label={action.ariaLabel}
          download={action.downloadName}
        >
          {action.target === "_blank" ? action.label : null}
          {actionIcon}
          {action.target === "_blank" ? null : action.label}
        </a>
      }
      variant="outline"
      size="xs"
    />
  );
}

type ResultsTableProps = {
  ariaLabel: string;
  items: string[];
  likedResults: Set<string>;
  minimumRowCount?: number;
  onToggleLike: (value: string) => void;
  rowAction: "like" | "remove";
};

function ResultsTable({
  ariaLabel,
  items,
  likedResults,
  minimumRowCount = 0,
  onToggleLike,
  rowAction,
}: ResultsTableProps) {
  return (
    <Table aria-label={ariaLabel} className="table-fixed">
      <TableBody className="before:hidden shadow-none *:[tr]:*:[td]:border-0 *:[tr]:*:[td]:bg-transparent *:[tr]:*:[td]:first:border-s-0 *:[tr]:*:[td]:last:border-e-0 *:[tr]:first:*:[td]:border-t-0">
        {buildResultRows(items, minimumRowCount).map((row) => (
          <TableRow key={`${ariaLabel}-${row.key}`}>
            {row.cells.map((cell) => (
              <TableCell
                key={`${ariaLabel}-${cell.key}`}
                className="w-1/2 max-w-0 min-w-0 overflow-hidden py-0"
              >
                <ResultCell
                  value={cell.value}
                  isLiked={likedResults.has(normalizeLikeValue(cell.value))}
                  onToggleLike={onToggleLike}
                  rowAction={rowAction}
                />
              </TableCell>
            ))}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

type ResultCellProps = {
  isLiked: boolean;
  onToggleLike: (value: string) => void;
  rowAction: "like" | "remove";
  value: string;
};

function ResultCell({ value, isLiked, onToggleLike, rowAction }: ResultCellProps) {
  if (!value) {
    return (
      <div className="flex h-11 min-w-0 items-center gap-2 px-4 sm:h-10">
        <span className="invisible block flex-1 truncate">Placeholder</span>
        <span aria-hidden="true" className="size-5 shrink-0" />
      </div>
    );
  }

  return (
    <Button
      aria-label={rowAction === "remove" ? `Remove ${value} from likes` : `Like ${value}`}
      aria-pressed={rowAction === "like" ? isLiked : undefined}
      size="xl"
      variant="ghost"
      className={cn(
        "max-w-full min-w-0 w-full justify-between overflow-hidden font-medium hover:text-foreground",
        rowAction === "like" && isLiked && "text-red-500 hover:text-red-500",
      )}
      onClick={() => onToggleLike(value)}
    >
      <span className="block min-w-0 flex-1 overflow-hidden truncate text-left">{value}</span>
      {rowAction === "remove" ? (
        <XIcon aria-hidden="true" className="shrink-0 text-muted-foreground" />
      ) : (
        <HeartIcon
          aria-hidden="true"
          className={cn("shrink-0 text-muted-foreground", isLiked && "fill-current text-red-500")}
        />
      )}
    </Button>
  );
}

type ButtonLabelState = {
  error: string;
  hasVisibleResults: boolean;
  isInitialLoading: boolean;
  isRefreshingSet: boolean;
  isWaitingForNextBatch: boolean;
};

function getButtonLabel({
  error,
  hasVisibleResults,
  isInitialLoading,
  isRefreshingSet,
  isWaitingForNextBatch,
}: ButtonLabelState) {
  if (isInitialLoading || isRefreshingSet) {
    return (
      <>
        <Spinner />
        Generating
      </>
    );
  }

  if (error && !hasVisibleResults) {
    return "Retry";
  }

  if (isWaitingForNextBatch) {
    return (
      <>
        <Spinner />
        Preparing
      </>
    );
  }

  return "Generate";
}

function buildResultRows(items: string[], minimumRowCount = 0) {
  const rowCount = Math.max(minimumRowCount, Math.ceil(items.length / DISPLAY_COLUMN_COUNT));

  return Array.from({ length: rowCount }, (_, rowIndex) => {
    const startIndex = rowIndex * DISPLAY_COLUMN_COUNT;
    const leftValue = items[startIndex] ?? "";
    const rightValue = items[startIndex + 1] ?? "";

    return {
      cells: [
        {
          key: leftValue ? `left-${rowIndex}-${leftValue}` : `left-${rowIndex}-empty`,
          value: leftValue,
        },
        {
          key: rightValue ? `right-${rowIndex}-${rightValue}` : `right-${rowIndex}-empty`,
          value: rightValue,
        },
      ],
      key: `${leftValue || `left-empty-${rowIndex}`}-${rightValue || `right-empty-${rowIndex}`}`,
    };
  });
}

function isModelPanelTab(value: unknown): value is ModelPanelTab {
  return typeof value === "string" && MODEL_PANEL_TABS.includes(value as ModelPanelTab);
}
