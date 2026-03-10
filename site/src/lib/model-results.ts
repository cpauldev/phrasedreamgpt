import { useCallback, useEffect, useRef, useState } from "react";

import { generateBundleSamples } from "@/lib/dreamphrase-browser";

type ResultPageSet = {
  pages: string[][];
  temperatureKey: string;
};

export const DISPLAY_COUNT = 10;
export const DISPLAY_COLUMN_COUNT = 2;
export const DISPLAY_ROW_COUNT = DISPLAY_COUNT / DISPLAY_COLUMN_COUNT;
const RESULT_PAGES_PER_SET = 5;
const RESULTS_PER_SET = DISPLAY_COUNT * RESULT_PAGES_PER_SET;

export const DEFAULT_TEMPERATURE = 0.8;
export const MIN_TEMPERATURE = 0.4;
export const MAX_TEMPERATURE = 1.4;
export const TEMPERATURE_TICK_STEP = 0.1;
export const TEMPERATURE_TICKS = buildTemperatureTicks();
const EMPTY_RESULTS = toDisplayRows([]);

export type ModelResultsState = {
  error: string;
  generateNextPage: () => void;
  isInitialLoading: boolean;
  isRefreshingSet: boolean;
  isWaitingForNextBatch: boolean;
  visibleResults: string[];
};

export function useModelResults(
  bundlePath: string,
  selectedTemperatureKey: string,
): ModelResultsState {
  const activeController = useRef<AbortController | null>(null);
  const initialTemperatureKey = useRef(selectedTemperatureKey);
  const requestId = useRef(0);

  const [currentSet, setCurrentSet] = useState<ResultPageSet | null>(null);
  const [currentPageIndex, setCurrentPageIndex] = useState(0);
  const [nextBatch, setNextBatch] = useState<ResultPageSet | null>(null);
  const [error, setError] = useState("");
  const [isInitialLoading, setIsInitialLoading] = useState(true);
  const [isPreparingNextSet, setIsPreparingNextSet] = useState(false);
  const [isQueuedToPrepareNextSet, setIsQueuedToPrepareNextSet] = useState(false);
  const [isWaitingForNextBatch, setIsWaitingForNextBatch] = useState(false);
  const [isRefreshingSet, setIsRefreshingSet] = useState(false);

  const visibleResults = currentSet?.pages[currentPageIndex] ?? EMPTY_RESULTS;
  const activeTemperatureKey = currentSet?.temperatureKey ?? null;
  const hasHiddenPages = Boolean(currentSet && currentPageIndex < currentSet.pages.length - 1);
  const isBlockedByNextBatch =
    isWaitingForNextBatch &&
    activeTemperatureKey === selectedTemperatureKey &&
    !hasHiddenPages &&
    !nextBatch &&
    (isQueuedToPrepareNextSet || isPreparingNextSet);

  const cancelActiveRequest = useCallback(() => {
    requestId.current += 1;
    activeController.current?.abort();
    activeController.current = null;
  }, []);

  const startRequest = useCallback(() => {
    cancelActiveRequest();

    const controller = new AbortController();
    activeController.current = controller;

    return {
      id: requestId.current,
      signal: controller.signal,
    };
  }, [cancelActiveRequest]);

  const loadFreshSet = useCallback(
    async (temperatureKey: string, isInitialRequest = false) => {
      const activeRequest = startRequest();

      if (isInitialRequest) {
        setIsInitialLoading(true);
      }

      setIsRefreshingSet(true);
      setIsQueuedToPrepareNextSet(false);
      setIsWaitingForNextBatch(false);
      setError("");
      setNextBatch(null);

      try {
        const nextPageSet = await requestPageSet(
          bundlePath,
          RESULTS_PER_SET,
          temperatureKey,
          activeRequest.signal,
        );

        if (activeRequest.id !== requestId.current) {
          return;
        }

        setCurrentSet(nextPageSet);
        setCurrentPageIndex(0);
      } catch (caughtError) {
        if (activeRequest.id !== requestId.current || isAbortError(caughtError)) {
          return;
        }

        setError(getErrorMessage(caughtError));
      } finally {
        if (activeRequest.id === requestId.current) {
          if (activeController.current?.signal === activeRequest.signal) {
            activeController.current = null;
          }
          setIsRefreshingSet(false);
          if (isInitialRequest) {
            setIsInitialLoading(false);
          }
        }
      }
    },
    [bundlePath, startRequest],
  );

  const prepareNextSet = useCallback(
    async (temperatureKey: string) => {
      const activeRequest = startRequest();

      setIsPreparingNextSet(true);
      setIsQueuedToPrepareNextSet(false);
      setError("");

      try {
        const nextPageSet = await requestPageSet(
          bundlePath,
          RESULTS_PER_SET,
          temperatureKey,
          activeRequest.signal,
        );

        if (activeRequest.id !== requestId.current) {
          return;
        }

        setIsWaitingForNextBatch(false);
        setNextBatch(nextPageSet);
      } catch (caughtError) {
        if (activeRequest.id !== requestId.current || isAbortError(caughtError)) {
          return;
        }

        setIsWaitingForNextBatch(false);
        setError(getErrorMessage(caughtError));
      } finally {
        if (activeRequest.id === requestId.current) {
          if (activeController.current?.signal === activeRequest.signal) {
            activeController.current = null;
          }
          setIsPreparingNextSet(false);
        }
      }
    },
    [bundlePath, startRequest],
  );

  useEffect(() => {
    return () => {
      requestId.current += 1;
      activeController.current?.abort();
      activeController.current = null;
    };
  }, []);

  useEffect(() => {
    void warmStart();

    async function warmStart() {
      setCurrentSet(null);
      setCurrentPageIndex(0);
      setNextBatch(null);
      await loadFreshSet(initialTemperatureKey.current, true);
    }
  }, [loadFreshSet]);

  useEffect(() => {
    if (!currentSet || selectedTemperatureKey === currentSet.temperatureKey) {
      return;
    }

    setNextBatch(null);
    setIsQueuedToPrepareNextSet(false);
    setIsWaitingForNextBatch(false);

    if (!isPreparingNextSet) {
      return;
    }

    cancelActiveRequest();
    setIsPreparingNextSet(false);
  }, [cancelActiveRequest, currentSet, isPreparingNextSet, selectedTemperatureKey]);

  useEffect(() => {
    if (!currentSet || selectedTemperatureKey !== currentSet.temperatureKey) {
      return;
    }

    if (nextBatch?.temperatureKey === currentSet.temperatureKey) {
      return;
    }

    if (isPreparingNextSet || isRefreshingSet) {
      return;
    }

    const remainingHiddenPages = currentSet.pages.length - currentPageIndex - 1;

    if (remainingHiddenPages > 0 && !isQueuedToPrepareNextSet) {
      return;
    }

    let timeoutId: number | null = null;
    const frameId = window.requestAnimationFrame(() => {
      timeoutId = window.setTimeout(() => {
        void prepareNextSet(currentSet.temperatureKey);
      }, 0);
    });

    return () => {
      window.cancelAnimationFrame(frameId);
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [
    currentPageIndex,
    currentSet,
    isPreparingNextSet,
    isQueuedToPrepareNextSet,
    isRefreshingSet,
    nextBatch?.temperatureKey,
    prepareNextSet,
    selectedTemperatureKey,
  ]);

  useEffect(() => {
    if (!isWaitingForNextBatch || isBlockedByNextBatch) {
      return;
    }

    setIsWaitingForNextBatch(false);
  }, [isBlockedByNextBatch, isWaitingForNextBatch]);

  function generateNextPage() {
    if (isInitialLoading || isRefreshingSet) {
      return;
    }

    if (!currentSet || selectedTemperatureKey !== currentSet.temperatureKey) {
      void loadFreshSet(selectedTemperatureKey);
      return;
    }

    const nextPageIndex = currentPageIndex + 1;

    if (nextPageIndex < currentSet.pages.length) {
      const isShowingLastVisiblePage = nextPageIndex === currentSet.pages.length - 1;

      if (isShowingLastVisiblePage && !nextBatch) {
        setIsQueuedToPrepareNextSet(true);
      }

      setCurrentPageIndex(nextPageIndex);
      setError("");
      return;
    }

    if (nextBatch && nextBatch.temperatureKey === currentSet.temperatureKey) {
      setIsWaitingForNextBatch(false);
      setCurrentSet(nextBatch);
      setCurrentPageIndex(0);
      setNextBatch(null);
      setError("");
      return;
    }

    if (isPreparingNextSet || isQueuedToPrepareNextSet) {
      setIsWaitingForNextBatch(true);
      return;
    }

    void loadFreshSet(currentSet.temperatureKey);
  }

  return {
    error,
    generateNextPage,
    isInitialLoading,
    isRefreshingSet,
    isWaitingForNextBatch: isBlockedByNextBatch,
    visibleResults,
  };
}

export function usePersistentLikes(storageKey: string) {
  const [likes, setLikes] = useState<string[]>(() =>
    typeof window === "undefined" ? [] : readStoredLikes(storageKey),
  );
  const previousStorageKey = useRef(storageKey);

  useEffect(() => {
    if (previousStorageKey.current === storageKey) {
      return;
    }

    previousStorageKey.current = storageKey;
    setLikes(readStoredLikes(storageKey));
  }, [storageKey]);

  useEffect(() => {
    try {
      window.localStorage.setItem(storageKey, JSON.stringify(likes));
    } catch {
      // Ignore storage failures and keep working with in-memory likes.
    }
  }, [likes, storageKey]);

  function toggleLike(value: string) {
    const normalizedValue = normalizeLikeValue(value);

    if (!normalizedValue) {
      return;
    }

    setLikes((currentLikes) =>
      currentLikes.includes(normalizedValue)
        ? currentLikes.filter((item) => item !== normalizedValue)
        : [normalizedValue, ...currentLikes],
    );
  }

  return {
    likes,
    toggleLike,
  };
}

export function formatTemperature(value: number) {
  return normalizeTemperature(value).toFixed(1);
}

export function normalizeLikeValue(value: string) {
  return value.trim();
}

export function normalizeTemperature(value: number) {
  if (!Number.isFinite(value)) {
    return MIN_TEMPERATURE;
  }

  const clampedValue = Math.min(MAX_TEMPERATURE, Math.max(MIN_TEMPERATURE, value));
  const roundedValue = Math.round(clampedValue / TEMPERATURE_TICK_STEP) * TEMPERATURE_TICK_STEP;

  return Number(roundedValue.toFixed(1));
}

async function requestPageSet(
  bundlePath: string,
  samples: number,
  temperatureKey: string,
  signal?: AbortSignal,
) {
  const items = await generateBundleSamples(bundlePath, {
    samples,
    signal,
    temperature: Number(temperatureKey),
  });

  return {
    pages: splitResultPages(items),
    temperatureKey,
  };
}

function toDisplayRows(items: string[]) {
  return Array.from({ length: DISPLAY_COUNT }, (_, index) => items[index] ?? "");
}

function splitResultPages(items: string[]) {
  const pageCount = Math.max(1, Math.ceil(items.length / DISPLAY_COUNT));

  return Array.from({ length: pageCount }, (_, pageIndex) =>
    toDisplayRows(items.slice(pageIndex * DISPLAY_COUNT, (pageIndex + 1) * DISPLAY_COUNT)),
  );
}

function buildTemperatureTicks() {
  const tickCount = Math.round((MAX_TEMPERATURE - MIN_TEMPERATURE) / TEMPERATURE_TICK_STEP);
  return Array.from({ length: tickCount + 1 }, (_, index) =>
    (MIN_TEMPERATURE + index * TEMPERATURE_TICK_STEP).toFixed(1),
  );
}

function getErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : "Generation failed for an unknown reason.";
}

function isAbortError(error: unknown) {
  return error instanceof Error && error.name === "AbortError";
}

function readStoredLikes(storageKey: string) {
  try {
    const storedLikes = window.localStorage.getItem(storageKey);

    if (!storedLikes) {
      return [];
    }

    return sanitizeLikes(JSON.parse(storedLikes));
  } catch {
    return [];
  }
}

function sanitizeLikes(value: unknown) {
  if (!Array.isArray(value)) {
    return [];
  }

  const uniqueLikes = new Set<string>();

  for (const item of value) {
    if (typeof item !== "string") {
      continue;
    }

    const normalizedItem = normalizeLikeValue(item);

    if (!normalizedItem) {
      continue;
    }

    uniqueLikes.add(normalizedItem);
  }

  return Array.from(uniqueLikes);
}
