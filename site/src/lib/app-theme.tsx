import { createContext, useContext, useEffect, useMemo, useState } from "react";

import { useMediaQuery } from "@/hooks/use-media-query";
import {
  applyThemePreference,
  getSystemPrefersDark,
  type ResolvedTheme,
  readStoredThemePreference,
  resolveThemePreference,
  THEME_STORAGE_KEY,
  type ThemePreference,
  writeStoredThemePreference,
} from "@/lib/theme";

type ThemeContextValue = {
  resolvedTheme: ResolvedTheme;
  setPreference: (preference: ThemePreference) => void;
};

const FALLBACK_THEME_CONTEXT: ThemeContextValue = {
  resolvedTheme: "light",
  setPreference: () => {},
};

const ThemeContext = createContext<ThemeContextValue>(FALLBACK_THEME_CONTEXT);

function getInitialThemePreference() {
  if (typeof window === "undefined") {
    return "system" satisfies ThemePreference;
  }

  return readStoredThemePreference(window.localStorage);
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const systemPrefersDark = useMediaQuery("(prefers-color-scheme: dark)");
  const [preference, setPreference] = useState<ThemePreference>(getInitialThemePreference);
  const resolvedTheme = useMemo(
    () => resolveThemePreference(preference, systemPrefersDark),
    [preference, systemPrefersDark],
  );

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    writeStoredThemePreference(preference, window.localStorage);
  }, [preference]);

  useEffect(() => {
    if (typeof document === "undefined") {
      return;
    }

    applyThemePreference(document.documentElement, preference, systemPrefersDark);
  }, [preference, systemPrefersDark]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const handleStorage = (event: StorageEvent) => {
      if (event.storageArea !== window.localStorage || event.key !== THEME_STORAGE_KEY) {
        return;
      }

      setPreference(readStoredThemePreference(window.localStorage));
    };

    window.addEventListener("storage", handleStorage);
    return () => window.removeEventListener("storage", handleStorage);
  }, []);

  const value = useMemo<ThemeContextValue>(
    () => ({
      resolvedTheme,
      setPreference,
    }),
    [resolvedTheme],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useAppTheme() {
  return useContext(ThemeContext);
}

export function initializeAppTheme() {
  if (typeof document === "undefined" || typeof window === "undefined") {
    return;
  }

  const preference = readStoredThemePreference(window.localStorage);
  applyThemePreference(document.documentElement, preference, getSystemPrefersDark(window));
}
