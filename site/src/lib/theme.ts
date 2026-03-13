export type ThemePreference = "system" | "light" | "dark";
export type ResolvedTheme = "light" | "dark";

export const THEME_STORAGE_KEY = "dreamphrasegpt-browser:theme";
const THEME_MEDIA_QUERY = "(prefers-color-scheme: dark)";

const THEME_COLOR_BY_MODE: Record<ResolvedTheme, string> = {
  dark: "#09090b",
  light: "#ffffff",
};

function isThemePreference(value: unknown): value is ThemePreference {
  return value === "system" || value === "light" || value === "dark";
}

export function readStoredThemePreference(
  storage?: Pick<Storage, "getItem"> | null,
): ThemePreference {
  try {
    const storedValue = storage?.getItem(THEME_STORAGE_KEY);
    return isThemePreference(storedValue) ? storedValue : "system";
  } catch {
    return "system";
  }
}

export function writeStoredThemePreference(
  preference: ThemePreference,
  storage?: Pick<Storage, "removeItem" | "setItem"> | null,
) {
  try {
    if (!storage) {
      return;
    }

    if (preference === "system") {
      storage.removeItem(THEME_STORAGE_KEY);
      return;
    }

    storage.setItem(THEME_STORAGE_KEY, preference);
  } catch {
    // Ignore storage write failures.
  }
}

export function resolveThemePreference(
  preference: ThemePreference,
  systemPrefersDark: boolean,
): ResolvedTheme {
  if (preference === "system") {
    return systemPrefersDark ? "dark" : "light";
  }

  return preference;
}

export function getSystemPrefersDark(windowObject?: Pick<Window, "matchMedia"> | null): boolean {
  try {
    return windowObject?.matchMedia(THEME_MEDIA_QUERY).matches ?? false;
  } catch {
    return false;
  }
}

function applyResolvedTheme(
  rootElement: HTMLElement,
  theme: ResolvedTheme,
  documentObject: Pick<Document, "querySelector"> = document,
) {
  rootElement.classList.toggle("dark", theme === "dark");
  rootElement.style.colorScheme = theme;

  const themeColorMeta = documentObject.querySelector('meta[name="theme-color"]');
  if (themeColorMeta instanceof HTMLMetaElement) {
    themeColorMeta.content = THEME_COLOR_BY_MODE[theme];
  }

  return theme;
}

export function applyThemePreference(
  rootElement: HTMLElement,
  preference: ThemePreference,
  systemPrefersDark: boolean,
  documentObject: Pick<Document, "querySelector"> = document,
) {
  const resolvedTheme = resolveThemePreference(preference, systemPrefersDark);
  return applyResolvedTheme(rootElement, resolvedTheme, documentObject);
}
