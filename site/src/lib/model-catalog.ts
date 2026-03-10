import { resolveBasePath } from "@/lib/utils";

export type ModelMetadataAction = {
  ariaLabel?: string;
  downloadName?: string;
  href: string;
  label: string;
  target?: "_blank";
};

export type ModelMetadataItem = {
  actions?: ModelMetadataAction[];
  helpText?: string;
  label: string;
  value: string;
};

export type ModelPanelConfig = {
  bundlePath: string;
  description: string;
  metadata: ModelMetadataItem[];
  title: string;
};

type SourceFilterMetadata = {
  bitCount: number;
  falsePositiveRate: number;
  hashCount: number;
  itemCount: number;
  kind: string;
  version: number;
};

type ModelBundleMetadata = {
  blockSize: number;
  bosId: number;
  bundleVersion: number;
  characterCount: number;
  datasetPath: string;
  exportedAt: string;
  fileSizeBytes: number;
  format: string;
  sourceFilter: SourceFilterMetadata;
  vocabSize: number;
};

type ModelCatalogEntry = {
  bundlePath: string;
  description: string;
  metadata: ModelBundleMetadata;
  title: string;
};

const MODEL_CATALOG: ModelCatalogEntry[] = [
  {
    bundlePath: "models/english_words.model",
    description: "Trained on about 370,000 English words in 27 seconds.",
    metadata: {
      blockSize: 32,
      bosId: 26,
      bundleVersion: 1,
      characterCount: 26,
      datasetPath: "datasets/english_words.txt",
      exportedAt: "2026-03-09T03:40:28",
      fileSizeBytes: 4644581,
      format: "dreamphrasegpt-onnx-bundle",
      sourceFilter: {
        bitCount: 7094957,
        falsePositiveRate: 0.0001,
        hashCount: 13,
        itemCount: 370105,
        kind: "bloom",
        version: 1,
      },
      vocabSize: 27,
    },
    title: "Words",
  },
  {
    bundlePath: "models/us_baby_names.model",
    description: "Trained on about 105,000 U.S. baby names in 27 seconds.",
    metadata: {
      blockSize: 32,
      bosId: 52,
      bundleVersion: 1,
      characterCount: 52,
      datasetPath: "datasets/us_baby_names.txt",
      exportedAt: "2026-03-09T05:03:37",
      fileSizeBytes: 3823717,
      format: "dreamphrasegpt-onnx-bundle",
      sourceFilter: {
        bitCount: 2009393,
        falsePositiveRate: 0.0001,
        hashCount: 13,
        itemCount: 104819,
        kind: "bloom",
        version: 1,
      },
      vocabSize: 53,
    },
    title: "Baby Names",
  },
];

export const MODEL_PANELS = MODEL_CATALOG.map(buildModelPanelConfig);

function buildModelPanelConfig({
  bundlePath,
  description,
  metadata,
  title,
}: ModelCatalogEntry): ModelPanelConfig {
  return {
    bundlePath,
    description,
    metadata: buildMetadata(bundlePath, metadata),
    title,
  };
}

function buildMetadata(
  bundlePath: string,
  {
    blockSize,
    bosId,
    bundleVersion,
    characterCount,
    datasetPath,
    exportedAt,
    fileSizeBytes,
    format,
    sourceFilter,
    vocabSize,
  }: ModelBundleMetadata,
): ModelMetadataItem[] {
  return [
    {
      label: "Bundle format",
      value: `${format} v${bundleVersion}`,
    },
    {
      actions: [createDownloadAction(bundlePath, "Download model bundle")],
      label: "Model size",
      value: `${formatFileSize(fileSizeBytes)} (${fileSizeBytes.toLocaleString("en-US")} bytes)`,
    },
    {
      label: "Exported",
      value: exportedAt,
    },
    {
      label: "Context window",
      helpText: "How many earlier characters the model can look at when predicting the next one.",
      value: `${blockSize} characters`,
    },
    {
      helpText: "How many distinct characters the model can generate from this dataset.",
      label: "Characters",
      value: `${characterCount} characters`,
    },
    {
      helpText: "The total number of tokenizer IDs, including the special start token.",
      label: "Tokenizer size",
      value: `${vocabSize} tokens`,
    },
    {
      helpText:
        "The ID of the special token used to begin generation. It is not a visible character.",
      label: "Start token ID",
      value: `${bosId}`,
    },
    {
      actions: [
        createViewAction(datasetPath, "View dataset text file"),
        createDownloadAction(datasetPath, "Download dataset text file"),
      ],
      label: "Source lines",
      value: sourceFilter.itemCount.toLocaleString("en-US"),
    },
    {
      helpText:
        "The filter used to reject outputs that look like exact lines from the source data.",
      label: "Source filter",
      value: `${capitalize(sourceFilter.kind)} v${sourceFilter.version}`,
    },
    {
      helpText:
        "How many bits are stored in the Bloom filter. More bits usually mean fewer accidental matches.",
      label: "Bloom bits",
      value: sourceFilter.bitCount.toLocaleString("en-US"),
    },
    {
      helpText: "How many hash positions each source line sets in the Bloom filter.",
      label: "Bloom hashes",
      value: `${sourceFilter.hashCount}`,
    },
    {
      helpText:
        "The approximate chance that the Bloom filter flags a new line as a source match by mistake.",
      label: "Bloom false-positive rate",
      value: `${(sourceFilter.falsePositiveRate * 100).toFixed(2)}%`,
    },
  ];
}

function formatFileSize(bytes: number) {
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function createDownloadAction(path: string, ariaLabel: string): ModelMetadataAction {
  return {
    ariaLabel,
    downloadName: getFileName(path),
    href: resolveBasePath(path),
    label: "Download",
  };
}

function createViewAction(path: string, ariaLabel: string): ModelMetadataAction {
  return {
    ariaLabel,
    href: resolveBasePath(path),
    label: "View",
    target: "_blank",
  };
}

function getFileName(path: string) {
  return path.split("/").pop();
}

function capitalize(value: string) {
  if (!value) {
    return value;
  }

  return `${value[0].toUpperCase()}${value.slice(1)}`;
}
